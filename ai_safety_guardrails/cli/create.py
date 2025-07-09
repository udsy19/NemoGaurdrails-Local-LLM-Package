"""
CLI commands for creating applications from templates.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import click
import yaml

from ..templates.creator import TemplateCreator
from ..utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def create_command():
    """Create applications from templates."""
    pass


@create_command.command()
@click.argument("app_name")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["chat", "api", "streamlit", "notebook"]),
    default="chat",
    help="Template type to use",
)
@click.option(
    "--llm",
    "-l",
    type=click.Choice(["openai", "ollama", "generic"]),
    default="openai",
    help="LLM provider to configure",
)
@click.option(
    "--detectors",
    "-d",
    multiple=True,
    help="Detectors to enable (default: toxicity,pii)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory (default: current directory)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing directory")
def app(app_name, template, llm, detectors, output, force):
    """Create a new AI safety application from template."""
    try:
        # Set up output directory
        if output:
            output_dir = Path(output) / app_name
        else:
            output_dir = Path.cwd() / app_name

        # Check if directory exists
        if output_dir.exists() and not force:
            click.echo(
                f"ERROR Directory {output_dir} already exists. Use --force to overwrite."
            )
            return 1

        # Set default detectors
        if not detectors:
            detectors = ["toxicity", "pii"]

        # Create template
        click.echo(f"Creating {template} application: {app_name}")
        click.echo(f"LLM provider: {llm}")
        click.echo(f"Detectors: {', '.join(detectors)}")
        click.echo(f"Output directory: {output_dir}")
        click.echo()

        creator = TemplateCreator()
        created_files = creator.create_app(
            app_name=app_name,
            template_type=template,
            llm_provider=llm,
            detectors=list(detectors),
            output_dir=output_dir,
            force=force,
        )

        # Display created files
        click.echo("Created files:")
        for file_path in created_files:
            rel_path = Path(file_path).relative_to(output_dir.parent)
            click.echo(f"  {rel_path}")

        click.echo()
        click.echo("Next steps:")
        click.echo(f"  cd {app_name}")
        click.echo("  pip install -r requirements.txt")

        if template == "streamlit":
            click.echo("  streamlit run app.py")
        elif template == "api":
            click.echo("  uvicorn main:app --reload")
        elif template == "notebook":
            click.echo("  jupyter notebook")
        else:
            click.echo("  python main.py")

    except Exception as e:
        click.echo(f"ERROR creating application: {e}", err=True)
        return 1


@create_command.command()
@click.argument("config_name")
@click.option("--detectors", "-d", multiple=True, help="Detectors to include in config")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def config(config_name, detectors, output):
    """Create a safety configuration file."""
    try:
        from ..core.detector_config import SafetyConfig

        # Create base config
        config_obj = SafetyConfig()
        config_dict = config_obj.to_dict()

        # Customize detectors if specified
        if detectors:
            # Disable all detectors first
            for detector in config_dict["detectors"]:
                config_dict["detectors"][detector]["enabled"] = False

            # Enable specified detectors
            for detector in detectors:
                if detector in config_dict["detectors"]:
                    config_dict["detectors"][detector]["enabled"] = True
                else:
                    click.echo(f"WARNING Unknown detector: {detector}")

        # Set output path
        if output:
            output_path = Path(output)
        else:
            output_path = Path.cwd() / f"{config_name}_config.yml"

        # Save config
        config_obj.config = config_dict
        config_obj.save(output_path)

        click.echo(f"OK Created configuration: {output_path}")

        if detectors:
            click.echo(f"Enabled detectors: {', '.join(detectors)}")

    except Exception as e:
        click.echo(f"ERROR creating config: {e}", err=True)
        return 1


@create_command.command()
def list_templates():
    """List available application templates."""
    templates = {
        "chat": "Interactive chat application with safety protection",
        "api": "FastAPI server with safety endpoints",
        "streamlit": "Streamlit web app with safety dashboard",
        "notebook": "Jupyter notebook with safety examples",
    }

    click.echo("Available templates:")
    click.echo()

    for name, description in templates.items():
        click.echo(f"  {name}")
        click.echo(f"     {description}")
        click.echo()


@create_command.command()
def list_llms():
    """List supported LLM providers."""
    providers = {
        "openai": "OpenAI GPT models (ChatGPT, GPT-4)",
        "ollama": "Local models via Ollama",
        "generic": "Generic LLM interface (bring your own)",
    }

    click.echo("Supported LLM providers:")
    click.echo()

    for name, description in providers.items():
        click.echo(f"  {name}")
        click.echo(f"     {description}")
        click.echo()


def create_app_cli():
    """Entry point for ai-safety-create command."""
    create_command()
