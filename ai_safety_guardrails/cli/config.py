"""
CLI commands for configuration management.
"""

from pathlib import Path

import click
import yaml

from ..core.detector_config import SafetyConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def config_command():
    """Manage safety configuration."""
    pass


@config_command.command()
@click.argument("config_file", type=click.Path(exists=True))
def show(config_file):
    """Display configuration file contents."""
    try:
        config = SafetyConfig.from_file(config_file)

        click.echo(f"Configuration: {config_file}")
        click.echo("=" * 50)
        click.echo(config.to_yaml())

    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        return 1


@config_command.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file):
    """Validate a configuration file."""
    try:
        config = SafetyConfig.from_file(config_file)
        issues = config.validate()

        if issues:
            click.echo(f"ERROR Configuration file {config_file} has issues:")
            for issue in issues:
                click.echo(f"  - {issue}")
            return 1
        else:
            click.echo(f"OK Configuration file {config_file} is valid")

    except Exception as e:
        click.echo(f"ERROR validating {config_file}: {e}", err=True)
        return 1


@config_command.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: ai_safety_config.yml)",
)
@click.option(
    "--detectors", "-d", multiple=True, help="Detectors to enable (default: all)"
)
def create_template(output, detectors):
    """Create a template configuration file."""
    try:
        config = SafetyConfig()

        # Customize enabled detectors if specified
        if detectors:
            config_dict = config.to_dict()

            # Disable all detectors first
            for detector in config_dict["detectors"]:
                config_dict["detectors"][detector]["enabled"] = False

            # Enable specified detectors
            for detector in detectors:
                if detector in config_dict["detectors"]:
                    config_dict["detectors"][detector]["enabled"] = True
                else:
                    click.echo(f"WARNING Unknown detector: {detector}")

            config.config = config_dict

        # Set output path
        if output:
            output_path = Path(output)
        else:
            output_path = Path.cwd() / "ai_safety_config.yml"

        config.save(output_path)
        click.echo(f"OK Created template configuration: {output_path}")

        if detectors:
            click.echo(f"Enabled detectors: {', '.join(detectors)}")

    except Exception as e:
        click.echo(f"ERROR creating template: {e}", err=True)
        return 1


@config_command.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--detector", "-d", required=True, help="Detector to enable/disable")
@click.option("--enable/--disable", default=True, help="Enable or disable the detector")
@click.option("--threshold", "-t", type=float, help="Set detector threshold (0.0-1.0)")
@click.option(
    "--sensitivity",
    "-s",
    type=click.Choice(["low", "medium", "high"]),
    help="Set detector sensitivity",
)
def update(config_file, detector, enable, threshold, sensitivity):
    """Update detector settings in configuration."""
    try:
        config = SafetyConfig.from_file(config_file)
        config_dict = config.to_dict()

        # Check if detector exists
        if detector not in config_dict["detectors"]:
            click.echo(f"ERROR Unknown detector: {detector}")
            available = list(config_dict["detectors"].keys())
            click.echo(f"Available detectors: {', '.join(available)}")
            return 1

        # Update settings
        detector_config = config_dict["detectors"][detector]
        detector_config["enabled"] = enable

        if threshold is not None:
            detector_config["threshold"] = threshold

        if sensitivity is not None:
            detector_config["sensitivity"] = sensitivity

        # Save updated config
        config.config = config_dict
        config.save(config_file)

        # Show what was updated
        status = "enabled" if enable else "disabled"
        click.echo(f"OK Detector '{detector}' {status}")

        if threshold is not None:
            click.echo(f"   Threshold set to: {threshold}")

        if sensitivity is not None:
            click.echo(f"   Sensitivity set to: {sensitivity}")

    except Exception as e:
        click.echo(f"ERROR updating config: {e}", err=True)
        return 1


@config_command.command()
@click.argument("config_file", type=click.Path(exists=True))
def list_detectors(config_file):
    """List all detectors in configuration."""
    try:
        config = SafetyConfig.from_file(config_file)
        config_dict = config.to_dict()

        click.echo(f"Detectors in {config_file}:")
        click.echo()

        for name, detector_config in config_dict["detectors"].items():
            status = "ENABLED" if detector_config.get("enabled", True) else "DISABLED"

            click.echo(f"{status} {name}")

            if "threshold" in detector_config:
                click.echo(f"   Threshold: {detector_config['threshold']}")

            if "sensitivity" in detector_config:
                click.echo(f"   Sensitivity: {detector_config['sensitivity']}")

            if "model" in detector_config:
                click.echo(f"   Model: {detector_config['model']}")

            click.echo()

    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        return 1


@config_command.command()
@click.argument("source_config", type=click.Path(exists=True))
@click.argument("target_config", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file (default: merged_config.yml)"
)
def merge(source_config, target_config, output):
    """Merge two configuration files."""
    try:
        # Load both configs
        source = SafetyConfig.from_file(source_config)
        target = SafetyConfig.from_file(target_config)

        source_dict = source.to_dict()
        target_dict = target.to_dict()

        # Merge configs (target takes precedence)
        merged_dict = source_dict.copy()

        # Merge detector configs
        for detector, config in target_dict.get("detectors", {}).items():
            if detector in merged_dict["detectors"]:
                merged_dict["detectors"][detector].update(config)
            else:
                merged_dict["detectors"][detector] = config

        # Merge other sections
        for section in ["models", "safety", "logging"]:
            if section in target_dict:
                merged_dict[section].update(target_dict[section])

        # Create merged config
        merged_config = SafetyConfig(merged_dict)

        # Set output path
        if output:
            output_path = Path(output)
        else:
            output_path = Path.cwd() / "merged_config.yml"

        merged_config.save(output_path)
        click.echo(f"OK Merged configuration saved to: {output_path}")

        # Validate merged config
        issues = merged_config.validate()
        if issues:
            click.echo("WARNING Merged configuration has issues:")
            for issue in issues:
                click.echo(f"  - {issue}")

    except Exception as e:
        click.echo(f"ERROR merging configs: {e}", err=True)
        return 1


# Alias for backward compatibility with tests
config_cli = config_command
