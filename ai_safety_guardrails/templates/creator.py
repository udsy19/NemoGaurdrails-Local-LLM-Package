"""
Template creator for generating AI safety applications.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, Template

from ..core.detector_config import SafetyConfig
from ..utils.exceptions import TemplateException
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TemplateCreator:
    """
    Creates applications from templates.

    Supports various template types:
    - chat: Interactive chat application
    - api: FastAPI server with safety endpoints
    - streamlit: Streamlit web app
    - notebook: Jupyter notebook examples
    """

    def __init__(self):
        self.templates_dir = Path(__file__).parent
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def create_app(
        self,
        app_name: str,
        template_type: str,
        llm_provider: str,
        detectors: List[str],
        output_dir: Path,
        force: bool = False,
    ) -> List[str]:
        """
        Create an application from a template.

        Args:
            app_name: Name of the application
            template_type: Type of template (chat, api, streamlit, notebook)
            llm_provider: LLM provider (openai, ollama, generic)
            detectors: List of detectors to enable
            output_dir: Output directory
            force: Overwrite existing directory

        Returns:
            List of created file paths
        """
        try:
            # Validate inputs
            self._validate_inputs(template_type, llm_provider, detectors)

            # Prepare output directory
            if output_dir.exists() and not force:
                raise TemplateException(f"Directory {output_dir} already exists")

            if output_dir.exists() and force:
                shutil.rmtree(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate template context
            context = self._create_template_context(
                app_name, template_type, llm_provider, detectors
            )

            # Get template files
            template_files = self._get_template_files(template_type)

            # Create files
            created_files = []
            for template_file, output_file in template_files.items():
                output_path = output_dir / output_file
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if template_file.endswith(".j2"):
                    # Render Jinja2 template
                    self._render_template_file(template_file, output_path, context)
                else:
                    # Copy static file
                    src_path = self.templates_dir / template_type / template_file
                    shutil.copy2(src_path, output_path)

                created_files.append(str(output_path))

            logger.info(f"Created {template_type} application: {app_name}")
            return created_files

        except Exception as e:
            logger.error(f"Failed to create application: {e}")
            raise TemplateException(f"Template creation failed: {e}")

    def _validate_inputs(
        self, template_type: str, llm_provider: str, detectors: List[str]
    ):
        """Validate input parameters."""
        valid_templates = ["chat", "api", "streamlit", "notebook"]
        if template_type not in valid_templates:
            raise TemplateException(f"Invalid template type: {template_type}")

        valid_providers = ["openai", "ollama", "generic"]
        if llm_provider not in valid_providers:
            raise TemplateException(f"Invalid LLM provider: {llm_provider}")

        # Check if template directory exists
        template_dir = self.templates_dir / template_type
        if not template_dir.exists():
            raise TemplateException(f"Template directory not found: {template_dir}")

    def _create_template_context(
        self, app_name: str, template_type: str, llm_provider: str, detectors: List[str]
    ) -> Dict[str, Any]:
        """Create context for template rendering."""

        # Create safety configuration
        config = SafetyConfig()
        config_dict = config.to_dict()

        # Enable only specified detectors
        for detector in config_dict["detectors"]:
            config_dict["detectors"][detector]["enabled"] = detector in detectors

        # LLM provider specific settings
        llm_config = self._get_llm_config(llm_provider)

        # Dependencies based on template and LLM
        dependencies = self._get_dependencies(template_type, llm_provider)

        return {
            "app_name": app_name,
            "template_type": template_type,
            "llm_provider": llm_provider,
            "detectors": detectors,
            "safety_config": config_dict,
            "llm_config": llm_config,
            "dependencies": dependencies,
            "package_name": "ai-safety-guardrails",
        }

    def _get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get LLM provider specific configuration."""
        configs = {
            "openai": {
                "import_statement": "import openai",
                "client_init": 'openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))',
                "model_default": "gpt-3.5-turbo",
                "env_vars": ["OPENAI_API_KEY"],
                "chat_method": "chat.completions.create",
                "response_accessor": "choices[0].message.content",
            },
            "ollama": {
                "import_statement": "import ollama",
                "client_init": 'ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))',
                "model_default": "llama2",
                "env_vars": ["OLLAMA_HOST"],
                "chat_method": "chat",
                "response_accessor": "message.content",
            },
            "generic": {
                "import_statement": "# Import your LLM library here",
                "client_init": "# Initialize your LLM client here",
                "model_default": "your-model-name",
                "env_vars": ["YOUR_API_KEY"],
                "chat_method": "your_chat_method",
                "response_accessor": "your_response_accessor",
            },
        }

        return configs.get(provider, configs["generic"])

    def _get_dependencies(self, template_type: str, llm_provider: str) -> List[str]:
        """Get dependencies for the template."""
        base_deps = ["ai-safety-guardrails"]

        # Template-specific dependencies
        template_deps = {
            "chat": [],
            "api": ["fastapi", "uvicorn[standard]"],
            "streamlit": ["streamlit", "plotly"],
            "notebook": ["jupyter", "ipywidgets"],
        }

        # LLM provider dependencies
        llm_deps = {
            "openai": ["openai"],
            "ollama": ["ollama"],
            "generic": [],
        }

        deps = (
            base_deps
            + template_deps.get(template_type, [])
            + llm_deps.get(llm_provider, [])
        )

        # Add common utilities
        deps.extend(["python-dotenv", "pyyaml"])

        return sorted(list(set(deps)))

    def _get_template_files(self, template_type: str) -> Dict[str, str]:
        """Get mapping of template files to output files."""
        files = {
            "chat": {
                "main.py.j2": "main.py",
                "config.yml.j2": "ai_safety_config.yml",
                "requirements.txt.j2": "requirements.txt",
                ".env.example.j2": ".env.example",
                "README.md.j2": "README.md",
            },
            "api": {
                "main.py.j2": "main.py",
                "config.yml.j2": "ai_safety_config.yml",
                "requirements.txt.j2": "requirements.txt",
                ".env.example.j2": ".env.example",
                "README.md.j2": "README.md",
            },
            "streamlit": {
                "app.py.j2": "app.py",
                "config.yml.j2": "ai_safety_config.yml",
                "requirements.txt.j2": "requirements.txt",
                ".env.example.j2": ".env.example",
                "README.md.j2": "README.md",
            },
            "notebook": {
                "safety_examples.ipynb.j2": "safety_examples.ipynb",
                "config.yml.j2": "ai_safety_config.yml",
                "requirements.txt.j2": "requirements.txt",
                ".env.example.j2": ".env.example",
                "README.md.j2": "README.md",
            },
        }

        return files.get(template_type, {})

    def _render_template_file(
        self, template_file: str, output_path: Path, context: Dict[str, Any]
    ):
        """Render a Jinja2 template file."""
        try:
            template = self.jinja_env.get_template(
                f"{context['template_type']}/{template_file}"
            )
            rendered = template.render(**context)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered)

        except Exception as e:
            raise TemplateException(f"Failed to render template {template_file}: {e}")


def create_app(
    app_name: str,
    template_type: str = "chat",
    llm_provider: str = "openai",
    detectors: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> List[str]:
    """
    Convenience function for creating applications.

    Args:
        app_name: Name of the application
        template_type: Type of template (default: chat)
        llm_provider: LLM provider (default: openai)
        detectors: List of detectors (default: ['toxicity', 'pii'])
        output_dir: Output directory (default: current directory)
        force: Overwrite existing directory

    Returns:
        List of created file paths
    """
    if detectors is None:
        detectors = ["toxicity", "pii"]

    if output_dir is None:
        output_dir = Path.cwd() / app_name

    creator = TemplateCreator()
    return creator.create_app(
        app_name=app_name,
        template_type=template_type,
        llm_provider=llm_provider,
        detectors=detectors,
        output_dir=output_dir,
        force=force,
    )
