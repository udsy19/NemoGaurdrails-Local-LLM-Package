"""
CLI commands for model management.
"""

import asyncio
from pathlib import Path

import click

from ..core.model_manager import ModelManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def models_command():
    """Manage AI safety models."""
    pass


@models_command.command()
@click.option(
    "--detector", "-d", multiple=True, help="Download models for specific detectors"
)
@click.option(
    "--all", "download_all", is_flag=True, help="Download all available models"
)
@click.option("--cache-dir", type=click.Path(), help="Custom cache directory")
def download(detector, download_all, cache_dir):
    """Download AI safety models."""

    async def _download():
        try:
            manager = ModelManager(cache_dir=cache_dir)
            await manager.initialize()

            # Get available detectors
            available = manager.get_available_detectors()

            if download_all:
                detectors_to_download = list(available.keys())
            elif detector:
                detectors_to_download = list(detector)
                # Validate detector names
                invalid = [d for d in detectors_to_download if d not in available]
                if invalid:
                    click.echo(f"ERROR Unknown detectors: {', '.join(invalid)}")
                    return 1
            else:
                click.echo("ERROR Specify --detector or --all")
                return 1

            click.echo(f"Downloading models for: {', '.join(detectors_to_download)}")
            click.echo()

            # Download models
            for detector_name in detectors_to_download:
                info = available[detector_name]

                if info["model_type"] in ["patterns", "heuristics"]:
                    click.echo(f"OK {detector_name}: No download needed (built-in)")
                    continue

                click.echo(f"DOWNLOADING {detector_name}: {info['default_model']}")

                try:
                    model_path = await manager.ensure_model_available(detector_name)
                    click.echo(f"OK {detector_name}: Ready")
                except Exception as e:
                    click.echo(f"ERROR {detector_name}: {e}")

            await manager.cleanup()

        except Exception as e:
            click.echo(f"ERROR: {e}", err=True)
            return 1

    return asyncio.run(_download())


@models_command.command()
@click.option("--cache-dir", type=click.Path(), help="Custom cache directory")
def list(cache_dir):
    """List available models and their status."""

    async def _list():
        try:
            manager = ModelManager(cache_dir=cache_dir)
            await manager.initialize()

            detectors = manager.get_available_detectors()

            click.echo("Available AI Safety Models:")
            click.echo()

            for name, info in detectors.items():
                status = "CACHED" if info["cached"] else "NOT_CACHED"
                loaded = "LOADED" if info["loaded"] else "NOT_LOADED"

                click.echo(f"{status} {loaded} {name}")
                click.echo(f"     Model: {info['default_model']}")
                click.echo(f"     Type: {info['model_type']}")

                if info["required_packages"]:
                    packages = ", ".join(info["required_packages"])
                    click.echo(f"     Requires: {packages}")

                if info["cached"] and info["model_path"]:
                    click.echo(f"     Path: {info['model_path']}")

                click.echo()

            # Cache info
            cache_info = manager.get_cache_size()
            click.echo(
                f"Cache size: {cache_info['total_size_mb']} MB ({cache_info['file_count']} files)"
            )
            click.echo(f"Cache directory: {manager.cache_dir}")

            await manager.cleanup()

        except Exception as e:
            click.echo(f"ERROR: {e}", err=True)
            return 1

    return asyncio.run(_list())


@models_command.command()
@click.option("--detector", "-d", help="Clear cache for specific detector")
@click.option("--all", "clear_all", is_flag=True, help="Clear entire cache")
@click.option("--cache-dir", type=click.Path(), help="Custom cache directory")
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear_cache(detector, clear_all, cache_dir):
    """Clear model cache."""

    async def _clear():
        try:
            manager = ModelManager(cache_dir=cache_dir)
            await manager.initialize()

            if clear_all:
                result = await manager.clear_cache()
                click.echo("OK Cleared entire model cache")
            elif detector:
                result = await manager.clear_cache(detector)
                if result["status"] == "success":
                    click.echo(f"OK Cleared cache for {detector}")
                else:
                    click.echo(f"ERROR: {result.get('error', 'Unknown error')}")
            else:
                click.echo("ERROR Specify --detector or --all")
                return 1

            await manager.cleanup()

        except Exception as e:
            click.echo(f"ERROR: {e}", err=True)
            return 1

    return asyncio.run(_clear())


@models_command.command()
@click.option("--cache-dir", type=click.Path(), help="Custom cache directory")
def info(cache_dir):
    """Show detailed model information."""

    async def _info():
        try:
            manager = ModelManager(cache_dir=cache_dir)
            await manager.initialize()

            # Model registry info
            click.echo("Model Registry:")
            click.echo()

            for detector, info in manager.model_registry.items():
                click.echo(f"DETECTOR {detector}")
                click.echo(f"   Default model: {info['default_model']}")
                click.echo(f"   Type: {info['model_type']}")
                click.echo(
                    f"   Packages: {', '.join(info['required_packages']) if info['required_packages'] else 'None'}"
                )
                click.echo()

            # Cache information
            cache_info = manager.get_cache_size()
            health = await manager.health_check()

            click.echo("Cache Information:")
            click.echo(f"  Directory: {manager.cache_dir}")
            click.echo(f"  Size: {cache_info['total_size_mb']} MB")
            click.echo(f"  Files: {cache_info['file_count']}")
            click.echo()

            click.echo("Status:")
            click.echo(f"  Healthy: {'OK' if health['healthy'] else 'ERROR'}")
            click.echo(f"  Initialized: {'OK' if health['initialized'] else 'ERROR'}")
            click.echo(f"  Total detectors: {health['total_detectors']}")
            click.echo(f"  Loaded models: {health['loaded_detectors']}")
            click.echo(f"  Cached models: {health['cached_models']}")

            await manager.cleanup()

        except Exception as e:
            click.echo(f"ERROR: {e}", err=True)
            return 1

    return asyncio.run(_info())


def models_cli():
    """Entry point for ai-safety-models command."""
    models_command()
