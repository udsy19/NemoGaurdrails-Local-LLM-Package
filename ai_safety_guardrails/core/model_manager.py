"""
Model manager for AI Safety Guardrails.

Handles model downloading, caching, and lifecycle management.
"""

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ModelLoadException
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manages AI models for detectors.

    Provides:
    - Lazy model downloading
    - Local caching
    - Model lifecycle management
    - Health monitoring
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory for model caching. Defaults to ~/.ai_safety_models
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.ai_safety_models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.is_initialized = False
        self.available_models: dict[str, dict] = {}
        self.loaded_models: dict[str, Any] = {}

        # Model registry - maps detector names to their model requirements
        self.model_registry = {
            "toxicity": {
                "default_model": "martin-ha/toxic-comment-model",
                "model_type": "transformers",
                "required_packages": ["transformers", "torch"],
            },
            "pii": {
                "default_model": "en_core_web_sm",
                "model_type": "spacy",
                "required_packages": ["spacy"],
            },
            "topics": {
                "default_model": "all-MiniLM-L6-v2",
                "model_type": "sentence_transformers",
                "required_packages": ["sentence-transformers"],
            },
            "prompt_injection": {
                "default_model": "patterns",
                "model_type": "patterns",
                "required_packages": [],
            },
            "fact_check": {
                "default_model": "heuristics",
                "model_type": "heuristics",
                "required_packages": [],
            },
            "spam": {
                "default_model": "patterns",
                "model_type": "patterns",
                "required_packages": [],
            },
        }

        logger.info(f"ModelManager initialized with cache dir: {self.cache_dir}")

    async def initialize(self) -> None:
        """Initialize the model manager."""
        try:
            logger.info("Initializing ModelManager")

            # Load model cache info
            await self._load_cache_info()

            # Check available models
            await self._check_available_models()

            self.is_initialized = True
            logger.info("ModelManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {e}")
            raise ModelLoadException(f"ModelManager initialization failed: {e}")

    async def _load_cache_info(self) -> None:
        """Load cached model information."""
        cache_info_path = self.cache_dir / "cache_info.json"

        if cache_info_path.exists():
            try:
                with open(cache_info_path, "r") as f:
                    cache_info = json.load(f)
                    self.available_models = cache_info.get("available_models", {})
                    logger.info(
                        f"Loaded cache info for {len(self.available_models)} models"
                    )
            except Exception as e:
                logger.warning(f"Failed to load cache info: {e}")
                self.available_models = {}
        else:
            self.available_models = {}

    async def _save_cache_info(self) -> None:
        """Save model cache information."""
        cache_info_path = self.cache_dir / "cache_info.json"

        try:
            cache_info = {
                "available_models": self.available_models,
                "last_updated": str(asyncio.get_event_loop().time()),
            }

            with open(cache_info_path, "w") as f:
                json.dump(cache_info, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save cache info: {e}")

    async def _check_available_models(self) -> None:
        """Check which models are available locally."""
        for detector_name, model_info in self.model_registry.items():
            model_name: str = str(model_info["default_model"])
            model_path = self.cache_dir / detector_name / model_name.replace("/", "_")

            if model_path.exists():
                self.available_models[detector_name] = {
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "cached": True,
                    "model_type": model_info["model_type"],
                }
            else:
                self.available_models[detector_name] = {
                    "model_name": model_name,
                    "model_path": None,
                    "cached": False,
                    "model_type": model_info["model_type"],
                }

    async def ensure_model_available(
        self, detector_name: str, model_name: Optional[str] = None
    ) -> str:
        """
        Ensure a model is available for a detector.

        Args:
            detector_name: Name of the detector
            model_name: Optional specific model name

        Returns:
            Path to the model or model identifier
        """
        if detector_name not in self.model_registry:
            raise ModelLoadException(f"Unknown detector: {detector_name}")

        # Use provided model name or default
        if model_name is None:
            model_name = str(self.model_registry[detector_name]["default_model"])

        model_info = self.model_registry[detector_name]
        model_type = model_info["model_type"]

        # For pattern-based and heuristic detectors, no download needed
        if model_type in ["patterns", "heuristics"]:
            return model_name

        # Check if model is already cached
        if (
            detector_name in self.available_models
            and self.available_models[detector_name]["cached"]
            and self.available_models[detector_name]["model_name"] == model_name
        ):
            return str(self.available_models[detector_name]["model_path"])

        # For transformers and sentence_transformers, return model name for lazy loading
        # The actual download happens when the model is first used
        if model_type in ["transformers", "sentence_transformers"]:
            await self._prepare_model_cache_dir(detector_name, model_name)
            return model_name

        # For spaCy models, check if installed
        if model_type == "spacy":
            try:
                import spacy

                spacy.load(model_name)
                logger.info(f"SpaCy model {model_name} is available")
                return model_name
            except OSError:
                logger.warning(
                    f"SpaCy model {model_name} not found. Install with: python -m spacy download {model_name}"
                )
                return model_name

        return model_name

    async def _prepare_model_cache_dir(
        self, detector_name: str, model_name: str
    ) -> None:
        """Prepare cache directory for a model."""
        cache_subdir = self.cache_dir / detector_name
        cache_subdir.mkdir(parents=True, exist_ok=True)

        # Update available models info
        self.available_models[detector_name] = {
            "model_name": model_name,
            "model_path": str(cache_subdir),
            "cached": False,  # Will be True after first download
            "model_type": self.model_registry[detector_name]["model_type"],
        }

    async def download_model(
        self, detector_name: str, show_progress: bool = True
    ) -> bool:
        """
        Explicitly download a model for a detector.

        Args:
            detector_name: Name of the detector
            show_progress: Whether to show download progress

        Returns:
            True if successful, False otherwise
        """
        try:
            if detector_name not in self.model_registry:
                raise ModelLoadException(f"Unknown detector: {detector_name}")

            model_info = self.model_registry[detector_name]
            model_name = str(model_info["default_model"])
            model_type = model_info["model_type"]

            # Skip pattern-based detectors
            if model_type in ["patterns", "heuristics"]:
                logger.info(f"Detector {detector_name} doesn't require model download")
                return True

            logger.info(f"Downloading model for {detector_name}: {model_name}")

            # Prepare cache directory
            await self._prepare_model_cache_dir(detector_name, model_name)
            cache_dir = self.cache_dir / detector_name

            if model_type == "transformers":
                # Download transformers model
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, cache_dir=cache_dir
                )

                # Mark as cached
                self.available_models[detector_name]["cached"] = True
                logger.info(f"Successfully downloaded transformers model: {model_name}")

            elif model_type == "sentence_transformers":
                # Download sentence transformers model
                from sentence_transformers import SentenceTransformer

                # This will download and cache the model
                model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

                # Mark as cached
                self.available_models[detector_name]["cached"] = True
                logger.info(
                    f"Successfully downloaded sentence transformer model: {model_name}"
                )

            elif model_type == "spacy":
                # For spaCy, we can't download programmatically in the same way
                # Just check if it's available
                import spacy

                try:
                    spacy.load(model_name)
                    self.available_models[detector_name]["cached"] = True
                    logger.info(f"SpaCy model {model_name} is available")
                except OSError:
                    logger.error(
                        f"SpaCy model {model_name} not found. Install with: python -m spacy download {model_name}"
                    )
                    return False

            # Save cache info
            await self._save_cache_info()
            return True

        except Exception as e:
            logger.error(f"Failed to download model for {detector_name}: {e}")
            return False

    def get_available_detectors(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available detectors and their models."""
        detector_info = {}

        for detector_name, model_info in self.model_registry.items():
            available_info = self.available_models.get(detector_name, {})

            detector_info[detector_name] = {
                "default_model": model_info["default_model"],
                "model_type": model_info["model_type"],
                "required_packages": model_info["required_packages"],
                "cached": available_info.get("cached", False),
                "model_path": available_info.get("model_path"),
                "loaded": detector_name in self.loaded_models,
            }

        return detector_info

    def register_model_loaded(self, detector_name: str, model_instance: Any) -> None:
        """Register that a model has been loaded."""
        self.loaded_models[detector_name] = {
            "model_instance": model_instance,
            "load_time": asyncio.get_event_loop().time(),
        }
        logger.info(f"Registered loaded model for detector: {detector_name}")

    def unload_model(self, detector_name: str) -> None:
        """Unload a model from memory."""
        if detector_name in self.loaded_models:
            del self.loaded_models[detector_name]
            logger.info(f"Unloaded model for detector: {detector_name}")

    async def health_check(self) -> Dict[str, Any]:
        """Check model manager health."""
        try:
            detector_info = self.get_available_detectors()
            total_detectors = len(detector_info)
            loaded_detectors = len(self.loaded_models)
            cached_models = sum(1 for info in detector_info.values() if info["cached"])

            return {
                "healthy": True,
                "initialized": self.is_initialized,
                "total_detectors": total_detectors,
                "loaded_detectors": loaded_detectors,
                "cached_models": cached_models,
                "cache_dir": str(self.cache_dir),
                "available_detectors": list(detector_info.keys()),
            }

        except Exception as e:
            logger.error(f"Model manager health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    def get_cache_size(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        try:
            total_size = 0
            file_count = 0

            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.exists():
                        total_size += file_path.stat().st_size
                        file_count += 1

            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "cache_dir": str(self.cache_dir),
            }

        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
            return {"error": str(e)}

    async def clear_cache(self, detector_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear model cache.

        Args:
            detector_name: Optional specific detector to clear. If None, clears all.

        Returns:
            Information about what was cleared
        """
        try:
            if detector_name:
                # Clear specific detector cache
                detector_cache_dir = self.cache_dir / detector_name
                if detector_cache_dir.exists():
                    shutil.rmtree(detector_cache_dir)
                    logger.info(f"Cleared cache for detector: {detector_name}")

                # Update available models
                if detector_name in self.available_models:
                    self.available_models[detector_name]["cached"] = False
                    self.available_models[detector_name]["model_path"] = None

                return {"cleared": detector_name, "status": "success"}
            else:
                # Clear entire cache
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)

                # Reset available models
                for detector_name in self.available_models:
                    self.available_models[detector_name]["cached"] = False
                    self.available_models[detector_name]["model_path"] = None

                logger.info("Cleared entire model cache")
                return {"cleared": "all", "status": "success"}

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up model manager resources."""
        logger.info("Cleaning up ModelManager")

        # Save cache info
        await self._save_cache_info()

        # Clear loaded models
        self.loaded_models.clear()

        self.is_initialized = False
