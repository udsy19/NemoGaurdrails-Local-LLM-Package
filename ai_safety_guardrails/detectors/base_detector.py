"""
Base detector class for AI safety guardrails.

This module provides the abstract base class that all detectors must inherit from.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Standard detection result format."""

    blocked: bool
    confidence: float
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    detector_name: Optional[str] = None
    processing_time: Optional[float] = None


class BaseDetector(ABC):
    """
    Abstract base class for all AI safety detectors.

    All detectors must implement the detect method and follow the standard
    interface for initialization, model loading, and detection.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the detector.

        Args:
            name: The detector name/identifier
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.is_loaded = False
        self.config = kwargs
        self.model_load_time = None
        self.total_detections = 0
        self.successful_detections = 0

    @property
    def enabled(self) -> bool:
        """Check if the detector is enabled."""
        return self.config.get("enabled", True)

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the detector enabled status."""
        self.config["enabled"] = value

    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the detector's model and initialize resources.

        This method should be implemented by each detector to load
        any required models, download resources, etc.

        Raises:
            ModelLoadException: If model loading fails
        """
        pass

    @abstractmethod
    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Perform detection on the input text.

        Args:
            text: The text to analyze
            context: Optional context information (user_id, conversation_id, etc.)

        Returns:
            DetectionResult: Standardized detection result

        Raises:
            DetectorException: If detection fails
        """
        pass

    async def ensure_loaded(self) -> None:
        """Ensure the model is loaded before detection."""
        if not self.is_loaded:
            start_time = time.time()
            await self.load_model()
            self.model_load_time = time.time() - start_time
            logger.info(f"Detector {self.name} loaded in {self.model_load_time:.2f}s")

    async def safe_detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Safely perform detection with error handling and metrics.

        Args:
            text: The text to analyze
            context: Optional context information

        Returns:
            DetectionResult: Detection result or safe fallback
        """
        start_time = time.time()
        self.total_detections += 1

        try:
            await self.ensure_loaded()
            result = await self.detect(text, context)

            # Add processing time and detector name
            result.processing_time = time.time() - start_time
            result.detector_name = self.name

            self.successful_detections += 1
            return result

        except Exception as e:
            logger.error(f"Detection failed for {self.name}: {e}")

            # Return safe fallback result
            return DetectionResult(
                blocked=False,  # Fail open by default
                confidence=0.0,
                reason=f"Detection failed: {str(e)}",
                detector_name=self.name,
                processing_time=time.time() - start_time,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get detector performance metrics."""
        success_rate = (
            self.successful_detections / self.total_detections
            if self.total_detections > 0
            else 0.0
        )

        return {
            "name": self.name,
            "is_loaded": self.is_loaded,
            "model_load_time": self.model_load_time,
            "total_detections": self.total_detections,
            "successful_detections": self.successful_detections,
            "success_rate": success_rate,
            "config": self.config,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get detector health status."""
        return {
            "name": self.name,
            "healthy": self.is_loaded,
            "status": "loaded" if self.is_loaded else "not_loaded",
            "last_error": None,  # Could be extended to track last error
        }

    async def cleanup(self) -> None:
        """Cleanup detector resources."""
        logger.info(f"Cleaning up detector {self.name}")
        self.is_loaded = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', loaded={self.is_loaded})"
