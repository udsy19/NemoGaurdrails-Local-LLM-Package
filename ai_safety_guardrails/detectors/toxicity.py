"""
Toxicity detector using transformer models.

Detects harmful, offensive, or toxic content in text.
"""

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger
from .base_detector import BaseDetector, DetectionResult

logger = get_logger(__name__)


class ToxicityDetector(BaseDetector):
    """
    Detects toxic and harmful content using pre-trained transformer models.

    Uses models like 'martin-ha/toxic-comment-model' for high-accuracy
    toxicity detection.
    """

    def __init__(self, model_name: str = "martin-ha/toxic-comment-model", **kwargs):
        super().__init__("toxicity", **kwargs)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load_model(self) -> None:
        """Load the toxicity detection model."""
        try:
            logger.info(f"Loading toxicity model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
            )

            self.is_loaded = True
            logger.info("Toxicity model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load toxicity model: {e}")
            raise ModelLoadException(f"Failed to load toxicity model: {e}")

    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect toxicity in the input text.

        Args:
            text: Text to analyze
            context: Optional context (threshold can be specified here)

        Returns:
            DetectionResult with toxicity analysis
        """
        try:
            if not text.strip():
                return DetectionResult(
                    blocked=False, confidence=0.0, reason="Empty text"
                )

            # Get threshold from context or use default
            threshold = 0.7
            if context and "threshold" in context:
                threshold = context["threshold"]
            elif "threshold" in self.config:
                threshold = self.config["threshold"]

            # Run toxicity detection
            results = self.pipeline(text)

            # Handle different model output formats
            if isinstance(results[0], list):
                scores = {item["label"]: item["score"] for item in results[0]}
            else:
                scores = {results[0]["label"]: results[0]["score"]}

            # Determine toxicity
            toxic_score = scores.get("TOXIC", scores.get("toxic", 0.0))
            is_toxic = toxic_score > threshold

            return DetectionResult(
                blocked=is_toxic,
                confidence=toxic_score,
                reason=(
                    f"Toxic content detected (confidence: {toxic_score:.2f})"
                    if is_toxic
                    else None
                ),
                details={
                    "scores": scores,
                    "threshold": threshold,
                    "model": self.model_name,
                },
            )

        except Exception as e:
            logger.error(f"Toxicity detection failed: {e}")
            raise DetectorException(f"Toxicity detection failed: {e}")

    async def cleanup(self) -> None:
        """Clean up model resources."""
        await super().cleanup()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
