"""
Topic detector using sentence transformers.

Classifies content topics and filters restricted categories.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger
from .base_detector import BaseDetector, DetectionResult

logger = get_logger(__name__)


class TopicsDetector(BaseDetector):
    """
    Detects and classifies content topics using sentence transformers.

    Can identify and block content related to:
    - Violence and harm
    - Hate speech
    - Illegal activities
    - Adult content
    - Custom restricted topics
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        super().__init__("topics", **kwargs)
        self.model_name = model_name
        self.model = None
        self.topic_embeddings: dict[str, Any] = {}

        # Predefined restricted topics with sample texts
        self.restricted_topics = {
            "violence": [
                "physical harm",
                "weapons",
                "assault",
                "fighting",
                "violence",
                "murder",
                "killing",
                "attack",
                "terrorism",
                "war",
            ],
            "hate_speech": [
                "racial slurs",
                "discrimination",
                "prejudice",
                "hatred",
                "racist",
                "sexist",
                "homophobic",
                "bigotry",
            ],
            "illegal_activities": [
                "drug dealing",
                "fraud",
                "hacking",
                "theft",
                "money laundering",
                "illegal substances",
                "criminal activities",
                "law breaking",
            ],
            "adult_content": [
                "explicit sexual content",
                "pornography",
                "adult material",
                "sexual acts",
                "inappropriate content",
            ],
        }

    async def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading topic detection model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            # Pre-compute embeddings for restricted topics
            self.topic_embeddings = {}
            for topic, examples in self.restricted_topics.items():
                embeddings = self.model.encode(examples)
                self.topic_embeddings[topic] = np.mean(embeddings, axis=0)

            self.is_loaded = True
            logger.info("Topic detection model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load topic detection model: {e}")
            raise ModelLoadException(f"Failed to load topic detection model: {e}")

    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect restricted topics in text.

        Args:
            text: Text to analyze
            context: Optional context (threshold, custom_topics can be specified)

        Returns:
            DetectionResult with topic analysis
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

            # Encode input text
            text_embedding = self.model.encode([text])[0]

            # Calculate similarities with restricted topics
            similarities = {}
            for topic, topic_embedding in self.topic_embeddings.items():
                similarity = np.dot(text_embedding, topic_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(topic_embedding)
                )
                similarities[topic] = float(similarity)

            # Find the most similar topic
            max_similarity = max(similarities.values()) if similarities else 0.0
            most_similar_topic = (
                max(similarities.items(), key=lambda x: x[1])[0]
                if similarities
                else None
            )

            has_restricted_topic = max_similarity > threshold

            return DetectionResult(
                blocked=has_restricted_topic,
                confidence=max_similarity,
                reason=(
                    f"Restricted topic detected: {most_similar_topic} (similarity: {max_similarity:.2f})"
                    if has_restricted_topic
                    else None
                ),
                details={
                    "similarities": similarities,
                    "most_similar_topic": most_similar_topic,
                    "max_similarity": max_similarity,
                    "threshold": threshold,
                    "model": self.model_name,
                },
            )

        except Exception as e:
            logger.error(f"Topic detection failed: {e}")
            raise DetectorException(f"Topic detection failed: {e}")

    def add_custom_topic(self, topic_name: str, examples: List[str]) -> None:
        """
        Add a custom restricted topic.

        Args:
            topic_name: Name of the topic
            examples: List of example texts for this topic
        """
        if not self.is_loaded:
            raise DetectorException("Model must be loaded before adding custom topics")

        embeddings = self.model.encode(examples)
        self.topic_embeddings[topic_name] = np.mean(embeddings, axis=0)

        logger.info(f"Added custom topic: {topic_name}")

    def remove_topic(self, topic_name: str) -> None:
        """Remove a topic from restricted list."""
        if topic_name in self.topic_embeddings:
            del self.topic_embeddings[topic_name]
            logger.info(f"Removed topic: {topic_name}")

    def get_topics(self) -> List[str]:
        """Get list of all configured topics."""
        return list(self.topic_embeddings.keys())

    async def cleanup(self) -> None:
        """Clean up model resources."""
        await super().cleanup()
        self.model = None
        self.topic_embeddings = {}
