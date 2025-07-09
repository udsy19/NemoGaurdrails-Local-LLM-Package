"""
PII (Personally Identifiable Information) detector.

Detects and optionally redacts personal information like emails, phone numbers, etc.
"""

import re
from typing import Any, Dict, List, Optional

import spacy

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger
from .base_detector import BaseDetector, DetectionResult

logger = get_logger(__name__)


class PIIDetector(BaseDetector):
    """
    Detects personally identifiable information using NLP and pattern matching.

    Combines spaCy NER with regex patterns to identify:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - Names and organizations
    """

    def __init__(self, model_name: str = "en_core_web_sm", **kwargs):
        super().__init__("pii", **kwargs)
        self.model_name = model_name
        self.nlp = None

        # Common PII patterns
        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        }

    async def load_model(self) -> None:
        """Load the spaCy NER model."""
        try:
            logger.info(f"Loading PII detection model: {self.model_name}")

            self.nlp = spacy.load(self.model_name)
            self.is_loaded = True

            logger.info("PII detection model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load PII model: {e}")
            raise ModelLoadException(f"Failed to load PII model: {e}")

    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect PII in the input text.

        Args:
            text: Text to analyze
            context: Optional context (sensitivity can be specified here)

        Returns:
            DetectionResult with PII analysis
        """
        try:
            if not text.strip():
                return DetectionResult(
                    blocked=False, confidence=0.0, reason="Empty text"
                )

            # Get sensitivity from context or use default
            sensitivity = "medium"
            if context and "sensitivity" in context:
                sensitivity = context["sensitivity"]
            elif "sensitivity" in self.config:
                sensitivity = self.config["sensitivity"]

            # NER detection
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                # Filter entities based on sensitivity
                if self._should_flag_entity(ent, sensitivity):
                    entities.append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "confidence": 1.0,
                        }
                    )

            # Pattern-based detection
            pattern_matches = []
            for pattern_name, pattern in self.patterns.items():
                matches = pattern.finditer(text)
                for match in matches:
                    pattern_matches.append(
                        {
                            "text": match.group(),
                            "type": pattern_name,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 1.0,
                        }
                    )

            has_pii = len(entities) > 0 or len(pattern_matches) > 0
            confidence = min(len(entities) * 0.5 + len(pattern_matches) * 0.7, 1.0)

            return DetectionResult(
                blocked=has_pii,
                confidence=confidence,
                reason=(
                    f"PII detected: {len(entities)} entities, {len(pattern_matches)} patterns"
                    if has_pii
                    else None
                ),
                details={
                    "entities": entities,
                    "patterns": pattern_matches,
                    "sensitivity": sensitivity,
                    "redact_available": True,
                },
            )

        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            raise DetectorException(f"PII detection failed: {e}")

    def _should_flag_entity(self, ent, sensitivity: str) -> bool:
        """Determine if an entity should be flagged based on sensitivity."""
        if sensitivity == "low":
            # Only flag obvious PII
            return ent.label_ in ["PERSON", "ORG"] and len(ent.text) > 5
        elif sensitivity == "high":
            # Flag more aggressively
            return ent.label_ in ["PERSON", "ORG", "DATE", "MONEY", "CARDINAL"]
        else:  # medium
            # Flag clear personal information
            if ent.label_ == "PERSON":
                # Skip common names/historical figures
                return len(ent.text.split()) > 1 or (
                    len(ent.text.split()) == 1 and ent.text[0].islower()
                )
            elif ent.label_ == "ORG" and len(ent.text) > 3:
                return True
            return False

    def redact_pii(
        self, text: str, detection_result: DetectionResult, redaction_char: str = "*"
    ) -> str:
        """
        Redact PII from text based on detection results.

        Args:
            text: Original text
            detection_result: Result from detect() method
            redaction_char: Character to use for redaction

        Returns:
            Text with PII redacted
        """
        if not detection_result.details:
            return text

        redacted_text = text
        offset = 0

        # Sort all matches by start position (reverse order for replacement)
        all_matches = []

        if "entities" in detection_result.details:
            for entity in detection_result.details["entities"]:
                all_matches.append((entity["start"], entity["end"], entity["text"]))

        if "patterns" in detection_result.details:
            for pattern in detection_result.details["patterns"]:
                all_matches.append((pattern["start"], pattern["end"], pattern["text"]))

        # Sort by start position (descending to avoid offset issues)
        all_matches.sort(key=lambda x: x[0], reverse=True)

        # Replace matches with redaction
        for start, end, original_text in all_matches:
            redaction = redaction_char * len(original_text)
            redacted_text = redacted_text[:start] + redaction + redacted_text[end:]

        return redacted_text

    async def cleanup(self) -> None:
        """Clean up model resources."""
        await super().cleanup()
        self.nlp = None
