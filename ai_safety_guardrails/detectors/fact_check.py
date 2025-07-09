"""
Fact check detector.

Identifies content that may need fact-checking using heuristic analysis.
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger
from .base_detector import BaseDetector, DetectionResult

logger = get_logger(__name__)


class FactCheckDetector(BaseDetector):
    """
    Detects content that may need fact-checking.

    Uses heuristic analysis to identify:
    - Statistical claims
    - Factual assertions
    - News-like content
    - Claims requiring verification
    """

    def __init__(self, **kwargs):
        super().__init__("fact_check", **kwargs)

        # Indicators of factual content
        self.factual_indicators = [
            "according to",
            "studies show",
            "research indicates",
            "statistics reveal",
            "data shows",
            "evidence suggests",
            "proven fact",
            "scientific evidence",
            "documented case",
            "survey found",
            "report states",
            "experts say",
            "analysis reveals",
            "findings show",
            "study confirms",
        ]

        # Statistical patterns
        self.stat_patterns = [
            r"\d+%",  # Percentages
            r"\d+\s*out\s*of\s*\d+",  # X out of Y
            r"\d+\s*in\s*\d+",  # X in Y
            r"\$\d+",  # Dollar amounts
            r"\d+\s*(million|billion|trillion)",  # Large numbers
            r"\d+\s*(times|fold)\s*(more|less|higher|lower)",  # Comparisons
        ]

    async def load_model(self) -> None:
        """Load fact-check detector (mainly pattern compilation)."""
        try:
            logger.info("Loading fact-check detector")

            # Compile statistical patterns
            self.compiled_stat_patterns = [
                re.compile(pattern, re.IGNORECASE) for pattern in self.stat_patterns
            ]

            self.is_loaded = True
            logger.info("Fact-check detector loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load fact-check detector: {e}")
            raise ModelLoadException(f"Failed to load fact-check detector: {e}")

    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect content that may need fact-checking.

        Args:
            text: Text to analyze
            context: Optional context (threshold can be specified here)

        Returns:
            DetectionResult with fact-check analysis
        """
        try:
            if not text.strip():
                return DetectionResult(
                    blocked=False, confidence=0.0, reason="Empty text"
                )

            # Get threshold from context or use default
            threshold = 0.5
            if context and "threshold" in context:
                threshold = context["threshold"]
            elif "threshold" in self.config:
                threshold = self.config["threshold"]

            text_lower = text.lower()

            # Check for factual indicators
            factual_claims = []
            for indicator in self.factual_indicators:
                if indicator in text_lower:
                    factual_claims.append(indicator)

            # Check for statistical patterns
            statistical_patterns = []
            for pattern in self.compiled_stat_patterns:
                matches = pattern.findall(text)
                statistical_patterns.extend(matches)

            # Additional heuristics
            has_numbers = bool(re.search(r"\d+", text))
            has_factual_language = len(factual_claims) > 0
            has_statistics = len(statistical_patterns) > 0
            has_definitive_claims = self._check_definitive_claims(text)
            has_comparative_claims = self._check_comparative_claims(text)

            # Calculate confidence
            confidence = 0.0

            if has_factual_language:
                confidence += 0.3
            if has_statistics:
                confidence += 0.4
            if has_definitive_claims:
                confidence += 0.2
            if has_comparative_claims:
                confidence += 0.2
            if has_numbers and has_factual_language:
                confidence += 0.2

            confidence = min(confidence, 1.0)
            needs_fact_check = confidence > threshold

            return DetectionResult(
                blocked=needs_fact_check,
                confidence=confidence,
                reason=(
                    f"Content may need fact-checking (confidence: {confidence:.2f})"
                    if needs_fact_check
                    else None
                ),
                details={
                    "factual_claims": factual_claims,
                    "statistical_patterns": statistical_patterns,
                    "has_numbers": has_numbers,
                    "has_factual_language": has_factual_language,
                    "has_statistics": has_statistics,
                    "has_definitive_claims": has_definitive_claims,
                    "has_comparative_claims": has_comparative_claims,
                    "threshold": threshold,
                },
            )

        except Exception as e:
            logger.error(f"Fact-check detection failed: {e}")
            raise DetectorException(f"Fact-check detection failed: {e}")

    def _check_definitive_claims(self, text: str) -> bool:
        """Check for definitive claims that should be verified."""
        definitive_patterns = [
            r"\bis\s+proven\b",
            r"\bis\s+fact\b",
            r"\bis\s+true\b",
            r"\bis\s+false\b",
            r"\bproven\s+to\s+be\b",
            r"\bknown\s+to\s+be\b",
            r"\bcertainly\b",
            r"\bdefinitely\b",
            r"\bundoubtedly\b",
        ]

        for pattern in definitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_comparative_claims(self, text: str) -> bool:
        """Check for comparative claims that should be verified."""
        comparative_patterns = [
            r"\bmore\s+than\b",
            r"\bless\s+than\b",
            r"\bhigher\s+than\b",
            r"\blower\s+than\b",
            r"\bincreased\s+by\b",
            r"\bdecreased\s+by\b",
            r"\bcompared\s+to\b",
            r"\bversus\b",
            r"\bvs\.?\b",
        ]

        for pattern in comparative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    async def cleanup(self) -> None:
        """Clean up resources."""
        await super().cleanup()
        self.compiled_stat_patterns = []
