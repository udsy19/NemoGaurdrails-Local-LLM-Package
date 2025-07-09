"""
Spam detector.

Identifies spam and unwanted promotional content.
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger
from .base_detector import BaseDetector, DetectionResult

logger = get_logger(__name__)


class SpamDetector(BaseDetector):
    """
    Detects spam and promotional content.

    Identifies:
    - Promotional language
    - Excessive capitalization
    - Spam keywords
    - Suspicious patterns
    """

    def __init__(self, **kwargs):
        super().__init__("spam", **kwargs)

        # Common spam indicators
        self.spam_indicators = [
            "click here",
            "buy now",
            "limited time",
            "act now",
            "urgent",
            "congratulations",
            "winner",
            "free money",
            "make money fast",
            "work from home",
            "get rich quick",
            "no risk",
            "guaranteed",
            "amazing deal",
            "special offer",
            "once in a lifetime",
            "don't miss out",
            "call now",
            "apply now",
            "order now",
            "sign up now",
            "subscribe now",
        ]

        # Promotional patterns
        self.promo_patterns = [
            r"\$\d+\s*(off|discount)",
            r"\d+%\s*off",
            r"free\s+\w+",
            r"save\s+\$\d+",
            r"only\s+\$\d+",
            r"starting\s+at\s+\$\d+",
        ]

    async def load_model(self) -> None:
        """Load spam detector (pattern compilation)."""
        try:
            logger.info("Loading spam detector")

            # Compile promotional patterns
            self.compiled_promo_patterns = [
                re.compile(pattern, re.IGNORECASE) for pattern in self.promo_patterns
            ]

            self.is_loaded = True
            logger.info("Spam detector loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load spam detector: {e}")
            raise ModelLoadException(f"Failed to load spam detector: {e}")

    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect spam content.

        Args:
            text: Text to analyze
            context: Optional context (threshold can be specified here)

        Returns:
            DetectionResult with spam analysis
        """
        try:
            if not text.strip():
                return DetectionResult(
                    blocked=False, confidence=0.0, reason="Empty text"
                )

            # Get threshold from context or use default
            threshold = 0.6
            if context and "threshold" in context:
                threshold = context["threshold"]
            elif "threshold" in self.config:
                threshold = self.config["threshold"]

            text_lower = text.lower()

            # Check for spam indicators
            found_indicators = []
            for indicator in self.spam_indicators:
                if indicator in text_lower:
                    found_indicators.append(indicator)

            # Check promotional patterns
            promo_matches = []
            for pattern in self.compiled_promo_patterns:
                matches = pattern.findall(text)
                promo_matches.extend(matches)

            # Additional spam characteristics
            excessive_caps = self._check_excessive_caps(text)
            excessive_punctuation = self._check_excessive_punctuation(text)
            suspicious_urls = self._check_suspicious_urls(text)
            repeated_words = self._check_repeated_words(text)

            # Calculate confidence
            confidence = 0.0

            # Spam indicators
            confidence += len(found_indicators) * 0.15

            # Promotional patterns
            confidence += len(promo_matches) * 0.2

            # Formatting issues
            if excessive_caps:
                confidence += 0.25
            if excessive_punctuation:
                confidence += 0.15
            if suspicious_urls:
                confidence += 0.3
            if repeated_words:
                confidence += 0.2

            confidence = min(confidence, 1.0)
            is_spam = confidence > threshold

            return DetectionResult(
                blocked=is_spam,
                confidence=confidence,
                reason=(
                    f"Spam content detected (confidence: {confidence:.2f})"
                    if is_spam
                    else None
                ),
                details={
                    "spam_indicators": found_indicators,
                    "promotional_patterns": promo_matches,
                    "excessive_caps": excessive_caps,
                    "excessive_punctuation": excessive_punctuation,
                    "suspicious_urls": suspicious_urls,
                    "repeated_words": repeated_words,
                    "threshold": threshold,
                },
            )

        except Exception as e:
            logger.error(f"Spam detection failed: {e}")
            raise DetectorException(f"Spam detection failed: {e}")

    def _check_excessive_caps(self, text: str) -> bool:
        """Check for excessive capitalization."""
        if not text:
            return False

        caps_count = len(re.findall(r"[A-Z]", text))
        total_letters = len(re.findall(r"[A-Za-z]", text))

        if total_letters == 0:
            return False

        caps_ratio = caps_count / total_letters
        return caps_ratio > 0.3

    def _check_excessive_punctuation(self, text: str) -> bool:
        """Check for excessive punctuation."""
        excessive_patterns = [r"[!]{2,}", r"[?]{2,}", r"[.]{3,}", r"[!?]{2,}"]

        for pattern in excessive_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _check_suspicious_urls(self, text: str) -> bool:
        """Check for suspicious URLs or domains."""
        # Basic URL pattern
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)

        if not urls:
            return False

        # Suspicious TLDs or patterns
        suspicious_patterns = [
            r"\.tk\b",
            r"\.ml\b",
            r"\.ga\b",
            r"\.cf\b",
            r"bit\.ly",
            r"tinyurl",
            r"shortlink",
        ]

        for url in urls:
            for pattern in suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return True

        return False

    def _check_repeated_words(self, text: str) -> bool:
        """Check for repeated words (common in spam)."""
        words = text.lower().split()

        if len(words) < 5:
            return False

        word_counts: dict[str, int] = {}
        for word in words:
            if len(word) > 2:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1

        # Check if any word appears too frequently
        max_count = max(word_counts.values()) if word_counts else 0
        total_words = len(words)

        if total_words > 0:
            max_ratio = max_count / total_words
            return max_ratio > 0.2  # More than 20% repetition

        return False

    async def cleanup(self) -> None:
        """Clean up resources."""
        await super().cleanup()
        self.compiled_promo_patterns = []
