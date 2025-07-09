"""
Prompt injection detector.

Detects attempts to manipulate AI behavior through prompt injection attacks.
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.exceptions import DetectorException, ModelLoadException
from ..utils.logger import get_logger
from .base_detector import BaseDetector, DetectionResult

logger = get_logger(__name__)


class PromptInjectionDetector(BaseDetector):
    """
    Detects prompt injection attacks using pattern matching.

    Identifies attempts to:
    - Override system instructions
    - Change AI behavior through role-playing
    - Inject malicious prompts
    - Bypass safety measures
    """

    def __init__(self, **kwargs):
        super().__init__("prompt_injection", **kwargs)

        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"ignore\s+the\s+above",
            r"disregard\s+the\s+above",
            r"forget\s+everything\s+above",
            r"you\s+are\s+now\s+playing\s+the\s+role\s+of",
            r"pretend\s+you\s+are",
            r"act\s+as\s+if\s+you\s+are",
            r"simulate\s+being",
            r"roleplay\s+as",
            r"behave\s+like",
            r"respond\s+as\s+if\s+you\s+were",
            r"from\s+now\s+on",
            r"new\s+instruction",
            r"override\s+your\s+programming",
            r"change\s+your\s+behavior",
            r"alter\s+your\s+responses",
            r"system\s+prompt",
            r"system\s+message",
            r"developer\s+mode",
            r"jailbreak",
            r"prompt\s+injection",
            r"bypass\s+safety",
            r"ignore\s+safety",
            r"disable\s+safety",
        ]

        # Compiled patterns for efficiency
        self.compiled_patterns = []

    async def load_model(self) -> None:
        """Load and compile regex patterns."""
        try:
            logger.info("Loading prompt injection detector")

            # Compile regex patterns
            self.compiled_patterns = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.injection_patterns
            ]

            self.is_loaded = True
            logger.info("Prompt injection detector loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load prompt injection detector: {e}")
            raise ModelLoadException(f"Failed to load prompt injection detector: {e}")

    async def detect(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect prompt injection attempts.

        Args:
            text: Text to analyze
            context: Optional context (threshold can be specified here)

        Returns:
            DetectionResult with injection analysis
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

            matched_patterns = []
            total_matches = 0

            # Check each pattern
            for i, pattern in enumerate(self.compiled_patterns):
                matches = pattern.findall(text)
                if matches:
                    matched_patterns.append(
                        {
                            "pattern": self.injection_patterns[i],
                            "matches": matches,
                            "count": len(matches),
                        }
                    )
                    total_matches += len(matches)

            # Additional heuristics
            suspicious_phrases = self._check_suspicious_phrases(text)
            role_playing = self._check_role_playing(text)
            instruction_override = self._check_instruction_override(text)

            # Calculate confidence based on various factors
            confidence = min(total_matches * 0.3, 1.0)

            if suspicious_phrases:
                confidence += 0.2
            if role_playing:
                confidence += 0.3
            if instruction_override:
                confidence += 0.4

            confidence = min(confidence, 1.0)
            is_injection = confidence > threshold

            return DetectionResult(
                blocked=is_injection,
                confidence=confidence,
                reason=(
                    f"Prompt injection detected (confidence: {confidence:.2f})"
                    if is_injection
                    else None
                ),
                details={
                    "matched_patterns": matched_patterns,
                    "total_matches": total_matches,
                    "suspicious_phrases": suspicious_phrases,
                    "role_playing": role_playing,
                    "instruction_override": instruction_override,
                    "threshold": threshold,
                },
            )

        except Exception as e:
            logger.error(f"Prompt injection detection failed: {e}")
            raise DetectorException(f"Prompt injection detection failed: {e}")

    def _check_suspicious_phrases(self, text: str) -> bool:
        """Check for suspicious phrases that might indicate injection."""
        suspicious = [
            "system",
            "prompt",
            "assistant",
            "ai",
            "model",
            "gpt",
            "chatbot",
            "instructions",
            "rules",
            "guidelines",
            "behavior",
            "personality",
        ]

        text_lower = text.lower()
        word_count = len(text.split())

        # If text is short and contains many AI-related terms, it's suspicious
        if word_count < 50:
            ai_term_count = sum(1 for term in suspicious if term in text_lower)
            return ai_term_count >= 3

        return False

    def _check_role_playing(self, text: str) -> bool:
        """Check for role-playing attempts."""
        role_patterns = [
            r"you\s+are\s+a\s+",
            r"pretend\s+to\s+be\s+",
            r"imagine\s+you\s+are\s+",
            r"act\s+like\s+a\s+",
            r"roleplay\s+as\s+a\s+",
        ]

        for pattern in role_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _check_instruction_override(self, text: str) -> bool:
        """Check for attempts to override instructions."""
        override_patterns = [
            r"ignore\s+",
            r"forget\s+",
            r"disregard\s+",
            r"override\s+",
            r"replace\s+",
            r"change\s+",
            r"modify\s+",
        ]

        instruction_words = ["instruction", "rule", "guideline", "system", "prompt"]

        text_lower = text.lower()

        for override_pattern in override_patterns:
            for instruction_word in instruction_words:
                pattern = override_pattern + r".*?" + instruction_word
                if re.search(pattern, text_lower):
                    return True

        return False

    async def cleanup(self) -> None:
        """Clean up resources."""
        await super().cleanup()
        self.compiled_patterns = []
