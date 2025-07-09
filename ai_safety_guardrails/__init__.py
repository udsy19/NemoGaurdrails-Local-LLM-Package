"""
AI Safety Guardrails - A comprehensive AI safety package for protecting LLM applications.

This package provides:
- SafetyGuard: Library API for explicit safety control
- @safe_ai: Decorator API for transparent safety protection
- Template system: Quick-start applications for common use cases
- CLI tools: Command-line interface for package management

Usage:
    # Library approach
    from ai_safety_guardrails import SafetyGuard, DetectorConfig

    guard = SafetyGuard(detectors=[
        DetectorConfig("toxicity", threshold=0.7),
        DetectorConfig("pii", sensitivity="high")
    ])

    result = guard.protect(user_input, my_llm_function)

    # Decorator approach
    from ai_safety_guardrails import safe_ai

    @safe_ai(detectors=["toxicity", "pii"], config="./safety_config.yml")
    def my_llm_function(prompt):
        return llm_response
"""

__version__ = "1.0.1"
__author__ = "Udaya Vijay Anand"
__email__ = "udayatejas2004@gmail.com"
__description__ = "Comprehensive AI safety package for LLM applications"

from .core.detector_config import DetectorConfig

# Core exports - these are the main public API
from .core.safety_guard import SafetyGuard
from .decorators import safe_ai
from .detectors.pii import PIIDetector
from .detectors.prompt_injection import PromptInjectionDetector
from .detectors.topics import TopicsDetector

# Detector exports
from .detectors.toxicity import ToxicityDetector
from .integrations.ollama_client import OllamaClient

# Integration exports
from .integrations.openai_client import OpenAIClient

from .templates.creator import create_app

# Exception exports
from .utils.exceptions import (
    AILLMSafetyException,
    DetectorException,
    ModelLoadException,
    SafetyGuardException,
)

# Main public API - what users import
__all__ = [
    # Core functionality
    "SafetyGuard",
    "DetectorConfig",
    "safe_ai",
    "create_app",
    # Detectors
    "ToxicityDetector",
    "PIIDetector",
    "TopicsDetector",
    "PromptInjectionDetector",
    # Integrations
    "OpenAIClient",
    "OllamaClient",
    # Exceptions
    "AILLMSafetyException",
    "SafetyGuardException",
    "DetectorException",
    "ModelLoadException",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]

