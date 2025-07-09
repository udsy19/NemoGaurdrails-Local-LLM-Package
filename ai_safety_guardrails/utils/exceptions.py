"""
Custom exceptions for AI Safety Guardrails package.
"""


class AILLMSafetyException(Exception):
    """Base exception for all AI safety related errors."""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class SafetyGuardException(AILLMSafetyException):
    """Exception raised when SafetyGuard encounters an error."""

    pass


class DetectorException(AILLMSafetyException):
    """Exception raised when a detector fails to process input."""

    pass


class ModelLoadException(AILLMSafetyException):
    """Exception raised when a model fails to load."""

    pass


class ConfigurationException(AILLMSafetyException):
    """Exception raised when configuration is invalid."""

    pass


class TemplateException(AILLMSafetyException):
    """Exception raised during template creation or processing."""

    pass


class IntegrationException(AILLMSafetyException):
    """Exception raised during LLM integration errors."""

    pass
