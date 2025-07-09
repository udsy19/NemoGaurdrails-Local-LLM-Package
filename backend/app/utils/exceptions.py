from typing import Optional

class AILLMSafetyException(Exception):
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ModelLoadingException(AILLMSafetyException):
    pass

class DetectionException(AILLMSafetyException):
    pass

class GuardrailsException(AILLMSafetyException):
    pass

class OllamaException(AILLMSafetyException):
    pass

class ConfigurationException(AILLMSafetyException):
    pass