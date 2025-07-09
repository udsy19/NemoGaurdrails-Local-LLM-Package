"""
Ollama integration for AI Safety Guardrails.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from ..utils.exceptions import IntegrationException
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """
    Ollama client with built-in safety integration.

    Provides convenient methods for calling Ollama APIs with
    automatic safety protection.
    """

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL (defaults to http://localhost:11434)
            **kwargs: Additional Ollama client parameters
        """
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        try:
            import ollama

            self.ollama = ollama

            # Initialize client
            self.client = ollama.Client(host=self.base_url, **kwargs)
            logger.info(f"Ollama client initialized with host: {self.base_url}")

        except ImportError:
            raise IntegrationException(
                "Ollama package not installed. Install with: pip install ollama"
            )
        except Exception as e:
            raise IntegrationException(f"Failed to initialize Ollama client: {e}")

    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Create a chat completion.

        Args:
            model: Model to use
            messages: List of message dictionaries
            **kwargs: Additional Ollama parameters

        Returns:
            The response content as a string
        """
        try:
            # Run in thread pool since Ollama client might be sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.chat(model=model, messages=messages, **kwargs)
            )

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise IntegrationException(f"Ollama API call failed: {e}")

    async def generate(self, model: str, prompt: str, **kwargs) -> str:
        """
        Generate text completion.

        Args:
            model: Model to use
            prompt: Text prompt
            **kwargs: Additional Ollama parameters

        Returns:
            The generated text
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.generate(model=model, prompt=prompt, **kwargs)
            )

            return response["response"]

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise IntegrationException(f"Ollama API call failed: {e}")

    def create_safe_chat_function(
        self, model: str, system_message: Optional[str] = None, **default_kwargs
    ):
        """
        Create a reusable chat function for use with SafetyGuard.

        Args:
            model: Model to use
            system_message: Optional system message
            **default_kwargs: Default parameters for chat

        Returns:
            Async function that can be used with SafetyGuard.protect()
        """

        async def safe_chat(user_message: str) -> str:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": user_message})

            return await self.chat(model=model, messages=messages, **default_kwargs)

        return safe_chat

    def create_safe_generate_function(self, model: str, **default_kwargs):
        """
        Create a reusable generate function for use with SafetyGuard.

        Args:
            model: Model to use
            **default_kwargs: Default parameters for generation

        Returns:
            Async function that can be used with SafetyGuard.protect()
        """

        async def safe_generate(prompt: str) -> str:
            return await self.generate(model=model, prompt=prompt, **default_kwargs)

        return safe_generate

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models = self.client.list()
            return [model["name"] for model in models["models"]]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def test_connection(self) -> bool:
        """Test the connection to Ollama server."""
        try:
            # Try to list models as a simple test
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False

    def pull_model(self, model: str) -> bool:
        """
        Pull a model to the Ollama server.

        Args:
            model: Model name to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.pull(model)
            logger.info(f"Successfully pulled model: {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
