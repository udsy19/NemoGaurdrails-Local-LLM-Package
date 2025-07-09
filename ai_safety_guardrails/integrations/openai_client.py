"""
OpenAI integration for AI Safety Guardrails.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import IntegrationException
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """
    OpenAI client with built-in safety integration.

    Provides convenient methods for calling OpenAI APIs with
    automatic safety protection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for API
            organization: Organization ID
            **kwargs: Additional OpenAI client parameters
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise IntegrationException("OpenAI API key is required")

        try:
            import openai

            self.openai = openai

            # Initialize client
            client_kwargs = {"api_key": self.api_key, **kwargs}

            if base_url:
                client_kwargs["base_url"] = base_url
            if organization:
                client_kwargs["organization"] = organization

            self.client = openai.OpenAI(**client_kwargs)
            logger.info("OpenAI client initialized successfully")

        except ImportError:
            raise IntegrationException(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise IntegrationException(f"Failed to initialize OpenAI client: {e}")

    async def chat_completion(
        self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", **kwargs
    ) -> str:
        """
        Create a chat completion.

        Args:
            messages: List of message dictionaries
            model: Model to use
            **kwargs: Additional OpenAI parameters

        Returns:
            The response content as a string
        """
        try:
            # Run in thread pool since OpenAI client is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                ),
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise IntegrationException(f"OpenAI API call failed: {e}")

    async def completion(
        self, prompt: str, model: str = "gpt-3.5-turbo-instruct", **kwargs
    ) -> str:
        """
        Create a text completion.

        Args:
            prompt: Text prompt
            model: Model to use
            **kwargs: Additional OpenAI parameters

        Returns:
            The completion text
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.completions.create(
                    model=model, prompt=prompt, **kwargs
                ),
            )

            return response.choices[0].text

        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise IntegrationException(f"OpenAI API call failed: {e}")

    def create_safe_chat_function(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: Optional[str] = None,
        **default_kwargs,
    ):
        """
        Create a reusable chat function for use with SafetyGuard.

        Args:
            model: Default model to use
            system_message: Optional system message
            **default_kwargs: Default parameters for chat completions

        Returns:
            Async function that can be used with SafetyGuard.protect()
        """

        async def safe_chat(user_message: str) -> str:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": user_message})

            return await self.chat_completion(
                messages=messages, model=model, **default_kwargs
            )

        return safe_chat

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def test_connection(self) -> bool:
        """Test the connection to OpenAI API."""
        try:
            # Try to list models as a simple test
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
