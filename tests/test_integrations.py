"""
Test integration functionality for AI Safety Guardrails.

This module tests the integration between the safety guardrails and various LLM providers.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from ai_safety_guardrails.integrations.openai_client import OpenAIClient
from ai_safety_guardrails.integrations.ollama_client import OllamaClient
from ai_safety_guardrails.utils.exceptions import AILLMSafetyException


class TestOpenAIClient:
    """Test OpenAI client functionality."""

    def test_openai_client_initialization(self):
        """Test OpenAI client initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            assert client.api_key == "test-key"
            assert client.timeout == 120

    def test_openai_client_with_custom_params(self):
        """Test OpenAI client with custom parameters."""
        client = OpenAIClient(
            api_key="custom-key",
            timeout=60,
            max_retries=5
        )
        assert client.api_key == "custom-key"
        assert client.timeout == 60
        assert client.max_retries == 5

    def test_openai_client_without_api_key(self):
        """Test OpenAI client without API key."""
        with patch.dict(os.environ, {}, clear=True):
            import pytest
            with pytest.raises(ValueError):
                OpenAIClient()


class TestOllamaClient:
    """Test Ollama client functionality."""

    def test_ollama_client_initialization(self):
        """Test Ollama client initialization."""
        client = OllamaClient()
        assert client.host == "http://localhost:11434"
        assert client.timeout == 120

    def test_ollama_client_with_custom_params(self):
        """Test Ollama client with custom parameters."""
        client = OllamaClient(
            host="http://custom-host:8000",
            timeout=60
        )
        assert client.host == "http://custom-host:8000"
        assert client.timeout == 60


@pytest.mark.integration
class TestRealIntegrationTests:
    """Real integration tests with actual services (requires API keys)."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY_TEST"),
        reason="Requires OPENAI_API_KEY_TEST environment variable"
    )
    @pytest.mark.asyncio
    async def test_real_openai_integration(self):
        """Test real OpenAI integration (requires API key)."""
        client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY_TEST"))
        
        # Test connection
        result = await client.test_connection()
        assert result is True
        
        # Test chat completion
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Say hello"}],
            model="gpt-3.5-turbo",
            max_tokens=10
        )
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skipif(
        not os.getenv("RUN_OLLAMA_TESTS"),
        reason="Requires local Ollama server and RUN_OLLAMA_TESTS=true"
    )
    @pytest.mark.asyncio
    async def test_real_ollama_integration(self):
        """Test real Ollama integration (requires local server)."""
        host = os.getenv("OLLAMA_HOST_TEST", "http://localhost:11434")
        client = OllamaClient(host=host)
        
        try:
            # Test connection
            result = await client.test_connection()
            assert result is True
            
            # Test generation (if models are available)
            response = await client.generate(
                model="llama2",  # Assume llama2 is available
                prompt="Say hello",
                options={"num_predict": 10}
            )
            assert isinstance(response, dict)
            assert "response" in response
            
        except AILLMSafetyException:
            pytest.skip("Ollama server not available or no models installed")