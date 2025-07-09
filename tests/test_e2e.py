"""
End-to-end tests for AI Safety Guardrails.

These tests verify complete workflows from user input to final output.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock

from ai_safety_guardrails import SafetyGuard, safe_ai
from ai_safety_guardrails.integrations.openai_client import OpenAIClient
from ai_safety_guardrails.integrations.ollama_client import OllamaClient



@pytest.mark.e2e
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_complete_safety_workflow(self, mock_detector, mock_llm_function):
        """Test complete safety workflow from input to output."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            # Initialize SafetyGuard
            guard = SafetyGuard(detectors=["toxicity", "pii", "prompt_injection"])
            
            # Test safe content workflow
            result = await guard.protect(
                input_text="Hello, how can you help me learn about AI?",
                llm_function=mock_llm_function,
                check_output=True
            )
            
            assert not result.blocked
            assert result.response is not None
            assert "AI Response:" in result.response
            assert result.safety_results is not None
            assert len(result.safety_results) == 3  # Three detectors
            
            # Verify all detectors ran
            for detector_name in ["toxicity", "pii", "prompt_injection"]:
                assert detector_name in result.safety_results
                assert not result.safety_results[detector_name].blocked

    @pytest.mark.asyncio
    async def test_blocked_input_workflow(self, mock_detector, mock_llm_function):
        """Test workflow when input is blocked."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that blocks on specific content
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def conditional_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    should_block = "block_this" in text.lower()
                    return DetectionResult(
                        blocked=should_block,
                        confidence=0.9 if should_block else 0.1,
                        reason="Contains blocked phrase" if should_block else None,
                        detector_name="mock_detector"
                    )
                
                detector.safe_detect = conditional_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Test blocked input
            result = await guard.protect(
                input_text="Please block_this content",
                llm_function=mock_llm_function
            )
            
            assert result.blocked
            assert result.response is None
            assert result.block_reason is not None
            assert "input" in result.block_reason.lower()

    @pytest.mark.asyncio
    async def test_blocked_output_workflow(self, mock_detector, mock_llm_function):
        """Test workflow when output is blocked."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that blocks LLM output but not input
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def conditional_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    should_block = "AI Response:" in text and "block_output" in text
                    return DetectionResult(
                        blocked=should_block,
                        confidence=0.9 if should_block else 0.1,
                        reason="Output contains blocked content" if should_block else None,
                        detector_name="mock_detector"
                    )
                
                detector.safe_detect = conditional_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            # Mock LLM that generates blocked output
            async def mock_llm_with_blocked_output(prompt):
                return "AI Response: This output contains block_output content"
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            result = await guard.protect(
                input_text="Generate some content",
                llm_function=mock_llm_with_blocked_output,
                check_output=True
            )
            
            assert result.blocked
            assert result.response is None
            assert result.block_reason is not None
            assert "output" in result.block_reason.lower()

    @pytest.mark.asyncio
    async def test_decorator_workflow(self, mock_detector):
        """Test end-to-end workflow using decorator API."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            # Create protected function using decorator
            @safe_ai(detectors=["toxicity", "pii"])
            async def protected_llm_function(prompt: str) -> str:
                """Protected LLM function."""
                return f"Protected response to: {prompt}"
            
            # Test safe usage
            result = await protected_llm_function("Hello, how are you?")
            assert result == "Protected response to: Hello, how are you?"

    @pytest.mark.asyncio
    async def test_decorator_blocking_workflow(self, mock_detector):
        """Test decorator API with blocking."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that blocks specific content
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def conditional_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    should_block = "unsafe_content" in text.lower()
                    return DetectionResult(
                        blocked=should_block,
                        confidence=0.9 if should_block else 0.1,
                        reason="Unsafe content detected" if should_block else None,
                        detector_name="mock_detector"
                    )
                
                detector.safe_detect = conditional_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            @safe_ai(detectors=["toxicity"])
            async def protected_function(prompt: str) -> str:
                return f"Response to: {prompt}"
            
            # Test blocked content
            from ai_safety_guardrails.utils.exceptions import SafetyGuardException
            
            with pytest.raises(SafetyGuardException):
                await protected_function("This contains unsafe_content")

    @pytest.mark.asyncio
    async def test_configuration_workflow(self, temp_config_file):
        """Test end-to-end workflow with configuration file."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            from ai_safety_guardrails.detectors.base_detector import BaseDetector, DetectionResult
            
            class ConfigurableDetector(BaseDetector):
                def __init__(self, **kwargs):
                    super().__init__("configurable", **kwargs)
                
                async def load_model(self):
                    self._loaded = True
                
                async def detect(self, text: str) -> DetectionResult:
                    # Use threshold from config
                    confidence = 0.8 if "high_confidence" in text else 0.2
                    blocked = confidence >= self.threshold
                    
                    return DetectionResult(
                        blocked=blocked,
                        confidence=confidence,
                        reason="High confidence detection" if blocked else None,
                        detector_name=self.name
                    )
            
            mock_get_detector.return_value = ConfigurableDetector
            
            # Initialize with config file
            guard = SafetyGuard(config_file=temp_config_file)
            
            # Test with different confidence levels
            low_result = await guard.analyze_text("This is normal text")
            high_result = await guard.analyze_text("This has high_confidence content")
            
            # Results should reflect configuration
            assert isinstance(low_result, dict)
            assert isinstance(high_result, dict)

    @pytest.mark.asyncio
    async def test_batch_workflow(self, mock_detector, sample_texts):
        """Test end-to-end batch processing workflow."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity", "pii"])
            
            # Test batch processing
            all_texts = sample_texts["safe"] + sample_texts["pii"][:2]
            results = await guard.batch_analyze(all_texts)
            
            assert len(results) == len(all_texts)
            
            for result_dict in results:
                assert isinstance(result_dict, dict)
                assert "toxicity" in result_dict
                assert "pii" in result_dict


@pytest.mark.e2e
class TestIntegrationWorkflows:
    """Test integration workflows with actual LLM clients."""

    @pytest.mark.asyncio
    async def test_openai_integration_workflow(self, mock_detector):
        """Test complete workflow with OpenAI integration."""
        with patch('ai_safety_guardrails.integrations.openai_client.openai.OpenAI') as mock_openai:
            with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
                # Setup mocks
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "OpenAI says hello!"
                
                mock_completion = AsyncMock(return_value=mock_response)
                mock_client.chat.completions.create = mock_completion
                mock_openai.return_value = mock_client
                
                mock_detector_instance = mock_detector(should_block=False)
                mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
                
                # Complete workflow
                openai_client = OpenAIClient(api_key="test-key")
                safety_guard = SafetyGuard(detectors=["toxicity", "pii"])
                
                # Create safe function
                safe_chat_fn = openai_client.create_safe_chat_function(model="gpt-3.5-turbo")
                
                # Protected execution
                result = await safety_guard.protect(
                    input_text="Tell me about artificial intelligence",
                    llm_function=safe_chat_fn,
                    check_output=True
                )
                
                assert not result.blocked
                assert result.response == "OpenAI says hello!"
                assert result.safety_results is not None

    @pytest.mark.asyncio
    async def test_ollama_integration_workflow(self, mock_detector):
        """Test complete workflow with Ollama integration."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
                # Setup mocks
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "response": "Ollama generated response"
                })
                mock_post.return_value.__aenter__.return_value = mock_response
                
                mock_detector_instance = mock_detector(should_block=False)
                mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
                
                # Complete workflow
                ollama_client = OllamaClient(host="http://localhost:11434")
                safety_guard = SafetyGuard(detectors=["toxicity"])
                
                # Create safe function
                safe_generate_fn = ollama_client.create_safe_generate_function(model="llama2")
                
                # Protected execution
                result = await safety_guard.protect(
                    input_text="Generate a story about AI",
                    llm_function=safe_generate_fn
                )
                
                assert not result.blocked
                assert result.response == "Ollama generated response"



@pytest.mark.e2e
class TestRealWorldScenarios:
    """Test realistic real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_customer_service_chatbot_scenario(self, mock_detector):
        """Test a customer service chatbot scenario."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that blocks PII and toxic content
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def conditional_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    
                    # Check for PII patterns
                    pii_patterns = ["@", "555-", "ssn:", "credit card"]
                    has_pii = any(pattern in text.lower() for pattern in pii_patterns)
                    
                    # Check for toxic content
                    toxic_words = ["hate", "angry", "terrible"]
                    has_toxicity = any(word in text.lower() for word in toxic_words)
                    
                    blocked = has_pii or has_toxicity
                    confidence = 0.9 if blocked else 0.1
                    reason = None
                    
                    if has_pii:
                        reason = "PII detected"
                    elif has_toxicity:
                        reason = "Toxic content detected"
                    
                    return DetectionResult(
                        blocked=blocked,
                        confidence=confidence,
                        reason=reason,
                        detector_name="conditional_detector"
                    )
                
                detector.safe_detect = conditional_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            # Simulate customer service LLM
            async def customer_service_llm(prompt: str) -> str:
                if "order status" in prompt.lower():
                    return "I'd be happy to help you check your order status. Could you provide your order number?"
                elif "refund" in prompt.lower():
                    return "I understand you'd like to request a refund. Let me help you with that process."
                else:
                    return "Thank you for contacting customer service. How can I assist you today?"
            
            guard = SafetyGuard(detectors=["pii", "toxicity"])
            
            # Test legitimate customer queries
            legitimate_queries = [
                "What's my order status?",
                "I need help with a refund",
                "How do I return an item?",
                "When will my package arrive?"
            ]
            
            for query in legitimate_queries:
                result = await guard.protect(
                    input_text=query,
                    llm_function=customer_service_llm,
                    check_output=True
                )
                
                assert not result.blocked, f"Legitimate query blocked: {query}"
                assert result.response is not None
            
            # Test problematic queries
            problematic_queries = [
                "My email is user@example.com and I'm angry about this terrible service",
                "Call me at 555-1234, this is hate-worthy",
                "Here's my credit card number: 4532-1234-5678-9012"
            ]
            
            for query in problematic_queries:
                result = await guard.protect(
                    input_text=query,
                    llm_function=customer_service_llm
                )
                
                assert result.blocked, f"Problematic query not blocked: {query}"

    @pytest.mark.asyncio
    async def test_content_moderation_scenario(self, mock_detector, sample_texts):
        """Test a content moderation scenario."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create sophisticated detector
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def content_moderation_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    
                    # Multiple criteria for content moderation
                    blocked_reasons = []
                    confidence = 0.0
                    
                    # Check for spam indicators
                    spam_indicators = ["buy now", "limited time", "click here", "!!!"]
                    spam_count = sum(1 for indicator in spam_indicators if indicator in text.lower())
                    if spam_count >= 2:
                        blocked_reasons.append("Spam content")
                        confidence = max(confidence, 0.8)
                    
                    # Check for inappropriate content
                    inappropriate_words = ["inappropriate", "offensive", "harmful"]
                    if any(word in text.lower() for word in inappropriate_words):
                        blocked_reasons.append("Inappropriate content")
                        confidence = max(confidence, 0.9)
                    
                    # Check text length (flag very long texts)
                    if len(text) > 1000:
                        blocked_reasons.append("Content too long")
                        confidence = max(confidence, 0.6)
                    
                    blocked = len(blocked_reasons) > 0
                    reason = "; ".join(blocked_reasons) if blocked else None
                    
                    return DetectionResult(
                        blocked=blocked,
                        confidence=confidence,
                        reason=reason,
                        detector_name="content_moderator"
                    )
                
                detector.safe_detect = content_moderation_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            guard = SafetyGuard(detectors=["content_moderator"])
            
            # Test various content types
            test_contents = [
                ("Normal post", "Just sharing some thoughts about my day", False),
                ("Spam content", "BUY NOW!!! Limited time offer, click here for amazing deals!!!", True),
                ("Inappropriate", "This content is inappropriate and harmful", True),
                ("Long content", "A" * 1100, True),  # Very long text
                ("Borderline", "Great deals available, check it out!", False),  # Borderline case
            ]
            
            for content_type, text, should_block in test_contents:
                result = await guard.analyze_text(text)
                moderation_result = result["content_moderator"]
                
                if should_block:
                    assert moderation_result.blocked, f"{content_type} should be blocked but wasn't"
                else:
                    assert not moderation_result.blocked, f"{content_type} shouldn't be blocked but was"

    @pytest.mark.asyncio
    async def test_educational_ai_tutor_scenario(self, mock_detector):
        """Test an educational AI tutor scenario."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Educational content detector
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def educational_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    
                    # Block inappropriate educational content
                    inappropriate_educational = [
                        "homework answers",
                        "cheat on test",
                        "plagiarize",
                        "copy assignment"
                    ]
                    
                    blocked = any(phrase in text.lower() for phrase in inappropriate_educational)
                    confidence = 0.9 if blocked else 0.1
                    reason = "Academic dishonesty detected" if blocked else None
                    
                    return DetectionResult(
                        blocked=blocked,
                        confidence=confidence,
                        reason=reason,
                        detector_name="educational_filter"
                    )
                
                detector.safe_detect = educational_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            # Educational AI tutor
            async def ai_tutor(prompt: str) -> str:
                if "explain" in prompt.lower():
                    return "I'd be happy to explain that concept. Let me break it down for you..."
                elif "help" in prompt.lower():
                    return "I'm here to help guide your learning. What specific part would you like to explore?"
                elif "example" in prompt.lower():
                    return "Here's an example to illustrate the concept..."
                else:
                    return "Let's work through this together. What would you like to learn about?"
            
            guard = SafetyGuard(detectors=["educational_filter"])
            
            # Test legitimate educational queries
            good_queries = [
                "Can you explain photosynthesis?",
                "Help me understand calculus derivatives",
                "Show me an example of a chemical reaction",
                "How does machine learning work?"
            ]
            
            for query in good_queries:
                result = await guard.protect(
                    input_text=query,
                    llm_function=ai_tutor,
                    check_output=True
                )
                
                assert not result.blocked, f"Educational query blocked: {query}"
                assert result.response is not None
            
            # Test inappropriate educational queries
            bad_queries = [
                "Give me homework answers for math test",
                "Help me cheat on my exam tomorrow",
                "How can I plagiarize this essay?",
                "Let me copy this assignment from you"
            ]
            
            for query in bad_queries:
                result = await guard.protect(
                    input_text=query,
                    llm_function=ai_tutor
                )
                
                assert result.blocked, f"Inappropriate query not blocked: {query}"

    @pytest.mark.asyncio
    async def test_enterprise_document_analysis_scenario(self, mock_detector):
        """Test enterprise document analysis scenario."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Enterprise security detector
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def enterprise_detect(text):
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    
                    # Check for sensitive enterprise information
                    sensitive_patterns = [
                        "confidential",
                        "internal only",
                        "proprietary",
                        "trade secret",
                        "api key:",
                        "password:",
                        "token:"
                    ]
                    
                    blocked = any(pattern in text.lower() for pattern in sensitive_patterns)
                    confidence = 0.95 if blocked else 0.05
                    reason = "Sensitive enterprise information detected" if blocked else None
                    
                    return DetectionResult(
                        blocked=blocked,
                        confidence=confidence,
                        reason=reason,
                        detector_name="enterprise_security"
                    )
                
                detector.safe_detect = enterprise_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            # Document analysis AI
            async def document_analyzer(document_text: str) -> str:
                word_count = len(document_text.split())
                return f"Document analysis complete. Word count: {word_count}. Summary: General business document discussing operational procedures."
            
            guard = SafetyGuard(detectors=["enterprise_security"])
            
            # Test safe documents
            safe_documents = [
                "This is a general business document about sales procedures.",
                "Meeting notes from the quarterly review session.",
                "Public marketing materials for product launch."
            ]
            
            for doc in safe_documents:
                result = await guard.protect(
                    input_text=doc,
                    llm_function=document_analyzer,
                    check_output=True
                )
                
                assert not result.blocked, f"Safe document blocked: {doc[:50]}..."
            
            # Test sensitive documents
            sensitive_documents = [
                "This confidential document contains proprietary algorithms.",
                "Internal only: API key: abc123xyz and password: secret123",
                "Trade secret information about our competitive advantage."
            ]
            
            for doc in sensitive_documents:
                result = await guard.protect(
                    input_text=doc,
                    llm_function=document_analyzer
                )
                
                assert result.blocked, f"Sensitive document not blocked: {doc[:50]}..."


@pytest.mark.e2e
class TestErrorRecoveryWorkflows:
    """Test error recovery and resilience workflows."""

    @pytest.mark.asyncio
    async def test_detector_failure_recovery(self, mock_detector, mock_llm_function):
        """Test recovery from detector failures."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that sometimes fails
            def failing_detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                detector.call_count = 0
                
                async def failing_detect(text):
                    detector.call_count += 1
                    if detector.call_count % 3 == 0:  # Fail every 3rd call
                        raise Exception("Detector failure")
                    
                    from ai_safety_guardrails.detectors.base_detector import DetectionResult
                    return DetectionResult(
                        blocked=False,
                        confidence=0.1,
                        reason=None,
                        detector_name="failing_detector"
                    )
                
                detector.safe_detect = failing_detect
                return detector
            
            mock_get_detector.return_value = failing_detector_factory
            
            guard = SafetyGuard(detectors=["failing_detector"], fail_open=True)
            
            # Test multiple calls with some failures
            for i in range(5):
                result = await guard.protect(
                    input_text=f"Test message {i}",
                    llm_function=mock_llm_function
                )
                
                # Should not block due to fail_open=True
                assert not result.blocked
                assert result.response is not None

    @pytest.mark.asyncio
    async def test_llm_failure_handling(self, mock_detector):
        """Test handling of LLM failures."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            # LLM that fails
            async def failing_llm(prompt: str) -> str:
                raise Exception("LLM service unavailable")
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Should propagate LLM failures
            with pytest.raises(Exception) as exc_info:
                await guard.protect(
                    input_text="Test input",
                    llm_function=failing_llm
                )
            
            assert "LLM service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_workflow(self, mock_detector, mock_llm_function):
        """Test circuit breaker functionality in workflows."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that fails consistently
            def failing_detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def always_failing_detect(text):
                    raise Exception("Persistent detector failure")
                
                detector.safe_detect = always_failing_detect
                return detector
            
            mock_get_detector.return_value = failing_detector_factory
            
            guard = SafetyGuard(
                detectors=["failing_detector"],
                circuit_breaker=True,
                fail_open=True
            )
            
            # Multiple calls should trigger circuit breaker
            for i in range(10):
                result = await guard.protect(
                    input_text=f"Test message {i}",
                    llm_function=mock_llm_function
                )
                
                # Should still work with fail_open
                assert not result.blocked
                assert result.response is not None