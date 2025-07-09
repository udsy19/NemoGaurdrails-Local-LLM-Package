"""
Unit tests for AI Safety Guardrails SafetyGuard core functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from ai_safety_guardrails import SafetyGuard, DetectorConfig
from ai_safety_guardrails.detectors.base_detector import DetectionResult
from ai_safety_guardrails.utils.exceptions import SafetyGuardException


class TestSafetyGuardInitialization:
    """Test SafetyGuard initialization and configuration."""

    def test_safety_guard_basic_init(self):
        """Test basic SafetyGuard initialization."""
        guard = SafetyGuard(detectors=["toxicity", "pii"])
        assert guard is not None
        assert len(guard.detectors) == 2

    def test_safety_guard_detector_config_init(self):
        """Test SafetyGuard initialization with DetectorConfig objects."""
        configs = [
            DetectorConfig("toxicity", threshold=0.8),
            DetectorConfig("pii", sensitivity="high")
        ]
        guard = SafetyGuard(detectors=configs)
        assert guard is not None
        assert len(guard.detectors) == 2

    def test_safety_guard_mixed_config_init(self):
        """Test SafetyGuard initialization with mixed detector configs."""
        configs = [
            "toxicity",
            DetectorConfig("pii", sensitivity="high"),
            "prompt_injection"
        ]
        guard = SafetyGuard(detectors=configs)
        assert guard is not None
        assert len(guard.detectors) == 3

    def test_safety_guard_config_file_init(self, temp_config_file):
        """Test SafetyGuard initialization with config file."""
        guard = SafetyGuard(config_file=temp_config_file)
        assert guard is not None
        # Should load detectors from config file

    def test_safety_guard_invalid_detector(self):
        """Test SafetyGuard with invalid detector name."""
        with pytest.raises(ValueError):
            SafetyGuard(detectors=["invalid_detector"])

    def test_safety_guard_empty_detectors(self):
        """Test SafetyGuard with empty detectors list."""
        guard = SafetyGuard(detectors=[])
        assert guard is not None
        assert len(guard.detectors) == 0


class TestSafetyGuardAnalysis:
    """Test SafetyGuard text analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_text_safe_content(self, mock_detector, sample_texts):
        """Test analyze_text with safe content."""
        # Create mock detectors that don't block
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity", "pii"])
            
            for text in sample_texts["safe"]:
                results = await guard.analyze_text(text)
                assert isinstance(results, dict)
                assert len(results) == 2  # Two detectors
                
                for detector_name, result in results.items():
                    assert isinstance(result, DetectionResult)
                    assert not result.blocked

    @pytest.mark.asyncio
    async def test_analyze_text_blocked_content(self, mock_detector):
        """Test analyze_text with content that should be blocked."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=True)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            results = await guard.analyze_text("This should be blocked")
            assert isinstance(results, dict)
            assert len(results) == 1
            
            result = results["toxicity"]
            assert isinstance(result, DetectionResult)
            assert result.blocked

    @pytest.mark.asyncio
    async def test_analyze_text_specific_detectors(self, mock_detector):
        """Test analyze_text with specific detector selection."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity", "pii", "spam"])
            
            # Test with specific detectors
            results = await guard.analyze_text("test text", detectors=["toxicity", "pii"])
            assert isinstance(results, dict)
            assert len(results) == 2
            assert "toxicity" in results
            assert "pii" in results
            assert "spam" not in results

    @pytest.mark.asyncio
    async def test_analyze_text_empty_text(self, mock_detector):
        """Test analyze_text with empty text."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            results = await guard.analyze_text("")
            assert isinstance(results, dict)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_analyze_text_detector_failure(self, mock_detector):
        """Test analyze_text when detector fails."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create a detector that fails
            failing_detector = mock_detector(should_block=False)
            
            async def failing_safe_detect(text):
                raise Exception("Detector failure")
            
            failing_detector.safe_detect = failing_safe_detect
            mock_get_detector.return_value = lambda **kwargs: failing_detector
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            results = await guard.analyze_text("test text")
            assert isinstance(results, dict)
            
            # Should handle failure gracefully
            result = results["toxicity"]
            assert isinstance(result, DetectionResult)


class TestSafetyGuardProtection:
    """Test SafetyGuard protect functionality."""

    @pytest.mark.asyncio
    async def test_protect_safe_content(self, mock_detector, mock_llm_function):
        """Test protect with safe content."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            result = await guard.protect(
                input_text="Hello, how are you?",
                llm_function=mock_llm_function
            )
            
            assert not result.blocked
            assert result.response is not None
            assert "AI Response:" in result.response

    @pytest.mark.asyncio
    async def test_protect_blocked_input(self, mock_detector, mock_llm_function):
        """Test protect with blocked input."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=True)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            result = await guard.protect(
                input_text="This should be blocked",
                llm_function=mock_llm_function
            )
            
            assert result.blocked
            assert result.response is None
            assert result.block_reason is not None

    @pytest.mark.asyncio
    async def test_protect_with_output_checking(self, mock_detector, mock_llm_function):
        """Test protect with output checking enabled."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Input is safe, but output might be blocked
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                # Override safe_detect to block on output
                original_safe_detect = detector.safe_detect
                async def conditional_safe_detect(text):
                    if "AI Response:" in text:
                        # Block output
                        return DetectionResult(
                            blocked=True,
                            confidence=0.9,
                            reason="Output blocked",
                            detector_name="mock_detector"
                        )
                    return await original_safe_detect(text)
                
                detector.safe_detect = conditional_safe_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            result = await guard.protect(
                input_text="Hello",
                llm_function=mock_llm_function,
                check_output=True
            )
            
            assert result.blocked
            assert "output" in result.block_reason.lower()

    @pytest.mark.asyncio
    async def test_protect_llm_failure(self, mock_detector, mock_failing_llm):
        """Test protect when LLM fails."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            with pytest.raises(Exception):
                await guard.protect(
                    input_text="Hello",
                    llm_function=mock_failing_llm
                )

    @pytest.mark.asyncio
    async def test_protect_with_context(self, mock_detector, mock_llm_function):
        """Test protect with additional context."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            context = {"user_id": "test_user", "session": "test_session"}
            result = await guard.protect(
                input_text="Hello",
                llm_function=mock_llm_function,
                context=context
            )
            
            assert not result.blocked
            assert result.context == context


class TestSafetyGuardBatchProcessing:
    """Test SafetyGuard batch processing functionality."""

    @pytest.mark.asyncio
    async def test_batch_analyze_safe_content(self, mock_detector, sample_texts):
        """Test batch_analyze with safe content."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            results = await guard.batch_analyze(sample_texts["safe"])
            assert isinstance(results, list)
            assert len(results) == len(sample_texts["safe"])
            
            for result_dict in results:
                assert isinstance(result_dict, dict)
                assert "toxicity" in result_dict
                assert not result_dict["toxicity"].blocked

    @pytest.mark.asyncio
    async def test_batch_analyze_mixed_content(self, mock_detector):
        """Test batch_analyze with mixed content."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            # Create detector that blocks based on content
            def detector_factory(**kwargs):
                detector = mock_detector(should_block=False)
                
                async def conditional_detect(text):
                    should_block = "block" in text.lower()
                    return DetectionResult(
                        blocked=should_block,
                        confidence=0.9 if should_block else 0.1,
                        reason="Contains 'block'" if should_block else None,
                        detector_name="mock_detector"
                    )
                
                detector.safe_detect = conditional_detect
                return detector
            
            mock_get_detector.return_value = detector_factory
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            texts = ["safe text", "this should block", "another safe text"]
            results = await guard.batch_analyze(texts)
            
            assert len(results) == 3
            assert not results[0]["toxicity"].blocked  # safe
            assert results[1]["toxicity"].blocked      # contains "block"
            assert not results[2]["toxicity"].blocked  # safe

    @pytest.mark.asyncio
    async def test_batch_analyze_empty_list(self, mock_detector):
        """Test batch_analyze with empty list."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            results = await guard.batch_analyze([])
            assert isinstance(results, list)
            assert len(results) == 0


class TestSafetyGuardHealthAndMetrics:
    """Test SafetyGuard health monitoring and metrics."""

    @pytest.mark.asyncio
    async def test_health_check(self, mock_detector, mock_health_status):
        """Test health check functionality."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_detector_instance.get_health_status.return_value = mock_health_status.detectors["toxicity"]
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            health = await guard.health_check()
            assert health.status in ["healthy", "unhealthy"]
            assert hasattr(health, "uptime")
            assert hasattr(health, "memory_usage")
            assert hasattr(health, "detectors")

    def test_get_metrics(self, mock_detector):
        """Test metrics collection."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_detector_instance.get_metrics.return_value = {
                "total_calls": 10,
                "successful_calls": 9,
                "failed_calls": 1,
                "blocks": 2
            }
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            metrics = guard.get_metrics()
            assert isinstance(metrics, dict)
            assert "total_requests" in metrics
            assert "detector_metrics" in metrics

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_detector):
        """Test cleanup functionality."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_detector_instance.cleanup = AsyncMock()
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            await guard.cleanup()
            # Should call cleanup on all detectors
            mock_detector_instance.cleanup.assert_called()


class TestSafetyGuardConfiguration:
    """Test SafetyGuard configuration management."""

    def test_update_detector_config(self, mock_detector):
        """Test updating detector configuration."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Update configuration
            new_config = {"threshold": 0.9}
            guard.update_detector_config("toxicity", new_config)
            
            # Should update the detector's configuration
            detector = guard.detectors["toxicity"]
            assert detector.threshold == 0.9

    def test_enable_disable_detector(self, mock_detector):
        """Test enabling and disabling detectors."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            
            # Disable detector
            guard.disable_detector("toxicity")
            assert not guard.detectors["toxicity"].enabled
            
            # Re-enable detector
            guard.enable_detector("toxicity")
            assert guard.detectors["toxicity"].enabled

    def test_add_remove_detector(self, mock_detector):
        """Test adding and removing detectors."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"])
            assert len(guard.detectors) == 1
            
            # Add detector
            guard.add_detector("pii")
            assert len(guard.detectors) == 2
            assert "pii" in guard.detectors
            
            # Remove detector
            guard.remove_detector("toxicity")
            assert len(guard.detectors) == 1
            assert "toxicity" not in guard.detectors


class TestSafetyGuardConcurrency:
    """Test SafetyGuard concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, mock_detector, sample_texts):
        """Test concurrent text analysis."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"], max_concurrent_detections=5)
            
            # Run multiple analyses concurrently
            tasks = [
                guard.analyze_text(text)
                for text in sample_texts["safe"]
            ]
            
            results = await asyncio.gather(*tasks)
            assert len(results) == len(sample_texts["safe"])
            
            for result_dict in results:
                assert isinstance(result_dict, dict)
                assert "toxicity" in result_dict

    @pytest.mark.asyncio
    async def test_concurrent_protection(self, mock_detector, mock_llm_function):
        """Test concurrent protection operations."""
        with patch('ai_safety_guardrails.core.safety_guard.get_detector_class') as mock_get_detector:
            mock_detector_instance = mock_detector(should_block=False)
            mock_get_detector.return_value = lambda **kwargs: mock_detector_instance
            
            guard = SafetyGuard(detectors=["toxicity"], max_concurrent_detections=3)
            
            # Run multiple protections concurrently
            tasks = [
                guard.protect(input_text=f"Message {i}", llm_function=mock_llm_function)
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            
            for result in results:
                assert not result.blocked
                assert result.response is not None