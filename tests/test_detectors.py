"""
Unit tests for AI Safety Guardrails detectors.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from ai_safety_guardrails.detectors.base_detector import BaseDetector, DetectionResult
from ai_safety_guardrails.detectors.toxicity import ToxicityDetector
from ai_safety_guardrails.detectors.pii import PIIDetector
from ai_safety_guardrails.detectors.prompt_injection import PromptInjectionDetector
from ai_safety_guardrails.detectors.spam import SpamDetector
from ai_safety_guardrails.detectors.topics import TopicsDetector
from ai_safety_guardrails.detectors.fact_check import FactCheckDetector


class TestBaseDetector:
    """Test the base detector functionality."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = BaseDetector("test_detector")
        assert detector.name == "test_detector"
        assert not detector._loaded
        assert detector.threshold == 0.5
        assert detector.enabled == True

    def test_detector_configuration(self):
        """Test detector configuration."""
        config = {
            "threshold": 0.8,
            "enabled": False,
            "custom_param": "value"
        }
        detector = BaseDetector("test", **config)
        assert detector.threshold == 0.8
        assert detector.enabled == False
        assert detector.config["custom_param"] == "value"

    @pytest.mark.asyncio
    async def test_safe_detect_success(self, mock_detector):
        """Test successful detection."""
        detector = mock_detector()
        result = await detector.safe_detect("test text")
        
        assert isinstance(result, DetectionResult)
        assert result.detector_name == "mock_detector"
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_safe_detect_failure_open_mode(self, mock_detector):
        """Test detection failure in open mode."""
        detector = mock_detector()
        detector.fail_open = True
        
        # Mock the detect method to raise an exception
        async def failing_detect(text):
            raise Exception("Test failure")
        
        detector.detect = failing_detect
        
        result = await detector.safe_detect("test text")
        
        assert isinstance(result, DetectionResult)
        assert not result.blocked  # Should not block in open mode
        assert "error" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_safe_detect_failure_closed_mode(self, mock_detector):
        """Test detection failure in closed mode."""
        detector = mock_detector()
        detector.fail_open = False
        
        # Mock the detect method to raise an exception
        async def failing_detect(text):
            raise Exception("Test failure")
        
        detector.detect = failing_detect
        
        result = await detector.safe_detect("test text")
        
        assert isinstance(result, DetectionResult)
        assert result.blocked  # Should block in closed mode
        assert "error" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_ensure_loaded(self, mock_detector):
        """Test ensure_loaded functionality."""
        detector = mock_detector()
        detector._loaded = False
        
        await detector.ensure_loaded()
        assert detector._loaded

    def test_get_metrics(self, mock_detector):
        """Test metrics collection."""
        detector = mock_detector()
        metrics = detector.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_calls" in metrics
        assert "successful_calls" in metrics
        assert "failed_calls" in metrics
        assert "blocks" in metrics

    def test_get_health_status(self, mock_detector):
        """Test health status reporting."""
        detector = mock_detector()
        health = detector.get_health_status()
        
        assert health.status in ["healthy", "unhealthy"]
        assert hasattr(health, "load_time")


class TestToxicityDetector:
    """Test the toxicity detector."""

    @pytest.mark.asyncio
    async def test_toxicity_safe_content(self, sample_texts):
        """Test toxicity detector with safe content."""
        with patch('ai_safety_guardrails.detectors.toxicity.pipeline') as mock_pipeline:
            # Mock the pipeline
            mock_classifier = MagicMock()
            mock_classifier.return_value = [{'label': 'TOXIC', 'score': 0.1}]
            mock_pipeline.return_value = mock_classifier
            
            detector = ToxicityDetector(threshold=0.5)
            await detector.load_model()
            
            for text in sample_texts["safe"]:
                result = await detector.detect(text)
                assert isinstance(result, DetectionResult)
                assert not result.blocked
                assert result.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_toxicity_toxic_content(self, sample_texts):
        """Test toxicity detector with toxic content."""
        with patch('ai_safety_guardrails.detectors.toxicity.pipeline') as mock_pipeline:
            # Mock the pipeline to return high toxicity score
            mock_classifier = MagicMock()
            mock_classifier.return_value = [{'label': 'TOXIC', 'score': 0.9}]
            mock_pipeline.return_value = mock_classifier
            
            detector = ToxicityDetector(threshold=0.5)
            await detector.load_model()
            
            result = await detector.detect("This is toxic content")
            assert isinstance(result, DetectionResult)
            assert result.blocked
            assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_toxicity_threshold_adjustment(self):
        """Test toxicity detector with different thresholds."""
        with patch('ai_safety_guardrails.detectors.toxicity.pipeline') as mock_pipeline:
            mock_classifier = MagicMock()
            mock_classifier.return_value = [{'label': 'TOXIC', 'score': 0.7}]
            mock_pipeline.return_value = mock_classifier
            
            # Test with low threshold - should block
            detector_low = ToxicityDetector(threshold=0.5)
            await detector_low.load_model()
            result_low = await detector_low.detect("borderline content")
            assert result_low.blocked
            
            # Test with high threshold - should not block
            detector_high = ToxicityDetector(threshold=0.9)
            await detector_high.load_model()
            result_high = await detector_high.detect("borderline content")
            assert not result_high.blocked


class TestPIIDetector:
    """Test the PII detector."""

    @pytest.mark.asyncio
    async def test_pii_email_detection(self):
        """Test PII detector with email addresses."""
        with patch('ai_safety_guardrails.detectors.pii.spacy.load') as mock_spacy:
            # Mock spaCy model
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            detector = PIIDetector(sensitivity="high")
            await detector.load_model()
            
            result = await detector.detect("Contact me at john.doe@example.com")
            assert isinstance(result, DetectionResult)
            assert result.blocked  # Should detect email pattern

    @pytest.mark.asyncio
    async def test_pii_phone_detection(self):
        """Test PII detector with phone numbers."""
        with patch('ai_safety_guardrails.detectors.pii.spacy.load') as mock_spacy:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            detector = PIIDetector(sensitivity="high")
            await detector.load_model()
            
            result = await detector.detect("Call me at 555-123-4567")
            assert isinstance(result, DetectionResult)
            assert result.blocked  # Should detect phone pattern

    @pytest.mark.asyncio
    async def test_pii_safe_content(self, sample_texts):
        """Test PII detector with safe content."""
        with patch('ai_safety_guardrails.detectors.pii.spacy.load') as mock_spacy:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            detector = PIIDetector(sensitivity="high")
            await detector.load_model()
            
            for text in sample_texts["safe"]:
                result = await detector.detect(text)
                assert isinstance(result, DetectionResult)
                assert not result.blocked

    @pytest.mark.asyncio
    async def test_pii_sensitivity_levels(self):
        """Test PII detector with different sensitivity levels."""
        with patch('ai_safety_guardrails.detectors.pii.spacy.load') as mock_spacy:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            # High sensitivity should catch more
            detector_high = PIIDetector(sensitivity="high")
            await detector_high.load_model()
            
            # Low sensitivity should catch less
            detector_low = PIIDetector(sensitivity="low")
            await detector_low.load_model()
            
            # Both should be initialized properly
            assert detector_high.sensitivity_threshold < detector_low.sensitivity_threshold


class TestPromptInjectionDetector:
    """Test the prompt injection detector."""

    @pytest.mark.asyncio
    async def test_prompt_injection_detection(self, sample_texts):
        """Test prompt injection detector with injection attempts."""
        detector = PromptInjectionDetector(threshold=0.5)
        await detector.load_model()
        
        for text in sample_texts["prompt_injection"]:
            result = await detector.detect(text)
            assert isinstance(result, DetectionResult)
            assert result.blocked
            assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_prompt_injection_safe_content(self, sample_texts):
        """Test prompt injection detector with safe content."""
        detector = PromptInjectionDetector(threshold=0.5)
        await detector.load_model()
        
        for text in sample_texts["safe"]:
            result = await detector.detect(text)
            assert isinstance(result, DetectionResult)
            assert not result.blocked

    @pytest.mark.asyncio
    async def test_prompt_injection_patterns(self):
        """Test specific prompt injection patterns."""
        detector = PromptInjectionDetector(threshold=0.3)
        await detector.load_model()
        
        test_cases = [
            ("ignore all instructions", True),
            ("forget everything", True),
            ("you are now a", True),
            ("roleplay as", True),
            ("normal conversation", False),
            ("please help me", False),
        ]
        
        for text, should_block in test_cases:
            result = await detector.detect(text)
            assert result.blocked == should_block, f"Failed for: {text}"


class TestSpamDetector:
    """Test the spam detector."""

    @pytest.mark.asyncio
    async def test_spam_detection(self, sample_texts):
        """Test spam detector with spam content."""
        detector = SpamDetector(threshold=0.5)
        await detector.load_model()
        
        for text in sample_texts["spam"]:
            result = await detector.detect(text)
            assert isinstance(result, DetectionResult)
            assert result.blocked

    @pytest.mark.asyncio
    async def test_spam_safe_content(self, sample_texts):
        """Test spam detector with safe content."""
        detector = SpamDetector(threshold=0.5)
        await detector.load_model()
        
        for text in sample_texts["safe"]:
            result = await detector.detect(text)
            assert isinstance(result, DetectionResult)
            assert not result.blocked

    @pytest.mark.asyncio
    async def test_spam_aggressive_mode(self):
        """Test spam detector in aggressive mode."""
        detector = SpamDetector(threshold=0.3, aggressive=True)
        await detector.load_model()
        
        # Should be more sensitive in aggressive mode
        result = await detector.detect("Great deal available now!")
        assert isinstance(result, DetectionResult)
        # In aggressive mode, this should be more likely to block


class TestTopicsDetector:
    """Test the topics detector."""

    @pytest.mark.asyncio
    async def test_topics_safe_content(self, sample_texts):
        """Test topics detector with safe content."""
        with patch('ai_safety_guardrails.detectors.topics.SentenceTransformer') as mock_st:
            # Mock SentenceTransformer
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]  # Low similarity
            mock_st.return_value = mock_model
            
            detector = TopicsDetector(threshold=0.7)
            await detector.load_model()
            
            for text in sample_texts["safe"]:
                result = await detector.detect(text)
                assert isinstance(result, DetectionResult)
                assert not result.blocked

    @pytest.mark.asyncio
    async def test_topics_blocked_content(self):
        """Test topics detector with potentially blocked content."""
        with patch('ai_safety_guardrails.detectors.topics.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            # High similarity to blocked topic
            mock_model.encode.return_value = [[0.9, 0.8, 0.7]]
            mock_st.return_value = mock_model
            
            detector = TopicsDetector(threshold=0.5)
            await detector.load_model()
            
            result = await detector.detect("content about violence")
            assert isinstance(result, DetectionResult)
            # Should block due to high similarity


class TestFactCheckDetector:
    """Test the fact check detector."""

    @pytest.mark.asyncio
    async def test_fact_check_detection(self):
        """Test fact check detector."""
        detector = FactCheckDetector(threshold=0.5)
        await detector.load_model()
        
        # Test factual claim
        result = await detector.detect("Studies show that this is true")
        assert isinstance(result, DetectionResult)
        
        # Test non-factual content
        result = await detector.detect("I think this might be interesting")
        assert isinstance(result, DetectionResult)

    @pytest.mark.asyncio
    async def test_fact_check_patterns(self):
        """Test fact check pattern detection."""
        detector = FactCheckDetector(threshold=0.3)
        await detector.load_model()
        
        factual_phrases = [
            "according to research",
            "studies show that",
            "data indicates",
            "proven fact",
            "statistics reveal"
        ]
        
        for phrase in factual_phrases:
            result = await detector.detect(f"This is a statement where {phrase} something.")
            assert isinstance(result, DetectionResult)
            # Should detect factual language patterns


class TestDetectorIntegration:
    """Integration tests for detectors."""

    @pytest.mark.asyncio
    async def test_multiple_detectors_safe_content(self, sample_texts):
        """Test multiple detectors with safe content."""
        detectors = [
            PromptInjectionDetector(),
            SpamDetector(),
            FactCheckDetector()
        ]
        
        for detector in detectors:
            await detector.load_model()
        
        for text in sample_texts["safe"]:
            for detector in detectors:
                result = await detector.safe_detect(text)
                assert isinstance(result, DetectionResult)
                # Most safe content should not be blocked
                # (allowing for some false positives)

    @pytest.mark.asyncio
    async def test_detector_performance(self, sample_texts):
        """Test detector performance characteristics."""
        detector = PromptInjectionDetector()
        await detector.load_model()
        
        # Test with various text lengths
        test_texts = [
            "short",
            "This is a medium length text for testing purposes.",
            "This is a much longer text " * 10 + " for performance testing."
        ]
        
        for text in test_texts:
            result = await detector.safe_detect(text)
            assert isinstance(result, DetectionResult)
            assert result.processing_time > 0
            # Performance should be reasonable (under 1 second for testing)
            assert result.processing_time < 1.0

    @pytest.mark.asyncio
    async def test_detector_edge_cases(self):
        """Test detectors with edge cases."""
        detector = PromptInjectionDetector()
        await detector.load_model()
        
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace
            "a" * 1000,  # Very long string
            "🎉 Unicode: café naïve résumé",  # Unicode characters
            "Line 1\nLine 2\nLine 3",  # Multi-line
        ]
        
        for text in edge_cases:
            result = await detector.safe_detect(text)
            assert isinstance(result, DetectionResult)
            # Should handle edge cases gracefully without crashing