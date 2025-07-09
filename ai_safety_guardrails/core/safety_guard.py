"""
SafetyGuard - Main library API for AI safety guardrails.

This is the primary interface for explicit safety control.
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..detectors import (
    FactCheckDetector,
    PIIDetector,
    PromptInjectionDetector,
    SpamDetector,
    TopicsDetector,
    ToxicityDetector,
)
from ..detectors.base_detector import BaseDetector, DetectionResult
from ..utils.exceptions import ConfigurationException, SafetyGuardException
from ..utils.logger import get_logger
from .detector_config import DetectorConfig, SafetyConfig
from .model_manager import ModelManager

logger = get_logger(__name__)

# Detector class mapping
DETECTOR_CLASSES = {
    "toxicity": ToxicityDetector,
    "pii": PIIDetector,
    "prompt_injection": PromptInjectionDetector,
    "spam": SpamDetector,
    "topics": TopicsDetector,
    "fact_check": FactCheckDetector,
}


def get_detector_class(detector_type: str) -> type:
    """Get detector class by type name."""
    if detector_type not in DETECTOR_CLASSES:
        raise ConfigurationException(f"Unknown detector type: {detector_type}")
    return DETECTOR_CLASSES[detector_type]


@dataclass
class DetectorHealth:
    """Health status for an individual detector."""

    status: str
    load_time: float
    errors: int = 0
    last_used: Optional[float] = None


@dataclass
class HealthStatus:
    """Overall health status of the SafetyGuard system."""

    status: str
    uptime: float
    memory_usage: float
    detectors: Dict[str, DetectorHealth]
    total_requests: int = 0
    failed_requests: int = 0


@dataclass
class SafetyResult:
    """Result of safety analysis and LLM execution."""

    blocked: bool
    response: Optional[str] = None
    block_reason: Optional[str] = None
    input_results: Dict[str, DetectionResult] = field(default_factory=dict)
    output_results: Dict[str, DetectionResult] = field(default_factory=dict)
    processing_time: Optional[float] = None
    llm_execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_results(self) -> Dict[str, DetectionResult]:
        """Get all detection results (input + output)."""
        results = {}
        results.update(self.input_results)
        # Prefix output results to distinguish them
        for name, result in self.output_results.items():
            results[f"output_{name}"] = result
        return results

    @property
    def max_confidence(self) -> float:
        """Get the highest confidence score from all detections."""
        all_confidences = [result.confidence for result in self.all_results.values()]
        return max(all_confidences) if all_confidences else 0.0

    @property
    def triggered_detectors(self) -> List[str]:
        """Get list of detectors that triggered blocks."""
        triggered = []
        for name, result in self.all_results.items():
            if result.blocked:
                triggered.append(name)
        return triggered


class SafetyGuard:
    """
    Main SafetyGuard class for explicit AI safety control.

    Provides comprehensive safety analysis for LLM applications with:
    - Multiple detector types (toxicity, PII, prompt injection, etc.)
    - Configurable thresholds and sensitivity
    - Input and output filtering
    - Performance monitoring
    - Health checking
    """

    # Built-in detector classes
    DETECTOR_CLASSES = {
        "toxicity": ToxicityDetector,
        "pii": PIIDetector,
        "prompt_injection": PromptInjectionDetector,
        "topics": TopicsDetector,
        "fact_check": FactCheckDetector,
        "spam": SpamDetector,
    }

    def __init__(
        self,
        detectors: Optional[List[Union[str, DetectorConfig, BaseDetector]]] = None,
        config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
        model_manager: Optional[ModelManager] = None,
        circuit_breaker: bool = False,
        fallback_mode: str = "open",
        config_file: Optional[Union[str, Path]] = None,
        max_concurrent_detections: int = 5,
    ):
        """
        Initialize SafetyGuard.

        Args:
            detectors: List of detectors to enable. Can be strings, DetectorConfig objects, or detector instances.
            config: Configuration dictionary, SafetyConfig object, or path to config file.
            model_manager: Optional model manager instance.
            circuit_breaker: Enable circuit breaker for fault tolerance.
            fallback_mode: "open" (allow on failure) or "closed" (block on failure).
            config_file: DEPRECATED: Use 'config' instead. Path to config file.
            max_concurrent_detections: Maximum number of concurrent detector operations.
        """
        # Handle deprecated config_file parameter
        if config_file is not None and config is None:
            config = config_file
        elif config_file is not None and config is not None:
            raise ConfigurationException(
                "Cannot specify both 'config' and 'config_file'. Use 'config' instead."
            )
        self.detectors: Dict[str, BaseDetector] = {}
        self.model_manager = model_manager or ModelManager()
        self.circuit_breaker = circuit_breaker
        self.fallback_mode = fallback_mode
        self.max_concurrent_detections = max_concurrent_detections

        # Load configuration
        self.config = self._load_config(config)

        # Performance tracking
        self.total_requests = 0
        self.blocked_requests = 0
        self.total_processing_time = 0.0
        self.detector_metrics: dict[str, dict] = {}

        # Initialize detectors
        self._initialize_detectors(detectors)

        # Health status
        self.is_healthy = True
        self.last_health_check = None

        logger.info(f"SafetyGuard initialized with {len(self.detectors)} detectors")

    def _load_config(
        self, config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]]
    ) -> SafetyConfig:
        """Load and validate configuration."""
        if config is None:
            return SafetyConfig()
        elif isinstance(config, SafetyConfig):
            return config
        elif isinstance(config, dict):
            return SafetyConfig(config)
        elif isinstance(config, (str, Path)):
            return SafetyConfig.from_file(config)
        else:
            raise ConfigurationException(f"Invalid config type: {type(config)}")

    def _initialize_detectors(
        self, detectors: Optional[List[Union[str, DetectorConfig, BaseDetector]]]
    ):
        """Initialize detector instances."""
        if detectors is None:
            # Use enabled detectors from config
            detectors = []
            for name, config in self.config.get_detector_config("").items():
                if config.get("enabled", True):
                    detectors.append(name)

        for detector_spec in detectors:
            if isinstance(detector_spec, str):
                # Simple string specification
                detector_config = DetectorConfig.from_string(detector_spec)
                detector = self._create_detector(detector_config)
            elif isinstance(detector_spec, DetectorConfig):
                # DetectorConfig object
                detector = self._create_detector(detector_spec)
            elif isinstance(detector_spec, BaseDetector):
                # Already instantiated detector
                detector = detector_spec
            else:
                raise ConfigurationException(
                    f"Invalid detector specification: {detector_spec}"
                )

            self.detectors[detector.name] = detector
            self.detector_metrics[detector.name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
            }

    def _create_detector(self, config: DetectorConfig) -> BaseDetector:
        """Create a detector instance from configuration."""
        if config.name not in self.DETECTOR_CLASSES:
            raise ConfigurationException(f"Unknown detector type: {config.name}")

        detector_class = self.DETECTOR_CLASSES[config.name]

        # Merge config with global settings
        merged_config = config.merge_with_global_config(self.config.to_dict())

        # Create detector instance
        if config.model:
            merged_config["model_name"] = config.model

        return detector_class(**merged_config)

    async def protect(
        self,
        input_text: str,
        llm_function: Callable[[str], Union[str, Awaitable[str]]],
        context: Optional[Dict[str, Any]] = None,
        check_output: bool = True,
    ) -> SafetyResult:
        """
        Protect an LLM interaction with safety analysis.

        Args:
            input_text: User input to analyze
            llm_function: Function that calls the LLM
            context: Optional context information
            check_output: Whether to check LLM output for safety

        Returns:
            SafetyResult with analysis and LLM response
        """
        start_time = time.time()
        self.total_requests += 1

        try:
            # Step 1: Analyze input
            input_results = await self._analyze_text(input_text, context, "input")

            # Step 2: Check if input should be blocked
            input_blocked, block_reason = self._should_block(input_results)

            if input_blocked:
                self.blocked_requests += 1
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time

                return SafetyResult(
                    blocked=True,
                    block_reason=block_reason,
                    input_results=input_results,
                    processing_time=processing_time,
                )

            # Step 3: Execute LLM function
            llm_start_time = time.time()

            if asyncio.iscoroutinefunction(llm_function):
                llm_response = await llm_function(input_text)
            else:
                llm_response = llm_function(input_text)

            llm_execution_time = time.time() - llm_start_time

            # Step 4: Analyze output (if enabled)
            output_results = {}
            if check_output and llm_response:
                output_results = await self._analyze_text(
                    llm_response, context, "output"
                )

                output_blocked, output_block_reason = self._should_block(output_results)

                if output_blocked:
                    self.blocked_requests += 1
                    processing_time = time.time() - start_time
                    self.total_processing_time += processing_time

                    return SafetyResult(
                        blocked=True,
                        block_reason=f"Output blocked: {output_block_reason}",
                        input_results=input_results,
                        output_results=output_results,
                        processing_time=processing_time,
                        llm_execution_time=llm_execution_time,
                    )

            # Step 5: Return successful result
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            return SafetyResult(
                blocked=False,
                response=llm_response,
                input_results=input_results,
                output_results=output_results,
                processing_time=processing_time,
                llm_execution_time=llm_execution_time,
            )

        except Exception as e:
            logger.error(f"SafetyGuard protection failed: {e}")

            # Handle failure based on fallback mode
            if self.fallback_mode == "open":
                # Fail open - allow the request but log the error
                try:
                    if asyncio.iscoroutinefunction(llm_function):
                        llm_response = await llm_function(input_text)
                    else:
                        llm_response = llm_function(input_text)

                    return SafetyResult(
                        blocked=False,
                        response=llm_response,
                        block_reason=f"Safety check failed: {str(e)}",
                        processing_time=time.time() - start_time,
                    )
                except Exception as llm_error:
                    raise SafetyGuardException(
                        f"Both safety check and LLM execution failed: {e}, {llm_error}"
                    )
            else:
                # Fail closed - block the request
                self.blocked_requests += 1
                return SafetyResult(
                    blocked=True,
                    block_reason=f"Safety system failure: {str(e)}",
                    processing_time=time.time() - start_time,
                )

    async def analyze_text(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DetectionResult]:
        """
        Analyze text with all enabled detectors.

        Args:
            text: Text to analyze
            context: Optional context information

        Returns:
            Dictionary of detection results by detector name
        """
        return await self._analyze_text(text, context, "analysis")

    async def _analyze_text(
        self, text: str, context: Optional[Dict[str, Any]], analysis_type: str
    ) -> Dict[str, DetectionResult]:
        """Internal method to analyze text with all detectors."""
        if not text.strip():
            return {}

        # Run all detectors concurrently
        tasks = []
        detector_names = []

        for name, detector in self.detectors.items():
            if not detector.config.get("enabled", True):
                continue

            task = self._run_detector_safely(detector, text, context)
            tasks.append(task)
            detector_names.append(name)

        # Wait for all detections to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        detection_results = {}
        for name, result in zip(detector_names, results):
            if isinstance(result, Exception):
                logger.error(f"Detector {name} failed: {result}")
                # Create failure result
                detection_results[name] = DetectionResult(
                    blocked=False if self.fallback_mode == "open" else True,
                    confidence=0.0,
                    reason=f"Detector failed: {str(result)}",
                    detector_name=name,
                )
            else:
                detection_results[name] = result

        return detection_results

    async def _run_detector_safely(
        self, detector: BaseDetector, text: str, context: Optional[Dict[str, Any]]
    ) -> DetectionResult:
        """Run a detector with error handling and metrics."""
        start_time = time.time()
        detector_name = detector.name

        try:
            self.detector_metrics[detector_name]["total_calls"] += 1

            result = await detector.safe_detect(text, context)

            # Update metrics
            execution_time = time.time() - start_time
            self.detector_metrics[detector_name]["successful_calls"] += 1
            self.detector_metrics[detector_name]["total_time"] += execution_time
            self.detector_metrics[detector_name]["avg_time"] = (
                self.detector_metrics[detector_name]["total_time"]
                / self.detector_metrics[detector_name]["successful_calls"]
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.detector_metrics[detector_name]["failed_calls"] += 1
            self.detector_metrics[detector_name]["total_time"] += execution_time

            logger.error(f"Detector {detector_name} failed: {e}")

            # Return safe fallback
            return DetectionResult(
                blocked=False if self.fallback_mode == "open" else True,
                confidence=0.0,
                reason=f"Detector error: {str(e)}",
                detector_name=detector_name,
                processing_time=execution_time,
            )

    def _should_block(
        self, results: Dict[str, DetectionResult]
    ) -> tuple[bool, Optional[str]]:
        """Determine if content should be blocked based on detection results."""
        blocked_detectors = []
        reasons = []

        for name, result in results.items():
            if result.blocked:
                blocked_detectors.append(name)
                if result.reason:
                    reasons.append(f"{name}: {result.reason}")

        if blocked_detectors:
            reason = (
                "; ".join(reasons)
                if reasons
                else f"Blocked by: {', '.join(blocked_detectors)}"
            )
            return True, reason

        return False, None

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all components."""
        self.last_health_check = time.time()

        detector_health = {}
        overall_healthy = True

        for name, detector in self.detectors.items():
            health = detector.get_health_status()
            detector_health[name] = health

            if not health["healthy"]:
                overall_healthy = False

        # Check model manager health
        model_manager_health = (
            await self.model_manager.health_check()
            if hasattr(self.model_manager, "health_check")
            else {"healthy": True}
        )

        if not model_manager_health["healthy"]:
            overall_healthy = False

        self.is_healthy = overall_healthy

        return {
            "overall_healthy": overall_healthy,
            "detectors": detector_health,
            "model_manager": model_manager_health,
            "metrics": self.get_metrics(),
            "timestamp": self.last_health_check,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics."""
        success_rate = (
            (self.total_requests - self.blocked_requests) / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        avg_processing_time = (
            self.total_processing_time / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "success_rate": success_rate,
            "block_rate": 1.0 - success_rate,
            "avg_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time,
            "detector_metrics": self.detector_metrics.copy(),
            "detectors_count": len(self.detectors),
            "enabled_detectors": [
                name
                for name, detector in self.detectors.items()
                if detector.config.get("enabled", True)
            ],
        }

    def get_detector(self, name: str) -> Optional[BaseDetector]:
        """Get a detector by name."""
        return self.detectors.get(name)

    def enable_detector(self, name: str) -> None:
        """Enable a detector."""
        if name in self.detectors:
            self.detectors[name].config["enabled"] = True
            logger.info(f"Enabled detector: {name}")
        else:
            raise SafetyGuardException(f"Detector not found: {name}")

    def disable_detector(self, name: str) -> None:
        """Disable a detector."""
        if name in self.detectors:
            self.detectors[name].config["enabled"] = False
            logger.info(f"Disabled detector: {name}")
        else:
            raise SafetyGuardException(f"Detector not found: {name}")

    def list_detectors(self) -> List[str]:
        """Get list of all detector names."""
        return list(self.detectors.keys())

    def update_detector_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific detector."""
        if name not in self.detectors:
            raise SafetyGuardException(f"Detector not found: {name}")
        
        detector = self.detectors[name]
        detector.config.update(config)
        logger.info(f"Updated configuration for detector: {name}")

    def add_detector(self, detector_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Add a new detector to the SafetyGuard."""
        if detector_name in self.detectors:
            raise SafetyGuardException(f"Detector already exists: {detector_name}")
        
        detector_config = DetectorConfig(
            name=detector_name,
            type=detector_name,
            config=config or {}
        )
        
        detector = self._create_detector(detector_config)
        self.detectors[detector_name] = detector
        logger.info(f"Added detector: {detector_name}")

    def remove_detector(self, name: str) -> None:
        """Remove a detector from the SafetyGuard."""
        if name not in self.detectors:
            raise SafetyGuardException(f"Detector not found: {name}")
        
        detector = self.detectors.pop(name)
        logger.info(f"Removed detector: {name}")

    async def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up SafetyGuard")

        for detector in self.detectors.values():
            try:
                await detector.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup detector {detector.name}: {e}")

        if hasattr(self.model_manager, "cleanup"):
            await self.model_manager.cleanup()

    def __repr__(self) -> str:
        return f"SafetyGuard(detectors={list(self.detectors.keys())}, healthy={self.is_healthy})"
