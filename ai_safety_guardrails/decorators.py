"""
Decorator API for AI Safety Guardrails.

Provides transparent safety protection through Python decorators.
"""

import asyncio
import functools
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .core.detector_config import DetectorConfig, SafetyConfig
from .core.safety_guard import SafetyGuard, SafetyResult
from .utils.exceptions import ConfigurationException, SafetyGuardException
from .utils.logger import get_logger

logger = get_logger(__name__)


class DecoratorSafetyGuard:
    """
    Shared SafetyGuard instance for decorators.

    This allows multiple @safe_ai decorators to share the same
    SafetyGuard instance for efficiency.
    """

    _instance: Optional[SafetyGuard] = None
    _config_hash: Optional[str] = None

    @classmethod
    async def get_instance(
        self,
        detectors: Optional[List[Union[str, DetectorConfig]]] = None,
        config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
        **kwargs,
    ) -> SafetyGuard:
        """Get or create a shared SafetyGuard instance."""
        import hashlib
        import json

        # Create a hash of the configuration to detect changes
        config_data = {
            "detectors": [str(d) for d in (detectors or [])],
            "config": str(config) if config else None,
            "kwargs": kwargs,
        }
        config_hash = hashlib.md5(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()

        # Create new instance if config changed or first time
        if self._instance is None or self._config_hash != config_hash:
            logger.info("Creating new shared SafetyGuard instance")
            self._instance = SafetyGuard(detectors=detectors, config=config, **kwargs)
            self._config_hash = config_hash

        return self._instance

    @classmethod
    async def cleanup(self):
        """Clean up the shared instance."""
        if self._instance:
            await self._instance.cleanup()
            self._instance = None
            self._config_hash = None


def safe_ai(
    detectors: Optional[List[Union[str, DetectorConfig]]] = None,
    config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
    check_output: bool = True,
    fail_mode: str = "open",
    shared_instance: bool = True,
    **kwargs,
):
    """
    Decorator for transparent AI safety protection.

    This decorator wraps LLM functions to automatically apply safety checks
    to both input and output without requiring code changes.

    Args:
        detectors: List of detectors to enable. If None, uses config defaults.
        config: Configuration file path, dict, or SafetyConfig object.
        check_output: Whether to check LLM output for safety.
        fail_mode: "open" (allow on failure) or "closed" (block on failure).
        shared_instance: Whether to share SafetyGuard instance across decorators.
        **kwargs: Additional SafetyGuard configuration.

    Example:
        @safe_ai(detectors=["toxicity", "pii"], config="./safety_config.yml")
        def chat_with_ai(user_input):
            return openai.chat.completions.create(...)

        # Usage is completely transparent
        response = chat_with_ai("Hello, how are you?")

    The decorator:
    1. Extracts the first string argument as input text
    2. Runs safety analysis on the input
    3. Calls the wrapped function if input is safe
    4. Optionally checks the output for safety
    5. Returns the result or raises an exception if blocked
    """

    def decorator(func: Callable) -> Callable:
        # Store function metadata
        func_name = func.__name__
        is_async = asyncio.iscoroutinefunction(func)

        # Initialize SafetyGuard (will be done lazily)
        _safety_guard: Optional[SafetyGuard] = None

        async def _get_safety_guard() -> SafetyGuard:
            nonlocal _safety_guard

            if _safety_guard is None:
                if shared_instance:
                    _safety_guard = await DecoratorSafetyGuard.get_instance(
                        detectors=detectors,
                        config=config,
                        fallback_mode=fail_mode,
                        **kwargs,
                    )
                else:
                    _safety_guard = SafetyGuard(
                        detectors=detectors,
                        config=config,
                        fallback_mode=fail_mode,
                        **kwargs,
                    )

            return _safety_guard

        def _extract_input_text(args, kwargs_dict) -> str:
            """Extract input text from function arguments."""
            # Try to find the first string argument
            for arg in args:
                if isinstance(arg, str) and arg.strip():
                    return arg

            # Try common parameter names
            common_names = ["text", "input", "message", "prompt", "query", "content"]
            for name in common_names:
                if name in kwargs_dict and isinstance(kwargs_dict[name], str):
                    return kwargs_dict[name]

            # Try any string parameter
            for value in kwargs_dict.values():
                if isinstance(value, str) and value.strip():
                    return value

            raise SafetyGuardException(
                f"Could not extract input text from function {func_name}. "
                "Ensure the function has a string parameter containing the text to analyze."
            )

        def _extract_context(args, kwargs_dict) -> Dict[str, Any]:
            """Extract context information from function arguments."""
            context = {"function_name": func_name, "is_async": is_async}

            # Look for context-related parameters
            context_names = [
                "context",
                "metadata",
                "user_id",
                "session_id",
                "conversation_id",
            ]
            for name in context_names:
                if name in kwargs_dict:
                    context[name] = kwargs_dict[name]

            return context

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs_dict):
                try:
                    # Extract input text and context
                    input_text = _extract_input_text(args, kwargs_dict)
                    context = _extract_context(args, kwargs_dict)

                    # Get SafetyGuard instance
                    safety_guard = await _get_safety_guard()

                    # Create a wrapper for the original function
                    async def llm_function(text: str) -> str:
                        # Call original function with original arguments
                        result = await func(*args, **kwargs_dict)
                        return str(result) if result is not None else ""

                    # Run safety protection
                    safety_result = await safety_guard.protect(
                        input_text=input_text,
                        llm_function=llm_function,
                        context=context,
                        check_output=check_output,
                    )

                    # Handle result
                    if safety_result.blocked:
                        raise SafetyGuardException(
                            f"Content blocked by AI safety: {safety_result.block_reason}"
                        )

                    # Return the original function result type
                    return await func(*args, **kwargs_dict)

                except SafetyGuardException:
                    # Re-raise safety exceptions
                    raise
                except Exception as e:
                    logger.error(f"Decorator wrapper failed for {func_name}: {e}")

                    if fail_mode == "open":
                        # Fall back to calling the original function
                        logger.warning(
                            f"Falling back to unprotected execution for {func_name}"
                        )
                        return await func(*args, **kwargs_dict)
                    else:
                        raise SafetyGuardException(f"Safety system failure: {e}")

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs_dict):
                try:
                    # Extract input text and context
                    input_text = _extract_input_text(args, kwargs_dict)
                    context = _extract_context(args, kwargs_dict)

                    # Run async operations in event loop
                    async def _async_protection():
                        # Get SafetyGuard instance
                        safety_guard = await _get_safety_guard()

                        # Create a wrapper for the original function
                        def llm_function(text: str) -> str:
                            # Call original function with original arguments
                            result = func(*args, **kwargs_dict)
                            return str(result) if result is not None else ""

                        # Run safety protection
                        return await safety_guard.protect(
                            input_text=input_text,
                            llm_function=llm_function,
                            context=context,
                            check_output=check_output,
                        )

                    # Run async protection
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    safety_result = loop.run_until_complete(_async_protection())

                    # Handle result
                    if safety_result.blocked:
                        raise SafetyGuardException(
                            f"Content blocked by AI safety: {safety_result.block_reason}"
                        )

                    # Return the original function result
                    return func(*args, **kwargs_dict)

                except SafetyGuardException:
                    # Re-raise safety exceptions
                    raise
                except Exception as e:
                    logger.error(f"Decorator wrapper failed for {func_name}: {e}")

                    if fail_mode == "open":
                        # Fall back to calling the original function
                        logger.warning(
                            f"Falling back to unprotected execution for {func_name}"
                        )
                        return func(*args, **kwargs_dict)
                    else:
                        raise SafetyGuardException(f"Safety system failure: {e}")

            return sync_wrapper

    return decorator


def safe_chat(
    detectors: Optional[List[Union[str, DetectorConfig]]] = None,
    config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
    **kwargs,
):
    """
    Specialized decorator for chat functions.

    This is a convenience decorator that assumes the function is for chat/conversation
    and automatically enables appropriate detectors.

    Args:
        detectors: List of detectors. If None, uses ["toxicity", "pii", "prompt_injection"]
        config: Configuration for safety system
        **kwargs: Additional SafetyGuard configuration

    Example:
        @safe_chat()
        def chat_response(user_message):
            return llm.generate(user_message)
    """

    if detectors is None:
        detectors = ["toxicity", "pii", "prompt_injection"]

    return safe_ai(detectors=detectors, config=config, check_output=True, **kwargs)


def safe_generation(
    detectors: Optional[List[Union[str, DetectorConfig]]] = None,
    config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
    **kwargs,
):
    """
    Specialized decorator for content generation functions.

    This decorator is optimized for content generation and includes
    fact-checking and spam detection.

    Args:
        detectors: List of detectors. If None, uses comprehensive set.
        config: Configuration for safety system
        **kwargs: Additional SafetyGuard configuration

    Example:
        @safe_generation()
        def generate_article(topic):
            return llm.generate_long_form(topic)
    """

    if detectors is None:
        detectors = ["toxicity", "pii", "topics", "fact_check", "spam"]

    return safe_ai(detectors=detectors, config=config, check_output=True, **kwargs)


def safe_qa(
    detectors: Optional[List[Union[str, DetectorConfig]]] = None,
    config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
    **kwargs,
):
    """
    Specialized decorator for Q&A functions.

    This decorator is optimized for question-answering scenarios.

    Args:
        detectors: List of detectors. If None, uses ["toxicity", "fact_check"]
        config: Configuration for safety system
        **kwargs: Additional SafetyGuard configuration

    Example:
        @safe_qa()
        def answer_question(question):
            return qa_system.answer(question)
    """

    if detectors is None:
        detectors = ["toxicity", "fact_check"]

    return safe_ai(detectors=detectors, config=config, check_output=True, **kwargs)


# Convenience function for manual safety checking
async def check_safety(
    text: str,
    detectors: Optional[List[Union[str, DetectorConfig]]] = None,
    config: Optional[Union[Dict[str, Any], SafetyConfig, str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Manually check text safety without executing an LLM function.

    Args:
        text: Text to analyze
        detectors: List of detectors to use
        config: Safety configuration
        **kwargs: Additional configuration

    Returns:
        Dictionary with safety analysis results

    Example:
        result = await check_safety("Hello world", detectors=["toxicity", "pii"])
        if result["blocked"]:
            print(f"Blocked: {result['reason']}")
    """

    # Get shared SafetyGuard instance
    safety_guard = await DecoratorSafetyGuard.get_instance(
        detectors=detectors, config=config, **kwargs
    )

    # Analyze text
    detection_results = await safety_guard.analyze_text(text)

    # Determine if blocked
    blocked_detectors = [
        name for name, result in detection_results.items() if result.blocked
    ]
    is_blocked = len(blocked_detectors) > 0

    reasons = []
    for name, result in detection_results.items():
        if result.blocked and result.reason:
            reasons.append(f"{name}: {result.reason}")

    block_reason = "; ".join(reasons) if reasons else None

    return {
        "blocked": is_blocked,
        "reason": block_reason,
        "triggered_detectors": blocked_detectors,
        "detection_results": {
            name: {
                "blocked": result.blocked,
                "confidence": result.confidence,
                "reason": result.reason,
                "processing_time": result.processing_time,
            }
            for name, result in detection_results.items()
        },
        "max_confidence": max(
            (result.confidence for result in detection_results.values()), default=0.0
        ),
    }
