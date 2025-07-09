"""
Retry Logic with Exponential Backoff and Jitter
Author: Udaya Vijay Anand
"""

import asyncio
import random
import time
import functools
from typing import Callable, Any, Type, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    stop_exceptions: Tuple[Type[Exception], ...] = ()
    timeout_per_attempt: Optional[float] = None


@dataclass
class RetryAttempt:
    attempt_number: int
    delay: float
    exception: Optional[Exception]
    start_time: float
    end_time: Optional[float] = None
    success: bool = False


class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted"""
    
    def __init__(self, attempts: list, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {len(attempts)} attempts. Last error: {last_exception}")


class RetryHandler:
    """Advanced retry handler with multiple strategies and detailed tracking"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempt_history: Dict[str, list] = {}
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic to a function"""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
    
    def _async_wrapper(self, func: Callable) -> Callable:
        """Async wrapper for retry logic"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute_async(func, *args, **kwargs)
        return wrapper
    
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Sync wrapper for retry logic"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_sync(func, *args, **kwargs)
        return wrapper
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic"""
        func_name = getattr(func, '__name__', 'unknown')
        attempts = []
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                delay=0,
                exception=None,
                start_time=time.time()
            )
            
            try:
                logger.info(f"Attempting {func_name} (attempt {attempt_num}/{self.config.max_attempts})")
                
                # Apply timeout if configured
                if self.config.timeout_per_attempt:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout_per_attempt
                    )
                else:
                    result = await func(*args, **kwargs)
                
                attempt.end_time = time.time()
                attempt.success = True
                attempts.append(attempt)
                
                # Log successful attempt
                duration = attempt.end_time - attempt.start_time
                logger.info(f"{func_name} succeeded on attempt {attempt_num} after {duration:.2f}s")
                
                # Store attempt history
                self._store_attempt_history(func_name, attempts)
                
                return result
                
            except self.config.stop_exceptions as e:
                # Don't retry for stop exceptions
                attempt.end_time = time.time()
                attempt.exception = e
                attempts.append(attempt)
                
                logger.error(f"{func_name} failed with stop exception on attempt {attempt_num}: {e}")
                self._store_attempt_history(func_name, attempts)
                raise e
                
            except self.config.retry_exceptions as e:
                attempt.end_time = time.time()
                attempt.exception = e
                attempts.append(attempt)
                
                if attempt_num == self.config.max_attempts:
                    # Final attempt failed
                    logger.error(f"{func_name} failed after {self.config.max_attempts} attempts")
                    self._store_attempt_history(func_name, attempts)
                    raise RetryExhaustedException(attempts, e)
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt_num)
                attempt.delay = delay
                
                logger.warning(f"{func_name} failed on attempt {attempt_num}: {e}. Retrying in {delay:.2f}s")
                
                # Wait before next attempt
                await asyncio.sleep(delay)
                
            except Exception as e:
                # Unexpected exception type
                attempt.end_time = time.time()
                attempt.exception = e
                attempts.append(attempt)
                
                logger.error(f"{func_name} failed with unexpected exception: {e}")
                self._store_attempt_history(func_name, attempts)
                raise e
        
        # This should never be reached
        raise RetryExhaustedException(attempts, attempts[-1].exception)
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with retry logic"""
        func_name = getattr(func, '__name__', 'unknown')
        attempts = []
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                delay=0,
                exception=None,
                start_time=time.time()
            )
            
            try:
                logger.info(f"Attempting {func_name} (attempt {attempt_num}/{self.config.max_attempts})")
                
                result = func(*args, **kwargs)
                
                attempt.end_time = time.time()
                attempt.success = True
                attempts.append(attempt)
                
                # Log successful attempt
                duration = attempt.end_time - attempt.start_time
                logger.info(f"{func_name} succeeded on attempt {attempt_num} after {duration:.2f}s")
                
                # Store attempt history
                self._store_attempt_history(func_name, attempts)
                
                return result
                
            except self.config.stop_exceptions as e:
                # Don't retry for stop exceptions
                attempt.end_time = time.time()
                attempt.exception = e
                attempts.append(attempt)
                
                logger.error(f"{func_name} failed with stop exception on attempt {attempt_num}: {e}")
                self._store_attempt_history(func_name, attempts)
                raise e
                
            except self.config.retry_exceptions as e:
                attempt.end_time = time.time()
                attempt.exception = e
                attempts.append(attempt)
                
                if attempt_num == self.config.max_attempts:
                    # Final attempt failed
                    logger.error(f"{func_name} failed after {self.config.max_attempts} attempts")
                    self._store_attempt_history(func_name, attempts)
                    raise RetryExhaustedException(attempts, e)
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt_num)
                attempt.delay = delay
                
                logger.warning(f"{func_name} failed on attempt {attempt_num}: {e}. Retrying in {delay:.2f}s")
                
                # Wait before next attempt
                time.sleep(delay)
                
            except Exception as e:
                # Unexpected exception type
                attempt.end_time = time.time()
                attempt.exception = e
                attempts.append(attempt)
                
                logger.error(f"{func_name} failed with unexpected exception: {e}")
                self._store_attempt_history(func_name, attempts)
                raise e
        
        # This should never be reached
        raise RetryExhaustedException(attempts, attempts[-1].exception)
    
    def _calculate_delay(self, attempt_num: int) -> float:
        """Calculate delay for next retry attempt"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt_num
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt_num - 1))
        else:
            # Custom strategy (default to exponential)
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt_num - 1))
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure delay is not negative
        
        return delay
    
    def _store_attempt_history(self, func_name: str, attempts: list):
        """Store attempt history for analysis"""
        if func_name not in self.attempt_history:
            self.attempt_history[func_name] = []
        
        self.attempt_history[func_name].append({
            "timestamp": time.time(),
            "attempts": attempts,
            "total_attempts": len(attempts),
            "success": attempts[-1].success if attempts else False,
            "total_duration": sum(a.end_time - a.start_time for a in attempts if a.end_time),
            "total_delay": sum(a.delay for a in attempts)
        })
        
        # Keep only last 100 executions per function
        if len(self.attempt_history[func_name]) > 100:
            self.attempt_history[func_name].pop(0)
    
    def get_retry_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics"""
        if func_name:
            if func_name not in self.attempt_history:
                return {"error": f"No history found for function {func_name}"}
            
            history = self.attempt_history[func_name]
        else:
            # Aggregate stats for all functions
            history = []
            for func_history in self.attempt_history.values():
                history.extend(func_history)
        
        if not history:
            return {"message": "No retry attempts recorded"}
        
        total_executions = len(history)
        successful_executions = sum(1 for h in history if h["success"])
        failed_executions = total_executions - successful_executions
        
        total_attempts = sum(h["total_attempts"] for h in history)
        avg_attempts = total_attempts / total_executions if total_executions > 0 else 0
        
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": success_rate,
            "total_attempts": total_attempts,
            "average_attempts_per_execution": avg_attempts,
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
                "strategy": self.config.strategy.value
            }
        }


# Predefined retry configurations
class RetryConfigs:
    """Common retry configurations"""
    
    # Fast retry for quick operations
    FAST = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        backoff_multiplier=2.0
    )
    
    # Standard retry for normal operations
    STANDARD = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0
    )
    
    # Patient retry for slow operations
    PATIENT = RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=120.0,
        backoff_multiplier=2.0
    )
    
    # Network operations
    NETWORK = RetryConfig(
        max_attempts=4,
        base_delay=1.0,
        max_delay=60.0,
        backoff_multiplier=2.0,
        retry_exceptions=(ConnectionError, TimeoutError, OSError)
    )
    
    # Database operations
    DATABASE = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        backoff_multiplier=2.0,
        timeout_per_attempt=30.0
    )


# Convenience decorators
def retry_fast(func):
    """Fast retry decorator"""
    return RetryHandler(RetryConfigs.FAST)(func)

def retry_standard(func):
    """Standard retry decorator"""
    return RetryHandler(RetryConfigs.STANDARD)(func)

def retry_patient(func):
    """Patient retry decorator"""
    return RetryHandler(RetryConfigs.PATIENT)(func)

def retry_network(func):
    """Network retry decorator"""
    return RetryHandler(RetryConfigs.NETWORK)(func)

def retry_database(func):
    """Database retry decorator"""
    return RetryHandler(RetryConfigs.DATABASE)(func)

def retry_custom(config: RetryConfig):
    """Custom retry decorator"""
    return RetryHandler(config)