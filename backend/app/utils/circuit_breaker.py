"""
Circuit Breaker Pattern Implementation for Service Resilience
Author: Udaya Vijay Anand
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    name: str = "circuit_breaker"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation for protecting against cascading failures.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Fast failure, calls immediately raise CircuitBreakerError
    - HALF_OPEN: Recovery testing, single call allowed to test health
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.call_count = 0
        self.success_count = 0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function"""
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.call_count += 1
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {self.config.name} moving to HALF_OPEN state")
            else:
                logger.warning(f"Circuit breaker {self.config.name} is OPEN, failing fast")
                raise CircuitBreakerError(f"Circuit breaker {self.config.name} is open")
        
        try:
            result = await self._execute_call(func, *args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    async def _execute_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the actual function call"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.success_count += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker {self.config.name} recovered, moving to CLOSED state")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker {self.config.name} tripped, moving to OPEN state")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(self.call_count, 1),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "is_healthy": self.state == CircuitBreakerState.CLOSED
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        logger.info(f"Circuit breaker {self.config.name} manually reset")


class CircuitBreakerRegistry:
    """Registry to manage multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def register(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker"""
        config.name = name
        breaker = CircuitBreaker(config)
        self._breakers[name] = breaker
        return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
    
    def get_unhealthy_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers that are not in closed state"""
        return {
            name: breaker for name, breaker in self._breakers.items()
            if breaker.state != CircuitBreakerState.CLOSED
        }


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()