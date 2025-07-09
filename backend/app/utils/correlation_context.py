"""
Correlation ID and Request Context Management
Author: Udaya Vijay Anand
"""

import asyncio
import contextvars
import uuid
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_trace_event(self, event_type: str, component: str, details: Dict[str, Any] = None):
        """Add a trace event to the request context"""
        trace_event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "component": component,
            "details": details or {},
            "elapsed_time": time.time() - self.start_time
        }
        self.trace_data.append(trace_event)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since request start"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "start_time": self.start_time,
            "elapsed_time": self.get_elapsed_time(),
            "metadata": self.metadata,
            "trace_data": self.trace_data
        }


# Context variable to store the current request context
_request_context: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    'request_context', 
    default=None
)


class CorrelationContextManager:
    """Manages correlation IDs and request context across async operations"""
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}"
    
    @staticmethod
    def create_context(
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RequestContext:
        """Create a new request context"""
        return RequestContext(
            correlation_id=correlation_id or CorrelationContextManager.generate_correlation_id(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id or str(uuid.uuid4()),
            metadata=metadata or {}
        )
    
    @staticmethod
    def set_context(context: RequestContext) -> RequestContext:
        """Set the current request context"""
        _request_context.set(context)
        logger.debug(f"Set request context [correlation_id: {context.correlation_id}]")
        return context
    
    @staticmethod
    def get_context() -> Optional[RequestContext]:
        """Get the current request context"""
        return _request_context.get()
    
    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get the current correlation ID"""
        context = CorrelationContextManager.get_context()
        return context.correlation_id if context else None
    
    @staticmethod
    def ensure_context(
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RequestContext:
        """Ensure a request context exists, create if needed"""
        context = CorrelationContextManager.get_context()
        if context is None:
            context = CorrelationContextManager.create_context(
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )
            CorrelationContextManager.set_context(context)
        return context
    
    @staticmethod
    def add_metadata(key: str, value: Any):
        """Add metadata to the current context"""
        context = CorrelationContextManager.get_context()
        if context:
            context.metadata[key] = value
    
    @staticmethod
    def add_trace_event(event_type: str, component: str, details: Dict[str, Any] = None):
        """Add a trace event to the current context"""
        context = CorrelationContextManager.get_context()
        if context:
            context.add_trace_event(event_type, component, details)
    
    @staticmethod
    def clear_context():
        """Clear the current request context"""
        _request_context.set(None)


class ContextualLogger:
    """Logger that automatically includes correlation context"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def _get_context_info(self) -> str:
        """Get context information for logging"""
        context = CorrelationContextManager.get_context()
        if context:
            parts = [f"correlation_id: {context.correlation_id}"]
            if context.user_id:
                parts.append(f"user_id: {context.user_id}")
            if context.session_id:
                parts.append(f"session_id: {context.session_id}")
            return f"[{', '.join(parts)}]"
        return ""
    
    def debug(self, message: str, *args, **kwargs):
        context_info = self._get_context_info()
        self.logger.debug(f"{message} {context_info}", *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        context_info = self._get_context_info()
        self.logger.info(f"{message} {context_info}", *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        context_info = self._get_context_info()
        self.logger.warning(f"{message} {context_info}", *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        context_info = self._get_context_info()
        self.logger.error(f"{message} {context_info}", *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        context_info = self._get_context_info()
        self.logger.critical(f"{message} {context_info}", *args, **kwargs)


def with_correlation_context(
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Decorator to ensure a function runs with correlation context"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Save existing context
                existing_context = CorrelationContextManager.get_context()
                
                # Create new context if needed
                if existing_context is None:
                    context = CorrelationContextManager.create_context(
                        correlation_id=correlation_id,
                        user_id=user_id,
                        session_id=session_id,
                        metadata=metadata
                    )
                    CorrelationContextManager.set_context(context)
                
                try:
                    # Add function start trace
                    CorrelationContextManager.add_trace_event(
                        "function_start",
                        func.__name__,
                        {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                    )
                    
                    result = await func(*args, **kwargs)
                    
                    # Add function end trace
                    CorrelationContextManager.add_trace_event(
                        "function_end",
                        func.__name__,
                        {"success": True}
                    )
                    
                    return result
                    
                except Exception as e:
                    # Add function error trace
                    CorrelationContextManager.add_trace_event(
                        "function_error",
                        func.__name__,
                        {"error": str(e), "error_type": type(e).__name__}
                    )
                    raise
                    
                finally:
                    # Restore original context if we created a new one
                    if existing_context is None:
                        CorrelationContextManager.clear_context()
                    else:
                        CorrelationContextManager.set_context(existing_context)
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Save existing context
                existing_context = CorrelationContextManager.get_context()
                
                # Create new context if needed
                if existing_context is None:
                    context = CorrelationContextManager.create_context(
                        correlation_id=correlation_id,
                        user_id=user_id,
                        session_id=session_id,
                        metadata=metadata
                    )
                    CorrelationContextManager.set_context(context)
                
                try:
                    # Add function start trace
                    CorrelationContextManager.add_trace_event(
                        "function_start",
                        func.__name__,
                        {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                    )
                    
                    result = func(*args, **kwargs)
                    
                    # Add function end trace
                    CorrelationContextManager.add_trace_event(
                        "function_end",
                        func.__name__,
                        {"success": True}
                    )
                    
                    return result
                    
                except Exception as e:
                    # Add function error trace
                    CorrelationContextManager.add_trace_event(
                        "function_error",
                        func.__name__,
                        {"error": str(e), "error_type": type(e).__name__}
                    )
                    raise
                    
                finally:
                    # Restore original context if we created a new one
                    if existing_context is None:
                        CorrelationContextManager.clear_context()
                    else:
                        CorrelationContextManager.set_context(existing_context)
            
            return sync_wrapper
    
    return decorator


def get_request_trace() -> Optional[Dict[str, Any]]:
    """Get the complete request trace"""
    context = CorrelationContextManager.get_context()
    if context:
        return context.to_dict()
    return None


def create_child_context(operation_name: str) -> RequestContext:
    """Create a child context for sub-operations"""
    parent_context = CorrelationContextManager.get_context()
    
    if parent_context:
        # Create child context with same correlation ID but new request ID
        child_context = CorrelationContextManager.create_context(
            correlation_id=parent_context.correlation_id,
            user_id=parent_context.user_id,
            session_id=parent_context.session_id,
            request_id=f"{parent_context.request_id}_{operation_name}",
            metadata=parent_context.metadata.copy()
        )
        
        # Add trace event for child creation
        parent_context.add_trace_event(
            "child_context_created",
            operation_name,
            {"child_request_id": child_context.request_id}
        )
        
        return child_context
    else:
        # No parent context, create new one
        return CorrelationContextManager.create_context(
            request_id=operation_name,
            metadata={"operation": operation_name}
        )


class RequestTracker:
    """Track and analyze request patterns"""
    
    def __init__(self):
        self.active_requests: Dict[str, RequestContext] = {}
        self.completed_requests: List[Dict[str, Any]] = []
        self.max_completed_history = 1000
    
    def start_tracking(self, context: RequestContext):
        """Start tracking a request"""
        self.active_requests[context.correlation_id] = context
        logger.debug(f"Started tracking request [correlation_id: {context.correlation_id}]")
    
    def complete_tracking(self, correlation_id: str):
        """Complete tracking a request"""
        if correlation_id in self.active_requests:
            context = self.active_requests.pop(correlation_id)
            
            # Add to completed requests
            completed_request = context.to_dict()
            completed_request["completed_at"] = time.time()
            
            self.completed_requests.append(completed_request)
            
            # Maintain history limit
            if len(self.completed_requests) > self.max_completed_history:
                self.completed_requests.pop(0)
            
            logger.debug(f"Completed tracking request [correlation_id: {correlation_id}]")
    
    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active requests"""
        return {
            corr_id: context.to_dict() 
            for corr_id, context in self.active_requests.items()
        }
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        active_count = len(self.active_requests)
        completed_count = len(self.completed_requests)
        
        if completed_count > 0:
            avg_duration = sum(
                req["elapsed_time"] for req in self.completed_requests
            ) / completed_count
            
            # Find requests with traces
            traced_requests = [req for req in self.completed_requests if req["trace_data"]]
            
            return {
                "active_requests": active_count,
                "completed_requests": completed_count,
                "average_duration": avg_duration,
                "traced_requests": len(traced_requests),
                "trace_coverage": len(traced_requests) / completed_count if completed_count > 0 else 0
            }
        else:
            return {
                "active_requests": active_count,
                "completed_requests": 0,
                "average_duration": 0,
                "traced_requests": 0,
                "trace_coverage": 0
            }


# Global request tracker instance
request_tracker = RequestTracker()