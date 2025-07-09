"""
Performance Monitoring and Metrics Collection
Author: Udaya Vijay Anand
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import functools
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    request_count: int = 0
    request_duration_total: float = 0.0
    request_duration_avg: float = 0.0
    request_duration_min: float = float('inf')
    request_duration_max: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "request_duration_total": self.request_duration_total,
            "request_duration_avg": self.request_duration_avg,
            "request_duration_min": self.request_duration_min if self.request_duration_min != float('inf') else 0.0,
            "request_duration_max": self.request_duration_max,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "active_connections": self.active_connections
        }


class MetricsCollector:
    """Collect and store time-series metrics"""
    
    def __init__(self, max_points: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.max_points = max_points
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric point"""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            self.metrics[name].append(point)
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        with self._lock:
            if name in self.metrics:
                return [point for point in self.metrics[name] if point.timestamp >= cutoff_time]
            return []
    
    def get_metric_stats(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for a metric"""
        points = self.get_metric_history(name, hours)
        
        if not points:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "latest": 0}
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            return {
                name: {
                    "current_value": points[-1].value if points else 0,
                    "point_count": len(points),
                    "stats": self.get_metric_stats(name, 1)
                }
                for name, points in self.metrics.items()
            }


class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitoring_task = None
        self.monitoring_interval = 30  # seconds
        
        # Request tracking
        self.active_requests = {}
        self.request_history = deque(maxlen=10000)
        
        # Function performance tracking
        self.function_stats = defaultdict(lambda: {
            "call_count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0
        })
        
        self._lock = threading.Lock()
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Record metrics
            self.metrics_collector.record_metric("system_memory_percent", memory.percent)
            self.metrics_collector.record_metric("system_cpu_percent", cpu_percent)
            self.metrics_collector.record_metric("process_memory_mb", process_memory.rss / 1024 / 1024)
            self.metrics_collector.record_metric("process_cpu_percent", process.cpu_percent())
            
            # Garbage collection metrics
            gc_counts = gc.get_count()
            self.metrics_collector.record_metric("gc_gen0_count", gc_counts[0])
            self.metrics_collector.record_metric("gc_gen1_count", gc_counts[1])
            self.metrics_collector.record_metric("gc_gen2_count", gc_counts[2])
            
            # Update performance metrics
            self.performance_metrics.memory_usage_mb = process_memory.rss / 1024 / 1024
            self.performance_metrics.cpu_usage_percent = cpu_percent
            
            # Active connections
            try:
                connections = len(process.connections())
                self.metrics_collector.record_metric("active_connections", connections)
                self.performance_metrics.active_connections = connections
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def track_request(self, request_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Start tracking a request"""
        with self._lock:
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "metadata": metadata or {}
            }
    
    def complete_request(self, request_id: str, success: bool = True, error_message: str = ""):
        """Complete tracking a request"""
        with self._lock:
            if request_id in self.active_requests:
                request_info = self.active_requests.pop(request_id)
                duration = time.time() - request_info["start_time"]
                
                # Update metrics
                self.performance_metrics.request_count += 1
                self.performance_metrics.request_duration_total += duration
                self.performance_metrics.request_duration_avg = (
                    self.performance_metrics.request_duration_total / self.performance_metrics.request_count
                )
                
                if duration < self.performance_metrics.request_duration_min:
                    self.performance_metrics.request_duration_min = duration
                if duration > self.performance_metrics.request_duration_max:
                    self.performance_metrics.request_duration_max = duration
                
                if not success:
                    self.performance_metrics.error_count += 1
                    self.performance_metrics.error_rate = (
                        self.performance_metrics.error_count / self.performance_metrics.request_count
                    )
                
                # Store in history
                self.request_history.append({
                    "request_id": request_id,
                    "duration": duration,
                    "success": success,
                    "error_message": error_message,
                    "timestamp": time.time(),
                    "metadata": request_info["metadata"]
                })
                
                # Record metrics
                self.metrics_collector.record_metric("request_duration", duration)
                self.metrics_collector.record_metric("request_count", 1)
                if not success:
                    self.metrics_collector.record_metric("error_count", 1)
    
    def track_function_performance(self, func_name: str, duration: float, success: bool = True):
        """Track function performance"""
        with self._lock:
            stats = self.function_stats[func_name]
            stats["call_count"] += 1
            stats["total_time"] += duration
            stats["avg_time"] = stats["total_time"] / stats["call_count"]
            
            if duration < stats["min_time"]:
                stats["min_time"] = duration
            if duration > stats["max_time"]:
                stats["max_time"] = duration
            
            if not success:
                stats["error_count"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "current_metrics": self.performance_metrics.to_dict(),
            "system_metrics": self.metrics_collector.get_all_metrics(),
            "active_requests": len(self.active_requests),
            "request_history_size": len(self.request_history),
            "function_stats": dict(self.function_stats),
            "monitoring_active": self.monitoring_active
        }
    
    def get_recent_requests(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent request history"""
        with self._lock:
            return list(self.request_history)[-count:]
    
    def get_slow_requests(self, threshold_seconds: float = 5.0, hours: int = 24) -> List[Dict[str, Any]]:
        """Get slow requests above threshold"""
        cutoff_time = time.time() - (hours * 3600)
        with self._lock:
            return [
                req for req in self.request_history 
                if req["duration"] > threshold_seconds and req["timestamp"] >= cutoff_time
            ]
    
    def get_error_requests(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get error requests"""
        cutoff_time = time.time() - (hours * 3600)
        with self._lock:
            return [
                req for req in self.request_history 
                if not req["success"] and req["timestamp"] >= cutoff_time
            ]
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self.performance_metrics = PerformanceMetrics()
            self.function_stats.clear()
            self.request_history.clear()
            self.active_requests.clear()
            logger.info("Performance metrics reset")


def performance_monitor_decorator(func_name: Optional[str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        actual_func_name = func_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    performance_monitor.track_function_performance(actual_func_name, duration, success)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    performance_monitor.track_function_performance(actual_func_name, duration, success)
            return sync_wrapper
    return decorator


# Global performance monitor instance
performance_monitor = PerformanceMonitor()