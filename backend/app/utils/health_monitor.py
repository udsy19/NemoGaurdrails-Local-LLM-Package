"""
Comprehensive System Health Monitoring
Author: Udaya Vijay Anand
"""

import asyncio
import psutil
import time
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    last_check: Optional[float] = None


@dataclass
class SystemHealth:
    overall_status: HealthStatus
    components: List[ComponentHealth]
    system_metrics: Dict[str, Any]
    timestamp: float
    correlation_id: str


class HealthChecker:
    """Individual health checker for a component"""
    
    def __init__(self, name: str, check_func, timeout: float = 5.0):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.last_result: Optional[ComponentHealth] = None
    
    async def check(self) -> ComponentHealth:
        """Perform health check with timeout protection"""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                health = ComponentHealth(
                    name=self.name,
                    status=status,
                    response_time=response_time,
                    last_check=time.time()
                )
            elif isinstance(result, dict):
                health = ComponentHealth(
                    name=self.name,
                    status=result.get("status", HealthStatus.HEALTHY),
                    response_time=response_time,
                    error_message=result.get("error"),
                    metrics=result.get("metrics"),
                    last_check=time.time()
                )
            else:
                health = ComponentHealth(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    last_check=time.time()
                )
            
            self.last_result = health
            return health
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name=self.name,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                error_message=f"Health check timeout after {self.timeout}s",
                last_check=time.time()
            )
            self.last_result = health
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                error_message=str(e),
                last_check=time.time()
            )
            self.last_result = health
            return health


class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self):
        self.checkers: Dict[str, HealthChecker] = {}
        self.last_system_health: Optional[SystemHealth] = None
        self.health_history: List[SystemHealth] = []
        self.max_history = 100
        
    def register_checker(self, name: str, check_func, timeout: float = 5.0) -> HealthChecker:
        """Register a health checker for a component"""
        checker = HealthChecker(name, check_func, timeout)
        self.checkers[name] = checker
        logger.info(f"Registered health checker for {name}")
        return checker
    
    async def check_all_components(self, correlation_id: str = None) -> SystemHealth:
        """Check health of all registered components"""
        correlation_id = correlation_id or f"health_{int(time.time())}"
        
        logger.info(f"Starting system health check [correlation_id: {correlation_id}]")
        start_time = time.time()
        
        # Run all health checks concurrently
        tasks = []
        for checker in self.checkers.values():
            task = asyncio.create_task(checker.check())
            tasks.append(task)
        
        # Wait for all checks to complete
        component_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        components = []
        for i, result in enumerate(component_results):
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                checker_name = list(self.checkers.keys())[i]
                components.append(ComponentHealth(
                    name=checker_name,
                    status=HealthStatus.CRITICAL,
                    error_message=f"Health check failed: {result}",
                    last_check=time.time()
                ))
            else:
                components.append(result)
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Determine overall status
        overall_status = self._determine_overall_status(components, system_metrics)
        
        # Create system health report
        system_health = SystemHealth(
            overall_status=overall_status,
            components=components,
            system_metrics=system_metrics,
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        
        # Store results
        self.last_system_health = system_health
        self._add_to_history(system_health)
        
        check_duration = time.time() - start_time
        logger.info(f"System health check completed in {check_duration:.2f}s - Status: {overall_status.value} [correlation_id: {correlation_id}]")
        
        return system_health
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_metrics = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
            
            # CPU metrics
            cpu_metrics = {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_metrics = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
            
            # Process metrics
            process = psutil.Process()
            process_metrics = {
                "memory_rss": process.memory_info().rss,
                "memory_vms": process.memory_info().vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
            
            return {
                "memory": memory_metrics,
                "cpu": cpu_metrics,
                "disk": disk_metrics,
                "process": process_metrics,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _determine_overall_status(self, components: List[ComponentHealth], system_metrics: Dict[str, Any]) -> HealthStatus:
        """Determine overall system health status"""
        # Check component health
        critical_count = sum(1 for c in components if c.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        
        # Check system resource health
        resource_issues = []
        
        if "memory" in system_metrics:
            memory_percent = system_metrics["memory"].get("percent", 0)
            if memory_percent > 90:
                resource_issues.append("high_memory")
            elif memory_percent > 80:
                resource_issues.append("elevated_memory")
        
        if "cpu" in system_metrics:
            cpu_percent = system_metrics["cpu"].get("percent", 0)
            if cpu_percent > 90:
                resource_issues.append("high_cpu")
            elif cpu_percent > 80:
                resource_issues.append("elevated_cpu")
        
        if "disk" in system_metrics:
            disk_percent = system_metrics["disk"].get("percent", 0)
            if disk_percent > 95:
                resource_issues.append("disk_full")
            elif disk_percent > 85:
                resource_issues.append("disk_low")
        
        # Determine status based on component and resource health
        if critical_count > 0 or "high_memory" in resource_issues or "disk_full" in resource_issues:
            return HealthStatus.CRITICAL
        elif unhealthy_count > 0 or "high_cpu" in resource_issues:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0 or len(resource_issues) > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _add_to_history(self, health: SystemHealth):
        """Add health check to history"""
        self.health_history.append(health)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health"""
        if not self.last_system_health:
            return {"status": "unknown", "message": "No health checks performed yet"}
        
        return {
            "status": self.last_system_health.overall_status.value,
            "timestamp": self.last_system_health.timestamp,
            "components": {
                comp.name: {
                    "status": comp.status.value,
                    "response_time": comp.response_time,
                    "error": comp.error_message
                }
                for comp in self.last_system_health.components
            },
            "system_metrics": self.last_system_health.system_metrics,
            "correlation_id": self.last_system_health.correlation_id
        }
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component"""
        if component_name in self.checkers:
            return self.checkers[component_name].last_result
        return None
    
    def get_health_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get health trends over specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        recent_checks = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        if not recent_checks:
            return {"error": "No recent health checks available"}
        
        # Calculate trends
        status_counts = {}
        avg_response_times = {}
        
        for health in recent_checks:
            status = health.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            for component in health.components:
                comp_name = component.name
                if comp_name not in avg_response_times:
                    avg_response_times[comp_name] = []
                if component.response_time:
                    avg_response_times[comp_name].append(component.response_time)
        
        # Calculate averages
        for comp_name in avg_response_times:
            times = avg_response_times[comp_name]
            avg_response_times[comp_name] = sum(times) / len(times) if times else 0
        
        return {
            "period_hours": hours,
            "total_checks": len(recent_checks),
            "status_distribution": status_counts,
            "average_response_times": avg_response_times,
            "latest_status": recent_checks[-1].overall_status.value if recent_checks else "unknown"
        }
    
    def to_dict(self, health: SystemHealth) -> Dict[str, Any]:
        """Convert SystemHealth to dictionary"""
        return {
            "overall_status": health.overall_status.value,
            "components": [asdict(comp) for comp in health.components],
            "system_metrics": health.system_metrics,
            "timestamp": health.timestamp,
            "correlation_id": health.correlation_id
        }


# Global health monitor instance
system_health_monitor = SystemHealthMonitor()