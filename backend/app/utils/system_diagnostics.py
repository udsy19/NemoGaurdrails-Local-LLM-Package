"""
System Diagnostics and Recovery Tools
Author: Udaya Vijay Anand
"""

import asyncio
import psutil
import time
import os
import gc
import sys
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import subprocess
import json

logger = logging.getLogger(__name__)


class DiagnosticLevel(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EMERGENCY = "emergency"


class RecoveryAction(Enum):
    RESTART_COMPONENT = "restart_component"
    CLEAR_CACHE = "clear_cache"
    RESTART_SERVICE = "restart_service"
    RESET_CONNECTIONS = "reset_connections"
    GARBAGE_COLLECT = "garbage_collect"
    RELOAD_CONFIG = "reload_config"
    RESET_METRICS = "reset_metrics"


@dataclass
class DiagnosticResult:
    component: str
    status: str
    details: Dict[str, Any]
    recommendations: List[str]
    severity: str
    timestamp: float


@dataclass
class RecoveryPlan:
    actions: List[RecoveryAction]
    estimated_downtime: float
    risk_level: str
    description: str
    prerequisites: List[str]


class SystemDiagnostics:
    """Comprehensive system diagnostics and recovery management"""
    
    def __init__(self):
        self.diagnostic_checkers: Dict[str, Callable] = {}
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self.diagnostic_history: List[DiagnosticResult] = []
        self.max_history = 100
        
        # Register default diagnostic checkers
        self._register_default_checkers()
        
        # Register default recovery handlers
        self._register_default_recovery_handlers()
    
    def _register_default_checkers(self):
        """Register default diagnostic checkers"""
        self.register_checker("memory", self._check_memory_usage)
        self.register_checker("cpu", self._check_cpu_usage)
        self.register_checker("disk", self._check_disk_usage)
        self.register_checker("network", self._check_network_connections)
        self.register_checker("processes", self._check_process_health)
        self.register_checker("file_descriptors", self._check_file_descriptors)
        self.register_checker("python_gc", self._check_python_gc)
        self.register_checker("async_tasks", self._check_async_tasks)
    
    def _register_default_recovery_handlers(self):
        """Register default recovery handlers"""
        self.register_recovery_handler(RecoveryAction.GARBAGE_COLLECT, self._garbage_collect)
        self.register_recovery_handler(RecoveryAction.CLEAR_CACHE, self._clear_caches)
        self.register_recovery_handler(RecoveryAction.RESET_CONNECTIONS, self._reset_connections)
        self.register_recovery_handler(RecoveryAction.RESET_METRICS, self._reset_metrics)
    
    def register_checker(self, name: str, checker_func: Callable):
        """Register a diagnostic checker"""
        self.diagnostic_checkers[name] = checker_func
        logger.info(f"Registered diagnostic checker: {name}")
    
    def register_recovery_handler(self, action: RecoveryAction, handler_func: Callable):
        """Register a recovery handler"""
        self.recovery_handlers[action] = handler_func
        logger.info(f"Registered recovery handler: {action.value}")
    
    async def run_diagnostics(self, level: DiagnosticLevel = DiagnosticLevel.DETAILED) -> List[DiagnosticResult]:
        """Run system diagnostics at specified level"""
        logger.info(f"Running system diagnostics at {level.value} level")
        start_time = time.time()
        
        results = []
        
        # Determine which checkers to run based on level
        checkers_to_run = self._get_checkers_for_level(level)
        
        # Run diagnostic checkers
        for checker_name in checkers_to_run:
            if checker_name in self.diagnostic_checkers:
                try:
                    result = await self._run_checker(checker_name)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Diagnostic checker {checker_name} failed: {e}")
                    results.append(DiagnosticResult(
                        component=checker_name,
                        status="error",
                        details={"error": str(e)},
                        recommendations=[f"Fix diagnostic checker: {checker_name}"],
                        severity="high",
                        timestamp=time.time()
                    ))
        
        # Store results in history
        for result in results:
            self._add_to_history(result)
        
        duration = time.time() - start_time
        logger.info(f"System diagnostics completed in {duration:.2f}s. Found {len(results)} results")
        
        return results
    
    def _get_checkers_for_level(self, level: DiagnosticLevel) -> List[str]:
        """Get checkers to run for diagnostic level"""
        basic_checkers = ["memory", "cpu", "disk"]
        detailed_checkers = basic_checkers + ["network", "processes", "file_descriptors"]
        comprehensive_checkers = detailed_checkers + ["python_gc", "async_tasks"]
        emergency_checkers = ["memory", "cpu", "processes"]  # Fast critical checks only
        
        if level == DiagnosticLevel.BASIC:
            return basic_checkers
        elif level == DiagnosticLevel.DETAILED:
            return detailed_checkers
        elif level == DiagnosticLevel.COMPREHENSIVE:
            return comprehensive_checkers
        elif level == DiagnosticLevel.EMERGENCY:
            return emergency_checkers
        else:
            return detailed_checkers
    
    async def _run_checker(self, checker_name: str) -> DiagnosticResult:
        """Run a single diagnostic checker"""
        checker_func = self.diagnostic_checkers[checker_name]
        
        try:
            if asyncio.iscoroutinefunction(checker_func):
                result = await checker_func()
            else:
                result = checker_func()
            
            return DiagnosticResult(
                component=checker_name,
                status=result.get("status", "unknown"),
                details=result.get("details", {}),
                recommendations=result.get("recommendations", []),
                severity=result.get("severity", "info"),
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Checker {checker_name} failed: {e}")
            raise
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system and process memory usage"""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Determine status and recommendations
        recommendations = []
        severity = "info"
        status = "healthy"
        
        if system_memory.percent > 90:
            status = "critical"
            severity = "critical"
            recommendations.append("System memory usage is critically high")
        elif system_memory.percent > 80:
            status = "warning"
            severity = "medium"
            recommendations.append("System memory usage is elevated")
        
        # Check for memory leaks (simplified heuristic)
        memory_mb = process_memory.rss / 1024 / 1024
        if memory_mb > 1000:  # More than 1GB
            recommendations.append("Process memory usage is high, consider garbage collection")
        
        return {
            "status": status,
            "severity": severity,
            "details": {
                "system_memory": {
                    "total_gb": system_memory.total / 1024 / 1024 / 1024,
                    "available_gb": system_memory.available / 1024 / 1024 / 1024,
                    "used_percent": system_memory.percent
                },
                "process_memory": {
                    "rss_mb": memory_mb,
                    "vms_mb": process_memory.vms / 1024 / 1024
                }
            },
            "recommendations": recommendations
        }
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage and load"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Get load average if available
        load_avg = None
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()
        
        recommendations = []
        severity = "info"
        status = "healthy"
        
        if cpu_percent > 90:
            status = "critical"
            severity = "critical"
            recommendations.append("CPU usage is critically high")
        elif cpu_percent > 80:
            status = "warning"
            severity = "medium"
            recommendations.append("CPU usage is elevated")
        
        return {
            "status": status,
            "severity": severity,
            "details": {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_average": load_avg
            },
            "recommendations": recommendations
        }
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        disk_usage = psutil.disk_usage('/')
        
        used_percent = (disk_usage.used / disk_usage.total) * 100
        
        recommendations = []
        severity = "info"
        status = "healthy"
        
        if used_percent > 95:
            status = "critical"
            severity = "critical"
            recommendations.append("Disk space is critically low")
        elif used_percent > 85:
            status = "warning"
            severity = "medium"
            recommendations.append("Disk space is getting low")
        
        return {
            "status": status,
            "severity": severity,
            "details": {
                "total_gb": disk_usage.total / 1024 / 1024 / 1024,
                "used_gb": disk_usage.used / 1024 / 1024 / 1024,
                "free_gb": disk_usage.free / 1024 / 1024 / 1024,
                "used_percent": used_percent
            },
            "recommendations": recommendations
        }
    
    def _check_network_connections(self) -> Dict[str, Any]:
        """Check network connections"""
        try:
            process = psutil.Process()
            connections = process.connections()
            
            connection_states = {}
            for conn in connections:
                state = conn.status
                connection_states[state] = connection_states.get(state, 0) + 1
            
            total_connections = len(connections)
            
            recommendations = []
            severity = "info"
            status = "healthy"
            
            if total_connections > 1000:
                status = "warning"
                severity = "medium"
                recommendations.append("High number of network connections")
            
            return {
                "status": status,
                "severity": severity,
                "details": {
                    "total_connections": total_connections,
                    "connection_states": connection_states
                },
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "status": "error",
                "severity": "low",
                "details": {"error": str(e)},
                "recommendations": ["Unable to check network connections"]
            }
    
    def _check_process_health(self) -> Dict[str, Any]:
        """Check process health metrics"""
        try:
            process = psutil.Process()
            
            details = {
                "pid": process.pid,
                "status": process.status(),
                "num_threads": process.num_threads(),
                "cpu_percent": process.cpu_percent(),
                "create_time": process.create_time(),
                "uptime_hours": (time.time() - process.create_time()) / 3600
            }
            
            recommendations = []
            severity = "info"
            status = "healthy"
            
            if details["num_threads"] > 100:
                recommendations.append("High number of threads")
                severity = "medium"
            
            return {
                "status": status,
                "severity": severity,
                "details": details,
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "status": "error",
                "severity": "medium",
                "details": {"error": str(e)},
                "recommendations": ["Unable to check process health"]
            }
    
    def _check_file_descriptors(self) -> Dict[str, Any]:
        """Check file descriptor usage"""
        try:
            process = psutil.Process()
            open_files = len(process.open_files())
            
            # Try to get file descriptor limits
            try:
                import resource
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            except:
                soft_limit = hard_limit = None
            
            recommendations = []
            severity = "info"
            status = "healthy"
            
            if soft_limit and open_files > soft_limit * 0.8:
                status = "warning"
                severity = "medium"
                recommendations.append("File descriptor usage is high")
            
            return {
                "status": status,
                "severity": severity,
                "details": {
                    "open_files": open_files,
                    "soft_limit": soft_limit,
                    "hard_limit": hard_limit
                },
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "status": "error",
                "severity": "low",
                "details": {"error": str(e)},
                "recommendations": ["Unable to check file descriptors"]
            }
    
    def _check_python_gc(self) -> Dict[str, Any]:
        """Check Python garbage collection status"""
        try:
            gc_stats = gc.get_stats()
            gc_counts = gc.get_count()
            gc_thresholds = gc.get_threshold()
            
            # Check for potential memory leaks
            generation_2_count = gc_counts[2] if len(gc_counts) > 2 else 0
            
            recommendations = []
            severity = "info"
            status = "healthy"
            
            if generation_2_count > 1000:
                recommendations.append("High generation 2 object count, consider manual garbage collection")
                severity = "low"
            
            return {
                "status": status,
                "severity": severity,
                "details": {
                    "gc_counts": gc_counts,
                    "gc_thresholds": gc_thresholds,
                    "gc_stats": gc_stats
                },
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "status": "error",
                "severity": "low",
                "details": {"error": str(e)},
                "recommendations": ["Unable to check garbage collection"]
            }
    
    async def _check_async_tasks(self) -> Dict[str, Any]:
        """Check asyncio task status"""
        try:
            loop = asyncio.get_running_loop()
            all_tasks = asyncio.all_tasks(loop)
            
            task_states = {}
            pending_tasks = 0
            
            for task in all_tasks:
                if task.done():
                    if task.cancelled():
                        state = "cancelled"
                    elif task.exception():
                        state = "failed"
                    else:
                        state = "completed"
                else:
                    state = "pending"
                    pending_tasks += 1
                
                task_states[state] = task_states.get(state, 0) + 1
            
            recommendations = []
            severity = "info"
            status = "healthy"
            
            if pending_tasks > 100:
                status = "warning"
                severity = "medium"
                recommendations.append("High number of pending async tasks")
            
            return {
                "status": status,
                "severity": severity,
                "details": {
                    "total_tasks": len(all_tasks),
                    "pending_tasks": pending_tasks,
                    "task_states": task_states
                },
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "status": "error",
                "severity": "low",
                "details": {"error": str(e)},
                "recommendations": ["Unable to check async tasks"]
            }
    
    def _garbage_collect(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform garbage collection"""
        logger.info("Performing garbage collection")
        
        before_counts = gc.get_count()
        collected = gc.collect()
        after_counts = gc.get_count()
        
        result = {
            "action": "garbage_collect",
            "success": True,
            "collected_objects": collected,
            "before_counts": before_counts,
            "after_counts": after_counts
        }
        
        logger.info(f"Garbage collection completed: {collected} objects collected")
        return result
    
    def _clear_caches(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Clear system caches"""
        logger.info("Clearing system caches")
        
        # This is a placeholder - specific cache clearing would be implemented
        # based on the actual caching mechanisms used
        
        return {
            "action": "clear_cache",
            "success": True,
            "message": "Cache clearing initiated"
        }
    
    def _reset_connections(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset network connections"""
        logger.info("Resetting network connections")
        
        # This is a placeholder - specific connection reset would be implemented
        # based on the actual connection pools used
        
        return {
            "action": "reset_connections",
            "success": True,
            "message": "Connection reset initiated"
        }
    
    def _reset_metrics(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset system metrics"""
        logger.info("Resetting system metrics")
        
        # Clear diagnostic history
        self.diagnostic_history.clear()
        
        return {
            "action": "reset_metrics",
            "success": True,
            "message": "Metrics reset completed"
        }
    
    async def execute_recovery_plan(self, plan: RecoveryPlan, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a recovery plan"""
        logger.warning(f"Executing recovery plan: {plan.description}")
        
        results = []
        context = context or {}
        
        for action in plan.actions:
            if action in self.recovery_handlers:
                try:
                    handler = self.recovery_handlers[action]
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(context)
                    else:
                        result = handler(context)
                    results.append(result)
                    logger.info(f"Recovery action {action.value} completed successfully")
                except Exception as e:
                    logger.error(f"Recovery action {action.value} failed: {e}")
                    results.append({
                        "action": action.value,
                        "success": False,
                        "error": str(e)
                    })
            else:
                logger.warning(f"No handler for recovery action: {action.value}")
                results.append({
                    "action": action.value,
                    "success": False,
                    "error": "No handler available"
                })
        
        success_count = sum(1 for r in results if r.get("success", False))
        
        return {
            "plan_description": plan.description,
            "total_actions": len(plan.actions),
            "successful_actions": success_count,
            "failed_actions": len(plan.actions) - success_count,
            "results": results,
            "overall_success": success_count == len(plan.actions)
        }
    
    def generate_recovery_plan(self, diagnostic_results: List[DiagnosticResult]) -> Optional[RecoveryPlan]:
        """Generate a recovery plan based on diagnostic results"""
        critical_issues = [r for r in diagnostic_results if r.severity == "critical"]
        medium_issues = [r for r in diagnostic_results if r.severity == "medium"]
        
        if not critical_issues and not medium_issues:
            return None
        
        actions = []
        description_parts = []
        estimated_downtime = 0.0
        risk_level = "low"
        
        # Handle critical issues
        for issue in critical_issues:
            if issue.component == "memory":
                actions.extend([RecoveryAction.GARBAGE_COLLECT, RecoveryAction.CLEAR_CACHE])
                description_parts.append("memory cleanup")
                estimated_downtime += 10.0
                risk_level = "medium"
            
            elif issue.component == "cpu":
                actions.append(RecoveryAction.RESET_CONNECTIONS)
                description_parts.append("reduce CPU load")
                estimated_downtime += 5.0
                risk_level = "medium"
            
            elif issue.component == "disk":
                actions.append(RecoveryAction.CLEAR_CACHE)
                description_parts.append("free disk space")
                estimated_downtime += 5.0
                risk_level = "low"
        
        # Handle medium issues
        for issue in medium_issues:
            if issue.component == "network":
                actions.append(RecoveryAction.RESET_CONNECTIONS)
                description_parts.append("reset network connections")
                estimated_downtime += 2.0
        
        if not actions:
            return None
        
        # Remove duplicates while preserving order
        unique_actions = []
        for action in actions:
            if action not in unique_actions:
                unique_actions.append(action)
        
        description = f"Recovery plan to address: {', '.join(description_parts)}"
        
        return RecoveryPlan(
            actions=unique_actions,
            estimated_downtime=estimated_downtime,
            risk_level=risk_level,
            description=description,
            prerequisites=["Ensure no critical operations are running"]
        )
    
    def _add_to_history(self, result: DiagnosticResult):
        """Add diagnostic result to history"""
        self.diagnostic_history.append(result)
        if len(self.diagnostic_history) > self.max_history:
            self.diagnostic_history.pop(0)
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get summary of recent diagnostics"""
        if not self.diagnostic_history:
            return {"message": "No diagnostic history available"}
        
        recent_results = self.diagnostic_history[-10:]  # Last 10 results
        
        severity_counts = {}
        status_counts = {}
        component_issues = {}
        
        for result in recent_results:
            severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            
            if result.status != "healthy":
                if result.component not in component_issues:
                    component_issues[result.component] = []
                component_issues[result.component].append({
                    "status": result.status,
                    "severity": result.severity,
                    "timestamp": result.timestamp
                })
        
        return {
            "total_checks": len(recent_results),
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "problematic_components": component_issues,
            "last_check": recent_results[-1].timestamp if recent_results else None
        }


# Global system diagnostics instance
system_diagnostics = SystemDiagnostics()