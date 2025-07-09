"""
Self-Healing and Optimization System
Author: Udaya Vijay Anand
"""

import asyncio
import time
import gc
import os
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import psutil

logger = logging.getLogger(__name__)


class HealingAction(Enum):
    RESTART_COMPONENT = "restart_component"
    CLEAR_MEMORY = "clear_memory"
    RELOAD_CONFIG = "reload_config"
    SCALE_RESOURCES = "scale_resources"
    RESET_CONNECTIONS = "reset_connections"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    THROTTLE_REQUESTS = "throttle_requests"
    CLEANUP_TEMP_FILES = "cleanup_temp_files"


class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    AGGRESSIVE = "aggressive"


@dataclass
class HealingRule:
    name: str
    trigger_condition: str
    healing_actions: List[HealingAction]
    cooldown_minutes: int = 30
    enabled: bool = True
    priority: int = 1  # 1=highest, 5=lowest
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingEvent:
    id: str
    rule_name: str
    triggered_at: float
    actions_taken: List[HealingAction]
    success: bool
    error_message: Optional[str] = None
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "triggered_at": self.triggered_at,
            "actions_taken": [action.value for action in self.actions_taken],
            "success": self.success,
            "error_message": self.error_message,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after
        }


class ResourceOptimizer:
    """Optimize system resources dynamically"""
    
    def __init__(self):
        self.optimization_level = OptimizationLevel.BALANCED
        self.last_optimization = 0
        self.optimization_history = []
        
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        logger.info("Starting memory optimization")
        
        before_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear internal caches (if any)
        # In a real implementation, you'd clear application-specific caches
        
        after_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = before_memory - after_memory
        
        result = {
            "memory_before_mb": before_memory,
            "memory_after_mb": after_memory,
            "memory_freed_mb": memory_freed,
            "objects_collected": collected,
            "optimization_level": self.optimization_level.value
        }
        
        logger.info(f"Memory optimization completed: freed {memory_freed:.2f}MB")
        return result
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance"""
        logger.info("Starting performance optimization")
        
        optimizations = []
        
        # CPU optimization
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            # Reduce background tasks
            optimizations.append("reduced_background_tasks")
        
        # Memory optimization
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            result = await self.optimize_memory()
            optimizations.append("memory_cleanup")
        
        # Connection optimization
        try:
            process = psutil.Process()
            connections = len(process.connections())
            if connections > 100:
                # In real implementation, close idle connections
                optimizations.append("connection_cleanup")
        except:
            pass
        
        result = {
            "optimizations_applied": optimizations,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "timestamp": time.time()
        }
        
        logger.info(f"Performance optimization completed: {optimizations}")
        return result
    
    async def cleanup_temporary_files(self) -> Dict[str, Any]:
        """Clean up temporary files"""
        logger.info("Starting temporary file cleanup")
        
        cleaned_files = 0
        cleaned_size = 0
        
        # In a real implementation, you'd scan and clean temp directories
        # This is a placeholder
        
        result = {
            "files_cleaned": cleaned_files,
            "size_freed_mb": cleaned_size / 1024 / 1024,
            "timestamp": time.time()
        }
        
        logger.info(f"Temporary file cleanup completed: {cleaned_files} files")
        return result


class SelfHealingSystem:
    """Advanced self-healing and optimization system"""
    
    def __init__(self):
        self.rules: Dict[str, HealingRule] = {}
        self.healing_history: List[HealingEvent] = []
        self.last_healing_times: Dict[str, float] = {}
        self.healing_handlers: Dict[HealingAction, Callable] = {}
        self.resource_optimizer = ResourceOptimizer()
        self.healing_active = False
        self.healing_task = None
        self.check_interval = 60  # seconds
        self.max_history = 1000
        
        # Register default healing handlers
        self._register_default_handlers()
        
        # Register default healing rules
        self._register_default_rules()
    
    def _register_default_handlers(self):
        """Register default healing action handlers"""
        self.healing_handlers[HealingAction.CLEAR_MEMORY] = self._handle_clear_memory
        self.healing_handlers[HealingAction.OPTIMIZE_PERFORMANCE] = self._handle_optimize_performance
        self.healing_handlers[HealingAction.CLEANUP_TEMP_FILES] = self._handle_cleanup_temp_files
        self.healing_handlers[HealingAction.RESET_CONNECTIONS] = self._handle_reset_connections
        self.healing_handlers[HealingAction.THROTTLE_REQUESTS] = self._handle_throttle_requests
    
    def _register_default_rules(self):
        """Register default self-healing rules"""
        
        # Memory management rules
        self.register_rule(HealingRule(
            name="high_memory_auto_cleanup",
            trigger_condition="memory_usage > 85",
            healing_actions=[HealingAction.CLEAR_MEMORY, HealingAction.OPTIMIZE_PERFORMANCE],
            cooldown_minutes=15,
            priority=1,
            description="Automatic memory cleanup when usage is high"
        ))
        
        # Performance optimization rules
        self.register_rule(HealingRule(
            name="cpu_overload_optimization",
            trigger_condition="cpu_usage > 90",
            healing_actions=[HealingAction.OPTIMIZE_PERFORMANCE, HealingAction.THROTTLE_REQUESTS],
            cooldown_minutes=10,
            priority=1,
            description="Optimize performance when CPU is overloaded"
        ))
        
        # Disk space management
        self.register_rule(HealingRule(
            name="low_disk_space_cleanup",
            trigger_condition="disk_usage > 90",
            healing_actions=[HealingAction.CLEANUP_TEMP_FILES],
            cooldown_minutes=60,
            priority=2,
            description="Clean up temporary files when disk space is low"
        ))
        
        # Connection management
        self.register_rule(HealingRule(
            name="too_many_connections",
            trigger_condition="active_connections > 500",
            healing_actions=[HealingAction.RESET_CONNECTIONS],
            cooldown_minutes=20,
            priority=2,
            description="Reset connections when count is too high"
        ))
        
        # Error rate optimization
        self.register_rule(HealingRule(
            name="high_error_rate_optimization",
            trigger_condition="error_rate > 0.1",
            healing_actions=[HealingAction.OPTIMIZE_PERFORMANCE],
            cooldown_minutes=30,
            priority=3,
            description="Optimize performance when error rate is high"
        ))
    
    def register_rule(self, rule: HealingRule):
        """Register a self-healing rule"""
        self.rules[rule.name] = rule
        logger.info(f"Registered self-healing rule: {rule.name}")
    
    def register_handler(self, action: HealingAction, handler: Callable):
        """Register a custom healing action handler"""
        self.healing_handlers[action] = handler
        logger.info(f"Registered healing handler for action: {action.value}")
    
    async def start_self_healing(self):
        """Start the self-healing system"""
        if not self.healing_active:
            self.healing_active = True
            self.healing_task = asyncio.create_task(self._healing_loop())
            logger.info("Self-healing system started")
    
    async def stop_self_healing(self):
        """Stop the self-healing system"""
        if self.healing_active:
            self.healing_active = False
            if self.healing_task:
                self.healing_task.cancel()
                try:
                    await self.healing_task
                except asyncio.CancelledError:
                    pass
            logger.info("Self-healing system stopped")
    
    async def _healing_loop(self):
        """Main self-healing loop"""
        while self.healing_active:
            try:
                await self._check_and_heal()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in self-healing loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_and_heal(self):
        """Check system status and apply healing if needed"""
        try:
            # Collect current metrics
            metrics = await self._collect_metrics()
            
            # Check each rule
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    should_heal = self._evaluate_condition(rule.trigger_condition, metrics)
                    
                    if should_heal:
                        await self._apply_healing(rule, metrics)
                        
                except Exception as e:
                    logger.error(f"Error evaluating healing rule {rule_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in check and heal process: {e}")
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Connection count
        try:
            connection_count = len(process.connections())
        except:
            connection_count = 0
        
        return {
            "memory_usage": memory.percent,
            "cpu_usage": cpu_percent,
            "disk_usage": (disk.used / disk.total) * 100,
            "process_memory_mb": process_memory.rss / 1024 / 1024,
            "active_connections": connection_count,
            "error_rate": 0.0,  # Would be populated from actual metrics
            "timestamp": time.time()
        }
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate healing condition"""
        try:
            # Replace metric names with actual values
            eval_condition = condition
            for key, value in metrics.items():
                eval_condition = eval_condition.replace(key, str(value))
            
            # Evaluate the condition
            result = eval(eval_condition)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _apply_healing(self, rule: HealingRule, metrics: Dict[str, Any]):
        """Apply healing actions for a rule"""
        current_time = time.time()
        
        # Check cooldown
        if rule.name in self.last_healing_times:
            time_since_last = current_time - self.last_healing_times[rule.name]
            if time_since_last < rule.cooldown_minutes * 60:
                return
        
        logger.warning(f"Applying self-healing rule: {rule.name}")
        
        # Create healing event
        event_id = f"{rule.name}_{int(current_time)}"
        event = HealingEvent(
            id=event_id,
            rule_name=rule.name,
            triggered_at=current_time,
            actions_taken=rule.healing_actions,
            success=False,
            metrics_before=metrics.copy()
        )
        
        try:
            # Apply healing actions
            for action in rule.healing_actions:
                if action in self.healing_handlers:
                    handler = self.healing_handlers[action]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(metrics)
                    else:
                        handler(metrics)
                    logger.info(f"Applied healing action: {action.value}")
                else:
                    logger.warning(f"No handler for healing action: {action.value}")
            
            # Collect metrics after healing
            post_healing_metrics = await self._collect_metrics()
            event.metrics_after = post_healing_metrics
            event.success = True
            
            # Update last healing time
            self.last_healing_times[rule.name] = current_time
            
            logger.info(f"Self-healing completed successfully for rule: {rule.name}")
            
        except Exception as e:
            event.error_message = str(e)
            logger.error(f"Self-healing failed for rule {rule.name}: {e}")
        
        # Store healing event
        self.healing_history.append(event)
        if len(self.healing_history) > self.max_history:
            self.healing_history.pop(0)
    
    async def _handle_clear_memory(self, metrics: Dict[str, Any]):
        """Handle memory clearing action"""
        result = await self.resource_optimizer.optimize_memory()
        logger.info(f"Memory clearing completed: {result}")
    
    async def _handle_optimize_performance(self, metrics: Dict[str, Any]):
        """Handle performance optimization action"""
        result = await self.resource_optimizer.optimize_performance()
        logger.info(f"Performance optimization completed: {result}")
    
    async def _handle_cleanup_temp_files(self, metrics: Dict[str, Any]):
        """Handle temporary file cleanup action"""
        result = await self.resource_optimizer.cleanup_temporary_files()
        logger.info(f"Temporary file cleanup completed: {result}")
    
    async def _handle_reset_connections(self, metrics: Dict[str, Any]):
        """Handle connection reset action"""
        # In a real implementation, this would reset connection pools
        logger.info("Connection reset action triggered")
    
    async def _handle_throttle_requests(self, metrics: Dict[str, Any]):
        """Handle request throttling action"""
        # In a real implementation, this would enable request throttling
        logger.info("Request throttling action triggered")
    
    def get_healing_stats(self) -> Dict[str, Any]:
        """Get self-healing statistics"""
        total_events = len(self.healing_history)
        successful_events = len([e for e in self.healing_history if e.success])
        
        # Count by rule
        rule_counts = {}
        for event in self.healing_history:
            rule_name = event.rule_name
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        # Count by action
        action_counts = {}
        for event in self.healing_history:
            for action in event.actions_taken:
                action_name = action.value
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        return {
            "healing_active": self.healing_active,
            "total_healing_events": total_events,
            "successful_healing_events": successful_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "rule_distribution": rule_counts,
            "action_distribution": action_counts,
            "registered_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled])
        }
    
    def get_recent_healing_events(self, hours: int = 24) -> List[HealingEvent]:
        """Get recent healing events"""
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.healing_history if event.triggered_at >= cutoff_time]
    
    def enable_rule(self, rule_name: str):
        """Enable a healing rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled healing rule: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable a healing rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled healing rule: {rule_name}")
    
    def get_rule_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all healing rules"""
        return {
            rule_name: {
                "enabled": rule.enabled,
                "priority": rule.priority,
                "actions": [action.value for action in rule.healing_actions],
                "cooldown_minutes": rule.cooldown_minutes,
                "description": rule.description,
                "last_triggered": self.last_healing_times.get(rule_name)
            }
            for rule_name, rule in self.rules.items()
        }


# Global self-healing system instance
self_healing_system = SelfHealingSystem()