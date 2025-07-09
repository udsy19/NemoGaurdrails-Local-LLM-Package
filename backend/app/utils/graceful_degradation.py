"""
Graceful Degradation Manager for System Resilience
Author: Udaya Vijay Anand
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    FULL = "full"                    # All features available
    REDUCED = "reduced"              # Some features disabled
    MINIMAL = "minimal"              # Only core features
    EMERGENCY = "emergency"          # Critical functions only


@dataclass
class DegradationRule:
    trigger_condition: str           # e.g., "ollama_down", "memory_high"
    degradation_level: DegradationLevel
    disabled_features: List[str]
    fallback_strategy: Optional[str] = None
    priority: int = 1               # Higher number = higher priority


@dataclass
class SystemState:
    degradation_level: DegradationLevel
    active_rules: List[DegradationRule]
    disabled_features: Set[str]
    fallback_strategies: Dict[str, str]
    last_updated: float
    reason: str


class GracefulDegradationManager:
    """Manages system degradation strategies for resilience"""
    
    def __init__(self):
        self.current_state = SystemState(
            degradation_level=DegradationLevel.FULL,
            active_rules=[],
            disabled_features=set(),
            fallback_strategies={},
            last_updated=time.time(),
            reason="System initialized"
        )
        
        self.degradation_rules: Dict[str, DegradationRule] = {}
        self.feature_registry: Dict[str, Dict[str, Any]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.state_change_callbacks: List[Callable] = []
        
        # Initialize default degradation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default degradation rules"""
        
        # Ollama service down
        self.register_rule(
            "ollama_down",
            DegradationRule(
                trigger_condition="ollama_down",
                degradation_level=DegradationLevel.REDUCED,
                disabled_features=["ai_generation", "chat_completion"],
                fallback_strategy="cached_responses",
                priority=3
            )
        )
        
        # High memory usage
        self.register_rule(
            "memory_high",
            DegradationRule(
                trigger_condition="memory_high",
                degradation_level=DegradationLevel.REDUCED,
                disabled_features=["advanced_analytics", "model_caching"],
                fallback_strategy="reduce_memory_usage",
                priority=2
            )
        )
        
        # Multiple detector failures
        self.register_rule(
            "detectors_failing",
            DegradationRule(
                trigger_condition="detectors_failing",
                degradation_level=DegradationLevel.MINIMAL,
                disabled_features=["advanced_detection", "custom_rules"],
                fallback_strategy="basic_safety_only",
                priority=2
            )
        )
        
        # System overload
        self.register_rule(
            "system_overload",
            DegradationRule(
                trigger_condition="system_overload",
                degradation_level=DegradationLevel.EMERGENCY,
                disabled_features=["non_essential_features", "background_tasks", "detailed_logging"],
                fallback_strategy="emergency_mode",
                priority=4
            )
        )
        
        # Guardrails failure
        self.register_rule(
            "guardrails_down",
            DegradationRule(
                trigger_condition="guardrails_down",
                degradation_level=DegradationLevel.MINIMAL,
                disabled_features=["advanced_guardrails"],
                fallback_strategy="basic_detection_only",
                priority=3
            )
        )
    
    def register_rule(self, name: str, rule: DegradationRule):
        """Register a degradation rule"""
        self.degradation_rules[name] = rule
        logger.info(f"Registered degradation rule: {name}")
    
    def register_feature(self, feature_name: str, metadata: Dict[str, Any]):
        """Register a feature that can be degraded"""
        self.feature_registry[feature_name] = {
            "metadata": metadata,
            "essential": metadata.get("essential", False),
            "fallback_available": metadata.get("fallback_available", False),
            "resource_intensive": metadata.get("resource_intensive", False)
        }
        logger.info(f"Registered feature: {feature_name}")
    
    def register_fallback_handler(self, strategy_name: str, handler: Callable):
        """Register a fallback handler for a degradation strategy"""
        self.fallback_handlers[strategy_name] = handler
        logger.info(f"Registered fallback handler: {strategy_name}")
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback to be called when system state changes"""
        self.state_change_callbacks.append(callback)
    
    async def evaluate_degradation(self, trigger_conditions: List[str], context: Dict[str, Any] = None) -> SystemState:
        """Evaluate and apply degradation based on trigger conditions"""
        context = context or {}
        
        logger.info(f"Evaluating degradation for conditions: {trigger_conditions}")
        
        # Find applicable rules
        applicable_rules = []
        for condition in trigger_conditions:
            if condition in self.degradation_rules:
                applicable_rules.append(self.degradation_rules[condition])
        
        if not applicable_rules:
            # No degradation needed
            if self.current_state.degradation_level != DegradationLevel.FULL:
                await self._restore_full_operation("No trigger conditions active")
            return self.current_state
        
        # Sort rules by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Determine the most severe degradation level needed
        highest_degradation = max(applicable_rules, key=lambda r: self._get_degradation_severity(r.degradation_level))
        
        # Collect all disabled features and fallback strategies
        disabled_features = set()
        fallback_strategies = {}
        
        for rule in applicable_rules:
            disabled_features.update(rule.disabled_features)
            if rule.fallback_strategy:
                fallback_strategies[rule.trigger_condition] = rule.fallback_strategy
        
        # Create new system state
        new_state = SystemState(
            degradation_level=highest_degradation.degradation_level,
            active_rules=applicable_rules,
            disabled_features=disabled_features,
            fallback_strategies=fallback_strategies,
            last_updated=time.time(),
            reason=f"Triggered by: {', '.join(trigger_conditions)}"
        )
        
        # Apply degradation if state changed
        if self._state_changed(new_state):
            await self._apply_degradation(new_state, context)
        
        return self.current_state
    
    async def _apply_degradation(self, new_state: SystemState, context: Dict[str, Any]):
        """Apply the degradation state"""
        old_state = self.current_state
        self.current_state = new_state
        
        logger.warning(f"Applying system degradation: {old_state.degradation_level.value} -> {new_state.degradation_level.value}")
        logger.warning(f"Reason: {new_state.reason}")
        
        # Disable features
        for feature in new_state.disabled_features:
            logger.warning(f"Disabling feature: {feature}")
        
        # Activate fallback strategies
        for condition, strategy in new_state.fallback_strategies.items():
            await self._activate_fallback_strategy(strategy, context)
        
        # Notify callbacks
        await self._notify_state_change_callbacks(old_state, new_state)
        
        logger.info(f"System degradation applied successfully")
    
    async def _restore_full_operation(self, reason: str):
        """Restore system to full operation"""
        if self.current_state.degradation_level == DegradationLevel.FULL:
            return
        
        old_state = self.current_state
        self.current_state = SystemState(
            degradation_level=DegradationLevel.FULL,
            active_rules=[],
            disabled_features=set(),
            fallback_strategies={},
            last_updated=time.time(),
            reason=reason
        )
        
        logger.info(f"Restoring system to full operation: {reason}")
        
        # Re-enable all features
        for feature in old_state.disabled_features:
            logger.info(f"Re-enabling feature: {feature}")
        
        # Deactivate fallback strategies
        for strategy in old_state.fallback_strategies.values():
            await self._deactivate_fallback_strategy(strategy)
        
        # Notify callbacks
        await self._notify_state_change_callbacks(old_state, self.current_state)
    
    async def _activate_fallback_strategy(self, strategy: str, context: Dict[str, Any]):
        """Activate a fallback strategy"""
        logger.info(f"Activating fallback strategy: {strategy}")
        
        if strategy in self.fallback_handlers:
            try:
                handler = self.fallback_handlers[strategy]
                if asyncio.iscoroutinefunction(handler):
                    await handler(context)
                else:
                    handler(context)
                logger.info(f"Fallback strategy {strategy} activated successfully")
            except Exception as e:
                logger.error(f"Failed to activate fallback strategy {strategy}: {e}")
        else:
            logger.warning(f"No handler registered for fallback strategy: {strategy}")
    
    async def _deactivate_fallback_strategy(self, strategy: str):
        """Deactivate a fallback strategy"""
        logger.info(f"Deactivating fallback strategy: {strategy}")
        # Implementation depends on specific strategy
        # This is a placeholder for strategy-specific cleanup
    
    async def _notify_state_change_callbacks(self, old_state: SystemState, new_state: SystemState):
        """Notify all registered callbacks about state change"""
        for callback in self.state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state)
                else:
                    callback(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _state_changed(self, new_state: SystemState) -> bool:
        """Check if the system state has changed"""
        return (
            self.current_state.degradation_level != new_state.degradation_level or
            self.current_state.disabled_features != new_state.disabled_features or
            self.current_state.fallback_strategies != new_state.fallback_strategies
        )
    
    def _get_degradation_severity(self, level: DegradationLevel) -> int:
        """Get numeric severity for degradation level"""
        severity_map = {
            DegradationLevel.FULL: 0,
            DegradationLevel.REDUCED: 1,
            DegradationLevel.MINIMAL: 2,
            DegradationLevel.EMERGENCY: 3
        }
        return severity_map.get(level, 0)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is currently enabled"""
        return feature_name not in self.current_state.disabled_features
    
    def get_current_state(self) -> SystemState:
        """Get current system state"""
        return self.current_state
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities"""
        enabled_features = []
        disabled_features = []
        
        for feature_name, feature_info in self.feature_registry.items():
            if self.is_feature_enabled(feature_name):
                enabled_features.append({
                    "name": feature_name,
                    "metadata": feature_info["metadata"]
                })
            else:
                disabled_features.append({
                    "name": feature_name,
                    "metadata": feature_info["metadata"],
                    "reason": "System degradation"
                })
        
        return {
            "degradation_level": self.current_state.degradation_level.value,
            "enabled_features": enabled_features,
            "disabled_features": disabled_features,
            "active_fallback_strategies": list(self.current_state.fallback_strategies.values()),
            "last_updated": self.current_state.last_updated,
            "reason": self.current_state.reason
        }
    
    async def force_degradation_level(self, level: DegradationLevel, reason: str):
        """Force system to a specific degradation level"""
        logger.warning(f"Force degradation to {level.value}: {reason}")
        
        # Find all rules that match or exceed this degradation level
        applicable_rules = []
        for rule in self.degradation_rules.values():
            if self._get_degradation_severity(rule.degradation_level) >= self._get_degradation_severity(level):
                applicable_rules.append(rule)
        
        # Create forced degradation state
        disabled_features = set()
        fallback_strategies = {}
        
        for rule in applicable_rules:
            disabled_features.update(rule.disabled_features)
            if rule.fallback_strategy:
                fallback_strategies[rule.trigger_condition] = rule.fallback_strategy
        
        forced_state = SystemState(
            degradation_level=level,
            active_rules=applicable_rules,
            disabled_features=disabled_features,
            fallback_strategies=fallback_strategies,
            last_updated=time.time(),
            reason=f"Forced degradation: {reason}"
        )
        
        await self._apply_degradation(forced_state, {})
    
    def get_degradation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get degradation history (placeholder for future implementation)"""
        # This would typically be stored in a database or log file
        return [{
            "timestamp": self.current_state.last_updated,
            "degradation_level": self.current_state.degradation_level.value,
            "reason": self.current_state.reason,
            "disabled_features": list(self.current_state.disabled_features)
        }]


# Global degradation manager instance
graceful_degradation_manager = GracefulDegradationManager()