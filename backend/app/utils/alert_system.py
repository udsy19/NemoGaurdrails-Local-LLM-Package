"""
Advanced Alert System with Multi-Channel Notifications
Author: Udaya Vijay Anand
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class AlertRule:
    name: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    description: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata
        }


class AlertManager:
    """Advanced alert management system with multiple channels and rules"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, float] = {}
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.max_history = 1000
        
        # Register default handlers
        self._register_default_handlers()
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_handlers(self):
        """Register default alert handlers"""
        self.alert_handlers[AlertChannel.LOG] = self._handle_log_alert
        self.alert_handlers[AlertChannel.CONSOLE] = self._handle_console_alert
        self.alert_handlers[AlertChannel.EMAIL] = self._handle_email_alert
        self.alert_handlers[AlertChannel.WEBHOOK] = self._handle_webhook_alert
    
    def _register_default_rules(self):
        """Register default alert rules"""
        
        # System resource alerts
        self.register_rule(AlertRule(
            name="high_memory_usage",
            condition="memory_usage > 90",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=10,
            description="System memory usage is critically high"
        ))
        
        self.register_rule(AlertRule(
            name="high_cpu_usage",
            condition="cpu_usage > 85",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG],
            cooldown_minutes=15,
            description="CPU usage is elevated"
        ))
        
        self.register_rule(AlertRule(
            name="low_disk_space",
            condition="disk_usage > 90",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=30,
            description="Disk space is critically low"
        ))
        
        # Circuit breaker alerts
        self.register_rule(AlertRule(
            name="circuit_breaker_open",
            condition="circuit_breaker_open == true",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=5,
            description="Circuit breaker is open"
        ))
        
        # Health check alerts
        self.register_rule(AlertRule(
            name="health_check_failed",
            condition="health_status == 'critical'",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=5,
            description="Health check failed"
        ))
        
        # Model loading alerts
        self.register_rule(AlertRule(
            name="model_loading_failed",
            condition="model_load_failures > 3",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=10,
            description="Multiple model loading failures"
        ))
    
    def register_rule(self, rule: AlertRule):
        """Register an alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """Register a custom alert handler"""
        self.alert_handlers[channel] = handler
        logger.info(f"Registered alert handler for channel: {channel.value}")
    
    async def evaluate_conditions(self, metrics: Dict[str, Any]):
        """Evaluate alert conditions against current metrics"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                should_alert = self._evaluate_condition(rule.condition, metrics)
                
                if should_alert:
                    await self._trigger_alert(rule, metrics)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate a single alert condition"""
        # Simple condition evaluation - in production, use a proper expression evaluator
        try:
            # Replace metric names with actual values
            eval_condition = condition
            for key, value in metrics.items():
                eval_condition = eval_condition.replace(key, str(value))
            
            # Handle boolean conditions
            eval_condition = eval_condition.replace("== true", "== True")
            eval_condition = eval_condition.replace("== false", "== False")
            
            # Evaluate the condition
            result = eval(eval_condition)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert if conditions are met"""
        current_time = time.time()
        
        # Check cooldown
        if rule.name in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[rule.name]
            if time_since_last < rule.cooldown_minutes * 60:
                return
        
        # Create alert
        alert_id = f"{rule.name}_{int(current_time)}"
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.description,
            timestamp=current_time,
            metadata={"metrics": metrics, "condition": rule.condition}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = current_time
        
        # Maintain history limit
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Send alert through channels
        for channel in rule.channels:
            if channel in self.alert_handlers:
                try:
                    await self._send_alert(channel, alert)
                except Exception as e:
                    logger.error(f"Error sending alert through {channel.value}: {e}")
        
        logger.warning(f"Alert triggered: {alert.rule_name} - {alert.message}")
    
    async def _send_alert(self, channel: AlertChannel, alert: Alert):
        """Send alert through specified channel"""
        handler = self.alert_handlers[channel]
        
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)
    
    async def _handle_log_alert(self, alert: Alert):
        """Handle log alert"""
        log_level = {
            AlertSeverity.LOW: logger.info,
            AlertSeverity.MEDIUM: logger.warning,
            AlertSeverity.HIGH: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.info)
        
        log_level(f"ALERT [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")
    
    async def _handle_console_alert(self, alert: Alert):
        """Handle console alert"""
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        severity_symbol = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }.get(alert.severity, "ℹ️")
        
        print(f"\n{severity_symbol} ALERT [{alert.severity.value.upper()}] {timestamp}")
        print(f"Rule: {alert.rule_name}")
        print(f"Message: {alert.message}")
        print(f"Alert ID: {alert.id}")
        print("-" * 50)
    
    async def _handle_email_alert(self, alert: Alert):
        """Handle email alert (placeholder)"""
        # In production, configure SMTP settings
        logger.info(f"Email alert would be sent: {alert.message}")
    
    async def _handle_webhook_alert(self, alert: Alert):
        """Handle webhook alert (placeholder)"""
        # In production, configure webhook endpoints
        logger.info(f"Webhook alert would be sent: {alert.message}")
    
    async def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_at = time.time()
            if resolution_note:
                alert.metadata["resolution_note"] = resolution_note
            
            logger.info(f"Alert resolved: {alert.rule_name} - {alert_id}")
            return alert
        return None
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_count = len(self.active_alerts)
        total_alerts = len(self.alert_history)
        
        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by rule
        rule_counts = {}
        for alert in self.alert_history:
            rule_name = alert.rule_name
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        return {
            "active_alerts": active_count,
            "total_alerts": total_alerts,
            "severity_distribution": severity_counts,
            "rule_distribution": rule_counts,
            "registered_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled])
        }
    
    def disable_rule(self, rule_name: str):
        """Disable an alert rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled alert rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """Enable an alert rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled alert rule: {rule_name}")
    
    def get_rule_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all alert rules"""
        return {
            rule_name: {
                "enabled": rule.enabled,
                "severity": rule.severity.value,
                "channels": [c.value for c in rule.channels],
                "cooldown_minutes": rule.cooldown_minutes,
                "description": rule.description,
                "last_triggered": self.last_alert_times.get(rule_name)
            }
            for rule_name, rule in self.rules.items()
        }


# Global alert manager instance
alert_manager = AlertManager()