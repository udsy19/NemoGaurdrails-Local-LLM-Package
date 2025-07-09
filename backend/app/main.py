from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import threading
from typing import Dict, Any
import asyncio
from contextlib import asynccontextmanager

from app.utils.logger import app_logger
from app.utils.exceptions import AILLMSafetyException
from app.utils.health_monitor import system_health_monitor, HealthStatus
from app.utils.graceful_degradation import graceful_degradation_manager, DegradationLevel
from app.utils.system_diagnostics import system_diagnostics, DiagnosticLevel
from app.utils.correlation_context import CorrelationContextManager, with_correlation_context
from app.utils.circuit_breaker import circuit_breaker_registry
from app.utils.alert_system import alert_manager, AlertSeverity
from app.utils.performance_monitor import performance_monitor
from app.utils.self_healing import self_healing_system
from app.models.model_manager import ModelManager
from app.services.detection_service import DetectionService
from app.services.chat_service import ChatService
from app.api.chat import chat_router
from app.api.detectors import detector_router  
from app.api.config import config_router

# Try to import NeMo Guardrails, fall back to simple implementation
try:
    from app.guardrails.guardrails_manager import GuardrailsManager
    NEMO_AVAILABLE = True
except ImportError:
    from app.guardrails.simple_guardrails import SimpleGuardrails as GuardrailsManager
    NEMO_AVAILABLE = False
    app_logger.warning("NeMo Guardrails not available, using simplified implementation")

# Global instances
model_manager: ModelManager = None
detection_service: DetectionService = None
chat_service: ChatService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, detection_service, chat_service
    
    # Initialize correlation context for startup
    startup_context = CorrelationContextManager.create_context(
        correlation_id="system_startup",
        metadata={"operation": "system_initialization"}
    )
    CorrelationContextManager.set_context(startup_context)
    
    app_logger.info("Starting AI Safety System with enhanced resilience...")
    
    try:
        # Setup health monitoring
        await _setup_health_monitoring()
        
        # Setup graceful degradation
        _setup_graceful_degradation()
        
        # Initialize model manager
        model_manager = ModelManager()
        await model_manager.initialize()
        
        # Initialize services
        detection_service = DetectionService(model_manager)
        chat_service = ChatService(model_manager, detection_service)
        await chat_service.initialize()
        
        # Start advanced monitoring systems
        await performance_monitor.start_monitoring()
        await self_healing_system.start_self_healing()
        
        # Store in app state
        app.state.model_manager = model_manager
        app.state.detection_service = detection_service
        app.state.chat_service = chat_service
        app.state.health_monitor = system_health_monitor
        app.state.degradation_manager = graceful_degradation_manager
        app.state.diagnostics = system_diagnostics
        app.state.alert_manager = alert_manager
        app.state.performance_monitor = performance_monitor
        app.state.self_healing_system = self_healing_system
        
        # Run initial system diagnostics
        initial_diagnostics = await system_diagnostics.run_diagnostics(DiagnosticLevel.BASIC)
        app_logger.info(f"Initial diagnostics completed: {len(initial_diagnostics)} checks")
        
        # Perform initial health check
        system_health = await system_health_monitor.check_all_components("system_startup")
        app_logger.info(f"System health status: {system_health.overall_status.value}")
        
        # Check if system needs degradation
        if system_health.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            trigger_conditions = ["system_startup_issues"]
            await graceful_degradation_manager.evaluate_degradation(trigger_conditions)
        
        app_logger.info(f"AI Safety System initialized successfully (NeMo: {NEMO_AVAILABLE})")
        
    except Exception as e:
        app_logger.error(f"Failed to initialize AI Safety System: {e}")
        # Try emergency diagnostics
        try:
            emergency_diagnostics = await system_diagnostics.run_diagnostics(DiagnosticLevel.EMERGENCY)
            app_logger.error(f"Emergency diagnostics: {len(emergency_diagnostics)} issues found")
        except:
            pass
        raise
    
    yield
    
    # Cleanup
    app_logger.info("Shutting down AI Safety System...")
    
    # Stop advanced monitoring systems
    await performance_monitor.stop_monitoring()
    await self_healing_system.stop_self_healing()
    
    # Clear correlation context
    CorrelationContextManager.clear_context()
    
    # Cleanup services
    if model_manager:
        await model_manager.cleanup()
    
    # Final health check
    try:
        final_health = await system_health_monitor.check_all_components("system_shutdown")
        app_logger.info(f"Final system health: {final_health.overall_status.value}")
    except Exception as e:
        app_logger.error(f"Failed final health check: {e}")


async def _setup_health_monitoring():
    """Setup health monitoring for all system components"""
    
    # Register health checkers
    system_health_monitor.register_checker(
        "ollama_client",
        _check_ollama_health,
        timeout=10.0
    )
    
    system_health_monitor.register_checker(
        "model_manager",
        _check_model_manager_health,
        timeout=5.0
    )
    
    system_health_monitor.register_checker(
        "detection_service",
        _check_detection_service_health,
        timeout=5.0
    )
    
    system_health_monitor.register_checker(
        "circuit_breakers",
        _check_circuit_breakers_health,
        timeout=2.0
    )
    
    app_logger.info("Health monitoring setup completed")


def _setup_graceful_degradation():
    """Setup graceful degradation features and fallback handlers"""
    
    # Register features that can be degraded
    graceful_degradation_manager.register_feature(
        "ai_generation",
        {
            "essential": True,
            "fallback_available": True,
            "resource_intensive": True,
            "description": "AI response generation"
        }
    )
    
    graceful_degradation_manager.register_feature(
        "advanced_detection",
        {
            "essential": False,
            "fallback_available": True,
            "resource_intensive": True,
            "description": "Advanced safety detection"
        }
    )
    
    graceful_degradation_manager.register_feature(
        "detailed_analytics",
        {
            "essential": False,
            "fallback_available": False,
            "resource_intensive": True,
            "description": "Detailed analytics and reporting"
        }
    )
    
    # Register fallback handlers
    graceful_degradation_manager.register_fallback_handler(
        "cached_responses",
        _handle_cached_responses
    )
    
    graceful_degradation_manager.register_fallback_handler(
        "basic_safety_only",
        _handle_basic_safety_only
    )
    
    graceful_degradation_manager.register_fallback_handler(
        "reduce_memory_usage",
        _handle_reduce_memory_usage
    )
    
    app_logger.info("Graceful degradation setup completed")


async def _check_ollama_health() -> Dict[str, Any]:
    """Health check for Ollama client"""
    try:
        if model_manager and model_manager.ollama_client:
            is_healthy = await model_manager.ollama_client.health_check()
            metrics = model_manager.ollama_client.get_metrics()
            
            if is_healthy:
                return {
                    "status": HealthStatus.HEALTHY,
                    "metrics": metrics
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "error": "Ollama health check failed",
                    "metrics": metrics
                }
        else:
            return {
                "status": HealthStatus.CRITICAL,
                "error": "Ollama client not initialized"
            }
    except Exception as e:
        return {
            "status": HealthStatus.CRITICAL,
            "error": str(e)
        }


async def _check_model_manager_health() -> Dict[str, Any]:
    """Health check for model manager"""
    try:
        if model_manager:
            available_detectors = model_manager.get_available_detectors()
            is_initialized = model_manager.is_initialized
            
            loaded_count = sum(1 for detector in available_detectors.values() if detector.get("loaded", False))
            total_count = len(available_detectors)
            
            if is_initialized and loaded_count == total_count:
                return {
                    "status": HealthStatus.HEALTHY,
                    "metrics": {
                        "total_detectors": total_count,
                        "loaded_detectors": loaded_count,
                        "detector_status": available_detectors
                    }
                }
            elif loaded_count > 0:
                return {
                    "status": HealthStatus.DEGRADED,
                    "error": f"Only {loaded_count}/{total_count} detectors loaded",
                    "metrics": {
                        "total_detectors": total_count,
                        "loaded_detectors": loaded_count
                    }
                }
            else:
                return {
                    "status": HealthStatus.CRITICAL,
                    "error": "No detectors loaded"
                }
        else:
            return {
                "status": HealthStatus.CRITICAL,
                "error": "Model manager not initialized"
            }
    except Exception as e:
        return {
            "status": HealthStatus.CRITICAL,
            "error": str(e)
        }


async def _check_detection_service_health() -> Dict[str, Any]:
    """Health check for detection service"""
    try:
        if detection_service:
            # Simple test detection
            test_result = await detection_service.detect_input("test message", {})
            
            return {
                "status": HealthStatus.HEALTHY,
                "metrics": {
                    "test_detection_successful": True,
                    "detectors_available": len(test_result.get("results", {}))
                }
            }
        else:
            return {
                "status": HealthStatus.CRITICAL,
                "error": "Detection service not initialized"
            }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "error": str(e)
        }


async def _check_circuit_breakers_health() -> Dict[str, Any]:
    """Health check for circuit breakers"""
    try:
        all_metrics = circuit_breaker_registry.get_all_metrics()
        unhealthy_breakers = circuit_breaker_registry.get_unhealthy_breakers()
        
        if not unhealthy_breakers:
            return {
                "status": HealthStatus.HEALTHY,
                "metrics": {
                    "total_breakers": len(all_metrics),
                    "healthy_breakers": len(all_metrics),
                    "breaker_metrics": all_metrics
                }
            }
        else:
            return {
                "status": HealthStatus.DEGRADED,
                "error": f"{len(unhealthy_breakers)} circuit breakers are open",
                "metrics": {
                    "total_breakers": len(all_metrics),
                    "unhealthy_breakers": len(unhealthy_breakers),
                    "unhealthy_breaker_names": list(unhealthy_breakers.keys())
                }
            }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "error": str(e)
        }


async def _handle_cached_responses(context: Dict[str, Any]):
    """Fallback handler for cached responses"""
    app_logger.info("Activating cached responses fallback")
    # Implementation would cache common responses


async def _handle_basic_safety_only(context: Dict[str, Any]):
    """Fallback handler for basic safety only"""
    app_logger.info("Activating basic safety only mode")
    # Implementation would disable advanced detectors


async def _handle_reduce_memory_usage(context: Dict[str, Any]):
    """Fallback handler to reduce memory usage"""
    app_logger.info("Activating memory reduction measures")
    # Implementation would clear caches, etc.

app = FastAPI(
    title="AI Safety System",
    description="Local AI Safety System with Multiple Detectors",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - simplified for Streamlit only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(AILLMSafetyException)
async def safety_exception_handler(request, exc: AILLMSafetyException):
    return JSONResponse(
        status_code=400,
        content={
            "error": "AI Safety Error",
            "message": exc.message,
            "error_code": exc.error_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    app_logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(detector_router, prefix="/api/detectors", tags=["detectors"])
app.include_router(config_router, prefix="/api/config", tags=["config"])

@app.get("/")
async def root():
    return {
        "message": "AI Safety System Backend", 
        "status": "running",
        "streamlit_interface": "http://localhost:8501",
        "nemo_guardrails": NEMO_AVAILABLE
    }

@app.get("/api")
async def api_root():
    return {
        "message": "AI Safety System API", 
        "status": "running",
        "nemo_guardrails": NEMO_AVAILABLE
    }

@app.get("/health")
@with_correlation_context()
async def health_check():
    """Enhanced health check with comprehensive system monitoring"""
    try:
        # Get comprehensive system health
        system_health = await system_health_monitor.check_all_components()
        
        # Get degradation status
        degradation_state = graceful_degradation_manager.get_current_state()
        
        # Get circuit breaker status
        circuit_breaker_metrics = circuit_breaker_registry.get_all_metrics()
        
        # Basic compatibility response
        basic_status = {
            "status": system_health.overall_status.value,
            "models_loaded": bool(model_manager and model_manager.is_initialized),
            "detectors_available": list(model_manager.get_available_detectors().keys()) if model_manager else [],
            "ollama_connected": any(
                comp.name == "ollama_client" and comp.status.value == "healthy" 
                for comp in system_health.components
            ),
            "nemo_guardrails": NEMO_AVAILABLE
        }
        
        # Enhanced status information
        enhanced_status = {
            **basic_status,
            "system_health": {
                "overall_status": system_health.overall_status.value,
                "components": [
                    {
                        "name": comp.name,
                        "status": comp.status.value,
                        "response_time": comp.response_time,
                        "error": comp.error_message
                    }
                    for comp in system_health.components
                ],
                "system_metrics": system_health.system_metrics,
                "timestamp": system_health.timestamp
            },
            "degradation_status": {
                "level": degradation_state.degradation_level.value,
                "disabled_features": list(degradation_state.disabled_features),
                "reason": degradation_state.reason,
                "last_updated": degradation_state.last_updated
            },
            "circuit_breakers": circuit_breaker_metrics,
            "correlation_id": system_health.correlation_id
        }
        
        return enhanced_status
        
    except Exception as e:
        app_logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "correlation_id": CorrelationContextManager.get_correlation_id()
        }


@app.get("/health/detailed")
@with_correlation_context()
async def detailed_health_check():
    """Detailed health check with diagnostics"""
    try:
        # Run comprehensive health check
        system_health = await system_health_monitor.check_all_components()
        
        # Run detailed diagnostics
        diagnostics = await system_diagnostics.run_diagnostics(DiagnosticLevel.DETAILED)
        
        # Get health trends
        health_trends = system_health_monitor.get_health_trends(hours=1)
        
        # Get degradation capabilities
        system_capabilities = graceful_degradation_manager.get_system_capabilities()
        
        return {
            "system_health": system_health_monitor.to_dict(system_health),
            "diagnostics": [
                {
                    "component": diag.component,
                    "status": diag.status,
                    "details": diag.details,
                    "recommendations": diag.recommendations,
                    "severity": diag.severity,
                    "timestamp": diag.timestamp
                }
                for diag in diagnostics
            ],
            "health_trends": health_trends,
            "system_capabilities": system_capabilities,
            "diagnostic_summary": system_diagnostics.get_diagnostic_summary()
        }
        
    except Exception as e:
        app_logger.error(f"Detailed health check failed: {e}")
        return {
            "error": str(e),
            "correlation_id": CorrelationContextManager.get_correlation_id()
        }


@app.post("/diagnostics/run")
@with_correlation_context()
async def run_diagnostics(level: str = "detailed"):
    """Run system diagnostics on demand"""
    try:
        # Map string to enum
        level_map = {
            "basic": DiagnosticLevel.BASIC,
            "detailed": DiagnosticLevel.DETAILED,
            "comprehensive": DiagnosticLevel.COMPREHENSIVE,
            "emergency": DiagnosticLevel.EMERGENCY
        }
        
        diagnostic_level = level_map.get(level.lower(), DiagnosticLevel.DETAILED)
        
        # Run diagnostics
        results = await system_diagnostics.run_diagnostics(diagnostic_level)
        
        # Generate recovery plan if needed
        recovery_plan = system_diagnostics.generate_recovery_plan(results)
        
        response = {
            "diagnostic_level": diagnostic_level.value,
            "results": [
                {
                    "component": result.component,
                    "status": result.status,
                    "details": result.details,
                    "recommendations": result.recommendations,
                    "severity": result.severity,
                    "timestamp": result.timestamp
                }
                for result in results
            ],
            "summary": {
                "total_checks": len(results),
                "critical_issues": len([r for r in results if r.severity == "critical"]),
                "warnings": len([r for r in results if r.severity == "medium"]),
                "healthy_components": len([r for r in results if r.status == "healthy"])
            }
        }
        
        if recovery_plan:
            response["recovery_plan"] = {
                "description": recovery_plan.description,
                "actions": [action.value for action in recovery_plan.actions],
                "estimated_downtime": recovery_plan.estimated_downtime,
                "risk_level": recovery_plan.risk_level,
                "prerequisites": recovery_plan.prerequisites
            }
        
        return response
        
    except Exception as e:
        app_logger.error(f"Diagnostics run failed: {e}")
        return {
            "error": str(e),
            "correlation_id": CorrelationContextManager.get_correlation_id()
        }


@app.post("/recovery/execute")
@with_correlation_context()
async def execute_recovery():
    """Execute automatic system recovery"""
    try:
        # Run diagnostics to identify issues
        diagnostics = await system_diagnostics.run_diagnostics(DiagnosticLevel.DETAILED)
        
        # Generate recovery plan
        recovery_plan = system_diagnostics.generate_recovery_plan(diagnostics)
        
        if not recovery_plan:
            return {
                "message": "No recovery actions needed",
                "diagnostics_summary": {
                    "total_checks": len(diagnostics),
                    "issues_found": len([d for d in diagnostics if d.status != "healthy"])
                }
            }
        
        # Execute recovery plan
        recovery_result = await system_diagnostics.execute_recovery_plan(recovery_plan)
        
        # Re-run diagnostics to check if issues are resolved
        post_recovery_diagnostics = await system_diagnostics.run_diagnostics(DiagnosticLevel.BASIC)
        
        return {
            "recovery_executed": True,
            "recovery_plan": recovery_plan.description,
            "recovery_result": recovery_result,
            "post_recovery_status": {
                "total_checks": len(post_recovery_diagnostics),
                "remaining_issues": len([d for d in post_recovery_diagnostics if d.status != "healthy"])
            }
        }
        
    except Exception as e:
        app_logger.error(f"Recovery execution failed: {e}")
        return {
            "error": str(e),
            "correlation_id": CorrelationContextManager.get_correlation_id()
        }


# Phase 3 & 4: Advanced Monitoring and Self-Healing Endpoints

@app.get("/alerts")
@with_correlation_context()
async def get_alerts():
    """Get all active alerts"""
    try:
        active_alerts = alert_manager.get_active_alerts()
        alert_stats = alert_manager.get_alert_stats()
        
        return {
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "stats": alert_stats,
            "rule_status": alert_manager.get_rule_status()
        }
    except Exception as e:
        app_logger.error(f"Failed to get alerts: {e}")
        return {"error": str(e)}


@app.get("/alerts/history")
@with_correlation_context()
async def get_alert_history(hours: int = 24):
    """Get alert history"""
    try:
        history = alert_manager.get_alert_history(hours)
        return {
            "alerts": [alert.to_dict() for alert in history],
            "timeframe_hours": hours,
            "total_alerts": len(history)
        }
    except Exception as e:
        app_logger.error(f"Failed to get alert history: {e}")
        return {"error": str(e)}


@app.post("/alerts/{alert_id}/resolve")
@with_correlation_context()
async def resolve_alert(alert_id: str, resolution_note: str = ""):
    """Resolve an active alert"""
    try:
        resolved_alert = await alert_manager.resolve_alert(alert_id, resolution_note)
        if resolved_alert:
            return {
                "message": "Alert resolved successfully",
                "alert": resolved_alert.to_dict()
            }
        else:
            return {"error": "Alert not found"}
    except Exception as e:
        app_logger.error(f"Failed to resolve alert: {e}")
        return {"error": str(e)}


@app.get("/performance")
@with_correlation_context()
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        performance_summary = performance_monitor.get_performance_summary()
        return performance_summary
    except Exception as e:
        app_logger.error(f"Failed to get performance metrics: {e}")
        return {"error": str(e)}


@app.get("/performance/requests")
@with_correlation_context()
async def get_request_performance(count: int = 100):
    """Get recent request performance data"""
    try:
        recent_requests = performance_monitor.get_recent_requests(count)
        slow_requests = performance_monitor.get_slow_requests()
        error_requests = performance_monitor.get_error_requests()
        
        return {
            "recent_requests": recent_requests,
            "slow_requests": slow_requests,
            "error_requests": error_requests,
            "request_count": len(recent_requests)
        }
    except Exception as e:
        app_logger.error(f"Failed to get request performance: {e}")
        return {"error": str(e)}


@app.get("/self-healing")
@with_correlation_context()
async def get_self_healing_status():
    """Get self-healing system status"""
    try:
        healing_stats = self_healing_system.get_healing_stats()
        rule_status = self_healing_system.get_rule_status()
        recent_events = self_healing_system.get_recent_healing_events()
        
        return {
            "stats": healing_stats,
            "rules": rule_status,
            "recent_events": [event.to_dict() for event in recent_events]
        }
    except Exception as e:
        app_logger.error(f"Failed to get self-healing status: {e}")
        return {"error": str(e)}


@app.post("/self-healing/rules/{rule_name}/enable")
@with_correlation_context()
async def enable_healing_rule(rule_name: str):
    """Enable a self-healing rule"""
    try:
        self_healing_system.enable_rule(rule_name)
        return {"message": f"Healing rule '{rule_name}' enabled successfully"}
    except Exception as e:
        app_logger.error(f"Failed to enable healing rule: {e}")
        return {"error": str(e)}


@app.post("/self-healing/rules/{rule_name}/disable")
@with_correlation_context()
async def disable_healing_rule(rule_name: str):
    """Disable a self-healing rule"""
    try:
        self_healing_system.disable_rule(rule_name)
        return {"message": f"Healing rule '{rule_name}' disabled successfully"}
    except Exception as e:
        app_logger.error(f"Failed to disable healing rule: {e}")
        return {"error": str(e)}


@app.post("/alerts/rules/{rule_name}/enable")
@with_correlation_context()
async def enable_alert_rule(rule_name: str):
    """Enable an alert rule"""
    try:
        alert_manager.enable_rule(rule_name)
        return {"message": f"Alert rule '{rule_name}' enabled successfully"}
    except Exception as e:
        app_logger.error(f"Failed to enable alert rule: {e}")
        return {"error": str(e)}


@app.post("/alerts/rules/{rule_name}/disable")
@with_correlation_context()
async def disable_alert_rule(rule_name: str):
    """Disable an alert rule"""
    try:
        alert_manager.disable_rule(rule_name)
        return {"message": f"Alert rule '{rule_name}' disabled successfully"}
    except Exception as e:
        app_logger.error(f"Failed to disable alert rule: {e}")
        return {"error": str(e)}


@app.get("/monitoring/overview")
@with_correlation_context()
async def get_monitoring_overview():
    """Get comprehensive monitoring overview"""
    try:
        # Collect metrics for alert evaluation
        system_health = await system_health_monitor.check_all_components()
        performance_summary = performance_monitor.get_performance_summary()
        healing_stats = self_healing_system.get_healing_stats()
        alert_stats = alert_manager.get_alert_stats()
        
        # Prepare metrics for alert evaluation
        metrics = {
            "memory_usage": performance_summary.get("current_metrics", {}).get("memory_usage_mb", 0),
            "cpu_usage": performance_summary.get("current_metrics", {}).get("cpu_usage_percent", 0),
            "disk_usage": 0,  # Would be calculated from actual disk metrics
            "error_rate": performance_summary.get("current_metrics", {}).get("error_rate", 0),
            "active_connections": performance_summary.get("current_metrics", {}).get("active_connections", 0),
            "health_status": system_health.overall_status.value
        }
        
        # Evaluate alert conditions
        await alert_manager.evaluate_conditions(metrics)
        
        return {
            "system_health": {
                "overall_status": system_health.overall_status.value,
                "component_count": len(system_health.components),
                "timestamp": system_health.timestamp
            },
            "performance": {
                "monitoring_active": performance_summary.get("monitoring_active", False),
                "active_requests": performance_summary.get("active_requests", 0),
                "memory_usage_mb": metrics["memory_usage"],
                "cpu_usage_percent": metrics["cpu_usage"]
            },
            "alerts": {
                "active_count": alert_stats.get("active_alerts", 0),
                "total_count": alert_stats.get("total_alerts", 0),
                "enabled_rules": alert_stats.get("enabled_rules", 0)
            },
            "self_healing": {
                "active": healing_stats.get("healing_active", False),
                "total_events": healing_stats.get("total_healing_events", 0),
                "success_rate": healing_stats.get("success_rate", 0),
                "enabled_rules": healing_stats.get("enabled_rules", 0)
            },
            "correlation_id": CorrelationContextManager.get_correlation_id()
        }
    except Exception as e:
        app_logger.error(f"Failed to get monitoring overview: {e}")
        return {"error": str(e)}


# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "chat_message":
                message = data.get("message", "")
                detector_config = data.get("detector_config", {})
                
                try:
                    # Process message through safety pipeline
                    response = await chat_service.process_message(message, detector_config)
                    await websocket.send_json(response)
                    
                except Exception as e:
                    app_logger.error(f"Error processing chat message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to process message",
                        "error": str(e)
                    })
            
    except WebSocketDisconnect:
        app_logger.info("WebSocket connection closed")
    except Exception as e:
        app_logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
