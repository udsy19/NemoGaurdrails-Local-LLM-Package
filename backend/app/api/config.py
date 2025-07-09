from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from app.utils.logger import app_logger

router = APIRouter()

# Pydantic models
class ConfigUpdate(BaseModel):
    config: Dict[str, Any]

class DetectorConfigUpdate(BaseModel):
    detector_name: str
    config: Dict[str, Any]

class ExportRequest(BaseModel):
    format: str = Field("yaml", pattern="^(yaml|json)$")
    include_user_configs: bool = True

class ImportRequest(BaseModel):
    config_data: str
    format: str = Field("yaml", pattern="^(yaml|json)$")

# Dependencies
def get_config_service():
    from app.main import app
    return app.state.config_service if hasattr(app.state, 'config_service') else None

def get_model_manager():
    from app.main import app
    return app.state.model_manager

@router.get("/")
async def get_system_config(
    config_service = Depends(get_config_service)
):
    """Get current system configuration"""
    try:
        if not config_service:
            # Fallback to basic config if service not available
            model_manager = get_model_manager()
            return {
                "detectors": model_manager.get_available_detectors(),
                "active_detectors": model_manager.get_active_detectors(),
                "timestamp": datetime.now().isoformat()
            }
        
        config = config_service.get_system_config()
        
        return {
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get system config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system config: {str(e)}")

@router.put("/")
async def update_system_config(
    request: ConfigUpdate,
    config_service = Depends(get_config_service)
):
    """Update system configuration"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        success = await config_service.update_system_config(request.config)
        
        if success:
            return {
                "message": "System configuration updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update system configuration")
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to update system config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update system config: {str(e)}")

@router.get("/detectors")
async def get_detector_configs(
    config_service = Depends(get_config_service),
    model_manager = Depends(get_model_manager)
):
    """Get configuration for all detectors"""
    try:
        if config_service:
            detector_configs = config_service.get_active_detectors_config()
        else:
            # Fallback to model manager
            detector_configs = {
                name: info["config"] 
                for name, info in model_manager.get_available_detectors().items()
            }
        
        return {
            "detector_configs": detector_configs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get detector configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detector configs: {str(e)}")

@router.put("/detectors")
async def update_detector_config(
    request: DetectorConfigUpdate,
    config_service = Depends(get_config_service),
    model_manager = Depends(get_model_manager)
):
    """Update configuration for a specific detector"""
    try:
        # Validate detector name
        available_detectors = model_manager.get_available_detectors()
        if request.detector_name not in available_detectors:
            raise HTTPException(
                status_code=404,
                detail=f"Detector not found: {request.detector_name}"
            )
        
        # Update via config service if available
        if config_service:
            success = await config_service.update_detector_config(
                request.detector_name,
                request.config
            )
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update detector configuration")
        else:
            # Fallback to model manager
            model_manager.update_detector_config(request.detector_name, request.config)
        
        return {
            "message": f"Configuration updated for detector: {request.detector_name}",
            "detector_name": request.detector_name,
            "config": request.config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to update detector config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update detector config: {str(e)}")

@router.get("/detectors/{detector_name}")
async def get_detector_config(
    detector_name: str,
    config_service = Depends(get_config_service),
    model_manager = Depends(get_model_manager)
):
    """Get configuration for a specific detector"""
    try:
        # Validate detector name
        available_detectors = model_manager.get_available_detectors()
        if detector_name not in available_detectors:
            raise HTTPException(
                status_code=404,
                detail=f"Detector not found: {detector_name}"
            )
        
        if config_service:
            config = config_service.get_detector_config(detector_name)
        else:
            config = available_detectors[detector_name]["config"]
        
        return {
            "detector_name": detector_name,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to get detector config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detector config: {str(e)}")

@router.get("/models")
async def get_model_info(
    model_manager = Depends(get_model_manager)
):
    """Get information about loaded models"""
    try:
        system_status = await model_manager.get_system_status()
        
        # Get Ollama model info
        ollama_models = []
        try:
            ollama_models_data = await model_manager.ollama_client.list_models()
            ollama_models = ollama_models_data.get("models", [])
        except Exception as e:
            app_logger.warning(f"Failed to get Ollama models: {e}")
        
        return {
            "system_status": system_status,
            "ollama_models": ollama_models,
            "detector_models": {
                name: {
                    "loaded": info["loaded"],
                    "healthy": info["healthy"]
                }
                for name, info in system_status["detector_status"].items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/reset")
async def reset_to_defaults(
    config_service = Depends(get_config_service)
):
    """Reset system configuration to defaults"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        success = await config_service.reset_to_defaults()
        
        if success:
            return {
                "message": "System configuration reset to defaults",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reset configuration")
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to reset config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset config: {str(e)}")

@router.post("/export")
async def export_config(
    request: ExportRequest,
    config_service = Depends(get_config_service)
):
    """Export system configuration"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        result = config_service.export_config(request.format)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "export_data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to export config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export config: {str(e)}")

@router.post("/import")
async def import_config(
    request: ImportRequest,
    config_service = Depends(get_config_service)
):
    """Import system configuration"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        success = await config_service.import_config(request.config_data, request.format)
        
        if success:
            return {
                "message": "Configuration imported successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to import configuration")
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to import config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import config: {str(e)}")

@router.post("/import/file")
async def import_config_file(
    file: UploadFile = File(...),
    format: Optional[str] = None,
    config_service = Depends(get_config_service)
):
    """Import system configuration from uploaded file"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        # Determine format from file extension if not provided
        if not format:
            if file.filename.endswith('.yml') or file.filename.endswith('.yaml'):
                format = 'yaml'
            elif file.filename.endswith('.json'):
                format = 'json'
            else:
                raise HTTPException(status_code=400, detail="Unable to determine file format")
        
        # Read file content
        content = await file.read()
        config_data = content.decode('utf-8')
        
        success = await config_service.import_config(config_data, format)
        
        if success:
            return {
                "message": f"Configuration imported successfully from {file.filename}",
                "filename": file.filename,
                "format": format,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to import configuration from file")
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to import config file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import config file: {str(e)}")

@router.get("/schema")
async def get_config_schema(
    config_service = Depends(get_config_service)
):
    """Get configuration schema for validation"""
    try:
        if not config_service:
            # Return basic schema if service not available
            return {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detectors": {"type": "object"},
                        "models": {"type": "object"}
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
        
        schema = config_service.get_config_schema()
        
        return {
            "schema": schema,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get config schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config schema: {str(e)}")

@router.get("/health")
async def get_config_health(
    config_service = Depends(get_config_service)
):
    """Get health status of configuration service"""
    try:
        if not config_service:
            return {
                "status": "unavailable",
                "message": "Configuration service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        health_status = config_service.get_health_status()
        
        return {
            "status": "healthy" if health_status["config_loaded"] else "unhealthy",
            "details": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get config health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config health: {str(e)}")

@router.post("/reload")
async def reload_config(
    config_service = Depends(get_config_service)
):
    """Reload configuration from files"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        # This would reload configs from files
        await config_service.load_system_config()
        
        return {
            "message": "Configuration reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to reload config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")

# User-specific configurations
@router.get("/users/{user_id}")
async def get_user_config(
    user_id: str,
    config_service = Depends(get_config_service)
):
    """Get user-specific configuration"""
    try:
        if not config_service:
            return {"user_id": user_id, "config": {}, "timestamp": datetime.now().isoformat()}
        
        config = config_service.get_user_config(user_id)
        
        return {
            "user_id": user_id,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get user config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user config: {str(e)}")

@router.put("/users/{user_id}")
async def set_user_config(
    user_id: str,
    request: ConfigUpdate,
    config_service = Depends(get_config_service)
):
    """Set user-specific configuration"""
    try:
        if not config_service:
            raise HTTPException(status_code=503, detail="Configuration service not available")
        
        config_service.set_user_config(user_id, request.config)
        
        return {
            "message": f"User configuration updated for {user_id}",
            "user_id": user_id,
            "config": request.config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to set user config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set user config: {str(e)}")

config_router = router