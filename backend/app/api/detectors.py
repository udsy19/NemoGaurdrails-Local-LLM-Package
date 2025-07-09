from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.utils.logger import app_logger

router = APIRouter()

# Pydantic models
class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    detectors: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None

class DetectorConfigUpdate(BaseModel):
    detector_name: str
    config: Dict[str, Any]

class DetectorToggle(BaseModel):
    detector_names: List[str]
    enabled: bool

class TestDetectorRequest(BaseModel):
    detector_name: str
    test_text: str
    config: Optional[Dict[str, Any]] = None

class BatchDetectionRequest(BaseModel):
    texts: List[str] = Field(..., max_items=20)
    detectors: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None

# Dependencies
def get_detection_service():
    from app.main import app
    return app.state.detection_service

def get_model_manager():
    from app.main import app
    return app.state.model_manager

@router.get("/")
async def list_detectors(
    model_manager = Depends(get_model_manager)
):
    """Get list of all available detectors with their status"""
    try:
        available_detectors = model_manager.get_available_detectors()
        active_detectors = model_manager.get_active_detectors()
        
        detector_list = []
        for detector_name, info in available_detectors.items():
            detector_info = {
                "name": detector_name,
                "loaded": info["loaded"],
                "active": info["active"],
                "config": info["config"],
                "description": get_detector_description(detector_name)
            }
            detector_list.append(detector_info)
        
        return {
            "detectors": detector_list,
            "total_detectors": len(available_detectors),
            "active_detectors": len(active_detectors),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to list detectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list detectors: {str(e)}")

@router.post("/detect")
async def run_detection(
    request: DetectionRequest,
    detection_service = Depends(get_detection_service)
):
    """Run detection on text with specified detectors"""
    try:
        result = await detection_service.run_detection(
            text=request.text,
            detector_names=request.detectors,
            detector_config=request.config
        )
        
        return {
            "detection_id": f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            **result
        }
        
    except Exception as e:
        app_logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/test")
async def test_detector(
    request: TestDetectorRequest,
    detection_service = Depends(get_detection_service)
):
    """Test a specific detector with given text"""
    try:
        result = await detection_service.test_detector(
            detector_name=request.detector_name,
            test_text=request.test_text
        )
        
        return {
            "test_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            **result
        }
        
    except Exception as e:
        app_logger.error(f"Detector test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detector test failed: {str(e)}")

@router.post("/batch")
async def batch_detection(
    request: BatchDetectionRequest,
    detection_service = Depends(get_detection_service)
):
    """Run detection on multiple texts"""
    try:
        results = await detection_service.batch_detection(
            texts=request.texts,
            detector_names=request.detectors
        )
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "text_count": len(request.texts),
            "results": results
        }
        
    except Exception as e:
        app_logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@router.get("/active")
async def get_active_detectors(
    model_manager = Depends(get_model_manager)
):
    """Get list of currently active detectors"""
    try:
        active_detectors = model_manager.get_active_detectors()
        
        return {
            "active_detectors": active_detectors,
            "count": len(active_detectors),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get active detectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active detectors: {str(e)}")

@router.post("/active")
async def set_active_detectors(
    detector_names: List[str],
    model_manager = Depends(get_model_manager)
):
    """Set which detectors are active"""
    try:
        # Validate detector names
        available_detectors = model_manager.get_available_detectors()
        invalid_detectors = [name for name in detector_names if name not in available_detectors]
        
        if invalid_detectors:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid detector names: {invalid_detectors}"
            )
        
        model_manager.set_active_detectors(detector_names)
        
        return {
            "message": "Active detectors updated successfully",
            "active_detectors": detector_names,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to set active detectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set active detectors: {str(e)}")

@router.put("/config")
async def update_detector_config(
    request: DetectorConfigUpdate,
    model_manager = Depends(get_model_manager)
):
    """Update configuration for a specific detector"""
    try:
        available_detectors = model_manager.get_available_detectors()
        
        if request.detector_name not in available_detectors:
            raise HTTPException(
                status_code=404, 
                detail=f"Detector not found: {request.detector_name}"
            )
        
        model_manager.update_detector_config(request.detector_name, request.config)
        
        return {
            "message": f"Configuration updated for {request.detector_name}",
            "detector_name": request.detector_name,
            "config": request.config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to update detector config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update detector config: {str(e)}")

@router.get("/config/{detector_name}")
async def get_detector_config(
    detector_name: str,
    model_manager = Depends(get_model_manager)
):
    """Get configuration for a specific detector"""
    try:
        available_detectors = model_manager.get_available_detectors()
        
        if detector_name not in available_detectors:
            raise HTTPException(
                status_code=404, 
                detail=f"Detector not found: {detector_name}"
            )
        
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

@router.post("/reload/{detector_name}")
async def reload_detector(
    detector_name: str,
    model_manager = Depends(get_model_manager)
):
    """Reload a specific detector"""
    try:
        available_detectors = model_manager.get_available_detectors()
        
        if detector_name not in available_detectors:
            raise HTTPException(
                status_code=404, 
                detail=f"Detector not found: {detector_name}"
            )
        
        await model_manager.reload_detector(detector_name)
        
        return {
            "message": f"Detector {detector_name} reloaded successfully",
            "detector_name": detector_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to reload detector: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload detector: {str(e)}")

@router.get("/stats")
async def get_detector_stats(
    detection_service = Depends(get_detection_service)
):
    """Get statistics about detector usage and performance"""
    try:
        stats = detection_service.get_detector_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            **stats
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get detector stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detector stats: {str(e)}")

@router.get("/health")
async def get_detectors_health(
    model_manager = Depends(get_model_manager)
):
    """Get health status of all detectors"""
    try:
        system_status = await model_manager.get_system_status()
        
        return {
            "status": "healthy" if system_status["initialized"] else "unhealthy",
            "details": system_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get detectors health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detectors health: {str(e)}")

# Presets and templates
@router.get("/presets")
async def get_detector_presets():
    """Get predefined detector configuration presets"""
    try:
        presets = {
            "strict": {
                "toxicity": {"threshold": 0.5},
                "pii": {"sensitivity": 0.9},
                "prompt_injection": {"threshold": 0.3},
                "topic": {"threshold": 0.6},
                "spam": {"threshold": 0.4}
            },
            "balanced": {
                "toxicity": {"threshold": 0.7},
                "pii": {"sensitivity": 0.8},
                "prompt_injection": {"threshold": 0.5},
                "topic": {"threshold": 0.7},
                "spam": {"threshold": 0.6}
            },
            "permissive": {
                "toxicity": {"threshold": 0.8},
                "pii": {"sensitivity": 0.7},
                "prompt_injection": {"threshold": 0.7},
                "topic": {"threshold": 0.8},
                "spam": {"threshold": 0.7}
            }
        }
        
        return {
            "presets": presets,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get detector presets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detector presets: {str(e)}")

@router.post("/presets/{preset_name}")
async def apply_detector_preset(
    preset_name: str,
    model_manager = Depends(get_model_manager)
):
    """Apply a predefined detector configuration preset"""
    try:
        presets_response = await get_detector_presets()
        presets = presets_response["presets"]
        
        if preset_name not in presets:
            raise HTTPException(
                status_code=404,
                detail=f"Preset not found: {preset_name}"
            )
        
        preset_config = presets[preset_name]
        
        # Apply configuration to each detector
        for detector_name, config in preset_config.items():
            if detector_name in model_manager.available_detectors:
                model_manager.update_detector_config(detector_name, config)
        
        return {
            "message": f"Preset '{preset_name}' applied successfully",
            "preset_name": preset_name,
            "applied_config": preset_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to apply detector preset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply detector preset: {str(e)}")

# Helper functions
def get_detector_description(detector_name: str) -> str:
    """Get description for a detector"""
    descriptions = {
        "toxicity": "Detects toxic, harmful, or inappropriate content using transformer models",
        "pii": "Identifies personally identifiable information like names, emails, phone numbers",
        "prompt_injection": "Detects attempts to manipulate or inject malicious prompts",
        "topic": "Classifies content topics and identifies restricted subject matter",
        "fact_check": "Identifies claims that may need fact-checking or verification",
        "spam": "Detects spam, promotional content, and low-quality messages"
    }
    return descriptions.get(detector_name, "No description available")

detector_router = router