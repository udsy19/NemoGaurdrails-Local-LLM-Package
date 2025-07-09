from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

from app.utils.logger import app_logger

router = APIRouter()

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    detector_config: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    blocked: bool
    blocking_reasons: List[str] = []
    warnings: List[str] = []
    session_id: str
    message_id: str
    timestamp: str
    input_analysis: Optional[Dict[str, Any]] = None
    output_analysis: Optional[Dict[str, Any]] = None

class ChatHistoryRequest(BaseModel):
    session_id: str
    limit: int = Field(50, ge=1, le=200)

class DetectorConfigUpdate(BaseModel):
    session_id: str
    detector_config: Dict[str, Any]

class ExportRequest(BaseModel):
    session_id: str
    format: str = Field("json", pattern="^(json|text)$")

# Dependency to get chat service
def get_chat_service():
    from app.main import app
    return app.state.chat_service

@router.post("/message", response_model=ChatResponse)
async def send_message(
    message_data: ChatMessage,
    chat_service = Depends(get_chat_service)
):
    """Send a message and get AI response through safety pipeline"""
    try:
        app_logger.info(f"Received chat message: {message_data.message[:100]}...")
        
        result = await chat_service.process_message(
            message=message_data.message,
            detector_config=message_data.detector_config,
            session_id=message_data.session_id,
            context=message_data.context
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        app_logger.error(f"Failed to process chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@router.get("/history")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    chat_service = Depends(get_chat_service)
):
    """Get chat history for a session"""
    try:
        history = await chat_service.get_chat_history(session_id, limit)
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.delete("/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    chat_service = Depends(get_chat_service)
):
    """Clear chat history for a session"""
    try:
        success = await chat_service.clear_chat_history(session_id)
        
        if success:
            return {"message": f"Chat history cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear chat history")
        
    except Exception as e:
        app_logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@router.post("/config")
async def update_detector_config(
    config_data: DetectorConfigUpdate,
    chat_service = Depends(get_chat_service)
):
    """Update detector configuration for a session"""
    try:
        success = await chat_service.update_detector_config(
            config_data.session_id,
            config_data.detector_config
        )
        
        if success:
            return {
                "message": f"Detector configuration updated for session {config_data.session_id}",
                "config": config_data.detector_config
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update detector configuration")
        
    except Exception as e:
        app_logger.error(f"Failed to update detector config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update detector config: {str(e)}")

@router.get("/config/{session_id}")
async def get_session_config(
    session_id: str,
    chat_service = Depends(get_chat_service)
):
    """Get detector configuration for a session"""
    try:
        config = chat_service.get_session_config(session_id)
        
        return {
            "session_id": session_id,
            "config": config
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get session config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session config: {str(e)}")

@router.get("/sessions")
async def get_active_sessions(
    chat_service = Depends(get_chat_service)
):
    """Get list of active chat sessions"""
    try:
        sessions = chat_service.get_active_sessions()
        
        return {
            "active_sessions": sessions,
            "session_count": len(sessions)
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get active sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")

@router.get("/stats/{session_id}")
async def get_session_stats(
    session_id: str,
    chat_service = Depends(get_chat_service)
):
    """Get statistics for a chat session"""
    try:
        stats = await chat_service.get_session_stats(session_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to get session stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session stats: {str(e)}")

@router.post("/export")
async def export_chat_history(
    export_data: ExportRequest,
    chat_service = Depends(get_chat_service)
):
    """Export chat history in specified format"""
    try:
        result = await chat_service.export_chat_history(
            export_data.session_id,
            export_data.format
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to export chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export chat history: {str(e)}")

@router.get("/health")
async def get_chat_health(
    chat_service = Depends(get_chat_service)
):
    """Get health status of chat service"""
    try:
        health_status = chat_service.get_health_status()
        
        return {
            "status": "healthy" if health_status["initialized"] else "unhealthy",
            "details": health_status
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get chat health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat health: {str(e)}")

@router.post("/test")
async def test_chat_pipeline(
    message_data: ChatMessage,
    background_tasks: BackgroundTasks,
    chat_service = Depends(get_chat_service)
):
    """Test the chat pipeline with a message (for debugging)"""
    try:
        # Process message in test mode
        result = await chat_service.process_message(
            message=f"[TEST] {message_data.message}",
            detector_config=message_data.detector_config,
            session_id=f"test_{uuid.uuid4()}",
            context={"test_mode": True}
        )
        
        # Add test metadata
        result["test_mode"] = True
        result["test_timestamp"] = datetime.now().isoformat()
        
        # Clean up test session in background
        background_tasks.add_task(
            chat_service.clear_chat_history,
            result["session_id"]
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        app_logger.error(f"Failed to test chat pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test chat pipeline: {str(e)}")

# Batch operations
@router.post("/batch")
async def process_batch_messages(
    messages: List[str],
    detector_config: Optional[Dict[str, Any]] = None,
    chat_service = Depends(get_chat_service)
):
    """Process multiple messages in batch"""
    try:
        if len(messages) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 messages per batch")
        
        results = []
        session_id = f"batch_{uuid.uuid4()}"
        
        for i, message in enumerate(messages):
            try:
                result = await chat_service.process_message(
                    message=message,
                    detector_config=detector_config,
                    session_id=session_id,
                    context={"batch_mode": True, "batch_index": i}
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "message_index": i,
                    "blocked": True,
                    "blocking_reasons": [f"Processing error: {str(e)}"]
                })
        
        return {
            "batch_id": session_id,
            "message_count": len(messages),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to process batch messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process batch messages: {str(e)}")

chat_router = router