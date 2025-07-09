# Simplified guardrails implementation for Python 3.13 compatibility
import asyncio
from typing import Dict, Any, List, Optional

class SimpleGuardrails:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.active_detectors = []
        
    async def initialize(self, active_detectors: Optional[List[str]] = None):
        if active_detectors is None:
            active_detectors = list(self.model_manager.available_detectors.keys())
        self.active_detectors = active_detectors
        
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Run all active detectors
        results = await self.model_manager.detect_all(message, active_only=True)
        
        # Check for blocking conditions
        blocked = False
        blocking_reasons = []
        
        for detector_name, result in results.items():
            if result.get("error"):
                continue
                
            detection_result = result.get("result", {})
            
            if detector_name == "toxicity" and detection_result.get("is_toxic", False):
                blocked = True
                blocking_reasons.append("Toxic content detected")
            elif detector_name == "pii" and detection_result.get("has_pii", False):
                blocked = True
                blocking_reasons.append("Personal information detected")
            elif detector_name == "prompt_injection" and detection_result.get("is_injection", False):
                blocked = True
                blocking_reasons.append("Prompt injection detected")
            elif detector_name == "topic" and detection_result.get("has_restricted_topic", False):
                blocked = True
                blocking_reasons.append("Restricted topic detected")
            elif detector_name == "spam" and detection_result.get("is_spam", False):
                blocked = True
                blocking_reasons.append("Spam detected")
        
        if blocked:
            return {
                "response": "I cannot process this message due to safety concerns.",
                "blocked": True,
                "blocking_reasons": blocking_reasons,
                "context": context or {}
            }
        
        # Generate response using Ollama
        try:
            ollama_response = await self.model_manager.ollama_client.generate(message)
            response_text = ollama_response.get("response", "I apologize, but I couldn't generate a response.")
        except Exception as e:
            response_text = f"Error generating response: {str(e)}"
        
        return {
            "response": response_text,
            "blocked": False,
            "blocking_reasons": [],
            "context": context or {}
        }
    
    async def update_active_detectors(self, detectors: List[str]):
        self.active_detectors = detectors
        
    def get_active_detectors(self) -> List[str]:
        return self.active_detectors.copy()
