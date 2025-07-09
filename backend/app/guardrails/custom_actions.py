from typing import Optional, Dict, Any
from nemoguardrails.actions import action
from app.utils.logger import app_logger

# Global model manager instance (will be set by guardrails_manager)
_model_manager = None

def set_model_manager(manager):
    global _model_manager
    _model_manager = manager

@action(is_system_action=True)
async def check_toxicity(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Check if text contains toxic content"""
    try:
        if not _model_manager:
            return {"is_toxic": False, "error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"is_toxic": False, "confidence": 0.0}
        
        result = await _model_manager.detect(text, "toxicity")
        
        app_logger.info(f"Toxicity check result: {result['result']['is_toxic']}")
        
        return {
            "is_toxic": result["result"]["is_toxic"],
            "confidence": result["result"]["confidence"],
            "scores": result["result"]["scores"]
        }
        
    except Exception as e:
        app_logger.error(f"Toxicity check failed: {e}")
        return {"is_toxic": False, "error": str(e)}

@action(is_system_action=True)
async def check_pii(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Check if text contains personally identifiable information"""
    try:
        if not _model_manager:
            return {"has_pii": False, "error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"has_pii": False, "entities": []}
        
        result = await _model_manager.detect(text, "pii")
        
        app_logger.info(f"PII check result: {result['result']['has_pii']}")
        
        return {
            "has_pii": result["result"]["has_pii"],
            "entities": result["result"]["entities"],
            "patterns": result["result"]["patterns"]
        }
        
    except Exception as e:
        app_logger.error(f"PII check failed: {e}")
        return {"has_pii": False, "error": str(e)}

@action(is_system_action=True)
async def check_prompt_injection(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Check if text contains prompt injection attempts"""
    try:
        if not _model_manager:
            return {"is_injection": False, "error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"is_injection": False, "confidence": 0.0}
        
        result = await _model_manager.detect(text, "prompt_injection")
        
        app_logger.info(f"Prompt injection check result: {result['result']['is_injection']}")
        
        return {
            "is_injection": result["result"]["is_injection"],
            "confidence": result["result"]["confidence"],
            "matched_patterns": result["result"]["matched_patterns"]
        }
        
    except Exception as e:
        app_logger.error(f"Prompt injection check failed: {e}")
        return {"is_injection": False, "error": str(e)}

@action(is_system_action=True)
async def check_topics(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Check if text contains restricted topics"""
    try:
        if not _model_manager:
            return {"has_restricted_topic": False, "error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"has_restricted_topic": False, "similarities": {}}
        
        result = await _model_manager.detect(text, "topic")
        
        app_logger.info(f"Topic check result: {result['result']['has_restricted_topic']}")
        
        return {
            "has_restricted_topic": result["result"]["has_restricted_topic"],
            "similarities": result["result"]["similarities"],
            "max_similarity": result["result"]["max_similarity"]
        }
        
    except Exception as e:
        app_logger.error(f"Topic check failed: {e}")
        return {"has_restricted_topic": False, "error": str(e)}

@action(is_system_action=True)
async def check_facts(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Check if text needs fact-checking"""
    try:
        if not _model_manager:
            return {"needs_fact_check": False, "error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"needs_fact_check": False, "confidence": 0.0}
        
        result = await _model_manager.detect(text, "fact_check")
        
        app_logger.info(f"Fact check result: {result['result']['needs_fact_check']}")
        
        return {
            "needs_fact_check": result["result"]["needs_fact_check"],
            "confidence": result["result"]["confidence"],
            "factual_claims": result["result"]["factual_claims"]
        }
        
    except Exception as e:
        app_logger.error(f"Fact check failed: {e}")
        return {"needs_fact_check": False, "error": str(e)}

@action(is_system_action=True)
async def check_spam(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Check if text is spam"""
    try:
        if not _model_manager:
            return {"is_spam": False, "error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"is_spam": False, "confidence": 0.0}
        
        result = await _model_manager.detect(text, "spam")
        
        app_logger.info(f"Spam check result: {result['result']['is_spam']}")
        
        return {
            "is_spam": result["result"]["is_spam"],
            "confidence": result["result"]["confidence"],
            "spam_indicators": result["result"]["spam_indicators"]
        }
        
    except Exception as e:
        app_logger.error(f"Spam check failed: {e}")
        return {"is_spam": False, "error": str(e)}

@action(is_system_action=True)
async def check_all_detectors(context: Optional[Dict[str, Any]] = None, text: str = None) -> Dict[str, Any]:
    """Run all available detectors on the text"""
    try:
        if not _model_manager:
            return {"error": "Model manager not initialized"}
        
        if not text:
            text = context.get("user_message", "") if context else ""
        
        if not text:
            return {"results": {}, "blocked": False}
        
        # Run all detectors
        results = await _model_manager.detect_all(text, active_only=True)
        
        # Determine if message should be blocked
        blocked = False
        blocking_reasons = []
        
        for detector_name, detector_result in results.items():
            if detector_result.get("error"):
                continue
                
            result = detector_result.get("result", {})
            
            # Check blocking conditions
            if detector_name == "toxicity" and result.get("is_toxic", False):
                blocked = True
                blocking_reasons.append("toxic content detected")
            elif detector_name == "pii" and result.get("has_pii", False):
                blocked = True
                blocking_reasons.append("PII detected")
            elif detector_name == "prompt_injection" and result.get("is_injection", False):
                blocked = True
                blocking_reasons.append("prompt injection detected")
            elif detector_name == "topic" and result.get("has_restricted_topic", False):
                blocked = True
                blocking_reasons.append("restricted topic detected")
            elif detector_name == "spam" and result.get("is_spam", False):
                blocked = True
                blocking_reasons.append("spam detected")
        
        app_logger.info(f"All detectors check - Blocked: {blocked}, Reasons: {blocking_reasons}")
        
        return {
            "results": results,
            "blocked": blocked,
            "blocking_reasons": blocking_reasons
        }
        
    except Exception as e:
        app_logger.error(f"All detectors check failed: {e}")
        return {"error": str(e), "blocked": False}

@action(is_system_action=True)
async def generate_with_ollama(context: Optional[Dict[str, Any]] = None, prompt: str = None) -> str:
    """Generate response using Ollama"""
    try:
        if not _model_manager:
            return "Error: Model manager not initialized"
        
        if not prompt:
            prompt = context.get("user_message", "") if context else ""
        
        if not prompt:
            return "Error: No prompt provided"
        
        # Generate response using Ollama
        response = await _model_manager.ollama_client.generate(prompt)
        
        return response.get("response", "No response generated")
        
    except Exception as e:
        app_logger.error(f"Ollama generation failed: {e}")
        return f"Error generating response: {str(e)}"

@action(is_system_action=True)
async def chat_with_ollama(context: Optional[Dict[str, Any]] = None, messages: list = None) -> str:
    """Chat with Ollama using message history"""
    try:
        if not _model_manager:
            return "Error: Model manager not initialized"
        
        if not messages:
            # Convert single message to chat format
            user_message = context.get("user_message", "") if context else ""
            if user_message:
                messages = [{"role": "user", "content": user_message}]
            else:
                return "Error: No messages provided"
        
        # Chat with Ollama
        response = await _model_manager.ollama_client.chat(messages)
        
        return response.get("message", {}).get("content", "No response generated")
        
    except Exception as e:
        app_logger.error(f"Ollama chat failed: {e}")
        return f"Error in chat: {str(e)}"