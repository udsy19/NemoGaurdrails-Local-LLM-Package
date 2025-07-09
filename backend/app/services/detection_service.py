import asyncio
from typing import Dict, Any, List, Optional
from app.utils.logger import app_logger
from app.utils.exceptions import DetectionException

class DetectionService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
    async def run_detection(
        self,
        text: str,
        detector_names: Optional[List[str]] = None,
        detector_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run detection on text with specified detectors"""
        try:
            if not text.strip():
                return {
                    "results": {},
                    "blocked": False,
                    "blocking_reasons": [],
                    "summary": "Empty input"
                }
            
            # Use active detectors if none specified
            if detector_names is None:
                detector_names = self.model_manager.get_active_detectors()
            
            # Apply detector configuration if provided
            if detector_config:
                for detector_name, config in detector_config.items():
                    if detector_name in detector_names:
                        self.model_manager.update_detector_config(detector_name, config)
            
            # Set active detectors to only the requested ones
            self.model_manager.set_active_detectors(detector_names)
            
            # Run detections only on active detectors
            results = await self.model_manager.detect_all(text, active_only=True)
            
            # Results are already filtered to requested detectors
            filtered_results = results
            
            # Analyze results and determine blocking
            analysis = self.analyze_results(filtered_results)
            
            return {
                "results": filtered_results,
                "blocked": analysis["blocked"],
                "blocking_reasons": analysis["blocking_reasons"],
                "summary": analysis["summary"],
                "text_length": len(text),
                "detectors_used": detector_names
            }
            
        except Exception as e:
            app_logger.error(f"Detection service failed: {e}")
            raise DetectionException(f"Detection service failed: {e}")
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detection results and determine blocking status"""
        try:
            blocked = False
            blocking_reasons = []
            warnings = []
            scores = {}
            
            for detector_name, detector_result in results.items():
                if detector_result.get("error"):
                    warnings.append(f"{detector_name}: {detector_result['error']}")
                    continue
                
                result = detector_result.get("result", {})
                
                # Analyze each detector type
                if detector_name == "toxicity":
                    is_toxic = result.get("is_toxic", False)
                    confidence = result.get("confidence", 0.0)
                    scores["toxicity"] = confidence
                    
                    if is_toxic:
                        blocked = True
                        blocking_reasons.append(f"Toxic content detected (confidence: {confidence:.2f})")
                    elif confidence > 0.5:
                        warnings.append(f"Potential toxicity detected (confidence: {confidence:.2f})")
                
                elif detector_name == "pii":
                    has_pii = result.get("has_pii", False)
                    entities = result.get("entities", [])
                    patterns = result.get("patterns", [])
                    
                    if has_pii:
                        blocked = True
                        pii_types = set()
                        for entity in entities:
                            pii_types.add(entity.get("label", "unknown"))
                        for pattern in patterns:
                            pii_types.add(pattern.get("type", "unknown"))
                        
                        blocking_reasons.append(f"PII detected: {', '.join(pii_types)}")
                    
                    scores["pii"] = 1.0 if has_pii else 0.0
                
                elif detector_name == "prompt_injection":
                    is_injection = result.get("is_injection", False)
                    confidence = result.get("confidence", 0.0)
                    scores["prompt_injection"] = confidence
                    
                    if is_injection:
                        blocked = True
                        blocking_reasons.append(f"Prompt injection detected (confidence: {confidence:.2f})")
                    elif confidence > 0.3:
                        warnings.append(f"Potential prompt injection (confidence: {confidence:.2f})")
                
                elif detector_name == "topic":
                    has_restricted_topic = result.get("has_restricted_topic", False)
                    max_similarity = result.get("max_similarity", 0.0)
                    scores["topic"] = max_similarity
                    
                    if has_restricted_topic:
                        blocked = True
                        blocking_reasons.append(f"Restricted topic detected (similarity: {max_similarity:.2f})")
                    elif max_similarity > 0.5:
                        warnings.append(f"Potentially restricted topic (similarity: {max_similarity:.2f})")
                
                elif detector_name == "spam":
                    is_spam = result.get("is_spam", False)
                    confidence = result.get("confidence", 0.0)
                    scores["spam"] = confidence
                    
                    if is_spam:
                        blocked = True
                        blocking_reasons.append(f"Spam detected (confidence: {confidence:.2f})")
                    elif confidence > 0.4:
                        warnings.append(f"Potential spam (confidence: {confidence:.2f})")
                
                elif detector_name == "fact_check":
                    needs_fact_check = result.get("needs_fact_check", False)
                    confidence = result.get("confidence", 0.0)
                    scores["fact_check"] = confidence
                    
                    if needs_fact_check and confidence > 0.7:
                        warnings.append(f"Response may contain claims needing fact-checking")
            
            # Generate summary
            summary = self.generate_summary(blocked, blocking_reasons, warnings, scores)
            
            return {
                "blocked": blocked,
                "blocking_reasons": blocking_reasons,
                "warnings": warnings,
                "scores": scores,
                "summary": summary
            }
            
        except Exception as e:
            app_logger.error(f"Failed to analyze results: {e}")
            return {
                "blocked": False,
                "blocking_reasons": [],
                "warnings": [f"Analysis error: {str(e)}"],
                "scores": {},
                "summary": "Analysis failed"
            }
    
    def generate_summary(
        self,
        blocked: bool,
        blocking_reasons: List[str],
        warnings: List[str],
        scores: Dict[str, float]
    ) -> str:
        """Generate a human-readable summary of detection results"""
        try:
            if blocked:
                return f"⚠️ Content blocked: {'; '.join(blocking_reasons)}"
            
            if warnings:
                return f"⚡ Warnings: {'; '.join(warnings)}"
            
            # Calculate overall safety score
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                if avg_score < 0.2:
                    return "✅ Content appears safe"
                elif avg_score < 0.5:
                    return "🟡 Content has minor safety concerns"
                else:
                    return "🟠 Content has moderate safety concerns"
            
            return "✅ No safety issues detected"
            
        except Exception as e:
            app_logger.error(f"Failed to generate summary: {e}")
            return "❓ Unable to generate summary"
    
    async def test_detector(self, detector_name: str, test_text: str) -> Dict[str, Any]:
        """Test a specific detector with given text"""
        try:
            if detector_name not in self.model_manager.available_detectors:
                raise ValueError(f"Unknown detector: {detector_name}")
            
            result = await self.model_manager.detect(test_text, detector_name)
            
            # Add analysis for this single detector
            analysis = self.analyze_results({detector_name: result})
            
            return {
                "detector": detector_name,
                "test_text": test_text,
                "result": result,
                "analysis": analysis
            }
            
        except Exception as e:
            app_logger.error(f"Failed to test detector {detector_name}: {e}")
            raise DetectionException(f"Failed to test detector {detector_name}: {e}")
    
    async def batch_detection(self, texts: List[str], detector_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run detection on multiple texts"""
        try:
            tasks = []
            for i, text in enumerate(texts):
                task = asyncio.create_task(
                    self.run_detection(text, detector_names),
                    name=f"detect_batch_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "text_index": i,
                        "error": str(result),
                        "results": {},
                        "blocked": False,
                        "blocking_reasons": [f"Processing error: {str(result)}"]
                    })
                else:
                    result["text_index"] = i
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            app_logger.error(f"Batch detection failed: {e}")
            raise DetectionException(f"Batch detection failed: {e}")
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get statistics about detector usage and performance"""
        try:
            available_detectors = self.model_manager.get_available_detectors()
            active_detectors = self.model_manager.get_active_detectors()
            
            stats = {
                "total_detectors": len(available_detectors),
                "active_detectors": len(active_detectors),
                "available_detectors": available_detectors,
                "active_detector_names": active_detectors
            }
            
            return stats
            
        except Exception as e:
            app_logger.error(f"Failed to get detector stats: {e}")
            return {"error": str(e)}
    
    async def validate_input(self, text: str, max_length: int = 10000) -> Dict[str, Any]:
        """Validate input text before processing"""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check text length
            if len(text) > max_length:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Text too long: {len(text)} characters (max: {max_length})")
            
            # Check for empty text
            if not text.strip():
                validation_result["valid"] = False
                validation_result["errors"].append("Empty text provided")
            
            # Check for suspicious patterns (basic validation)
            if text.count('\n') > 100:
                validation_result["warnings"].append("Text contains many line breaks")
            
            if len(set(text)) < 10 and len(text) > 50:
                validation_result["warnings"].append("Text has low character diversity")
            
            return validation_result
            
        except Exception as e:
            app_logger.error(f"Input validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }