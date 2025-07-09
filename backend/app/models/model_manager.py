import asyncio
from typing import Dict, Any, Optional, List
from app.models.ollama_client import OllamaClient
from app.models.detector_models import (
    ToxicityDetector, PIIDetector, PromptInjectionDetector,
    TopicDetector, FactCheckDetector, SpamDetector
)
from app.utils.logger import app_logger
from app.utils.exceptions import ModelLoadingException

class ModelManager:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.detectors = {}
        self.is_initialized = False
        
        # Define available detectors
        self.available_detectors = {
            'toxicity': ToxicityDetector,
            'pii': PIIDetector,
            'prompt_injection': PromptInjectionDetector,
            'topic': TopicDetector,
            'fact_check': FactCheckDetector,
            'spam': SpamDetector
        }
        
        # Default detector configurations
        self.detector_configs = {
            'toxicity': {'threshold': 0.7},
            'pii': {'sensitivity': 0.8},
            'prompt_injection': {'threshold': 0.5},
            'topic': {'threshold': 0.7},
            'fact_check': {'threshold': 0.5},
            'spam': {'threshold': 0.6}
        }
        
        # Active detectors (can be modified at runtime)
        self.active_detectors = set()
        
    async def initialize(self, load_detectors: Optional[List[str]] = None):
        """Initialize the model manager and load specified detectors"""
        try:
            app_logger.info("Initializing Model Manager...")
            
            # Check Ollama connection
            if not await self.ollama_client.health_check():
                app_logger.warning("Ollama not available, some features may be limited")
            else:
                # Ensure Llama3.1 is available
                await self.ollama_client.ensure_model_available("llama3.1:8b")
                app_logger.info("Ollama client initialized successfully")
            
            # Load detectors
            detectors_to_load = load_detectors or list(self.available_detectors.keys())
            
            for detector_name in detectors_to_load:
                if detector_name in self.available_detectors:
                    await self.load_detector(detector_name)
            
            self.is_initialized = True
            app_logger.info("Model Manager initialized successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to initialize Model Manager: {e}")
            raise ModelLoadingException(f"Model Manager initialization failed: {e}")
    
    async def load_detector(self, detector_name: str):
        """Load a specific detector"""
        try:
            if detector_name not in self.available_detectors:
                raise ValueError(f"Unknown detector: {detector_name}")
            
            app_logger.info(f"Loading detector: {detector_name}")
            
            detector_class = self.available_detectors[detector_name]
            detector_instance = detector_class()
            
            # Load the detector model
            await detector_instance.load_model()
            
            self.detectors[detector_name] = detector_instance
            self.active_detectors.add(detector_name)
            
            app_logger.info(f"Detector {detector_name} loaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to load detector {detector_name}: {e}")
            raise ModelLoadingException(f"Failed to load detector {detector_name}: {e}")
    
    async def unload_detector(self, detector_name: str):
        """Unload a specific detector"""
        try:
            if detector_name in self.detectors:
                del self.detectors[detector_name]
                self.active_detectors.discard(detector_name)
                app_logger.info(f"Detector {detector_name} unloaded")
            
        except Exception as e:
            app_logger.error(f"Failed to unload detector {detector_name}: {e}")
    
    async def detect(self, text: str, detector_name: str, **kwargs) -> Dict[str, Any]:
        """Run detection with a specific detector"""
        try:
            if detector_name not in self.detectors:
                raise ValueError(f"Detector {detector_name} not loaded")
            
            detector = self.detectors[detector_name]
            
            # Merge default config with provided kwargs
            config = self.detector_configs.get(detector_name, {})
            config.update(kwargs)
            
            result = await detector.detect(text, **config)
            
            return {
                "detector": detector_name,
                "result": result,
                "text_length": len(text),
                "config": config
            }
            
        except Exception as e:
            app_logger.error(f"Detection failed for {detector_name}: {e}")
            raise
    
    async def detect_all(self, text: str, active_only: bool = True) -> Dict[str, Any]:
        """Run detection with all available detectors"""
        results = {}
        
        detectors_to_use = self.active_detectors if active_only else set(self.detectors.keys())
        
        # Run all detectors concurrently
        tasks = []
        for detector_name in detectors_to_use:
            if detector_name in self.detectors:
                task = asyncio.create_task(
                    self.detect(text, detector_name),
                    name=f"detect_{detector_name}"
                )
                tasks.append((detector_name, task))
        
        # Wait for all tasks to complete
        for detector_name, task in tasks:
            try:
                result = await task
                results[detector_name] = result
            except Exception as e:
                app_logger.error(f"Detection failed for {detector_name}: {e}")
                results[detector_name] = {
                    "detector": detector_name,
                    "error": str(e),
                    "result": None
                }
        
        return results
    
    def get_available_detectors(self) -> Dict[str, Any]:
        """Get information about available detectors"""
        return {
            detector_name: {
                "loaded": detector_name in self.detectors,
                "active": detector_name in self.active_detectors,
                "config": self.detector_configs.get(detector_name, {})
            }
            for detector_name in self.available_detectors
        }
    
    def get_active_detectors(self) -> List[str]:
        """Get list of active detector names"""
        return list(self.active_detectors)
    
    def set_active_detectors(self, detector_names: List[str]):
        """Set which detectors are active"""
        self.active_detectors = set(detector_names) & set(self.detectors.keys())
        app_logger.info(f"Active detectors updated: {list(self.active_detectors)}")
    
    def update_detector_config(self, detector_name: str, config: Dict[str, Any]):
        """Update configuration for a specific detector"""
        if detector_name in self.detector_configs:
            self.detector_configs[detector_name].update(config)
            app_logger.info(f"Updated config for {detector_name}: {config}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        ollama_healthy = await self.ollama_client.health_check()
        
        detector_status = {}
        for detector_name in self.available_detectors:
            detector_status[detector_name] = {
                "loaded": detector_name in self.detectors,
                "active": detector_name in self.active_detectors,
                "healthy": detector_name in self.detectors and self.detectors[detector_name].is_loaded
            }
        
        return {
            "initialized": self.is_initialized,
            "ollama_healthy": ollama_healthy,
            "total_detectors": len(self.available_detectors),
            "loaded_detectors": len(self.detectors),
            "active_detectors": len(self.active_detectors),
            "detector_status": detector_status
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            app_logger.info("Cleaning up Model Manager...")
            
            # Close Ollama client
            await self.ollama_client.close()
            
            # Clear detectors
            self.detectors.clear()
            self.active_detectors.clear()
            
            app_logger.info("Model Manager cleanup completed")
            
        except Exception as e:
            app_logger.error(f"Error during cleanup: {e}")
    
    async def reload_detector(self, detector_name: str):
        """Reload a specific detector"""
        try:
            if detector_name in self.detectors:
                await self.unload_detector(detector_name)
            
            await self.load_detector(detector_name)
            app_logger.info(f"Detector {detector_name} reloaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to reload detector {detector_name}: {e}")
            raise