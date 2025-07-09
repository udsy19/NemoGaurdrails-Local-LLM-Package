import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.utils.logger import app_logger
from app.utils.exceptions import ConfigurationException

class ConfigService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.config_file = Path("configs/app_config.yml")
        self.user_configs = {}
        self.system_config = {}
        
    async def initialize(self):
        """Initialize configuration service"""
        try:
            await self.load_system_config()
            app_logger.info("Configuration service initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize configuration service: {e}")
            raise ConfigurationException(f"Configuration service initialization failed: {e}")
    
    async def load_system_config(self):
        """Load system configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.system_config = yaml.safe_load(f)
                app_logger.info("System configuration loaded from file")
            else:
                self.system_config = self.get_default_system_config()
                await self.save_system_config()
                app_logger.info("Default system configuration created")
                
        except Exception as e:
            app_logger.error(f"Failed to load system config: {e}")
            raise ConfigurationException(f"Failed to load system config: {e}")
    
    def get_default_system_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            "app": {
                "name": "AI Safety System",
                "version": "1.0.0",
                "debug": False
            },
            "models": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3",
                    "timeout": 120
                },
                "huggingface": {
                    "cache_dir": "~/.cache/huggingface",
                    "device": "auto"
                }
            },
            "detectors": {
                "toxicity": {
                    "enabled": True,
                    "threshold": 0.7,
                    "model": "martin-ha/toxic-comment-model"
                },
                "pii": {
                    "enabled": True,
                    "sensitivity": 0.8,
                    "model": "en_core_web_sm"
                },
                "prompt_injection": {
                    "enabled": True,
                    "threshold": 0.5
                },
                "topic": {
                    "enabled": True,
                    "threshold": 0.7,
                    "model": "all-MiniLM-L6-v2"
                },
                "fact_check": {
                    "enabled": True,
                    "threshold": 0.5
                },
                "spam": {
                    "enabled": True,
                    "threshold": 0.6
                }
            },
            "guardrails": {
                "input_enabled": True,
                "output_enabled": True,
                "config_dir": "configs/guardrails"
            },
            "chat": {
                "max_history": 100,
                "max_message_length": 10000,
                "session_timeout": 3600
            },
            "api": {
                "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
                "rate_limit": {
                    "enabled": False,
                    "requests_per_minute": 60
                }
            }
        }
    
    async def save_system_config(self):
        """Save system configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                yaml.dump(self.system_config, f, default_flow_style=False, indent=2)
            
            app_logger.info("System configuration saved")
            
        except Exception as e:
            app_logger.error(f"Failed to save system config: {e}")
            raise ConfigurationException(f"Failed to save system config: {e}")
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get current system configuration"""
        return self.system_config.copy()
    
    async def update_system_config(self, updates: Dict[str, Any]) -> bool:
        """Update system configuration"""
        try:
            # Deep merge updates into system config
            self.deep_merge(self.system_config, updates)
            
            # Save updated configuration
            await self.save_system_config()
            
            # Apply configuration changes
            await self.apply_config_changes(updates)
            
            app_logger.info("System configuration updated successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to update system config: {e}")
            return False
    
    def deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source dictionary into target dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self.deep_merge(target[key], value)
            else:
                target[key] = value
    
    async def apply_config_changes(self, updates: Dict[str, Any]):
        """Apply configuration changes to running system"""
        try:
            # Update detector configurations
            if "detectors" in updates:
                detector_updates = updates["detectors"]
                
                for detector_name, config in detector_updates.items():
                    if detector_name in self.model_manager.available_detectors:
                        self.model_manager.update_detector_config(detector_name, config)
                        
                        # Enable/disable detectors based on config
                        if "enabled" in config:
                            active_detectors = self.model_manager.get_active_detectors()
                            
                            if config["enabled"] and detector_name not in active_detectors:
                                active_detectors.append(detector_name)
                            elif not config["enabled"] and detector_name in active_detectors:
                                active_detectors.remove(detector_name)
                            
                            self.model_manager.set_active_detectors(active_detectors)
            
            # Update Ollama configuration
            if "models" in updates and "ollama" in updates["models"]:
                ollama_config = updates["models"]["ollama"]
                
                if "default_model" in ollama_config:
                    self.model_manager.ollama_client.current_model = ollama_config["default_model"]
                
                if "base_url" in ollama_config:
                    # Note: Changing base_url requires restart
                    app_logger.warning("Ollama base_url changed, restart required for full effect")
            
            app_logger.info("Configuration changes applied successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to apply config changes: {e}")
            raise ConfigurationException(f"Failed to apply config changes: {e}")
    
    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific detector"""
        return self.system_config.get("detectors", {}).get(detector_name, {})
    
    async def update_detector_config(self, detector_name: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a specific detector"""
        try:
            if "detectors" not in self.system_config:
                self.system_config["detectors"] = {}
            
            if detector_name not in self.system_config["detectors"]:
                self.system_config["detectors"][detector_name] = {}
            
            self.system_config["detectors"][detector_name].update(config)
            
            await self.save_system_config()
            await self.apply_config_changes({"detectors": {detector_name: config}})
            
            app_logger.info(f"Detector configuration updated: {detector_name}")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to update detector config: {e}")
            return False
    
    def get_user_config(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific configuration"""
        return self.user_configs.get(user_id, {})
    
    def set_user_config(self, user_id: str, config: Dict[str, Any]):
        """Set user-specific configuration"""
        self.user_configs[user_id] = config
        app_logger.info(f"User configuration set for: {user_id}")
    
    def get_active_detectors_config(self) -> Dict[str, Any]:
        """Get configuration for currently active detectors"""
        active_detectors = self.model_manager.get_active_detectors()
        config = {}
        
        for detector_name in active_detectors:
            config[detector_name] = self.get_detector_config(detector_name)
        
        return config
    
    async def reset_to_defaults(self) -> bool:
        """Reset system configuration to defaults"""
        try:
            self.system_config = self.get_default_system_config()
            await self.save_system_config()
            await self.apply_config_changes(self.system_config)
            
            app_logger.info("System configuration reset to defaults")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def export_config(self, format: str = "yaml") -> Dict[str, Any]:
        """Export configuration in specified format"""
        try:
            config_data = {
                "system_config": self.system_config,
                "user_configs": self.user_configs,
                "export_timestamp": app_logger.info(),
                "format": format
            }
            
            if format == "yaml":
                return {
                    "content": yaml.dump(config_data, default_flow_style=False, indent=2),
                    "filename": "ai_safety_config.yml"
                }
            elif format == "json":
                return {
                    "content": json.dumps(config_data, indent=2),
                    "filename": "ai_safety_config.json"
                }
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            app_logger.error(f"Failed to export config: {e}")
            return {"error": str(e)}
    
    async def import_config(self, config_data: str, format: str = "yaml") -> bool:
        """Import configuration from data"""
        try:
            if format == "yaml":
                imported_config = yaml.safe_load(config_data)
            elif format == "json":
                imported_config = json.loads(config_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Validate imported configuration
            if not self.validate_config(imported_config):
                return False
            
            # Update system configuration
            if "system_config" in imported_config:
                self.system_config = imported_config["system_config"]
                await self.save_system_config()
                await self.apply_config_changes(self.system_config)
            
            # Update user configurations
            if "user_configs" in imported_config:
                self.user_configs.update(imported_config["user_configs"])
            
            app_logger.info("Configuration imported successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to import config: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        try:
            # Check required keys
            if "system_config" not in config:
                app_logger.error("Missing system_config in imported configuration")
                return False
            
            system_config = config["system_config"]
            
            # Validate system config structure
            required_sections = ["app", "models", "detectors", "guardrails", "chat", "api"]
            for section in required_sections:
                if section not in system_config:
                    app_logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate detector configurations
            detectors_config = system_config["detectors"]
            for detector_name, detector_config in detectors_config.items():
                if not isinstance(detector_config, dict):
                    app_logger.error(f"Invalid detector config for {detector_name}")
                    return False
                
                if "enabled" not in detector_config:
                    app_logger.error(f"Missing 'enabled' flag for detector {detector_name}")
                    return False
            
            return True
            
        except Exception as e:
            app_logger.error(f"Config validation failed: {e}")
            return False
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation"""
        return {
            "type": "object",
            "properties": {
                "system_config": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "object"},
                        "models": {"type": "object"},
                        "detectors": {"type": "object"},
                        "guardrails": {"type": "object"},
                        "chat": {"type": "object"},
                        "api": {"type": "object"}
                    },
                    "required": ["app", "models", "detectors", "guardrails", "chat", "api"]
                },
                "user_configs": {"type": "object"}
            },
            "required": ["system_config"]
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of configuration service"""
        return {
            "config_loaded": bool(self.system_config),
            "config_file_exists": self.config_file.exists(),
            "active_detectors": len(self.model_manager.get_active_detectors()),
            "user_configs": len(self.user_configs)
        }