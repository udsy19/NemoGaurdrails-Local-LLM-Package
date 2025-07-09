import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.utils.logger import app_logger
from app.utils.exceptions import ConfigurationException

class ConfigLoader:
    def __init__(self, config_dir: str = "configs/guardrails"):
        self.config_dir = Path(config_dir)
        self.base_config = None
        self.detector_configs = {}
        self.active_detectors = set()
        
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base guardrails configuration"""
        try:
            config_path = self.config_dir / "base_config.yml"
            
            if not config_path.exists():
                app_logger.warning(f"Base config not found at {config_path}, using default")
                return self.get_default_base_config()
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.base_config = config
            app_logger.info("Base configuration loaded successfully")
            return config
            
        except Exception as e:
            app_logger.error(f"Failed to load base config: {e}")
            raise ConfigurationException(f"Failed to load base config: {e}")
    
    def load_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Load configuration for a specific detector"""
        try:
            config_path = self.config_dir / f"{detector_name}_config.yml"
            
            if not config_path.exists():
                app_logger.warning(f"Detector config not found at {config_path}, using default")
                return self.get_default_detector_config(detector_name)
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.detector_configs[detector_name] = config
            app_logger.info(f"Detector configuration loaded for {detector_name}")
            return config
            
        except Exception as e:
            app_logger.error(f"Failed to load detector config for {detector_name}: {e}")
            raise ConfigurationException(f"Failed to load detector config for {detector_name}: {e}")
    
    def generate_dynamic_config(self, active_detectors: List[str]) -> Dict[str, Any]:
        """Generate a dynamic configuration based on active detectors"""
        try:
            if not self.base_config:
                self.load_base_config()
            
            # Start with base config
            config = self.base_config.copy()
            
            # Load detector-specific configs
            for detector_name in active_detectors:
                if detector_name not in self.detector_configs:
                    self.load_detector_config(detector_name)
            
            # Merge detector configs into base config
            self.merge_detector_configs(config, active_detectors)
            
            self.active_detectors = set(active_detectors)
            app_logger.info(f"Dynamic configuration generated for detectors: {active_detectors}")
            
            return config
            
        except Exception as e:
            app_logger.error(f"Failed to generate dynamic config: {e}")
            raise ConfigurationException(f"Failed to generate dynamic config: {e}")
    
    def merge_detector_configs(self, base_config: Dict[str, Any], active_detectors: List[str]):
        """Merge detector-specific configurations into base config"""
        try:
            # Merge input rails
            if 'rails' not in base_config:
                base_config['rails'] = {'input': {'flows': []}, 'output': {'flows': []}}
            
            # Add detector flows to input rails
            for detector_name in active_detectors:
                detector_config = self.detector_configs.get(detector_name, {})
                
                if 'input_flows' in detector_config:
                    base_config['rails']['input']['flows'].extend(detector_config['input_flows'])
                
                if 'output_flows' in detector_config:
                    base_config['rails']['output']['flows'].extend(detector_config['output_flows'])
            
            # Add custom actions
            if 'actions' not in base_config:
                base_config['actions'] = []
            
            # Add detector actions
            for detector_name in active_detectors:
                detector_config = self.detector_configs.get(detector_name, {})
                
                if 'actions' in detector_config:
                    base_config['actions'].extend(detector_config['actions'])
            
        except Exception as e:
            app_logger.error(f"Failed to merge detector configs: {e}")
            raise ConfigurationException(f"Failed to merge detector configs: {e}")
    
    def get_default_base_config(self) -> Dict[str, Any]:
        """Get default base configuration"""
        return {
            'instructions': [
                {
                    'type': 'general',
                    'content': 'You are a helpful AI assistant with built-in safety guardrails.'
                }
            ],
            'sample_conversation': [
                {
                    'user': 'Hello!',
                    'assistant': 'Hello! How can I help you today?'
                }
            ],
            'rails': {
                'input': {
                    'flows': []
                },
                'output': {
                    'flows': []
                }
            },
            'actions': [
                {
                    'name': 'generate_with_ollama',
                    'type': 'system'
                },
                {
                    'name': 'chat_with_ollama',
                    'type': 'system'
                }
            ]
        }
    
    def get_default_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get default configuration for a detector"""
        configs = {
            'toxicity': {
                'input_flows': [
                    'check toxicity'
                ],
                'output_flows': [
                    'check output toxicity'
                ],
                'actions': [
                    {
                        'name': 'check_toxicity',
                        'type': 'system'
                    }
                ]
            },
            'pii': {
                'input_flows': [
                    'check pii'
                ],
                'actions': [
                    {
                        'name': 'check_pii',
                        'type': 'system'
                    }
                ]
            },
            'prompt_injection': {
                'input_flows': [
                    'check prompt injection'
                ],
                'actions': [
                    {
                        'name': 'check_prompt_injection',
                        'type': 'system'
                    }
                ]
            },
            'topic': {
                'input_flows': [
                    'check topics'
                ],
                'actions': [
                    {
                        'name': 'check_topics',
                        'type': 'system'
                    }
                ]
            },
            'fact_check': {
                'output_flows': [
                    'check facts'
                ],
                'actions': [
                    {
                        'name': 'check_facts',
                        'type': 'system'
                    }
                ]
            },
            'spam': {
                'input_flows': [
                    'check spam'
                ],
                'actions': [
                    {
                        'name': 'check_spam',
                        'type': 'system'
                    }
                ]
            }
        }
        
        return configs.get(detector_name, {})
    
    def save_config(self, config: Dict[str, Any], config_name: str):
        """Save configuration to file"""
        try:
            config_path = self.config_dir / f"{config_name}.yml"
            
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            app_logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            app_logger.error(f"Failed to save config {config_name}: {e}")
            raise ConfigurationException(f"Failed to save config {config_name}: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        try:
            required_keys = ['instructions', 'rails', 'actions']
            
            for key in required_keys:
                if key not in config:
                    app_logger.error(f"Missing required key in config: {key}")
                    return False
            
            # Validate rails structure
            if 'input' not in config['rails'] or 'output' not in config['rails']:
                app_logger.error("Invalid rails structure")
                return False
            
            return True
            
        except Exception as e:
            app_logger.error(f"Config validation failed: {e}")
            return False
    
    def get_active_detectors(self) -> List[str]:
        """Get currently active detectors"""
        return list(self.active_detectors)
    
    def set_active_detectors(self, detectors: List[str]):
        """Set active detectors"""
        self.active_detectors = set(detectors)
        app_logger.info(f"Active detectors set to: {detectors}")
    
    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific detector"""
        return self.detector_configs.get(detector_name, {})
    
    def update_detector_config(self, detector_name: str, config: Dict[str, Any]):
        """Update configuration for a specific detector"""
        if detector_name not in self.detector_configs:
            self.detector_configs[detector_name] = {}
        
        self.detector_configs[detector_name].update(config)
        app_logger.info(f"Updated configuration for detector {detector_name}")
    
    def reload_all_configs(self):
        """Reload all configurations from files"""
        try:
            self.base_config = None
            self.detector_configs.clear()
            
            self.load_base_config()
            
            # Reload active detector configs
            for detector_name in self.active_detectors:
                self.load_detector_config(detector_name)
            
            app_logger.info("All configurations reloaded successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to reload configs: {e}")
            raise ConfigurationException(f"Failed to reload configs: {e}")