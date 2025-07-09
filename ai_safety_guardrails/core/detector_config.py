"""
Detector configuration class.

Provides a clean interface for configuring individual detectors.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..utils.exceptions import ConfigurationException
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectorConfig:
    """
    Configuration for an individual detector.

    Supports both simple string initialization and detailed configuration.
    """

    name: str
    enabled: bool = True
    threshold: Optional[float] = None
    sensitivity: Optional[str] = None
    model: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sensitivity and self.sensitivity not in ["low", "medium", "high"]:
            raise ConfigurationException(
                f"Invalid sensitivity '{self.sensitivity}'. Must be 'low', 'medium', or 'high'"
            )

        if self.threshold is not None and not (0.0 <= self.threshold <= 1.0):
            raise ConfigurationException(
                f"Invalid threshold {self.threshold}. Must be between 0.0 and 1.0"
            )

    @classmethod
    def from_dict(cls, name: str, config_dict: Dict[str, Any]) -> "DetectorConfig":
        """Create DetectorConfig from dictionary."""
        return cls(
            name=name,
            enabled=config_dict.get("enabled", True),
            threshold=config_dict.get("threshold"),
            sensitivity=config_dict.get("sensitivity"),
            model=config_dict.get("model"),
            config=config_dict.get("config", {}),
        )

    @classmethod
    def from_string(cls, detector_spec: str) -> "DetectorConfig":
        """
        Create DetectorConfig from string specification.

        Examples:
            "toxicity"
            "toxicity:0.8"
            "pii:high"
            "toxicity:0.7:martin-ha/toxic-comment-model"
        """
        parts = detector_spec.split(":")

        if len(parts) == 1:
            return cls(name=parts[0])
        elif len(parts) == 2:
            name, param = parts
            # Try to parse as float (threshold) or string (sensitivity)
            try:
                threshold = float(param)
                return cls(name=name, threshold=threshold)
            except ValueError:
                return cls(name=name, sensitivity=param)
        elif len(parts) == 3:
            name, threshold_str, model = parts
            try:
                threshold = float(threshold_str)
                return cls(name=name, threshold=threshold, model=model)
            except ValueError:
                return cls(name=name, sensitivity=threshold_str, model=model)
        else:
            raise ConfigurationException(
                f"Invalid detector specification: {detector_spec}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"enabled": self.enabled, "config": self.config}

        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.sensitivity is not None:
            result["sensitivity"] = self.sensitivity
        if self.model is not None:
            result["model"] = self.model

        return result

    def merge_with_global_config(self, global_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge detector config with global configuration."""
        merged = {}

        # Start with global config for this detector
        if self.name in global_config.get("detectors", {}):
            merged.update(global_config["detectors"][self.name])

        # Override with detector-specific config
        detector_dict = self.to_dict()
        merged.update(detector_dict)

        # Add detector-specific values
        if self.threshold is not None:
            merged["threshold"] = self.threshold
        if self.sensitivity is not None:
            merged["sensitivity"] = self.sensitivity
        if self.model is not None:
            merged["model"] = self.model

        return merged

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with fallback."""
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return f"DetectorConfig(name='{self.name}', enabled={self.enabled}, threshold={self.threshold})"


class SafetyConfig:
    """
    Global configuration for the safety system.

    Manages loading from YAML/JSON files and provides access to
    detector configurations and global settings.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary. If None, uses defaults.
        """
        self.config = config_dict or self._get_default_config()

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "SafetyConfig":
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationException(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                else:
                    raise ConfigurationException(
                        f"Unsupported config file format: {config_path.suffix}"
                    )

            return cls(config_dict)

        except Exception as e:
            raise ConfigurationException(
                f"Failed to load config from {config_path}: {e}"
            )

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "SafetyConfig":
        """Load configuration from YAML string."""
        try:
            config_dict = yaml.safe_load(yaml_string)
            return cls(config_dict)
        except Exception as e:
            raise ConfigurationException(f"Failed to parse YAML config: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "detectors": {
                "toxicity": {
                    "enabled": True,
                    "threshold": 0.7,
                    "model": "martin-ha/toxic-comment-model",
                },
                "pii": {
                    "enabled": True,
                    "sensitivity": "medium",
                    "model": "en_core_web_sm",
                },
                "prompt_injection": {"enabled": True, "threshold": 0.5},
                "topics": {
                    "enabled": True,
                    "threshold": 0.7,
                    "model": "all-MiniLM-L6-v2",
                },
                "fact_check": {"enabled": False, "threshold": 0.5},
                "spam": {"enabled": True, "threshold": 0.6},
            },
            "models": {
                "cache_dir": "~/.ai_safety_models",
                "auto_download": True,
                "download_timeout": 300,
            },
            "safety": {
                "fail_mode": "open",  # "open" or "closed"
                "max_concurrent_detections": 5,
                "detection_timeout": 30,
            },
            "logging": {"level": "INFO", "file": None},
        }

    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific detector."""
        return self.config.get("detectors", {}).get(detector_name, {})

    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.config.get("models", {})

    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety configuration."""
        return self.config.get("safety", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                elif config_path.suffix.lower() == ".json":
                    json.dump(self.config, f, indent=2)
                else:
                    raise ConfigurationException(
                        f"Unsupported config file format: {config_path.suffix}"
                    )

        except Exception as e:
            raise ConfigurationException(f"Failed to save config to {config_path}: {e}")

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate detector configurations
        for detector_name, detector_config in self.config.get("detectors", {}).items():
            if not isinstance(detector_config, dict):
                issues.append(f"Detector '{detector_name}' config must be a dictionary")
                continue

            # Check threshold values
            if "threshold" in detector_config:
                threshold = detector_config["threshold"]
                if not isinstance(threshold, (int, float)) or not (
                    0.0 <= threshold <= 1.0
                ):
                    issues.append(
                        f"Detector '{detector_name}' threshold must be between 0.0 and 1.0"
                    )

            # Check sensitivity values
            if "sensitivity" in detector_config:
                sensitivity = detector_config["sensitivity"]
                if sensitivity not in ["low", "medium", "high"]:
                    issues.append(
                        f"Detector '{detector_name}' sensitivity must be 'low', 'medium', or 'high'"
                    )

        # Validate models configuration
        models_config = self.config.get("models", {})
        if "download_timeout" in models_config:
            timeout = models_config["download_timeout"]
            if not isinstance(timeout, int) or timeout <= 0:
                issues.append("Models download_timeout must be a positive integer")

        return issues
