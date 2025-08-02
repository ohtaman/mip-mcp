"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from ..models.config import Config


class ConfigManager:
    """Manages application configuration from files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to config directory or file. If None, uses default.
        """
        self.config_dir = self._resolve_config_path(config_path)
        self.config = self._load_config()
    
    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Resolve configuration directory path."""
        if config_path:
            path = Path(config_path)
            if path.is_file():
                return path.parent
            return path
        
        # Default: use config directory relative to this module
        return Path(__file__).parent.parent / "config"
    
    def _load_config(self) -> Config:
        """Load configuration from files and environment variables."""
        # Start with default config
        config_data = self._load_yaml("default.yaml")
        
        # Override with environment-specific config if exists
        env = os.getenv("ENVIRONMENT", "").lower()
        if env and env != "default":
            env_config = self._load_yaml(f"{env}.yaml")
            config_data = self._merge_dict(config_data, env_config)
        
        # Override with environment variables
        self._apply_env_vars(config_data)
        
        return Config.from_dict(config_data)
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            if filename == "default.yaml":
                raise FileNotFoundError(f"Default config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")
    
    def _merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_vars(self, config_data: Dict[str, Any]) -> None:
        """Apply environment variable overrides."""
        # Define environment variable mappings
        env_mappings = {
            "MIP_MCP_LOG_LEVEL": ("logging", "level"),
            "MIP_MCP_EXECUTOR_TIMEOUT": ("executor", "timeout"),
            "MIP_MCP_SOLVER_DEFAULT": ("solvers", "default"),
            "MIP_MCP_SOLVER_TIMEOUT": ("solvers", "timeout"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in config_data:
                    config_data[section] = {}
                
                # Convert string values to appropriate types
                if key in ["timeout"]:
                    try:
                        config_data[section][key] = int(value)
                    except ValueError:
                        pass  # Keep original value if conversion fails
                else:
                    config_data[section][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "logging.level")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config.model_dump()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_config(self) -> Config:
        """Get the complete configuration object."""
        return self.config