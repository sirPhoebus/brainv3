import yaml
import os
from typing import Any, Dict

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    """
    if config_path is None:
        # Default path relative to project root
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "swarm_config.yaml")
    
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Global config instance
DEFAULT_CONFIG = load_config()
