"""Configuration management for omaster."""
from pathlib import Path
import yaml
from .errors import ErrorCode, ReleaseError

VALID_MODELS = ["gpt-4o", "gpt-4o-mini"]
DEFAULT_CONFIG = {
    "ai": {
        "model": "gpt-4o-mini"  # Default to the smaller model
    }
}

class Config:
    """Configuration manager for omaster."""
    
    def __init__(self, project_path: Path):
        """Initialize configuration.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = project_path
        self.config_path = project_path / ".omaster.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load and validate configuration.
        
        Returns:
            dict: Validated configuration
            
        Raises:
            ReleaseError: If configuration is invalid
        """
        # Start with default config
        config = DEFAULT_CONFIG.copy()
        
        # Load user config if exists
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    user_config = yaml.safe_load(f)
                if user_config:
                    # Merge user config with defaults
                    if "ai" in user_config:
                        config["ai"].update(user_config["ai"])
            except Exception as e:
                raise ReleaseError(
                    ErrorCode.CONFIG_ERROR,
                    f"Failed to load .omaster.yaml: {str(e)}"
                )
        
        # Validate model
        model = config["ai"]["model"]
        if model not in VALID_MODELS:
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                f"Invalid model: {model}. Must be one of: {', '.join(VALID_MODELS)}"
            )
            
        return config
    
    @property
    def model(self) -> str:
        """Get the configured AI model."""
        return self.config["ai"]["model"] 