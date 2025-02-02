"""Configuration management for the release process."""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from .errors import ReleaseError, ErrorCode

# Default configuration values
DEFAULT_CONFIG = {
    "project": {
        "name": "",
        "version": "0.1.0",
        "description": "",
        "author": "",
        "email": "",
        "url": ""
    },
    "release": {
        "branch": "main",
        "tag_prefix": "v",
        "changelog": "CHANGELOG.md"
    },
    "build": {
        "clean": True,
        "test": True,
        "docs": True
    },
    "publish": {
        "repository": "pypi",
        "skip_existing": False
    },
    "ai": {
        "model": "gpt-4",
        "api_key": None
    },
    "github": {
        "org": None,
        "repo": None,
        "private": False
    },
    "quality": {
        "max_complexity": 15,
        "max_duplicates": 1.0
    }
}

def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from .omaster.yaml file.

    Args:
        path: Path to the project directory

    Returns:
        Configuration dictionary

    Raises:
        ReleaseError: If configuration is invalid
    """
    config_file = path / ".omaster.yaml"
    if not config_file.exists():
        raise ReleaseError(
            ErrorCode.CONFIG_ERROR,
            f"Configuration file not found: {config_file}"
        )

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ReleaseError(
            ErrorCode.CONFIG_ERROR,
            f"Failed to load configuration: {e}"
        )

    # Validate and merge with defaults
    return validate_config(config)

def _validate_section(config: Dict[str, Any], section: str, validated: Dict[str, Any]) -> None:
    """Validate a configuration section and update the validated config.

    Args:
        config: Input configuration dictionary
        section: Name of the section to validate
        validated: Validated configuration to update

    Raises:
        ReleaseError: If section configuration is invalid
    """
    if section in config:
        section_config = config[section]
        if not isinstance(section_config, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                f"Invalid {section} configuration: must be a dictionary"
            )
        validated[section].update(section_config)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        Validated configuration dictionary

    Raises:
        ReleaseError: If configuration is invalid
    """
    # Start with default config
    validated = DEFAULT_CONFIG.copy()

    # Validate each section
    sections = ["project", "release", "build", "publish", "ai", "github"]
    for section in sections:
        _validate_section(config, section, validated)

    return validated

class Config:
    """Configuration manager."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize configuration.

        Args:
            data: Configuration dictionary
        """
        self.data = data

    @property
    def model(self) -> str:
        """Get the AI model name."""
        return self.data["ai"]["model"]

    @property
    def api_key(self) -> Optional[str]:
        """Get the OpenAI API key."""
        return self.data["ai"]["api_key"]

    @property
    def github_org(self) -> Optional[str]:
        """Get the GitHub organization name."""
        return self.data["github"]["org"]

    @property
    def github_repo(self) -> Optional[str]:
        """Get the GitHub repository name."""
        return self.data["github"]["repo"]

    @property
    def github_private(self) -> bool:
        """Get whether the GitHub repository should be private."""
        return self.data["github"]["private"]