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

    # Validate project section
    if "project" in config:
        project_config = config["project"]
        if not isinstance(project_config, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Invalid project configuration: must be a dictionary"
            )
        validated["project"].update(project_config)

    # Validate release section
    if "release" in config:
        release_config = config["release"]
        if not isinstance(release_config, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Invalid release configuration: must be a dictionary"
            )
        validated["release"].update(release_config)

    # Validate build section
    if "build" in config:
        build_config = config["build"]
        if not isinstance(build_config, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Invalid build configuration: must be a dictionary"
            )
        validated["build"].update(build_config)

    # Validate publish section
    if "publish" in config:
        publish_config = config["publish"]
        if not isinstance(publish_config, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Invalid publish configuration: must be a dictionary"
            )
        validated["publish"].update(publish_config)

    # Validate AI section
    if "ai" in config:
        ai_config = config["ai"]
        if not isinstance(ai_config, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Invalid AI configuration: must be a dictionary"
            )
        validated["ai"].update(ai_config)

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
    def project_name(self) -> str:
        """Get the project name."""
        return self.data["project"]["name"]

    @property
    def project_version(self) -> str:
        """Get the project version."""
        return self.data["project"]["version"]

    @property
    def release_branch(self) -> str:
        """Get the release branch name."""
        return self.data["release"]["branch"]

    @property
    def tag_prefix(self) -> str:
        """Get the release tag prefix."""
        return self.data["release"]["tag_prefix"]

    @property
    def changelog_file(self) -> str:
        """Get the changelog file path."""
        return self.data["release"]["changelog"]

    @property
    def clean_build(self) -> bool:
        """Get whether to clean before building."""
        return self.data["build"]["clean"]

    @property
    def run_tests(self) -> bool:
        """Get whether to run tests."""
        return self.data["build"]["test"]

    @property
    def build_docs(self) -> bool:
        """Get whether to build documentation."""
        return self.data["build"]["docs"]

    @property
    def publish_repository(self) -> str:
        """Get the publish repository."""
        return self.data["publish"]["repository"]

    @property
    def skip_existing(self) -> bool:
        """Get whether to skip existing versions."""
        return self.data["publish"]["skip_existing"]

    @property
    def model(self) -> str:
        """Get the AI model name."""
        return self.data["ai"]["model"]

    @property
    def api_key(self) -> Optional[str]:
        """Get the OpenAI API key."""
        return self.data["ai"]["api_key"]