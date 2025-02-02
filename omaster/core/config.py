"""Configuration management for omaster."""
from pathlib import Path
import os
import yaml
from typing import Dict, Any, Optional
from .errors import ErrorCode, ReleaseError
import logging

logger = logging.getLogger(__name__)

VALID_MODELS = ["gpt-4o", "gpt-4o-mini"]

# Default weights for different issue severities (0-1 scale)
DEFAULT_SEVERITY_WEIGHTS = {
    "critical": 1.0,    # Critical issues (security, crashes)
    "high": 0.8,        # High severity (complexity, performance)
    "medium": 0.5,      # Medium severity (maintainability, style)
    "low": 0.2,         # Low severity (minor style issues)
    "info": 0.1         # Informational issues
}

DEFAULT_CONFIG = {
    "ai": {
        "model": "gpt-4o-mini"  # Default to the smaller model
    },
    "quality": {
        # Complexity thresholds
        "complexity": {
            "max_cyclomatic": 15,
            "max_cognitive": 20,
            "min_maintainability": 65,
            "max_halstead_difficulty": 30,
            "min_halstead_language_level": 0.8,
            "max_bug_prediction": 0.4,
            "max_oop_complexity": 50,
            "weights": {
                "cyclomatic": 0.8,
                "cognitive": 0.7,
                "maintainability": 0.6,
                "halstead": 0.5,
                "oop": 0.4
            }
        },
        # Dead code thresholds
        "dead_code": {
            "unused_import_threshold": 0.2,
            "unused_variable_threshold": 0.3,
            "unused_function_threshold": 0.5,
            "unused_class_threshold": 0.6,
            "unreachable_code_threshold": 0.8,
            "weights": {
                "unused_import": 0.2,
                "unused_variable": 0.3,
                "unused_function": 0.5,
                "unused_class": 0.6,
                "unreachable_code": 0.8
            }
        },
        # Similarity thresholds
        "similarity": {
            "exact_match_threshold": 1.0,
            "ast_similarity_threshold": 0.7,
            "token_similarity_threshold": 0.8,
            "cfg_similarity_threshold": 0.6,
            "semantic_similarity_threshold": 0.85,
            "min_lines": 6,
            "weights": {
                "exact_match": 1.0,
                "ast_similarity": 0.8,
                "token_similarity": 0.6,
                "cfg_similarity": 0.7,
                "semantic_similarity": 0.5
            }
        },
        # Global severity weights
        "severity_weights": DEFAULT_SEVERITY_WEIGHTS.copy()
    }
}


class Config:
    """Configuration manager."""

    def __init__(self, config_data: Dict[str, Any]):
        """Initialize configuration.

        Args:
            config_data: Configuration data
        """
        self.data = config_data

    @classmethod
    def load(cls, project_path: Optional[Path] = None) -> 'Config':
        """Load configuration from .omaster.yaml file.

        Args:
            project_path: Optional path to project directory

        Returns:
            Config instance

        Raises:
            ReleaseError: If configuration is invalid
        """
        config_data = DEFAULT_CONFIG.copy()

        # Find config file
        if project_path is None:
            project_path = Path.cwd()

        config_path = project_path / ".omaster.yaml"
        logger.info(f"Looking for configuration at {config_path}")

        if config_path.exists():
            try:
                logger.info("Loading configuration from .omaster.yaml")
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                
                if user_config:
                    # Merge user config with defaults
                    cls._merge_configs(config_data, user_config)
                    logger.info("✓ Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load configuration: {str(e)}")
                raise ReleaseError(
                    ErrorCode.CONFIG_ERROR,
                    f"Failed to load configuration: {str(e)}"
                )
        else:
            logger.info("No .omaster.yaml found, using default configuration")

        # Validate configuration
        cls._validate_config(config_data)
        return cls(config_data)

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override config into base config.

        Args:
            base: Base configuration
            override: Override configuration
        """
        for key, value in override.items():
            if (
                key in base 
                and isinstance(base[key], dict) 
                and isinstance(value, dict)
            ):
                Config._merge_configs(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate

        Raises:
            ReleaseError: If configuration is invalid
        """
        # Validate AI model
        ai_config = config.get("ai", {})
        model = ai_config.get("model")
        if model and model not in VALID_MODELS:
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                f"Invalid AI model: {model}. Must be one of: {', '.join(VALID_MODELS)}"
            )

        # Validate quality thresholds
        quality_config = config.get("quality", {})
        for section in ["complexity", "dead_code", "similarity"]:
            if section in quality_config:
                section_config = quality_config[section]
                if not isinstance(section_config, dict):
                    raise ReleaseError(
                        ErrorCode.CONFIG_ERROR,
                        f"Invalid quality.{section} configuration: must be a dictionary"
                    )

                # Validate weights
                weights = section_config.get("weights", {})
                if not isinstance(weights, dict):
                    raise ReleaseError(
                        ErrorCode.CONFIG_ERROR,
                        f"Invalid quality.{section}.weights configuration: must be a dictionary"
                    )
                for weight_name, weight_value in weights.items():
                    if not isinstance(weight_value, (int, float)) or not 0 <= weight_value <= 1:
                        raise ReleaseError(
                            ErrorCode.CONFIG_ERROR,
                            f"Invalid weight value for {section}.{weight_name}: {weight_value}. Must be between 0 and 1"
                        )

        # Validate severity weights
        severity_weights = quality_config.get("severity_weights", {})
        if not isinstance(severity_weights, dict):
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Invalid severity_weights configuration: must be a dictionary"
            )
        for severity, weight in severity_weights.items():
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise ReleaseError(
                    ErrorCode.CONFIG_ERROR,
                    f"Invalid severity weight for {severity}: {weight}. Must be between 0 and 1"
                )

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            *keys: Key path
            default: Default value if key not found

        Returns:
            Configuration value
        """
        value = self.data
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
            if value is None:
                return default
        return value

    @property
    def model(self) -> str:
        """Get the configured AI model."""
        return self.data["ai"]["model"]

    @property
    def github_repo(self) -> str:
        """Get the configured GitHub repository name."""
        return self.data["github"]["repo_name"]

    @property
    def github_org(self) -> str | None:
        """Get the configured GitHub organization (if any)."""
        return self.data["github"]["org"]

    @property
    def github_private(self) -> bool:
        """Get whether the repository should be private."""
        return self.data["github"]["private"]

    @property
    def quality_config(self) -> dict:
        """Get the code quality configuration."""
        return self.data["quality"]

    def get_severity_weight(self, severity: str) -> float:
        """Get the weight for a severity level.

        Args:
            severity: Severity level (critical, high, medium, low, info)

        Returns:
            float: Weight between 0 and 1
        """
        return self.data["quality"]["severity_weights"].get(
            severity.lower(),
            DEFAULT_SEVERITY_WEIGHTS["info"]
        )