"""Configuration management for omaster."""
from pathlib import Path
import os
import yaml
from typing import Dict, Any
from .errors import ErrorCode, ReleaseError

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
    "github": {
        "repo_name": None,  # Will default to pyproject.toml name if not set
        "org": None,  # Optional, will use user's account if not set
        "private": False
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
        # Style thresholds
        "style": {
            "max_line_length": 100,
            "max_function_args": 5,
            "require_docstrings": True,
            "weights": {
                "line_length": 0.3,
                "naming": 0.4,
                "docstrings": 0.5,
                "imports": 0.4,
                "whitespace": 0.2
            }
        },
        # Global severity weights
        "severity_weights": DEFAULT_SEVERITY_WEIGHTS.copy()
    }
}

class Config:
    """Configuration management class."""
    
    def __init__(self, project_path: Path):
        """Initialize configuration.
        
        Args:
            project_path: Path to the project root
        """
        self.project_path = project_path
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from .omaster.yaml file.
        
        Returns:
            dict: Merged configuration
        """
        config = DEFAULT_CONFIG.copy()
        
        config_file = self.project_path / ".omaster.yaml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    user_config = yaml.safe_load(f)
                if user_config:
                    # Deep merge user config
                    self._deep_merge(config, user_config)
            except Exception as e:
                raise ReleaseError(
                    ErrorCode.CONFIG_ERROR,
                    f"Failed to load .omaster.yaml: {str(e)}"
                )
                
        # Validate model choice
        if config["ai"]["model"] not in VALID_MODELS:
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                f"Invalid model: {config['ai']['model']}. Must be one of {VALID_MODELS}"
            )
            
        # If repo_name not set, try to get from pyproject.toml
        if not config["github"]["repo_name"]:
            try:
                with open(self.project_path / "pyproject.toml", "rb") as f:
                    import tomli
                    pyproject = tomli.load(f)
                    config["github"]["repo_name"] = pyproject["project"]["name"]
            except Exception as e:
                raise ReleaseError(
                    ErrorCode.CONFIG_ERROR,
                    "Repository name must be set in .omaster.yaml or pyproject.toml"
                )
            
        return config
        
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary to merge into
            update: Dictionary to merge from
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
                
    @property
    def model(self) -> str:
        """Get the configured AI model."""
        return self.config["ai"]["model"]
    
    @property
    def github_repo(self) -> str:
        """Get the configured GitHub repository name."""
        return self.config["github"]["repo_name"]
    
    @property
    def github_org(self) -> str | None:
        """Get the configured GitHub organization (if any)."""
        return self.config["github"]["org"]
    
    @property
    def github_private(self) -> bool:
        """Get whether the repository should be private."""
        return self.config["github"]["private"]
    
    @property
    def quality_config(self) -> dict:
        """Get the code quality configuration."""
        return self.config["quality"]
    
    def get_severity_weight(self, severity: str) -> float:
        """Get the weight for a severity level.
        
        Args:
            severity: Severity level (critical, high, medium, low, info)
            
        Returns:
            float: Weight between 0 and 1
        """
        return self.config["quality"]["severity_weights"].get(
            severity.lower(), 
            DEFAULT_SEVERITY_WEIGHTS["info"]
        ) 