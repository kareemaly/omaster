"""Base validator class for implementing different validators."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List


class ValidationResult:
    """Result of a validation check."""
    def __init__(self, passed: bool, messages: List[str]):
        self.passed = passed
        self.messages = messages


class Validator(ABC):
    """Base class for all validators."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the validator."""
        pass
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """Run validation and return result."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get validator configuration. Override if needed."""
        return {} 