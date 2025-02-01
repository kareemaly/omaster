"""Base class for code quality analyzers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

class BaseAnalyzer(ABC):
    """Base class for all code quality analyzers."""
    
    def __init__(self, project_path: Path, config: Optional[Dict[str, Any]] = None):
        """Initialize the analyzer.
        
        Args:
            project_path: Path to the project root directory
            config: Optional configuration dictionary
        """
        self.project_path = project_path
        self.config = config or {}
        
    @abstractmethod
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze code quality.
        
        Returns:
            List of issues found. Each issue should be a dictionary with:
            - file: str (relative path to file)
            - line: int (line number)
            - message: str (issue description)
            - type: str (error type)
            - severity: str (critical, high, medium, low, info)
            - weight: float (0-1 scale, optional)
        """
        pass
        
    def _is_excluded(self, file_path: Path) -> bool:
        """Check if a file should be excluded from analysis.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if file should be excluded
        """
        # Standard exclusions
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "build",
            "dist",
            "*.egg-info",
        }
        
        # Add custom exclusions from config
        if "exclude_patterns" in self.config:
            excluded_dirs.update(self.config["exclude_patterns"])
            
        # Check path parts
        parts = file_path.parts
        return any(
            any(part.startswith(exclude) for exclude in excluded_dirs)
            for part in parts
        )
        
    def _make_issue(self, 
                    file_path: Path, 
                    line: int,
                    message: str,
                    issue_type: str = "error",
                    severity: str = "medium",
                    weight: float = 0.5,
                    **details: Any) -> Dict[str, Any]:
        """Create a standardized issue dictionary.
        
        Args:
            file_path: Path to the file
            line: Line number
            message: Issue description
            issue_type: Type of issue
            severity: Issue severity
            weight: Issue weight (0-1)
            **details: Additional issue details
            
        Returns:
            Dict containing issue information
        """
        return {
            "file": str(file_path.relative_to(self.project_path)),
            "line": line,
            "message": message,
            "type": issue_type,
            "severity": severity,
            "weight": weight,
            **details
        } 