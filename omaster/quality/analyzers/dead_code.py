"""Dead code analyzer for detecting unused and unreachable code.

This analyzer uses Vulture to detect:
1. Unused imports
2. Unused variables
3. Unused functions
4. Unused classes
5. Unused attributes
6. Unreachable code
"""
import logging
from pathlib import Path
from typing import Dict, List, Any
import vulture

from .base import BaseAnalyzer
from ..quality_issue import QualityIssue, QualityMetric

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "min_confidence": 60,  # Minimum confidence percentage for reporting
    "ignore_names": ["test_*"],  # Patterns to ignore
    "ignore_decorators": ["@app.*", "@pytest.*"],  # Decorator patterns to ignore
    "severity_weights": {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    }
}

class DeadCodeAnalyzer(BaseAnalyzer):
    """Analyzer for detecting dead code using Vulture."""

    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the analyzer.

        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary
        """
        super().__init__(project_path, config)

        # Get dead code config with defaults
        dead_code_config = config.get("quality", {}).get("dead_code", DEFAULT_CONFIG)
        
        # Initialize Vulture with configuration
        self.vulture = vulture.Vulture()
        self.vulture.minimum_confidence = dead_code_config.get("min_confidence", DEFAULT_CONFIG["min_confidence"])
        
        # Configure ignore patterns
        for pattern in dead_code_config.get("ignore_names", DEFAULT_CONFIG["ignore_names"]):
            self.vulture.ignore_names.append(pattern)
        for pattern in dead_code_config.get("ignore_decorators", DEFAULT_CONFIG["ignore_decorators"]):
            self.vulture.ignore_decorators.append(pattern)

        # Get severity weights with defaults
        self.severity_weights = dead_code_config.get("severity_weights", DEFAULT_CONFIG["severity_weights"])

    def _get_severity(self, item_type: str, confidence: int) -> str:
        """Determine severity based on item type and confidence.

        Args:
            item_type: Type of dead code item
            confidence: Confidence percentage

        Returns:
            Severity level string
        """
        if confidence >= 90:
            if item_type in {"import", "function", "class"}:
                return "high"
            return "medium"
        elif confidence >= 75:
            return "medium"
        return "low"

    def analyze(self) -> List[QualityIssue]:
        """Analyze code for dead code issues.

        Returns:
            List of dead code issues found
        """
        issues = []
        logger.info("Starting dead code analysis with Vulture...")

        try:
            # Scan all Python files in the project
            python_files = list(self.project_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to analyze")

            # Exclude files if needed
            files_to_analyze = [
                str(f) for f in python_files 
                if not self._is_excluded(f)
            ]

            # Scan files with Vulture
            self.vulture.scavenge(files_to_analyze)

            # Process results
            for item in self.vulture.get_unused_code():
                # Skip items below confidence threshold
                if item.confidence < self.vulture.minimum_confidence:
                    continue

                # Determine severity based on type and confidence
                severity = self._get_severity(item.typ, item.confidence)
                weight = self.severity_weights[severity]

                # Create quality issue
                rel_path = str(Path(item.filename).relative_to(self.project_path))
                message = f"Unused {item.typ}: {item.name} (confidence: {item.confidence}%)"
                
                issues.append(QualityIssue(
                    type=QualityMetric.DEAD_CODE,
                    file_path=rel_path,
                    line=item.first_lineno,
                    end_line=item.last_lineno,
                    message=message,
                    severity=severity,
                    weight=weight,
                    details={
                        "type": item.typ,
                        "name": item.name,
                        "confidence": item.confidence,
                        "size": item.size
                    }
                ))

            logger.info(f"Found {len(issues)} potential dead code issues")
            return issues

        except Exception as e:
            logger.error(f"Error during dead code analysis: {str(e)}", exc_info=True)
            issues.append(QualityIssue(
                type=QualityMetric.DEAD_CODE,
                file_path="",
                line=1,
                end_line=None,
                message=f"Failed to analyze dead code: {str(e)}",
                severity="critical",
                weight=self.severity_weights["critical"]
            ))
            return issues