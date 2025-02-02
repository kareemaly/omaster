"""Code style analyzer for enforcing consistent coding style.

This analyzer implements comprehensive style checking:
1. PEP 8 compliance checking
2. Documentation style and completeness
3. Naming convention enforcement
4. Line length and complexity
5. Import organization
6. Comment quality and placement
"""
import ast
import tokenize
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import re

from .base import BaseAnalyzer
from ..quality_issue import QualityIssue, QualityMetric


# Default configuration values
DEFAULT_CONFIG = {
    "max_line_length": 100,
    "max_doc_length": 72,
    "min_doc_length": 10,
    "max_function_length": 50,
    "max_class_length": 500,
    "max_module_length": 1000,
    "min_name_length": 2,
    "max_name_length": 30,
    "required_docstring_sections": [
        "Args",
        "Returns",
        "Raises"
    ],
    "weights": {
        "line_length": 0.3,
        "doc_style": 0.4,
        "naming": 0.5,
        "imports": 0.2,
        "comments": 0.3
    },
    "severity_weights": {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    }
}


class StyleAnalyzer(BaseAnalyzer):
    """Analyzer for code style and documentation."""

    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the style analyzer.

        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary with style rules and weights
        """
        super().__init__(project_path, config)

        # Get style thresholds from config with defaults
        style_config = config.get("quality", {}).get("style", DEFAULT_CONFIG)
        self.max_line_length = style_config.get(
            "max_line_length",
            DEFAULT_CONFIG["max_line_length"]
        )
        self.max_doc_length = style_config.get(
            "max_doc_length",
            DEFAULT_CONFIG["max_doc_length"]
        )
        self.min_doc_length = style_config.get(
            "min_doc_length",
            DEFAULT_CONFIG["min_doc_length"]
        )
        self.max_function_length = style_config.get(
            "max_function_length",
            DEFAULT_CONFIG["max_function_length"]
        )
        self.max_class_length = style_config.get(
            "max_class_length",
            DEFAULT_CONFIG["max_class_length"]
        )
        self.max_module_length = style_config.get(
            "max_module_length",
            DEFAULT_CONFIG["max_module_length"]
        )
        self.min_name_length = style_config.get(
            "min_name_length",
            DEFAULT_CONFIG["min_name_length"]
        )
        self.max_name_length = style_config.get(
            "max_name_length",
            DEFAULT_CONFIG["max_name_length"]
        )
        self.required_docstring_sections = style_config.get(
            "required_docstring_sections",
            DEFAULT_CONFIG["required_docstring_sections"]
        )

        # Get metric weights with defaults
        self.weights = style_config.get("weights", DEFAULT_CONFIG["weights"])

        # Get severity weights with defaults
        self.severity_weights = style_config.get("severity_weights", DEFAULT_CONFIG["severity_weights"])

        # Compile regex patterns
        self.snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        self.pascal_case_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.camel_case_pattern = re.compile(r'^[a-z][a-zA-Z0-9]*$')

    def analyze(self) -> List[QualityIssue]:
        """Analyze code style and documentation.

        Returns:
            List of style issues found
        """
        issues = []

        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue

            try:
                with open(file_path) as f:
                    content = f.read()
                    tree = ast.parse(content)

                # Check line length and basic formatting
                issues.extend(self._check_line_formatting(
                    str(file_path.relative_to(self.project_path)),
                    content
                ))

                # Check documentation
                issues.extend(self._check_documentation(
                    str(file_path.relative_to(self.project_path)),
                    tree
                ))

                # Check naming conventions
                issues.extend(self._check_naming(
                    str(file_path.relative_to(self.project_path)),
                    tree
                ))

                # Check imports
                issues.extend(self._check_imports(
                    str(file_path.relative_to(self.project_path)),
                    tree
                ))

                # Check comments
                issues.extend(self._check_comments(
                    str(file_path.relative_to(self.project_path)),
                    content
                ))

            except Exception as e:
                issues.append(QualityIssue(
                    type=QualityMetric.STYLE,
                    file_path=str(file_path),
                    line=1,
                    end_line=None,
                    message=f"Failed to analyze style: {str(e)}",
                    severity="critical",
                    weight=self.severity_weights["critical"]
                ))

        return issues

    def _check_line_formatting(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check line length and basic formatting.

        Args:
            file_path: Path to the file being checked
            content: File content

        Returns:
            List of formatting issues
        """
        issues = []

        for i, line in enumerate(content.splitlines(), 1):
            # Check line length
            if len(line.rstrip()) > self.max_line_length:
                issues.append(QualityIssue(
                    type=QualityMetric.STYLE,
                    file_path=file_path,
                    line=i,
                    end_line=i,
                    message=f"Line too long ({len(line)} > {self.max_line_length} characters)",
                    severity="low",
                    weight=self.severity_weights["low"]
                ))

            # Check trailing whitespace
            if line.rstrip() != line:
                issues.append(QualityIssue(
                    type=QualityMetric.STYLE,
                    file_path=file_path,
                    line=i,
                    end_line=i,
                    message="Line contains trailing whitespace",
                    severity="info",
                    weight=self.severity_weights["low"] * 0.5
                ))

        return issues

    def _check_documentation(self, file_path: str, tree: ast.AST) -> List[QualityIssue]:
        """Check documentation style and completeness.

        Args:
            file_path: Path to the file being checked
            tree: AST of the file

        Returns:
            List of documentation issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)

                if not docstring:
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        message=f"Missing docstring in {type(node).__name__.lower()}",
                        severity="medium",
                        weight=self.severity_weights["medium"]
                    ))
                    continue

                # Check docstring length
                if len(docstring) < self.min_doc_length:
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        message=f"Docstring too short ({len(docstring)} < {self.min_doc_length})",
                        severity="low",
                        weight=self.severity_weights["low"]
                    ))

                # Check required sections
                if isinstance(node, ast.FunctionDef):
                    missing_sections = []
                    for section in self.required_docstring_sections:
                        if section not in docstring:
                            missing_sections.append(section)

                    if missing_sections:
                        issues.append(QualityIssue(
                            type=QualityMetric.STYLE,
                            file_path=file_path,
                            line=node.lineno,
                            end_line=node.end_lineno,
                            message=f"Missing docstring sections: {', '.join(missing_sections)}",
                            severity="medium",
                            weight=self.severity_weights["medium"]
                        ))

        return issues

    def _check_naming(self, file_path: str, tree: ast.AST) -> List[QualityIssue]:
        """Check naming conventions.

        Args:
            file_path: Path to the file being checked
            tree: AST of the file

        Returns:
            List of naming issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id

                # Check name length
                if len(name) < self.min_name_length:
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        message=f"Name too short: {name}",
                        severity="low",
                        weight=self.severity_weights["low"]
                    ))
                elif len(name) > self.max_name_length:
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        message=f"Name too long: {name}",
                        severity="low",
                        weight=self.severity_weights["low"]
                    ))

                # Check naming convention
                if isinstance(node.ctx, ast.Store):
                    if isinstance(node.parent, ast.ClassDef):
                        if not self.pascal_case_pattern.match(name):
                            issues.append(QualityIssue(
                                type=QualityMetric.STYLE,
                                file_path=file_path,
                                line=node.lineno,
                                end_line=node.end_lineno,
                                message=f"Class name should be PascalCase: {name}",
                                severity="medium",
                                weight=self.severity_weights["medium"]
                            ))
                    elif isinstance(node.parent, ast.FunctionDef):
                        if not self.snake_case_pattern.match(name):
                            issues.append(QualityIssue(
                                type=QualityMetric.STYLE,
                                file_path=file_path,
                                line=node.lineno,
                                end_line=node.end_lineno,
                                message=f"Function name should be snake_case: {name}",
                                severity="medium",
                                weight=self.severity_weights["medium"]
                            ))

        return issues

    def _check_imports(self, file_path: str, tree: ast.AST) -> List[QualityIssue]:
        """Check import organization.

        Args:
            file_path: Path to the file being checked
            tree: AST of the file

        Returns:
            List of import issues
        """
        issues = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)

        # Check import order
        if len(imports) > 1:
            for i in range(len(imports) - 1):
                curr = imports[i]
                next_import = imports[i + 1]

                if curr.lineno > next_import.lineno:
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=curr.lineno,
                        end_line=curr.end_lineno,
                        message="Imports not in alphabetical order",
                        severity="low",
                        weight=self.severity_weights["low"]
                    ))

        return issues

    def _check_comments(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check comment quality and placement.

        Args:
            file_path: Path to the file being checked
            content: File content

        Returns:
            List of comment issues
        """
        issues = []

        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()

            if line.startswith('#'):
                # Check comment style
                if not line.startswith('# '):
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=i,
                        end_line=i,
                        message="Comment should have a space after #",
                        severity="info",
                        weight=self.severity_weights["low"]
                    ))

                # Check comment content
                comment = line[1:].strip()
                if len(comment) < self.min_doc_length:
                    issues.append(QualityIssue(
                        type=QualityMetric.STYLE,
                        file_path=file_path,
                        line=i,
                        end_line=i,
                        message=f"Comment too short ({len(comment)} < {self.min_doc_length})",
                        severity="info",
                        weight=self.severity_weights["low"]
                    ))

        return issues