"""Style analyzer for checking code style and formatting.

This analyzer implements comprehensive style checks:
1. PEP 8 compliance
2. Naming conventions
3. Docstring formatting
4. Import ordering
5. Line length
6. Code complexity
"""
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Pattern, Optional

from .base import BaseAnalyzer


@dataclass
class StylePattern:
    """Pattern for style violation detection."""
    
    name: str
    pattern: Pattern
    message: str
    severity: str
    weight: float


class StyleAnalyzer(BaseAnalyzer):
    """Analyzer for code style and formatting."""
    
    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the style analyzer.
        
        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary with thresholds and weights
        """
        super().__init__(project_path, config)
        
        # Get style thresholds from config
        style_config = config["quality"]["style"]
        self.max_line_length = style_config["max_line_length"]
        self.max_function_args = style_config["max_function_args"]
        self.require_docstrings = style_config["require_docstrings"]
        
        # Get metric weights
        self.weights = style_config["weights"]
        
        # Get severity weights
        self.severity_weights = config["quality"]["severity_weights"]
        
        # Initialize style patterns
        self.patterns = [
            # Variable naming
            StylePattern(
                name="invalid_variable_name",
                pattern=re.compile(r"[A-Z]+[a-z0-9]*\s*="),
                message="Variable names should be lowercase with underscores",
                severity="low",
                weight=self.weights["naming"]
            ),
            StylePattern(
                name="single_char_name",
                pattern=re.compile(r"\b[a-z]\s*="),
                message="Avoid single-character variable names",
                severity="low",
                weight=self.weights["naming"]
            ),
            
            # Import style
            StylePattern(
                name="multiple_imports",
                pattern=re.compile(r"from\s+\S+\s+import\s+[^(][^\\]*,"),
                message="Multiple imports should use parentheses",
                severity="low",
                weight=self.weights["imports"]
            ),
            StylePattern(
                name="star_import",
                pattern=re.compile(r"from\s+\S+\s+import\s+\*"),
                message="Wildcard imports should be avoided",
                severity="medium",
                weight=self.weights["imports"]
            ),
            
            # Whitespace
            StylePattern(
                name="trailing_whitespace",
                pattern=re.compile(r"[ \t]+$", re.MULTILINE),
                message="Line contains trailing whitespace",
                severity="info",
                weight=self.weights["whitespace"]
            ),
            StylePattern(
                name="missing_newline",
                pattern=re.compile(r"[^\n]\Z"),
                message="File should end with a newline",
                severity="info",
                weight=self.weights["whitespace"]
            ),
            
            # Operators
            StylePattern(
                name="operator_spacing",
                pattern=re.compile(r"[^=!<>]=|=[^=]|\+=|-=|\*=|/=|%=|&=|\|=|\^=|>>=|<<=|\*\*="),
                message="Operators should be surrounded by spaces",
                severity="info",
                weight=self.weights["whitespace"]
            ),
        ]
        
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze code for style issues.
        
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
                    
                # Check line length
                for i, line in enumerate(content.splitlines(), 1):
                    if len(line) > self.max_line_length:
                        severity = self._calculate_severity(
                            len(line) / self.max_line_length,
                            self.weights["line_length"]
                        )
                        issues.append(self._make_issue(
                            file_path,
                            i,
                            f"Line too long ({len(line)} > {self.max_line_length} characters)",
                            "style",
                            severity,
                            self.weights["line_length"]
                        ))
                        
                # Pattern-based checks
                for pattern in self.patterns:
                    for match in pattern.pattern.finditer(content):
                        line_no = content.count('\n', 0, match.start()) + 1
                        issues.append(self._make_issue(
                            file_path,
                            line_no,
                            pattern.message,
                            "style",
                            pattern.severity,
                            pattern.weight
                        ))
                        
                # AST-based checks
                visitor = StyleVisitor(
                    file_path,
                    self.project_path,
                    self.max_function_args,
                    self.require_docstrings,
                    self.weights
                )
                visitor.visit(tree)
                issues.extend(visitor.issues)
                
            except Exception as e:
                issues.append(self._make_issue(
                    file_path,
                    1,
                    f"Failed to analyze file: {str(e)}",
                    "error",
                    "critical",
                    1.0
                ))
                
        return issues
        
    def _calculate_severity(self, ratio: float, weight: float) -> str:
        """Calculate severity level based on ratio and weight.
        
        Args:
            ratio: Ratio of actual value to threshold
            weight: Weight of the metric
            
        Returns:
            str: Severity level (critical, high, medium, low, info)
        """
        score = ratio * weight
        
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "info"


class StyleVisitor(ast.NodeVisitor):
    """AST visitor for style violation detection."""
    
    def __init__(self, file_path: Path, project_path: Path, 
                 max_args: int, require_docstrings: bool,
                 weights: Dict[str, float]):
        """Initialize the visitor.
        
        Args:
            file_path: Path to the file being analyzed
            project_path: Path to the project root
            max_args: Maximum number of function arguments
            require_docstrings: Whether docstrings are required
            weights: Style check weights
        """
        self.file_path = file_path
        self.project_path = project_path
        self.max_args = max_args
        self.require_docstrings = require_docstrings
        self.weights = weights
        self.issues: List[Dict[str, Any]] = []
        self.class_names: List[str] = []
        self.function_names: List[str] = []
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class definition style."""
        # Check class naming
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.issues.append({
                "file": str(self.file_path.relative_to(self.project_path)),
                "line": node.lineno,
                "message": "Class names should use CapWords convention",
                "type": "style",
                "severity": "low",
                "weight": self.weights["naming"]
            })
            
        # Check docstring
        if self.require_docstrings and not self._has_docstring(node):
            self.issues.append({
                "file": str(self.file_path.relative_to(self.project_path)),
                "line": node.lineno,
                "message": "Missing class docstring",
                "type": "style",
                "severity": "medium",
                "weight": self.weights["docstrings"]
            })
            
        self.class_names.append(node.name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definition style."""
        # Check function naming
        if not node.name.startswith('_') and not re.match(r'^[a-z][a-z0-9_]*$', node.name):
            self.issues.append({
                "file": str(self.file_path.relative_to(self.project_path)),
                "line": node.lineno,
                "message": "Function names should be lowercase with underscores",
                "type": "style",
                "severity": "low",
                "weight": self.weights["naming"]
            })
            
        # Check docstring
        if self.require_docstrings and not self._has_docstring(node):
            self.issues.append({
                "file": str(self.file_path.relative_to(self.project_path)),
                "line": node.lineno,
                "message": "Missing function docstring",
                "type": "style",
                "severity": "medium",
                "weight": self.weights["docstrings"]
            })
            
        # Check argument count
        if len(node.args.args) > self.max_args:
            self.issues.append({
                "file": str(self.file_path.relative_to(self.project_path)),
                "line": node.lineno,
                "message": f"Too many arguments ({len(node.args.args)} > {self.max_args})",
                "type": "style",
                "severity": "medium",
                "weight": self.weights["naming"]
            })
            
        self.function_names.append(node.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check import style."""
        # Check relative imports
        if node.level > 0:
            self.issues.append({
                "file": str(self.file_path.relative_to(self.project_path)),
                "line": node.lineno,
                "message": "Relative imports should be avoided",
                "type": "style",
                "severity": "low",
                "weight": self.weights["imports"]
            })
            
        self.generic_visit(node)
        
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a node has a docstring.
        
        Args:
            node: AST node to check
            
        Returns:
            True if the node has a docstring
        """
        return ast.get_docstring(node) is not None 