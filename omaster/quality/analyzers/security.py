"""Security analyzer for detecting potential vulnerabilities.

This analyzer implements comprehensive security checks:
1. Hardcoded secrets detection
2. SQL injection vulnerability detection
3. Command injection vulnerability detection
4. Unsafe file operations
5. Insecure cryptographic usage
6. XSS vulnerability detection
"""
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Pattern

from .base import BaseAnalyzer
from ..quality_issue import QualityIssue, QualityMetric

# Default configuration
DEFAULT_CONFIG = {
    "severity_weights": {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
        "info": 0.1
    }
}

@dataclass
class SecurityPattern:
    """Pattern for security vulnerability detection."""
    name: str
    pattern: Pattern
    message: str
    severity: str
    weight: float


class SecurityAnalyzer(BaseAnalyzer):
    """Analyzer for security vulnerabilities."""

    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the security analyzer.

        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary
        """
        super().__init__(project_path, config)

        # Get severity weights with defaults
        security_config = config.get("quality", {}).get("security", DEFAULT_CONFIG)
        self.severity_weights = security_config.get("severity_weights", DEFAULT_CONFIG["severity_weights"])

        # Initialize security patterns
        self.patterns = [
            # Hardcoded secrets
            SecurityPattern(
                name="hardcoded_password",
                pattern=re.compile(
                    r"password\s*=\s*['\"][\w\-+=@#$%^&*(){}[\]|\\:;<>,.?/~`]+['\"]",
                    re.IGNORECASE
                ),
                message="Hardcoded password detected",
                severity="critical",
                weight=self.severity_weights["critical"]
            ),
            SecurityPattern(
                name="hardcoded_key",
                pattern=re.compile(
                    r"(?:api_?key|auth_?token|secret)\s*=\s*['\"][\w\-+=@#$%^&*(){}[\]|\\:;<>,.?/~`]+['\"]",
                    re.IGNORECASE
                ),
                message="Hardcoded API key or secret detected",
                severity="critical",
                weight=self.severity_weights["critical"]
            ),

            # SQL injection
            SecurityPattern(
                name="sql_injection",
                pattern=re.compile(
                    r"execute\(['\"].*?\%.*?['\"].*?\)",
                    re.IGNORECASE
                ),
                message="Potential SQL injection vulnerability (use parameterized queries)",
                severity="high",
                weight=self.severity_weights["high"]
            ),

            # Command injection
            SecurityPattern(
                name="command_injection",
                pattern=re.compile(
                    r"(?:os\.system|subprocess\.(?:call|run|Popen))\(['\"].*?\+.*?['\"].*?\)",
                    re.IGNORECASE
                ),
                message="Potential command injection vulnerability (use shlex.quote)",
                severity="high",
                weight=self.severity_weights["high"]
            ),

            # Unsafe file operations
            SecurityPattern(
                name="unsafe_file_read",
                pattern=re.compile(
                    r"open\(.*?\+.*?\)",
                    re.IGNORECASE
                ),
                message="Unsafe file operation (potential path traversal)",
                severity="medium",
                weight=self.severity_weights["medium"]
            ),

            # Insecure crypto
            SecurityPattern(
                name="weak_crypto",
                pattern=re.compile(
                    r"(?:md5|sha1)\(",
                    re.IGNORECASE
                ),
                message="Use of weak cryptographic hash function",
                severity="medium",
                weight=self.severity_weights["medium"]
            ),

            # XSS
            SecurityPattern(
                name="xss",
                pattern=re.compile(
                    r"render_template\(.*?\+.*?\)",
                    re.IGNORECASE
                ),
                message="Potential XSS vulnerability (use template escaping)",
                severity="high",
                weight=self.severity_weights["high"]
            )
        ]

    def analyze(self) -> List[QualityIssue]:
        """Analyze code for security issues.

        Returns:
            List of security issues found
        """
        issues = []

        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue

            try:
                with open(file_path) as f:
                    content = f.read()
                    tree = ast.parse(content)

                # Pattern-based checks
                for pattern in self.patterns:
                    for match in pattern.pattern.finditer(content):
                        line_no = content.count('\n', 0, match.start()) + 1
                        issues.append(QualityIssue(
                            type=QualityMetric.ERROR,
                            file_path=str(file_path.relative_to(self.project_path)),
                            line=line_no,
                            end_line=None,
                            message=pattern.message,
                            severity=pattern.severity,
                            weight=pattern.weight
                        ))

                # AST-based checks
                visitor = SecurityVisitor(file_path, self.project_path, self.severity_weights)
                visitor.visit(tree)
                issues.extend(visitor.issues)

            except Exception as e:
                issues.append(QualityIssue(
                    type=QualityMetric.ERROR,
                    file_path=str(file_path.relative_to(self.project_path)),
                    line=1,
                    end_line=None,
                    message=f"Failed to analyze file: {str(e)}",
                    severity="critical",
                    weight=self.severity_weights["critical"]
                ))

        return issues


class SecurityVisitor(ast.NodeVisitor):
    """AST visitor for security vulnerability detection."""

    def __init__(self, file_path: Path, project_path: Path, severity_weights: Dict[str, float]):
        """Initialize the visitor.

        Args:
            file_path: Path to the file being analyzed
            project_path: Path to the project root
            severity_weights: Severity weight configuration
        """
        self.file_path = file_path
        self.project_path = project_path
        self.severity_weights = severity_weights
        self.issues: List[QualityIssue] = []

    def visit_Try(self, node: ast.Try) -> None:
        """Check for overly broad exception handling."""
        for handler in node.handlers:
            if (isinstance(handler.type, ast.Name) and
                handler.type.id == 'Exception'):
                self.issues.append(QualityIssue(
                    type=QualityMetric.ERROR,
                    file_path=str(self.file_path.relative_to(self.project_path)),
                    line=handler.lineno,
                    end_line=None,
                    message="Overly broad exception handler (catch specific exceptions)",
                    severity="low",
                    weight=self.severity_weights["low"]
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for dangerous function calls."""
        if isinstance(node.func, ast.Name):
            # Check for eval/exec usage
            if node.func.id in {'eval', 'exec'}:
                self.issues.append(QualityIssue(
                    type=QualityMetric.ERROR,
                    file_path=str(self.file_path.relative_to(self.project_path)),
                    line=node.lineno,
                    end_line=None,
                    message=f"Use of dangerous function '{node.func.id}'",
                    severity="critical",
                    weight=self.severity_weights["critical"]
                ))

            # Check for pickle usage
            elif node.func.id in {'loads', 'load'} and self._is_from_pickle(node):
                self.issues.append(QualityIssue(
                    type=QualityMetric.ERROR,
                    file_path=str(self.file_path.relative_to(self.project_path)),
                    line=node.lineno,
                    end_line=None,
                    message="Unsafe deserialization using pickle",
                    severity="high",
                    weight=self.severity_weights["high"]
                ))

        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Check for assertions that might be removed."""
        self.issues.append(QualityIssue(
            type=QualityMetric.ERROR,
            file_path=str(self.file_path.relative_to(self.project_path)),
            line=node.lineno,
            end_line=None,
            message="Security controls using assertions (assertions may be disabled)",
            severity="medium",
            weight=self.severity_weights["medium"]
        ))
        self.generic_visit(node)

    def _is_from_pickle(self, node: ast.Call) -> bool:
        """Check if a call is from the pickle module."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return node.func.value.id == 'pickle'
        return False