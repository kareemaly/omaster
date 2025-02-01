"""Step 1.5: Code Quality Analysis.

This step runs various code quality checks:
1. Complexity analysis (cyclomatic, cognitive, maintainability)
2. Dead code detection (unused functions, classes, imports)
3. Code similarity detection (duplicate code patterns)
"""

import ast
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table

from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config
from ...quality.analyzers import (
    ComplexityAnalyzer,
    DeadCodeAnalyzer,
    SimilarityAnalyzer,
    StyleAnalyzer
)


class QualityMetric(Enum):
    """Types of quality metrics."""
    COMPLEXITY = "complexity"
    DEAD_CODE = "dead_code"
    SIMILARITY = "similarity"
    STYLE = "style"


@dataclass
class QualityIssue:
    """A code quality issue."""
    type: QualityMetric
    file_path: str  # Relative to project root
    line: int
    end_line: Optional[int]
    message: str
    details: Optional[Dict[str, Any]] = None


# Standard Python project directories to exclude
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "references",  # Exclude reference code
    "*.egg-info",
}

# Standard Python file patterns to exclude
EXCLUDED_FILES = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
}


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor for calculating code complexity metrics."""
    
    def __init__(self):
        self.cyclomatic_complexity = 0
        self.cognitive_complexity = 0
        self.functions = []
        self.current_function = None
        self.nesting_level = 0
        self.loc = 0
        
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        old_function = self.current_function
        old_complexity = self.cyclomatic_complexity
        old_cognitive = self.cognitive_complexity
        old_nesting = self.nesting_level
        
        # Create function metrics
        self.current_function = {
            'name': node.name,
            'cyclomatic_complexity': 1,  # Base complexity
            'cognitive_complexity': 0,
            'loc': len(node.body),
            'line': node.lineno,
            'end_line': node.end_lineno or node.lineno
        }
        self.cyclomatic_complexity = 1
        self.cognitive_complexity = 0
        self.nesting_level = 0
        
        # Visit function body
        self.generic_visit(node)
        
        # Update function metrics
        self.current_function['cyclomatic_complexity'] = self.cyclomatic_complexity
        self.current_function['cognitive_complexity'] = self.cognitive_complexity
        self.functions.append(self.current_function)
        
        # Update total LOC
        self.loc += len(node.body)
        
        # Restore state
        self.current_function = old_function
        self.cyclomatic_complexity = old_complexity
        self.cognitive_complexity = old_cognitive
        self.nesting_level = old_nesting
        
    def visit_If(self, node):
        """Visit if statement."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (1 + self.nesting_level)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_While(self, node):
        """Visit while loop."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (2 + self.nesting_level)  # Loops are more complex
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_For(self, node):
        """Visit for loop."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += (2 + self.nesting_level)  # Loops are more complex
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_Try(self, node):
        """Visit try block."""
        self.cyclomatic_complexity += len(node.handlers) + len(node.finalbody)
        self.cognitive_complexity += 1  # Error handling complexity
        self.generic_visit(node)
        
    def visit_ExceptHandler(self, node):
        """Visit except handler."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1
        self.generic_visit(node)
        
    def visit_BoolOp(self, node):
        """Visit boolean operation."""
        self.cyclomatic_complexity += len(node.values) - 1
        self.cognitive_complexity += (len(node.values) - 1)  # Each boolean operation adds complexity
        self.generic_visit(node)


def should_analyze_file(file_path: Path) -> bool:
    """Check if a file should be analyzed.
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if the file should be analyzed
    """
    # Convert to relative path from CWD
    try:
        rel_path = file_path.relative_to(Path.cwd())
    except ValueError:
        return False
        
    # Check directory exclusions
    for part in rel_path.parts:
        if any(part.startswith(pattern.rstrip('/*')) for pattern in EXCLUDED_DIRS):
            return False
            
    # Check file pattern exclusions
    for pattern in EXCLUDED_FILES:
        if rel_path.match(pattern):
            return False
            
    return True


def analyze_complexity(file_path: Path, config: Config) -> List[QualityIssue]:
    """Analyze code complexity.
    
    Args:
        file_path: Path to the file to analyze
        config: Configuration object
        
    Returns:
        List[QualityIssue]: List of complexity issues
    """
    issues = []
    complexity_config = config.quality_config["complexity"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Get relative path from CWD
        rel_path = file_path.relative_to(Path.cwd())
        
        for func in visitor.functions:
            # Check cyclomatic complexity
            if func['cyclomatic_complexity'] > complexity_config['max_cyclomatic']:
                issues.append(QualityIssue(
                    type=QualityMetric.COMPLEXITY,
                    file_path=str(rel_path),
                    line=func['line'],
                    end_line=func['end_line'],
                    message=f"Function '{func['name']}' has high cyclomatic complexity ({func['cyclomatic_complexity']})",
                    details={
                        'name': func['name'],
                        'cyclomatic_complexity': func['cyclomatic_complexity'],
                        'threshold': complexity_config['max_cyclomatic'],
                        'weight': complexity_config['weights']['cyclomatic']
                    }
                ))
            
            # Check cognitive complexity
            if func['cognitive_complexity'] > complexity_config['max_cognitive']:
                issues.append(QualityIssue(
                    type=QualityMetric.COMPLEXITY,
                    file_path=str(rel_path),
                    line=func['line'],
                    end_line=func['end_line'],
                    message=f"Function '{func['name']}' has high cognitive complexity ({func['cognitive_complexity']})",
                    details={
                        'name': func['name'],
                        'cognitive_complexity': func['cognitive_complexity'],
                        'threshold': complexity_config['max_cognitive'],
                        'weight': complexity_config['weights']['cognitive']
                    }
                ))
                
    except Exception as e:
        issues.append(QualityIssue(
            type=QualityMetric.COMPLEXITY,
            file_path=str(file_path),
            line=1,
            end_line=None,
            message=f"Failed to analyze complexity: {str(e)}"
        ))
        
    return issues


def format_issues_table(issues: List[QualityIssue]) -> str:
    """Format quality issues into a rich table.
    
    Args:
        issues: List of quality issues
        
    Returns:
        str: Formatted table string
    """
    console = Console(record=True)
    
    # Create table
    table = Table(title="🚨 Code Quality Issues")
    table.add_column("File", style="cyan")
    table.add_column("Line", justify="right", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Message", style="red")
    table.add_column("Details", style="yellow")
    
    # Add rows
    for issue in issues:
        details = ""
        if issue.details:
            details = ", ".join(f"{k}: {v}" for k, v in issue.details.items())
            
        table.add_row(
            issue.file_path,
            str(issue.line),
            issue.type.value,
            issue.message,
            details
        )
        
    # Render table
    console.print(table)
    return str(console.export_text())


def run_quality_analysis(project_path: Path, config: Config) -> List[QualityIssue]:
    """Run all code quality analyzers.
    
    Args:
        project_path: Path to the project root
        config: Configuration object
        
    Returns:
        List[QualityIssue]: List of all quality issues found
    """
    issues = []
    
    # Initialize analyzers
    analyzers = [
        ComplexityAnalyzer(project_path, config.quality_config),
        DeadCodeAnalyzer(project_path, config.quality_config),
        SimilarityAnalyzer(project_path, config.quality_config),
        StyleAnalyzer(project_path, config.quality_config)
    ]
    
    # Run each analyzer
    for analyzer in analyzers:
        try:
            analyzer_issues = analyzer.analyze()
            issues.extend(analyzer_issues)
        except Exception as e:
            issues.append(QualityIssue(
                type=QualityMetric.COMPLEXITY,  # Use complexity as default
                file_path="",
                line=1,
                end_line=None,
                message=f"Failed to run {analyzer.__class__.__name__}: {str(e)}"
            ))
    
    return issues


def format_issues_table(issues: List[QualityIssue]) -> Table:
    """Format issues into a rich table.
    
    Args:
        issues: List of quality issues
        
    Returns:
        Table: Formatted table of issues
    """
    table = Table(title="Code Quality Issues")
    
    table.add_column("Type", style="cyan")
    table.add_column("File", style="blue")
    table.add_column("Line", justify="right", style="green")
    table.add_column("Message", style="yellow")
    table.add_column("Details", style="magenta")
    
    for issue in issues:
        # Format line range
        line_str = str(issue.line)
        if issue.end_line and issue.end_line != issue.line:
            line_str += f"-{issue.end_line}"
            
        # Format details
        details_str = ""
        if issue.details:
            details_str = ", ".join(f"{k}: {v}" for k, v in issue.details.items())
            
        table.add_row(
            issue.type.value,
            issue.file_path,
            line_str,
            issue.message,
            details_str
        )
        
    return table


def analyze_code_quality(project_path: Path, config: Config) -> None:
    """Analyze code quality and print results.
    
    Args:
        project_path: Path to the project root
        config: Configuration object
        
    Raises:
        ReleaseError: If code quality issues are found
    """
    console = Console()
    
    # Run analysis
    issues = run_quality_analysis(project_path, config)
    
    if issues:
        # Format and display issues
        table = format_issues_table(issues)
        console.print(table)
        
        # Calculate total severity score
        total_score = 0.0
        max_score = 0.0
        
        for issue in issues:
            if issue.details and 'weight' in issue.details:
                weight = float(issue.details['weight'])
                severity_weight = config.get_severity_weight(issue.details.get('severity', 'medium'))
                total_score += weight * severity_weight
                max_score += weight
                
        if max_score > 0:
            quality_ratio = 1.0 - (total_score / max_score)
            if quality_ratio < 0.8:  # Allow up to 20% quality issues
                raise ReleaseError(
                    ErrorCode.CODE_QUALITY_ERROR,
                    f"Code quality score ({quality_ratio:.2%}) is below threshold (80%)"
                )
        else:
            # If no weights found, fail if any issues exist
            raise ReleaseError(
                ErrorCode.CODE_QUALITY_ERROR,
                "Code quality issues found"
            )


def run(project_path: Path) -> bool:
    """Run code quality analysis.
    
    Args:
        project_path: Path to the project root
        
    Returns:
        bool: True if analysis passed
        
    Raises:
        ReleaseError: If code quality issues are found
    """
    console = Console()
    console.print("\nStep 1.5: Analyzing code quality...")
    
    try:
        # Load config
        config = Config(project_path)
        
        # Run analysis
        analyze_code_quality(project_path, config)
        
        console.print("✓ Code quality analysis passed", style="green")
        return True
        
    except ReleaseError as e:
        console.print(f"\n🚨 Code quality error: {str(e)}", style="red")
        return False
        
    except Exception as e:
        console.print(f"\n🚨 Unexpected error: {str(e)}", style="red")
        return False