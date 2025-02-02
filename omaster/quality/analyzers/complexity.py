"""Code complexity analyzer implementing scientific metrics.

This module implements the following complexity metrics:
1. McCabe's Cyclomatic Complexity (CC) with weighted edges
2. Enhanced Halstead Complexity Metrics with additional indicators
3. Cognitive Complexity with weighted nesting
4. Advanced Maintainability Index with documentation factors
5. Object-Oriented Complexity Metrics
"""
import ast
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Set, Counter as CounterType
from dataclasses import dataclass

from .base import BaseAnalyzer
from ..quality_issue import QualityIssue, QualityMetric


# Default configuration values
DEFAULT_CONFIG = {
    "max_cyclomatic": 10,
    "max_cognitive": 15,
    "min_maintainability": 20,
    "max_halstead_difficulty": 30,
    "min_halstead_language_level": 0.8,
    "max_bug_prediction": 0.4,
    "max_oop_complexity": 50,
    "metric_weights": {
        "cyclomatic": 0.4,
        "cognitive": 0.4,
        "maintainability": 0.2
    },
    "severity_weights": {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    }
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


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyzer for code complexity metrics."""

    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the analyzer.

        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary
        """
        super().__init__(project_path, config)

        # Get thresholds
        self.max_cyclomatic = self.config.get("max_cyclomatic", DEFAULT_CONFIG["max_cyclomatic"])
        self.max_cognitive = self.config.get("max_cognitive", DEFAULT_CONFIG["max_cognitive"])
        self.min_maintainability = self.config.get("min_maintainability", DEFAULT_CONFIG["min_maintainability"])

        # Get weights
        self.metric_weights = self.config.get("metric_weights", DEFAULT_CONFIG["metric_weights"])
        self.severity_weights = self.config.get("severity_weights", DEFAULT_CONFIG["severity_weights"])

    def analyze_file(self, file_path: Path) -> List[QualityIssue]:
        """Analyze a single file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            List[QualityIssue]: List of complexity issues
        """
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            visitor = ComplexityVisitor()
            visitor.visit(tree)

            # Get relative path from project root
            rel_path = file_path.relative_to(self.project_path)

            for func in visitor.functions:
                # Check cyclomatic complexity
                if func['cyclomatic_complexity'] > self.max_cyclomatic:
                    issues.append(QualityIssue(
                        type=QualityMetric.COMPLEXITY,
                        file_path=str(rel_path),
                        line=func['line'],
                        end_line=func['end_line'],
                        message=f"Function '{func['name']}' has high cyclomatic complexity ({func['cyclomatic_complexity']})",
                        severity="high",
                        weight=self.severity_weights["high"],
                        details={
                            'name': func['name'],
                            'cyclomatic_complexity': func['cyclomatic_complexity'],
                            'threshold': self.max_cyclomatic
                        }
                    ))

                # Check cognitive complexity
                if func['cognitive_complexity'] > self.max_cognitive:
                    issues.append(QualityIssue(
                        type=QualityMetric.COMPLEXITY,
                        file_path=str(rel_path),
                        line=func['line'],
                        end_line=func['end_line'],
                        message=f"Function '{func['name']}' has high cognitive complexity ({func['cognitive_complexity']})",
                        severity="high",
                        weight=self.severity_weights["high"],
                        details={
                            'name': func['name'],
                            'cognitive_complexity': func['cognitive_complexity'],
                            'threshold': self.max_cognitive
                        }
                    ))

        except Exception as e:
            issues.append(QualityIssue(
                type=QualityMetric.COMPLEXITY,
                file_path=str(file_path),
                line=1,
                end_line=None,
                message=f"Failed to analyze complexity: {str(e)}",
                severity="critical",
                weight=self.severity_weights["critical"]
            ))

        return issues

    def analyze(self) -> List[QualityIssue]:
        """Analyze all Python files in the project.

        Returns:
            List[QualityIssue]: List of complexity issues
        """
        issues = []

        # Find all Python files
        for file_path in self.project_path.rglob("*.py"):
            # Skip test files and migrations
            if "test" in str(file_path) or "migrations" in str(file_path):
                continue

            issues.extend(self.analyze_file(file_path))

        return issues

    def _calc_weighted_cyclomatic_complexity(self, node: ast.AST) -> float:
        """Calculate weighted cyclomatic complexity.

        Enhances McCabe's formula with nesting weights:
        WCC = Σ(w_i * e_i) - n + 2p
        where:
        - w_i = weight of control structure at nesting level i
        - e_i = number of edges at nesting level i
        - n = number of nodes
        - p = number of connected components

        Args:
            node: AST node to analyze

        Returns:
            Weighted cyclomatic complexity score
        """
        complexity = 1.0  # Base complexity
        nesting_level = 0

        def visit_node(node: ast.AST, level: int) -> None:
            nonlocal complexity
            weight = 1.0 + (level * 0.1)  # Increase weight with nesting

            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += weight
            elif isinstance(node, ast.BoolOp):
                complexity += weight * (len(node.values) - 1)
            elif isinstance(node, (ast.With, ast.Assert)):
                complexity += weight * 0.8  # Lower weight for simpler constructs
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += weight * 0.5  # Lower weight for comprehensions

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                    visit_node(child, level + 1)
                else:
                    visit_node(child, level)

        visit_node(node, nesting_level)
        return complexity

    def _calc_enhanced_cognitive_complexity(self, node: ast.AST) -> float:
        """Calculate enhanced cognitive complexity.

        Enhances SonarSource algorithm with:
        1. Variable name complexity
        2. Structural patterns
        3. Working memory model

        Args:
            node: AST node to analyze

        Returns:
            Enhanced cognitive complexity score
        """
        complexity = 0.0
        nesting = 0
        working_memory = set()  # Track variables in scope

        def visit(node: ast.AST, level: int = 0) -> None:
            nonlocal complexity, nesting

            # B1: Enhanced nesting penalties
            nesting_weight = 1.0 + (level * 0.2)  # Progressive nesting penalty

            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += level * nesting_weight

            # B2: Enhanced structural complexity
            if isinstance(node, ast.If):
                complexity += 1 * nesting_weight
                if node.orelse:  # Extra penalty for else clauses
                    complexity += 0.5 * nesting_weight
            elif isinstance(node, (ast.While, ast.For)):
                complexity += 1.5 * nesting_weight  # Higher weight for loops
            elif isinstance(node, ast.Try):
                complexity += 1 * nesting_weight
                complexity += len(node.handlers) * 0.5  # Penalty per except clause

            # B3: Enhanced cognitive load factors
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    working_memory.add(node.id)
                    # Penalize complex variable names
                    if len(node.id) > 20 or sum(1 for c in node.id if c.isupper()) > 2:
                        complexity += 0.2
            elif isinstance(node, ast.BoolOp):
                complexity += (len(node.values) - 1) * 0.5 * nesting_weight

            # Working memory model penalty
            if len(working_memory) > 7:  # Miller's Law: 7±2 items
                complexity += 0.1 * (len(working_memory) - 7)

            for child in ast.iter_child_nodes(node):
                visit(child, level + 1)

        visit(node)
        return complexity

    def _calc_enhanced_halstead_metrics(self, node: ast.AST) -> Dict[str, float]:
        """Calculate enhanced Halstead complexity metrics.

        Adds additional Halstead indicators:
        - Intelligence Content (I)
        - Language Level (λ)
        - Program Level (PL)
        - Bug Prediction (B)

        Args:
            node: AST node to analyze

        Returns:
            Dictionary containing enhanced Halstead metrics
        """
        operators: CounterType[str] = Counter()
        operands: CounterType[str] = Counter()

        def collect_operators_operands(node: ast.AST) -> None:
            if isinstance(node, ast.BinOp):
                operators[type(node.op).__name__] += 1
            elif isinstance(node, ast.UnaryOp):
                operators[type(node.op).__name__] += 1
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    operators[type(op).__name__] += 1
            elif isinstance(node, ast.Name):
                operands[node.id] += 1
            elif isinstance(node, ast.Num):
                operands[str(node.n)] += 1
            elif isinstance(node, ast.Str):
                operands[node.s] += 1

            for child in ast.iter_child_nodes(node):
                collect_operators_operands(child)

        collect_operators_operands(node)

        # Calculate base metrics
        n1 = len(operators)  # Distinct operators
        n2 = len(operands)   # Distinct operands
        N1 = sum(operators.values())  # Total operators
        N2 = sum(operands.values())   # Total operands

        # Handle edge cases
        if n1 == 0 or n2 == 0:
            return {
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
                "time": 0,
                "intelligence": 0,
                "language_level": 1.0,
                "program_level": 1.0,
                "bug_prediction": 0
            }

        # Calculate enhanced Halstead metrics
        N = N1 + N2  # Program length
        n = n1 + n2  # Vocabulary size
        V = N * math.log2(n)  # Volume
        D = (n1 / 2) * (N2 / n2)  # Difficulty
        E = D * V  # Effort
        T = E / 18  # Time to program (seconds)

        # New metrics
        L = 1 / D  # Program level (inverse of difficulty)
        I = L * V  # Intelligence content
        lambda_val = (L ** 2) * (V / n1 / N2)  # Language level
        B = V / 3000  # Bug prediction (bugs per KLOC)

        return {
            "volume": V,
            "difficulty": D,
            "effort": E,
            "time": T,
            "intelligence": I,
            "language_level": lambda_val,
            "program_level": L,
            "bug_prediction": B
        }

    def _calc_enhanced_maintainability_index(self,
                                           cyclomatic: float,
                                           halstead: Dict[str, float],
                                           loc: int,
                                           node: ast.AST) -> float:
        """Calculate enhanced maintainability index.

        Enhanced formula:
        MI = 171 - 5.2ln(HV) - 0.23CC - 16.2ln(LOC) + 50sin(√(2.4CD))
        where:
        - HV = Halstead Volume
        - CC = Cyclomatic Complexity
        - LOC = Lines of Code
        - CD = Comment Density (0-1)

        Args:
            cyclomatic: Weighted cyclomatic complexity
            halstead: Dictionary of enhanced Halstead metrics
            loc: Lines of code
            node: AST node for additional analysis

        Returns:
            Enhanced maintainability index (0-171)
        """
        if loc == 0 or halstead["volume"] == 0:
            return 171

        # Calculate comment density
        comment_lines = sum(1 for child in ast.walk(node)
                          if isinstance(child, ast.Expr) and
                          isinstance(child.value, ast.Str))
        comment_density = comment_lines / max(loc, 1)

        # Enhanced maintainability formula
        mi = (171
              - 5.2 * math.log(halstead["volume"])
              - 0.23 * cyclomatic
              - 16.2 * math.log(loc)
              + 50 * math.sin(math.sqrt(2.4 * comment_density)))

        # Normalize to 0-100 scale
        return max(0, min(100, mi * 100 / 171))

    def _calc_oop_complexity(self, node: ast.AST) -> float:
        """Calculate object-oriented programming complexity.

        Measures:
        1. Inheritance depth
        2. Number of overridden methods
        3. Coupling between objects
        4. Cohesion of methods

        Args:
            node: AST node to analyze

        Returns:
            OOP complexity score
        """
        complexity = 0.0

        # Check if node is in a class
        if not self._is_in_class(node):
            return 0.0

        # Analyze class hierarchy
        parent_classes = self._get_parent_classes(node)
        complexity += len(parent_classes) * 2  # Inheritance depth penalty

        # Check for method overrides
        if self._is_override(node):
            complexity += 1.5

        # Analyze coupling (method calls to other classes)
        coupling = self._analyze_coupling(node)
        complexity += coupling * 0.5

        # Analyze cohesion (method interactions within class)
        cohesion = self._analyze_cohesion(node)
        complexity += (1 - cohesion) * 3  # Lower cohesion increases complexity

        return complexity

    def _is_in_class(self, node: ast.AST) -> bool:
        """Check if a node is inside a class definition."""
        parent = getattr(node, 'parent', None)
        while parent:
            if isinstance(parent, ast.ClassDef):
                return True
            parent = getattr(parent, 'parent', None)
        return False

    def _get_parent_classes(self, node: ast.AST) -> List[str]:
        """Get list of parent classes."""
        parent = getattr(node, 'parent', None)
        while parent and not isinstance(parent, ast.ClassDef):
            parent = getattr(parent, 'parent', None)

        if parent and isinstance(parent, ast.ClassDef):
            return [base.id for base in parent.bases
                   if isinstance(base, ast.Name)]
        return []

    def _is_override(self, node: ast.AST) -> bool:
        """Check if a method overrides a parent method."""
        if not isinstance(node, ast.FunctionDef):
            return False

        # Check for override decorator
        return any(isinstance(d, ast.Name) and d.id == 'override'
                  for d in node.decorator_list)

    def _analyze_coupling(self, node: ast.AST) -> int:
        """Analyze coupling between objects."""
        external_calls = 0

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    external_calls += 1

        return external_calls

    def _analyze_cohesion(self, node: ast.AST) -> float:
        """Analyze method cohesion within class."""
        method_vars = set()
        class_vars = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load):
                    method_vars.add(child.id)
                elif isinstance(child.ctx, ast.Store):
                    class_vars.add(child.id)

        if not class_vars:
            return 1.0

        return len(method_vars.intersection(class_vars)) / len(class_vars)

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