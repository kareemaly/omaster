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

from .base import BaseAnalyzer


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyzer for code complexity metrics using scientific calculations."""

    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the complexity analyzer.
        
        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary with thresholds and weights
        """
        super().__init__(project_path)
        
        # Get complexity thresholds from config
        complexity_config = config["quality"]["complexity"]
        self.max_cyclomatic = complexity_config["max_cyclomatic"]
        self.max_cognitive = complexity_config["max_cognitive"]
        self.min_maintainability = complexity_config["min_maintainability"]
        self.max_halstead_difficulty = complexity_config["max_halstead_difficulty"]
        self.min_halstead_language_level = complexity_config["min_halstead_language_level"]
        self.max_bug_prediction = complexity_config["max_bug_prediction"]
        self.max_oop_complexity = complexity_config["max_oop_complexity"]
        
        # Get metric weights
        self.weights = complexity_config["weights"]
        
        # Get severity weights
        self.severity_weights = config["quality"]["severity_weights"]
        
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze code complexity metrics.
        
        Returns:
            List of complexity issues found
        """
        issues = []
        
        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue
                
            try:
                with open(file_path) as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Calculate enhanced complexity metrics
                        cyclo = self._calc_weighted_cyclomatic_complexity(node)
                        cogn = self._calc_enhanced_cognitive_complexity(node)
                        halstead = self._calc_enhanced_halstead_metrics(node)
                        maintainability = self._calc_enhanced_maintainability_index(
                            cyclo, halstead, content.count('\n'), node
                        )
                        
                        # Calculate new OOP metrics if in class context
                        if self._is_in_class(node):
                            oop_complexity = self._calc_oop_complexity(node)
                            if oop_complexity > self.max_oop_complexity:
                                severity = self._calculate_severity(
                                    oop_complexity / self.max_oop_complexity,
                                    self.weights["oop"]
                                )
                                issues.append({
                                    "file": str(file_path.relative_to(self.project_path)),
                                    "line": node.lineno,
                                    "message": f"Method '{node.name}' has high OOP complexity of {oop_complexity}",
                                    "type": "error",
                                    "severity": severity,
                                    "weight": self.weights["oop"]
                                })
                        
                        # Check cyclomatic complexity
                        if cyclo > self.max_cyclomatic:
                            severity = self._calculate_severity(
                                cyclo / self.max_cyclomatic,
                                self.weights["cyclomatic"]
                            )
                            issues.append({
                                "file": str(file_path.relative_to(self.project_path)),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has weighted cyclomatic complexity of {cyclo}",
                                "type": "error",
                                "severity": severity,
                                "weight": self.weights["cyclomatic"]
                            })
                            
                        # Check cognitive complexity
                        if cogn > self.max_cognitive:
                            severity = self._calculate_severity(
                                cogn / self.max_cognitive,
                                self.weights["cognitive"]
                            )
                            issues.append({
                                "file": str(file_path.relative_to(self.project_path)),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has enhanced cognitive complexity of {cogn}",
                                "type": "error", 
                                "severity": severity,
                                "weight": self.weights["cognitive"]
                            })
                            
                        # Check enhanced Halstead metrics
                        if halstead['difficulty'] > self.max_halstead_difficulty:
                            severity = self._calculate_severity(
                                halstead['difficulty'] / self.max_halstead_difficulty,
                                self.weights["halstead"]
                            )
                            issues.append({
                                "file": str(file_path.relative_to(self.project_path)),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has high Halstead difficulty: {halstead['difficulty']:.2f}",
                                "type": "error",
                                "severity": severity,
                                "weight": self.weights["halstead"]
                            })
                            
                        # Check language level
                        if halstead['language_level'] < self.min_halstead_language_level:
                            severity = self._calculate_severity(
                                (self.min_halstead_language_level - halstead['language_level']) / self.min_halstead_language_level,
                                self.weights["halstead"]
                            )
                            issues.append({
                                "file": str(file_path.relative_to(self.project_path)),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has low language level: {halstead['language_level']:.2f}",
                                "type": "warning",
                                "severity": severity,
                                "weight": self.weights["halstead"]
                            })
                            
                        # Check bug prediction
                        if halstead['bug_prediction'] > self.max_bug_prediction:
                            severity = self._calculate_severity(
                                halstead['bug_prediction'] / self.max_bug_prediction,
                                self.weights["halstead"]
                            )
                            issues.append({
                                "file": str(file_path.relative_to(self.project_path)),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has high bug prediction: {halstead['bug_prediction']:.2f}",
                                "type": "warning",
                                "severity": severity,
                                "weight": self.weights["halstead"]
                            })
                            
                        # Check maintainability index
                        if maintainability < self.min_maintainability:
                            severity = self._calculate_severity(
                                (self.min_maintainability - maintainability) / self.min_maintainability,
                                self.weights["maintainability"]
                            )
                            issues.append({
                                "file": str(file_path.relative_to(self.project_path)),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has low maintainability: {maintainability:.2f}",
                                "type": "error",
                                "severity": severity,
                                "weight": self.weights["maintainability"]
                            })
                            
            except Exception as e:
                issues.append({
                    "file": str(file_path.relative_to(self.project_path)),
                    "line": 1,
                    "message": f"Failed to analyze file: {str(e)}",
                    "type": "error",
                    "severity": "critical",
                    "weight": 1.0
                })
                
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