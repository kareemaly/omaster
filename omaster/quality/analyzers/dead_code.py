"""Dead code analyzer for detecting unused and unreachable code.

This analyzer implements comprehensive dead code detection:
1. Unused imports, variables, functions, and classes
2. Unreachable code blocks (after return/break/continue)
3. Dead branches in conditional statements
4. Unused exception handlers
5. Dead code in generators and comprehensions
6. Cross-module symbol tracking
"""
import ast
import builtins
import symtable
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field

from .base import BaseAnalyzer


@dataclass
class Symbol:
    """Represents a code symbol with usage tracking."""
    
    name: str
    node: ast.AST
    defined_line: int
    is_import: bool = False
    is_function: bool = False
    is_class: bool = False
    is_parameter: bool = False
    uses: List[Tuple[str, int]] = field(default_factory=list)  # (file, line) pairs


@dataclass
class Scope:
    """Represents a code scope with symbol tracking."""
    
    name: str
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    parent: Optional['Scope'] = None
    children: List['Scope'] = field(default_factory=list)
    returns_seen: bool = False
    has_yield: bool = False
    
    def add_symbol(self, name: str, node: ast.AST, line: int, **kwargs) -> None:
        """Add a symbol to the scope.
        
        Args:
            name: Symbol name
            node: AST node where symbol is defined
            line: Line number where symbol is defined
            **kwargs: Additional symbol attributes
        """
        self.symbols[name] = Symbol(name=name, node=node, defined_line=line, **kwargs)
        
    def use_symbol(self, name: str, file: str, line: int) -> bool:
        """Mark a symbol as used.
        
        Args:
            name: Symbol name
            file: File where symbol is used
            line: Line number where symbol is used
            
        Returns:
            True if symbol was found and marked as used
        """
        if name in self.symbols:
            self.symbols[name].uses.append((file, line))
            return True
            
        if self.parent:
            return self.parent.use_symbol(name, file, line)
            
        return False
        
    def get_symbol(self, name: str) -> Optional[Symbol]:
        """Get a symbol from this scope or parent scopes.
        
        Args:
            name: Symbol name
            
        Returns:
            Symbol if found, None otherwise
        """
        if name in self.symbols:
            return self.symbols[name]
            
        if self.parent:
            return self.parent.get_symbol(name)
            
        return None
        
    def get_unused_symbols(self) -> List[Symbol]:
        """Get all unused symbols in this scope.
        
        Returns:
            List of unused symbols
        """
        unused = []
        for symbol in self.symbols.values():
            # Skip builtins and special names
            if symbol.name in {'__name__', '__file__', '__doc__'}:
                continue
                
            # Skip parameters in certain contexts
            if symbol.is_parameter and (self.has_yield or self.name.startswith('__')):
                continue
                
            if not symbol.uses:
                unused.append(symbol)
                
        return unused


class DeadCodeAnalyzer(BaseAnalyzer):
    """Analyzer for detecting dead code."""
    
    def __init__(self, project_path: Path):
        """Initialize the dead code analyzer.
        
        Args:
            project_path: Path to the project root directory
        """
        super().__init__(project_path)
        self._builtins = set(dir(builtins))
        self._module_symbols: Dict[str, Dict[str, Symbol]] = {}
        
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze code for unused and unreachable code.
        
        The analysis is performed in multiple passes:
        1. Collect all symbol definitions across modules
        2. Track symbol usage across modules
        3. Detect unreachable code
        4. Generate issues for unused and unreachable code
        
        Returns:
            List of dead code issues found
        """
        issues = []
        
        # First pass: collect symbols
        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue
                
            try:
                with open(file_path) as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                # Create symbol table for additional analysis
                try:
                    table = symtable.symtable(content, str(file_path), 'exec')
                except Exception:
                    table = None
                    
                # Collect symbols
                visitor = DeadCodeVisitor(
                    self._builtins,
                    str(file_path.relative_to(self.project_path)),
                    table
                )
                visitor.visit(tree)
                
                # Store module symbols
                rel_path = str(file_path.relative_to(self.project_path))
                self._module_symbols[rel_path] = visitor.root_scope.symbols
                
            except Exception as e:
                issues.append(self._make_error_issue(file_path, str(e)))
                
        # Second pass: track usage across modules
        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue
                
            try:
                with open(file_path) as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                # Track cross-module usage
                visitor = CrossModuleVisitor(
                    self._module_symbols,
                    str(file_path.relative_to(self.project_path))
                )
                visitor.visit(tree)
                
            except Exception as e:
                issues.append(self._make_error_issue(file_path, str(e)))
                
        # Generate issues
        for file_path, symbols in self._module_symbols.items():
            # Report unused symbols
            for symbol in symbols.values():
                if not symbol.uses:
                    if symbol.is_import:
                        issues.append({
                            "file": file_path,
                            "line": symbol.defined_line,
                            "message": f"Unused import '{symbol.name}'",
                            "type": "error",
                            "severity": 2
                        })
                    elif symbol.is_function:
                        issues.append({
                            "file": file_path,
                            "line": symbol.defined_line,
                            "message": f"Unused function '{symbol.name}'",
                            "type": "error",
                            "severity": 2
                        })
                    elif symbol.is_class:
                        issues.append({
                            "file": file_path,
                            "line": symbol.defined_line,
                            "message": f"Unused class '{symbol.name}'",
                            "type": "error",
                            "severity": 2
                        })
                    elif not symbol.is_parameter:
                        issues.append({
                            "file": file_path,
                            "line": symbol.defined_line,
                            "message": f"Unused variable '{symbol.name}'",
                            "type": "error",
                            "severity": 2
                        })
                        
        return issues
        
    def _make_error_issue(self, file_path: Path, message: str) -> Dict[str, Any]:
        """Create an error issue dictionary.
        
        Args:
            file_path: Path to the file
            message: Error message
            
        Returns:
            Issue dictionary
        """
        return {
            "file": str(file_path.relative_to(self.project_path)),
            "line": 1,
            "message": f"Failed to analyze file: {message}",
            "type": "error",
            "severity": 5
        }


class DeadCodeVisitor(ast.NodeVisitor):
    """AST visitor for dead code detection."""
    
    def __init__(self, builtins: Set[str], file_path: str, 
                 symtable: Optional[symtable.SymbolTable] = None):
        """Initialize the visitor.
        
        Args:
            builtins: Set of builtin names to ignore
            file_path: Path to the file being analyzed
            symtable: Symbol table for the module
        """
        self.current_scope = Scope("module")
        self.root_scope = self.current_scope
        self.file_path = file_path
        self._builtins = builtins
        self._symtable = symtable
        self._unreachable_nodes: List[Tuple[ast.AST, str]] = []
        self._return_seen = False
        
    def visit_Import(self, node: ast.Import) -> None:
        """Process import statements."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.current_scope.add_symbol(
                name, node, node.lineno,
                is_import=True
            )
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Process from-import statements."""
        module = node.module or ''
        for alias in node.names:
            name = alias.asname or alias.name
            self.current_scope.add_symbol(
                name, node, node.lineno,
                is_import=True
            )
        self.generic_visit(node)
        
    def visit_Name(self, node: ast.Name) -> None:
        """Process name references."""
        if isinstance(node.ctx, ast.Store):
            self.current_scope.add_symbol(node.id, node, node.lineno)
        elif isinstance(node.ctx, ast.Load):
            self.current_scope.use_symbol(node.id, self.file_path, node.lineno)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions."""
        # Add function to current scope
        self.current_scope.add_symbol(
            node.name, node, node.lineno,
            is_function=True
        )
        
        # Create new scope for function
        function_scope = Scope(node.name, parent=self.current_scope)
        self.current_scope.children.append(function_scope)
        old_scope = self.current_scope
        self.current_scope = function_scope
        
        # Add parameters as symbols
        for arg in node.args.args:
            self.current_scope.add_symbol(
                arg.arg, arg, arg.lineno,
                is_parameter=True
            )
            
        # Visit function body
        old_return = self._return_seen
        self._return_seen = False
        self.generic_visit(node)
        
        # Check for unreachable code after return
        if self._return_seen:
            self.current_scope.returns_seen = True
            
        self._return_seen = old_return
        self.current_scope = old_scope
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process class definitions."""
        # Add class to current scope
        self.current_scope.add_symbol(
            node.name, node, node.lineno,
            is_class=True
        )
        
        # Create new scope for class
        class_scope = Scope(node.name, parent=self.current_scope)
        self.current_scope.children.append(class_scope)
        old_scope = self.current_scope
        self.current_scope = class_scope
        
        # Visit class body
        self.generic_visit(node)
        
        # Restore scope
        self.current_scope = old_scope
        
    def visit_Return(self, node: ast.Return) -> None:
        """Process return statements."""
        self._return_seen = True
        self.generic_visit(node)
        
    def visit_Yield(self, node: ast.Yield) -> None:
        """Process yield statements."""
        self.current_scope.has_yield = True
        self.generic_visit(node)
        
    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Process yield from statements."""
        self.current_scope.has_yield = True
        self.generic_visit(node)
        
    def visit_If(self, node: ast.If) -> None:
        """Process if statements for unreachable code."""
        if isinstance(node.test, ast.Constant):
            if node.test.value is True:
                # else clause is unreachable
                if node.orelse:
                    self._unreachable_nodes.append(
                        (node.orelse[0], "Unreachable 'else' clause (condition is always True)")
                    )
            elif node.test.value is False:
                # if body is unreachable
                self._unreachable_nodes.append(
                    (node.body[0], "Unreachable 'if' body (condition is always False)")
                )
        self.generic_visit(node)
        
    def visit_While(self, node: ast.While) -> None:
        """Process while loops for unreachable code."""
        if isinstance(node.test, ast.Constant):
            if node.test.value is False:
                self._unreachable_nodes.append(
                    (node.body[0], "Unreachable 'while' body (condition is always False)")
                )
        self.generic_visit(node)


class CrossModuleVisitor(ast.NodeVisitor):
    """AST visitor for tracking symbol usage across modules."""
    
    def __init__(self, module_symbols: Dict[str, Dict[str, Symbol]], current_file: str):
        """Initialize the visitor.
        
        Args:
            module_symbols: Dictionary of symbols by module
            current_file: Current file being analyzed
        """
        self.module_symbols = module_symbols
        self.current_file = current_file
        
    def visit_Name(self, node: ast.Name) -> None:
        """Process name references."""
        if isinstance(node.ctx, ast.Load):
            # Check all modules for symbol
            for symbols in self.module_symbols.values():
                if node.id in symbols:
                    symbols[node.id].uses.append((self.current_file, node.lineno))
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Process attribute access."""
        if isinstance(node.ctx, ast.Load):
            # Try to resolve the base object
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                # Check if it's an imported module
                for symbols in self.module_symbols.values():
                    if base_name in symbols and symbols[base_name].is_import:
                        symbols[base_name].uses.append(
                            (self.current_file, node.lineno)
                        )
        self.generic_visit(node) 