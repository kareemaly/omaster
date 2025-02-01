"""Code similarity analyzer for detecting duplicated and similar code.

This analyzer implements advanced code similarity detection:
1. Exact code duplication detection
2. AST structural similarity analysis
3. Token-based fuzzy matching
4. Control flow graph similarity
5. Semantic similarity using embeddings
6. Context-aware code block extraction
"""
import ast
import difflib
import tokenize
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
import networkx as nx

from .base import BaseAnalyzer


@dataclass
class CodeBlock:
    """Represents a block of code for similarity analysis."""
    
    file: str
    start_line: int
    end_line: int
    content: str
    normalized: str  # Normalized content for comparison
    tokens: List[str]  # Tokenized representation
    ast_hash: str  # Hash of AST structure
    cfg_hash: str  # Hash of control flow graph
    semantic_vec: Optional[List[float]] = None  # Semantic embedding
    
    @property
    def size(self) -> int:
        """Get the size of the block in lines."""
        return self.end_line - self.start_line + 1


class SimilarityAnalyzer(BaseAnalyzer):
    """Analyzer for detecting code duplication and similarity."""
    
    def __init__(self, project_path: Path, config: Dict[str, Any]):
        """Initialize the similarity analyzer.
        
        Args:
            project_path: Path to the project root directory
            config: Configuration dictionary with thresholds and weights
        """
        super().__init__(project_path, config)
        
        # Get similarity thresholds from config
        similarity_config = config["quality"]["similarity"]
        self.min_lines = similarity_config["min_lines"]
        self.exact_match_threshold = similarity_config["exact_match_threshold"]
        self.ast_similarity_threshold = similarity_config["ast_similarity_threshold"]
        self.token_similarity_threshold = similarity_config["token_similarity_threshold"]
        self.cfg_similarity_threshold = similarity_config["cfg_similarity_threshold"]
        self.semantic_similarity_threshold = similarity_config["semantic_similarity_threshold"]
        
        # Get metric weights
        self.weights = similarity_config["weights"]
        
        # Get severity weights
        self.severity_weights = config["quality"]["severity_weights"]
        
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze code for duplicated or similar code blocks.
        
        The analysis is performed using multiple techniques:
        1. Exact hash matching
        2. AST structural comparison
        3. Token sequence alignment
        4. Control flow graph matching
        5. Semantic similarity comparison
        
        Returns:
            List of similarity issues found
        """
        issues = []
        code_blocks: List[CodeBlock] = []
        
        # First pass: collect and analyze code blocks
        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue
                
            try:
                with open(file_path) as f:
                    content = f.read()
                    
                blocks = self._extract_code_blocks(
                    str(file_path.relative_to(self.project_path)),
                    content
                )
                code_blocks.extend(blocks)
                
            except Exception as e:
                issues.append({
                    "file": str(file_path.relative_to(self.project_path)),
                    "line": 1,
                    "message": f"Failed to analyze file: {str(e)}",
                    "type": "error",
                    "severity": 5
                })
                
        # Second pass: multi-level similarity detection
        exact_duplicates = self._find_exact_duplicates(code_blocks)
        ast_similar = self._find_ast_similar_blocks(code_blocks)
        token_similar = self._find_token_similar_blocks(code_blocks)
        cfg_similar = self._find_cfg_similar_blocks(code_blocks)
        semantic_similar = self._find_semantic_similar_blocks(code_blocks)
        
        # Report exact duplicates (highest priority)
        for blocks in exact_duplicates:
            primary = blocks[0]
            others = blocks[1:]
            
            for other in others:
                issues.append({
                    "file": other.file,
                    "line": other.start_line,
                    "message": (f"Code block is identical to block at {primary.file}:"
                              f"{primary.start_line}-{primary.end_line} "
                              f"({other.size} lines)"),
                    "type": "error",
                    "severity": 4
                })
                
        # Report AST structural similarities
        for block1, block2, similarity in ast_similar:
            issues.append({
                "file": block2.file,
                "line": block2.start_line,
                "message": (f"Code block has {similarity:.1%} structural similarity "
                          f"with block at {block1.file}:{block1.start_line}"),
                "type": "warning",
                "severity": 3
            })
            
        # Report token sequence similarities
        for block1, block2, similarity in token_similar:
            issues.append({
                "file": block2.file,
                "line": block2.start_line,
                "message": (f"Code block has {similarity:.1%} token sequence similarity "
                          f"with block at {block1.file}:{block1.start_line}"),
                "type": "warning",
                "severity": 2
            })
            
        # Report control flow similarities
        for block1, block2, similarity in cfg_similar:
            issues.append({
                "file": block2.file,
                "line": block2.start_line,
                "message": (f"Code block has {similarity:.1%} control flow similarity "
                          f"with block at {block1.file}:{block1.start_line}"),
                "type": "warning",
                "severity": 2
            })
            
        # Report semantic similarities
        for block1, block2, similarity in semantic_similar:
            issues.append({
                "file": block2.file,
                "line": block2.start_line,
                "message": (f"Code block has {similarity:.1%} semantic similarity "
                          f"with block at {block1.file}:{block1.start_line}"),
                "type": "info",
                "severity": 1
            })
            
        return issues
        
    def _extract_code_blocks(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract code blocks from file content.
        
        Uses smart block boundary detection:
        1. Function and class definitions
        2. Logical code groups (imports, setup, etc.)
        3. Nested function handling
        4. Comment-separated blocks
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of code blocks
        """
        blocks = []
        
        try:
            # Tokenize the content first
            tokens = list(tokenize.generate_tokens(StringIO(content).readline))
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Track logical groups
            current_group = []
            current_group_start = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Get the source lines for this node
                    start = node.lineno
                    end = node.end_lineno or start
                    
                    if end - start + 1 < self.min_lines:
                        continue
                        
                    # Extract the block content
                    block_content = "\n".join(content.splitlines()[start-1:end])
                    
                    # Create normalized versions
                    normalized = self._normalize_code(block_content)
                    token_list = self._tokenize_code(block_content)
                    ast_hash = self._hash_ast_structure(node)
                    cfg_hash = self._hash_control_flow(node)
                    
                    blocks.append(CodeBlock(
                        file=file_path,
                        start_line=start,
                        end_line=end,
                        content=block_content,
                        normalized=normalized,
                        tokens=token_list,
                        ast_hash=ast_hash,
                        cfg_hash=cfg_hash
                    ))
                    
                    # Handle nested functions
                    if isinstance(node, ast.FunctionDef):
                        for child in ast.walk(node):
                            if (isinstance(child, ast.FunctionDef) and 
                                child is not node):
                                nested_start = child.lineno
                                nested_end = child.end_lineno or nested_start
                                
                                if nested_end - nested_start + 1 < self.min_lines:
                                    continue
                                    
                                nested_content = "\n".join(
                                    content.splitlines()[nested_start-1:nested_end]
                                )
                                
                                blocks.append(CodeBlock(
                                    file=file_path,
                                    start_line=nested_start,
                                    end_line=nested_end,
                                    content=nested_content,
                                    normalized=self._normalize_code(nested_content),
                                    tokens=self._tokenize_code(nested_content),
                                    ast_hash=self._hash_ast_structure(child),
                                    cfg_hash=self._hash_control_flow(child)
                                ))
                                
        except Exception:
            # If parsing fails, fall back to simpler extraction
            lines = content.splitlines()
            i = 0
            while i < len(lines):
                if len(lines[i].strip()) > 0:
                    # Found non-empty line, look for block
                    start = i + 1
                    while (i < len(lines) and 
                           (len(lines[i].strip()) > 0 or 
                            i < start + self.min_lines)):
                        i += 1
                    end = i
                    
                    if end - start + 1 >= self.min_lines:
                        block_content = "\n".join(lines[start-1:end])
                        blocks.append(CodeBlock(
                            file=file_path,
                            start_line=start,
                            end_line=end,
                            content=block_content,
                            normalized=block_content,  # No normalization in fallback
                            tokens=block_content.split(),  # Simple tokenization
                            ast_hash="",  # No AST in fallback
                            cfg_hash=""  # No CFG in fallback
                        ))
                i += 1
                
        return blocks
        
    def _normalize_code(self, content: str) -> str:
        """Normalize code for comparison.
        
        Enhanced normalization:
        1. Remove comments and docstrings
        2. Normalize whitespace
        3. Type-aware variable renaming
        4. Preserve semantic structure
        5. Handle string literals and numbers
        
        Args:
            content: Code content to normalize
            
        Returns:
            Normalized code
        """
        try:
            # Parse the code
            tree = ast.parse(content)
            
            # Remove docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if (ast.get_docstring(node) and 
                        node.body and 
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Str)):
                        node.body = node.body[1:]
                        
            # Create a normalized version
            class TypeAwareNormalizer(ast.NodeTransformer):
                def __init__(self):
                    self.type_counters = defaultdict(int)
                    self.var_map = {}
                    
                def visit_Name(self, node):
                    # Type-aware variable renaming
                    if isinstance(node.ctx, ast.Store):
                        var_type = self._infer_type(node)
                        if node.id not in self.var_map:
                            self.type_counters[var_type] += 1
                            self.var_map[node.id] = f"{var_type}{self.type_counters[var_type]}"
                        node.id = self.var_map[node.id]
                    elif node.id in self.var_map:
                        node.id = self.var_map[node.id]
                    return node
                    
                def visit_Str(self, node):
                    # Preserve string length characteristics
                    return ast.Str("S" * min(len(node.s), 3))
                    
                def visit_Num(self, node):
                    # Preserve number type
                    if isinstance(node.n, int):
                        return ast.Num(0)
                    return ast.Num(0.0)
                    
                def _infer_type(self, node):
                    # Simple type inference
                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.ClassDef):
                        return "cls"
                    elif isinstance(parent, ast.FunctionDef):
                        return "func"
                    elif isinstance(parent, ast.For):
                        return "iter"
                    return "var"
                    
            normalized = TypeAwareNormalizer().visit(tree)
            return ast.unparse(normalized)
            
        except Exception:
            # If normalization fails, return original
            return content
            
    def _tokenize_code(self, content: str) -> List[str]:
        """Tokenize code for sequence comparison.
        
        Args:
            content: Code content to tokenize
            
        Returns:
            List of tokens
        """
        tokens = []
        try:
            for tok in tokenize.generate_tokens(StringIO(content).readline):
                if tok.type in {tokenize.NAME, tokenize.OP, tokenize.KEYWORD}:
                    tokens.append(tok.string)
        except Exception:
            # Fallback to simple splitting
            tokens = content.split()
        return tokens
        
    def _hash_ast_structure(self, node: ast.AST) -> str:
        """Create a hash of the AST structure.
        
        Preserves structural information while ignoring specific values.
        
        Args:
            node: AST node to hash
            
        Returns:
            Hash of the AST structure
        """
        def serialize_node(node):
            if isinstance(node, ast.AST):
                fields = []
                for field, value in ast.iter_fields(node):
                    if field not in {'lineno', 'col_offset', 'end_lineno', 'end_col_offset'}:
                        fields.append(f"{field}:{serialize_node(value)}")
                return f"{type(node).__name__}({','.join(fields)})"
            elif isinstance(node, list):
                return f"[{','.join(serialize_node(x) for x in node)}]"
            return "_"
            
        return sha256(serialize_node(node).encode()).hexdigest()
        
    def _hash_control_flow(self, node: ast.AST) -> str:
        """Create a hash of the control flow graph.
        
        Args:
            node: AST node to analyze
            
        Returns:
            Hash of the control flow graph
        """
        def build_cfg(node):
            graph = nx.DiGraph()
            current_block = []
            block_id = 0
            
            def add_block():
                nonlocal block_id
                if current_block:
                    graph.add_node(block_id, statements=current_block[:])
                    block_id += 1
                    current_block.clear()
                return block_id - 1
                
            def visit(node, parent_id=None):
                nonlocal block_id
                
                if isinstance(node, ast.If):
                    # Add condition block
                    current_block.append('if')
                    cond_id = add_block()
                    if parent_id is not None:
                        graph.add_edge(parent_id, cond_id)
                        
                    # Process true branch
                    for stmt in node.body:
                        visit(stmt, cond_id)
                        
                    # Process false branch
                    if node.orelse:
                        for stmt in node.orelse:
                            visit(stmt, cond_id)
                            
                elif isinstance(node, (ast.For, ast.While)):
                    # Add loop block
                    current_block.append('loop')
                    loop_id = add_block()
                    if parent_id is not None:
                        graph.add_edge(parent_id, loop_id)
                        
                    # Process loop body
                    for stmt in node.body:
                        visit(stmt, loop_id)
                    graph.add_edge(loop_id, loop_id)  # Loop back edge
                    
                else:
                    current_block.append(type(node).__name__)
                    
            visit(node)
            add_block()  # Add any remaining statements
            return graph
            
        try:
            cfg = build_cfg(node)
            return sha256(str(sorted(cfg.edges())).encode()).hexdigest()
        except Exception:
            return ""
            
    def _find_exact_duplicates(self, blocks: List[CodeBlock]) -> List[List[CodeBlock]]:
        """Find exactly duplicated code blocks.
        
        Args:
            blocks: List of code blocks to compare
            
        Returns:
            List of duplicate block groups
        """
        duplicates: Dict[str, List[CodeBlock]] = defaultdict(list)
        
        # Group blocks by normalized content hash
        for block in blocks:
            hash_key = sha256(block.normalized.encode()).hexdigest()
            duplicates[hash_key].append(block)
            
        # Return only groups with multiple blocks
        return [group for group in duplicates.values() if len(group) > 1]
        
    def _find_ast_similar_blocks(self, 
                                blocks: List[CodeBlock]) -> List[Tuple[CodeBlock, CodeBlock, float]]:
        """Find blocks with similar AST structure.
        
        Args:
            blocks: List of code blocks to compare
            
        Returns:
            List of (block1, block2, similarity_ratio) tuples
        """
        similar = []
        seen: Set[Tuple[str, str]] = set()
        
        for i, block1 in enumerate(blocks):
            for block2 in blocks[i+1:]:
                # Skip if already compared or from same file
                if (block1.file == block2.file or
                    (block1.ast_hash, block2.ast_hash) in seen):
                    continue
                    
                # Compare AST hashes
                if block1.ast_hash and block2.ast_hash:
                    similarity = difflib.SequenceMatcher(
                        None, 
                        block1.ast_hash, 
                        block2.ast_hash
                    ).ratio()
                    
                    if similarity >= self.ast_similarity_threshold:
                        similar.append((block1, block2, similarity))
                        
                seen.add((block1.ast_hash, block2.ast_hash))
                seen.add((block2.ast_hash, block1.ast_hash))
                
        return similar
        
    def _find_token_similar_blocks(self, 
                                  blocks: List[CodeBlock]) -> List[Tuple[CodeBlock, CodeBlock, float]]:
        """Find blocks with similar token sequences.
        
        Args:
            blocks: List of code blocks to compare
            
        Returns:
            List of (block1, block2, similarity_ratio) tuples
        """
        similar = []
        seen: Set[Tuple[str, str]] = set()
        
        for i, block1 in enumerate(blocks):
            for block2 in blocks[i+1:]:
                # Skip if already compared or from same file
                if (block1.file == block2.file or
                    (block1.ast_hash, block2.ast_hash) in seen):
                    continue
                    
                # Compare token sequences
                similarity = difflib.SequenceMatcher(
                    None,
                    block1.tokens,
                    block2.tokens
                ).ratio()
                
                if similarity >= self.token_similarity_threshold:
                    similar.append((block1, block2, similarity))
                    
                seen.add((block1.ast_hash, block2.ast_hash))
                seen.add((block2.ast_hash, block1.ast_hash))
                
        return similar
        
    def _find_cfg_similar_blocks(self, 
                                blocks: List[CodeBlock]) -> List[Tuple[CodeBlock, CodeBlock, float]]:
        """Find blocks with similar control flow.
        
        Args:
            blocks: List of code blocks to compare
            
        Returns:
            List of (block1, block2, similarity_ratio) tuples
        """
        similar = []
        seen: Set[Tuple[str, str]] = set()
        
        for i, block1 in enumerate(blocks):
            for block2 in blocks[i+1:]:
                # Skip if already compared or from same file
                if (block1.file == block2.file or
                    (block1.cfg_hash, block2.cfg_hash) in seen):
                    continue
                    
                # Compare control flow graphs
                if block1.cfg_hash and block2.cfg_hash:
                    similarity = difflib.SequenceMatcher(
                        None,
                        block1.cfg_hash,
                        block2.cfg_hash
                    ).ratio()
                    
                    if similarity >= self.cfg_similarity_threshold:
                        similar.append((block1, block2, similarity))
                        
                seen.add((block1.cfg_hash, block2.cfg_hash))
                seen.add((block2.cfg_hash, block1.cfg_hash))
                
        return similar
        
    def _find_semantic_similar_blocks(self, 
                                    blocks: List[CodeBlock]) -> List[Tuple[CodeBlock, CodeBlock, float]]:
        """Find blocks with similar semantic meaning.
        
        Note: This is a placeholder for semantic similarity.
        In a real implementation, this would use code embeddings
        or other semantic analysis techniques.
        
        Args:
            blocks: List of code blocks to compare
            
        Returns:
            List of (block1, block2, similarity_ratio) tuples
        """
        # Placeholder for semantic similarity
        return [] 