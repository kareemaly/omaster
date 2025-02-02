"""Shared types for quality analysis."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any


class QualityMetric(Enum):
    """Types of quality metrics."""
    COMPLEXITY = "complexity"
    DEAD_CODE = "dead_code"
    SIMILARITY = "similarity"
    STYLE = "style"
    ERROR = "error"
    DUPLICATION = "duplication"
    AST_SIMILARITY = "ast_similarity"
    TOKEN_SIMILARITY = "token_similarity"
    CFG_SIMILARITY = "cfg_similarity"
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class QualityIssue:
    """A code quality issue."""
    type: QualityMetric
    file_path: str  # Relative to project root
    line: int
    end_line: Optional[int]
    message: str
    severity: str  # critical, high, medium, low, info
    weight: float  # 0-1 scale
    details: Optional[Dict[str, Any]] = None