"""Error handling for the release process."""
from enum import Enum

class ErrorCode(Enum):
    """Error codes for the release pipeline."""

    UNKNOWN_ERROR = 100
    CONFIG_ERROR = 101
    CODE_QUALITY_ERROR = 102
    OPENAI_API_ERROR = 103

class ReleaseError(Exception):
    """Base class for release pipeline errors."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Initialize the error.
        
        Args:
            code: The error code
            message: The error message
        """
        self.code = code
        self.message = message
        super().__init__(message)