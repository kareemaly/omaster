"""Error handling for the release process."""
from enum import Enum

class ErrorCode(Enum):
    """Error codes for the release pipeline."""

    UNKNOWN_ERROR = 100
    CONFIG_ERROR = 101
    CODE_QUALITY_ERROR = 102
    OPENAI_API_ERROR = 103
    BUILD_FAILED = 104
    BUILD_CLEAN_FAILED = 105
    PUBLISH_ERROR = 106
    GIT_NO_CHANGES = 107
    GIT_OPERATION_FAILED = 108
    VERSION_UPDATE_FAILED = 109
    OPENAI_API_KEY_MISSING = 110
    MISSING_PYPROJECT = 111
    INVALID_PYPROJECT = 112
    MISSING_README = 113
    INVALID_README = 114

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