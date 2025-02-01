"""Centralized error handling system."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCode(Enum):
    """Error codes for all possible errors in the system."""
    # Validation errors (100-199)
    VALIDATION_FAILED = 100
    MISSING_PYPROJECT = 101
    INVALID_PYPROJECT = 102
    MISSING_README = 103
    INVALID_README = 104
    
    # Git errors (200-299)
    GIT_NO_CHANGES = 200
    GIT_ADD_FAILED = 201
    GIT_COMMIT_FAILED = 202
    GIT_PUSH_FAILED = 203
    
    # Build errors (300-399)
    BUILD_CLEAN_FAILED = 300
    BUILD_FAILED = 301
    
    # Publish errors (400-499)
    PUBLISH_FAILED = 400
    PACKAGE_EXISTS = 401
    
    # Version errors (500-599)
    VERSION_BUMP_FAILED = 500
    INVALID_VERSION = 501
    
    # OpenAI errors (600-699)
    OPENAI_API_KEY_MISSING = 600
    OPENAI_API_ERROR = 601
    
    # System errors (900-999)
    UNKNOWN_ERROR = 999


@dataclass
class ErrorTemplate:
    """Template for an error message."""
    title: str
    description: str
    fix_instructions: str
    example: Optional[str] = None


ERROR_TEMPLATES = {
    ErrorCode.VALIDATION_FAILED: ErrorTemplate(
        title="Validation Failed",
        description="One or more validation checks failed",
        fix_instructions="Review the specific validation errors above and fix each one",
        example="See validation error messages for details"
    ),
    ErrorCode.MISSING_PYPROJECT: ErrorTemplate(
        title="Missing pyproject.toml",
        description="The pyproject.toml file is required but was not found",
        fix_instructions="Create a pyproject.toml file in your project root",
        example="""[project]
name = "your-package"
version = "0.1.0"
description = "Your package description"
"""
    ),
    ErrorCode.INVALID_PYPROJECT: ErrorTemplate(
        title="Invalid pyproject.toml",
        description="The pyproject.toml file is invalid or missing required fields",
        fix_instructions="Ensure all required fields are present and valid",
        example="""[project]
name = "package-name"  # Required
version = "0.1.0"     # Required
description = "..."   # Required
readme = "README.md"  # Required
requires-python = ">=3.8"
"""
    ),
    ErrorCode.MISSING_README: ErrorTemplate(
        title="Missing README.md",
        description="The README.md file is required but was not found",
        fix_instructions="Create a README.md file in your project root",
        example="""# Your Package Name

A brief description of your package.

## Installation

\`\`\`bash
uv pip install your-package
\`\`\`

## Usage

Basic usage examples...
"""
    ),
    ErrorCode.INVALID_README: ErrorTemplate(
        title="Invalid README.md",
        description="The README.md file is missing required sections or is too short",
        fix_instructions="Ensure README includes Installation and Usage sections, and is at least 50 characters",
        example="See README.md template above"
    ),
    ErrorCode.GIT_NO_CHANGES: ErrorTemplate(
        title="No Git Changes",
        description="No changes detected in git working directory",
        fix_instructions="Make changes to your code before releasing",
        example="git status to check current changes"
    ),
    ErrorCode.GIT_ADD_FAILED: ErrorTemplate(
        title="Git Add Failed",
        description="Failed to stage changes with git add",
        fix_instructions="Check git status and resolve any conflicts",
        example="git status to see what went wrong"
    ),
    ErrorCode.GIT_COMMIT_FAILED: ErrorTemplate(
        title="Git Commit Failed",
        description="Failed to commit changes",
        fix_instructions="Ensure you have configured git user.name and user.email",
        example="""git config --global user.name "Your Name"
git config --global user.email "you@example.com\""""
    ),
    ErrorCode.GIT_PUSH_FAILED: ErrorTemplate(
        title="Git Push Failed",
        description="Failed to push changes to remote repository",
        fix_instructions="Pull latest changes and resolve any conflicts",
        example="git pull --rebase"
    ),
    ErrorCode.BUILD_CLEAN_FAILED: ErrorTemplate(
        title="Build Clean Failed",
        description="Failed to clean old build files",
        fix_instructions="Manually remove dist directory and try again",
        example="rm -rf dist/*"
    ),
    ErrorCode.BUILD_FAILED: ErrorTemplate(
        title="Build Failed",
        description="Failed to build package",
        fix_instructions="Check build logs for specific errors",
        example="uv build"
    ),
    ErrorCode.PUBLISH_FAILED: ErrorTemplate(
        title="Publish Failed",
        description="Failed to publish package to PyPI",
        fix_instructions="Check PyPI credentials and package version",
        example="uv publish"
    ),
    ErrorCode.PACKAGE_EXISTS: ErrorTemplate(
        title="Package Already Exists",
        description="Package version already exists on PyPI",
        fix_instructions="Bump version number in pyproject.toml",
        example="Current: 0.1.0 -> New: 0.1.1"
    ),
    ErrorCode.VERSION_BUMP_FAILED: ErrorTemplate(
        title="Version Bump Failed",
        description="Failed to bump package version",
        fix_instructions="Manually update version in pyproject.toml",
        example="""[project]
version = "0.1.1"  # Increment version number"""
    ),
    ErrorCode.INVALID_VERSION: ErrorTemplate(
        title="Invalid Version",
        description="Package version is invalid",
        fix_instructions="Use semantic versioning (MAJOR.MINOR.PATCH)",
        example="0.1.0, 1.0.0, 2.3.4"
    ),
    ErrorCode.OPENAI_API_KEY_MISSING: ErrorTemplate(
        title="OpenAI API Key Missing",
        description="OPENAI_API_KEY environment variable not set",
        fix_instructions="Set OPENAI_API_KEY environment variable",
        example="export OPENAI_API_KEY='your-api-key'"
    ),
    ErrorCode.OPENAI_API_ERROR: ErrorTemplate(
        title="OpenAI API Error",
        description="Error calling OpenAI API",
        fix_instructions="Check API key and error message",
        example="Check OpenAI status page for service issues"
    ),
    ErrorCode.UNKNOWN_ERROR: ErrorTemplate(
        title="Unknown Error",
        description="An unexpected error occurred",
        fix_instructions="Check error message and stack trace",
        example="Contact support if issue persists"
    ),
}


class ReleaseError(Exception):
    """Custom exception for release errors."""
    def __init__(self, code: ErrorCode, context: Optional[str] = None):
        self.code = code
        self.context = context
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message."""
        template = ERROR_TEMPLATES[self.code]
        
        # Build the message
        message = [
            "🚨 Error 🚨",
            f"Code: {self.code.value} - {template.title}",
            "",
            "Description:",
            template.description,
            "",
            "How to fix:",
            template.fix_instructions,
        ]
        
        if template.example:
            message.extend([
                "",
                "Example:",
                template.example
            ])
            
        if self.context:
            message.extend([
                "",
                "Additional context:",
                self.context
            ])
            
        return "\n".join(message) 