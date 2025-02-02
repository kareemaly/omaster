"""Error handling for the release process."""
from enum import IntEnum
from typing import Dict, Optional
import traceback
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

console = Console()

class ErrorCode(IntEnum):
    """Error codes for the release process."""

    # General errors (1-99)
    UNEXPECTED_ERROR = 1
    CONFIG_ERROR = 2

    # Validation errors (100-199)
    PYPROJECT_ERROR = 100
    README_ERROR = 101
    VERSION_ERROR = 102

    # Quality analysis errors (200-299)
    ANALYZER_INIT_ERROR = 200
    ANALYSIS_ERROR = 201
    QUALITY_ERROR = 202

    # Git errors (300-399)
    GIT_ERROR = 300

    # Build errors (400-499)
    BUILD_ERROR = 400

    # Publish errors (500-599)
    PUBLISH_ERROR = 500


# Error message templates with rich formatting
ERROR_TEMPLATES = {
    # General errors
    ErrorCode.UNEXPECTED_ERROR: "[red]An unexpected error occurred:[/red] {message}",
    ErrorCode.CONFIG_ERROR: "[red]Configuration error:[/red] {message}",

    # Validation errors
    ErrorCode.PYPROJECT_ERROR: "[red]pyproject.toml error:[/red] {message}",
    ErrorCode.README_ERROR: "[red]README.md error:[/red] {message}",
    ErrorCode.VERSION_ERROR: "[red]Version error:[/red] {message}",

    # Quality analysis errors
    ErrorCode.ANALYZER_INIT_ERROR: "[red]Failed to initialize analyzer:[/red] {message}",
    ErrorCode.ANALYSIS_ERROR: "[red]Analysis error:[/red] {message}",
    ErrorCode.QUALITY_ERROR: "[red]Quality check failed:[/red] {message}",

    # Git errors
    ErrorCode.GIT_ERROR: "[red]Git error:[/red] {message}",

    # Build errors
    ErrorCode.BUILD_ERROR: "[red]Build error:[/red] {message}",

    # Publish errors
    ErrorCode.PUBLISH_ERROR: "[red]Publish error:[/red] {message}"
}


class ReleaseError(Exception):
    """Custom exception for release process errors."""

    def __init__(self, code: ErrorCode, message: str, details: Optional[Dict] = None):
        """Initialize the error.

        Args:
            code: Error code
            message: Error message
            details: Optional error details
        """
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message using the template.

        Returns:
            Formatted error message
        """
        template = ERROR_TEMPLATES[self.code]
        formatted = template.format(message=self.message)

        if self.details:
            formatted += "\n\nDetails:"
            for key, value in self.details.items():
                formatted += f"\n• {key}: {value}"

        return formatted

    def with_traceback(self) -> str:
        """Get error message with traceback.

        Returns:
            Error message with traceback
        """
        return f"{self._format_message()}\n\n{traceback.format_exc()}"


def handle_error(error: Exception) -> None:
    """Global error handler.

    Args:
        error: The exception to handle
    """
    console.print("\n")
    
    if isinstance(error, ReleaseError):
        # Create main error panel
        error_text = Text()
        error_text.append("🚨 Error 🚨\n", style="red bold")
        error_text.append(f"Code: {error.code.value} - {error.code.name}\n\n", style="yellow")
        error_text.append(error.message)

        main_panel = Panel(
            error_text,
            title="[red]Release Pipeline Error",
            border_style="red"
        )
        console.print(main_panel)

        # Show details if present
        if error.details:
            details_text = Text()
            for key, value in error.details.items():
                if key == "traceback" and value is True:
                    continue  # Skip traceback flag, we'll handle it separately
                details_text.append(f"• {key}: ", style="yellow")
                details_text.append(f"{value}\n", style="white")

            details_panel = Panel(
                details_text,
                title="[yellow]Additional Details",
                border_style="yellow"
            )
            console.print(details_panel)

        # Show traceback if requested
        if error.details and error.details.get("traceback"):
            console.print("\n[red]Traceback:[/red]")
            console.print(Traceback.from_exception(
                type(error),
                error,
                traceback.extract_tb(sys.exc_info()[2])
            ))
    else:
        # Handle unexpected exceptions
        error_panel = Panel(
            Text(str(error), style="red"),
            title="[red]Unexpected Error",
            border_style="red"
        )
        console.print(error_panel)
        console.print("\n[red]Traceback:[/red]")
        console.print(Traceback.from_exception(
            type(error),
            error,
            traceback.extract_tb(sys.exc_info()[2])
        ))

    console.print("\n")
    sys.exit(1)  # Exit with failure code