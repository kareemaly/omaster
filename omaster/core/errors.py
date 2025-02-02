"""Error handling for the release process."""
from enum import Enum
from typing import Dict, Optional
import traceback
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

console = Console()

class ErrorCode(Enum):
    """Error codes for the release process."""

    # General errors (100-199)
    UNKNOWN_ERROR = 100
    CONFIG_ERROR = 101
    VALIDATION_ERROR = 102
    ANALYSIS_ERROR = 103
    BUILD_ERROR = 104
    PUBLISH_ERROR = 105
    OPENAI_API_ERROR = 106
    GIT_NO_CHANGES = 107
    OPENAI_API_KEY_MISSING = 108
    GIT_OPERATION_FAILED = 109
    VERSION_UPDATE_FAILED = 110
    BUILD_FAILED = 111
    BUILD_CLEAN_FAILED = 112

# Error messages for each error code
ERROR_MESSAGES: Dict[ErrorCode, str] = {
    # General errors
    ErrorCode.UNKNOWN_ERROR: "[red]Unexpected error:[/red] {message}",
    ErrorCode.CONFIG_ERROR: "[red]Configuration error:[/red] {message}",
    ErrorCode.VALIDATION_ERROR: "[red]Validation failed:[/red] {message}",
    ErrorCode.ANALYSIS_ERROR: "[red]Analysis failed:[/red] {message}",
    ErrorCode.BUILD_ERROR: "[red]Build failed:[/red] {message}",
    ErrorCode.PUBLISH_ERROR: "[red]Publish failed:[/red] {message}",
    ErrorCode.OPENAI_API_ERROR: "[red]OpenAI API error:[/red] {message}",
    ErrorCode.GIT_NO_CHANGES: "[red]No git changes found:[/red] {message}",
    ErrorCode.OPENAI_API_KEY_MISSING: "[red]OpenAI API key missing:[/red] Please set OPENAI_API_KEY environment variable",
    ErrorCode.GIT_OPERATION_FAILED: "[red]Git operation failed:[/red] {message}",
    ErrorCode.VERSION_UPDATE_FAILED: "[red]Version update failed:[/red] {message}",
    ErrorCode.BUILD_FAILED: "[red]Build failed:[/red] {message}",
    ErrorCode.BUILD_CLEAN_FAILED: "[red]Build clean failed:[/red] {message}"
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
        template = ERROR_MESSAGES[self.code]
        formatted = template.format(message=self.message)

        if self.details:
            formatted += "\n\nDetails:"
            for key, value in self.details.items():
                formatted += f"\n• {key}: {value}"

        return formatted


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