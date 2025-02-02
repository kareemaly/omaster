"""Terminal UI layout manager."""
from typing import Dict, List
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.text import Text
from rich import box

class LogEntry:
    """A log entry with level and message."""
    def __init__(self, message: str, level: str = "info", style: str = "white"):
        self.message = message
        self.level = level
        self.style = style

class ReleaseUI:
    """Manages the terminal UI layout."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the UI manager.
        
        Args:
            verbose: Whether to show debug logs
        """
        self.console = Console()
        self.verbose = verbose
        
        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
            console=self.console
        )
        self.task_id = None
        
        # Create log buffers by level
        self.debug_logs: List[LogEntry] = []
        self.info_logs: List[LogEntry] = []
        self.error_logs: List[LogEntry] = []
        
    def __enter__(self):
        """Start the UI."""
        self.console.clear()
        self.task_id = self.progress.add_task("", total=100)
        return self
        
    def __exit__(self, *_):
        """Clean up the UI and show final logs."""
        self.progress.stop()
        self.console.line()
        
        # Show debug logs if verbose
        if self.verbose and self.debug_logs:
            self.console.print(Panel(
                Group(*[Text(log.message, style="dim") for log in self.debug_logs]),
                title="Debug Logs",
                border_style="dim",
                box=box.ROUNDED,
                padding=(1, 2)
            ))
            self.console.line()
        
        # Show info logs
        if self.info_logs:
            self.console.print(Panel(
                Group(*[Text(log.message, style=log.style) for log in self.info_logs]),
                title="Release Summary",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2)
            ))
            self.console.line()
        
        # Show error logs
        if self.error_logs:
            self.console.print(Panel(
                Group(*[Text(log.message, style="red") for log in self.error_logs]),
                title="Errors",
                border_style="red",
                box=box.ROUNDED,
                padding=(1, 2)
            ))
            self.console.line()
        
    def update_progress(self, description: str, percentage: int):
        """Update the progress bar.
        
        Args:
            description: Current step description
            percentage: Progress percentage (0-100)
        """
        self.progress.update(
            self.task_id,
            description=description,
            completed=percentage
        )
        
    def log(self, message: str, level: str = "info", style: str = "white"):
        """Add a log message.
        
        Args:
            message: The message to log
            level: Log level (debug, info, error)
            style: Rich style string for the message
        """
        entry = LogEntry(message, level, style)
        if level == "debug":
            self.debug_logs.append(entry)
        elif level == "error":
            self.error_logs.append(entry)
        else:
            self.info_logs.append(entry)
        
    def log_error(self, error: Exception):
        """Log an error message.
        
        Args:
            error: The error to log
        """
        if hasattr(error, 'code'):
            self.log(f"🚨 Error 🚨\n\nCode: {error.code.value} - {error.code.name}\n\n{str(error)}", level="error")
        else:
            self.log(f"🚨 Error 🚨\n\n{str(error)}", level="error") 