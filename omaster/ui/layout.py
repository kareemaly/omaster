"""Terminal UI layout manager."""
from typing import Dict
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

class ReleaseUI:
    """Manages the terminal UI layout."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the UI manager.
        
        Args:
            verbose: Whether to show debug logs
        """
        self.console = Console()
        self.verbose = verbose
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )
        self.task_id = None
        
    def __enter__(self):
        """Start the UI."""
        self.task_id = self.progress.add_task("Starting...", total=100)
        self.progress.start()
        return self
        
    def __exit__(self, *_):
        """Clean up the UI."""
        self.progress.stop()
        
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
        if level == "debug" and not self.verbose:
            return
            
        if style == "green":
            message = f"✓ {message}"
        elif style == "red":
            message = f"✗ {message}"
            
        self.console.print(message, style=style)
        
    def log_error(self, error: Exception):
        """Log an error message.
        
        Args:
            error: The error to log
        """
        if hasattr(error, 'code'):
            self.log(f"Error: {error.code.name} - {str(error)}", level="error", style="red")
        else:
            self.log(f"Error: {str(error)}", level="error", style="red") 