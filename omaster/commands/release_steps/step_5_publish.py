"""Step 6: Publish.

This step publishes the package to PyPI.
"""
from pathlib import Path
from rich.progress import Progress
from contextlib import nullcontext

from ...core.errors import ReleaseError, ErrorCode
from ...utils import run_command

def run(project_path: Path, progress: Progress = None, task_id: int = None) -> bool:
    """Run publish step.

    Args:
        project_path: Path to the project directory
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        bool: True if publish succeeds

    Raises:
        ReleaseError: If publish fails
    """
    # Create dummy progress if none provided
    dummy_progress = False
    if progress is None:
        from rich.console import Console
        progress = Progress(console=Console())
        task_id = progress.add_task("[cyan]Step 6: Publish", total=100)
        dummy_progress = True

    try:
        with progress if dummy_progress else nullcontext():
            if progress and task_id:
                progress.update(task_id, advance=30, description="[cyan]Step 6: Publishing to PyPI...")

            if not run_command("uv publish", project_path, ErrorCode.PUBLISH_ERROR):
                return False

            if progress and task_id:
                progress.update(task_id, advance=70, description="[cyan]Step 6: Package published successfully")
            return True

    finally:
        if dummy_progress:
            progress.stop()