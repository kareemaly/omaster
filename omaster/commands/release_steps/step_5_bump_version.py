"""Step 4: Bump Version.

This step updates the version number based on the commit analysis.
"""
import logging
from pathlib import Path
from typing import Dict, Any
from rich.progress import Progress
from contextlib import nullcontext

from ...core.errors import ReleaseError, ErrorCode
from ...utils.version import update_version

logger = logging.getLogger(__name__)

def run(project_path: Path, bump_type: str, progress: Progress = None, task_id: int = None) -> bool:
    """Run version bump step.

    Args:
        project_path: Path to the project directory
        bump_type: Type of version bump (major, minor, patch)
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        bool: True if version bump succeeds

    Raises:
        ReleaseError: If version bump fails
    """
    # Create dummy progress if none provided
    dummy_progress = False
    if progress is None:
        from rich.console import Console
        progress = Progress(console=Console())
        task_id = progress.add_task("[magenta]Step 4: Version Update", total=100)
        dummy_progress = True

    try:
        with progress if dummy_progress else nullcontext():
            if progress and task_id:
                progress.update(task_id, advance=30, description="[magenta]Step 4: Reading current version...")

            if not bump_type:
                raise ReleaseError(
                    ErrorCode.VERSION_UPDATE_FAILED,
                    "No version bump type provided"
                )

            if progress and task_id:
                progress.update(task_id, advance=30, description="[magenta]Step 4: Updating version...")

            new_version = update_version(project_path, bump_type)

            if progress and task_id:
                progress.update(task_id, advance=40, description=f"[magenta]Step 4: Version bumped to {new_version}")
            return True

    except Exception as e:
        logger.error(f"Failed to bump version: {str(e)}")
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            f"Failed to bump version: {str(e)}"
        )
    finally:
        if dummy_progress:
            progress.stop()