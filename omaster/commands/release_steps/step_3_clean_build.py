"""Step 5: Clean old builds and build package."""
import shutil
from pathlib import Path
from rich.progress import Progress
from ...core.errors import ErrorCode, ReleaseError
from ...utils import run_command

def clean_dist(project_path: Path, progress: Progress = None, task_id: int = None) -> bool:
    """Clean up old build files.
    
    Args:
        project_path: Path to project directory
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar
        
    Returns:
        bool: True if cleaning succeeds
    """
    try:
        if progress and task_id:
            progress.update(task_id, advance=10, description="[red]Step 5: Cleaning old builds...")
            
        dist_path = project_path / "dist"
        if dist_path.exists():
            shutil.rmtree(dist_path)
        dist_path.mkdir(exist_ok=True)
        
        if progress and task_id:
            progress.update(task_id, advance=20, description="[red]Step 5: Clean successful")
        return True
        
    except Exception as e:
        raise ReleaseError(ErrorCode.BUILD_CLEAN_FAILED, str(e))

def run(project_path: Path, progress: Progress = None, task_id: int = None) -> bool:
    """Clean old builds and build package.

    Args:
        project_path: Path to the project directory
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        bool: True if build successful

    Raises:
        ReleaseError: If cleaning or building fails
    """
    # Create dummy progress if none provided
    dummy_progress = False
    if progress is None:
        from rich.console import Console
        progress = Progress(console=Console())
        task_id = progress.add_task("[red]Step 5: Build", total=100)
        dummy_progress = True

    try:
        with progress if dummy_progress else nullcontext():
            # Clean dist directory
            if not clean_dist(project_path, progress, task_id):
                return False

            # Build package
            if progress and task_id:
                progress.update(task_id, advance=30, description="[red]Step 5: Building package...")
                
            if not run_command("uv build", project_path, ErrorCode.BUILD_FAILED):
                return False
                
            if progress and task_id:
                progress.update(task_id, advance=40, description="[red]Step 5: Build successful")
            return True

    finally:
        if dummy_progress:
            progress.stop()