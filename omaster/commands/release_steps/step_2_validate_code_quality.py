"""Step 2: Validate code quality using radon, jscpd, and vulture."""
import subprocess
import json
from pathlib import Path
from typing import Dict, Any
from rich.progress import Progress

from ...core.errors import ReleaseError, ErrorCode
from ...core.config import Config, load_config

def run_radon(project_path: Path, max_complexity: int = 15, progress: Progress = None, task_id: int = None) -> bool:
    """Run radon complexity check.

    Args:
        project_path: Path to project directory
        max_complexity: Maximum allowed complexity
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        bool: True if check passes
    """
    try:
        if progress and task_id:
            progress.update(task_id, advance=10, description="[yellow]Step 2: Running complexity analysis...")
            
        result = subprocess.run(
            ["radon", "cc", "-j", "-n", "C", str(project_path)],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        
        # Check each file's complexity
        for file_path, blocks in data.items():
            for block in blocks:
                if block["complexity"] > max_complexity:
                    raise ReleaseError(
                        ErrorCode.CODE_QUALITY_ERROR,
                        f"Code complexity too high in {file_path}: {block['name']} ({block['complexity']})"
                    )
                    
        if progress and task_id:
            progress.update(task_id, advance=20, description="[yellow]Step 2: Complexity check passed")
        return True
        
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run radon: {str(e)}"
        )

def run_jscpd(project_path: Path, max_duplicates: float = 1.0) -> bool:
    """Run jscpd duplicate code check.

    Args:
        project_path: Path to project directory
        max_duplicates: Maximum allowed duplicate percentage

    Returns:
        bool: True if check passes
    """
    try:
        # Run jscpd without --json flag as it's not supported in newer versions
        result = subprocess.run(
            ["jscpd", str(project_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to find duplication percentage
        for line in result.stdout.splitlines():
            if "%" in line and "clones found" in line.lower():
                percentage = float(line.split("%")[0].strip().split()[-1])
                if percentage > max_duplicates:
                    raise ReleaseError(
                        ErrorCode.CODE_QUALITY_ERROR,
                        f"Too much code duplication: {percentage}% (max {max_duplicates}%)"
                    )
                break
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run jscpd: {str(e)}"
        )

def run_vulture(progress: Progress = None, task_id: int = None) -> bool:
    """Run vulture to check for dead code.
    
    Args:
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar
        
    Returns:
        bool: True if check passes
    """
    try:
        if progress and task_id:
            progress.update(task_id, advance=10, description="[yellow]Step 2: Checking for dead code...")
            
        result = subprocess.run(
            ["vulture", "omaster", "--exclude", ".venv/*"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ReleaseError(
                ErrorCode.CODE_QUALITY_ERROR,
                f"Dead code check failed with exit status {result.returncode}:\n{result.stdout}",
            )
            
        if progress and task_id:
            progress.update(task_id, advance=20, description="[yellow]Step 2: Dead code check passed")
        return True
        
    except Exception as e:
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run vulture: {str(e)}",
        )

def run(project_path: Path, progress: Progress = None, task_id: int = None) -> bool:
    """Run code quality validation.

    Args:
        project_path: Path to the project directory
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        bool: True if validation passes

    Raises:
        ReleaseError: If validation fails
    """
    # Create dummy progress if none provided
    dummy_progress = False
    if progress is None:
        from rich.console import Console
        progress = Progress(console=Console())
        task_id = progress.add_task("[yellow]Step 2: Code Quality", total=100)
        dummy_progress = True

    try:
        with progress if dummy_progress else nullcontext():
            # Load configuration
            if progress and task_id:
                progress.update(task_id, advance=10, description="[yellow]Step 2: Loading configuration...")
            
            config_data = load_config(project_path)
            config = Config(config_data)

            # Get thresholds from config or use defaults
            quality_config = config.data.get("quality", {})
            max_complexity = quality_config.get("max_complexity", 15)
            max_duplicates = quality_config.get("max_duplicates", 1.0)

            # Run radon
            if not run_radon(project_path, max_complexity, progress, task_id):
                return False

            # Run jscpd
            print("Running duplication check...")
            if run_jscpd(project_path, max_duplicates):
                print("✓ Duplication check passed")

            # Run vulture
            if not run_vulture(progress, task_id):
                return False

            if progress and task_id:
                progress.update(task_id, advance=40, description="[yellow]Step 2: All quality checks passed")
            return True

    finally:
        if dummy_progress:
            progress.stop() 