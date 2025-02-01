"""Step 4: Clean old builds and build package."""
import shutil
import subprocess
from pathlib import Path
from ...core.errors import ErrorCode, ReleaseError

def run_command(cmd: str, cwd: Path) -> bool:
    """Run a shell command."""
    try:
        subprocess.run(cmd.split(), check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(ErrorCode.BUILD_FAILED, str(e))

def clean_dist(project_path: Path) -> bool:
    """Clean up old build files."""
    try:
        dist_path = project_path / "dist"
        if dist_path.exists():
            shutil.rmtree(dist_path)
        dist_path.mkdir(exist_ok=True)
        return True
    except Exception as e:
        raise ReleaseError(ErrorCode.BUILD_CLEAN_FAILED, str(e))

def run(project_path: Path) -> bool:
    """Clean old builds and build package.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        bool: True if build successful, False otherwise
        
    Raises:
        ReleaseError: If cleaning or building fails
    """
    print("Step 4: Cleaning and building package...")
    
    # Clean dist
    if not clean_dist(project_path):
        raise ReleaseError(ErrorCode.BUILD_CLEAN_FAILED)
    print("✓ Clean successful")
    
    # Build
    try:
        if not run_command("uv build", project_path):
            raise ReleaseError(ErrorCode.BUILD_FAILED)
        print("✓ Build successful\n")
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(ErrorCode.BUILD_FAILED, str(e))