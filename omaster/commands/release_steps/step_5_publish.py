"""Step 5: Publish package to PyPI."""
import subprocess
from pathlib import Path
from ...core.errors import ErrorCode, ReleaseError

def run_command(cmd: str, cwd: Path) -> bool:
    """Run a shell command."""
    try:
        subprocess.run(cmd.split(), check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in str(e):
            raise ReleaseError(ErrorCode.PACKAGE_EXISTS, str(e))
        raise ReleaseError(ErrorCode.PUBLISH_FAILED, str(e))

def run(project_path: Path) -> bool:
    """Publish package to PyPI.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        bool: True if publish successful, False otherwise
        
    Raises:
        ReleaseError: If publishing fails
    """
    print("Step 5: Publishing package...")
    try:
        if not run_command("uv publish", project_path):
            raise ReleaseError(ErrorCode.PUBLISH_FAILED)
        print("✓ Publish successful\n")
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in str(e):
            raise ReleaseError(ErrorCode.PACKAGE_EXISTS, str(e))
        raise ReleaseError(ErrorCode.PUBLISH_FAILED, str(e)) 