"""Step 5: Publish.

This step publishes the package to PyPI.
"""
import subprocess
from pathlib import Path

from ...core.errors import ReleaseError, ErrorCode

def run_command(cmd: str, cwd: Path) -> bool:
    """Run a shell command."""
    try:
        subprocess.run(cmd.split(), check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.PUBLISH_ERROR,
            str(e)
        )

def run(project_path: Path) -> bool:
    """Run publish step.

    Args:
        project_path: Path to the project directory

    Returns:
        bool: True if publish succeeds

    Raises:
        ReleaseError: If publish fails
    """
    print("Step 5: Publishing package...")
    if not run_command("uv publish", project_path):
        return False

    print("✓ Package published successfully\n")
    return True