"""Step 5: Publish.

This step publishes the package to PyPI.
"""
from pathlib import Path
from ...core.errors import ReleaseError, ErrorCode
from ...utils import run_command

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
    if not run_command("uv publish", project_path, ErrorCode.PUBLISH_ERROR):
        return False

    print("✓ Package published successfully\n")
    return True