"""Step 4: Bump Version.

This step updates the version number based on the commit analysis.
"""
import logging
from pathlib import Path
from typing import Dict, Any
from ...core.errors import ReleaseError, ErrorCode
from ...utils.version import update_version
from ...ui.layout import ReleaseUI

logger = logging.getLogger(__name__)

def run(project_path: Path, bump_type: str, ui: ReleaseUI) -> bool:
    """Run version bump step.

    Args:
        project_path: Path to the project directory
        bump_type: Type of version bump (major, minor, patch)
        ui: UI manager instance

    Returns:
        bool: True if version bump succeeds

    Raises:
        ReleaseError: If version bump fails
    """
    ui.log("Starting version update...", style="blue")

    if not bump_type:
        ui.log("No version bump type provided", level="error")
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            "No version bump type provided"
        )

    try:
        ui.update_progress("Reading current version...", 65)
        ui.log("Reading current version...", level="debug")

        ui.update_progress("Updating version...", 70)
        ui.log(f"Bumping version ({bump_type})...", level="debug")

        new_version = update_version(project_path, bump_type)

        ui.update_progress(f"Version bumped to {new_version}", 75)
        ui.log(f"✓ Version bumped to {new_version}", style="green")
        return True

    except Exception as e:
        ui.log(f"Failed to bump version: {e}", level="error")
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            f"Failed to bump version: {str(e)}"
        )