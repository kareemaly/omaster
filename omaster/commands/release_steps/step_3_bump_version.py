"""Step 3: Bump Version.

This step updates the version number based on the commit analysis.
"""
import logging
from pathlib import Path
from typing import Dict, Any

from ...core.errors import ReleaseError, ErrorCode
from ...utils.version import update_version

logger = logging.getLogger(__name__)

def run(project_path: Path, commit_info: Dict[str, Any]) -> bool:
    """Run version bump step.

    Args:
        project_path: Path to the project directory
        commit_info: Commit information from analysis step

    Returns:
        bool: True if version bump succeeds

    Raises:
        ReleaseError: If version bump fails
    """
    logger.info("Step 3: Bumping Version")
    
    try:
        bump_type = commit_info.get("bump_type")
        if not bump_type:
            raise ReleaseError(
                ErrorCode.VERSION_UPDATE_FAILED,
                "No version bump type provided"
            )

        new_version = update_version(project_path, bump_type)
        logger.info(f"✓ Version bumped to {new_version}")
        return True

    except Exception as e:
        logger.error(f"Failed to bump version: {str(e)}")
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            f"Failed to bump version: {str(e)}"
        )