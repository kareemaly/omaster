"""Version management utilities."""
import re
import logging
import tomli
import tomli_w
from pathlib import Path
from typing import Tuple

from ..core.errors import ReleaseError, ErrorCode

logger = logging.getLogger(__name__)

def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse version string into components.

    Args:
        version: Version string (e.g., "1.2.3")

    Returns:
        Tuple of (major, minor, patch) version numbers

    Raises:
        ReleaseError: If version string is invalid
    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            f"Invalid version format: {version}"
        )
    return tuple(map(int, match.groups()))

def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version according to semver rules.

    Args:
        current_version: Current version string
        bump_type: Type of version bump (major, minor, patch)

    Returns:
        New version string

    Raises:
        ReleaseError: If version string or bump type is invalid
    """
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            f"Invalid bump type: {bump_type}"
        )

def update_version(project_path: Path, bump_type: str) -> str:
    """Update version in pyproject.toml.

    Args:
        project_path: Path to project directory
        bump_type: Type of version bump (major, minor, patch)

    Returns:
        New version string

    Raises:
        ReleaseError: If version update fails
    """
    pyproject_path = project_path / "pyproject.toml"
    if not pyproject_path.exists():
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            "pyproject.toml not found"
        )

    try:
        logger.info("Reading current version...")
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)

        current_version = pyproject["project"]["version"]
        new_version = bump_version(current_version, bump_type)
        logger.info(f"Bumping version: {current_version} -> {new_version}")

        pyproject["project"]["version"] = new_version

        logger.info("Updating pyproject.toml...")
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(pyproject, f)

        logger.info("✓ Version updated successfully")
        return new_version

    except Exception as e:
        raise ReleaseError(
            ErrorCode.VERSION_UPDATE_FAILED,
            f"Failed to update version: {str(e)}"
        )