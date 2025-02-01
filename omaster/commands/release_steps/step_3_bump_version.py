"""Step 3: Bump version based on changes."""
from pathlib import Path
from typing import TypedDict
from ...utils.version import bump_version
from ...core.errors import ErrorCode, ReleaseError

class CommitInfo(TypedDict):
    title: str
    description: str
    bump_type: str

def run(project_path: Path, commit_info: CommitInfo) -> bool:
    """Bump version based on changes.
    
    Args:
        project_path: Path to the project directory
        commit_info: Information about the changes
        
    Returns:
        bool: True if version bump successful, False otherwise
        
    Raises:
        ReleaseError: If version bump fails
    """
    print("Step 3: Bumping version...")
    try:
        if not bump_version(project_path, commit_info["bump_type"]):
            raise ReleaseError(ErrorCode.VERSION_BUMP_FAILED)
        print("✓ Version bumped\n")
        return True
    except ValueError as e:
        raise ReleaseError(ErrorCode.INVALID_VERSION, str(e))
    except Exception as e:
        raise ReleaseError(ErrorCode.VERSION_BUMP_FAILED, str(e)) 