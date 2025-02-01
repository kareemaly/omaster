"""Version bumping utilities."""
from pathlib import Path
import tomli
import tomli_w
from typing import Tuple, Literal

BumpType = Literal["major", "minor", "patch"]

def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse version string into (major, minor, patch) tuple."""
    try:
        major, minor, patch = map(int, version.split("."))
        return major, minor, patch
    except ValueError:
        raise ValueError(f"Invalid version format: {version}. Expected 'major.minor.patch'")


def bump_version(project_path: Path, bump_type: BumpType) -> bool:
    """Bump version in pyproject.toml based on bump type.
    
    Args:
        project_path: Path to the project directory
        bump_type: Type of version bump (major, minor, patch)
        
    Returns:
        bool: True if version was bumped successfully
    """
    pyproject_path = project_path / "pyproject.toml"
    
    try:
        # Read current version
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)
        
        current_version = pyproject["project"]["version"]
        major, minor, patch = parse_version(current_version)
        
        # Bump version based on type
        if bump_type == "major":
            new_version = f"{major + 1}.0.0"
        elif bump_type == "minor":
            new_version = f"{major}.{minor + 1}.0"
        else:  # patch
            new_version = f"{major}.{minor}.{patch + 1}"
            
        pyproject["project"]["version"] = new_version
        
        # Write updated version
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(pyproject, f)
        
        print(f"Version bumped: {current_version} -> {new_version}")
        return True
        
    except Exception as e:
        print(f"Error bumping version: {str(e)}")
        return False