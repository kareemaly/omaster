"""Step 1: Validate the project.

This step validates the project structure and required files.
"""
import os
from pathlib import Path
import tomli
from rich.progress import Progress
from ...core.errors import ErrorCode, ReleaseError


def validate_pyproject(project_path: Path) -> bool:
    """Validate pyproject.toml file.

    Args:
        project_path: Path to project directory

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ReleaseError: If validation fails
    """
    pyproject_path = project_path / "pyproject.toml"

    # Check file exists
    if not pyproject_path.exists():
        raise ReleaseError(ErrorCode.MISSING_PYPROJECT)

    # Read and parse file
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)
    except Exception as e:
        raise ReleaseError(ErrorCode.INVALID_PYPROJECT, str(e))

    # Check required fields
    required_fields = ["name", "version", "description", "readme"]
    for field in required_fields:
        if field not in pyproject.get("project", {}):
            raise ReleaseError(
                ErrorCode.INVALID_PYPROJECT,
                f"Missing required field: {field}"
            )

    return True


def validate_readme(project_path: Path) -> bool:
    """Validate README.md file.

    Args:
        project_path: Path to project directory

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ReleaseError: If validation fails
    """
    readme_path = project_path / "README.md"

    # Check file exists
    if not readme_path.exists():
        raise ReleaseError(ErrorCode.MISSING_README)

    # Read file
    try:
        content = readme_path.read_text()
    except Exception as e:
        raise ReleaseError(ErrorCode.INVALID_README, str(e))

    # Check length
    if len(content) < 50:
        raise ReleaseError(
            ErrorCode.INVALID_README,
            "README is too short (minimum 50 characters)"
        )

    # Check required sections
    required_sections = ["Installation", "Usage"]
    for section in required_sections:
        if f"## {section}" not in content:
            raise ReleaseError(
                ErrorCode.INVALID_README,
                f"Missing required section: {section}"
            )

    return True


def run(project_path: Path, progress: Progress = None, task_id: int = None) -> bool:
    """Run validation step.

    Args:
        project_path: Path to project directory
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        bool: True if validation passes

    Raises:
        ReleaseError: If validation fails
    """
    if progress and task_id:
        progress.update(task_id, advance=30, description="[green]Step 1: Validating pyproject.toml...")
    
    if not validate_pyproject(project_path):
        return False
        
    if progress and task_id:
        progress.update(task_id, advance=30, description="[green]Step 1: Validating README.md...")
    
    if not validate_readme(project_path):
        return False
        
    if progress and task_id:
        progress.update(task_id, advance=40, description="[green]Step 1: All validations passed")
    
    return True