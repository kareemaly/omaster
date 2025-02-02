"""Step 1: Validate the project.

This step validates the project structure and required files.
"""
import os
from pathlib import Path
import tomli
from ...core.errors import ErrorCode, ReleaseError
from ...ui.layout import ReleaseUI


def validate_pyproject(project_path: Path, ui: ReleaseUI) -> bool:
    """Validate pyproject.toml file.

    Args:
        project_path: Path to project directory
        ui: UI manager instance

    Returns:
        bool: True if valid

    Raises:
        ReleaseError: If validation fails
    """
    pyproject_path = project_path / "pyproject.toml"

    # Check file exists
    if not pyproject_path.exists():
        ui.log("pyproject.toml not found", level="error")
        raise ReleaseError(ErrorCode.MISSING_PYPROJECT)

    # Read and parse file
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)
    except Exception as e:
        ui.log(f"Failed to parse pyproject.toml: {e}", level="error")
        raise ReleaseError(ErrorCode.INVALID_PYPROJECT, str(e))

    # Check required fields
    required_fields = ["name", "version", "description", "readme"]
    for field in required_fields:
        if field not in pyproject.get("project", {}):
            ui.log(f"Missing required field in pyproject.toml: {field}", level="error")
            raise ReleaseError(
                ErrorCode.INVALID_PYPROJECT,
                f"Missing required field: {field}"
            )

    ui.log("✓ pyproject.toml validation passed", style="green")
    return True


def validate_readme(project_path: Path, ui: ReleaseUI) -> bool:
    """Validate README.md file.

    Args:
        project_path: Path to project directory
        ui: UI manager instance

    Returns:
        bool: True if valid

    Raises:
        ReleaseError: If validation fails
    """
    readme_path = project_path / "README.md"

    # Check file exists
    if not readme_path.exists():
        ui.log("README.md not found", level="error")
        raise ReleaseError(ErrorCode.MISSING_README)

    # Read file
    try:
        content = readme_path.read_text()
    except Exception as e:
        ui.log(f"Failed to read README.md: {e}", level="error")
        raise ReleaseError(ErrorCode.INVALID_README, str(e))

    # Check length
    if len(content) < 50:
        ui.log("README.md is too short (minimum 50 characters)", level="error")
        raise ReleaseError(
            ErrorCode.INVALID_README,
            "README is too short (minimum 50 characters)"
        )

    # Check required sections
    required_sections = ["Installation", "Usage"]
    for section in required_sections:
        if f"## {section}" not in content:
            ui.log(f"Missing required section in README.md: {section}", level="error")
            raise ReleaseError(
                ErrorCode.INVALID_README,
                f"Missing required section: {section}"
            )

    ui.log("✓ README.md validation passed", style="green")
    return True


def run(project_path: Path, ui: ReleaseUI) -> bool:
    """Run validation step.

    Args:
        project_path: Path to project directory
        ui: UI manager instance

    Returns:
        bool: True if validation passes

    Raises:
        ReleaseError: If validation fails
    """
    ui.log("Starting project validation...", style="blue")
    
    # Validate pyproject.toml
    ui.update_progress("Validating pyproject.toml...", 5)
    if not validate_pyproject(project_path, ui):
        return False
        
    # Validate README.md
    ui.update_progress("Validating README.md...", 15)
    if not validate_readme(project_path, ui):
        return False
        
    ui.log("✓ All validations passed", style="green")
    return True