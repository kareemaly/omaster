"""Step 2: Validate code quality using radon, jscpd, and vulture."""
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

from ...core.errors import ReleaseError, ErrorCode
from ...core.config import Config, load_config

def run_radon(project_path: Path, max_complexity: int = 15) -> bool:
    """Run radon complexity check.

    Args:
        project_path: Path to project directory
        max_complexity: Maximum allowed complexity

    Returns:
        bool: True if check passes
    """
    try:
        result = subprocess.run(
            ["radon", "cc", "-j", "-n", "C", str(project_path)],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        
        # Check each file's complexity
        for file_path, blocks in data.items():
            for block in blocks:
                if block["complexity"] > max_complexity:
                    raise ReleaseError(
                        ErrorCode.CODE_QUALITY_ERROR,
                        f"Code complexity too high in {file_path}: {block['name']} ({block['complexity']})"
                    )
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run radon: {str(e)}"
        )

def run_jscpd(project_path: Path, max_duplicates: float = 1.0) -> bool:
    """Run jscpd duplicate code check.

    Args:
        project_path: Path to project directory
        max_duplicates: Maximum allowed duplicate percentage

    Returns:
        bool: True if check passes
    """
    try:
        # Run jscpd without --json flag as it's not supported in newer versions
        result = subprocess.run(
            ["jscpd", str(project_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to find duplication percentage
        for line in result.stdout.splitlines():
            if "%" in line and "clones found" in line.lower():
                percentage = float(line.split("%")[0].strip().split()[-1])
                if percentage > max_duplicates:
                    raise ReleaseError(
                        ErrorCode.CODE_QUALITY_ERROR,
                        f"Too much code duplication: {percentage}% (max {max_duplicates}%)"
                    )
                break
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run jscpd: {str(e)}"
        )

def run_vulture() -> None:
    """Run vulture to check for dead code."""
    try:
        result = subprocess.run(
            ["vulture", "omaster", "--exclude", ".venv/*"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ReleaseError(
                ErrorCode.CODE_QUALITY_ERROR,
                f"Dead code check failed with exit status {result.returncode}:\n{result.stdout}",
            )
    except Exception as e:
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run vulture: {str(e)}",
        )

def run(project_path: Path) -> bool:
    """Run code quality validation.

    Args:
        project_path: Path to the project directory

    Returns:
        bool: True if validation passes

    Raises:
        ReleaseError: If validation fails
    """
    print("Step 2: Validating code quality...")

    # Load configuration
    config_data = load_config(project_path)
    config = Config(config_data)

    # Get thresholds from config or use defaults
    quality_config = config.data.get("quality", {})
    max_complexity = quality_config.get("max_complexity", 15)
    max_duplicates = quality_config.get("max_duplicates", 1.0)

    # Run radon
    print("Running complexity check...")
    if run_radon(project_path, max_complexity):
        print("✓ Complexity check passed")

    # Run jscpd
    print("Running duplication check...")
    if run_jscpd(project_path, max_duplicates):
        print("✓ Duplication check passed")

    # Run vulture
    print("Running dead code check...")
    if run_vulture():
        print("✓ Dead code check passed")

    print("✓ Code quality validation passed\n")
    return True 