"""Step 2: Validate code quality using radon, jscpd, and vulture."""
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any
from ...core.errors import ReleaseError, ErrorCode
from ...core.config import Config, load_config
from ...ui.layout import ReleaseUI

def run_radon(project_path: Path, max_complexity: int = 15, ui: ReleaseUI = None) -> bool:
    """Run radon complexity check.

    Args:
        project_path: Path to project directory
        max_complexity: Maximum allowed complexity
        ui: UI manager instance

    Returns:
        bool: True if check passes
    """
    try:
        if ui:
            ui.log("Running complexity analysis...", level="debug")
            
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
                    if ui:
                        ui.log(f"High complexity in {file_path}: {block['name']} ({block['complexity']})", level="error")
                    raise ReleaseError(
                        ErrorCode.CODE_QUALITY_ERROR,
                        f"Code complexity too high in {file_path}: {block['name']} ({block['complexity']})"
                    )
                    
        if ui:
            ui.log("✓ Complexity check passed", style="green")
        return True
        
    except subprocess.CalledProcessError as e:
        if ui:
            ui.log(f"Failed to run radon: {e}", level="error")
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run radon: {str(e)}"
        )

def run_jscpd(project_path: Path, max_duplicates: float = 1.0, ui: ReleaseUI = None) -> bool:
    """Run jscpd duplicate code check.

    Args:
        project_path: Path to project directory
        max_duplicates: Maximum allowed duplicate percentage
        ui: UI manager instance

    Returns:
        bool: True if check passes
    """
    # Check if jscpd is installed
    if not shutil.which("jscpd"):
        if ui:
            ui.log("jscpd not found - this tool is required for code quality checks", level="error")
            ui.log("To install jscpd: npm install -g jscpd", style="yellow")
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            "jscpd is required but not installed. Install with: npm install -g jscpd"
        )
        
    try:
        if ui:
            ui.log("Running duplication check...", level="debug")
            
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
                    if ui:
                        ui.log(f"Too much code duplication: {percentage}% (max {max_duplicates}%)", level="error")
                    raise ReleaseError(
                        ErrorCode.CODE_QUALITY_ERROR,
                        f"Too much code duplication: {percentage}% (max {max_duplicates}%)"
                    )
                break
                
        if ui:
            ui.log("✓ Duplication check passed", style="green")
        return True
        
    except subprocess.CalledProcessError as e:
        if ui:
            ui.log(f"Failed to run jscpd: {e}", level="error")
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run jscpd: {str(e)}"
        )

def run_vulture(ui: ReleaseUI = None) -> bool:
    """Run vulture to check for dead code.
    
    Args:
        ui: UI manager instance
        
    Returns:
        bool: True if check passes
    """
    try:
        if ui:
            ui.log("Running dead code check...", level="debug")
            
        result = subprocess.run(
            ["vulture", "omaster", "--exclude", ".venv/*"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if ui:
                ui.log(f"Dead code check failed:\n{result.stdout}", level="error")
            raise ReleaseError(
                ErrorCode.CODE_QUALITY_ERROR,
                f"Dead code check failed with exit status {result.returncode}:\n{result.stdout}",
            )
            
        if ui:
            ui.log("✓ Dead code check passed", style="green")
        return True
        
    except Exception as e:
        if ui:
            ui.log(f"Failed to run vulture: {e}", level="error")
        raise ReleaseError(
            ErrorCode.CODE_QUALITY_ERROR,
            f"Failed to run vulture: {str(e)}",
        )

def run(project_path: Path, ui: ReleaseUI) -> bool:
    """Run code quality validation.

    Args:
        project_path: Path to the project directory
        ui: UI manager instance

    Returns:
        bool: True if validation passes

    Raises:
        ReleaseError: If validation fails
    """
    ui.log("Starting code quality validation...", style="blue")
    
    # Load configuration
    ui.update_progress("Loading configuration...", 10)
    config_data = load_config(project_path)
    config = Config(config_data)

    # Get thresholds from config or use defaults
    quality_config = config.data.get("quality", {})
    max_complexity = quality_config.get("max_complexity", 15)
    max_duplicates = quality_config.get("max_duplicates", 1.0)

    # Run radon
    ui.update_progress("Running complexity analysis...", 30)
    if not run_radon(project_path, max_complexity, ui):
        return False

    # Run jscpd
    ui.update_progress("Running duplication check...", 60)
    if not run_jscpd(project_path, max_duplicates, ui):
        return False

    # Run vulture
    ui.update_progress("Running dead code check...", 90)
    if not run_vulture(ui):
        return False

    ui.log("✓ All quality checks passed", style="green")
    return True 