"""Step 5: Clean old builds and build package."""
import shutil
from pathlib import Path
from ...core.errors import ErrorCode, ReleaseError
from ...utils import run_command
from ...ui.layout import ReleaseUI

def clean_dist(project_path: Path, ui: ReleaseUI = None) -> bool:
    """Clean up old build files."""
    try:
        if ui:
            ui.log("Cleaning old build files...", level="debug")
            
        dist_path = project_path / "dist"
        if dist_path.exists():
            shutil.rmtree(dist_path)
        dist_path.mkdir(exist_ok=True)
        
        if ui:
            ui.log("✓ Clean successful", style="green")
        return True
        
    except Exception as e:
        if ui:
            ui.log(f"Failed to clean build files: {e}", level="error")
        raise ReleaseError(ErrorCode.BUILD_CLEAN_FAILED, str(e))

def run(project_path: Path, ui: ReleaseUI) -> bool:
    """Clean old builds and build package."""
    ui.log("Starting build process...", style="blue")
    
    # Clean dist directory
    ui.update_progress("Cleaning old builds...", 85)
    if not clean_dist(project_path, ui):
        return False

    # Build package
    ui.update_progress("Building package...", 90)
    try:
        if not run_command("uv build", project_path, ErrorCode.BUILD_FAILED):
            ui.log("Build failed", level="error")
            return False
            
        ui.log("✓ Build successful", style="green")
        return True
        
    except ReleaseError as e:
        ui.log(f"Build failed: {e}", level="error")
        raise