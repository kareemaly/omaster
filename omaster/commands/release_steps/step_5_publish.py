"""Step 6: Publish.

This step publishes the package to PyPI.
"""
from pathlib import Path
from ...core.errors import ReleaseError, ErrorCode
from ...utils import run_command
from ...ui.layout import ReleaseUI

def run(project_path: Path, ui: ReleaseUI) -> bool:
    """Run publish step."""
    ui.log("Starting package publish...", style="blue")
    
    ui.update_progress("Publishing to PyPI...", 95)
    ui.log("Publishing package to PyPI...", level="debug")

    try:
        if not run_command("uv publish", project_path, ErrorCode.PUBLISH_ERROR):
            ui.log("Failed to publish package", level="error")
            return False

        ui.log("✓ Package published successfully", style="green")
        return True

    except ReleaseError as e:
        ui.log(f"Failed to publish package: {e}", level="error")
        raise