"""Main entry point for the release pipeline."""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from .core.errors import ReleaseError, ErrorCode
from .ui.layout import ReleaseUI
from .commands.release_steps import (
    step_1_validate,
    step_2_validate_code_quality,
    step_3_clean_build,
    step_4_analyze_changes,
    step_5_bump_version,
    step_5_publish,
    step_6_git_commit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format since we'll use rich for output
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    """Run the release pipeline."""
    project_path = Path.cwd()
    verbose = os.getenv("OMASTER_VERBOSE", "0") == "1"
    
    with ReleaseUI(verbose=verbose) as ui:
        try:
            ui.log("🚀 Starting Release Pipeline", style="bold blue")
            
            # Step 1: Validation
            ui.update_progress("Step 1: Project Validation", 0)
            step_1_validate.run(project_path, ui)
            ui.update_progress("Step 1: ✓ Validation passed", 20)
            
            # Step 2: Code Quality
            ui.update_progress("Step 2: Code Quality", 20)
            step_2_validate_code_quality.run(project_path, ui)
            ui.update_progress("Step 2: ✓ Code quality passed", 40)
            
            # Step 3: Change Analysis
            ui.update_progress("Step 3: Change Analysis", 40)
            commit_message, version_bump_type = step_4_analyze_changes.run(project_path, ui)
            ui.update_progress(f"Step 3: ✓ Changes analyzed - {commit_message}", 60)
            
            # Step 4: Version Update
            ui.update_progress("Step 4: Version Update", 60)
            step_5_bump_version.run(project_path, version_bump_type, ui)
            ui.update_progress("Step 4: ✓ Version updated", 80)
            
            # Step 5: Build & Publish
            ui.update_progress("Step 5: Build & Publish", 80)
            step_3_clean_build.run(project_path, ui)
            step_5_publish.run(project_path, ui)
            ui.update_progress("Step 5: ✓ Build and publish completed", 100)
            
            ui.log("✨ Release process completed successfully!", style="bold green")
            
        except ReleaseError as e:
            ui.log_error(e)
            raise
        except Exception as e:
            error = ReleaseError(ErrorCode.UNKNOWN_ERROR, str(e))
            ui.log_error(error)
            raise error

def main() -> int:
    """Main entry point."""
    try:
        run_pipeline()
        return 0
    except ReleaseError as e:
        return e.code.value
    except Exception:
        return ErrorCode.UNKNOWN_ERROR.value

if __name__ == "__main__":
    main()