"""Main entry point for the release pipeline."""
import logging
from pathlib import Path
from typing import Optional

from .core.errors import ReleaseError, ErrorCode, handle_error
from .core.validator import validate_project
from .commands.release_steps import (
    step_1_validate,
    step_1_5_code_quality,
    step_2_analyze_changes,
    step_3_bump_version,
    step_4_clean_build,
    step_5_publish,
    step_6_git_commit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_path(project_path: Optional[str] = None) -> Path:
    """Get and validate the project path.

    Args:
        project_path: Optional path to project directory

    Returns:
        Path to project directory

    Raises:
        ReleaseError: If path is invalid
    """
    path = Path(project_path) if project_path else Path.cwd()
    logger.info(f"Validating project path: {path}")
    if not path.is_dir():
        logger.error(f"Invalid project path: {path} is not a directory")
        raise ReleaseError(
            ErrorCode.CONFIG_ERROR,
            "Invalid project path. Must be a directory."
        )
    logger.info("✓ Project path validation passed")
    return path


def run_release_pipeline(project_path: Optional[str] = None) -> bool:
    """Run the release pipeline.

    Args:
        project_path: Optional path to project directory

    Returns:
        bool: True if release was successful

    Raises:
        ReleaseError: If any step fails
    """
    try:
        logger.info("\n" + "="*50)
        logger.info("Starting omaster release pipeline")
        logger.info("="*50 + "\n")
        
        path = get_project_path(project_path)
        logger.info(f"\nProject directory: {path}")

        # Step 1: Validate project structure
        logger.info("\nStep 1: Project Structure Validation")
        logger.info("-"*30)
        logger.info("Checking required files and configurations...")
        if not validate_project(path):
            logger.error("Project validation failed")
            raise ReleaseError(
                ErrorCode.CONFIG_ERROR,
                "Project validation failed"
            )
        logger.info("✓ All required files present")
        logger.info("✓ Project structure validation passed")

        # Step 1.5: Code quality analysis
        logger.info("\nStep 1.5: Code Quality Analysis")
        logger.info("-"*30)
        logger.info("Running code quality checks...")
        logger.info("This may take a few moments...")
        if not step_1_5_code_quality.run(path):
            logger.error("Code quality checks failed")
            raise ReleaseError(
                ErrorCode.QUALITY_ERROR,
                "Code quality checks failed"
            )
        logger.info("✓ Code quality checks passed")

        logger.info("\n" + "="*50)
        logger.info("Release pipeline completed successfully")
        logger.info("="*50 + "\n")
        return True

    except ReleaseError as e:
        logger.error(f"Release pipeline failed: {str(e)}")
        handle_error(e)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in release pipeline: {str(e)}")
        handle_error(e)
        return False


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        success = run_release_pipeline()
        return 0 if success else 1
    except Exception as e:
        handle_error(e)
        return 1

if __name__ == "__main__":
    main()