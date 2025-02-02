"""Main entry point for the release pipeline."""
import logging
from pathlib import Path
from typing import Optional, Tuple

from .core.errors import ReleaseError, ErrorCode, handle_error
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        path = Path.cwd()
        logger.info("Starting release process...")

        # Step 1: Validate repository state
        logger.info("\nStep 1: Validation")
        logger.info("Validating repository state...")
        if not step_1_validate.run(path):
            logger.error("❌ Validation failed")
            return ErrorCode.VALIDATION_ERROR.value
        logger.info("✓ Validation passed")

        # Step 2: Validate code quality
        logger.info("\nStep 2: Code Quality")
        logger.info("Validating code quality...")
        if not step_2_validate_code_quality.run(path):
            logger.error("❌ Code quality validation failed")
            return ErrorCode.CODE_QUALITY_ERROR.value
        logger.info("✓ Code quality validation passed")

        # Step 3: Clean and build
        logger.info("\nStep 3: Clean and Build")
        logger.info("Cleaning and building project...")
        if not step_3_clean_build.run(path):
            logger.error("❌ Clean and build failed")
            return ErrorCode.BUILD_ERROR.value
        logger.info("✓ Clean and build passed")

        # Step 4: Analyze changes
        logger.info("\nStep 4: Change Analysis")
        logger.info("Analyzing changes...")
        success, commit_info = step_4_analyze_changes.run(path)
        if not success:
            logger.error("❌ Change analysis failed")
            return ErrorCode.ANALYSIS_ERROR.value
        logger.info("✓ Change analysis passed")
        logger.info(f"✓ Changes committed: {commit_info.title}")
        logger.info(f"✓ Version bump type: {commit_info.bump_type}")

        # Step 5: Bump version
        logger.info("\nStep 5: Version Update")
        logger.info("Updating version...")
        if not step_5_bump_version.run(path, commit_info.model_dump()):
            logger.error("❌ Version update failed")
            return ErrorCode.VERSION_UPDATE_FAILED.value
        logger.info("✓ Version updated")

        # Step 6: Publish
        logger.info("\nStep 6: Publish")
        logger.info("Publishing release...")
        if not step_5_publish.run(path):
            logger.error("❌ Publish failed")
            return ErrorCode.PUBLISH_ERROR.value
        logger.info("✓ Publish completed")

        logger.info("\n✨ Release process completed successfully!")
        return 0

    except ReleaseError as e:
        handle_error(e)  # Use the rich error handler
        return e.code.value
    except Exception as e:
        handle_error(e)  # Use the rich error handler for unexpected errors too
        return ErrorCode.UNKNOWN_ERROR.value

if __name__ == "__main__":
    main()