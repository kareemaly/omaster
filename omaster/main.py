"""Main entry point for the release pipeline."""
import logging
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .core.errors import ReleaseError, ErrorCode
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

def print_error(error: ReleaseError) -> None:
    """Print a formatted error message.
    
    Args:
        error: The error to print
    """
    console = Console()
    error_text = Text()
    error_text.append("🚨 Error 🚨\n", style="bold red")
    error_text.append(f"\nCode: {error.code.value} - {error.code.name}\n\n")
    error_text.append(str(error))

    panel = Panel(
        error_text,
        title="Release Pipeline Error",
        expand=True,
    )
    console.print(panel)

def run_pipeline() -> None:
    """Run the release pipeline."""
    logger.info("Starting release process...")
    project_path = Path.cwd()

    # Step 1: Validation
    logger.info("\nStep 1: Validation")
    logger.info("Validating repository state...")
    step_1_validate.run(project_path)
    logger.info("✓ Validation passed")

    # Step 2: Code Quality
    logger.info("\nStep 2: Code Quality")
    logger.info("Validating code quality...")
    step_2_validate_code_quality.run(project_path)
    logger.info("✓ Code quality validation passed")

    # Step 3: Change Analysis
    logger.info("\nStep 3: Change Analysis")
    logger.info("Analyzing changes...")
    commit_message, version_bump_type = step_4_analyze_changes.run(project_path)
    logger.info("✓ Change analysis passed")
    logger.info(f"✓ Changes committed: {commit_message}")
    logger.info(f"✓ Version bump type: {version_bump_type}")

    # Step 4: Version Update
    logger.info("\nStep 4: Version Update")
    logger.info("Updating version...")
    step_5_bump_version.run(project_path, version_bump_type)
    logger.info("✓ Version updated")

    # Step 5: Clean and Build
    logger.info("\nStep 5: Clean and Build")
    logger.info("Cleaning and building project...")
    step_3_clean_build.run(project_path)
    logger.info("✓ Clean and build passed")

    # Step 6: Publish
    logger.info("\nStep 6: Publish")
    logger.info("Publishing release...")
    step_5_publish.run(project_path)
    logger.info("✓ Release published")


def main() -> int:
    """Main entry point."""
    try:
        run_pipeline()
        return 0
    except ReleaseError as e:
        print_error(e)
        return e.code.value
    except Exception as e:
        print_error(ReleaseError(ErrorCode.UNKNOWN_ERROR, str(e)))
        return ErrorCode.UNKNOWN_ERROR.value

if __name__ == "__main__":
    main()