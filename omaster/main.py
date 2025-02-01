"""Main entry point for the omaster tool."""
import sys
from pathlib import Path
from typing import Optional

from .commands import release_steps
from .core.errors import ReleaseError, ErrorCode


def run_release_pipeline(project_path: Optional[str] = None) -> bool:
    """Run the complete release pipeline.
    
    Steps:
    1. Validate
    2. Analyze changes
    3. Bump version
    4. Clean and build
    5. Publish
    6. Git commit and push
    
    Args:
        project_path: Optional path to the project directory. Defaults to current directory.
        
    Returns:
        bool: True if release successful, False otherwise
    """
    path = Path(project_path or ".").resolve()
    print(f"\nReleasing project: {path}\n")
    
    try:
        # Step 1: Validate
        if not release_steps.step_1_validate.run(path):
            return False
            
        # Step 2: Analyze changes
        success, commit_info = release_steps.step_2_analyze_changes.analyze_changes(path)
        if not success:
            return False
            
        # Step 3: Bump version
        if not release_steps.step_3_bump_version.run(path, commit_info):
            return False
            
        # Step 4: Clean and build
        if not release_steps.step_4_clean_build.run(path):
            return False
            
        # Step 5: Publish
        if not release_steps.step_5_publish.run(path):
            return False
            
        # Step 6: Git commit and push
        if not release_steps.step_6_git_commit.run(path, commit_info):
            return False
        
        print("Release completed successfully!")
        return True
        
    except ReleaseError as e:
        print(str(e))
        return False
    except Exception as e:
        print(ReleaseError(ErrorCode.UNKNOWN_ERROR, str(e)))
        return False


def main():
    """Main entry point."""
    try:
        # Get project path from command line args if provided
        project_path = sys.argv[1] if len(sys.argv) > 1 else None
        
        success = run_release_pipeline(project_path)
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main() 