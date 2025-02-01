"""Main entry point for omaster."""
import os
import sys
from pathlib import Path

from .commands import release_steps
from .core.errors import ReleaseError


def get_project_path(path: str = None) -> Path:
    """Get the project path.
    
    Args:
        path: Optional path to project directory
        
    Returns:
        Path: Absolute path to project directory
    """
    if path:
        project_path = Path(path).resolve()
    else:
        project_path = Path.cwd()
        
    if not project_path.is_dir():
        raise ReleaseError(
            "Invalid project path. Must be a directory."
        )
        
    return project_path


def run_release_pipeline(project_path: str = None) -> bool:
    """Run the complete release pipeline.
    
    Args:
        project_path: Optional path to project directory
        
    Returns:
        bool: True if release successful
    """
    try:
        # Get project path
        path = get_project_path(project_path)
        print(f"\nReleasing project: {path}\n")
        
        # Step 1: Validate project
        release_steps.step_1_validate.run(path)
        
        # Step 1.5: Code quality checks
        release_steps.step_1_5_code_quality.run(path)
        
        # Step 2: Analyze changes
        success, commit_info = release_steps.step_2_analyze_changes.run(path)
        if not success:
            return False
            
        # Step 3: Bump version
        success = release_steps.step_3_bump_version.run(path, commit_info)
        if not success:
            return False
            
        # Step 4: Clean and build
        success = release_steps.step_4_clean_build.run(path)
        if not success:
            return False
            
        # Step 5: Publish
        success = release_steps.step_5_publish.run(path)
        if not success:
            return False
            
        # Step 6: Git commit and push
        success = release_steps.step_6_git_commit.run(path, commit_info)
        if not success:
            return False
            
        print("Release completed successfully!")
        return True
        
    except ReleaseError as e:
        print(str(e))
        return False
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return False
    except Exception as e:
        print(f"\n🚨 Unexpected error: {str(e)}")
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