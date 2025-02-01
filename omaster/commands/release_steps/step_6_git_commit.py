"""Step 6: Commit and push changes to git."""
import subprocess
from pathlib import Path
from typing import TypedDict
from ...core.errors import ErrorCode, ReleaseError

class CommitInfo(TypedDict):
    title: str
    description: str
    bump_type: str

def run_command(cmd: str, cwd: Path) -> bool:
    """Run a shell command."""
    try:
        subprocess.run(cmd.split(), check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(ErrorCode.GIT_PUSH_FAILED, str(e))

def run(project_path: Path, commit_info: CommitInfo) -> bool:
    """Commit and push changes.
    
    Args:
        project_path: Path to the project directory
        commit_info: Information about the changes
        
    Returns:
        bool: True if git operations successful, False otherwise
        
    Raises:
        ReleaseError: If git operations fail
    """
    print("Step 6: Committing and pushing changes...")
    
    # Add all changes
    try:
        if not run_command("git add --all", project_path):
            raise ReleaseError(ErrorCode.GIT_ADD_FAILED)
        
        # Create commit message
        commit_msg = f"{commit_info['title']}\n\n{commit_info['description']}"
        commit_cmd = ["git", "commit", "-m", commit_msg]
        try:
            subprocess.run(commit_cmd, check=True, cwd=project_path)
        except subprocess.CalledProcessError as e:
            raise ReleaseError(ErrorCode.GIT_COMMIT_FAILED, str(e))
        
        # Push changes
        if not run_command("git push", project_path):
            raise ReleaseError(ErrorCode.GIT_PUSH_FAILED)
            
        print("✓ Changes committed and pushed\n")
        return True
        
    except subprocess.CalledProcessError as e:
        if "nothing to commit" in str(e):
            raise ReleaseError(ErrorCode.GIT_NO_CHANGES, str(e))
        raise ReleaseError(ErrorCode.GIT_ADD_FAILED, str(e)) 