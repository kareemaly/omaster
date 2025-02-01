"""Step 2: Analyze changes and generate commit info using OpenAI."""
import os
import subprocess
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI

from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config

class CommitInfo(BaseModel):
    title: str
    description: str
    bump_type: str  # major, minor, or patch

def get_git_diff() -> str:
    """Get git diff of staged and unstaged changes."""
    try:
        staged = subprocess.check_output(['git', 'diff', '--cached'], text=True)
        unstaged = subprocess.check_output(['git', 'diff'], text=True)
        return staged + unstaged
    except subprocess.CalledProcessError as e:
        raise ReleaseError(ErrorCode.GIT_NO_CHANGES, str(e))

def analyze_changes(project_path: Path) -> tuple[bool, dict]:
    """Analyze changes and generate commit info.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        tuple[bool, dict]: Success status and commit info dictionary
        
    Raises:
        ReleaseError: If analysis fails
    """
    print("Step 2: Analyzing changes...")
    
    # Load configuration
    config = Config(project_path)
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ReleaseError(ErrorCode.OPENAI_API_KEY_MISSING)
        
    # Get git diff
    try:
        diff = get_git_diff()
        if not diff:
            raise ReleaseError(ErrorCode.GIT_NO_CHANGES)
    except subprocess.CalledProcessError as e:
        raise ReleaseError(ErrorCode.GIT_NO_CHANGES, str(e))
        
    try:
        client = OpenAI()
        completion = client.beta.chat.completions.parse(
            model=config.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a commit message generator. Analyze the git diff and generate a structured commit message."
                },
                {"role": "user", "content": f"Git diff:\n{diff}"}
            ],
            response_format=CommitInfo,
        )
        
        commit_info = completion.choices[0].message.parsed
        print("✓ Changes analyzed\n")
        return True, commit_info.model_dump()
        
    except Exception as e:
        raise ReleaseError(ErrorCode.OPENAI_API_ERROR, str(e))