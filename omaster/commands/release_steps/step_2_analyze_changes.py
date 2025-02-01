"""Step 2: Analyze changes and generate commit info using OpenAI."""
import os
import subprocess
from pathlib import Path
from typing import TypedDict
from openai import OpenAI

from ...core.errors import ErrorCode, ReleaseError

class CommitInfo(TypedDict):
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

def analyze_changes(project_path: Path) -> tuple[bool, CommitInfo]:
    """Analyze changes and generate commit info.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        tuple[bool, CommitInfo]: Success status and commit info
        
    Raises:
        ReleaseError: If analysis fails
    """
    print("Step 2: Analyzing changes...")
    
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the git diff and generate a commit message and version bump type."},
                {"role": "user", "content": f"Git diff:\n{diff}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short commit title summarizing changes"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of changes"
                        },
                        "bump_type": {
                            "type": "string",
                            "enum": ["major", "minor", "patch"],
                            "description": "Version bump type based on changes"
                        }
                    },
                    "required": ["title", "description", "bump_type"],
                    "additionalProperties": False
                }
            }
        )
        
        commit_info = response.choices[0].message.content
        print("✓ Changes analyzed\n")
        return True, commit_info
        
    except Exception as e:
        raise ReleaseError(ErrorCode.OPENAI_API_ERROR, str(e)) 