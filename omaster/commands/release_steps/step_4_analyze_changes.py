"""Step 3: Analyze changes and generate commit info using OpenAI."""
import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from rich.progress import Progress
from contextlib import nullcontext

from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config, load_config

logger = logging.getLogger(__name__)

class CommitInfo(BaseModel):
    """Model for commit information."""
    title: str
    description: str
    bump_type: str  # major, minor, or patch

def get_git_diff(progress: Progress = None, task_id: int = None) -> str:
    """Get git diff of staged and unstaged changes.

    Args:
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        str: Git diff output

    Raises:
        ReleaseError: If git command fails
    """
    try:
        if progress and task_id:
            progress.update(task_id, advance=10, description="[blue]Step 3: Getting git diff...")
            
        staged = subprocess.check_output(['git', 'diff', '--cached'], text=True)
        unstaged = subprocess.check_output(['git', 'diff'], text=True)
        if not (staged or unstaged):
            # Check for unpushed commits
            unpushed = subprocess.check_output(['git', 'log', '@{u}..'], text=True)
            if not unpushed:
                raise ReleaseError(
                    ErrorCode.GIT_NO_CHANGES,
                    "No changes found in git repository"
                )
            if progress and task_id:
                progress.update(task_id, advance=10, description="[blue]Step 3: Found unpushed commits")
            return unpushed
            
        if progress and task_id:
            progress.update(task_id, advance=10, description="[blue]Step 3: Git diff retrieved")
        return staged + unstaged
        
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.GIT_NO_CHANGES,
            f"Failed to get git diff: {str(e)}"
        )

def commit_and_push(commit_info: CommitInfo, progress: Progress = None, task_id: int = None) -> None:
    """Commit changes and push to remote.

    Args:
        commit_info: Commit information from AI
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Raises:
        ReleaseError: If git operations fail
    """
    try:
        # Check if there are any changes to commit
        status = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        if status:
            if progress and task_id:
                progress.update(task_id, advance=10, description="[blue]Step 3: Staging changes...")
                
            subprocess.run(['git', 'add', '.'], check=True)
            
            if progress and task_id:
                progress.update(task_id, advance=10, description="[blue]Step 3: Creating commit...")
                
            commit_msg = f"{commit_info.title}\n\n{commit_info.description}"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            
            if progress and task_id:
                progress.update(task_id, advance=10, description="[blue]Step 3: Changes committed")
        else:
            if progress and task_id:
                progress.update(task_id, advance=20, description="[blue]Step 3: No changes to commit")

        if progress and task_id:
            progress.update(task_id, advance=10, description="[blue]Step 3: Pushing to remote...")
            
        subprocess.run(['git', 'push'], check=True)
        
        if progress and task_id:
            progress.update(task_id, advance=10, description="[blue]Step 3: Changes pushed")

    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.GIT_OPERATION_FAILED,
            f"Failed to perform git operation: {str(e)}"
        )

def analyze_with_openai(diff: str, model: str, api_key: str, progress: Progress = None, task_id: int = None) -> CommitInfo:
    """Analyze git diff with OpenAI.

    Args:
        diff: Git diff to analyze
        model: OpenAI model to use
        api_key: OpenAI API key
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        CommitInfo: Generated commit information

    Raises:
        ReleaseError: If analysis fails
    """
    try:
        if progress and task_id:
            progress.update(task_id, advance=10, description="[blue]Step 3: Analyzing with OpenAI...")
            
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a commit message generator. Analyze the git diff "
                        "and generate a structured commit message. Your response must be a JSON object "
                        "with exactly these fields:\n"
                        "- title: A concise title following conventional commits format\n"
                        "- description: A detailed description of the changes\n"
                        "- bump_type: One of 'major', 'minor', or 'patch' based on semantic versioning\n\n"
                        "Example response format:\n"
                        "{\n"
                        '  "title": "feat: add new feature",\n'
                        '  "description": "Added new feature that does X",\n'
                        '  "bump_type": "minor"\n'
                        "}"
                    )
                },
                {"role": "user", "content": f"Git diff:\n{diff}"}
            ],
            response_format={"type": "json_object"}
        )

        commit_info = CommitInfo.model_validate_json(completion.choices[0].message.content)
        
        if progress and task_id:
            progress.update(task_id, advance=10, description=f"[blue]Step 3: Changes analyzed")
            
        return commit_info

    except Exception as e:
        raise ReleaseError(
            ErrorCode.OPENAI_API_ERROR,
            f"Failed to analyze changes: {str(e)}"
        )

def run(project_path: Path, progress: Progress = None, task_id: int = None) -> Tuple[str, str]:
    """Analyze changes and generate commit info.

    Args:
        project_path: Path to the project directory
        progress: Optional Progress instance for updating status
        task_id: Optional task ID for the progress bar

    Returns:
        Tuple[str, str]: Commit message and version bump type

    Raises:
        ReleaseError: If analysis fails
    """
    # Create dummy progress if none provided
    dummy_progress = False
    if progress is None:
        from rich.console import Console
        progress = Progress(console=Console())
        task_id = progress.add_task("[blue]Step 3: Change Analysis", total=100)
        dummy_progress = True

    try:
        with progress if dummy_progress else nullcontext():
            # Load configuration
            if progress and task_id:
                progress.update(task_id, advance=10, description="[blue]Step 3: Loading configuration...")
                
            config_data = load_config(project_path)
            config = Config(config_data)
            
            if progress and task_id:
                progress.update(task_id, advance=5, description="[blue]Step 3: Configuration loaded")

            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ReleaseError(
                    ErrorCode.OPENAI_API_KEY_MISSING,
                    "OpenAI API key not found in environment"
                )
                
            if progress and task_id:
                progress.update(task_id, advance=5, description="[blue]Step 3: API key validated")

            # Get git diff
            diff = get_git_diff(progress, task_id)

            # Analyze changes with OpenAI
            commit_info = analyze_with_openai(diff, config.model, api_key, progress, task_id)

            # Commit and push changes
            commit_and_push(commit_info, progress, task_id)

            return commit_info.title, commit_info.bump_type

    finally:
        if dummy_progress:
            progress.stop()