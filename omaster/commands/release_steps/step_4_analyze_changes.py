"""Step 3: Analyze changes and generate commit info using OpenAI."""
import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config, load_config
from ...ui.layout import ReleaseUI

logger = logging.getLogger(__name__)

class CommitInfo(BaseModel):
    """Model for commit information."""
    title: str
    description: str
    bump_type: str  # major, minor, or patch

def get_git_diff(ui: ReleaseUI = None) -> str:
    """Get git diff of staged and unstaged changes.

    Args:
        ui: UI manager instance

    Returns:
        str: Git diff output

    Raises:
        ReleaseError: If git command fails
    """
    try:
        if ui:
            ui.log("Getting git diff...", level="debug")
            
        staged = subprocess.check_output(['git', 'diff', '--cached'], text=True)
        unstaged = subprocess.check_output(['git', 'diff'], text=True)
        if not (staged or unstaged):
            # Check for unpushed commits
            unpushed = subprocess.check_output(['git', 'log', '@{u}..'], text=True)
            if not unpushed:
                if ui:
                    ui.log("No changes found in repository", level="error")
                raise ReleaseError(
                    ErrorCode.GIT_NO_CHANGES,
                    "No changes found in git repository"
                )
            if ui:
                ui.log("Found unpushed commits", level="debug")
            return unpushed
            
        if ui:
            ui.log("Git diff retrieved", style="green")
        return staged + unstaged
        
    except subprocess.CalledProcessError as e:
        if ui:
            ui.log(f"Failed to get git diff: {e}", level="error")
        raise ReleaseError(
            ErrorCode.GIT_NO_CHANGES,
            f"Failed to get git diff: {str(e)}"
        )

def commit_and_push(commit_info: CommitInfo, ui: ReleaseUI = None) -> None:
    """Commit changes and push to remote.

    Args:
        commit_info: Commit information from AI
        ui: UI manager instance

    Raises:
        ReleaseError: If git operations fail
    """
    try:
        # Check if there are any changes to commit
        status = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        if status:
            if ui:
                ui.log("Staging changes...", level="debug")
                
            subprocess.run(['git', 'add', '.'], check=True)
            
            if ui:
                ui.log("Creating commit...", level="debug")
                
            commit_msg = f"{commit_info.title}\n\n{commit_info.description}"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            
            if ui:
                ui.log("✓ Changes committed", style="green")
        else:
            if ui:
                ui.log("No changes to commit", level="debug")

        if ui:
            ui.log("Pushing to remote...", level="debug")
            
        subprocess.run(['git', 'push'], check=True)
        
        if ui:
            ui.log("✓ Changes pushed", style="green")

    except subprocess.CalledProcessError as e:
        if ui:
            ui.log(f"Git operation failed: {e}", level="error")
        raise ReleaseError(
            ErrorCode.GIT_OPERATION_FAILED,
            f"Failed to perform git operation: {str(e)}"
        )

def analyze_with_openai(diff: str, model: str, api_key: str, ui: ReleaseUI = None) -> CommitInfo:
    """Analyze git diff with OpenAI.

    Args:
        diff: Git diff to analyze
        model: OpenAI model to use
        api_key: OpenAI API key
        ui: UI manager instance

    Returns:
        CommitInfo: Generated commit information

    Raises:
        ReleaseError: If analysis fails
    """
    try:
        if ui:
            ui.log("Analyzing changes with OpenAI...", level="debug")
            
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
        
        if ui:
            ui.log(f"✓ Changes analyzed: {commit_info.title}", style="green")
            
        return commit_info

    except Exception as e:
        if ui:
            ui.log(f"Failed to analyze changes: {e}", level="error")
        raise ReleaseError(
            ErrorCode.OPENAI_API_ERROR,
            f"Failed to analyze changes: {str(e)}"
        )

def run(project_path: Path, ui: ReleaseUI) -> Tuple[str, str]:
    """Analyze changes and generate commit info.

    Args:
        project_path: Path to the project directory
        ui: UI manager instance

    Returns:
        Tuple[str, str]: Commit message and version bump type

    Raises:
        ReleaseError: If analysis fails
    """
    ui.log("Starting change analysis...", style="blue")
    
    # Load configuration
    ui.update_progress("Loading configuration...", 45)
    config_data = load_config(project_path)
    config = Config(config_data)
    ui.log("Configuration loaded", level="debug")

    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        ui.log("OpenAI API key not found", level="error")
        raise ReleaseError(
            ErrorCode.OPENAI_API_KEY_MISSING,
            "OpenAI API key not found in environment"
        )
    ui.log("OpenAI API key found", level="debug")

    # Get git diff
    ui.update_progress("Getting git diff...", 50)
    diff = get_git_diff(ui)

    # Analyze changes with OpenAI
    ui.update_progress("Analyzing changes with OpenAI...", 55)
    commit_info = analyze_with_openai(diff, config.model, api_key, ui)

    # Commit and push changes
    ui.update_progress("Committing changes...", 58)
    commit_and_push(commit_info, ui)

    return commit_info.title, commit_info.bump_type