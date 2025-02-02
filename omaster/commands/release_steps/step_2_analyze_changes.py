"""Step 2: Analyze changes and generate commit info using OpenAI."""
import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from pydantic import BaseModel
from openai import OpenAI

from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config, load_config

logger = logging.getLogger(__name__)

class CommitInfo(BaseModel):
    """Model for commit information."""
    title: str
    description: str
    bump_type: str  # major, minor, or patch

def get_git_diff() -> str:
    """Get git diff of staged and unstaged changes.

    Returns:
        str: Git diff output

    Raises:
        ReleaseError: If git command fails
    """
    try:
        logger.info("Getting git diff...")
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
            logger.info("✓ Found unpushed commits")
            return unpushed
        logger.info("✓ Git diff retrieved successfully")
        return staged + unstaged
    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.GIT_NO_CHANGES,
            f"Failed to get git diff: {str(e)}"
        )

def commit_and_push(commit_info: CommitInfo) -> None:
    """Commit changes and push to remote.

    Args:
        commit_info: Commit information from AI

    Raises:
        ReleaseError: If git operations fail
    """
    try:
        # Check if there are any changes to commit
        status = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        if status:
            logger.info("Staging all changes...")
            subprocess.run(['git', 'add', '.'], check=True)
            logger.info("✓ Changes staged")

            logger.info("Creating commit...")
            commit_msg = f"{commit_info.title}\n\n{commit_info.description}"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            logger.info("✓ Changes committed")
        else:
            logger.info("No changes to commit, using existing commit")

        logger.info("Pushing to remote...")
        subprocess.run(['git', 'push'], check=True)
        logger.info("✓ Changes pushed to remote")

    except subprocess.CalledProcessError as e:
        raise ReleaseError(
            ErrorCode.GIT_OPERATION_FAILED,
            f"Failed to perform git operation: {str(e)}"
        )

def run(project_path: Path) -> Tuple[bool, CommitInfo]:
    """Analyze changes and generate commit info.

    Args:
        project_path: Path to the project directory

    Returns:
        Tuple[bool, CommitInfo]: Success status and commit info

    Raises:
        ReleaseError: If analysis fails
    """
    logger.info("Step 2: Analyzing changes...")

    # Load configuration
    logger.info("Loading configuration...")
    config_data = load_config(project_path)
    config = Config(config_data)
    logger.info("✓ Configuration loaded")

    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ReleaseError(
            ErrorCode.OPENAI_API_KEY_MISSING,
            "OpenAI API key not found in environment"
        )
    logger.info("✓ OpenAI API key found")

    # Get git diff
    diff = get_git_diff()

    try:
        logger.info("Analyzing changes with OpenAI...")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=config.model,
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
        logger.info(f"✓ Changes analyzed: {commit_info.title}")

        # Commit and push changes
        commit_and_push(commit_info)

        return True, commit_info

    except Exception as e:
        raise ReleaseError(
            ErrorCode.OPENAI_API_ERROR,
            f"Failed to analyze changes: {str(e)}"
        )