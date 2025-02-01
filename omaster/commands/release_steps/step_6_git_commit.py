"""Step 6: Commit changes and push to remote."""
import os
import subprocess
from pathlib import Path
import requests
from typing import TypedDict

from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config

class CommitInfo(TypedDict):
    title: str
    description: str
    bump_type: str

def create_github_repo(config: Config) -> str:
    """Create a GitHub repository if it doesn't exist.
    
    Args:
        config: Configuration object
        
    Returns:
        str: Repository URL
        
    Raises:
        ReleaseError: If repository creation fails
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ReleaseError(
            ErrorCode.GIT_PUSH_FAILED,
            "GITHUB_TOKEN environment variable must be set"
        )
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Determine API endpoint based on org vs user
    if config.github_org:
        api_url = f"https://api.github.com/orgs/{config.github_org}/repos"
    else:
        api_url = "https://api.github.com/user/repos"
    
    # Create repository
    data = {
        "name": config.github_repo,
        "private": config.github_private,
        "auto_init": False
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        repo_data = response.json()
        return repo_data["clone_url"]
    except requests.exceptions.RequestException as e:
        if response.status_code == 422:  # Repository exists
            if config.github_org:
                return f"https://github.com/{config.github_org}/{config.github_repo}.git"
            else:
                # Get user info to construct URL
                user_response = requests.get("https://api.github.com/user", headers=headers)
                user_response.raise_for_status()
                username = user_response.json()["login"]
                return f"https://github.com/{username}/{config.github_repo}.git"
        raise ReleaseError(
            ErrorCode.GIT_PUSH_FAILED,
            f"Failed to create GitHub repository: {str(e)}"
        )

def run(project_path: Path, commit_info: CommitInfo) -> bool:
    """Commit changes and push to remote.
    
    Args:
        project_path: Path to the project directory
        commit_info: Information about the changes
        
    Returns:
        bool: True if commit and push successful
        
    Raises:
        ReleaseError: If commit or push fails
    """
    print("Step 6: Committing and pushing changes...")
    try:
        # Stage all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # Create commit with message
        subprocess.run(
            ["git", "commit", "-m", commit_info["title"], "-m", commit_info["description"]],
            check=True
        )
        
        # Check if remote exists
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # No remote exists, create repository and add remote
            config = Config(project_path)
            repo_url = create_github_repo(config)
            subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        
        # Push changes
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        
        print("✓ Changes committed and pushed\n")
        return True
        
    except subprocess.CalledProcessError as e:
        raise ReleaseError(ErrorCode.GIT_PUSH_FAILED, str(e))
    except Exception as e:
        raise ReleaseError(ErrorCode.GIT_PUSH_FAILED, str(e)) 