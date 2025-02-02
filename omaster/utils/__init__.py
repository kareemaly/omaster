"""Utility functions and helpers."""
import subprocess
from pathlib import Path
from ..core.errors import ReleaseError, ErrorCode

def run_command(cmd: str, cwd: Path, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR) -> bool:
    """Run a shell command.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        error_code: Error code to use if command fails
        
    Returns:
        bool: True if command succeeds
        
    Raises:
        ReleaseError: If command fails
    """
    try:
        subprocess.run(cmd.split(), check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        raise ReleaseError(error_code, str(e)) 