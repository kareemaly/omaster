"""Main entry point for the release pipeline."""
import logging
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .core.errors import ReleaseError, ErrorCode
from .commands.release_steps import (
    step_1_validate,
    step_2_validate_code_quality,
    step_3_clean_build,
    step_4_analyze_changes,
    step_5_bump_version,
    step_5_publish,
    step_6_git_commit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format since we'll use rich for output
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create console for rich output
console = Console()

def print_error(error: ReleaseError) -> None:
    """Print a formatted error message.
    
    Args:
        error: The error to print
    """
    error_text = Text()
    error_text.append("🚨 Error 🚨\n", style="bold red")
    error_text.append(f"\nCode: {error.code.value} - {error.code.name}\n\n")
    error_text.append(str(error))

    panel = Panel(
        error_text,
        title="Release Pipeline Error",
        expand=True,
    )
    console.print(panel)

def run_pipeline() -> None:
    """Run the release pipeline."""
    project_path = Path.cwd()
    
    # Print welcome message
    console.print("\n[bold blue]🚀 Starting Release Pipeline[/bold blue]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=False,  # Keep the progress bars after completion
    ) as progress:
        overall_task = progress.add_task("[cyan]Overall Progress", total=6)
        
        # Step 1: Validation
        step1 = progress.add_task("[green]Step 1: Project Validation", total=100)
        progress.update(step1, advance=10, description="[green]Step 1: Validating project structure...")
        step_1_validate.run(project_path)
        progress.update(step1, completed=100, description="[green]Step 1: ✓ Validation passed")
        progress.update(overall_task, advance=1)
        
        # Step 2: Code Quality
        step2 = progress.add_task("[yellow]Step 2: Code Quality", total=100)
        progress.update(step2, advance=10, description="[yellow]Step 2: Running code quality checks...")
        step_2_validate_code_quality.run(project_path)
        progress.update(step2, completed=100, description="[yellow]Step 2: ✓ Code quality passed")
        progress.update(overall_task, advance=1)
        
        # Step 3: Change Analysis
        step3 = progress.add_task("[blue]Step 3: Change Analysis", total=100)
        progress.update(step3, advance=10, description="[blue]Step 3: Analyzing changes...")
        commit_message, version_bump_type = step_4_analyze_changes.run(project_path)
        progress.update(step3, completed=100, description=f"[blue]Step 3: ✓ Changes analyzed - {commit_message}")
        progress.update(overall_task, advance=1)
        
        # Step 4: Version Update
        step4 = progress.add_task("[magenta]Step 4: Version Update", total=100)
        progress.update(step4, advance=10, description="[magenta]Step 4: Updating version...")
        step_5_bump_version.run(project_path, version_bump_type)
        progress.update(step4, completed=100, description="[magenta]Step 4: ✓ Version updated")
        progress.update(overall_task, advance=1)
        
        # Step 5: Clean and Build
        step5 = progress.add_task("[red]Step 5: Build", total=100)
        progress.update(step5, advance=10, description="[red]Step 5: Building package...")
        step_3_clean_build.run(project_path)
        progress.update(step5, completed=100, description="[red]Step 5: ✓ Build completed")
        progress.update(overall_task, advance=1)
        
        # Step 6: Publish
        step6 = progress.add_task("[cyan]Step 6: Publish", total=100)
        progress.update(step6, advance=10, description="[cyan]Step 6: Publishing package...")
        step_5_publish.run(project_path)
        progress.update(step6, completed=100, description="[cyan]Step 6: ✓ Package published")
        progress.update(overall_task, advance=1)
        
        # Final success message
        console.print("\n[bold green]✨ Release process completed successfully![/bold green]\n")

def main() -> int:
    """Main entry point."""
    try:
        run_pipeline()
        return 0
    except ReleaseError as e:
        print_error(e)
        return e.code.value
    except Exception as e:
        print_error(ReleaseError(ErrorCode.UNKNOWN_ERROR, str(e)))
        return ErrorCode.UNKNOWN_ERROR.value

if __name__ == "__main__":
    main()