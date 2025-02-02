"""Step 1.5: Code Quality Analysis.

This step runs various code quality checks:
1. Complexity analysis (cyclomatic, cognitive, maintainability)
2. Dead code detection (unused functions, classes, imports)
3. Code similarity detection (duplicate code patterns)
"""

import logging
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ...core.errors import ErrorCode, ReleaseError
from ...core.config import Config
from ...quality.analyzers import (
    ComplexityAnalyzer,
    DeadCodeAnalyzer,
    SecurityAnalyzer,
    SimilarityAnalyzer,
    StyleAnalyzer
)
from ...quality.quality_issue import QualityIssue, QualityMetric

# Configure logging
logger = logging.getLogger(__name__)
console = Console()

# Timeout for each analyzer in seconds
ANALYZER_TIMEOUT = 60

class AnalyzerTimeoutError(Exception):
    """Raised when an analyzer times out."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise AnalyzerTimeoutError("Analysis timed out")

def validate_quality_issues(issues: List[Union[Dict[str, Any], QualityIssue]], analyzer_name: str) -> List[QualityIssue]:
    """Validate and convert analyzer issues to QualityIssue format.

    Args:
        issues: List of issues from an analyzer
        analyzer_name: Name of the analyzer for error reporting

    Returns:
        List of validated QualityIssue objects

    Raises:
        ReleaseError: If issues are not in the correct format
    """
    validated_issues = []

    for i, issue in enumerate(issues):
        try:
            # Convert dictionary issues to QualityIssue
            if isinstance(issue, dict):
                # Map analyzer type to QualityMetric
                try:
                    metric_type = QualityMetric(issue.get("type", "error"))
                except ValueError:
                    metric_type = QualityMetric.COMPLEXITY  # Default to complexity

                # Create QualityIssue
                validated_issue = QualityIssue(
                    type=metric_type,
                    file_path=issue["file"],
                    line=issue["line"],
                    end_line=issue.get("end_line"),
                    message=issue["message"],
                    severity=issue.get("severity", "high"),
                    weight=float(issue.get("weight", 1.0)),
                    details=issue.get("details")
                )
            elif isinstance(issue, QualityIssue):
                validated_issue = issue
            else:
                raise ValueError(f"Invalid issue type: {type(issue)}")

            validated_issues.append(validated_issue)

        except Exception as e:
            logger.error(
                f"Invalid issue from {analyzer_name} at index {i}: {str(e)}",
                exc_info=True
            )
            raise ReleaseError(
                ErrorCode.ANALYSIS_ERROR,
                f"Invalid issue format from {analyzer_name}: {str(e)}"
            )

    return validated_issues


def format_issues_table(issues: List[QualityIssue]) -> Table:
    """Format issues into a rich table.

    Args:
        issues: List of quality issues

    Returns:
        Rich table with formatted issues
    """
    table = Table(
        title="Code Quality Issues",
        show_header=True,
        header_style="bold magenta",
        border_style="blue"
    )
    
    table.add_column("Type", style="cyan")
    table.add_column("Severity", style="red")
    table.add_column("File", style="green")
    table.add_column("Line", justify="right", style="yellow")
    table.add_column("Message", style="white")

    for issue in sorted(issues, key=lambda x: (x.severity, x.file_path, x.line)):
        severity_style = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "green",
            "info": "blue"
        }.get(issue.severity, "white")

        table.add_row(
            str(issue.type.value),
            Text(issue.severity, style=severity_style),
            issue.file_path,
            str(issue.line),
            issue.message
        )

    return table

def run_analyzer_with_timeout(analyzer: Any, timeout: int = ANALYZER_TIMEOUT) -> List[QualityIssue]:
    """Run an analyzer with timeout.

    Args:
        analyzer: The analyzer instance to run
        timeout: Timeout in seconds

    Returns:
        List of quality issues

    Raises:
        AnalyzerTimeoutError: If analyzer times out
        Exception: If analyzer fails
    """
    logger.info(f"Running {analyzer.__class__.__name__}...")
    
    def analyze_wrapper():
        return analyzer.analyze()

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(analyze_wrapper)
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error(f"{analyzer.__class__.__name__} timed out after {timeout} seconds")
            raise AnalyzerTimeoutError(f"{analyzer.__class__.__name__} analysis timed out after {timeout} seconds")

def run(project_path: Path) -> bool:
    """Run code quality analysis.

    Args:
        project_path: Path to the project directory

    Returns:
        bool: True if quality checks pass

    Raises:
        ReleaseError: If quality checks fail
    """
    logger.info("Starting code quality analysis...")
    
    try:
        # Load configuration
        config = Config.load(project_path)
        quality_config = config.data.get("quality", {})

        # Initialize analyzers with configuration
        analyzers = [
            ComplexityAnalyzer(project_path, quality_config),
            DeadCodeAnalyzer(project_path, quality_config),
            SecurityAnalyzer(project_path, quality_config),
            SimilarityAnalyzer(project_path, quality_config),
            StyleAnalyzer(project_path, quality_config)
        ]

        # Run analysis
        all_issues = []
        failed_analyzers = []

        with console.status("[bold blue]Running code quality analysis...") as status:
            for analyzer in analyzers:
                analyzer_name = analyzer.__class__.__name__
                try:
                    status.update(f"[bold blue]Running {analyzer_name}...")
                    issues = run_analyzer_with_timeout(analyzer)
                    all_issues.extend(issues)
                    logger.info(f"✓ {analyzer_name} completed")
                except AnalyzerTimeoutError as e:
                    logger.error(str(e))
                    failed_analyzers.append((analyzer_name, str(e)))
                except Exception as e:
                    logger.error(f"Error in {analyzer_name}: {str(e)}", exc_info=True)
                    failed_analyzers.append((analyzer_name, str(e)))

        # Handle failed analyzers
        if failed_analyzers:
            error_panel = Panel(
                "\n".join([f"• {name}: {error}" for name, error in failed_analyzers]),
                title="[red]Analyzer Failures",
                border_style="red"
            )
            console.print(error_panel)
            raise ReleaseError(
                ErrorCode.ANALYZER_INIT_ERROR,
                "Some analyzers failed to complete",
                {"failed_analyzers": dict(failed_analyzers)}
            )

        # Format and display results
        if all_issues:
            table = format_issues_table(all_issues)
            console.print("\n")
            console.print(table)

            # Check if any critical issues
            critical_issues = [i for i in all_issues if i.severity == "critical"]
            if critical_issues:
                critical_panel = Panel(
                    "\n".join([f"• {i.message} in {i.file_path}:{i.line}" for i in critical_issues]),
                    title="[red]Critical Quality Issues",
                    border_style="red"
                )
                console.print(critical_panel)
                raise ReleaseError(
                    ErrorCode.QUALITY_ERROR,
                    f"Found {len(critical_issues)} critical quality issues",
                    {"critical_issues": [{"file": i.file_path, "line": i.line, "message": i.message} 
                                       for i in critical_issues]}
                )

        logger.info("Code quality analysis completed successfully")
        return True

    except ReleaseError:
        raise
    except Exception as e:
        logger.error("Unexpected error during code quality analysis", exc_info=True)
        error_panel = Panel(
            str(e),
            title="[red]Unexpected Error",
            border_style="red"
        )
        console.print(error_panel)
        raise ReleaseError(
            ErrorCode.UNEXPECTED_ERROR,
            "Unexpected error during code quality analysis",
            {"error": str(e), "traceback": True}
        )