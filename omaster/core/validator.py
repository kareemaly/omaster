"""Base validator class for implementing different validators."""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, Any, List
from .errors import ReleaseError, ErrorCode

logger = logging.getLogger(__name__)

class ValidationResult:
    """Result of a validation check."""
    def __init__(self, passed: bool, messages: List[str]):
        self.passed = passed
        self.messages = messages


class Validator(ABC):
    """Base class for all validators."""

    def __init__(self, project_path: Path):
        self.project_path = project_path

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the validator."""
        pass

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Run validation and return result."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get validator configuration. Override if needed."""
        return {}


def validate_project(project_path: Path) -> bool:
    """Validate entire project.

    Args:
        project_path: Path to project directory

    Returns:
        bool: True if validation passed, False otherwise

    Raises:
        ReleaseError: If validation fails
    """
    logger.info("Starting project structure validation...")
    logger.info(f"Project path: {project_path}")

    try:
        # Check pyproject.toml
        logger.info("\nChecking pyproject.toml...")
        pyproject_path = project_path / "pyproject.toml"
        if not pyproject_path.exists():
            logger.error("pyproject.toml not found")
            raise ReleaseError(
                ErrorCode.MISSING_PYPROJECT,
                f"pyproject.toml not found in {project_path}"
            )
        logger.info("✓ pyproject.toml found")

        # Check README.md
        logger.info("\nChecking README.md...")
        readme_path = project_path / "README.md"
        if not readme_path.exists():
            logger.error("README.md not found")
            raise ReleaseError(
                ErrorCode.MISSING_README,
                f"README.md not found in {project_path}"
            )
        logger.info("✓ README.md found")

        # Read and validate pyproject.toml
        logger.info("\nValidating pyproject.toml contents...")
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                pyproject = tomli.load(f)

            # Check required fields
            required_fields = ["name", "version", "description", "readme"]
            missing_fields = []
            for field in required_fields:
                if field not in pyproject.get("project", {}):
                    missing_fields.append(field)
            
            if missing_fields:
                logger.error(f"Missing required fields in pyproject.toml: {', '.join(missing_fields)}")
                raise ReleaseError(
                    ErrorCode.INVALID_PYPROJECT,
                    f"Missing required fields in pyproject.toml: {', '.join(missing_fields)}"
                )
            logger.info("✓ All required fields present in pyproject.toml")

        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {str(e)}")
            raise ReleaseError(
                ErrorCode.INVALID_PYPROJECT,
                f"Failed to parse pyproject.toml: {str(e)}"
            )

        # Read and validate README.md
        logger.info("\nValidating README.md contents...")
        try:
            content = readme_path.read_text()

            # Check length
            if len(content) < 50:
                logger.error("README.md is too short (minimum 50 characters)")
                raise ReleaseError(
                    ErrorCode.INVALID_README,
                    "README.md is too short (minimum 50 characters)"
                )
            logger.info("✓ README.md length is sufficient")

            # Check required sections
            required_sections = ["Installation", "Usage"]
            missing_sections = []
            for section in required_sections:
                if f"## {section}" not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.error(f"Missing required sections in README.md: {', '.join(missing_sections)}")
                raise ReleaseError(
                    ErrorCode.INVALID_README,
                    f"Missing required sections in README.md: {', '.join(missing_sections)}"
                )
            logger.info("✓ All required sections present in README.md")

        except ReleaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to read README.md: {str(e)}")
            raise ReleaseError(
                ErrorCode.INVALID_README,
                f"Failed to read README.md: {str(e)}"
            )

        logger.info("\n✓ Project structure validation completed successfully")
        return True

    except ReleaseError:
        raise
    except Exception as e:
        logger.error("Unexpected error during project validation", exc_info=True)
        raise ReleaseError(
            ErrorCode.UNKNOWN_ERROR,
            "Unexpected error during project validation",
            {"error": str(e)}
        )