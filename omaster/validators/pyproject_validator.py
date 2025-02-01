"""Validator for pyproject.toml file."""
import tomli
from pathlib import Path
from typing import Dict, Any, List
from ..core.validator import Validator, ValidationResult


class PyProjectValidator(Validator):
    """Validates the pyproject.toml file."""
    
    @property
    def name(self) -> str:
        return "pyproject.toml validator"
    
    def _check_project_section(self, project: Dict[str, Any]) -> List[str]:
        """Check the [project] section."""
        messages = []
        
        # Check required fields
        required_fields = {
            "name": "Package name",
            "version": "Version number",
            "description": "Package description",
            "readme": "README file",
            "requires-python": "Python version requirement"
        }
        
        for field, desc in required_fields.items():
            if field not in project:
                messages.append(f"Missing {desc} in [project] section")
        
        # Check version format
        if "version" in project:
            version = project["version"]
            if not all(part.isdigit() for part in version.split(".")):
                messages.append(f"Invalid version format: {version}. Expected 'major.minor.patch'")
        
        # Check dependencies
        if "dependencies" not in project:
            messages.append("No dependencies specified in [project] section")
        elif not isinstance(project["dependencies"], list):
            messages.append("Dependencies must be a list")
        
        return messages
    
    def _check_build_system(self, pyproject: Dict[str, Any]) -> List[str]:
        """Check the [build-system] section."""
        messages = []
        
        if "build-system" not in pyproject:
            messages.append("Missing [build-system] section")
            return messages
        
        build_system = pyproject["build-system"]
        
        # Check required fields
        if "requires" not in build_system:
            messages.append("Missing 'requires' in [build-system]")
        elif not isinstance(build_system["requires"], list):
            messages.append("'requires' in [build-system] must be a list")
        
        if "build-backend" not in build_system:
            messages.append("Missing 'build-backend' in [build-system]")
        
        return messages
    
    def _check_scripts(self, project: Dict[str, Any]) -> List[str]:
        """Check the [project.scripts] section."""
        messages = []
        
        scripts = project.get("scripts", {})
        if not scripts:
            messages.append("No entry points defined in [project.scripts]")
        else:
            for script_name, entry_point in scripts.items():
                if ":" not in entry_point:
                    messages.append(f"Invalid entry point format for {script_name}: {entry_point}")
                    continue
                
                module, func = entry_point.split(":", 1)
                if not all(part.isidentifier() for part in module.split(".")):
                    messages.append(f"Invalid module path in entry point: {module}")
                if not func.isidentifier():
                    messages.append(f"Invalid function name in entry point: {func}")
        
        return messages
    
    def validate(self) -> ValidationResult:
        """Validate the pyproject.toml file."""
        messages = []
        pyproject_path = self.project_path / "pyproject.toml"
        
        if not pyproject_path.exists():
            return ValidationResult(False, ["pyproject.toml not found"])
        
        try:
            with open(pyproject_path, "rb") as f:
                pyproject = tomli.load(f)
            
            # Check project section
            if "project" not in pyproject:
                return ValidationResult(False, ["Missing [project] section"])
            
            messages.extend(self._check_project_section(pyproject["project"]))
            messages.extend(self._check_build_system(pyproject))
            messages.extend(self._check_scripts(pyproject["project"]))
            
            passed = len(messages) == 0
            return ValidationResult(passed, messages)
            
        except Exception as e:
            return ValidationResult(False, [f"Error parsing pyproject.toml: {str(e)}"]) 