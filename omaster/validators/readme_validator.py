"""Validator for README.md file."""
from pathlib import Path
from ..core.validator import Validator, ValidationResult


class ReadmeValidator(Validator):
    """Validates the README.md file."""
    
    @property
    def name(self) -> str:
        return "README.md validator"
    
    def validate(self) -> ValidationResult:
        """Validate the README.md file."""
        messages = []
        readme_path = self.project_path / "README.md"
        
        if not readme_path.exists():
            return ValidationResult(False, ["README.md not found"])
        
        try:
            content = readme_path.read_text()
            
            # Check minimum content
            if len(content.strip()) < 50:
                messages.append("README.md is too short (minimum 50 characters)")
            
            # Check sections
            required_sections = ["Installation", "Usage"]
            found_sections = [
                line.strip("# ").lower()
                for line in content.split("\n")
                if line.strip().startswith("#")
            ]
            
            missing = [
                section for section in required_sections
                if section.lower() not in found_sections
            ]
            
            if missing:
                messages.append(f"Missing required sections: {', '.join(missing)}")
            
            # Check code examples
            if "```" not in content:
                messages.append("No code examples found in README.md")
            
            passed = len(messages) == 0
            return ValidationResult(passed, messages)
            
        except Exception as e:
            return ValidationResult(False, [f"Error reading README.md: {str(e)}"]) 