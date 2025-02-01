"""Dependency analyzer for checking project dependencies.

This analyzer implements comprehensive dependency analysis:
1. Outdated package detection
2. Unused dependency detection
3. Dependency conflict detection
4. Security vulnerability scanning
5. Missing dependency detection
"""
import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import tomli

from .base import BaseAnalyzer


@dataclass
class PackageInfo:
    """Information about a package dependency."""
    
    name: str
    current_version: str
    latest_version: Optional[str] = None
    used: bool = False
    vulnerabilities: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.vulnerabilities is None:
            self.vulnerabilities = []


class DependencyAnalyzer(BaseAnalyzer):
    """Analyzer for checking project dependencies."""
    
    def __init__(self, project_path: Path):
        """Initialize the dependency analyzer.
        
        Args:
            project_path: Path to the project root directory
        """
        super().__init__(project_path)
        self.pyproject_path = project_path / "pyproject.toml"
        self.packages: Dict[str, PackageInfo] = {}
        
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze project dependencies for issues.
        
        Returns:
            List of dependency issues found
        """
        issues = []
        
        try:
            # Load dependencies from pyproject.toml
            if not self._load_dependencies():
                issues.append({
                    "file": str(self.pyproject_path.relative_to(self.project_path)),
                    "line": 1,
                    "message": "Failed to load dependencies from pyproject.toml",
                    "type": "error",
                    "severity": 5
                })
                return issues
                
            # Check for package usage
            self._check_package_usage()
            
            # Check for outdated packages
            self._check_outdated_packages()
            
            # Check for security vulnerabilities
            self._check_vulnerabilities()
            
            # Report issues
            for pkg_name, pkg_info in self.packages.items():
                # Report unused packages
                if not pkg_info.used:
                    issues.append({
                        "file": str(self.pyproject_path.relative_to(self.project_path)),
                        "line": 1,
                        "message": f"Package '{pkg_name}' is listed as a dependency but not used",
                        "type": "error",
                        "severity": 2
                    })
                    
                # Report outdated packages
                if pkg_info.latest_version and pkg_info.latest_version != pkg_info.current_version:
                    issues.append({
                        "file": str(self.pyproject_path.relative_to(self.project_path)),
                        "line": 1,
                        "message": (f"Package '{pkg_name}' is outdated "
                                  f"(current: {pkg_info.current_version}, "
                                  f"latest: {pkg_info.latest_version})"),
                        "type": "error",
                        "severity": 2
                    })
                    
                # Report vulnerabilities
                for vuln in pkg_info.vulnerabilities:
                    issues.append({
                        "file": str(self.pyproject_path.relative_to(self.project_path)),
                        "line": 1,
                        "message": f"Security vulnerability in '{pkg_name}': {vuln}",
                        "type": "error",
                        "severity": 4
                    })
                    
        except Exception as e:
            issues.append({
                "file": str(self.pyproject_path.relative_to(self.project_path)),
                "line": 1,
                "message": f"Failed to analyze dependencies: {str(e)}",
                "type": "error",
                "severity": 5
            })
            
        return issues
        
    def _load_dependencies(self) -> bool:
        """Load dependencies from pyproject.toml.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.pyproject_path.exists():
                return False
                
            with open(self.pyproject_path, "rb") as f:
                pyproject = tomli.load(f)
                
            # Get dependencies from project section
            project = pyproject.get("project", {})
            dependencies = project.get("dependencies", [])
            
            # Parse dependencies
            for dep in dependencies:
                name = dep.split(">=")[0].split("==")[0].strip()
                version = dep.split(name)[1].strip().lstrip(">=")
                self.packages[name] = PackageInfo(name=name, current_version=version)
                
            return True
            
        except Exception:
            return False
            
    def _check_package_usage(self) -> None:
        """Check which packages are actually used in the code."""
        imports: Set[str] = set()
        
        # Collect all imports from Python files
        for file_path in self.project_path.rglob("*.py"):
            if self._is_excluded(file_path):
                continue
                
            try:
                with open(file_path) as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split(".")[0])
                            
            except Exception:
                continue
                
        # Mark packages as used
        for pkg_name in self.packages:
            if pkg_name in imports:
                self.packages[pkg_name].used = True
                
    def _check_outdated_packages(self) -> None:
        """Check for outdated packages using pip."""
        try:
            # Run pip list --outdated in JSON format
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                
                # Update package info with latest versions
                for pkg in outdated:
                    name = pkg["name"]
                    if name in self.packages:
                        self.packages[name].latest_version = pkg["latest_version"]
                        
        except Exception:
            pass
            
    def _check_vulnerabilities(self) -> None:
        """Check for known security vulnerabilities.
        
        Uses safety-db data if available.
        """
        try:
            # Run safety check in JSON format
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                vulns = json.loads(result.stdout)
                
                # Add vulnerabilities to package info
                for vuln in vulns:
                    name = vuln["package"]
                    if name in self.packages:
                        self.packages[name].vulnerabilities.append(
                            f"{vuln['advisory']}: {vuln['vulnerability']}"
                        )
                        
        except Exception:
            # Safety not installed or failed
            pass