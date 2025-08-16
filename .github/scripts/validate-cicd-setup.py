#!/usr/bin/env python3
"""
CI/CD Setup Validation Script for AIrsenal Enhanced Components

This script validates that all enhanced CI/CD components are properly configured
and can detect common setup issues before they cause pipeline failures.
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

class CICDValidator:
    """Validates the CI/CD setup for AIrsenal enhanced components."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating Enhanced CI/CD Setup for AIrsenal...")
        print("=" * 60)
        
        checks = [
            ("Workflow Files", self.validate_workflow_files),
            ("Dependencies", self.validate_dependencies),
            ("Test Structure", self.validate_test_structure),
            ("Configuration Files", self.validate_configuration_files),
            ("Security Setup", self.validate_security_setup),
            ("Enhanced Components", self.validate_enhanced_components),
        ]
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}:")
            try:
                check_func()
                print(f"  ✅ {check_name} validation passed")
            except Exception as e:
                self.errors.append(f"{check_name}: {str(e)}")
                print(f"  ❌ {check_name} validation failed: {e}")
        
        self.print_summary()
        return len(self.errors) == 0
    
    def validate_workflow_files(self):
        """Validate GitHub Actions workflow files exist and are properly structured."""
        workflows_dir = self.repo_root / ".github" / "workflows"
        
        required_workflows = {
            "enhanced-ci.yml": "Main enhanced CI/CD pipeline",
            "performance-tests.yml": "Performance testing suite", 
            "security-scan.yml": "Security scanning workflow",
            "deployment.yml": "Deployment automation",
            "notification-hub.yml": "Notification and reporting hub"
        }
        
        for workflow_file, description in required_workflows.items():
            workflow_path = workflows_dir / workflow_file
            if not workflow_path.exists():
                raise FileNotFoundError(f"Missing workflow: {workflow_file} ({description})")
            
            # Validate YAML syntax
            try:
                with open(workflow_path, 'r') as f:
                    workflow_config = yaml.safe_load(f)
                
                # Basic structure validation
                if 'name' not in workflow_config:
                    self.warnings.append(f"{workflow_file}: Missing 'name' field")
                
                if 'on' not in workflow_config:
                    raise ValueError(f"{workflow_file}: Missing trigger configuration ('on' field)")
                
                if 'jobs' not in workflow_config:
                    raise ValueError(f"{workflow_file}: Missing jobs configuration")
                
                self.info.append(f"  ✓ {workflow_file}: Valid structure")
                
            except yaml.YAMLError as e:
                raise ValueError(f"{workflow_file}: Invalid YAML syntax - {e}")
    
    def validate_dependencies(self):
        """Validate that required dependencies are present in pyproject.toml."""
        pyproject_path = self.repo_root / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError("Missing pyproject.toml file")
        
        try:
            import tomllib
            with open(pyproject_path, 'rb') as f:
                config = tomllib.load(f)
        except ImportError:
            # Fallback for Python < 3.11
            try:
                import tomli as tomllib
                with open(pyproject_path, 'rb') as f:
                    config = tomllib.load(f)
            except ImportError:
                self.warnings.append("Cannot validate pyproject.toml - tomllib/tomli not available")
                return
        
        # Check for enhanced component dependencies
        dependencies = config.get('project', {}).get('dependencies', [])
        dependency_strings = ' '.join(dependencies)
        
        required_deps = {
            'redis': 'Redis caching support',
            'jax': 'JAX for ML models', 
            'structlog': 'Structured logging',
            'alembic': 'Database migrations',
            'sqlalchemy': 'Database ORM'
        }
        
        for dep, description in required_deps.items():
            if dep not in dependency_strings.lower():
                self.warnings.append(f"Missing recommended dependency: {dep} ({description})")
            else:
                self.info.append(f"  ✓ Found dependency: {dep}")
        
        # Check dev dependencies
        dev_deps = config.get('project', {}).get('optional-dependencies', {}).get('dev', [])
        dev_strings = ' '.join(dev_deps)
        
        required_dev_deps = {
            'pytest': 'Testing framework',
            'pytest-cov': 'Coverage reporting', 
            'ruff': 'Linting and formatting',
            'mypy': 'Type checking'
        }
        
        for dep, description in required_dev_deps.items():
            if dep not in dev_strings.lower():
                self.errors.append(f"Missing required dev dependency: {dep} ({description})")
            else:
                self.info.append(f"  ✓ Found dev dependency: {dep}")
    
    def validate_test_structure(self):
        """Validate that enhanced component tests exist."""
        tests_dir = self.repo_root / "airsenal" / "tests"
        if not tests_dir.exists():
            raise FileNotFoundError("Missing tests directory")
        
        required_test_files = {
            "test_feature_store.py": "Feature store infrastructure tests",
            "test_model_versioning.py": "Model versioning system tests",
            "test_database_versioning.py": "Database versioning tests",
            "test_player_attributes_extended.py": "Extended PlayerAttributes tests",
            "test_base_models.py": "Base model interface tests",
            "test_redis_cache.py": "Redis caching tests"
        }
        
        for test_file, description in required_test_files.items():
            test_path = tests_dir / test_file
            if not test_path.exists():
                self.warnings.append(f"Missing test file: {test_file} ({description})")
            else:
                # Basic validation that file contains tests
                with open(test_path, 'r') as f:
                    content = f.read()
                    if 'def test_' not in content and 'class Test' not in content:
                        self.warnings.append(f"{test_file}: No test functions/classes found")
                    else:
                        self.info.append(f"  ✓ {test_file}: Contains tests")
    
    def validate_configuration_files(self):
        """Validate CI/CD configuration files."""
        config_files = {
            ".pre-commit-config.yaml": "Pre-commit hooks configuration",
            ".gitleaks.toml": "Secret scanning configuration"
        }
        
        for config_file, description in config_files.items():
            config_path = self.repo_root / config_file
            if not config_path.exists():
                if config_file == ".gitleaks.toml":
                    self.errors.append(f"Missing {config_file} ({description})")
                else:
                    self.warnings.append(f"Missing {config_file} ({description})")
            else:
                self.info.append(f"  ✓ Found {config_file}")
                
                # Validate specific configurations
                if config_file == ".pre-commit-config.yaml":
                    self._validate_precommit_config(config_path)
                elif config_file == ".gitleaks.toml":
                    self._validate_gitleaks_config(config_path)
    
    def _validate_precommit_config(self, config_path: Path):
        """Validate pre-commit configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            repos = config.get('repos', [])
            hooks_found = []
            
            for repo in repos:
                for hook in repo.get('hooks', []):
                    hooks_found.append(hook.get('id'))
            
            required_hooks = ['ruff', 'ruff-format', 'mypy']
            for hook in required_hooks:
                if hook not in hooks_found:
                    self.warnings.append(f"Pre-commit missing recommended hook: {hook}")
                
        except Exception as e:
            self.warnings.append(f"Error validating pre-commit config: {e}")
    
    def _validate_gitleaks_config(self, config_path: Path):
        """Validate GitLeaks configuration."""
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            required_sections = ['[[rules]]', '[allowlist]']
            for section in required_sections:
                if section not in content:
                    self.warnings.append(f"GitLeaks config missing section: {section}")
                    
        except Exception as e:
            self.warnings.append(f"Error validating GitLeaks config: {e}")
    
    def validate_security_setup(self):
        """Validate security configuration."""
        # Check if GitLeaks config includes AIrsenal-specific patterns
        gitleaks_path = self.repo_root / ".gitleaks.toml"
        if gitleaks_path.exists():
            with open(gitleaks_path, 'r') as f:
                content = f.read()
            
            airsernal_patterns = ['fpl-credentials', 'redis-url', 'airsenal-db-path']
            for pattern in airsernal_patterns:
                if pattern not in content:
                    self.warnings.append(f"GitLeaks missing AIrsenal pattern: {pattern}")
                else:
                    self.info.append(f"  ✓ Found security pattern: {pattern}")
        
        # Check for secrets in common locations
        potential_secret_files = [
            ".env",
            "secrets.json", 
            "config.secret.json",
            "private_key.pem"
        ]
        
        for secret_file in potential_secret_files:
            if (self.repo_root / secret_file).exists():
                self.warnings.append(f"Potential secret file in repo: {secret_file}")
    
    def validate_enhanced_components(self):
        """Validate that enhanced components are properly implemented."""
        framework_dir = self.repo_root / "airsenal" / "framework"
        
        required_components = {
            "feature_store.py": "Feature store implementation",
            "model_versioning.py": "Model versioning system",
            "database_versioning.py": "Database versioning system", 
            "redis_cache.py": "Redis caching layer",
            "base_models.py": "Base model interfaces",
            "logging_config.py": "Structured logging configuration",
            "cicd_integration.py": "CI/CD integration utilities"
        }
        
        for component_file, description in required_components.items():
            component_path = framework_dir / component_file
            if not component_path.exists():
                self.warnings.append(f"Missing component: {component_file} ({description})")
            else:
                # Basic validation that component has expected structure
                with open(component_path, 'r') as f:
                    content = f.read()
                    
                if 'class ' not in content and 'def ' not in content:
                    self.warnings.append(f"{component_file}: No classes or functions found")
                else:
                    self.info.append(f"  ✓ {component_file}: Contains implementation")
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.info:
            print(f"\n✅ INFO ({len(self.info)}):")
            for info in self.info:
                print(f"   {info}")
        
        print("\n" + "=" * 60)
        
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print("🎉 PERFECT! All validations passed with no issues.")
                print("   Your enhanced CI/CD setup is ready to go!")
            else:
                print("✅ GOOD! No critical errors found.")
                print("   Consider addressing warnings for optimal setup.")
        else:
            print("❌ ISSUES FOUND! Please fix errors before proceeding.")
            print("   Some CI/CD features may not work correctly.")
        
        print("=" * 60)

def main():
    """Main validation entry point."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent  # Go up two levels from .github/scripts/
    
    if not (repo_root / "pyproject.toml").exists():
        print("❌ Error: Not in AIrsenal repository root")
        print("   Please run this script from the repository root or ensure pyproject.toml exists")
        sys.exit(1)
    
    validator = CICDValidator(repo_root)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()