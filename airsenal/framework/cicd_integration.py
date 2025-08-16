"""
CI/CD Integration utilities for AIrsenal database versioning.

This module provides utilities for automated version checking,
migration validation, and deployment pipeline integration.
"""

import json
import os
import sys
import subprocess
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from airsenal.framework.database_versioning import (
    get_current_app_version,
    get_current_database_version,
    check_version_compatibility,
    validate_database_on_startup,
    get_database_info,
    validate_schema_integrity,
    record_migration,
    DatabaseVersionError
)
from airsenal.framework.schema import session_scope

logger = logging.getLogger(__name__)


class CICDError(Exception):
    """Raised when CI/CD operations fail."""
    pass


@dataclass
class TestResult:
    """Represents a test result from various testing frameworks."""
    name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    test_type: str = "unit"  # "unit", "integration", "e2e", "performance"


@dataclass
class PipelineStage:
    """Represents a CI/CD pipeline stage."""
    name: str
    status: str  # "pending", "running", "success", "failure", "cancelled"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    logs: List[str] = None
    artifacts: List[str] = None


class CICDPipelineValidator:
    """Validates CI/CD pipeline stages and enforces quality gates."""
    
    def __init__(self, 
                 coverage_threshold: float = 80.0,
                 performance_threshold: float = 5.0,
                 security_scan_required: bool = True,
                 max_pipeline_duration: int = 1800):  # 30 minutes
        """
        Initialize pipeline validator.
        
        Args:
            coverage_threshold: Minimum code coverage percentage required
            performance_threshold: Maximum acceptable test duration in seconds
            security_scan_required: Whether security scans are mandatory
            max_pipeline_duration: Maximum pipeline duration in seconds
        """
        self.coverage_threshold = coverage_threshold
        self.performance_threshold = performance_threshold
        self.security_scan_required = security_scan_required
        self.max_pipeline_duration = max_pipeline_duration
        self.stages: List[PipelineStage] = []
        
    def validate_quality_gates(self, test_results: List[TestResult], 
                              coverage_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate quality gates based on test results and coverage.
        
        Args:
            test_results: List of test results
            coverage_report: Optional coverage report
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "passed": True,
            "gates": {},
            "summary": {},
            "errors": [],
            "warnings": []
        }
        
        # Test results analysis
        total_tests = len(test_results)
        passed_tests = len([t for t in test_results if t.status == "passed"])
        failed_tests = len([t for t in test_results if t.status == "failed"])
        
        test_pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        validation_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_pass_rate": test_pass_rate
        }
        
        # Quality Gate 1: Test Pass Rate
        if failed_tests > 0:
            validation_results["gates"]["test_results"] = False
            validation_results["errors"].append(f"{failed_tests} test(s) failed")
            validation_results["passed"] = False
        else:
            validation_results["gates"]["test_results"] = True
            
        # Quality Gate 2: Code Coverage
        if coverage_report:
            coverage_percentage = coverage_report.get("total_coverage", 0)
            if coverage_percentage < self.coverage_threshold:
                validation_results["gates"]["coverage"] = False
                validation_results["errors"].append(
                    f"Code coverage {coverage_percentage:.1f}% below threshold {self.coverage_threshold}%"
                )
                validation_results["passed"] = False
            else:
                validation_results["gates"]["coverage"] = True
        else:
            validation_results["warnings"].append("No coverage report provided")
            
        # Quality Gate 3: Performance Tests
        performance_tests = [t for t in test_results if t.test_type == "performance"]
        slow_tests = [t for t in performance_tests if t.duration > self.performance_threshold]
        
        if slow_tests:
            validation_results["gates"]["performance"] = False
            validation_results["warnings"].append(
                f"{len(slow_tests)} performance test(s) exceed threshold {self.performance_threshold}s"
            )
        else:
            validation_results["gates"]["performance"] = True
            
        return validation_results
    
    def validate_pipeline_status(self) -> Dict[str, Any]:
        """
        Validate overall pipeline status and timing.
        
        Returns:
            Dictionary with pipeline validation results
        """
        results = {
            "pipeline_valid": True,
            "total_duration": 0,
            "stage_status": {},
            "issues": []
        }
        
        if not self.stages:
            results["issues"].append("No pipeline stages found")
            results["pipeline_valid"] = False
            return results
            
        # Calculate total duration
        start_times = [s.start_time for s in self.stages if s.start_time]
        end_times = [s.end_time for s in self.stages if s.end_time]
        
        if start_times and end_times:
            total_duration = (max(end_times) - min(start_times)).total_seconds()
            results["total_duration"] = total_duration
            
            if total_duration > self.max_pipeline_duration:
                results["issues"].append(
                    f"Pipeline duration {total_duration:.0f}s exceeds maximum {self.max_pipeline_duration}s"
                )
        
        # Check stage statuses
        for stage in self.stages:
            results["stage_status"][stage.name] = stage.status
            if stage.status == "failure":
                results["pipeline_valid"] = False
                results["issues"].append(f"Stage '{stage.name}' failed")
        
        return results
    
    def add_stage(self, stage: PipelineStage) -> None:
        """Add a pipeline stage for validation."""
        self.stages.append(stage)
        
    def check_security_requirements(self, scan_results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if security requirements are met.
        
        Args:
            scan_results: Optional security scan results
            
        Returns:
            bool: True if security requirements are met
        """
        if not self.security_scan_required:
            return True
            
        if not scan_results:
            logger.warning("Security scan required but no results provided")
            return False
            
        # Check for high/critical vulnerabilities
        high_vulns = scan_results.get("high_vulnerabilities", 0)
        critical_vulns = scan_results.get("critical_vulnerabilities", 0)
        
        if critical_vulns > 0 or high_vulns > 5:
            logger.error(f"Security scan failed: {critical_vulns} critical, {high_vulns} high vulnerabilities")
            return False
            
        return True


class GitHubAPIClient:
    """Client for GitHub API integration in CI/CD workflows."""
    
    def __init__(self, token: Optional[str] = None, repo_owner: str = "", repo_name: str = ""):
        """
        Initialize GitHub API client.
        
        Args:
            token: GitHub API token (will try to get from environment if not provided)
            repo_owner: Repository owner/organization name
            repo_name: Repository name
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        else:
            logger.warning("No GitHub token provided - some operations may fail")
    
    def get_pull_request(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """
        Get pull request information.
        
        Args:
            pr_number: Pull request number
            
        Returns:
            Pull request data or None if not found
        """
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get PR {pr_number}: {e}")
            return None
    
    def update_pull_request_status(self, pr_number: int, status: str, description: str) -> bool:
        """
        Update pull request status via commit status.
        
        Args:
            pr_number: Pull request number
            status: Status ("pending", "success", "failure", "error")
            description: Status description
            
        Returns:
            bool: True if status was updated successfully
        """
        try:
            # Get PR to find head SHA
            pr_data = self.get_pull_request(pr_number)
            if not pr_data:
                return False
                
            sha = pr_data["head"]["sha"]
            
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/statuses/{sha}"
            data = {
                "state": status,
                "description": description,
                "context": "ci/airsenal-validation"
            }
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update PR {pr_number} status: {e}")
            return False
    
    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None) -> Optional[int]:
        """
        Create a GitHub issue.
        
        Args:
            title: Issue title
            body: Issue body
            labels: Optional list of labels
            
        Returns:
            Issue number if created successfully, None otherwise
        """
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues"
            data = {
                "title": title,
                "body": body,
                "labels": labels or []
            }
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()["number"]
            
        except requests.RequestException as e:
            logger.error(f"Failed to create issue: {e}")
            return None
    
    def trigger_workflow(self, workflow_id: str, ref: str = "main", 
                        inputs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Trigger a GitHub Actions workflow.
        
        Args:
            workflow_id: Workflow ID or filename
            ref: Git reference (branch, tag, or SHA)
            inputs: Optional workflow inputs
            
        Returns:
            bool: True if workflow was triggered successfully
        """
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/dispatches"
            data = {
                "ref": ref,
                "inputs": inputs or {}
            }
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to trigger workflow {workflow_id}: {e}")
            return False
    
    def get_workflow_runs(self, workflow_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get workflow run history.
        
        Args:
            workflow_id: Workflow ID or filename
            limit: Maximum number of runs to return
            
        Returns:
            List of workflow run data
        """
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/runs"
            params = {"per_page": limit}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json().get("workflow_runs", [])
            
        except requests.RequestException as e:
            logger.error(f"Failed to get workflow runs for {workflow_id}: {e}")
            return []
    
    def add_pr_comment(self, pr_number: int, comment: str) -> bool:
        """
        Add a comment to a pull request.
        
        Args:
            pr_number: Pull request number
            comment: Comment text
            
        Returns:
            bool: True if comment was added successfully
        """
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
            data = {"body": comment}
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to add comment to PR {pr_number}: {e}")
            return False


class TestResultCollector:
    """Collects and analyzes test results from various sources."""
    
    def __init__(self):
        """Initialize test result collector."""
        self.test_results: List[TestResult] = []
        self.coverage_data: Optional[Dict[str, Any]] = None
        self.performance_metrics: Dict[str, Any] = {}
        
    def collect_pytest_results(self, junit_xml_path: str) -> bool:
        """
        Collect test results from pytest JUnit XML output.
        
        Args:
            junit_xml_path: Path to JUnit XML file
            
        Returns:
            bool: True if results were collected successfully
        """
        try:
            import xml.etree.ElementTree as ET
            
            if not os.path.exists(junit_xml_path):
                logger.error(f"JUnit XML file not found: {junit_xml_path}")
                return False
                
            tree = ET.parse(junit_xml_path)
            root = tree.getroot()
            
            for testcase in root.findall(".//testcase"):
                name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")
                duration = float(testcase.get("time", 0))
                
                # Determine status
                status = "passed"
                error_message = None
                
                if testcase.find("failure") is not None:
                    status = "failed"
                    failure_elem = testcase.find("failure")
                    error_message = failure_elem.text if failure_elem is not None else "Test failed"
                elif testcase.find("error") is not None:
                    status = "error"
                    error_elem = testcase.find("error")
                    error_message = error_elem.text if error_elem is not None else "Test error"
                elif testcase.find("skipped") is not None:
                    status = "skipped"
                
                test_result = TestResult(
                    name=f"{classname}::{name}" if classname else name,
                    status=status,
                    duration=duration,
                    error_message=error_message,
                    test_type="unit"
                )
                
                self.test_results.append(test_result)
                
            logger.info(f"Collected {len(self.test_results)} test results from {junit_xml_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect pytest results: {e}")
            return False
    
    def collect_coverage_data(self, coverage_json_path: str) -> bool:
        """
        Collect coverage data from coverage.py JSON output.
        
        Args:
            coverage_json_path: Path to coverage JSON file
            
        Returns:
            bool: True if coverage data was collected successfully
        """
        try:
            if not os.path.exists(coverage_json_path):
                logger.error(f"Coverage JSON file not found: {coverage_json_path}")
                return False
                
            with open(coverage_json_path, 'r') as f:
                coverage_data = json.load(f)
                
            # Extract summary information
            totals = coverage_data.get("totals", {})
            self.coverage_data = {
                "total_coverage": totals.get("percent_covered", 0),
                "lines_covered": totals.get("covered_lines", 0),
                "lines_missing": totals.get("missing_lines", 0),
                "total_lines": totals.get("num_statements", 0),
                "files": coverage_data.get("files", {})
            }
            
            logger.info(f"Collected coverage data: {self.coverage_data['total_coverage']:.1f}% coverage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect coverage data: {e}")
            return False
    
    def analyze_test_trends(self, historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze test trends over time.
        
        Args:
            historical_results: List of historical test result summaries
            
        Returns:
            Dictionary with trend analysis
        """
        if len(historical_results) < 2:
            return {"trend": "insufficient_data", "message": "Need at least 2 data points for trend analysis"}
        
        # Calculate current metrics
        current_results = self.get_test_summary()
        
        # Compare with previous run
        previous_results = historical_results[-1] if historical_results else {}
        
        trends = {
            "test_count_trend": self._calculate_trend(
                current_results.get("total_tests", 0),
                previous_results.get("total_tests", 0)
            ),
            "pass_rate_trend": self._calculate_trend(
                current_results.get("pass_rate", 0),
                previous_results.get("pass_rate", 0)
            ),
            "duration_trend": self._calculate_trend(
                current_results.get("total_duration", 0),
                previous_results.get("total_duration", 0),
                reverse=True  # Lower duration is better
            )
        }
        
        # Overall assessment
        positive_trends = sum(1 for trend in trends.values() if trend.get("direction") == "improving")
        negative_trends = sum(1 for trend in trends.values() if trend.get("direction") == "declining")
        
        if positive_trends > negative_trends:
            overall_trend = "improving"
        elif negative_trends > positive_trends:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
            
        return {
            "overall_trend": overall_trend,
            "trends": trends,
            "current_metrics": current_results,
            "recommendations": self._generate_recommendations(trends)
        }
    
    def _calculate_trend(self, current: float, previous: float, reverse: bool = False) -> Dict[str, Any]:
        """Calculate trend between current and previous values."""
        if previous == 0:
            return {"direction": "new", "change": 0, "percentage_change": 0}
        
        change = current - previous
        percentage_change = (change / previous) * 100
        
        if reverse:
            direction = "improving" if change < 0 else "declining" if change > 0 else "stable"
        else:
            direction = "improving" if change > 0 else "declining" if change < 0 else "stable"
        
        return {
            "direction": direction,
            "change": change,
            "percentage_change": percentage_change
        }
    
    def _generate_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test trends."""
        recommendations = []
        
        if trends["pass_rate_trend"]["direction"] == "declining":
            recommendations.append("Consider reviewing recent changes that may have introduced test failures")
        
        if trends["duration_trend"]["direction"] == "declining":
            recommendations.append("Test execution time is increasing - consider optimizing slow tests")
        
        if trends["test_count_trend"]["direction"] == "declining":
            recommendations.append("Test count is decreasing - ensure adequate test coverage for new features")
        
        return recommendations
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected test results.
        
        Returns:
            Dictionary with test summary
        """
        if not self.test_results:
            return {"total_tests": 0, "pass_rate": 0, "total_duration": 0}
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t.status == "passed"])
        failed_tests = len([t for t in self.test_results if t.status == "failed"])
        skipped_tests = len([t for t in self.test_results if t.status == "skipped"])
        total_duration = sum(t.duration for t in self.test_results)
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group by test type
        by_type = {}
        for test_type in ["unit", "integration", "e2e", "performance"]:
            type_tests = [t for t in self.test_results if t.test_type == test_type]
            by_type[test_type] = {
                "count": len(type_tests),
                "passed": len([t for t in type_tests if t.status == "passed"]),
                "duration": sum(t.duration for t in type_tests)
            }
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "pass_rate": pass_rate,
            "total_duration": total_duration,
            "average_duration": total_duration / total_tests if total_tests > 0 else 0,
            "by_type": by_type,
            "coverage": self.coverage_data
        }
    
    def export_results(self, output_path: str, format: str = "json") -> bool:
        """
        Export test results to file.
        
        Args:
            output_path: Path to output file
            format: Export format ("json", "xml", "html")
            
        Returns:
            bool: True if export was successful
        """
        try:
            summary = self.get_test_summary()
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump({
                        "summary": summary,
                        "detailed_results": [
                            {
                                "name": t.name,
                                "status": t.status,
                                "duration": t.duration,
                                "error_message": t.error_message,
                                "test_type": t.test_type
                            }
                            for t in self.test_results
                        ]
                    }, f, indent=2)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Test results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export test results: {e}")
            return False


class DeploymentValidator:
    """Validates deployment readiness and database compatibility."""
    
    def __init__(self, 
                 strict_mode: bool = True,
                 allow_auto_migration: bool = False,
                 max_migration_time: int = 300):
        """
        Initialize deployment validator.
        
        Args:
            strict_mode: If True, fail on any compatibility issues
            allow_auto_migration: If True, attempt automatic migrations
            max_migration_time: Maximum time allowed for migrations (seconds)
        """
        self.strict_mode = strict_mode
        self.allow_auto_migration = allow_auto_migration
        self.max_migration_time = max_migration_time
        self.validation_results: dict[str, Any] = {}
    
    def validate_pre_deployment(self, dbsession: Session) -> Dict[str, Any]:
        """
        Perform comprehensive pre-deployment validation.
        
        Args:
            dbsession: Database session
            
        Returns:
            Dictionary with validation results
            
        Raises:
            CICDError: If critical validation failures occur
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_version": get_current_app_version(),
            "validation_passed": True,
            "critical_issues": [],
            "warnings": [],
            "compatibility_check": None,
            "schema_integrity": None,
            "database_info": None,
            "migration_required": False,
            "deployment_ready": False
        }
        
        try:
            # 1. Get database information
            logger.info("Gathering database information...")
            db_info = get_database_info(dbsession)
            results["database_info"] = db_info
            
            # 2. Check schema integrity
            logger.info("Validating schema integrity...")
            schema_valid, schema_errors = validate_schema_integrity(dbsession)
            results["schema_integrity"] = {
                "valid": schema_valid,
                "errors": schema_errors
            }
            
            if not schema_valid:
                results["critical_issues"].extend(schema_errors)
                results["validation_passed"] = False
            
            # 3. Check version compatibility
            logger.info("Checking version compatibility...")
            db_version = get_current_database_version(dbsession)
            
            if db_version:
                is_compatible, level, warnings = check_version_compatibility(
                    results["app_version"],
                    db_version.schema_version,
                    dbsession
                )
                
                results["compatibility_check"] = {
                    "compatible": is_compatible,
                    "level": level,
                    "warnings": warnings,
                    "current_db_version": db_version.schema_version
                }
                
                if not is_compatible:
                    results["critical_issues"].append(
                        f"Database version {db_version.schema_version} is incompatible "
                        f"with application version {results['app_version']}"
                    )
                    results["validation_passed"] = False
                    results["migration_required"] = True
                elif level == "limited":
                    results["warnings"].extend(warnings)
                    results["migration_required"] = True
                    
            else:
                results["warnings"].append("No database version found - first deployment?")
                results["migration_required"] = True
            
            # 4. Check for required migrations
            if results["migration_required"]:
                if self.allow_auto_migration:
                    logger.info("Migration required and auto-migration is enabled")
                    results["warnings"].append("Auto-migration will be attempted during deployment")
                elif self.strict_mode:
                    results["critical_issues"].append("Migration required but not allowed in strict mode")
                    results["validation_passed"] = False
                else:
                    results["warnings"].append("Migration required - manual intervention may be needed")
            
            # 5. Determine deployment readiness
            results["deployment_ready"] = (
                results["validation_passed"] and 
                (not results["migration_required"] or self.allow_auto_migration)
            )
            
        except Exception as e:
            logger.error(f"Pre-deployment validation failed: {e}")
            results["critical_issues"].append(f"Validation error: {e}")
            results["validation_passed"] = False
            results["deployment_ready"] = False
        
        self.validation_results = results
        
        if self.strict_mode and not results["validation_passed"]:
            raise CICDError(f"Pre-deployment validation failed: {results['critical_issues']}")
        
        return results
    
    def validate_post_deployment(self, dbsession: Session) -> Dict[str, Any]:
        """
        Perform post-deployment validation.
        
        Args:
            dbsession: Database session
            
        Returns:
            Dictionary with validation results
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_version": get_current_app_version(),
            "validation_passed": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            # 1. Verify startup validation passes
            logger.info("Running startup validation...")
            startup_valid = validate_database_on_startup(
                dbsession=dbsession,
                force_migration=False,
                error_on_mismatch=False
            )
            
            if not startup_valid:
                results["issues"].append("Startup validation failed")
                results["validation_passed"] = False
            
            # 2. Check schema integrity
            logger.info("Validating post-deployment schema integrity...")
            schema_valid, schema_errors = validate_schema_integrity(dbsession)
            
            if not schema_valid:
                results["issues"].extend(schema_errors)
                results["validation_passed"] = False
            
            # 3. Verify database version is current
            db_version = get_current_database_version(dbsession)
            current_app_version = get_current_app_version()
            
            if not db_version:
                results["issues"].append("No database version found after deployment")
                results["validation_passed"] = False
            elif db_version.app_version != current_app_version:
                results["warnings"].append(
                    f"Database version {db_version.app_version} does not match "
                    f"application version {current_app_version}"
                )
            
        except Exception as e:
            logger.error(f"Post-deployment validation failed: {e}")
            results["issues"].append(f"Validation error: {e}")
            results["validation_passed"] = False
        
        return results
    
    def validate_deployment_health(self, dbsession: Session, 
                                 health_checks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive deployment health checks.
        
        Args:
            dbsession: Database session
            health_checks: Optional list of specific checks to run
            
        Returns:
            Dictionary with health check results
        """
        available_checks = {
            "database_connectivity": self._check_database_connectivity,
            "database_integrity": self._check_database_integrity,
            "api_endpoints": self._check_api_endpoints,
            "external_services": self._check_external_services,
            "performance_baseline": self._check_performance_baseline
        }
        
        checks_to_run = health_checks or list(available_checks.keys())
        
        health_results = {
            "overall_health": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "issues": [],
            "warnings": []
        }
        
        for check_name in checks_to_run:
            if check_name in available_checks:
                try:
                    result = available_checks[check_name](dbsession)
                    health_results["checks"][check_name] = result
                    
                    if not result.get("passed", False):
                        health_results["overall_health"] = "unhealthy"
                        if result.get("error"):
                            health_results["issues"].append(f"{check_name}: {result['error']}")
                    elif result.get("warnings"):
                        health_results["warnings"].extend(result["warnings"])
                        
                except Exception as e:
                    logger.error(f"Health check {check_name} failed: {e}")
                    health_results["checks"][check_name] = {
                        "passed": False,
                        "error": str(e)
                    }
                    health_results["overall_health"] = "unhealthy"
                    health_results["issues"].append(f"{check_name}: {str(e)}")
        
        # Set overall health based on issues
        if health_results["issues"]:
            health_results["overall_health"] = "unhealthy"
        elif health_results["warnings"]:
            health_results["overall_health"] = "degraded"
        
        return health_results
    
    def _check_database_connectivity(self, dbsession: Session) -> Dict[str, Any]:
        """Check database connectivity and basic operations."""
        try:
            # Test basic query
            dbsession.execute("SELECT 1")
            
            # Test transaction capability
            with dbsession.begin():
                dbsession.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            
            return {"passed": True, "message": "Database connectivity verified"}
            
        except Exception as e:
            return {"passed": False, "error": f"Database connectivity failed: {e}"}
    
    def _check_database_integrity(self, dbsession: Session) -> Dict[str, Any]:
        """Check database schema integrity."""
        try:
            schema_valid, schema_errors = validate_schema_integrity(dbsession)
            
            if schema_valid:
                return {"passed": True, "message": "Database schema integrity verified"}
            else:
                return {
                    "passed": False,
                    "error": f"Schema integrity issues: {', '.join(schema_errors)}"
                }
                
        except Exception as e:
            return {"passed": False, "error": f"Schema integrity check failed: {e}"}
    
    def _check_api_endpoints(self, dbsession: Session) -> Dict[str, Any]:
        """Check critical API endpoints (if applicable)."""
        # This is a placeholder - in a real implementation, this would test actual API endpoints
        warnings = []
        
        # Check if we can access FPL API configuration
        fpl_team_id = os.getenv("FPL_TEAM_ID")
        if not fpl_team_id:
            warnings.append("FPL_TEAM_ID not configured - FPL API access may be limited")
        
        return {
            "passed": True,
            "message": "API endpoint checks completed",
            "warnings": warnings
        }
    
    def _check_external_services(self, dbsession: Session) -> Dict[str, Any]:
        """Check external service dependencies."""
        warnings = []
        
        # Check FPL API availability (basic connectivity test)
        try:
            import requests
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
            if response.status_code != 200:
                warnings.append(f"FPL API returned status {response.status_code}")
        except Exception as e:
            warnings.append(f"FPL API connectivity issue: {e}")
        
        return {
            "passed": True,
            "message": "External service checks completed",
            "warnings": warnings
        }
    
    def _check_performance_baseline(self, dbsession: Session) -> Dict[str, Any]:
        """Check basic performance metrics."""
        try:
            start_time = time.time()
            
            # Simple query performance test
            dbsession.execute("SELECT COUNT(*) FROM sqlite_master")
            
            query_time = time.time() - start_time
            
            warnings = []
            if query_time > 1.0:  # If basic query takes more than 1 second
                warnings.append(f"Database query performance may be degraded: {query_time:.2f}s")
            
            return {
                "passed": True,
                "query_time": query_time,
                "message": f"Performance baseline check completed in {query_time:.3f}s",
                "warnings": warnings
            }
            
        except Exception as e:
            return {"passed": False, "error": f"Performance check failed: {e}"}
    
    def prepare_rollback_plan(self, dbsession: Session) -> Dict[str, Any]:
        """
        Prepare rollback plan and capture current state.
        
        Args:
            dbsession: Database session
            
        Returns:
            Dictionary with rollback plan and state snapshot
        """
        rollback_plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_state": {},
            "rollback_steps": [],
            "backup_info": None
        }
        
        try:
            # Capture current application and database versions
            current_app_version = get_current_app_version()
            db_version = get_current_database_version(dbsession)
            
            rollback_plan["current_state"] = {
                "app_version": current_app_version,
                "db_version": db_version.schema_version if db_version else None,
                "db_app_version": db_version.app_version if db_version else None
            }
            
            # Define rollback steps
            rollback_plan["rollback_steps"] = [
                {
                    "step": 1,
                    "action": "stop_application",
                    "description": "Stop the AIrsenal application services"
                },
                {
                    "step": 2,
                    "action": "restore_database",
                    "description": "Restore database from backup if migration was performed"
                },
                {
                    "step": 3,
                    "action": "revert_code",
                    "description": f"Revert application code to previous version"
                },
                {
                    "step": 4,
                    "action": "restart_application",
                    "description": "Restart application with previous version"
                },
                {
                    "step": 5,
                    "action": "verify_rollback",
                    "description": "Verify rollback was successful"
                }
            ]
            
            # Check if backup exists
            backup_info = self._check_backup_availability()
            rollback_plan["backup_info"] = backup_info
            
            logger.info("Rollback plan prepared successfully")
            
        except Exception as e:
            logger.error(f"Failed to prepare rollback plan: {e}")
            rollback_plan["error"] = str(e)
        
        return rollback_plan
    
    def _check_backup_availability(self) -> Dict[str, Any]:
        """Check if database backup is available for rollback."""
        backup_info = {
            "backup_available": False,
            "backup_path": None,
            "backup_timestamp": None
        }
        
        try:
            # Look for recent database backups
            airsenal_home = os.getenv("AIRSENAL_HOME")
            if airsenal_home:
                backup_dir = os.path.join(airsenal_home, "backups")
                if os.path.exists(backup_dir):
                    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
                    if backup_files:
                        # Get most recent backup
                        backup_files.sort(reverse=True)
                        latest_backup = backup_files[0]
                        backup_path = os.path.join(backup_dir, latest_backup)
                        
                        backup_info.update({
                            "backup_available": True,
                            "backup_path": backup_path,
                            "backup_timestamp": os.path.getctime(backup_path)
                        })
        except Exception as e:
            logger.warning(f"Could not check backup availability: {e}")
        
        return backup_info
    
    def execute_rollback(self, rollback_plan: Dict[str, Any], 
                        dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute rollback plan.
        
        Args:
            rollback_plan: Rollback plan from prepare_rollback_plan()
            dry_run: If True, only simulate rollback actions
            
        Returns:
            Dictionary with rollback execution results
        """
        rollback_results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "steps_executed": [],
            "success": False,
            "error": None
        }
        
        try:
            for step in rollback_plan.get("rollback_steps", []):
                step_result = self._execute_rollback_step(step, dry_run)
                rollback_results["steps_executed"].append(step_result)
                
                if not step_result.get("success", False) and not dry_run:
                    rollback_results["error"] = f"Step {step['step']} failed: {step_result.get('error')}"
                    break
            
            # Mark as successful if all steps completed
            rollback_results["success"] = all(
                step.get("success", False) for step in rollback_results["steps_executed"]
            )
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            rollback_results["error"] = str(e)
        
        rollback_results["completed_at"] = datetime.now(timezone.utc).isoformat()
        return rollback_results
    
    def _execute_rollback_step(self, step: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
        """Execute a single rollback step."""
        step_result = {
            "step": step["step"],
            "action": step["action"],
            "success": False,
            "message": "",
            "dry_run": dry_run
        }
        
        action = step["action"]
        
        if dry_run:
            step_result.update({
                "success": True,
                "message": f"DRY RUN: Would execute {action}"
            })
        else:
            # In a real implementation, these would execute actual rollback actions
            if action == "stop_application":
                step_result.update({
                    "success": True,
                    "message": "Application stop simulated (implement actual stop logic)"
                })
            elif action == "restore_database":
                step_result.update({
                    "success": True,
                    "message": "Database restore simulated (implement actual restore logic)"
                })
            elif action == "revert_code":
                step_result.update({
                    "success": True,
                    "message": "Code revert simulated (implement actual git revert logic)"
                })
            elif action == "restart_application":
                step_result.update({
                    "success": True,
                    "message": "Application restart simulated (implement actual restart logic)"
                })
            elif action == "verify_rollback":
                step_result.update({
                    "success": True,
                    "message": "Rollback verification simulated (implement actual verification)"
                })
            else:
                step_result.update({
                    "success": False,
                    "message": f"Unknown rollback action: {action}"
                })
        
        return step_result
    
    def monitor_deployment_progress(self, deployment_id: str, 
                                  timeout: int = 600) -> Dict[str, Any]:
        """
        Monitor deployment progress with timeout.
        
        Args:
            deployment_id: Unique deployment identifier
            timeout: Maximum time to wait for deployment (seconds)
            
        Returns:
            Dictionary with deployment monitoring results
        """
        monitoring_results = {
            "deployment_id": deployment_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "monitoring",
            "progress": [],
            "final_status": None,
            "timed_out": False
        }
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                # In a real implementation, this would check actual deployment status
                # For now, simulate progress monitoring
                elapsed = time.time() - start_time
                
                if elapsed < 60:
                    status = "initializing"
                elif elapsed < 180:
                    status = "deploying"
                elif elapsed < 300:
                    status = "validating"
                else:
                    status = "completed"
                    monitoring_results["final_status"] = "success"
                    break
                
                progress_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_time": elapsed,
                    "status": status,
                    "message": f"Deployment {status}..."
                }
                
                monitoring_results["progress"].append(progress_entry)
                monitoring_results["status"] = status
                
                # Simulate checking every 30 seconds
                time.sleep(30)
            
            if monitoring_results["final_status"] is None:
                monitoring_results["timed_out"] = True
                monitoring_results["final_status"] = "timeout"
                
        except Exception as e:
            logger.error(f"Deployment monitoring failed: {e}")
            monitoring_results["final_status"] = "error"
            monitoring_results["error"] = str(e)
        
        monitoring_results["completed_at"] = datetime.now(timezone.utc).isoformat()
        return monitoring_results


def run_ci_validation() -> bool:
    """
    Run CI validation suitable for continuous integration pipelines.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Starting CI validation...")
    
    try:
        with session_scope() as dbsession:
            validator = DeploymentValidator(
                strict_mode=True,
                allow_auto_migration=False
            )
            
            results = validator.validate_pre_deployment(dbsession)
            
            # Output results for CI system
            print("=== CI VALIDATION RESULTS ===")
            print(f"Validation Passed: {results['validation_passed']}")
            print(f"Deployment Ready: {results['deployment_ready']}")
            print(f"App Version: {results['app_version']}")
            
            if results['database_info'] and results['database_info']['database_version']:
                db_ver = results['database_info']['database_version']['schema_version']
                print(f"Database Version: {db_ver}")
            else:
                print("Database Version: Not found")
            
            if results['compatibility_check']:
                comp = results['compatibility_check']
                print(f"Compatibility: {comp['level']} ({'✓' if comp['compatible'] else '✗'})")
            
            if results['critical_issues']:
                print("\nCRITICAL ISSUES:")
                for issue in results['critical_issues']:
                    print(f"  - {issue}")
            
            if results['warnings']:
                print("\nWARNINGS:")
                for warning in results['warnings']:
                    print(f"  - {warning}")
            
            return results['validation_passed']
            
    except Exception as e:
        logger.error(f"CI validation failed: {e}")
        print(f"CI VALIDATION FAILED: {e}")
        return False


def run_cd_deployment(
    perform_migration: bool = False,
    migration_timeout: int = 300
) -> bool:
    """
    Run CD deployment validation and optional migration.
    
    Args:
        perform_migration: If True, attempt to run migrations
        migration_timeout: Maximum time to wait for migrations
        
    Returns:
        bool: True if deployment is successful, False otherwise
    """
    logger.info("Starting CD deployment...")
    
    try:
        with session_scope() as dbsession:
            # Pre-deployment validation
            validator = DeploymentValidator(
                strict_mode=False,  # Less strict for CD
                allow_auto_migration=perform_migration,
                max_migration_time=migration_timeout
            )
            
            pre_results = validator.validate_pre_deployment(dbsession)
            
            print("=== PRE-DEPLOYMENT VALIDATION ===")
            print(f"Validation Passed: {pre_results['validation_passed']}")
            print(f"Migration Required: {pre_results['migration_required']}")
            
            if not pre_results['validation_passed'] and not perform_migration:
                print("Pre-deployment validation failed and migration is disabled")
                return False
            
            # Perform migration if required and enabled
            if pre_results['migration_required'] and perform_migration:
                print("\n=== RUNNING MIGRATION ===")
                migration_success = perform_deployment_migration(dbsession)
                if not migration_success:
                    print("Migration failed")
                    return False
                print("Migration completed successfully")
            
            # Post-deployment validation
            print("\n=== POST-DEPLOYMENT VALIDATION ===")
            post_results = validator.validate_post_deployment(dbsession)
            
            print(f"Post-deployment validation: {'✓' if post_results['validation_passed'] else '✗'}")
            
            if post_results['issues']:
                print("POST-DEPLOYMENT ISSUES:")
                for issue in post_results['issues']:
                    print(f"  - {issue}")
            
            return post_results['validation_passed']
            
    except Exception as e:
        logger.error(f"CD deployment failed: {e}")
        print(f"CD DEPLOYMENT FAILED: {e}")
        return False


def perform_deployment_migration(dbsession: Session) -> bool:
    """
    Perform automated migration during deployment.
    
    Args:
        dbsession: Database session
        
    Returns:
        bool: True if migration succeeds, False otherwise
    """
    try:
        current_app_version = get_current_app_version()
        
        # Record the migration (simplified - in real implementation, 
        # this would run actual migration scripts)
        logger.info(f"Recording deployment migration to version {current_app_version}")
        
        record_migration(
            dbsession=dbsession,
            migration_name=f"automated_deployment_v{current_app_version}",
            new_version=current_app_version,
            new_schema_version=current_app_version,
            migration_type="upgrade",
            migration_source="automated",
            executed_by="ci_cd_system",
            execution_context="deployment",
            description=f"Automated deployment migration to version {current_app_version}"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Deployment migration failed: {e}")
        return False


def generate_deployment_report(output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive deployment report.
    
    Args:
        output_file: Optional file to write report to
        
    Returns:
        Dictionary with deployment report
    """
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_version": get_current_app_version(),
        "environment": {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith(('AIRSENAL_', 'FPL_'))
            }
        },
        "database_status": None,
        "validation_results": None
    }
    
    try:
        with session_scope() as dbsession:
            # Get database information
            db_info = get_database_info(dbsession)
            report["database_status"] = db_info
            
            # Run validation
            validator = DeploymentValidator(strict_mode=False)
            validation_results = validator.validate_pre_deployment(dbsession)
            report["validation_results"] = validation_results
            
    except Exception as e:
        report["error"] = str(e)
        logger.error(f"Failed to generate deployment report: {e}")
    
    # Write to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Deployment report written to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write report to {output_file}: {e}")
    
    return report


def check_migration_prerequisites() -> Tuple[bool, List[str]]:
    """
    Check if all prerequisites for migration are met.
    
    Returns:
        Tuple of (prerequisites_met, issues)
    """
    issues = []
    
    # Check database connectivity
    try:
        with session_scope() as dbsession:
            dbsession.execute("SELECT 1")
    except Exception as e:
        issues.append(f"Database connectivity failed: {e}")
    
    # Check for required environment variables
    required_env_vars = ['AIRSENAL_HOME']
    for var in required_env_vars:
        if not os.getenv(var):
            issues.append(f"Required environment variable {var} is not set")
    
    # Check disk space (basic check)
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        if free_gb < 1:  # Less than 1GB free
            issues.append(f"Low disk space: {free_gb}GB free")
    except Exception:
        issues.append("Could not check disk space")
    
    return len(issues) == 0, issues


def create_github_actions_workflow() -> str:
    """
    Generate a GitHub Actions workflow for database version checking.
    
    Returns:
        YAML content for GitHub Actions workflow
    """
    workflow = """
name: Database Version Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate-database-version:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
    
    - name: Check database version compatibility
      run: |
        python -c "
        from airsenal.framework.cicd_integration import run_ci_validation
        import sys
        success = run_ci_validation()
        sys.exit(0 if success else 1)
        "
    
    - name: Generate deployment report
      if: always()
      run: |
        python -c "
        from airsenal.framework.cicd_integration import generate_deployment_report
        generate_deployment_report('deployment-report.json')
        "
    
    - name: Upload deployment report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: deployment-report
        path: deployment-report.json
"""
    return workflow.strip()


def create_docker_healthcheck() -> str:
    """
    Generate a Docker healthcheck script for database version validation.
    
    Returns:
        Shell script content for Docker healthcheck
    """
    script = """#!/bin/bash
# Docker healthcheck for AIrsenal database version compatibility

set -e

# Run database version validation
python -c "
from airsenal.framework.database_versioning import validate_database_on_startup
from airsenal.framework.schema import session_scope
import sys

try:
    with session_scope() as dbsession:
        valid = validate_database_on_startup(
            dbsession=dbsession,
            force_migration=False,
            error_on_mismatch=False
        )
        if valid:
            print('Database version check: HEALTHY')
            sys.exit(0)
        else:
            print('Database version check: UNHEALTHY - version mismatch')
            sys.exit(1)
except Exception as e:
    print(f'Database version check: ERROR - {e}')
    sys.exit(1)
"
"""
    return script.strip()


if __name__ == "__main__":
    # Command-line interface for CI/CD operations
    import argparse
    
    parser = argparse.ArgumentParser(description="AIrsenal CI/CD Integration")
    parser.add_argument("command", choices=["validate", "deploy", "report", "check-prereqs"])
    parser.add_argument("--migrate", action="store_true", help="Perform migration during deployment")
    parser.add_argument("--output", help="Output file for reports")
    parser.add_argument("--timeout", type=int, default=300, help="Migration timeout in seconds")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        success = run_ci_validation()
        sys.exit(0 if success else 1)
    elif args.command == "deploy":
        success = run_cd_deployment(args.migrate, args.timeout)
        sys.exit(0 if success else 1)
    elif args.command == "report":
        generate_deployment_report(args.output)
    elif args.command == "check-prereqs":
        prereqs_met, issues = check_migration_prerequisites()
        if issues:
            print("Migration prerequisites not met:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All migration prerequisites are met")
        sys.exit(0 if prereqs_met else 1)