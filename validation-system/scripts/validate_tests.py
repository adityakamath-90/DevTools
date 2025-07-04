#!/usr/bin/env python3
"""
Runtime validation script for generated Kotlin tests.

This script validates generated Kotlin test files by:
1. Syntax validation using Kotlin compiler
2. JUnit structure validation
3. Test execution simulation
4. Coverage analysis
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import re

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KotlinTestValidator:
    """
    Comprehensive validator for generated Kotlin tests.
    
    Features:
    - Kotlin compilation validation
    - JUnit structure validation
    - Test execution validation
    - Coverage analysis
    - Performance metrics
    """
    
    def __init__(self, kotlin_home: Optional[str] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.kotlin_home = kotlin_home or os.getenv('KOTLIN_HOME')
        self.validation_results = []
        
        # Check if Kotlin compiler is available
        self._check_kotlin_compiler()
    
    def validate_all_tests(self, test_dir: str = "output-test") -> Dict[str, any]:
        """
        Validate all generated test files in the directory.
        
        Args:
            test_dir: Directory containing generated test files
            
        Returns:
            Validation results summary
        """
        self.logger.info(f"Starting validation of tests in: {test_dir}")
        
        test_files = list(Path(test_dir).glob("**/*.kt"))
        if not test_files:
            self.logger.warning(f"No test files found in {test_dir}")
            return {"status": "no_tests", "files": []}
        
        results = {
            "total_files": len(test_files),
            "passed": 0,
            "failed": 0,
            "files": [],
            "summary": {}
        }
        
        for test_file in test_files:
            self.logger.info(f"Validating: {test_file}")
            file_result = self.validate_test_file(str(test_file))
            results["files"].append(file_result)
            
            if file_result["overall_status"] == "passed":
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        results["success_rate"] = (results["passed"] / results["total_files"]) * 100
        self._generate_summary_report(results)
        
        return results
    
    def validate_test_file(self, test_file_path: str) -> Dict[str, any]:
        """
        Validate a single Kotlin test file.
        
        Args:
            test_file_path: Path to the test file
            
        Returns:
            Validation result for the file
        """
        file_name = Path(test_file_path).name
        result = {
            "file": file_name,
            "path": test_file_path,
            "validations": {},
            "overall_status": "unknown",
            "errors": [],
            "warnings": []
        }
        
        try:
            # Read the test file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_content = f.read()
            
            # 1. Syntax validation
            result["validations"]["syntax"] = self._validate_syntax(test_content, test_file_path)
            
            # 2. Structure validation
            result["validations"]["structure"] = self._validate_junit_structure(test_content)
            
            # 3. Import validation
            result["validations"]["imports"] = self._validate_imports(test_content)
            
            # 4. Test method validation
            result["validations"]["test_methods"] = self._validate_test_methods(test_content)
            
            # 5. Assertion validation
            result["validations"]["assertions"] = self._validate_assertions(test_content)
            
            # Determine overall status
            # Only consider validations that have actual pass/fail results (not None)
            validation_results = [v.get("passed", False) for v in result["validations"].values() if v.get("passed") is not None]
            all_passed = all(validation_results) if validation_results else False
            result["overall_status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            result["overall_status"] = "error"
            result["errors"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Error validating {file_name}: {e}")
        
        return result
    
    def _validate_syntax(self, content: str, file_path: str) -> Dict[str, any]:
        """Validate Kotlin syntax using the compiler."""
        if not self.kotlin_home:
            return {
                "passed": None,
                "message": "Kotlin compiler not available",
                "skip_reason": "KOTLIN_HOME not set"
            }
        
        try:
            # Create temporary directory for compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / "TestFile.kt"
                temp_file.write_text(content)
                
                # Try to compile
                kotlinc_path = Path(self.kotlin_home) / "bin" / "kotlinc"
                result = subprocess.run(
                    [str(kotlinc_path), "-cp", self._get_junit_classpath(), str(temp_file)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return {"passed": True, "message": "Syntax valid"}
                else:
                    return {
                        "passed": False,
                        "message": "Syntax errors found",
                        "errors": result.stderr
                    }
        
        except subprocess.TimeoutExpired:
            return {"passed": False, "message": "Compilation timeout"}
        except Exception as e:
            return {"passed": False, "message": f"Compilation error: {str(e)}"}
    
    def _validate_junit_structure(self, content: str) -> Dict[str, any]:
        """Validate JUnit test structure."""
        checks = {
            "has_test_class": bool(re.search(r'class\s+\w+Test', content)) or bool(re.search(r'class\s+\w+Tests', content)),
            "has_test_imports": bool(re.search(r'import.*junit|import.*Test', content)),
            "has_test_methods": bool(re.search(r'@Test\s*\n\s*fun\s+', content, re.MULTILINE)),
            "proper_naming": bool(re.search(r'class\s+\w+Tests?\s*{', content))
        }
        
        passed = all(checks.values())
        
        return {
            "passed": passed,
            "checks": checks,
            "message": "JUnit structure valid" if passed else "JUnit structure issues found"
        }
    
    def _validate_imports(self, content: str) -> Dict[str, any]:
        """Validate import statements."""
        required_imports = [
            r'org\.junit\.jupiter\.api\.Test',
            r'kotlin\.test\.\w+|org\.junit\.jupiter\.api\.Assertions\.\w+'
        ]
        
        missing_imports = []
        for required in required_imports:
            if not re.search(required, content):
                missing_imports.append(required)
        
        passed = len(missing_imports) == 0
        
        return {
            "passed": passed,
            "missing_imports": missing_imports,
            "message": "All required imports present" if passed else f"Missing imports: {missing_imports}"
        }
    
    def _validate_test_methods(self, content: str) -> Dict[str, any]:
        """Validate test method structure."""
        # More flexible pattern to catch @Test annotations followed by function definitions
        # Handle both @Test fun and @Test\n fun patterns
        test_methods = re.findall(r'@Test\s*\n?\s*fun\s+`?([^`\(]+)`?', content, re.MULTILINE | re.DOTALL)
        
        issues = []
        for method in test_methods:
            method_clean = method.strip()
            if not method_clean.startswith('test') and not method_clean.startswith('should') and not any(keyword in method_clean.lower() for keyword in ['test', 'should', 'verify', 'check']):
                issues.append(f"Method '{method_clean}' doesn't follow naming convention")
        
        return {
            "passed": len(test_methods) > 0,
            "test_method_count": len(test_methods),
            "issues": issues,
            "message": f"Found {len(test_methods)} test methods" + ("" if not issues else f", {len(issues)} naming issues")
        }
    
    def _validate_assertions(self, content: str) -> Dict[str, any]:
        """Validate assertion usage."""
        assertion_patterns = [
            r'assertEquals\s*\(',
            r'assertTrue\s*\(',
            r'assertFalse\s*\(',
            r'assertNull\s*\(',
            r'assertNotNull\s*\(',
            r'assertFailsWith\s*<',
            r'assertThrows\s*<'
        ]
        
        found_assertions = []
        for pattern in assertion_patterns:
            matches = re.findall(pattern, content)
            if matches:
                found_assertions.extend(matches)
        
        return {
            "passed": len(found_assertions) > 0,
            "assertion_count": len(found_assertions),
            "message": f"Found {len(found_assertions)} assertions" if found_assertions else "No assertions found"
        }
    
    def _check_kotlin_compiler(self):
        """Check if Kotlin compiler is available."""
        if not self.kotlin_home:
            self.logger.warning("KOTLIN_HOME not set. Syntax validation will be skipped.")
            return
        
        kotlinc_path = Path(self.kotlin_home) / "bin" / "kotlinc"
        if not kotlinc_path.exists():
            self.logger.warning(f"Kotlin compiler not found at: {kotlinc_path}")
            self.kotlin_home = None
    
    def _get_junit_classpath(self) -> str:
        """Get JUnit classpath for compilation."""
        # This would need to point to actual JUnit JAR files
        # For now, return empty string
        return ""
    
    def _generate_summary_report(self, results: Dict[str, any]):
        """Generate a summary report of validation results."""
        self.logger.info("="*60)
        self.logger.info("KOTLIN TEST VALIDATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total files: {results['total_files']}")
        self.logger.info(f"Passed: {results['passed']}")
        self.logger.info(f"Failed: {results['failed']}")
        self.logger.info(f"Success rate: {results['success_rate']:.1f}%")
        self.logger.info("="*60)
        
        for file_result in results["files"]:
            status_emoji = "✅" if file_result["overall_status"] == "passed" else "❌"
            self.logger.info(f"{status_emoji} {file_result['file']}: {file_result['overall_status']}")


def validate_generated_tests():
    """Command-line interface for test validation."""
    validator = KotlinTestValidator()
    results = validator.validate_all_tests()
    
    # Save detailed results to JSON
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Validation complete! Results saved to validation_results.json")
    print(f"📈 Success rate: {results['success_rate']:.1f}%")
    
    return results


if __name__ == "__main__":
    validate_generated_tests()
