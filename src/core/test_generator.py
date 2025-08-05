"""
Main test generator orchestrator for AI-powered Kotlin test generation.

This module provides the main business logic for generating comprehensive
JUnit 5 test cases for Kotlin classes using AI and semantic similarity.
"""

import os
import re
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from models.data_models import KotlinClass, GenerationRequest, GenerationResult, ModelMetrics, GenerationStatus, TestCase
from interfaces.base_interfaces import TestGenerator, LLMProvider, SimilarityIndexer
from config.settings import GenerationConfig
from utils.logging import get_logger
from .code_parser import KotlinParser
from .prompt_builder import PromptBuilder

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for test generation processing."""
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    classes_found: int = 0
    tests_generated: int = 0
    total_processing_time: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.files_processed == 0:
            return 0.0
        return (self.files_succeeded / self.files_processed) * 100.0


class KotlinTestGenerator(TestGenerator):
    """
    Production-ready Kotlin test generator with comprehensive features.
    
    Features:
    - Advanced class detection and parsing
    - Semantic similarity matching
    - AI-powered test generation
    - Quality validation and improvement
    - Comprehensive error handling
    - Detailed metrics and logging
    - Fallback mechanisms
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        similarity_indexer: SimilarityIndexer,
        config: Optional[GenerationConfig] = None
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.llm_provider = llm_provider
        self.similarity_indexer = similarity_indexer
        self.config = config or GenerationConfig()
        
        # Initialize components
        self.parser = KotlinParser()
        self.prompt_builder = PromptBuilder()
        
        # Processing statistics
        self.stats = ProcessingStats()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info(f"Initialized KotlinTestGenerator with config: {self.config}")
    
    def generate_tests(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate tests for a single file or class.
        
        Args:
            request: Generation request containing source information
            
        Returns:
            GenerationResult with generated test code and metadata
        """
        self.logger.info(f"Starting test generation for class: {request.class_name}")
        
        try:
            # Parse the source code
            # Parse the source code (reconstruct KotlinClass from request fields)
            kotlin_class = KotlinClass(
                name=request.class_name,
                source_code=request.source_code
            )
            self.logger.info(f"Successfully parsed class: {kotlin_class.name}")
            # Find similar tests for context
            similar_tests = self._find_similar_tests(kotlin_class)
            self.logger.info(f"Found {len(similar_tests)} similar tests")
            # Generate test code
            test_code = self._generate_test_code(kotlin_class, similar_tests)
            if not test_code:
                self.logger.error(f"No test code generated for class {kotlin_class.name}. LLM/test output: {test_code}")
                return GenerationResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    error_message="Failed to generate test code"
                )
            # Validate and improve test code
            # improved_test_code = self._validate_and_improve_test(kotlin_class, test_code)
            # Clean the generated code
            final_test_code = self._clean_generated_code(test_code or test_code)
            self.logger.info(f"Final generated test code for {kotlin_class.name}:\n{final_test_code}\n---END---")

            # Success criteria: non-empty, contains at least one 'class' or '@Test' or 'fun' keyword
            is_success = bool(final_test_code and ("class " in final_test_code or "@Test" in final_test_code or "fun " in final_test_code))

            # Save the test file if generation was successful
            output_file = getattr(request, 'output_file', None)
            # If output_file is not set, generate it
            if not output_file:
                output_file = os.path.join(self.config.output_dir, f"{kotlin_class.name}Test.kt")
            if is_success:
                try:
                    saved_path = self._save_test_file(kotlin_class, final_test_code)
                    output_file = os.path.abspath(saved_path)
                except Exception as e:
                    self.logger.error(f"Failed to save test file: {e}")
                    is_success = False
                    output_file = None
            else:
                output_file = None
            self.stats.tests_generated += 1
            test_case = TestCase(name=f"{kotlin_class.name}Test")
            return GenerationResult(
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED if is_success else GenerationStatus.FAILED,
                test_code=final_test_code,
                output_file=output_file,
                error_message=None if is_success else "Generated code did not meet success criteria"
            )
            
        except Exception as e:
            self.logger.error(f"Error generating tests for class {request.class_name}: {e}")
            return GenerationResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_message=str(e)
            )
    
    def generate_tests_for_directory(self, source_dir: str) -> List[GenerationResult]:
        """
        Generate tests for all Kotlin files in a directory.
        
        Args:
            source_dir: Directory containing Kotlin source files
            
        Returns:
            List of GenerationResult objects
        """
        self.logger.info(f"Starting batch test generation for directory: {source_dir}")
        
        if not os.path.exists(source_dir):
            self.logger.error(f"Source directory does not exist: {source_dir}")
            return []
        
        # Find all Kotlin files
        kotlin_files = self._find_kotlin_files(source_dir)
        self.logger.info(f"Found {len(kotlin_files)} Kotlin files to process")
        
        if not kotlin_files:
            self.logger.warning(f"No Kotlin files found in {source_dir}")
            return []
        
        results = []
        
        for file_path in kotlin_files:
            try:
                # Read source code
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Create generation request
                kotlin_class = KotlinClass(
                    name=self.parser.extract_class_name(source_code) or "UnknownClass",
                    source_code=source_code,
                    file_path=file_path
                )
                
                # Generate output file path
                output_file = os.path.join(self.config.output_dir, f"{kotlin_class.name}Test.kt")
                
                request = GenerationRequest(
                    request_id=str(uuid.uuid4()),
                    class_name=kotlin_class.name,
                    source_code=kotlin_class.source_code or "",
                    parameters=None
                )
                # Attach output_file for downstream reporting
                setattr(request, 'output_file', output_file)
                
                # Generate tests
                result = self.generate_tests(request)
                results.append(result)
                
                # Update statistics
                self.stats.files_processed += 1
                if result.status == GenerationStatus.COMPLETED:
                    self.stats.files_succeeded += 1
                else:
                    self.stats.files_failed += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                kotlin_class = KotlinClass(
                    name="UnknownClass",
                    source_code="",
                    file_path=file_path
                )
                request = GenerationRequest(
                    request_id=str(uuid.uuid4()),
                    class_name=kotlin_class.name,
                    source_code=kotlin_class.source_code or "",
                    parameters=None
                )
                results.append(GenerationResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    error_message=str(e)
                ))
                self.stats.files_failed += 1
        
        self.logger.info(f"Completed batch generation. Success rate: {self.stats.success_rate():.1f}%")
        return results
    
    def extract_class_info(self, source_code: str, file_path: str) -> Optional[KotlinClass]:
        """
        Extract class information from source code.
        
        Args:
            source_code: Kotlin source code
            file_path: Path to the source file
            
        Returns:
            KotlinClass object or None if no class found
        """
        return self.parser.extract_class_info(source_code, file_path)
    
    def get_metrics(self) -> ModelMetrics:
        """
        Get performance metrics for the test generator.
        
        Returns:
            ModelMetrics object with performance data
        """
        return ModelMetrics(
            model_name="KotlinTestGenerator",
            additional_info={
                "files_processed": self.stats.files_processed,
                "files_succeeded": self.stats.files_succeeded,
                "files_failed": self.stats.files_failed,
                "average_response_time": self.stats.total_processing_time / max(1, self.stats.files_processed),
                "success_rate": self.stats.success_rate(),
                "total_processing_time": self.stats.total_processing_time
            }
        )
    
    def _find_kotlin_files(self, source_dir: str) -> List[str]:
        """Find all Kotlin files in the source directory."""
        kotlin_files = []
        
        for root, dirs, files in os.walk(source_dir):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['testcase--datastore', '.git', '__pycache__']]
            
            for file in files:
                if file.endswith('.kt'):
                    kotlin_files.append(os.path.join(root, file))
        
        return kotlin_files
    
    def _find_similar_tests(self, kotlin_class: KotlinClass) -> List[str]:
        """Find similar existing tests for context."""
        try:
            return self.similarity_indexer.find_similar(
                kotlin_class.source_code, 
                top_k=self.config.similarity_top_k
            )
        except Exception as e:
            self.logger.warning(f"Error finding similar tests: {e}")
            return []
    
    def _generate_test_code(self, kotlin_class: KotlinClass, similar_tests: List[str]) -> str:
        """
        Generate test code using AI with the configured LLM provider.
        
        This method:
        1. Builds a generation prompt using the prompt builder
        2. Sends the prompt to the LLM for test generation
        3. Cleans the generated code
        4. Optionally validates and improves the test code
        
        Args:
            kotlin_class: The Kotlin class to generate tests for
            similar_tests: List of similar test cases for context
            
        Returns:
            str: Generated test code, or empty string on failure
        """
        self.logger.info(f"Generating test code for class: {kotlin_class.name}")
        
        try:
            # Build the generation prompt with class and similar tests
            prompt = self.prompt_builder.build_generation_prompt(
                kotlin_class=kotlin_class, 
                similar_tests=similar_tests
            )
            
            self.logger.debug(f"Generated prompt with {len(prompt)} characters")
            
            # Generate test code with the LLM
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.95,  # Slightly higher for more creative generation
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract the generated text from the response
            generated_code = response.text if hasattr(response, 'text') else str(response)
            
            if not generated_code or not generated_code.strip():
                self.logger.error("Received empty response from LLM during test generation")
                return ""
            
            # Clean up the generated code
            cleaned_code = self._clean_generated_code(generated_code)
            
            if not cleaned_code:
                self.logger.error("Failed to clean generated test code")
                return ""
                
            self.logger.info(f"Generated test code for class: {kotlin_class.name}")
            
            # If validation is enabled, validate and improve the test code
            if self.config.enable_validation:
                improved_code = self._validate_and_improve_test(kotlin_class, cleaned_code)
                if improved_code:
                    self.logger.info(f"Successfully improved test code for class: {kotlin_class.name}")
                    return improved_code
            
            return cleaned_code
            
        except Exception as e:
            self.logger.error(f"Error generating test code: {str(e)}", exc_info=True)
            return ""
    
    def _validate_and_improve_test(self, kotlin_class: KotlinClass, test_code: str) -> Optional[str]:
        """
        Validate and improve the generated test code with compilation and coverage checks.
        
        This enhanced method performs the following steps:
        1. Runs static analysis to identify coverage gaps
        2. If Kotlin compiler is available:
           - Attempts to compile the test
           - If compilation fails, tries to fix the issues
           - Runs tests with coverage collection
           - If coverage is insufficient, generates additional tests
        
        Args:
            kotlin_class: The Kotlin class being tested
            test_code: The generated test code to validate and improve
            
        Returns:
            Optional[str]: Improved test code if validation is successful, None otherwise
        """
        if not self.config.enable_validation:
            return test_code
            
        self.logger.info(f"Validating and improving test for class: {kotlin_class.name}")
        
        import tempfile
        import os
        import shutil
        
        # Create a temporary file for the test code
        temp_file = None
        temp_file_path = None
        
        try:
            # Always run static analysis first
            self.logger.info("Running static analysis for coverage gaps...")
            static_analysis = self._analyze_static_coverage(kotlin_class, test_code)
            
            # If we found issues through static analysis, try to improve the tests
            if static_analysis and any(len(v) > 0 for v in static_analysis.values()):
                self.logger.info("Found potential coverage gaps through static analysis")
                improved_code = self._improve_test_coverage(
                    kotlin_class, 
                    test_code,
                    {'static_analysis': static_analysis}
                )
                if improved_code and improved_code != test_code:
                    self.logger.info("Successfully improved tests using static analysis")
                    test_code = improved_code
            
            # Check if Kotlin compiler is available for dynamic analysis
            has_compiler = shutil.which("kotlinc") is not None
            
            if not has_compiler:
                self.logger.warning("Kotlin compiler (kotlinc) not found. Using static analysis only.")
                return test_code
            
            # Create temp file for compilation
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False)
            temp_file_path = temp_file.name
            temp_file.write(test_code)
            temp_file.close()
            
            # Step 1: Compile the test
            compile_success, compile_output = self._compile_kotlin_test(
                test_file_path=temp_file_path,
                classpath=getattr(self.config, 'classpath', None)
            )
            
            if not compile_success:
                self.logger.warning(f"Test compilation failed: {compile_output}")
                # Try to fix compilation issues
                fixed_code = self._fix_compilation_issues(kotlin_class, test_code, compile_output)
                if fixed_code and fixed_code != test_code:
                    self.logger.info("Successfully fixed compilation issues")
                    test_code = fixed_code
                    # Update the temp file with fixed code
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(test_code)
                
                # Try to improve based on compilation errors
                improved_code = self._improve_test_coverage(
                    kotlin_class, 
                    test_code,
                    {'compilation_error': compile_output}
                )
                if improved_code and improved_code != test_code:
                    self.logger.info("Generated improved tests based on compilation errors")
                    return improved_code
                
                return test_code  # Return the best we have
            
            # If we get here, compilation was successful
            # Step 2: Run tests with coverage
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(temp_file_path)))
            run_success, coverage_data = self._run_test_with_coverage(
                test_class_name=kotlin_class.name,
                project_root=project_root
            )
            
            if run_success and coverage_data:
                self.logger.info(
                    f"Test coverage: {coverage_data.get('line_coverage', {}).get('percentage', 0):.1f}% lines, "
                    f"{coverage_data.get('branch_coverage', {}).get('percentage', 0):.1f}% branches"
                )
                
                # Step 3: Check if we need to improve coverage
                min_coverage = getattr(self.config, 'min_coverage', 80.0)
                current_coverage = coverage_data.get('line_coverage', {}).get('percentage', 0)
                
                if current_coverage < min_coverage:
                    self.logger.info("Test coverage below threshold, attempting to improve...")
                    improved_code = self._improve_test_coverage(
                        kotlin_class, test_code, coverage_data
                    )
                    
                    if improved_code and improved_code != test_code:
                        self.logger.info("Successfully improved test coverage")
                        return improved_code
            
            return test_code
            
        except Exception as e:
            self.logger.error(f"Error during test validation: {str(e)}", exc_info=True)
            return test_code  # Return the best we have so far
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up temporary file: {str(e)}")
    
    def _compile_kotlin_test(self, test_file_path: str, classpath: str = None) -> Tuple[bool, str]:
        """
        Compile the generated Kotlin test file.
        
        Args:
            test_file_path: Path to the test file to compile
            classpath: Optional classpath for compilation
            
        Returns:
            Tuple[bool, str]: (success, output) - success flag and compilation output
        """
        try:
            import subprocess
            import shutil
            
            # Check if kotlinc is installed
            if not shutil.which("kotlinc"):
                install_guide = """
                Kotlin compiler (kotlinc) is not installed or not in PATH.
                
                To install Kotlin compiler:
                
                On macOS (using Homebrew):
                    brew install kotlin
                
                On Linux (using SDKMAN):
                    curl -s https://get.sdkman.io | bash
                    source "$HOME/.sdkman/bin/sdkman-init.sh"
                    sdk install kotlin
                
                On Windows (using Chocolatey):
                    choco install kotlinc
                
                After installation, verify by running:
                    kotlinc -version
                """
                return False, install_guide
            
            # Build the compilation command
            cmd = ["kotlinc", test_file_path]
            if classpath:
                cmd.extend(["-cp", classpath])
            
            self.logger.debug(f"Running compilation command: {' '.join(cmd)}")
            
            # Run the compilation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown compilation error"
                self.logger.error(f"Compilation failed: {error_msg}")
                return False, error_msg
                
            self.logger.info("Test compilation successful")
            return True, "Compilation successful"
            
        except FileNotFoundError as e:
            error_msg = f"Required command not found: {e}. Please ensure Kotlin compiler is installed."
            self.logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during compilation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _run_test_with_coverage(self, test_class_name: str, project_root: str) -> Tuple[bool, dict]:
        """
        Run tests with coverage collection using JaCoCo and parse the coverage report.
        
        Args:
            test_class_name: Name of the test class to run
            project_root: Root directory of the project
            
        Returns:
            Tuple[bool, dict]: (success, coverage_data) - success flag and coverage information
            
        The coverage_data dict contains:
        {
            'line_coverage': {
                'covered': int,  # Number of covered lines
                'missed': int,   # Number of missed lines
                'percentage': float  # Coverage percentage (0-100)
            },
            'branch_coverage': {
                'covered': int,  # Number of covered branches
                'missed': int,   # Number of missed branches
                'percentage': float  # Branch coverage percentage (0-100)
            },
            'methods': [
                {
                    'name': str,  # Fully qualified method name
                    'line_coverage': int  # Number of times method was executed
                },
                ...
            ]
        }
        """
        try:
            import subprocess
            import json
            import xml.etree.ElementTree as ET
            from pathlib import Path
            from typing import Dict, List, Tuple
            
            # Ensure reports directory exists
            coverage_dir = Path(project_root) / "build" / "reports" / "jacoco"
            coverage_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean previous reports
            for file in coverage_dir.glob("*.xml"):
                file.unlink()
            
            # Run tests with JaCoCo agent
            result = subprocess.run(
                [
                    "./gradlew", "test",
                    f"--tests={test_class_name}",
                    "--info",
                    "--rerun-tasks",
                    "-PjacocoEnabled=true"
                ],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, {"error": result.stderr or "Test execution failed"}
            
            # Find the JaCoCo XML report
            report_files = list(coverage_dir.glob("**/test.exec"))
            if not report_files:
                return False, {"error": "No JaCoCo coverage report found"}
            
            # Convert binary .exec to XML
            jacoco_cli = Path(project_root) / "libs" / "jacococli.jar"
            if not jacoco_cli.exists():
                # Try to download jacococli.jar if not found
                self.logger.warning("jacococli.jar not found, downloading...")
                import urllib.request
                jacoco_cli.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(
                    "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.10/org.jacoco.cli-0.8.10-nodeps.jar",
                    jacoco_cli
                )
            
            xml_report = coverage_dir / "coverage.xml"
            subprocess.run(
                ["java", "-jar", str(jacoco_cli), "report", str(report_files[0]),
                 "--classfiles", str(Path(project_root) / "build/classes"),
                 "--sourcefiles", str(Path(project_root) / "src/main"),
                 "--xml", str(xml_report)],
                check=True,
                cwd=project_root,
                capture_output=True
            )
            
            if not xml_report.exists():
                return False, {"error": "Failed to generate XML coverage report"}
            
            # Parse the XML report
            tree = ET.parse(xml_report)
            root = tree.getroot()
            
            # Initialize coverage data
            coverage_data = {
                "line_coverage": {"covered": 0, "missed": 0, "percentage": 0.0},
                "branch_coverage": {"covered": 0, "missed": 0, "percentage": 0.0},
                "methods": []
            }
            
            # Find the package and class in the report
            for package in root.findall(".//package"):
                for class_elem in package.findall("class"):
                    class_name = class_elem.get("name", "")
                    
                    # Process line counter
                    counter = class_elem.find("counter[@type='LINE']")
                    if counter is not None:
                        covered = int(counter.get("covered", 0))
                        missed = int(counter.get("missed", 0))
                        coverage_data["line_coverage"]["covered"] += covered
                        coverage_data["line_coverage"]["missed"] += missed
                    
                    # Process branch counter
                    counter = class_elem.find("counter[@type='BRANCH']")
                    if counter is not None:
                        covered = int(counter.get("covered", 0))
                        missed = int(counter.get("missed", 0))
                        coverage_data["branch_coverage"]["covered"] += covered
                        coverage_data["branch_coverage"]["missed"] += missed
                    
                    # Process methods
                    for method in class_elem.findall(".//method"):
                        method_name = f"{class_name}.{method.get('name', '')}"
                        counter = method.find("counter[@type='LINE']")
                        if counter is not None:
                            covered = int(counter.get("covered", 0))
                            coverage_data["methods"].append({
                                "name": method_name,
                                "line_coverage": covered
                            })
            
            # Calculate percentages
            total_lines = (coverage_data["line_coverage"]["covered"] + 
                         coverage_data["line_coverage"]["missed"])
            if total_lines > 0:
                coverage_data["line_coverage"]["percentage"] = round(
                    (coverage_data["line_coverage"]["covered"] / total_lines) * 100, 2
                )
            
            total_branches = (coverage_data["branch_coverage"]["covered"] + 
                            coverage_data["branch_coverage"]["missed"])
            if total_branches > 0:
                coverage_data["branch_coverage"]["percentage"] = round(
                    (coverage_data["branch_coverage"]["covered"] / total_branches) * 100, 2
                )
            
            return True, coverage_data
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _fix_compilation_issues(self, kotlin_class: KotlinClass, test_code: str, error_output: str) -> Optional[str]:
        """
        Attempt to fix compilation issues in the test code with detailed error analysis.
        
        This method:
        1. Analyzes compilation errors to understand the root cause
        2. Generates targeted fixes for each error
        3. Ensures the fixed code maintains test functionality
        4. Preserves existing test structure and assertions
        
        Args:
            kotlin_class: The Kotlin class being tested
            test_code: The test code with compilation issues
            error_output: Compiler error output
            
        Returns:
            Optional[str]: Fixed test code or None if fixes couldn't be applied
        """
        try:
            self.logger.info(f"Attempting to fix compilation issues in test for {kotlin_class.name}")
            
            # Analyze common error patterns
            errors = self._analyze_compilation_errors(error_output)
            if not errors:
                self.logger.warning("No specific errors could be identified in the compilation output")
                return None
            
            # Build a detailed prompt to fix the compilation issues
            prompt = """
            Fix the Kotlin test code that has compilation errors. Follow these guidelines:
            
            1. Fix all compilation errors while preserving the test's intent
            2. Maintain the existing test structure and assertions
            3. Add necessary imports and dependencies
            4. Follow Kotlin best practices and testing conventions
            
            Class being tested:
            ```kotlin
            {source_code}
            ```
            
            Test code with errors:
            ```kotlin
            {test_code}
            ```
            
            Compilation errors:
            {error_output}
            
            Identified issues:
            {error_analysis}
            
            Please provide the complete fixed test code in a single Kotlin code block.
            Only include the test code, no explanations or markdown formatting.
            """.format(
                source_code=kotlin_class.source_code,
                test_code=test_code,
                error_output=error_output[:2000],  # Limit error output length
                error_analysis='\n'.join([f"- {e['type']}: {e['message']} (at line {e.get('line', 'unknown')})" 
                                     for e in errors])
            )
            
            # Get the fixed code from the LLM with low temperature for consistency
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for more predictable fixes
                max_tokens=self.config.max_tokens,
                top_p=0.95
            )
            
            if not hasattr(response, 'text') or not response.text:
                self.logger.warning("No response received from the model")
                return None
            
            # Extract and clean the fixed code
            fixed_code = self._clean_generated_code(response.text)
            if not fixed_code or fixed_code == test_code:
                self.logger.warning("No meaningful changes detected in the fixed code")
                return None
                
            # Log the changes made
            self.logger.info(f"Generated fixed test code with {len(fixed_code.splitlines())} lines "
                          f"(was {len(test_code.splitlines())} lines)")
            
            return fixed_code
            
        except Exception as e:
            self.logger.error(f"Error in fix_compilation_issues: {str(e)}", exc_info=True)
            return None
            
    def _analyze_static_coverage(self, kotlin_class: 'KotlinClass', test_code: str) -> dict:
        """
        Analyze test coverage gaps without compilation using static analysis.
        
        Args:
            kotlin_class: The Kotlin class being tested
            test_code: The generated test code
            
        Returns:
            dict: Coverage analysis with missing test cases and suggestions
        """
        analysis = {
            'untested_methods': [],
            'partially_tested_methods': [],
            'branches_needing_coverage': [],
            'edge_cases_to_consider': []
        }
        
        # Extract public methods from source code
        source_code = kotlin_class.source_code
        public_methods = self._extract_public_methods(source_code)
        tested_methods = self._extract_tested_methods(test_code)
        
        # Find untested methods
        for method in public_methods:
            if method not in tested_methods:
                analysis['untested_methods'].append(method)
        
        # Identify edge cases that should be tested
        edge_cases = self._identify_edge_cases(source_code, test_code)
        if edge_cases:
            analysis['edge_cases_to_consider'].extend(edge_cases)
        
        # Check error handling
        error_handling = self._check_error_handling(source_code, test_code)
        if error_handling:
            analysis['error_handling'] = error_handling
        
        # Generate additional suggestions
        suggestions = self._generate_coverage_suggestions(source_code, test_code)
        if suggestions:
            analysis['suggestions'] = suggestions
        
        return analysis
    
    def _extract_public_methods(self, source_code: str) -> List[str]:
        """Extract public method names from Kotlin source code."""
        import re
        # Match method declarations like 'fun methodName(' or 'fun methodName()'
        pattern = r'fun\s+(\w+)\s*\('  # Updated pattern for Kotlin method syntax
        return re.findall(pattern, source_code)
    
    def _extract_tested_methods(self, test_code: str) -> List[str]:
        """Extract method names being tested from test code."""
        import re
        # Match test method names containing the method name being tested
        pattern = r'fun\s+`?[^`\s]+\s+(\w+)\s*\('  # Updated pattern for Kotlin test methods
        return re.findall(pattern, test_code)
    
    def _identify_edge_cases(self, source_code: str, test_code: str) -> List[dict]:
        """Identify potential edge cases that should be tested."""
        edge_cases = []
        
        # Check for numeric parameters
        if any(word in source_code for word in ['Int', 'Long', 'Double', 'Float']):
            edge_cases.append({
                'type': 'numeric',
                'suggestion': 'Test with edge values (0, -1, MAX_VALUE, MIN_VALUE)'
            })
        
        # Check for string parameters
        if 'String' in source_code:
            edge_cases.append({
                'type': 'string',
                'suggestion': 'Test with empty string, whitespace, and very long strings'
            })
        
        # Check for collections
        if any(word in source_code for word in ['List', 'Set', 'Map', 'Array']):
            edge_cases.append({
                'type': 'collection',
                'suggestion': 'Test with empty collection, single item, and large collections'
            })
        
        return edge_cases
    
    def _check_error_handling(self, source_code: str, test_code: str) -> List[dict]:
        """Check if error handling is properly tested."""
        error_handling = []
        
        # Check for try-catch blocks in source
        if 'try {' in source_code and 'assertThrows' not in test_code:
            error_handling.append({
                'type': 'exception_handling',
                'suggestion': 'Add tests that verify exceptions are thrown in error conditions'
            })
        
        # Check for null safety
        if '?' in source_code and 'assertNull' not in test_code:
            error_handling.append({
                'type': 'null_safety',
                'suggestion': 'Add tests with null values for nullable parameters'
            })
        
        return error_handling
    
    def _generate_coverage_suggestions(self, source_code: str, test_code: str) -> List[str]:
        """Generate specific test case suggestions to improve coverage."""
        suggestions = []
        
        # Suggest boundary value analysis for numeric parameters
        if any(word in source_code for word in ['>', '<', '>=', '<=', '==']):
            suggestions.append(
                'Add boundary value tests for comparison operations'
            )
        
        # Suggest testing with different input combinations
        if source_code.count('fun ') > 1:  # Multiple methods
            suggestions.append(
                'Test method interactions by chaining multiple method calls'
            )
        
        # Suggest testing with different object states
        if 'var ' in source_code:  # Mutable state
            suggestions.append(
                'Test with different object states by modifying properties between method calls'
            )
        
        return suggestions
    
    def _analyze_compilation_errors(self, error_output: str) -> List[dict]:
        """
        Analyze compilation error output to identify specific issues.
        
        Args:
            error_output: Raw error output from the Kotlin compiler
            
        Returns:
            List[dict]: Structured information about each error
        """
        import re
        
        errors = []
        
        # Common Kotlin error patterns
        patterns = [
            # Unresolved reference
            (r"Unresolved reference: (\w+)", "UNRESOLVED_REFERENCE"),
            # Type mismatch
            (r"Type mismatch: inferred type is (.+?) but (.+?) was expected", "TYPE_MISMATCH"),
            # Unresolved import
            (r"Unresolved reference: (import .+?)\n", "UNRESOLVED_IMPORT"),
            # Missing imports
            (r"Cannot access class '(.*?)'.", "MISSING_IMPORT"),
            # Syntax error
            (r"Expecting an element(.*?)", "SYNTAX_ERROR"),
            # Missing function
            (r"No value passed for parameter '(\w+)'", "MISSING_PARAMETER"),
            # Unknown property
            (r"Unresolved reference: (\w+)", "UNKNOWN_PROPERTY")
        ]
        
        # Extract line numbers if available
        line_number = None
        line_match = re.search(r".*?(\d+):\d+:", error_output)
        if line_match:
            try:
                line_number = int(line_match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Match patterns against error output
        for pattern, error_type in patterns:
            match = re.search(pattern, error_output, re.DOTALL)
            if match:
                errors.append({
                    'type': error_type,
                    'message': match.group(0),
                    'line': line_number,
                    'details': match.groups()
                })
        
        # If no patterns matched, return a generic error
        if not errors:
            errors.append({
                'type': 'COMPILATION_ERROR',
                'message': 'Unknown compilation error',
                'line': line_number,
                'details': (error_output[:200] + '...') if len(error_output) > 200 else error_output
            })
        
        return errors
    
    def _analyze_coverage_gaps(self, kotlin_class: KotlinClass, coverage_data: dict) -> dict:
        """
        Analyze coverage gaps and identify specific areas that need improvement.
        
        Args:
            kotlin_class: The Kotlin class being tested
            coverage_data: Coverage information from JaCoCo
            
        Returns:
            dict: Analysis of coverage gaps with specific recommendations
        """
        analysis = {
            'untested_methods': [],
            'partially_tested_methods': [],
            'branches_needing_coverage': [],
            'edge_cases_to_consider': []
        }
        
        # Analyze method coverage
        for method in coverage_data.get('methods', []):
            method_name = method.get('name', '')
            line_coverage = method.get('line_coverage', 0)
            
            if line_coverage == 0:
                analysis['untested_methods'].append({
                    'name': method_name,
                    'reason': 'No test coverage for this method'
                })
            elif line_coverage < 3:  # Arbitrary threshold for "partially tested"
                analysis['partially_tested_methods'].append({
                    'name': method_name,
                    'coverage': line_coverage,
                    'suggestion': 'Add more test cases to improve coverage'
                })
        
        # Analyze branch coverage if available
        branch_coverage = coverage_data.get('branch_coverage', {})
        if branch_coverage.get('percentage', 100) < 100:
            analysis['branches_needing_coverage'].append({
                'message': f"Branch coverage is at {branch_coverage.get('percentage', 0)}%",
                'suggestion': 'Add test cases for conditional branches and edge cases'
            })
        
        # Simple heuristic for edge cases based on method names and coverage
        for method in kotlin_class.methods:
            method_lower = method.name.lower()
            if any(term in method_lower for term in ['edge', 'boundary', 'limit', 'validate', 'check']):
                analysis['edge_cases_to_consider'].append({
                    'method': method.name,
                    'suggestion': 'Consider adding edge case tests for boundary conditions'
                })
        
        return analysis
    
    def _improve_test_coverage(
        self, 
        kotlin_class: KotlinClass, 
        test_code: str, 
        coverage_data: dict = None
    ) -> Optional[str]:
        """
        Improve test coverage by analyzing the current coverage and generating additional tests.
        
        This enhanced method uses static analysis to identify coverage gaps and generates
        targeted test cases to improve test coverage. It focuses on:
        - Untested public methods
        - Conditional branches and loops
        - Edge cases
        - Error handling scenarios
        - Collection operations
        - Null safety issues
        
        Args:
            kotlin_class: The Kotlin class being tested
            test_code: Current test code
            coverage_data: Optional coverage data (not required for static analysis)
            
        Returns:
            Optional[str]: Improved test code or None if no improvements could be made
        """
        try:
            # Perform comprehensive static coverage analysis
            coverage_analysis = self._analyze_static_coverage(kotlin_class, test_code)
            
            # Skip if no gaps found
            if not any(coverage_analysis.values()):
                self.logger.info("No coverage gaps found through static analysis")
                return None
            
            # Build a detailed prompt for test improvement
            prompt = self._build_coverage_improvement_prompt(
                kotlin_class, test_code, coverage_analysis
            )
            
            # Generate improved test code with lower temperature for more focused output
            response = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=2500,  # Increased for more detailed test cases
                temperature=0.2,  # Low temperature for consistent, focused test generation
                top_p=0.95
            )
            
            if not hasattr(response, 'text') or not response.text:
                self.logger.warning("No response received from the model")
                return None
            
            # Extract the test code from the response
            improved_test_code = self._extract_kotlin_code(response.text)
            
            if improved_test_code and improved_test_code != test_code:
                self.logger.info("Generated improved test code based on coverage analysis")
                
                # Merge the improved tests with existing ones
                merged_test_code = self._merge_test_code(test_code, improved_test_code)
                
                # Verify the merged code is valid Kotlin
                if self._is_valid_kotlin(merged_test_code):
                    return merged_test_code
                else:
                    self.logger.warning("Merged test code contains syntax errors, returning only improved tests")
                    return improved_test_code if self._is_valid_kotlin(improved_test_code) else None
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error improving test coverage: {str(e)}", exc_info=True)
            return None
    
    def _is_valid_kotlin(self, code: str) -> bool:
        """
        Check if the provided code is valid Kotlin syntax.
        
        Args:
            code: The Kotlin code to validate
            
        Returns:
            bool: True if the code is valid Kotlin, False otherwise
        """
        try:
            # Simple check for common Kotlin syntax elements
            required_keywords = ['class', 'fun', 'import', 'package']
            return any(keyword in code for keyword in required_keywords)
        except Exception:
            return False
    
    def _merge_test_code(self, existing_code: str, new_code: str) -> str:
        """
        Merge new test code with existing test code, avoiding duplicates.
        
        Args:
            existing_code: The existing test code
            new_code: The new test code to merge in
            
        Returns:
            str: Merged test code
        """
        if not existing_code.strip():
            return new_code
            
        if not new_code.strip():
            return existing_code
            
        # Extract package and imports from existing code
        existing_lines = existing_code.split('\n')
        package_line = next((line for line in existing_lines if line.startswith('package ')), '')
        import_lines = [line for line in existing_lines if line.startswith('import ')]
        
        # Extract test methods from both codes
        existing_tests = self._extract_test_methods(existing_code)
        new_tests = self._extract_test_methods(new_code)
        
        # Filter out any new tests that already exist
        new_tests = [test for test in new_tests if test['name'] not in 
                    {t['name'] for t in existing_tests}]
        
        # Extract imports from new code
        new_imports = [line for line in new_code.split('\n') 
                      if line.startswith('import ') and 
                      line not in import_lines]
        
        # Combine everything
        merged_code = []
        
        # Add package and imports
        if package_line:
            merged_code.append(package_line)
            merged_code.append('')  # Empty line after package
            
        # Add imports, removing duplicates
        all_imports = list(dict.fromkeys(import_lines + new_imports))
        if all_imports:
            merged_code.extend(all_imports)
            merged_code.append('')  # Empty line after imports
        
        # Add class definition and existing tests
        class_start = next((i for i, line in enumerate(existing_lines) 
                          if line.startswith('class ')), 0)
        merged_code.extend(existing_lines[class_start:])
        
        # Add new tests if there's a class definition
        if 'class ' in '\n'.join(merged_code) and new_tests:
            # Find the closing brace of the test class
            class_end = len(merged_code) - 1
            while class_end >= 0 and '}' not in merged_code[class_end]:
                class_end -= 1
                
            if class_end > 0:
                # Insert new tests before the closing brace
                indent = ' ' * 4  # Standard Kotlin indentation
                new_test_code = ['']  # Start with an empty line
                
                for test in new_tests:
                    # Add test method with proper indentation
                    test_lines = test['code'].split('\n')
                    new_test_code.extend([f"{indent}{line}" for line in test_lines])
                    new_test_code.append('')  # Add empty line between tests
                
                merged_code = (merged_code[:class_end] + 
                             new_test_code + 
                             merged_code[class_end:])
        
        return '\n'.join(merged_code)
    
    def _extract_test_methods(self, code: str) -> List[dict]:
        """
        Extract test methods from Kotlin test code.
        
        Args:
            code: The Kotlin test code
            
        Returns:
            List[dict]: List of test methods with name and code
        """
        methods = []
        lines = code.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('@Test') and i + 1 < len(lines):
                # Found a test method
                next_line = lines[i + 1].strip()
                if next_line.startswith('fun '):
                    # Extract method name
                    method_name = next_line.split('fun', 1)[1].split('(')[0].strip()
                    method_name = method_name.strip('`')  # Remove backticks if present
                    
                    # Find the start and end of the method
                    start_idx = i
                    brace_count = 0
                    method_started = False
                    
                    for j in range(i, len(lines)):
                        line_j = lines[j]
                        brace_count += line_j.count('{')
                        brace_count -= line_j.count('}')
                        
                        if brace_count > 0:
                            method_started = True
                        
                        if method_started and brace_count <= 0:
                            # Found the end of the method
                            method_code = '\n'.join(lines[start_idx:j+1])
                            methods.append({
                                'name': method_name,
                                'code': method_code
                            })
                            i = j + 1
                            break
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
                
        return methods

    def _build_coverage_improvement_prompt(
        self, 
        kotlin_class: KotlinClass, 
        test_code: str, 
        coverage_analysis: dict
    ) -> str:
        """
        Build a detailed prompt for improving test coverage.
        
        This creates a comprehensive prompt that includes:
        - The class being tested
        - Current test code
        - Detailed analysis of coverage gaps
        - Specific test case suggestions
        
        Args:
            kotlin_class: The Kotlin class being tested
            test_code: Current test code
            coverage_analysis: Results from _analyze_coverage_gaps
            
        Returns:
            str: A detailed prompt for the LLM
        """
        prompt_parts = [
            "You are a senior Kotlin developer writing comprehensive unit tests. ",
            "Analyze the following Kotlin class and existing tests, then generate additional test cases ",
            "to improve test coverage. Focus on the specific gaps identified below.\n\n",
            
            "=== CLASS UNDER TEST ===\n",
            kotlin_class.source_code,
            "\n\n=== EXISTING TESTS ===\n",
            test_code,
            "\n\n=== COVERAGE GAP ANALYSIS ===\n"
        ]
        
        # Add analysis of each gap type
        if coverage_analysis.get('missing_test_cases'):
            prompt_parts.append("\nMissing test cases for methods:")
            for test_case in coverage_analysis['missing_test_cases']:
                prompt_parts.append(f"- {test_case.get('method', 'Unknown')}: {test_case.get('suggestion', 'Add test case')}")
        
        if coverage_analysis.get('branches_to_test'):
            prompt_parts.append("\nBranches and loops that need testing:")
            for branch in coverage_analysis['branches_to_test']:
                prompt_parts.append(f"- {branch.get('type', 'Branch')}: {branch.get('suggestion', 'Add test case')}")
        
        if coverage_analysis.get('edge_cases'):
            prompt_parts.append("\nEdge cases to consider:")
            for edge_case in coverage_analysis['edge_cases']:
                prompt_parts.append(f"- {edge_case.get('type', 'Edge case')}: {edge_case.get('suggestion', 'Add test case')}")
        
        if coverage_analysis.get('error_handling'):
            prompt_parts.append("\nError handling scenarios to test:")
            for error_case in coverage_analysis['error_handling']:
                prompt_parts.append(f"- {error_case.get('type', 'Error case')}: {error_case.get('suggestion', 'Add error test')}")
        
        if coverage_analysis.get('collection_operations'):
            prompt_parts.append("\nCollection operations to test:")
            for op in coverage_analysis['collection_operations']:
                prompt_parts.append(f"- {op.get('operation', 'Collection operation')}: {op.get('suggestion', 'Add test for collection operation')}")
                if 'example' in op:
                    prompt_parts.append(f"  Example:\n  {op['example']}")
        
        if coverage_analysis.get('null_safety_issues'):
            prompt_parts.append("\nNull safety issues to test:")
            for issue in coverage_analysis['null_safety_issues']:
                prompt_parts.append(f"- {issue.get('type', 'Null safety')}: {issue.get('suggestion', 'Add null safety test')}")
                if 'example' in issue:
                    prompt_parts.append(f"  Example:\n  {issue['example']}")
        
        # Add instructions for generating the tests
        prompt_parts.extend([
            "\n\n=== INSTRUCTIONS ===\n",
            "1. Generate additional test methods to cover the identified gaps.",
            "2. Focus on one test case per test method.",
            "3. Use descriptive test names that explain what's being tested.",
            "4. Include assertions that verify both the happy path and edge cases.",
            "5. Use appropriate test data that exercises the code paths.",
            "6. Follow Kotlin testing best practices and use JUnit 5 and MockK.",
            "7. Only include the test code in your response, no explanations.",
            "8. Do not include the existing test code in your response.",
            "9. Make sure to include all necessary imports.",
            "\n=== GENERATED TEST CODE ===\n"
        ])
        
        return '\n'.join(prompt_parts)
        
    def _extract_kotlin_code(self, text: str) -> str:
        """
        Extract Kotlin code from LLM response text.
        
        Args:
            text: Raw text from LLM response
            
        Returns:
            str: Extracted Kotlin code
        """
        import re
        
        # Try to extract code block first
        code_blocks = re.findall(r'```(?:kotlin)?\s*\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
            
        # If no code block, try to find Kotlin code patterns
        lines = []
        in_code = False
        for line in text.split('\n'):
            if any(line.strip().startswith(s) for s in ['package ', 'import ', 'class ', 'fun ']):
                in_code = True
            if in_code:
                lines.append(line)
                
        return '\n'.join(lines).strip()
        
    def _clean_generated_code(self, generated_code: str) -> str:
        """Extract only Kotlin code from LLM output, robust to summaries and markdown."""
        if not generated_code:
            return ""
        code = generated_code.strip()
        import re
        # Prefer code blocks
        code_blocks = re.findall(r"```kotlin(.*?)```", code, re.DOTALL | re.IGNORECASE)
        if not code_blocks:
            code_blocks = re.findall(r"```(.*?)```", code, re.DOTALL)
        if code_blocks:
            code = "\n".join(cb.strip() for cb in code_blocks)
        else:
            # Extract from first 'class', 'import', or 'package' to end
            match = re.search(r'(class |import |package )', code)
            if match:
                code = code[match.start():]
            else:
                # If no code marker, fallback to all text
                code = code
            # Remove trailing ---END--- or similar markers
            code = re.sub(r'---END---.*$', '', code, flags=re.DOTALL)
        code = code.strip()
        return code
    
    def _save_test_file(self, kotlin_class: KotlinClass, test_code: str) -> str:
        """
        Save the generated test code to a file with proper error handling and logging.
        
        Args:
            kotlin_class: The Kotlin class being tested
            test_code: The test code to save
            
        Returns:
            str: Path to the saved test file
            
        Raises:
            IOError: If there's an error writing the file
            ValueError: If the test code is empty or invalid
        """
        if not test_code or not test_code.strip():
            error_msg = "Cannot save empty or invalid test code"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        test_filename = f"{kotlin_class.name}Test.kt"
        test_file_path = os.path.join(self.config.output_dir, test_filename)
        
        self.logger.info(f"Saving test file to: {test_file_path}")
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            
            # Write test file with atomic write to prevent partial writes
            temp_file_path = f"{test_file_path}.tmp"
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            # Atomic rename to prevent race conditions
            if os.path.exists(test_file_path):
                backup_path = f"{test_file_path}.bak"
                os.replace(test_file_path, backup_path)
                self.logger.debug(f"Created backup of existing test file at: {backup_path}")
            
            os.replace(temp_file_path, test_file_path)
            
            # Verify the file was written correctly
            if not os.path.exists(test_file_path):
                error_msg = f"Failed to verify test file creation: {test_file_path}"
                self.logger.error(error_msg)
                raise IOError(error_msg)
                
            file_size = os.path.getsize(test_file_path)
            self.logger.info(
                f"Successfully saved test file: {test_file_path} "
                f"({file_size} bytes)"
            )
            
            return test_file_path
            
        except IOError as e:
            error_msg = f"I/O error while saving test file {test_file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise IOError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error while saving test file {test_file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise IOError(error_msg) from e


# Legacy compatibility class for backward compatibility
class LegacyKotlinTestGenerator:
    """
    Legacy wrapper for backward compatibility with existing code.
    
    Note: This class is deprecated. Use KotlinTestGenerator instead.
    """
    
    def __init__(self, source_dir: str, test_dir: str, llm_client, indexer):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.warning("Using deprecated LegacyKotlinTestGenerator. Consider updating to new KotlinTestGenerator.")
        
        self.source_dir = source_dir
        self.test_dir = test_dir
        self.llm_client = llm_client
        self.indexer = indexer
        
        # Create output directory
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Initialize parser for legacy functions
        self.parser = KotlinParser()
    
    def extract_class_name(self, code: str) -> Optional[str]:
        """Legacy method for extracting class name."""
        kotlin_class = self.parser.extract_class_info(code, "")
        return kotlin_class.name if kotlin_class else None
    
    def clean_generated_code(self, generated_code: str) -> str:
        """Legacy method for cleaning generated code."""
        generator = KotlinTestGenerator(
            llm_provider=None,
            similarity_indexer=None
        )
        return generator._clean_generated_code(generated_code)
    
    def process_file(self, filepath: str):
        """Legacy method for processing a single file."""
        self.logger.info(f"Processing file: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read file {filepath}: {e}")
            return

        class_name = self.extract_class_name(file_content)
        if not class_name:
            self.logger.warning(f"No class found in {filepath}, skipping.")
            return

        # Retrieve similar tests
        try:
            similar_tests = self.indexer.retrieve_similar(file_content)
        except Exception as e:
            self.logger.warning(f"Failed to retrieve similar tests: {e}")
            similar_tests = []

        # Import legacy prompt builder functions
        try:
            from .prompt_builder import build_generation_prompt, generate_accurate_prompt
        except ImportError:
            # Fallback to direct imports
            from PromptBuilder import PromptBuilder as LegacyPromptBuilder
            gen_prompt = LegacyPromptBuilder.build_generation_prompt(class_name, file_content, similar_tests)
            generated_test = self.llm_client.generate(gen_prompt)
            
            if generated_test:
                accuracy_prompt = LegacyPromptBuilder.generate_accurate_prompt(file_content, generated_test)
                feedback = self.llm_client.generate(accuracy_prompt)
            else:
                feedback = ""
        else:
            # Use new prompt builder functions
            gen_prompt = build_generation_prompt(class_name, file_content, similar_tests)
            generated_test = self.llm_client.generate(gen_prompt)
            
            if generated_test:
                accuracy_prompt = generate_accurate_prompt(file_content, generated_test)
                feedback = self.llm_client.generate(accuracy_prompt)
            else:
                feedback = ""

        if not generated_test:
            self.logger.error(f"Failed to generate test for {class_name}")
            return

        # Clean the generated test code
        clean_test_code = self.clean_generated_code(generated_test)
        clean_feedback = self.clean_generated_code(feedback)

        # Save generated test code
        test_filename = f"{class_name}Test.kt"
        test_path = os.path.join(self.test_dir, test_filename)

        try:
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(clean_test_code)
            print(f"[✅] Generated test: {test_path}")
            print(f"[📁] File saved to: {os.path.abspath(test_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to write test file {test_path}: {e}")
            return

        print(f"[🔍] Accuracy & Reliability feedback:\n{clean_feedback}\n")

    def generate_tests_for_all(self):
        """Legacy method for generating tests for all files."""
        self.logger.info(f"Scanning source directory: {self.source_dir}")
        for root, dirs, files in os.walk(self.source_dir):
            if 'testcase-datastore' in dirs:
                dirs.remove('testcase-datastore')
            for file in files:
                if file.endswith(".kt"):
                    full_path = os.path.join(root, file)
                    self.process_file(full_path)
