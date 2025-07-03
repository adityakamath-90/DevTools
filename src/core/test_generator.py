"""
Main test generator orchestrator for AI-powered Kotlin test generation.

This module provides the main business logic for generating comprehensive
JUnit 5 test cases for Kotlin classes using AI and semantic similarity.
"""

import os
import re
import uuid
from typing import List, Optional, Dict, Any
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
        self.logger.info(f"Starting test generation for: {request.source_file}")
        
        try:
            # Parse the source code
            kotlin_class = request.kotlin_class
            if not kotlin_class:
                return GenerationResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    source_file=request.source_file,
                    output_file=request.output_file,
                    error_message="Failed to parse Kotlin class"
                )
            
            self.logger.info(f"Successfully parsed class: {kotlin_class.name}")
            
            # Find similar tests for context
            similar_tests = self._find_similar_tests(kotlin_class)
            self.logger.info(f"Found {len(similar_tests)} similar tests")
            
            # Generate test code
            test_code = self._generate_test_code(kotlin_class, similar_tests)
            if not test_code:
                return GenerationResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    source_file=request.source_file,
                    output_file=request.output_file,
                    error_message="Failed to generate test code"
                )
            
            # Validate and improve test code
            improved_test_code = self._validate_and_improve_test(kotlin_class, test_code)
            
            # Clean the generated code
            final_test_code = self._clean_generated_code(improved_test_code or test_code)
            
            # Save the test file
            test_file_path = self._save_test_file(kotlin_class, final_test_code)
            
            # Update statistics
            self.stats.tests_generated += 1
            
            # Create test case
            test_case = TestCase(
                id=str(uuid.uuid4()),
                class_name=kotlin_class.name,
                test_name=f"{kotlin_class.name}Test",
                test_code=final_test_code
            )
            
            return GenerationResult(
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED,
                source_file=request.source_file,
                output_file=request.output_file,
                generated_tests=[test_case],
                similar_tests_count=len(similar_tests)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating tests for {request.kotlin_class.file_path}: {e}")
            return GenerationResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                source_file=request.source_file,
                output_file=request.output_file,
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
                    source_file=file_path,
                    kotlin_class=kotlin_class,
                    output_file=output_file
                )
                
                # Generate tests
                result = self.generate_tests(request)
                results.append(result)
                
                # Update statistics
                self.stats.files_processed += 1
                if result.is_successful:
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
                    source_file=file_path,
                    kotlin_class=kotlin_class,
                    output_file=""
                )
                results.append(GenerationResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    source_file=request.source_file,
                    output_file=request.output_file,
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
            total_requests=self.stats.files_processed,
            successful_requests=self.stats.files_succeeded,
            failed_requests=self.stats.files_failed,
            average_response_time=self.stats.total_processing_time / max(1, self.stats.files_processed),
            success_rate=self.stats.success_rate(),
            total_processing_time=self.stats.total_processing_time
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
        """Generate test code using AI."""
        try:
            prompt = self.prompt_builder.build_generation_prompt(kotlin_class, similar_tests)
            
            # Generate with LLM
            result = self.llm_provider.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating test code: {e}")
            return ""
    
    def _validate_and_improve_test(self, kotlin_class: KotlinClass, test_code: str) -> Optional[str]:
        """Validate and improve the generated test code."""
        if not self.config.enable_validation:
            return None
        
        try:
            validation_prompt = self.prompt_builder.build_validation_prompt(kotlin_class, test_code)
            
            improved_code = self.llm_provider.generate(
                validation_prompt,
                temperature=0.1,  # Lower temperature for validation
                max_tokens=self.config.max_tokens
            )
            
            return improved_code
            
        except Exception as e:
            self.logger.warning(f"Error validating test code: {e}")
            return None
    
    def _clean_generated_code(self, generated_code: str) -> str:
        """Clean up the generated code by removing markdown formatting."""
        if not generated_code:
            return ""
        
        # Remove markdown code blocks
        code = generated_code.strip()
        
        # Remove kotlin markdown blocks
        if code.startswith("```kotlin"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _save_test_file(self, kotlin_class: KotlinClass, test_code: str) -> str:
        """Save the generated test code to a file."""
        test_filename = f"{kotlin_class.name}Test.kt"
        test_file_path = os.path.join(self.config.output_dir, test_filename)
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            
            # Write test file
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.logger.info(f"Generated test file: {test_file_path}")
            return test_file_path
            
        except Exception as e:
            self.logger.error(f"Error saving test file: {e}")
            raise


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
            if 'testcase--datastore' in dirs:
                dirs.remove('testcase--datastore')
            for file in files:
                if file.endswith(".kt"):
                    full_path = os.path.join(root, file)
                    self.process_file(full_path)
