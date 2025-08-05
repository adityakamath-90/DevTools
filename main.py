#!/usr/bin/env python3
"""
Main entry point for the AI-powered Kotlin test generation system.

This script demonstrates the new modular architecture and provides both
new and legacy interfaces for generating Kotlin tests and KDoc comments.
"""

import os
os.environ["ALLOW_MODEL_DOWNLOAD"] = "true"
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import GenerationConfig, LLMConfig, EmbeddingConfig
from src.services.llm_service import LLMService
from src.services.embedding_service import EmbeddingIndexerService, SimpleEmbeddingIndexerService
from src.services.kdoc_service import KDocService
from src.core.test_generator import KotlinTestGenerator
from src.utils.logging import get_logger
from src.models.data_models import GenerationStatus

logger = get_logger(__name__)


class GenAIApplication:
    """
    Main application class for the AI-powered Kotlin code generation system.
    
    This class demonstrates the new modular architecture and provides
    a clean interface for generating tests and documentation.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize services
        self.llm_service = None
        self.embedding_service = None
        self.kdoc_service = None
        self.test_generator = None
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services."""
        try:
            self.logger.info("Initializing AI services...")
            
            # Initialize LLM service
            llm_config = LLMConfig()
            self.llm_service = LLMService(llm_config)
            
            # Initialize embedding service (with fallback)
            try:
                embedding_config = EmbeddingConfig(
                    test_cases_dir=self.config.existing_tests_dir
                )
                self.embedding_service = EmbeddingIndexerService(embedding_config)
                self.logger.info("Using advanced embedding service with CodeBERT")
            except Exception as e:
                self.logger.warning(f"Advanced embedding service failed: {e}")
                self.logger.info("Falling back to simple embedding service")
                self.embedding_service = SimpleEmbeddingIndexerService(
                    self.config.existing_tests_dir
                )
            
            # Initialize KDoc service
            self.kdoc_service = KDocService(self.llm_service)
            
            # Initialize test generator
            self.test_generator = KotlinTestGenerator(
                llm_provider=self.llm_service,
                similarity_indexer=self.embedding_service,
                config=self.config
            )
            
            self.logger.info("Successfully initialized all services")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise
    
    def improve_tests_with_feedback(
        self, 
        source_code: str, 
        generated_test_code: str, 
        user_feedback: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Improve generated tests based on user feedback.
        
        Args:
            source_code: The original Kotlin source code
            generated_test_code: Previously generated test code
            user_feedback: User feedback for improving the tests
            output_dir: Directory to save improved test files
            
        Returns:
            str: Improved test code
        """
        self.logger.info("Improving tests based on user feedback...")
        output_dir = output_dir or self.config.output_dir
        
        try:
            # Create a prompt that includes the feedback and previous test code
            prompt = f"""
            Please improve the following test code based on the user feedback.
            
            Original Kotlin code:
            ```kotlin
            {source_code}
            ```
            
            Generated test code (to be improved):
            ```kotlin
            {generated_test_code}
            ```
            
            User feedback: {user_feedback}
            
            Please generate an improved version of the test code that addresses the feedback.
            Focus on:
            1. Addressing the specific feedback provided
            2. Maintaining or improving test coverage
            3. Following Kotlin testing best practices
            4. Adding more assertions if needed
            
            Return only the improved test code, without any additional explanation or markdown formatting.
            """
            
            # Generate improved test code
            response = self.llm_service.generate(
                prompt,
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more focused improvements
            )
            
            # Extract the text from the LLMResponse object
            improved_test = response.text if hasattr(response, 'text') and response.text else generated_test_code
            
            # Save the improved test to a file
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                test_file = os.path.join(output_dir, 'ImprovedTest.kt')
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(improved_test)
                self.logger.info(f"Saved improved test to {test_file}")
            
            return improved_test
            
        except Exception as e:
            self.logger.error(f"Error improving tests with feedback: {e}")
            raise
    
    def generate_tests(self, source_dir: Optional[str] = None) -> bool:
        """
        Generate tests for Kotlin source files.
        
        Args:
            source_dir: Directory containing Kotlin source files
            
        Returns:
            True if generation was successful, False otherwise
        """
        source_dir = source_dir or self.config.source_dir
        
        try:
            self.logger.info(f"Starting test generation for: {source_dir}")
            
            # Check if source directory exists
            if not os.path.exists(source_dir):
                self.logger.error(f"Source directory does not exist: {source_dir}")
                return False
            
            # Generate tests for all files in directory
            results = self.test_generator.generate_tests_for_directory(source_dir)
            
            # Report results
            successful = sum(1 for r in results if (getattr(r, 'status', None) == GenerationStatus.COMPLETED or getattr(r, 'status', None) == 'completed' or getattr(r, 'status', None).value == 'completed'))
            total = len(results)
            self.logger.info(f"Test generation completed: {successful}/{total} files successful")
            # Print detailed results
            for result in results:
                status = getattr(result, 'status', None)
                is_success = (
                    status == GenerationStatus.COMPLETED or
                    status == 'completed' or
                    (hasattr(status, 'value') and status.value == 'completed')
                )
                if is_success:
                    print(f"✅ Generated test: {getattr(result, 'output_file', '[no file]')}")
                else:
                    print(f"❌ Failed: {getattr(result, 'output_file', '[no file]')} - {result.error_message}")
            return successful > 0
            
        except Exception as e:
            self.logger.error(f"Error in test generation: {e}")
            return False
    
    def generate_kdoc(self, source_dir: Optional[str] = None) -> bool:
        """
        Generate KDoc comments for Kotlin source files.
        
        Args:
            source_dir: Directory containing Kotlin source files
            
        Returns:
            True if generation was successful, False otherwise
        """
        source_dir = source_dir or self.config.source_dir
        
        try:
            self.logger.info(f"Starting KDoc generation for: {source_dir}")
            
            # Generate KDoc for directory
            result = self.kdoc_service.generate_kdoc_for_directory(source_dir)
            
            self.logger.info(f"KDoc generation completed: {result.success_rate():.1f}% success rate")
            
            # Print detailed results
            for file_result in result.results:
                if file_result.success:
                    if file_result.changes_made:
                        print(f"✅ Updated: {file_result.file_path}")
                    else:
                        print(f"➖ No changes: {file_result.file_path}")
                else:
                    print(f"❌ Failed: {file_result.file_path} - {file_result.error_message}")
            
            return result.successful_files > 0
            
        except Exception as e:
            self.logger.error(f"Error in KDoc generation: {e}")
            return False
    
    def check_health(self) -> bool:
        """
        Check the health of all services.
        
        Returns:
            True if all services are healthy, False otherwise
        """
        try:
            self.logger.info("Checking service health...")
            
            # Check LLM service
            llm_available = self.llm_service.is_available()
            print(f"LLM Service: {'✅ Available' if llm_available else '❌ Unavailable'}")
            
            # Check embedding service
            embedding_available = hasattr(self.embedding_service, 'find_similar')
            print(f"Embedding Service: {'✅ Available' if embedding_available else '❌ Unavailable'}")
            
            # Check KDoc service
            kdoc_available = self.kdoc_service is not None
            print(f"KDoc Service: {'✅ Available' if kdoc_available else '❌ Unavailable'}")
            
            overall_health = llm_available and embedding_available and kdoc_available
            print(f"Overall Health: {'✅ Healthy' if overall_health else '❌ Issues detected'}")
            
            return overall_health
            
        except Exception as e:
            self.logger.error(f"Error checking health: {e}")
            return False
    
    def get_metrics(self) -> dict:
        """Get performance metrics from all services."""
        try:
            metrics = {
                'test_generator': self.test_generator.get_metrics().__dict__,
                'llm_service': self.llm_service.get_model_info(),
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {}


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AI-powered Kotlin test generation and documentation system"
    )
    
    parser.add_argument(
        'command',
        choices=['test', 'kdoc', 'health', 'metrics', 'improve'],
        help='''Command to execute:
        test - Generate tests for Kotlin files
        kdoc - Generate KDoc documentation
        health - Check service health
        metrics - Get performance metrics
        improve - Improve existing tests with feedback
        '''
    )
    
    # Feedback improvement arguments
    feedback_group = parser.add_argument_group('Feedback Improvement')
    feedback_group.add_argument(
        '--source-file',
        help='Path to the source Kotlin file (for improve command)'
    )
    feedback_group.add_argument(
        '--test-file',
        help='Path to the generated test file to improve (for improve command)'
    )
    feedback_group.add_argument(
        '--feedback',
        help='User feedback for improving the tests (for improve command)'
    )
    feedback_group.add_argument(
        '--output-file',
        help='Path to save the improved test file (default: overwrites test-file)'
    )
    
    # Common arguments
    parser.add_argument(
        '--source-dir',
        default='input-src',
        help='Directory containing Kotlin source files'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output-test',
        help='Directory for generated test files'
    )
    
    parser.add_argument(
        '--existing-tests-dir',
        default='testcase-datastore',
        help='Directory containing existing test cases for context'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = GenerationConfig(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            existing_tests_dir=args.existing_tests_dir
        )
        
        # Initialize application
        app = GenAIApplication(config)
        
        # Execute the requested command
        if args.command == 'test':
            success = app.generate_tests(args.source_dir)
            sys.exit(0 if success else 1)
            
        elif args.command == 'kdoc':
            success = app.generate_kdoc(args.source_dir)
            sys.exit(0 if success else 1)
            
        elif args.command == 'health':
            healthy = app.check_health()
            sys.exit(0 if healthy else 1)
            
        elif args.command == 'metrics':
            metrics = app.get_metrics()
            print("\nPerformance Metrics:")
            print("-" * 40)
            print(json.dumps(metrics, indent=2))
            sys.exit(0)
            
        elif args.command == 'improve':
            if not all([args.source_file, args.test_file, args.feedback]):
                print("Error: --source-file, --test-file, and --feedback are required for the improve command")
                parser.print_help()
                sys.exit(1)
                
            try:
                # Read source and test code
                with open(args.source_file, 'r') as f:
                    source_code = f.read()
                    
                with open(args.test_file, 'r') as f:
                    test_code = f.read()
                    
                # Generate improved test code
                output_file = args.output_file or args.test_file
                improved_test = app.improve_tests_with_feedback(
                    source_code=source_code,
                    generated_test_code=test_code,
                    user_feedback=args.feedback,
                    output_dir=os.path.dirname(output_file) or None
                )
                
                # Save the improved test
                with open(output_file, 'w') as f:
                    f.write(improved_test)
                    
                print(f"✅ Successfully improved test code saved to: {output_file}")
                sys.exit(0)
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error improving test: {e}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
