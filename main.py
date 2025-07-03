#!/usr/bin/env python3
"""
Main entry point for the AI-powered Kotlin test generation system.

This script demonstrates the new modular architecture and provides both
new and legacy interfaces for generating Kotlin tests and KDoc comments.
"""

import os
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
            successful = sum(1 for r in results if r.is_successful)
            total = len(results)
            
            self.logger.info(f"Test generation completed: {successful}/{total} files successful")
            
            # Print detailed results
            for result in results:
                if result.is_successful:
                    print(f"✅ Generated test: {result.output_file}")
                else:
                    print(f"❌ Failed: {result.source_file} - {result.error_message}")
            
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
        choices=['test', 'kdoc', 'health', 'metrics'],
        help='Command to execute'
    )
    
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
        
        # Execute command
        if args.command == 'test':
            success = app.generate_tests()
            sys.exit(0 if success else 1)
        
        elif args.command == 'kdoc':
            success = app.generate_kdoc()
            sys.exit(0 if success else 1)
        
        elif args.command == 'health':
            healthy = app.check_health()
            sys.exit(0 if healthy else 1)
        
        elif args.command == 'metrics':
            metrics = app.get_metrics()
            print("Performance Metrics:")
            for service, data in metrics.items():
                print(f"\n{service.title()}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
            sys.exit(0)
    
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
