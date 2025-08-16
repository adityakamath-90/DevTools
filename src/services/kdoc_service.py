"""
KDoc generation service for automatic Kotlin documentation.

This module provides comprehensive KDoc generation capabilities using AI
to enhance Kotlin source code with professional documentation.
"""

import os
import shutil
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from interfaces.base_interfaces import LLMProvider
from src.config.settings import LLMConfig
from utils.logging import get_logger
from .llm_service import LLMService

logger = get_logger(__name__)


@dataclass
class KDocResult:
    """Result of KDoc generation operation."""
    success: bool
    file_path: str
    original_content: str
    enhanced_content: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    changes_made: bool = False
    backup_created: bool = False


@dataclass
class BatchKDocResult:
    """Result of batch KDoc generation."""
    total_files: int
    processed_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float
    results: List[KDocResult]
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.processed_files == 0:
            return 0.0
        return (self.successful_files / self.processed_files) * 100.0


class KDocService:
    """
    Production-ready KDoc generation service.
    
    Features:
    - AI-powered KDoc generation
    - Batch processing for multiple files
    - Automatic backup creation
    - Comprehensive error handling
    - File validation and safety checks
    - Detailed logging and metrics
    - Rollback capabilities
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.llm_provider = llm_provider or LLMService()
        
        # KDoc generation template
        self.kdoc_template = self._get_kdoc_template()
        
        self.logger.info("Initialized KDocService")
    
    def generate_kdoc_for_file(self, file_path: str, create_backup: bool = True) -> KDocResult:
        """
        Generate KDoc comments for a single Kotlin file.
        
        Args:
            file_path: Path to the Kotlin file
            create_backup: Whether to create a backup before modification
            
        Returns:
            KDocResult with generation details
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating KDoc for file: {file_path}")
            
            # Validate file
            if not self._validate_file(file_path):
                return KDocResult(
                    success=False,
                    file_path=file_path,
                    original_content="",
                    processing_time=time.time() - start_time,
                    error_message="File validation failed"
                )
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup if requested
            backup_created = False
            if create_backup:
                backup_created = self._create_backup(file_path)
            
            # Generate enhanced content
            enhanced_content = self._generate_enhanced_content(original_content)
            
            if not enhanced_content:
                return KDocResult(
                    success=False,
                    file_path=file_path,
                    original_content=original_content,
                    processing_time=time.time() - start_time,
                    error_message="Failed to generate enhanced content",
                    backup_created=backup_created
                )
            
            # Check if content actually changed
            changes_made = enhanced_content != original_content
            
            # Save enhanced content if changes were made
            if changes_made:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                
                self.logger.info(f"Successfully updated {file_path}")
            else:
                self.logger.info(f"No changes needed for {file_path}")
            
            return KDocResult(
                success=True,
                file_path=file_path,
                original_content=original_content,
                enhanced_content=enhanced_content,
                processing_time=time.time() - start_time,
                changes_made=changes_made,
                backup_created=backup_created
            )
            
        except Exception as e:
            self.logger.error(f"Error generating KDoc for {file_path}: {e}")
            
            # Restore from backup if something went wrong
            if create_backup:
                self._restore_from_backup(file_path)
            
            return KDocResult(
                success=False,
                file_path=file_path,
                original_content="",
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def generate_kdoc_for_directory(self, directory_path: str, create_backups: bool = True) -> BatchKDocResult:
        """
        Generate KDoc comments for all Kotlin files in a directory.
        
        Args:
            directory_path: Path to the directory containing Kotlin files
            create_backups: Whether to create backups before modification
            
        Returns:
            BatchKDocResult with processing details
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting batch KDoc generation for directory: {directory_path}")
            
            # Find all Kotlin files
            kotlin_files = self._find_kotlin_files(directory_path)
            
            if not kotlin_files:
                self.logger.warning(f"No Kotlin files found in {directory_path}")
                return BatchKDocResult(
                    total_files=0,
                    processed_files=0,
                    successful_files=0,
                    failed_files=0,
                    total_processing_time=time.time() - start_time,
                    results=[]
                )
            
            self.logger.info(f"Found {len(kotlin_files)} Kotlin files to process")
            
            # Process each file
            results = []
            successful_files = 0
            failed_files = 0
            
            for file_path in kotlin_files:
                result = self.generate_kdoc_for_file(file_path, create_backups)
                results.append(result)
                
                if result.success:
                    successful_files += 1
                else:
                    failed_files += 1
            
            total_processing_time = time.time() - start_time
            
            self.logger.info(f"Completed batch KDoc generation. "
                           f"Success rate: {(successful_files/len(kotlin_files))*100:.1f}%")
            
            return BatchKDocResult(
                total_files=len(kotlin_files),
                processed_files=len(kotlin_files),
                successful_files=successful_files,
                failed_files=failed_files,
                total_processing_time=total_processing_time,
                results=results
            )
            
        except Exception as e:
            self.logger.error(f"Error in batch KDoc generation: {e}")
            return BatchKDocResult(
                total_files=0,
                processed_files=0,
                successful_files=0,
                failed_files=1,
                total_processing_time=time.time() - start_time,
                results=[]
            )
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate that the file is a readable Kotlin file."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check if it's a Kotlin file
            if not file_path.endswith('.kt'):
                self.logger.error(f"Not a Kotlin file: {file_path}")
                return False
            
            # Check if file is readable
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            
            return True
            
        except Exception as e:
            self.logger.error(f"File validation failed for {file_path}: {e}")
            return False
    
    def _create_backup(self, file_path: str) -> bool:
        """Create a backup of the original file."""
        try:
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return False
    
    def _restore_from_backup(self, file_path: str) -> bool:
        """Restore file from backup."""
        try:
            backup_path = f"{file_path}.backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                self.logger.info(f"Restored {file_path} from backup")
                return True
            else:
                self.logger.warning(f"Backup not found: {backup_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def _find_kotlin_files(self, directory_path: str) -> List[str]:
        """Find all Kotlin files in the directory."""
        kotlin_files = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'build']]
                
                for file in files:
                    if file.endswith('.kt'):
                        kotlin_files.append(os.path.join(root, file))
        
        except Exception as e:
            self.logger.error(f"Error finding Kotlin files: {e}")
        
        return kotlin_files
    
    def _generate_enhanced_content(self, original_content: str) -> Optional[str]:
        """Generate enhanced content with KDoc comments."""
        try:
            # Build prompt for KDoc generation
            prompt = self.kdoc_template.format(file_content=original_content)
            
            # Generate enhanced content
            enhanced_content = self.llm_provider.generate(prompt)
            
            # Validate the result
            if self._validate_enhanced_content(original_content, enhanced_content):
                return enhanced_content
            else:
                self.logger.warning("Generated content validation failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating enhanced content: {e}")
            return None
    
    def _validate_enhanced_content(self, original: str, enhanced: str) -> bool:
        """Validate that the enhanced content is reasonable."""
        if not enhanced:
            return False
        
        # Check if enhanced content is significantly shorter than original
        if len(enhanced) < len(original) * 0.5:
            self.logger.warning("Enhanced content seems too short")
            return False
        
        # Check if it contains KDoc comments
        if "/**" not in enhanced:
            self.logger.warning("Enhanced content doesn't contain KDoc comments")
            return False
        
        return True
    
    def _get_kdoc_template(self) -> str:
        """Get the KDoc generation template."""
        return """You are a senior Kotlin developer and technical writer. You will be given an entire Kotlin source file.

Your task: add idiomatic, detailed, and concise KDoc comments (using /** ... */) for all classes, functions, properties, and public fields that are missing documentation.

Keep existing KDocs unchanged.

Document:
- Parameters with @param
- Return values with @return
- Exceptions with @throws
- Edge cases and assumptions
- Generics, lambdas, coroutines where applicable

Use valid KDoc syntax compatible with Dokka.

Return the full Kotlin file content with new KDocs inserted appropriately.

Source file:
{file_content}

Respond only with complete Kotlin source code including KDoc comments."""


# Legacy compatibility functions
def generate_kdoc_for_file(file_content: str) -> str:
    """
    Legacy function for backward compatibility.
    
    Note: This function is deprecated. Use KDocService instead.
    """
    logger.warning("Using deprecated generate_kdoc_for_file function. Consider using KDocService.")
    
    service = KDocService()
    
    # For legacy compatibility, we'll work with content directly
    try:
        enhanced_content = service._generate_enhanced_content(file_content)
        return enhanced_content or file_content
    except Exception as e:
        logger.error(f"Error in legacy KDoc generation: {e}")
        return file_content


def update_kdocs_in_file(filepath: str):
    """
    Legacy function for backward compatibility.
    
    Note: This function is deprecated. Use KDocService instead.
    """
    logger.warning("Using deprecated update_kdocs_in_file function. Consider using KDocService.")
    
    service = KDocService()
    result = service.generate_kdoc_for_file(filepath)
    
    if result.success:
        if result.changes_made:
            print(f"✓ Updated {filepath}")
        else:
            print(f"- No changes needed for {filepath}")
    else:
        print(f"✗ Failed to update {filepath}: {result.error_message}")


def update_kdocs_in_directory(directory: str):
    """
    Legacy function for backward compatibility.
    
    Note: This function is deprecated. Use KDocService instead.
    """
    logger.warning("Using deprecated update_kdocs_in_directory function. Consider using KDocService.")
    
    service = KDocService()
    result = service.generate_kdoc_for_directory(directory)
    
    print(f"Processed {result.processed_files} files. Success rate: {result.success_rate():.1f}%")
