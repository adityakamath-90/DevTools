"""
Advanced Kotlin code parser for extracting class information and structure.

This module provides sophisticated parsing capabilities for Kotlin source code,
including class detection, method extraction, and property analysis.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from models.data_models import KotlinClass, KotlinMethod, KotlinProperty
from interfaces.base_interfaces import CodeParser
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a Kotlin source file."""
    success: bool
    kotlin_class: Optional[KotlinClass] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class KotlinParser(CodeParser):
    """
    Production-ready Kotlin source code parser with comprehensive analysis.
    
    Features:
    - Comment-aware parsing
    - Multiple class detection with prioritization
    - Method and property extraction
    - Comprehensive error handling
    - Detailed logging and metrics
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._class_pattern = re.compile(
            r'^\s*(data\s+)?(class|object|interface|enum\s+class)\s+([A-Za-z0-9_]+)', 
            re.MULTILINE
        )
        self._method_pattern = re.compile(
            r'^\s*(override\s+)?(fun)\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*:\s*([^{=\n]*)?[{=]?',
            re.MULTILINE
        )
        self._property_pattern = re.compile(
            r'^\s*(val|var)\s+([A-Za-z0-9_]+)\s*:\s*([^=\n]*)?',
            re.MULTILINE
        )
    
    def parse_file(self, file_path: str, source_code: str) -> ParseResult:
        """
        Parse a Kotlin source file and extract class information.
        
        Args:
            file_path: Path to the source file
            source_code: Content of the source file
            
        Returns:
            ParseResult with extracted class information
        """
        self.logger.info(f"Parsing Kotlin file: {file_path}")
        
        try:
            # Clean source code by removing comments
            clean_code = self._remove_comments(source_code)
            
            # Extract primary class
            class_name = self._extract_primary_class_name(clean_code)
            if not class_name:
                return ParseResult(
                    success=False,
                    error_message="No class found in source code"
                )
            
            # Extract class information
            kotlin_class = self._extract_class_info(
                class_name, clean_code, file_path
            )
            
            self.logger.info(f"Successfully parsed class: {class_name}")
            return ParseResult(
                success=True,
                kotlin_class=kotlin_class
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
            return ParseResult(
                success=False,
                error_message=f"Parsing error: {str(e)}"
            )
    
    def extract_class_info(self, source_code: str, file_path: str) -> Optional[KotlinClass]:
        """
        Extract class information from source code.
        
        Args:
            source_code: Kotlin source code
            file_path: Path to the source file
            
        Returns:
            KotlinClass object or None if no class found
        """
        result = self.parse_file(file_path, source_code)
        return result.kotlin_class if result.success else None
    
    def parse_kotlin_file(self, file_path: str) -> List[KotlinClass]:
        """Parse a Kotlin file and extract class information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
            
            kotlin_class = self.extract_class_info(source_code, file_path)
            return [kotlin_class] if kotlin_class else []
            
        except Exception as e:
            self.logger.error(f"Failed to parse file {file_path}: {e}")
            return []
    
    def extract_methods(self, class_code: str) -> List[str]:
        """Extract method signatures from class code."""
        methods = []
        
        # Pattern to match function definitions
        function_pattern = re.compile(
            r'^\s*(?:public\s+|private\s+|protected\s+|internal\s+)?'
            r'(?:override\s+|open\s+|abstract\s+|final\s+)?'
            r'(?:suspend\s+)?'
            r'fun\s+(\w+)\s*\([^)]*\)\s*(?::\s*[^{=\n]+)?',
            re.MULTILINE
        )
        
        matches = function_pattern.findall(class_code)
        for match in matches:
            methods.append(match)
        
        return methods
    
    def extract_properties(self, class_code: str) -> List[str]:
        """Extract property definitions from class code."""
        properties = []
        
        # Pattern to match property declarations
        property_pattern = re.compile(
            r'^\s*(?:public\s+|private\s+|protected\s+|internal\s+)?'
            r'(?:val|var)\s+(\w+)\s*:\s*[^=\n]+',
            re.MULTILINE
        )
        
        matches = property_pattern.findall(class_code)
        for match in matches:
            properties.append(match)
        
        return properties
    
    def is_data_class(self, class_code: str) -> bool:
        """Check if a class is a data class."""
        data_class_pattern = re.compile(r'^\s*data\s+class\s+', re.MULTILINE)
        return bool(data_class_pattern.search(class_code))
    
    def _remove_comments(self, code: str) -> str:
        """Remove single-line and multi-line comments from code."""
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        return code
    
    def _extract_primary_class_name(self, code: str) -> Optional[str]:
        """
        Extract the primary class name from cleaned Kotlin code.
        
        Prioritizes non-data classes over data classes.
        """
        matches = self._class_pattern.findall(code)
        if not matches:
            return None
        
        # If there's only one class/object, return it
        if len(matches) == 1:
            return matches[0][2]  # Return the class name (third group)
        
        # If there are multiple classes, prefer non-data classes
        for match in matches:
            data_modifier, class_type, class_name = match
            if not data_modifier:  # Not a data class
                return class_name
        
        # If all are data classes, return the first one found
        return matches[0][2]
    
    def _extract_class_info(self, class_name: str, source_code: str, file_path: str) -> KotlinClass:
        """Extract comprehensive class information."""
        # Extract methods
        methods = self._extract_methods(source_code)
        
        # Extract properties
        properties = self._extract_properties(source_code)
        
        # Determine if it's a data class
        is_data_class = 'data class' in source_code
        
        return KotlinClass(
            name=class_name,
            file_path=file_path,
            source_code=source_code,
            methods=methods,
            properties=properties,
            is_data_class=is_data_class,
            package_name=self._extract_package_name(source_code),
            imports=self._extract_imports(source_code)
        )
    
    def _extract_methods(self, source_code: str) -> List[KotlinMethod]:
        """Extract method information from source code."""
        methods = []
        matches = self._method_pattern.findall(source_code)
        
        for match in matches:
            override_modifier, fun_keyword, method_name, return_type = match
            
            # Extract parameters (simplified)
            param_pattern = re.compile(
                rf'fun\s+{re.escape(method_name)}\s*\(([^)]*)\)',
                re.MULTILINE
            )
            param_match = param_pattern.search(source_code)
            parameters = []
            if param_match:
                param_str = param_match.group(1).strip()
                if param_str:
                    # Simple parameter parsing (can be enhanced)
                    param_parts = param_str.split(',')
                    for param in param_parts:
                        param = param.strip()
                        if ':' in param:
                            param_name, param_type = param.split(':', 1)
                            parameters.append({
                                'name': param_name.strip(),
                                'type': param_type.strip()
                            })
            
            methods.append(KotlinMethod(
                name=method_name,
                parameters=parameters,
                return_type=return_type.strip() if return_type else None,
                is_override=bool(override_modifier)
            ))
        
        return methods
    
    def _extract_properties(self, source_code: str) -> List[KotlinProperty]:
        """Extract property information from source code."""
        properties = []
        matches = self._property_pattern.findall(source_code)
        
        for match in matches:
            prop_type, prop_name, prop_kotlin_type = match
            properties.append(KotlinProperty(
                name=prop_name,
                type=prop_kotlin_type.strip() if prop_kotlin_type else None,
                is_mutable=prop_type == 'var'
            ))
        
        return properties
    
    def _extract_package_name(self, source_code: str) -> Optional[str]:
        """Extract package name from source code."""
        package_pattern = re.compile(r'^\s*package\s+([A-Za-z0-9_.]+)', re.MULTILINE)
        match = package_pattern.search(source_code)
        return match.group(1) if match else None
    
    def _extract_imports(self, source_code: str) -> List[str]:
        """Extract import statements from source code."""
        import_pattern = re.compile(r'^\s*import\s+([A-Za-z0-9_.]+)', re.MULTILINE)
        matches = import_pattern.findall(source_code)
        return matches
    
    def extract_class_name(self, source_code: str) -> Optional[str]:
        """Extract the primary class name from source code."""
        return self._extract_primary_class_name(source_code)
