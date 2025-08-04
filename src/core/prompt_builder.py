"""
Intelligent prompt builder for AI-powered Kotlin test generation.

This module provides sophisticated prompt construction capabilities for different
types of code generation tasks, with template-based approach and context awareness.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from models.data_models import KotlinClass, PromptTemplate, DEFAULT_GENERATION_TEMPLATE, DEFAULT_ACCURACY_TEMPLATE
from interfaces.base_interfaces import PromptBuilder as IPromptBuilder
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptContext:
    """Context information for prompt generation."""
    kotlin_class: KotlinClass
    similar_tests: List[str]
    additional_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_context is None:
            self.additional_context = {}


class PromptBuilder(IPromptBuilder):
    """
    Production-ready prompt builder with template-based approach.
    
    Features:
    - Template-based prompt construction
    - Context-aware generation
    - Support for multiple prompt types
    - Comprehensive error handling
    - Extensible template system
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._templates = {
            'generation': DEFAULT_GENERATION_TEMPLATE,
            'accuracy': DEFAULT_ACCURACY_TEMPLATE
        }
    
    def build_generation_prompt(
        self, 
        kotlin_class: KotlinClass, 
        similar_tests: List[str]
    ) -> str:
        """
        Build a comprehensive prompt for test generation.
        
        Args:
            kotlin_class: The Kotlin class to generate tests for
            similar_tests: List of similar existing test cases
            
        Returns:
            Formatted prompt string for test generation
        """
        self.logger.info(f"Building generation prompt for class: {kotlin_class.name}")
        try:
            context = self._build_context_string(similar_tests)
            methods_info = self._build_methods_info(kotlin_class)
            template = self._templates['generation']
            prompt = template.template.format(
                class_name=kotlin_class.name,
                class_code=kotlin_class.source_code,
                similar_tests=context,
                methods_info=methods_info
            )
            self.logger.debug(f"Generated prompt length: {len(prompt)} characters")
            return prompt
        except Exception as e:
            self.logger.error(f"Error building generation prompt: {e}")
            # Fallback to basic prompt
            return self._build_basic_generation_prompt(kotlin_class, similar_tests)
    
    def build_validation_prompt(
        self, 
        kotlin_class: KotlinClass, 
        generated_test: str
    ) -> str:
        """
        Build a prompt for test validation and improvement.
        
        Args:
            kotlin_class: The original Kotlin class
            generated_test: The generated test code to validate
            
        Returns:
            Formatted prompt string for test validation
        """
        self.logger.info(f"Building validation prompt for class: {kotlin_class.name}")
        try:
            template = self._templates['accuracy']
            prompt = template.template.format(
                class_code=kotlin_class.source_code,
                test_code=generated_test  # Changed from generated_test to test_code to match template
            )
            return prompt
        except Exception as e:
            self.logger.error(f"Error building validation prompt: {e}")
            # Fallback to basic prompt
            return self._build_basic_validation_prompt(kotlin_class, generated_test)
    
    def build_improvement_prompt(
        self, 
        kotlin_class: KotlinClass, 
        test_code: str, 
        feedback: str
    ) -> str:
        """
        Build a prompt for test improvement based on feedback.
        
        Args:
            kotlin_class: The original Kotlin class
            test_code: The current test code
            feedback: Feedback for improvement
            
        Returns:
            Formatted prompt string for test improvement
        """
        self.logger.info(f"Building improvement prompt for class: {kotlin_class.name}")
        
        return (
            f"You are a senior Android Kotlin developer improving unit tests based on feedback.\n\n"
            f"Original class:\n{kotlin_class.source_code}\n\n"
            f"Current test code:\n{test_code}\n\n"
            f"Feedback for improvement:\n{feedback}\n\n"
            "Please improve the test code based on the feedback provided.\n"
            "Maintain idiomatic Kotlin test style with JUnit 5 and MockK.\n"
            "Return only the improved Kotlin test source code.\n"
        )
    
    def add_custom_template(self, name: str, template: PromptTemplate) -> None:
        """Add a custom template to the builder."""
        self._templates[name] = template
        self.logger.info(f"Added custom template: {name}")
    
    def _build_context_string(self, similar_tests: List[str]) -> str:
        """Build context string from similar tests."""
        if not similar_tests:
            return "// No relevant test cases found in corpus."
        
        return "\n\n---\n\n".join(similar_tests)
    
    def _build_methods_info(self, kotlin_class: KotlinClass) -> str:
        """Build a summary of methods in the class."""
        if not kotlin_class.methods:
            return "// No methods detected in class."
        
        method_summaries = []
        for method in kotlin_class.methods:
            params = ", ".join([f"{p.get('name', 'param')}: {p.get('type', 'Any')}" 
                              for p in method.parameters])
            return_type = method.return_type or "Unit"
            method_summaries.append(f"- {method.name}({params}): {return_type}")
        
        return "Methods to test:\n" + "\n".join(method_summaries)
    
    def _build_basic_generation_prompt(self, kotlin_class: KotlinClass, similar_tests: List[str]) -> str:
        """Fallback basic generation prompt."""
        context = self._build_context_string(similar_tests)
        
        return (
            f"You are a senior Android Kotlin developer tasked with writing comprehensive unit tests for the following Kotlin class: `{kotlin_class.name}`.\n\n"
            f"Class Code:\n{kotlin_class.source_code}\n\n"
            f"You may refer to these similar existing unit tests for context:\n{context}\n\n"
            "Requirements:\n"
            "- Cover all public methods in the class.\n"
            "- Include test cases for typical usage scenarios, edge cases, and error handling.\n"
            "- Use idiomatic Kotlin test style with JUnit 5 and MockK.\n"
            "- Use assertions such as assertEquals, assertTrue, assertFailsWith, etc.\n"
            "- Write meaningful test function names that describe what each test is verifying.\n"
            "- Return only pure Kotlin unit test code.\n"
            "- Do NOT include comments, explanations, markdown, or annotations beyond necessary test-related syntax.\n"
            "\nRespond ONLY with the Kotlin test source code."
        )
    
    def _build_basic_validation_prompt(self, kotlin_class: KotlinClass, generated_test: str) -> str:
        """Fallback basic validation prompt."""
        return (
            "You are a senior Android Kotlin developer reviewing a set of proposed unit tests.\n\n"
            f"Here is the Kotlin class being tested:\n{kotlin_class.source_code}\n\n"
            f"Here are the proposed unit tests:\n{generated_test}\n\n"
            "Please do the following:\n"
            "1. Confirm that the tests cover all key behaviors, public methods, edge cases, and exception paths.\n"
            "2. Identify any missing tests, logical flaws, or testing anti-patterns.\n"
            "3. Improve or rewrite tests where necessary to ensure full, accurate coverage.\n"
            "4. Use JUnit 5 and MockK idiomatically in Kotlin.\n"
            "\nOutput ONLY the corrected Kotlin unit test source code. Do NOT include explanations, comments, markdown, or any introductory text."
        )


# Legacy compatibility functions for backward compatibility
def build_generation_prompt(class_name: str, class_code: str, similar_tests: List[str]) -> str:
    """
    Legacy function for backward compatibility.
    
    Note: This function is deprecated. Use PromptBuilder class instead.
    """
    logger.warning("Using deprecated build_generation_prompt function. Consider using PromptBuilder class.")
    
    # Create a temporary KotlinClass object for compatibility
    from models.data_models import KotlinClass
    kotlin_class = KotlinClass(
        name=class_name,
        file_path="",
        source_code=class_code,
        methods=[],
        properties=[],
        is_data_class=False
    )
    
    builder = PromptBuilder()
    return builder.build_generation_prompt(kotlin_class, similar_tests)


def generate_accurate_prompt(class_code: str, generated_test: str) -> str:
    """
    Legacy function for backward compatibility.
    
    Note: This function is deprecated. Use PromptBuilder class instead.
    """
    logger.warning("Using deprecated generate_accurate_prompt function. Consider using PromptBuilder class.")
    
    # Create a temporary KotlinClass object for compatibility
    from models.data_models import KotlinClass
    kotlin_class = KotlinClass(
        name="UnknownClass",
        file_path="",
        source_code=class_code,
        methods=[],
        properties=[],
        is_data_class=False
    )
    
    builder = PromptBuilder()
    return builder.build_validation_prompt(kotlin_class, generated_test)
