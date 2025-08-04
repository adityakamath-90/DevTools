"""
Prompt templates for test generation using LangChain.

This module contains prompt templates for generating Kotlin test code
using the LangChain framework with Ollama's CodeLLaMA model.
"""

from langchain.prompts import PromptTemplate
from typing import Dict, Any, List, Optional


def get_test_generation_prompt() -> PromptTemplate:
    """
    Get the prompt template for generating Kotlin test code.
    
    Returns:
        A LangChain PromptTemplate instance for test generation
    """
    template = """You are an expert Kotlin developer. Generate a JUnit 5 test class for the following Kotlin class.
    Follow these guidelines:
    1. Use descriptive test method names
    2. Test both success and error cases
    3. Include assertions for all important behaviors
    4. Follow Kotlin coding standards
    5. Use JUnit 5 annotations
    6. Include proper setup and teardown methods if needed
    7. Mock external dependencies
    8. Add appropriate test data
    9. Include comments explaining complex test cases
    
    Class to test:
    ```kotlin
    {class_code}
    ```
    
    Similar test examples:
    {similar_tests}
    
    Generate a complete test class with all necessary imports. Only output the test class code, no explanations.
    
    Test class for {class_name}:
    """
    
    return PromptTemplate(
        input_variables=["class_code", "similar_tests", "class_name"],
        template=template,
        template_format="f-string"
    )


def get_test_validation_prompt() -> PromptTemplate:
    """
    Get the prompt template for validating generated test code.
    
    Returns:
        A LangChain PromptTemplate instance for test validation
    """
    template = """Analyze the following test code and provide feedback:
    
    Original class:
    ```kotlin
    {class_code}
    ```
    
    Generated test:
    ```kotlin
    {test_code}
    ```
    
    Validation errors (if any):
    {errors}
    
    Provide feedback in the following JSON format:
    {{
        "is_valid": boolean,
        "feedback": "General feedback on the test quality",
        "suggested_fixes": ["list", "of", "suggested", "fixes"],
        "test_coverage_score": 0-100,
        "test_quality_score": 0-100
    }}
    """
    
    return PromptTemplate(
        input_variables=["class_code", "test_code", "errors"],
        template=template,
        template_format="f-string"
    )


def get_test_improvement_prompt() -> PromptTemplate:
    """
    Get the prompt template for improving existing test code.
    
    Returns:
        A LangChain PromptTemplate instance for test improvement
    """
    template = """Improve the following test code based on the feedback:
    
    Original class:
    ```kotlin
    {class_code}
    ```
    
    Current test:
    ```kotlin
    {test_code}
    ```
    
    Feedback:
    {feedback}
    
    Generate an improved version of the test code. Only output the improved test class code, no explanations.
    """
    
    return PromptTemplate(
        input_variables=["class_code", "test_code", "feedback"],
        template=template,
        template_format="f-string"
    )
