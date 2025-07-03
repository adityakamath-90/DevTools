#!/usr/bin/env python3
"""
Legacy entry point for backward compatibility.

This script provides backward compatibility with the existing TestCaseGenerator.py
while using the new modular architecture underneath.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import legacy wrapper classes for backward compatibility
from core.test_generator import LegacyKotlinTestGenerator as KotlinTestGenerator
from services.llm_service import LLMClient
from services.embedding_service import EmbeddingIndexer

# Import legacy prompt builder functions
try:
    from core.prompt_builder import build_generation_prompt, generate_accurate_prompt
    from core.prompt_builder import PromptBuilder
    
    # Create a legacy-compatible PromptBuilder class
    class LegacyPromptBuilder:
        @staticmethod
        def build_generation_prompt(class_name: str, class_code: str, similar_tests: list) -> str:
            return build_generation_prompt(class_name, class_code, similar_tests)
        
        @staticmethod
        def generate_accurate_prompt(class_code: str, generated_test: str) -> str:
            return generate_accurate_prompt(class_code, generated_test)
    
    # Override for direct import compatibility
    PromptBuilder = LegacyPromptBuilder

except ImportError as e:
    print(f"[WARN] Could not import new prompt builder: {e}")
    print("[INFO] Falling back to original PromptBuilder")
    
    # Fallback to original implementation
    class PromptBuilder:
        @staticmethod
        def build_generation_prompt(class_name: str, class_code: str, similar_tests: list) -> str:
            context = "\n\n---\n\n".join(similar_tests) if similar_tests else "// No relevant test cases found in corpus."
            return (
                f"You are a senior Android Kotlin developer tasked with writing comprehensive unit tests for the following Kotlin class: `{class_name}`.\n\n"
                f"Class Code:\n{class_code}\n\n"
                f"You may refer to these similar existing unit tests for context:\n{context}\n\n"
                "Requirements:\n"
                "- Cover all public methods in the class.\n"
                "- Include test cases for:\n"
                "  • Typical usage scenarios\n"
                "  • Edge cases\n"
                "  • Error and exception handling\n"
                "- Use idiomatic Kotlin test style with:\n"
                "  • JUnit 5 for structure\n"
                "  • MockK for mocking any dependencies\n"
                "- Use assertions such as assertEquals, assertTrue, assertFailsWith, etc.\n"
                "- Write meaningful test function names that describe what each test is verifying.\n"
                "- Return only pure Kotlin unit test code.\n"
                "- Do NOT include comments, explanations, markdown, or annotations beyond necessary test-related syntax.\n"
                "\nRespond ONLY with the Kotlin test source code."
            )

        @staticmethod
        def generate_accurate_prompt(class_code: str, generated_test: str) -> str:
            return (
                "You are a senior Android Kotlin developer reviewing a set of proposed unit tests.\n\n"
                f"Here is the Kotlin class being tested:\n{class_code}\n\n"
                f"Here are the proposed unit tests:\n{generated_test}\n\n"
                "Please do the following:\n"
                "1. Confirm that the tests cover all key behaviors, public methods, edge cases, and exception paths.\n"
                "2. Identify any missing tests, logical flaws, or testing anti-patterns.\n"
                "3. Improve or rewrite tests where necessary to ensure full, accurate coverage.\n"
                "4. Use JUnit 5 and MockK idiomatically in Kotlin.\n"
                "\nOutput ONLY the corrected Kotlin unit test source code. Do NOT include explanations, comments, markdown, or any introductory text."
            )

# Configuration constants for backward compatibility
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "codellama:instruct"

# Export the main TestCaseGenerator class for backward compatibility
TestCaseGenerator = KotlinTestGenerator

# Export other classes for backward compatibility
__all__ = [
    'TestCaseGenerator',
    'KotlinTestGenerator', 
    'LLMClient',
    'EmbeddingIndexer',
    'PromptBuilder'
]

# === Main execution (legacy compatibility) ===
if __name__ == "__main__":
    print("[INFO] Starting TestCaseGenerator (Legacy Mode)...")
    print("[INFO] This is running on the new modular architecture for better maintainability.")
    
    source_dir = sys.argv[1] if len(sys.argv) > 1 else ("src/input-src" if os.path.exists("src/input-src") else ".")
    test_dir = "output-test" 
    existing_tests_dir = "src/testcase--datastore"
    
    print(f"[INFO] Source directory: {source_dir}")
    print(f"[INFO] Test output directory: {test_dir}")
    print(f"[INFO] Existing tests directory: {existing_tests_dir}")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory {source_dir} does not exist")
        sys.exit(1)
    
    # Check if existing tests directory exists
    if not os.path.exists(existing_tests_dir):
        print(f"[WARN] Existing tests directory {existing_tests_dir} does not exist")
        print("[INFO] Creating empty existing tests directory...")
        os.makedirs(existing_tests_dir, exist_ok=True)
    
    try:
        print("[INFO] Initializing LLMClient...")
        llm_client = LLMClient(api_url=OLLAMA_API_URL, model_name=MODEL_NAME)
        
        print("[INFO] Initializing EmbeddingIndexer...")
        indexer = EmbeddingIndexer(test_dir=existing_tests_dir)
        
        print("[INFO] Initializing KotlinTestGenerator...")
        generator = KotlinTestGenerator(source_dir=source_dir, test_dir=test_dir, llm_client=llm_client, indexer=indexer)
        
        print("[INFO] Starting test generation...")
        generator.generate_tests_for_all()
        
        print("[INFO] Test generation completed!")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize components: {e}")
        print("Make sure all required dependencies are installed:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
        sys.exit(1)
