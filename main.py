#!/usr/bin/env python3
"""
DevTools - AI-Powered Kotlin Documentation & Test Generator
Main entry point for Docker container deployment
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from KdocGenerator import generate_kdoc_for_file, update_kdocs_in_file
    from TestCaseGenerator import KotlinTestGenerator
    from LLMClient import LLMClient
    from EmbeddingIndexer import EmbeddingIndexer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the src/ directory")
    sys.exit(1)

def find_kotlin_files(directory):
    """Find all Kotlin files in the given directory."""
    kotlin_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.kt'):
                kotlin_files.append(os.path.join(root, file))
    return kotlin_files

def generate_kdoc(input_dir, output_dir):
    """Generate KDoc documentation for Kotlin files."""
    print("=== Generating KDoc Documentation ===")
    
    kotlin_files = find_kotlin_files(input_dir)
    if not kotlin_files:
        print(f"No Kotlin files found in {input_dir}")
        return
    
    # Create output directory structure
    kdoc_output_dir = os.path.join(output_dir, "kdocs")
    os.makedirs(kdoc_output_dir, exist_ok=True)
    
    processed_count = 0
    for kotlin_file in kotlin_files:
        try:
            # Read the original file
            with open(kotlin_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Generate KDoc
            updated_content = generate_kdoc_for_file(original_content)
            
            # Create output file path maintaining directory structure
            rel_path = os.path.relpath(kotlin_file, input_dir)
            output_file = os.path.join(kdoc_output_dir, rel_path)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write updated content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ Generated KDoc for: {rel_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {kotlin_file}: {e}")
    
    print(f"Processed {processed_count} Kotlin files")

def generate_tests(input_dir, output_dir, data_dir, llm_client, indexer):
    """Generate test cases for Kotlin files."""
    print("=== Generating Test Cases ===")
    
    test_output_dir = os.path.join(output_dir, "tests")
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        test_generator = KotlinTestGenerator(input_dir, test_output_dir, llm_client, indexer)
        test_generator.generate_tests()
        print("✓ Test generation completed")
    except Exception as e:
        print(f"✗ Error generating tests: {e}")

def main():
    parser = argparse.ArgumentParser(description="DevTools - AI-Powered Kotlin Documentation & Test Generator")
    parser.add_argument("command", choices=["kdoc", "test", "both"], 
                       help="Operation to perform")
    parser.add_argument("--input-dir", default="/app/input", 
                       help="Input directory containing Kotlin files")
    parser.add_argument("--output-dir", default="/app/output", 
                       help="Output directory for generated files")
    parser.add_argument("--data-dir", default="/app/data", 
                       help="Directory containing existing test cases for RAG")
    parser.add_argument("--ollama-url", 
                       default=os.getenv("OLLAMA_API_URL", "http://ollama:11434/api/generate"),
                       help="Ollama API URL")
    parser.add_argument("--model", 
                       default=os.getenv("MODEL_NAME", "codellama:instruct"),
                       help="AI model name")
    
    args = parser.parse_args()
    
    # Print header
    print("DevTools - Test Case Generator and KDoc Generator")
    print("=================================================")
    print()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    # Find Kotlin files
    kotlin_files = find_kotlin_files(args.input_dir)
    if not kotlin_files:
        print(f"No Kotlin files found in {args.input_dir}")
        return 1
    
    print(f"Found {len(kotlin_files)} Kotlin files to process")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components for test generation
    llm_client = None
    indexer = None
    
    if args.command in ["test", "both"]:
        try:
            llm_client = LLMClient(args.ollama_url, args.model)
            indexer = EmbeddingIndexer(args.data_dir)
        except Exception as e:
            print(f"Warning: Could not initialize AI components: {e}")
            if args.command == "test":
                return 1
    
    # Execute commands
    try:
        if args.command in ["kdoc", "both"]:
            generate_kdoc(args.input_dir, args.output_dir)
        
        if args.command in ["test", "both"]:
            if llm_client and indexer:
                generate_tests(args.input_dir, args.output_dir, args.data_dir, llm_client, indexer)
            else:
                print("Skipping test generation due to initialization errors")
        
        print()
        print("=== Processing Complete ===")
        print(f"Output files are available in: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
