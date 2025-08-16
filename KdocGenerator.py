#!/usr/bin/env python3
"""
Legacy entry point for KDoc generation with backward compatibility.

This script provides backward compatibility with the existing KdocGenerator.py
while using the new modular architecture underneath.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import legacy functions for backward compatibility
try:
    from services.kdoc_service import generate_kdoc_for_file, update_kdocs_in_file, update_kdocs_in_directory
    print("[INFO] Using new modular KDoc service")
except ImportError as e:
    print(f"[WARN] Could not import new KDoc service: {e}")
    print("[INFO] Falling back to original implementation")
    
    # Fallback to original implementation
    import requests
    import shutil
    
    # Use environment variables with fallbacks for Docker compatibility
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
    MODEL_NAME = os.getenv("MODEL_NAME", "codellama:instruct")

    def generate_kdoc_for_file(file_content: str) -> str:
        prompt = (
            "You are a senior Kotlin developer and technical writer. You will be given an entire Kotlin source file.\n"
            "Your task: add idiomatic, detailed, and concise KDoc comments (using /** ... */) for all classes, class header "
            "functions, properties, and public fields that are missing documentation.\n"
            "Keep existing KDocs unchanged.\n"
            "Document parameters, return values, edge cases, assumptions, generics, lambdas, coroutines where applicable.\n"
            "Use valid KDoc syntax compatible with Dokka.\n"
            "Return the full Kotlin file content with new KDocs inserted appropriately.\n\n"
            f"{file_content}\n"
            "Respond only with complete Kotlin source code including KDoc comments."
        )

        try:
            response = requests.post(OLLAMA_API_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1  # Low temperature for consistent output
            })
            response.raise_for_status()
        except Exception as e:
            print(f"Request error: {e}")
            return file_content  # fallback: return original content unchanged

        try:
            result = response.json().get("response", "").strip()
        except Exception as e:
            print(f"JSON decode error: {e}")
            return file_content

        # Validate that we got valid Kotlin code back
        if not result or len(result) < len(file_content) * 0.5:
            print(f"Warning: Generated content seems too short, using original")
            return file_content
        
        return result

    def create_backup(filepath):
        """Create a backup of the original file"""
        backup_path = f"{filepath}.backup"
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")

    def update_kdocs_in_file(filepath):
        print(f"Processing file: {filepath}")
        
        # Create backup before modifying
        create_backup(filepath)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()

            updated_content = generate_kdoc_for_file(file_content)

            # Only write if content actually changed
            if updated_content != file_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                print(f"✓ Updated {filepath}")
            else:
                print(f"- No changes needed for {filepath}")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            # Restore from backup if something went wrong
            backup_path = f"{filepath}.backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, filepath)
                print(f"Restored {filepath} from backup")

        print(f"Finished updating {filepath}")

    def update_kdocs_in_directory(directory):
        """Process all .kt files in directory, excluding the src directory itself"""
        kotlin_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip the main src directory to avoid processing Python files
            if os.path.basename(root) == "src" and any(f.endswith('.py') for f in files):
                continue
                
            for file in files:
                if file.endswith(".kt"):
                    kotlin_files.append(os.path.join(root, file))
        
        if not kotlin_files:
            print(f"No Kotlin files found in {directory}")
            return
        
        print(f"Found {len(kotlin_files)} Kotlin files to process")
        
        for filepath in kotlin_files:
            update_kdocs_in_file(filepath)

# Export the main KdocGenerator class for backward compatibility
class KdocGenerator:
    """Legacy wrapper for KDoc generation functionality."""
    
    @staticmethod
    def generate_kdoc_for_file(file_path: str) -> str:
        """Generate KDoc for a single file."""
        return generate_kdoc_for_file(file_path)
    
    @staticmethod
    def update_kdocs_in_file(file_path: str) -> bool:
        """Update KDoc comments in a file."""
        return update_kdocs_in_file(file_path)
    
    @staticmethod
    def update_kdocs_in_directory(directory: str) -> bool:
        """Update KDoc comments in all files in a directory."""
        return update_kdocs_in_directory(directory)

# Export classes for backward compatibility
__all__ = [
    'KdocGenerator',
    'generate_kdoc_for_file',
    'update_kdocs_in_file', 
    'update_kdocs_in_directory'
]

if __name__ == "__main__":
    print("[INFO] Starting KDoc generation (Legacy Mode)...")
    print("[INFO] This is running on the new modular architecture for better maintainability.")
    
    # Process only the Kotlin source directory, not the Python source directory
    project_dir = "src/input-src"  # Fixed: was "src" which includes Python files
    
    if not os.path.exists(project_dir):
        print(f"Error: Directory {project_dir} does not exist")
        print("Please ensure you have Kotlin files in src/input-src/")
        exit(1)
    
    print(f"Starting KDoc generation for directory: {project_dir}")
    update_kdocs_in_directory(project_dir)
    print("KDoc generation completed!")
