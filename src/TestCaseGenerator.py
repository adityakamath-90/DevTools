import os
import re
import sys
from typing import Optional

from PromptBuilder import PromptBuilder
from LLMClient import LLMClient

# Try to import the complex EmbeddingIndexer, fallback to simple one
try:
    from EmbeddingIndexer import EmbeddingIndexer
    print("[INFO] Using advanced EmbeddingIndexer")
except ImportError as e:
    print(f"[WARN] Could not import EmbeddingIndexer: {e}")
    print("[INFO] Falling back to SimpleEmbeddingIndexer")
    from SimpleEmbeddingIndexer import SimpleEmbeddingIndexer as EmbeddingIndexer

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "codellama:instruct"

class KotlinTestGenerator:
    def __init__(self, source_dir: str, test_dir: str, llm_client: LLMClient, indexer: EmbeddingIndexer):
        self.source_dir = source_dir
        self.test_dir = test_dir
        self.llm_client = llm_client
        self.indexer = indexer

        os.makedirs(self.test_dir, exist_ok=True)

    def extract_class_name(self, code: str) -> Optional[str]:
        # Remove comments to avoid false matches
        code_without_comments = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code_without_comments = re.sub(r'//.*', '', code_without_comments)
        
        pattern = re.compile(r'^\s*(data\s+)?(class|object)\s+([A-Za-z0-9_]+)', re.MULTILINE)
        matches = pattern.findall(code_without_comments)
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
        
        # If all are data classes, return the last one found
        return matches[-1][2]
    
    def clean_generated_code(self, generated_code: str) -> str:
        """Clean up the generated code by removing markdown formatting and extra text."""
        # Remove markdown code blocks
        code = generated_code.strip()
        if code.startswith("```kotlin"):
            code = code[9:]  # Remove ```kotlin
        elif code.startswith("```"):
            code = code[3:]   # Remove ```
        if code.endswith("```"):
            code = code[:-3]  # Remove closing ```
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        return code

    def process_file(self, filepath: str):
        print(f"[INFO] Processing file: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read file {filepath}: {e}")
            return

        class_name = self.extract_class_name(file_content)
        if not class_name:
            print(f"[WARN] No class found in {filepath}, skipping.")
            return

        # Retrieve similar tests
        try:
            similar_tests = self.indexer.retrieve_similar(file_content)
        except Exception as e:
            print(f"[WARN] Failed to retrieve similar tests: {e}")
            similar_tests = []

        # Build generation prompt
        gen_prompt = PromptBuilder.build_generation_prompt(class_name, file_content, similar_tests)
        generated_test = self.llm_client.generate(gen_prompt)

        if not generated_test:
            print(f"[ERROR] Failed to generate test for {class_name}")
            return

        # Build accuracy check prompt
        accuracy_prompt = PromptBuilder.generate_accurate_prompt(file_content, generated_test)
        feedback = self.llm_client.generate(accuracy_prompt)

        # Clean the generated test code
        clean_test_code = self.clean_generated_code(generated_test)
        clean_feedback = self.clean_generated_code(feedback)

        # Save generated test code
        filename = os.path.basename(filepath)
        base_name = filename.replace(".kt", "")
        test_filename = f"{class_name}Test.kt"
        test_path = os.path.join(self.test_dir, test_filename)

        try:
            # Ensure the test directory exists
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
        print(f"[INFO] Scanning source directory: {self.source_dir}")
        for root, dirs, files in os.walk(self.source_dir):
            if 'testcase--datastore' in dirs:
                dirs.remove('testcase--datastore')
            for file in files:
                if file.endswith(".kt"):
                    full_path = os.path.join(root, file)
                    self.process_file(full_path)

# === Main execution ===
if __name__ == "__main__":
    print("[INFO] Starting TestCaseGenerator...")
    
    source_dir = sys.argv[1] if len(sys.argv) > 1 else ("src/input-src" if os.path.exists("src/input-src") else ".")
    test_dir = "src/output-test" 
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