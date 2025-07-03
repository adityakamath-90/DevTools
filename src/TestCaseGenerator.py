import os
import re
from typing import Optional

from PromptBuilder import PromptBuilder
from EmbeddingIndexer import EmbeddingIndexer
from LLMClient import LLMClient

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
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()

        class_name = self.extract_class_name(file_content)
        if not class_name:
            print(f"[WARN] No class found in {filepath}, skipping.")
            return

        # Retrieve similar tests
        similar_tests = self.indexer.retrieve_similar(file_content)

        # Build generation prompt
        gen_prompt = PromptBuilder.build_generation_prompt(class_name, file_content, similar_tests)
        generated_test = self.llm_client.generate(gen_prompt)

        if not generated_test:
            print(f"[ERROR] Failed to generate output-test for {class_name}")
            return

        # Build accuracy check prompt
        accuracy_prompt = PromptBuilder.generate_accurate_prompt(file_content, generated_test)
        feedback = self.llm_client.generate(accuracy_prompt)

        # Clean the generated test code
        clean_test_code = self.clean_generated_code(generated_test)
        clean_feedback = self.clean_generated_code(feedback)

        # Save generated output-test code
        filename = os.path.basename(filepath)
        base_name = filename.replace(".kt", "")
        test_filename = f"{class_name}Test.kt"
        test_path = os.path.join(self.test_dir, test_filename)

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(clean_test_code)

        print(f"[✅] Generated output-test: {test_path}")
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
    import sys

    source_dir = sys.argv[1] if len(sys.argv) > 1 else ("src/input-src" if os.path.exists("src/input-src") else ".")
    test_dir = "src/output-test"
    existing_tests_dir = "src/testcase--datastore"
    llm_client = LLMClient(api_url=OLLAMA_API_URL, model_name=MODEL_NAME)
    indexer = EmbeddingIndexer(test_dir=existing_tests_dir)
    generator = KotlinTestGenerator(source_dir=source_dir, test_dir=test_dir, llm_client=llm_client, indexer=indexer)
    generator.generate_tests_for_all()