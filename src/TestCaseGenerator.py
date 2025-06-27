import os
import re
from typing import Optional

import PromptBuilder
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
        pattern = re.compile(r"(class|object)\s+([A-Za-z0-9_]+)")
        match = pattern.search(code)
        return match.group(1) if match else None

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
        gen_prompt = PromptBuilder.PromptBuilder.build_generation_prompt(class_name, file_content, similar_tests)
        generated_test = self.llm_client.generate(gen_prompt)

        if not generated_test:
            print(f"[ERROR] Failed to generate test for {class_name}")
            return

        # Build accuracy check prompt
        accuracy_prompt = PromptBuilder.PromptBuilder.generate_accurate_prompt(file_content, generated_test)
        feedback = self.llm_client.generate(accuracy_prompt)

        # Save generated test code
        filename = os.path.basename(filepath)
        test_filename = filename.replace(".kt", f"{class_name}Test.kt")
        test_path = os.path.join(self.test_dir, test_filename)

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(generated_test)

        print(f"[✅] Generated test: {test_path}")
        print(f"[🔍] Accuracy & Reliability feedback:\n{feedback}\n")

    def generate_tests_for_all(self):
        print(f"[INFO] Scanning source directory: {self.source_dir}")
        for root, dirs, files in os.walk(self.source_dir):
            if 'datastore' in dirs:
                dirs.remove('datastore')
            for file in files:
                if file.endswith(".kt"):
                    full_path = os.path.join(root, file)
                    self.process_file(full_path)

# === Main execution ===
if __name__ == "__main__":
    import sys

    source_dir = sys.argv[1] if len(sys.argv) > 1 else ("src" if os.path.exists("") else ".")
    test_dir = "test"
    existing_tests_dir = "datastore"
    llm_client = LLMClient(api_url=OLLAMA_API_URL, model_name=MODEL_NAME)
    indexer = EmbeddingIndexer(test_dir=existing_tests_dir)
    generator = KotlinTestGenerator(source_dir=source_dir, test_dir=test_dir, llm_client=llm_client, indexer=indexer)
    generator.generate_tests_for_all()