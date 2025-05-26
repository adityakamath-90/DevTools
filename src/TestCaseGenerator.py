import os
import re
import requests

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "codellama:instruct"


def generate_test_stub(class_name: str, file_content: str) -> str:
    prompt = (
        f"You are a senior Kotlin developer writing unit tests.\n"
        f"Generate comprehensive unit tests for the Kotlin class `{class_name}` based on the code below:\n\n"
        f"{file_content}\n\n"
        "Requirements:\n"
        "- Include test functions for typical cases, edge cases, and exceptions.\n"
        "- Use idiomatic Kotlin test style (JUnit or Kotest).\n"
        "- Use assertEquals, assertTrue, assertFailsWith, etc.\n"
        "- Do NOT include explanation or markdown — return pure Kotlin test code only.\n"
        "Respond only with source code"
    )

    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return f"// TODO: Unable to generate tests for {class_name}\n"


def extract_class_name(file_content: str) -> str:
    class_pattern = re.compile(r"class\s+([A-Za-z0-9_]+)")
    match = class_pattern.search(file_content)
    return match.group(1) if match else None


class KotlinTestGenerator:
    def __init__(self, source_dir="src", test_dir="test"):
        self.source_dir = source_dir
        self.test_dir = test_dir
        os.makedirs(test_dir, exist_ok=True)

    def process_file(self, filepath: str):
        print(f"[INFO] Processing file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()

        class_name = extract_class_name(file_content)
        if not class_name:
            print(f"[WARN] No class found in {filepath}, skipping.")
            return

        test_code = generate_test_stub(class_name, file_content)

        # Create test file path
        filename = os.path.basename(filepath)
        test_filename = filename.replace(".kt", f"{class_name}Test.kt")
        test_path = os.path.join(self.test_dir, test_filename)

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code)

        print(f"[✅] Generated test: {test_path}")

    def generate_tests_for_all(self):
        print(f"[INFO] Scanning source directory: {self.source_dir}")
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith(".kt"):
                    full_path = os.path.join(root, file)
                    self.process_file(full_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        src_dir = "src" if os.path.exists("src") else "."

    generator = KotlinTestGenerator(source_dir=src_dir)
    generator.generate_tests_for_all()
