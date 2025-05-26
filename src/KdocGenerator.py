import os
import requests

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "codellama:instruct"

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
        "Respond only with  Kdocs source code"
    )

    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
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

    return result

def update_kdocs_in_file(filepath):
    print(f"Processing file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        file_content = f.read()

    updated_content = generate_kdoc_for_file(file_content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"Finished updating {filepath}")

def update_kdocs_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".kt"):
                update_kdocs_in_file(os.path.join(root, file))

if __name__ == "__main__":
    project_dir = "src"  # Your Kotlin source folder path
    update_kdocs_in_directory(project_dir)
