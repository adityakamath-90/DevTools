import requests

class LLMClient:
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0,
                "top_p": 0.8,
            })
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            return ""