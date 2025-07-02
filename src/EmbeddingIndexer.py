import os
import glob
import torch
import faiss
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel


class EmbeddingIndexer:
    def __init__(self, test_dir: str, embedding_model_name: str = "microsoft/codebert-base"):
        self.test_dir = test_dir
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        self.test_cases: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.dimension = 0

        self._load_and_index()

    def _load_and_index(self):
        print("[INFO] Loading test cases for indexing...")
        test_files = glob.glob(os.path.join(self.test_dir, "**/*.kt"), recursive=True)

        for file_path in test_files:
            with open(file_path, "r", encoding="utf-8") as f:
                self.test_cases.append(f.read())

        if not self.test_cases:
            print(f"[WARN] No test cases found in: {self.test_dir}")
            return

        print(f"[INFO] Loaded {len(self.test_cases)} test cases.")
        self.embeddings = self._encode(self.test_cases)
        self.dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings.numpy())
        print("[INFO] FAISS index built successfully.")

    def _encode(self, texts: List[str]) -> torch.Tensor:
        print("[INFO] Generating embeddings...")
        self.model.eval()
        with torch.no_grad():
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**encoded_input)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            return embeddings

    def retrieve_similar(self, code: str, top_k: int = 3) -> List[str]:
        print("[INFO] Retrieving similar test cases...")

        if not self.index:
            print("[ERROR] FAISS index is not initialized.")
            return []

        query_embedding = self._encode([code])
        distances, indices = self.index.search(query_embedding.numpy(), top_k)

        print(f"[INFO] Found top-{top_k} matches.")
        return [self.test_cases[i] for i in indices[0] if i < len(self.test_cases)]
