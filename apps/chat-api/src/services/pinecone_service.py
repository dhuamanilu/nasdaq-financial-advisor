
import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone

HF_API_URL = "https://router.huggingface.co/pipeline/feature-extraction/{model}"

class PineconeService:
    def __init__(self, namespace=""):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "financial-docs")
        self.namespace = namespace
        self.pc = Pinecone(api_key=self.api_key)
        self.dense_index = self.pc.Index(self.index_name)
        self.model_name = os.getenv("HAYSTACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.hf_url = HF_API_URL.format(model=self.model_name)

    def _embed_query(self, text):
        """Genera embedding usando la API gratuita de HuggingFace (mismo modelo del ETL)."""
        response = requests.post(
            self.hf_url,
            json={"inputs": text, "options": {"wait_for_model": True}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def search(self, query, top_k=2):
        try:
            # Generar embedding via HuggingFace API (mismo modelo del ETL)
            query_vector = self._embed_query(query)

            # Buscar en Pinecone con query()
            results = self.dense_index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )

            # Extraer contextos de los resultados
            matches = results.get("matches", [])
            if matches:
                contexts = []
                for match in matches:
                    md = match.get("metadata", {})
                    text = md.get("text_preview", "") or md.get("section", "")
                    if text:
                        contexts.append(text)
                return " ".join(contexts) if contexts else "No relevant financial context found."
            else:
                return "No relevant financial context found."

        except Exception as e:
            print(f"Error in Pinecone search: {e}")
            return "Financial context not available due to search service error."


