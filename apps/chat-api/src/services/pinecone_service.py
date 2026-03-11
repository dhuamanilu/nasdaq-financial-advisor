
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from fastembed import TextEmbedding

class PineconeService:
    def __init__(self, namespace=""):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "financial-docs")
        self.namespace = namespace
        self.pc = Pinecone(api_key=self.api_key)
        self.dense_index = self.pc.Index(self.index_name)
        # Mismo modelo usado en el ETL (all-MiniLM-L6-v2, 384 dims)
        self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def _embed_query(self, text):
        """Genera embedding local con ONNX (mismo modelo del ETL, sin torch)."""
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def search(self, query, top_k=2):
        try:
            query_vector = self._embed_query(query)

            results = self.dense_index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )

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
