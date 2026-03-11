
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

class PineconeService:
    def __init__(self, namespace="default"):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "financial-docs")
        self.namespace = namespace
        self.pc = Pinecone(api_key=self.api_key)
        self.dense_index = self.pc.Index(self.index_name)
        # Mismo modelo usado en el ETL para generar los embeddings
        model_name = os.getenv("HAYSTACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=2):
        try:
            # Generar embedding del query con el mismo modelo del ETL
            query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()

            # Buscar en Pinecone con query() (vectores locales)
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

