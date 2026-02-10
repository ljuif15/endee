import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore("knowledge_base")

with open("data/documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

embeddings = model.encode(documents)

for text, vector in zip(documents, embeddings):
    store.add(vector=vector, metadata={"text": text})

store.save()

print("✅ Documents ingested successfully into vector store.")
