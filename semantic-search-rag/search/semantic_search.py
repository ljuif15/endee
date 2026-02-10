import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore.load("knowledge_base")

query = input("Enter your search query: ")

query_vector = model.encode([query])
results = store.search(query_vector, top_k=3)

print("\n🔍 Top Semantic Matches:")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['metadata']['text']}")
