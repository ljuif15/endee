import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from vector_store import VectorStore
from openai import OpenAI

# Load OpenAI client (API key from environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Load embedding model and vector store
model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore.load("knowledge_base")

# User question
question = input("Ask a question: ")

# Embed question
question_vector = model.encode([question])

# Retrieve relevant documents
results = store.search(question_vector, top_k=3)
context = " ".join([r["metadata"]["text"] for r in results])

# Prompt for RAG
prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""

# Call OpenAI (NEW API)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n🤖 AI Answer:")
print(response.choices[0].message.content)
