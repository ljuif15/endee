# Semantic Search & RAG using Endee

## Overview
This project demonstrates a practical AI/ML application where vector search is the core
mechanism. It implements Semantic Search and Retrieval Augmented Generation (RAG) on top
of the Endee repository.

## Use Case
- Semantic document search
- AI-powered question answering
- Vector-based knowledge retrieval

## Architecture
Documents → Embeddings → Vector Store → Similarity Search → LLM Answer

## Tech Stack
- Python
- Sentence Transformers
- Vector similarity search
- OpenAI GPT (for RAG)

## Setup
```bash
git clone https://github.com/ljuif15/endee
cd endee/semantic-search-rag
pip install -r requirements.txt
python ingestion/ingest_documents.py
python main.py
