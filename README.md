# Implementing RAG with LangChain
Retrieval Augmented Generation (RAG) system using LangChain, Google Gemini, and Pinecone.

## Overview
This code builds a complete RAG pipeline:
1. **Document Ingestion** - Load, chunk, embed, and store documents in a vector database
2. **Naive RAG** - Implement a basic retrieval chain using manual function calls
3. **LCEL RAG** - Refactor to use LangChain Expression Language for a cleaner, more powerful approach

## Technologies

- **LangChain** - Framework for building LLM applications
- **Google Gemini** - Embeddings and chat completions
- **Pinecone** - Vector database for similarity search
- **Python** - 3.12+
