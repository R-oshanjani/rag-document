# RAG Document Question Answering (GenAI Project)

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline for answering questions using private documents instead of relying on an LLM alone.

The goal of this project is to reduce hallucination by grounding answers in retrieved document content. This setup is suitable for real-world use cases such as internal documentation search, HR policy assistants, and knowledge-base Q&A systems.

## Project Overview

Large Language Models are powerful but unreliable when used without context. They often:
- Hallucinate incorrect answers
- Cannot access private or internal documents

This project solves that problem using **RAG**, which combines:
- **Vector-based retrieval** (FAISS)
- **Semantic embeddings** (Sentence Transformers)
- **Grounded question answering**

## Architecture

The RAG pipeline works as follows:

1. User asks a question  
2. The question is converted into an embedding  
3. FAISS searches for similar document embeddings  
4. Relevant document text is retrieved  
5. The answer is generated based on retrieved content  


