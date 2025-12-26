# Building a Retrieval-Augmented Generation (RAG) System for Document Question Answering

## Abstract

Large Language Models (LLMs) are capable of generating fluent and human-like text, but they suffer from two major limitations: **hallucination** and **lack of access to private or domain-specific data**. These limitations make standalone LLMs unreliable for enterprise use cases such as internal documentation search or HR policy assistants.

Retrieval-Augmented Generation (RAG) addresses these issues by combining **document retrieval** with **language generation**. Instead of generating answers purely from model memory, RAG grounds responses in retrieved documents.

This article presents a complete, end-to-end explanation of building a RAG-based document question-answering system, covering system architecture, implementation steps, query-time execution, results, and limitations.


## 1. Problem Statement

Traditional LLM-based systems exhibit the following problems:

- LLMs may confidently generate incorrect information (**hallucination**)
- They cannot access private or proprietary documents
- Generated answers are not verifiable against source data

For real-world applications such as HR assistants, internal knowledge bases, and compliance systems, this behavior is unacceptable. We need a system where answers are **grounded in trusted documents**, not probabilistic guesses.

---

## 2. Solution Overview: What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines:

- **Retrieval**: Fetching relevant documents using vector similarity search  
- **Generation**: Producing answers using a Large Language Model conditioned on retrieved text  

Instead of relying solely on model parameters, RAG injects external context at query time. This significantly reduces hallucination and improves factual accuracy.

## 3. End-to-End System Architecture

### High-Level RAG Architecture

```mermaid
flowchart LR
    A[User Question]
    B[Embedding Model]
    C[Vector Database (FAISS)]
    D[Retrieved Documents]
    E[Language Model]
    F[Final Answer]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
Architecture Explanation
The user submits a question

The question is converted into a vector embedding

The vector database performs similarity search

Relevant document chunks are retrieved

The LLM generates a grounded answer using retrieved content

This ensures answers are document-backed, not hallucinated.

4. Step-by-Step Implementation
Step 1: Dataset Preparation
For this project, a small text document containing company policy information is used.
This simulates real enterprise data such as HR policies or internal guidelines.

Example content includes:

Employee leave policy

Sick leave rules

Carry-forward limitations

Step 2: Document Ingestion
python
Copy code
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/company_policy.txt")
documents = loader.load()
Raw text is converted into structured document objects that can be processed further.

Step 3: Generating Embeddings
python
Copy code
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Embeddings capture semantic meaning, enabling similarity search based on meaning rather than keywords.

Step 4: Storing Vectors in FAISS
python
Copy code
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")
FAISS enables fast and scalable vector similarity search, making it suitable for production systems.

Step 5: Retrieving Relevant Documents
python
Copy code
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("How many leave days are allowed?")
The retrieved documents act as the ground truth context for answer generation.

5. Query-Time Execution Flow
mermaid
Copy code
sequenceDiagram
    participant User
    participant Retriever
    participant VectorDB
    participant LLM

    User->>Retriever: Ask Question
    Retriever->>VectorDB: Vector Similarity Search
    VectorDB-->>Retriever: Relevant Documents
    Retriever->>LLM: Context + Question
    LLM-->>User: Grounded Answer
Retrieval and generation operate together during inference to ensure factual grounding.

6. Results
Query:

How many leave days are allowed?

Retrieved Document Content:

Employees are entitled to 20 days of paid leave per year.

Observations
Correct document retrieval

Answer grounded in source data

Hallucination avoided

This validates the effectiveness of the RAG pipeline.

7. Comparison: With vs Without RAG
Without RAG
Generic or incorrect answers

No access to private documents

High hallucination risk

With RAG
Answers grounded in real documents

Reduced hallucination

Verifiable responses

RAG is essential for building reliable enterprise GenAI systems.

8. Limitations
Despite its advantages, RAG has limitations:

Retrieval quality directly affects answer quality

Embeddings may miss subtle semantic nuances

Vector search introduces additional latency

Poor document chunking reduces retrieval accuracy

These issues can be mitigated through improved chunking strategies, better embeddings, and system-level optimizations.

9. Conclusion
This article demonstrated a complete, end-to-end implementation of a Retrieval-Augmented Generation system for document question answering. By combining vector-based retrieval with language generation, RAG enables LLMs to operate reliably on private and domain-specific data while significantly reducing hallucination.

RAG is a practical and scalable solution for real-world applications such as internal knowledge assistants, HR policy bots, and enterprise document search systems.
