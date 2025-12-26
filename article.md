Building a Retrieval-Augmented Generation (RAG) System for Document Question Answering
Abstract

Large Language Models (LLMs) are capable of generating fluent and human-like text, but they suffer from two major limitations: hallucination and lack of access to private or domain-specific data. These issues make standalone LLMs unreliable for enterprise use cases such as internal documentation search or HR policy assistants.

Retrieval-Augmented Generation (RAG) addresses these problems by combining document retrieval with language generation. Instead of generating answers purely from model memory, RAG grounds responses in retrieved documents.

This article provides a complete, end-to-end explanation of building a RAG-based document question-answering system, covering architecture, implementation steps, results, and limitations.

1. Problem Statement

Traditional LLM usage has the following issues:

LLMs may confidently generate incorrect information (hallucination)

They cannot access private documents such as company policies

Answers are not verifiable against source data

For real-world applications—HR assistants, internal knowledge bases, or compliance systems—this behavior is unacceptable. We need a system where answers are grounded in trusted documents.

2. Solution Overview: What is RAG?

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines:

Retrieval: Fetching relevant documents using vector similarity search

Generation: Producing answers using an LLM conditioned on retrieved text

Instead of relying solely on model parameters, the LLM is provided with context at query time, which significantly reduces hallucination and improves factual accuracy.

3. End-to-End System Architecture
Picture 1: High-Level RAG Architecture
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


Explanation:

The user asks a question

The question is converted into a vector embedding

The vector database finds similar document embeddings

Relevant document text is retrieved

The final answer is generated using retrieved content

This ensures that answers are based on documents, not guesses.

4. Step-by-Step Implementation
Step 1: Preparing the Dataset

For this project, a small text document containing company policy information is used.
This simulates real enterprise data such as HR policies or internal guidelines.

Example content:

Employee leave policy

Sick leave rules

Carry-forward limitations

Step 2: Document Ingestion

The first step is loading the document into the system.

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/company_policy.txt")
documents = loader.load()


This converts raw text into structured document objects that can be processed further.

Step 3: Generating Embeddings

Documents are converted into numerical vectors using a pre-trained sentence embedding model.

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


Embeddings capture semantic meaning, enabling similarity search based on meaning rather than keywords.

Step 4: Storing Vectors in FAISS

The embeddings are stored in a FAISS vector database.

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")


FAISS allows fast and scalable similarity search, even for large datasets.

Step 5: Retrieving Relevant Documents

At query time, the system retrieves the most relevant document chunks.

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("How many leave days are allowed?")


These retrieved documents serve as the ground truth context for answering the question.

5. Query-Time Flow (Detailed)
Picture 2: Query-Time Execution Flow
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


This flow shows how retrieval and generation work together during inference.

6. Results

When the question below is asked:

“How many leave days are allowed?”

The system retrieves the correct document content:

Employees are entitled to 20 days of paid leave per year.


This confirms:

Retrieval works correctly

Answers are grounded in documents

Hallucination is avoided

7. Comparison: With vs Without RAG
Without RAG

Generic or incorrect answers

No access to private documents

High hallucination risk

With RAG

Answers based on real documents

Reduced hallucination

Responses are verifiable

This comparison highlights why RAG is essential for reliable GenAI systems.

8. Limitations

Despite its advantages, RAG has limitations:

Retrieval quality directly affects answer quality

Embeddings may miss subtle semantic nuances

Vector search adds latency

Poor document chunking can reduce accuracy

These issues can be mitigated with better chunking strategies, improved embeddings, and system optimization.

9. Conclusion

This project demonstrates an end-to-end implementation of a Retrieval-Augmented Generation system for document question answering. By combining vector-based retrieval with language generation, RAG enables LLMs to work reliably with private and domain-specific data while significantly reducing hallucination.

RAG is a practical and scalable solution for real-world enterprise applications such as internal knowledge assistants, HR policy bots, and document search systems.