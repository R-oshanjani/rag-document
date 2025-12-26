print("RETRIEVER SCRIPT STARTED")

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

loader = TextLoader("data/company_policy.txt")
documents = loader.load()

print("Loaded documents")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded")

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

print("FAISS index saved at:", os.path.abspath("faiss_index"))
print("RETRIEVER SCRIPT FINISHED")
