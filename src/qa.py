print("QA SCRIPT STARTED")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings loaded")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Vectorstore loaded")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("Retriever ready")

docs = retriever.invoke(
    "How many leave days are allowed?"
)

print("RETRIEVED DOCUMENTS:")
for d in docs:
    print("-", d.page_content)

print("QA SCRIPT FINISHED")
