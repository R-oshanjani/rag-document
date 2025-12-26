print("INGEST SCRIPT STARTED")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/company_policy.txt")
documents = loader.load()

print("DOCUMENTS LOADED:")
print(documents)

print("INGEST SCRIPT FINISHED")
