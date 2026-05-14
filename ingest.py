import json
import csv
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
load_dotenv()

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_store"

with open(os.path.join(DATA_DIR, "user_roles.json")) as f:
    role_config = json.load(f)

def get_roles_for_source(source_key):
    allowed = []
    for role, info in role_config["roles"].items():
        if source_key in info["allowed_sources"]:
            allowed.append(role)
    return allowed

def load_txt(filename, source_key):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        content = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source": filename,
                "source_key": source_key,
                "chunk_index": i,
                "data_type": "text",
                "access_roles": ",".join(get_roles_for_source(source_key))
            }
        ))
    return docs

def load_csv(filename, source_key):
    path = os.path.join(DATA_DIR, filename)
    docs = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for i, row in enumerate(rows):
        text = " | ".join([f"{k}: {v}" for k, v in row.items()])
        docs.append(Document(
            page_content=text,
            metadata={
                "source": filename,
                "source_key": source_key,
                "chunk_index": i,
                "data_type": "csv",
                "access_roles": ",".join(get_roles_for_source(source_key))
            }
        ))
    return docs

def load_json(filename, source_key):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        entries = json.load(f)
    docs = []
    for i, entry in enumerate(entries):
        text = " | ".join([f"{k}: {v}" for k, v in entry.items()])
        docs.append(Document(
            page_content=text,
            metadata={
                "source": filename,
                "source_key": source_key,
                "chunk_index": i,
                "data_type": "json",
                "access_roles": ",".join(get_roles_for_source(source_key))
            }
        ))
    return docs

if __name__ == "__main__":
    all_docs = []

    print("Loading hr_policy.txt...")
    all_docs += load_txt("hr_policy.txt", "hr_policy")

    print("Loading employees.csv...")
    all_docs += load_csv("employees.csv", "employees")

    print("Loading system_audit_log.json...")
    all_docs += load_json("system_audit_log.json", "system_audit_log")

    print("Loading compliance_report.txt...")
    all_docs += load_txt("compliance_report.txt", "compliance_report")

    print(f"\nTotal chunks created: {len(all_docs)}")
    print("Sample chunk:")
    print(f"  source: {all_docs[0].metadata['source']}")
    print(f"  roles:  {all_docs[0].metadata['access_roles']}")
    print(f"  text:   {all_docs[0].page_content[:80]}...")

    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"\nDone! {len(all_docs)} chunks stored in {CHROMA_DIR}")