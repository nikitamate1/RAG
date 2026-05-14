import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "./chroma_store"
DATA_DIR = "./data"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

with open(os.path.join(DATA_DIR, "user_roles.json")) as f:
    role_config = json.load(f)

def get_user_role(user_id):
    for role, info in role_config["roles"].items():
        if user_id in info["users"]:
            return role
    return None

def retrieve_with_rbac(query, role, vectorstore, k=5):
    all_results = vectorstore.similarity_search_with_score(query, k=20)
    filtered = []
    for doc, score in all_results:
        allowed_roles = doc.metadata.get("access_roles", "").split(",")
        if role in allowed_roles or role == "admin":
            filtered.append((doc, score))
        if len(filtered) == k:
            break
    return filtered

def build_context(results):
    context_parts = []
    citations = []
    for i, (doc, score) in enumerate(results):
        ref_id = f"[{i+1}]"
        context_parts.append(f"{ref_id} {doc.page_content}")
        citations.append({
            "ref": ref_id,
            "source": doc.metadata["source"],
            "data_type": doc.metadata["data_type"],
            "similarity": round(1 - score, 3)
        })
    return "\n\n".join(context_parts), citations

def query_rag(user_id, question):
    role = get_user_role(user_id)
    if not role:
        return {
            "answer": "Access denied. User not recognised.",
            "citations": [],
            "role": None
        }

    print(f"\nUser: {user_id} | Role: {role}")
    print(f"Question: {question}\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    results = retrieve_with_rbac(question, role, vectorstore)

    if not results:
        return {
            "answer": "No relevant information found within your access permissions.",
            "citations": [],
            "role": role
        }

    context, citations = build_context(results)
    avg_confidence = round(sum(c["similarity"] for c in citations) / len(citations), 3)

    system_prompt = """You are an enterprise knowledge assistant.
Answer the user's question using ONLY the provided context.
Each snippet is labeled [1], [2], etc. Always cite inline.
If context is insufficient, say so. Never make up information."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer with inline citations like [1], [2]:"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "answer": response.content,
        "citations": citations,
        "role": role,
        "confidence": avg_confidence
    }

def format_response(result):
    print("=" * 60)
    print(f"ANSWER  (Role: {result['role']} | Confidence: {result.get('confidence', 'N/A')})")
    print("=" * 60)
    print(result["answer"])
    print("\nSOURCES:")
    for c in result["citations"]:
        print(f"  {c['ref']} {c['source']} ({c['data_type']}) - similarity: {c['similarity']}")
    print("=" * 60)

if __name__ == "__main__":
    # test 1 - hr_manager asking about leave (should work)
    result = query_rag("EMP009", "How many days of annual leave do employees get?")
    format_response(result)

    # test 2 - analyst trying to see audit logs (should be blocked)
    result = query_rag("EMP003", "Show me failed login attempts")
    format_response(result)

    # test 3 - engineer asking about logs (should work)
    result = query_rag("EMP001", "Which users had failed login attempts?")
    format_response(result)

    # test 4 - admin cross-source query (sees everything)
    result = query_rag("EMP010", "What compliance issues are open and who is the compliance officer?")
    format_response(result)