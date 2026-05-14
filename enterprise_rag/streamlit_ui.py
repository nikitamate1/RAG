import streamlit as st
import json, os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_store"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Enterprise RAG", page_icon="🔐", layout="centered")
st.title("🔐 Enterprise Knowledge Assistant")

with open(os.path.join(DATA_DIR, "user_roles.json")) as f:
    role_config = json.load(f)

# build user dropdown
users = {}
for role, info in role_config["roles"].items():
    for uid in info["users"]:
        users[uid] = role

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# ------------------------------------------------------------------
# Role-aware quick questions
# ------------------------------------------------------------------
QUICK_QUESTIONS = {
    "hr_manager": [
        "How many days of annual leave do employees get?",
        "What is the policy on sick leave?",
        "List all employees and their departments.",
        "What are the performance review guidelines?",
    ],
    "engineer": [
        "Which users had failed login attempts?",
        "Show me recent system audit events.",
        "What are the latest compliance findings?",
        "List any critical security incidents in the audit log.",
    ],
    "analyst": [
        "What is the overall compliance status?",
        "Summarise the HR policy for new hires.",
        "What open compliance issues exist?",
        "Give me an overview of the company data sources.",
    ],
    "admin": [
        "What compliance issues are open and who is the compliance officer?",
        "Show all failed login attempts across the system.",
        "List all employees with their roles and salaries.",
        "Give a full summary of HR policy and audit findings.",
    ],
}

# ------------------------------------------------------------------
# Sidebar — login
# ------------------------------------------------------------------
st.sidebar.header("Login")
selected_user = st.sidebar.selectbox("Select User", list(users.keys()))
role = users[selected_user]
st.sidebar.success(f"Role: **{role}**")

allowed_sources = role_config["roles"][role]["allowed_sources"]
st.sidebar.markdown("**Data Access:**")
for src, meta in role_config["source_metadata"].items():
    icon = "✅" if src in allowed_sources else "🔒"
    st.sidebar.markdown(f"{icon} `{meta['file']}`")

# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------
if "question_text" not in st.session_state:
    st.session_state.question_text = ""
if "run_now" not in st.session_state:
    st.session_state.run_now = False

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
st.markdown(f"Logged in as **{selected_user}** with role `{role}`")

# Quick question buttons
st.markdown("#### 💡 Quick Questions")
st.caption("Click any question below to run it instantly:")

quick_qs = QUICK_QUESTIONS.get(role, [])
cols = st.columns(2)
for i, q in enumerate(quick_qs):
    if cols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
        st.session_state.question_text = q  # push into the text box
        st.session_state.run_now = True      # auto-run flag

st.divider()

# Text input — driven by session state
question = st.text_input(
    "Or type your own question, Please copy and paste the question if clicking does not work:",
    value=st.session_state.question_text,
    key="manual_input"
)

# Sync manual edits back to session state
if question != st.session_state.question_text:
    st.session_state.question_text = question
    st.session_state.run_now = False

ask_clicked = st.button("Ask", type="primary")

# Trigger on either Ask button OR quick question click
should_run = (ask_clicked or st.session_state.run_now) and st.session_state.question_text

# Reset the auto-run flag so it doesn't loop
if st.session_state.run_now:
    st.session_state.run_now = False

# ------------------------------------------------------------------
# Query execution
# ------------------------------------------------------------------
if should_run:
    active_question = st.session_state.question_text
    vectorstore = get_vectorstore()

    all_results = vectorstore.similarity_search_with_score(active_question, k=20)
    filtered = []
    for doc, score in all_results:
        doc_roles = doc.metadata.get("access_roles", "").split(",")
        if role in doc_roles or role == "admin":
            filtered.append((doc, score))
        if len(filtered) == 5:
            break

    if not filtered:
        st.error("🔒 Access denied — no relevant data available for your role.")
    else:
        context = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, (doc, _) in enumerate(filtered)]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
        response = llm.invoke([
            SystemMessage(content="Answer using ONLY the context provided. Cite sources inline as [1], [2] etc. Be concise."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {active_question}")
        ])

        st.markdown("### Answer")
        st.write(response.content)

        st.markdown("### Sources")
        for i, (doc, score) in enumerate(filtered):
            clamped = min(score, 2.0)
            relevance_pct = round((1 - clamped / 2.0) * 100, 1)
            relevance_pct = max(relevance_pct, 0)
            st.caption(
                f"[{i+1}] `{doc.metadata['source']}` "
                f"({doc.metadata['data_type']}) — "
                f"**{relevance_pct}% relevance** "
                f"_(L2 distance: {round(score, 3)})_"
            )