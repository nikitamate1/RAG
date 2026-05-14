# 🔐 Enterprise RAG Intelligence System

A production-style **Retrieval-Augmented Generation (RAG)** system built for enterprise use cases — where not everyone should see everything. This project demonstrates how to combine vector search, LLMs, and role-based access control into a single, working AI assistant.

---

## 💡 What This Actually Does

Most RAG demos just dump all your data into a vector store and let anyone ask anything. This one doesn't.

This system ingests multiple company data sources — HR policies, employee records, system audit logs, compliance reports — and enforces **role-based access control (RBAC) at retrieval time**. That means an analyst can't accidentally (or intentionally) pull up audit logs they're not cleared for, even if they ask the right question. The access control isn't a UI trick — it's baked into the vector search itself.

On top of that, every answer the LLM generates is **grounded in retrieved context only**, with inline citations so you always know which document the answer came from.

---

## 🏗️ Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (runs locally, no API cost) |
| Vector Store | ChromaDB (persistent, local) |
| LLM | Google Gemini 2.5 Flash via LangChain |
| Access Control | Metadata-filtered retrieval (custom RBAC logic) |
| Frontend | Streamlit |
| Orchestration | LangChain Core |

---

## 👥 Roles & Access

| Role | What They Can See |
|---|---|
| `engineer` | Audit logs, compliance reports |
| `hr_manager` | HR policy, employee records |
| `analyst` | HR policy, compliance reports |
| `admin` | Everything |

Test users: `EMP001` (engineer), `EMP003` (analyst), `EMP009` (hr_manager), `EMP010` (admin)

---

## 🚀 Setup & Run

### 1. Clone / unzip the project
```
cd enterprise_rag
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
Create a `.env` file in the root:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at → https://aistudio.google.com/app/apikey

### 5. Ingest the data (run once)
```bash
python ingest.py
```
This chunks all data files, embeds them locally, and stores them in ChromaDB with role metadata attached.

### 6. Launch the app
```bash
streamlit run streamlit_ui.py
```

---

## ✨ Features

- **RBAC at retrieval level** — access control isn't just cosmetic, it filters which chunks the LLM ever sees
- **Inline citations** — every answer references the exact source chunks it used
- **Role-aware quick questions** — the UI suggests relevant questions based on who's logged in
- **Local embeddings** — no third-party embedding API calls, runs fully on CPU
- **Multi-format ingestion** — handles `.txt`, `.csv`, and `.json` data sources out of the box
- **CLI mode** — run `python main.py` for a terminal-based interactive session without Streamlit

---

## 📁 Project Structure

```
enterprise_rag/
├── data/
│   ├── compliance_report.txt
│   ├── employees.csv
│   ├── hr_policy.txt
│   ├── system_audit_log.json
│   └── user_roles.json          # defines roles, users, and access rules
├── chroma_store/                # auto-generated after ingest.py
├── ingest.py                    # loads, chunks, embeds, and stores data
├── query_engine.py              # RBAC retrieval + LLM generation logic
├── main.py                      # CLI interface
├── streamlit_ui.py              # web UI
├── requirements.txt
└── .env                         # your API key (never commit this)
```

---

Built with LangChain, ChromaDB, and Gemini — focused on making RAG actually safe and useful in a real enterprise context, not just a demo.
