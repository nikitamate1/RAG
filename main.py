"""
Enterprise RAG Intelligence System - Meridian Technologies Demo
--------------------------------------------------------------
Run order:
  1. Set your OpenAI key: export OPENAI_API_KEY=your_key
  2. Ingest data:         python ingest.py
  3. Run queries:         python query_engine.py

Or run this file for an interactive CLI session.
"""

from query_engine import query_rag, format_response

def main():
    print("\nEnterprise RAG System - Interactive Mode")
    print("Test users: EMP001 (engineer), EMP003 (analyst), EMP009 (hr_manager), EMP010 (admin)")
    print("Type 'exit' to quit\n")

    while True:
        user_id = input("Enter user ID: ").strip()
        if user_id.lower() == "exit":
            break

        question = input("Enter question: ").strip()
        if not question:
            continue

        result = query_rag(user_id, question)
        format_response(result)
        print()


if __name__ == "__main__":
    main()