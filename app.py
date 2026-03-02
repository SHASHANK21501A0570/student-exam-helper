from core.pdf_loader import load_pdf
from core.chunking import chunk_text
from core.embeddings import EmbeddingStore
from core.retriever import Retriever
from core.llm_loader import LLM


# -----------------------------
# 1️⃣ Setup (runs once)
# -----------------------------

print("Loading document...")

pdf_path = "data/sample.pdf"

text = load_pdf(pdf_path)
chunks = chunk_text(text)

store = EmbeddingStore()
store.create_embeddings(chunks)

retriever = Retriever(
    store.get_index(),
    store.get_chunks()
)

llm = LLM()

print("\n📘 Study Copilot Ready!")
print("Type 'exit' to quit.\n")


# -----------------------------
# 2️⃣ Interactive Loop
# -----------------------------

while True:
    query = input("🧑 You: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Retrieve context
    results = retriever.retrieve(query)
    context = "\n\n".join([r[0] for r in results])

    # Build prompt
    prompt = f"""
You are a helpful study assistant.

Answer ONLY using the provided context.
If the answer is not in context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer
    answer = llm.generate(prompt)

    print("\n🤖 Copilot:\n")
    print(answer)
    print("\n" + "-"*50 + "\n")
    print("\n📚 Sources:")

    for _, idx in results:
        print(f"- Chunk {idx}")