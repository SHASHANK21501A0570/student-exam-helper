from core.pdf_loader import load_pdf
from core.chunking import chunk_text
from core.embeddings import EmbeddingStore
from core.retriever import Retriever
from core.llm_loader import LLM


print("Loading document...")

pdf_path = "data/sample.pdf"


text = load_pdf(pdf_path)
chunks = chunk_text(text)

store = EmbeddingStore()

if not store.load_index():
    print("Creating embeddings...")
    store.create_embeddings(chunks)
    store.save_index()


retriever = Retriever(
store.get_index(),
store.get_chunks()
)


llm = LLM()



chat_history = []

print("\n📘 Study Copilot Ready!")
print("Type 'exit' to quit.\n")


while True:


    query = input("🧑 You: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    history_text = ""

    for role, message in chat_history[-4:]:
        history_text += f"{role.upper()}: {message}\n"

    rewritten_query = llm.rewrite_query(query, history_text)

    print(f"\n🔎 Rewritten Query: {rewritten_query}\n")

    results = retriever.retrieve(rewritten_query, top_k=5)

    context = "\n\n".join([r[0] for r in results])

    prompt = f"""
```

You are a helpful academic study assistant.

Use the conversation history and provided context to answer the question.

Rules:

* Answer ONLY using the provided context.
* If the answer is not in the context, say: "The document does not provide this information."
* Be concise and clear.

Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.generate(prompt)

    print("\n🤖 Copilot:\n")
    print(answer)

    print("\n📚 Sources:")

    for _, idx in results:
        print(f"- Chunk {idx}")

    print("\n" + "-"*50 + "\n")

    chat_history.append(("user", query))
    chat_history.append(("assistant", answer))

