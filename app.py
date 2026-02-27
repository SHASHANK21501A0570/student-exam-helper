from core.pdf_loader import load_pdf
from core.chunking import chunk_text
from core.embeddings import EmbeddingStore

pdf_path = "data/sample.pdf"

# 1️⃣ Load PDF
text = load_pdf(pdf_path)

# 2️⃣ Chunk
chunks = chunk_text(text)

print(f"Chunks created: {len(chunks)}")

# 3️⃣ Create embeddings + FAISS
store = EmbeddingStore()
store.create_embeddings(chunks)

print("\nFAISS index ready!")