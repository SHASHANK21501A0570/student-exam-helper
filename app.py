from core.pdf_loader import load_pdf
from core.chunking import chunk_text
from core.embeddings import EmbeddingStore
from core.retriever import Retriever

# Load PDF
pdf_path = "data/sample.pdf"
text = load_pdf(pdf_path)

#  Chunk text
chunks = chunk_text(text)

print(f"Chunks created: {len(chunks)}")

# Create embeddings + FAISS index
store = EmbeddingStore()
store.create_embeddings(chunks)


# Build retriever
retriever = Retriever(
    store.get_index(),
    store.get_chunks()
)

# Test query
query = "Explain the main idea of this document"

results = retriever.retrieve(query)

print("\n--- Retrieved Chunks ---")

for i, r in enumerate(results):
    print(f"\nResult {i+1}:\n{r[:300]}")