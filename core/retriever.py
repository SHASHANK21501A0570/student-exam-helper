from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi


class Retriever:
    """
    Retrieves most relevant text chunks using semantic similarity.
    """

    def __init__(self, index, chunks, model_name="BAAI/bge-small-en-v1.5"):
        self.index = index
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        # Build BM25 corpus
        self.corpus = [chunk["text"] for chunk in chunks]
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5):

        # ----- FAISS semantic search -----
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding, top_k)

        semantic_results = set(indices[0])

        # ----- BM25 keyword search -----
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

        keyword_results = set(bm25_indices)

        combined_indices = list(semantic_results.union(keyword_results))
        combined_indices = combined_indices[:top_k]

        results = [
            (
                self.chunks[i]["text"],
                self.chunks[i]["source"],
                int(i)
            )
            for i in combined_indices if i != -1
        ]

        return results