from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    """
    Retrieves most relevant text chunks using semantic similarity.
    """

    def __init__(self, index, chunks, model_name="BAAI/bge-small-en-v1.5"):
        self.index = index
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve top_k most relevant chunks for a query.

        Args:
            query (str): user question
            top_k (int): number of results to return

        Returns:
            list[str]: relevant text chunks
        """

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding, top_k)

        results = [self.chunks[i] for i in indices[0]]

        return results