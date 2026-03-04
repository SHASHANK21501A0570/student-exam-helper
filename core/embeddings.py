from importlib.resources import path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
class EmbeddingStore:
    """
    Class handles the embedding generation and faiss index creation
    """
    def __init__(self,model_name:str="BAAI/bge-small-en-v1.5"):
        print("Load embedding model")
        self.model=SentenceTransformer(model_name)
        print("Embedding model loaded successfully")
        self.index=None
        self.chunks=0
    
    def create_embeddings(self,chunks:list[str]):
        """
        Generate embeddings for the given text chunks and create a faiss index.

        Args:
            chunks (list[str]): List of text chunks to be embedded.
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding.")
        print("Generating embeddings for chunks...")

        embeddings=self.model.encode(chunks,convert_to_numpy=True,show_progress_bar=True)
        dimension = embeddings.shape[1]

        print(f"Embedding dimension: {dimension}")

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.chunks = chunks

        print(f"Stored {self.index.ntotal} vectors in FAISS index.")

    def get_index(self):
        return self.index

    def get_chunks(self):
        return self.chunks
    
    def save_index(self, path="vectorstore"):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print("Vector database saved.")

    def load_index(self, path="vectorstore"):
        if not os.path.exists(f"{path}/index.faiss"):
            return False

        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Vector database loaded.")

        return True



