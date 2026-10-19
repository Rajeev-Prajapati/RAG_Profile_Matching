
import hashlib
from pathlib import Path

import chromadb
import numpy as np


class ChromaStore:

    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="resume_chunks"
    ):
        self.project_root = Path(__file__).resolve().parent.parent

        db_path = Path(persist_directory)

        if not db_path.is_absolute():
            db_path = self.project_root / db_path

        self.db_path = db_path.resolve()

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "embedding_model": (
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            }
        )

        print(f"ChromaDB location: {self.db_path}")
        print(f"Collection: {collection_name}")

    def generate_chunk_id(self, chunk):
        """
        Create a deterministic ID using resume path
        and chunk position.
        """

        resume_path = str(
            Path(chunk["resume_path"]).resolve()
        )

        chunk_index = chunk["metadata"]["chunk_index"]

        unique_key = f"{resume_path}|{chunk_index}"

        return hashlib.sha256(
            unique_key.encode("utf-8")
        ).hexdigest()

    
    def prepare_metadata(self, chunk):

        source = chunk.get("metadata", {})

        skills = source.get("skills") or []

        if isinstance(skills, list):
            skills = ", ".join(skills)

        metadata = {
            "candidate_name": source.get("candidate_name") or "Unknown",
            "skills": str(skills),
            "education": source.get("education") or "Unknown",
            "section": chunk["section"],
            "chunk_index": int(source["chunk_index"]),
            "resume_path": str(chunk["resume_path"])
        }

        experience = source.get("experience_years")

        if experience is not None:
            metadata["experience_years"] = float(experience)

        return metadata


    def store_chunks(self, chunks, embeddings):
        """
        Insert or update resume chunks and embeddings.
        """

        if not chunks:
            print("No chunks to store.")
            return 0

        vectors = np.asarray(embeddings)

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array"
            )

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks and embeddings must match"
            )

        if vectors.shape[1] != 384:
            raise ValueError(
                "Expected 384-dimensional embeddings"
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            chunk_id = self.generate_chunk_id(chunk)

            ids.append(chunk_id)

            documents.append(
                f"Section: {chunk['section']}\n"
                f"{chunk['text']}"
            )

            metadatas.append(
                self.prepare_metadata(chunk)
            )

        # Prevent accidental duplicate IDs in the same batch
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk IDs found in the batch"
            )

        # Upsert updates an existing ID or inserts a new one
        batch_size = 100

        for start in range(0, len(ids), batch_size):

            end = start + batch_size

            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end].tolist()
            )

        print(f"Stored/updated {len(ids)} chunks.")

        return len(ids)

    def count_chunks(self):
        """
        Return the number of stored chunks.
        """

        return self.collection.count()

    def get_chunks(self, limit=5):
        """
        Retrieve stored chunks for inspection.
        """

        return self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )

    def get_chunk_with_embedding(self, chunk_id):
        """
        Retrieve one chunk including its vector.
        """

        return self.collection.get(
            ids=[chunk_id],
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

    
    def search_similar_chunks(self, query_embedding, top_k=10):
        """
        Search ChromaDB for the most similar resume chunks.

        Args:
            query_embedding: NumPy array or list of floats
            top_k: Maximum number of chunks to retrieve

        Returns:
            ChromaDB query results
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        query_vector = np.asarray(
            query_embedding,
            dtype=float
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "Query embedding must be a 1D vector"
            )

        if query_vector.shape[0] != 384:
            raise ValueError(
                "Expected a 384-dimensional query embedding"
            )

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results

    
    def replace_resume_chunks(self, chunks, embeddings):
        """
        Replace all indexed chunks for one resume.

        Important:
        This is not an atomic transaction. If the upsert
        fails after deletion, the resume must be reindexed.
        """

        if not chunks:
            raise ValueError(
                "At least one chunk is required"
            )

        resume_paths = {
            str(chunk["resume_path"])
            for chunk in chunks
        }

        if len(resume_paths) != 1:
            raise ValueError(
                "All chunks must belong to the same resume"
            )

        resume_path = next(iter(resume_paths))

        existing = self.collection.get(
            where={"resume_path": resume_path}
        )

        if existing["ids"]:
            self.collection.delete(
                ids=existing["ids"]
            )

        return self.store_chunks(
            chunks=chunks,
            embeddings=embeddings
        )




if __name__ == "__main__":

    store = ChromaStore()

    print("Stored chunks:", store.count_chunks())

    result = store.get_chunks(limit=5)

    for index, chunk_id in enumerate(result["ids"]):

        print("\nChunk ID:", chunk_id)

        print("Metadata:", result["metadatas"][index])

        print("Document:", result["documents"][index][:200])
