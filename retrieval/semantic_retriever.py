
from collections import defaultdict

from embeddings.embedding_service import EmbeddingService
from vector_store.chroma_store import ChromaStore


class SemanticRetriever:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = ChromaStore(
        persist_directory="chroma_db",
        collection_name="synthetic_resume_chunks"
    )

    def search_chunks(self, job_description, top_k=10):
        """
        Convert a job description into an embedding,
        then retrieve similar resume chunks.
        """

        if (
            not isinstance(job_description, str)
            or not job_description.strip()
        ):
            raise ValueError(
                "Job description must be a non-empty string"
            )

        print("\nGenerating job description embedding...")

        query_embedding = (
            self.embedding_service.generate_embedding(
                job_description
            )
        )

        print("Searching ChromaDB...")

        results = self.vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k
        )

        matches = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for index, chunk_id in enumerate(ids):

            distance = float(distances[index])

            similarity = 1.0 - distance

            matches.append({
                "chunk_id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": distance,
                "similarity": similarity
            })

        return matches

    def group_by_candidate(self, matches):
        """
        Group matching chunks by resume path.
        """

        grouped = defaultdict(list)

        for match in matches:

            resume_path = (
                match["metadata"]["resume_path"]
            )

            grouped[resume_path].append(match)

        candidates = []

        for resume_path, chunks in grouped.items():

            # Sort each candidate's chunks by similarity
            chunks.sort(
                key=lambda chunk: chunk["similarity"],
                reverse=True
            )

            best_chunk = chunks[0]
            metadata = best_chunk["metadata"]

            candidates.append({
                "candidate_name": metadata.get(
                    "candidate_name", "Unknown"
                ),
                "resume_path": resume_path,
                "skills": metadata.get("skills", ""),
                "experience_years": metadata.get(
                    "experience_years"
                ),
                "education": metadata.get(
                    "education", "Unknown"
                ),
                "semantic_score": best_chunk["similarity"],
                "relevant_chunks": chunks
            })

        candidates.sort(
            key=lambda candidate: candidate["semantic_score"],
            reverse=True
        )

        return candidates

    
    def search_candidates(
        self,
        job_description,
        top_k=10,
        retrieve_all=True
    ):
        """
        Retrieve chunks and group them by candidate.

        retrieve_all=True:
            Search all indexed chunks.
            Suitable for small assignment datasets.

        retrieve_all=False:
            Retrieve only top_k chunks.
        """

        if retrieve_all:
            top_k = self.vector_store.count_chunks()

        if top_k == 0:
            return []

        matches = self.search_chunks(
            job_description=job_description,
            top_k=top_k
        )

        candidates = self.group_by_candidate(matches)

        return candidates



if __name__ == "__main__":

    job_description = """
    Looking for a Python Backend Developer
    with experience in FastAPI, Django,
    PostgreSQL, Docker, REST APIs,
    and microservices development.
    """

    retriever = SemanticRetriever()

    candidates = retriever.search_candidates(
        job_description,
        top_k=10
    )

    print("\n" + "=" * 55)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 55)

    for index, candidate in enumerate(candidates, start=1):

        print(f"\nRank {index}")
        print("Candidate:", candidate["candidate_name"])
        print("Resume:", candidate["resume_path"])

        print(
            "Semantic similarity:",
            round(candidate["semantic_score"], 4)
        )

        print("Skills:", candidate["skills"])

        print(
            "Experience:",
            candidate["experience_years"]
        )

        print("Relevant excerpts:")

        for chunk in candidate["relevant_chunks"][:3]:

            print(
                f"\n[{chunk['metadata']['section']}] "
                f"Similarity: {chunk['similarity']:.4f}"
            )

            print(chunk["text"][:300])
