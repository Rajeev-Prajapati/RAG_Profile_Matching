
from vector_store.chroma_store import ChromaStore


def main():

    store = ChromaStore()

    result = store.get_chunks(limit=1)

    if not result["ids"]:
        print("No chunks found.")
        return

    chunk_id = result["ids"][0]

    stored = store.get_chunk_with_embedding(chunk_id)

    embedding = stored["embeddings"][0]

    print("Chunk ID:", chunk_id)
    print("Embedding dimension:", len(embedding))
    print("Candidate:", stored["metadatas"][0]["candidate_name"])
    print("Section:", stored["metadatas"][0]["section"])
    print("Text:", stored["documents"][0][:200])

    assert len(embedding) == 384

    print("\nChromaDB verification successful!")


if __name__ == "__main__":
    main()
