
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        print(f"Loading embedding model: {model_name}")

        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        print("Model loaded successfully!")
        print("Embedding dimension:", self.embedding_dimension)

    def generate_embedding(self, text):
        """
        Generate an embedding for a single text.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding

    def generate_embeddings(self, texts):
        """
        Generate embeddings for multiple texts.
        """

        if not isinstance(texts, list) or not texts:
            raise ValueError("Texts must be a non-empty list")

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("All texts must be non-empty strings")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity.
        """

        vector1 = np.asarray(embedding1, dtype=float)
        vector2 = np.asarray(embedding2, dtype=float)

        if vector1.shape != vector2.shape:
            raise ValueError(
                "Embeddings must have matching dimensions"
            )

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Cannot compare zero vectors")

        similarity = np.dot(vector1, vector2) / (
            norm1 * norm2
        )

        return float(similarity)


if __name__ == "__main__":

    service = EmbeddingService()

    texts = [
        "Developed REST APIs using Python and FastAPI",
        "Built backend services using FastAPI",
        "Worked in financial accounting and taxation"
    ]

    embeddings = service.generate_embeddings(texts)

    print("\nEmbedding shape:", embeddings.shape)

    print("\nFirst 10 embedding values:")
    print(embeddings[0][:10])

    similarity1 = service.calculate_similarity(
        embeddings[0],
        embeddings[1]
    )

    similarity2 = service.calculate_similarity(
        embeddings[0],
        embeddings[2]
    )

    print("\nPython vs Backend similarity:", similarity1)
    print("Python vs Accounting similarity:", similarity2)
