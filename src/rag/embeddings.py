
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate vector embeddings from text."""

    def __init__(self) -> None:
        # Load the embedding model.
        # It generates vectors containing 384 numerical values.
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_text(self, text: str) -> list[float]:
        """Convert a single text string into a vector."""

        # Reject empty input.
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        # Generate a normalized embedding.
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        # Convert the NumPy array into a Python list.
        return vector.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Convert multiple text chunks into vectors."""

        # Return an empty list if there are no documents.
        if not texts:
            return []

        # Reject empty text chunks.
        if any(not text.strip() for text in texts):
            raise ValueError(
                "Documents cannot contain empty text."
            )

        # Generate embeddings for all chunks in one batch.
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Return one vector for each text chunk.
        return vectors.tolist()
