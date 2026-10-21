
import pytest

import src.rag.embeddings as embeddings_module


class FakeArray:
    """Mimic the tolist() method used by model output."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSentenceTransformer:
    """Fake embedding model that avoids downloading model files."""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        # Return predictable vectors for testing.
        if isinstance(texts, str):
            return FakeArray([0.1, 0.2, 0.3])

        return FakeArray(
            [[0.1, 0.2, 0.3] for _ in texts]
        )


@pytest.fixture
def embedding_service(monkeypatch):
    # Replace the real model with our fake model.
    monkeypatch.setattr(
        embeddings_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    return embeddings_module.EmbeddingService()


def test_embed_text_returns_vector(embedding_service):
    vector = embedding_service.embed_text("Pump maintenance")

    assert vector == [0.1, 0.2, 0.3]


def test_embed_text_rejects_empty_text(embedding_service):
    with pytest.raises(ValueError, match="Text cannot be empty"):
        embedding_service.embed_text("   ")


def test_embed_documents_returns_one_vector_per_text(
    embedding_service,
):
    vectors = embedding_service.embed_documents(
        ["Pump maintenance", "Motor inspection"]
    )

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.1, 0.2, 0.3]


def test_embed_documents_returns_empty_list(embedding_service):
    assert embedding_service.embed_documents([]) == []


def test_embed_documents_rejects_empty_chunk(embedding_service):
    with pytest.raises(
        ValueError,
        match="Documents cannot contain empty text",
    ):
        embedding_service.embed_documents(
            ["Pump maintenance", "   "]
        )
