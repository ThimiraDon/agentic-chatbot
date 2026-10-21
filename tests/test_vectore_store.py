
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.rag.vector_store as vector_store_module


class FakeIndex:
    """Simulate a Pinecone index for unit testing."""

    def __init__(self):
        self.upsert_calls = []

        # Simulate one matching record returned by Pinecone.
        self.query_response = SimpleNamespace(
            matches=[
                SimpleNamespace(
                    id="chunk-1",
                    score=0.95,
                    metadata={
                        "text": "Pump maintenance instructions.",
                        "source": "manual.pdf",
                        "page": 2,
                    },
                )
            ]
        )

    def upsert(self, vectors, namespace):
        # Record the submitted data so tests can inspect it.
        self.upsert_calls.append(
            {
                "vectors": vectors,
                "namespace": namespace,
            }
        )

    def query(
        self,
        vector,
        top_k,
        namespace,
        include_metadata,
    ):
        # Return our predefined search result.
        return self.query_response


class FakePinecone:
    """Simulate the Pinecone client."""

    def __init__(self, api_key):
        assert api_key == "test-api-key"
        self.fake_index = FakeIndex()

    def Index(self, index_name):
        assert index_name == "test-index"
        return self.fake_index


@pytest.fixture
def store(monkeypatch):
    """Create a vector store connected to a fake Pinecone index."""

    # Supply test credentials without changing the real .env file.
    monkeypatch.setenv("PINECONE_API_KEY", "test-api-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    # Replace the real Pinecone client with our fake client.
    monkeypatch.setattr(
        vector_store_module,
        "Pinecone",
        FakePinecone,
    )

    return vector_store_module.PineconeVectorStore()


def test_upsert_documents_stores_text_and_vector(store):
    """A document and its embedding should be sent to Pinecone."""

    document = Document(
        page_content="Pump maintenance instructions.",
        metadata={
            "source": "manual.pdf",
            "page": 2,
            "start_index": 0,
        },
    )

    count = store.upsert_documents(
        documents=[document],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    assert count == 1

    record = store.index.upsert_calls[0]["vectors"][0]

    assert record["values"] == [0.1, 0.2, 0.3]
    assert record["metadata"]["text"] == document.page_content
    assert record["metadata"]["source"] == "manual.pdf"
    assert record["metadata"]["page"] == 2
    assert record["metadata"]["start_index"] == 0


def test_upsert_documents_rejects_mismatched_lengths(store):
    """Documents and embeddings must have equal lengths."""

    document = Document(page_content="Pump maintenance.")

    with pytest.raises(ValueError, match="must match"):
        store.upsert_documents(
            documents=[document],
            embeddings=[],
        )


def test_upsert_documents_handles_empty_input(store):
    """An empty batch should return zero without contacting Pinecone."""

    result = store.upsert_documents([], [])

    assert result == 0
    assert store.index.upsert_calls == []


def test_upsert_documents_rejects_empty_chunk(store):
    """A chunk containing only whitespace must be rejected."""

    document = Document(page_content="   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        store.upsert_documents(
            documents=[document],
            embeddings=[[0.1, 0.2, 0.3]],
        )


def test_search_returns_matching_chunks(store):
    """Search results should contain text and source metadata."""

    results = store.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["id"] == "chunk-1"
    assert results[0]["text"] == "Pump maintenance instructions."
    assert results[0]["source"] == "manual.pdf"
    assert results[0]["page"] == 2
    assert results[0]["score"] == 0.95


def test_search_rejects_empty_vector(store):
    """An empty query vector should be rejected."""

    with pytest.raises(ValueError, match="cannot be empty"):
        store.search([])


def test_search_rejects_invalid_top_k(store):
    """The requested number of results must be positive."""

    with pytest.raises(ValueError, match="greater than zero"):
        store.search(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=0,
        )


def test_missing_pinecone_configuration(monkeypatch):
    """Initialization should fail when the API key is missing."""

    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    with pytest.raises(
        ValueError,
        match="PINECONE_API_KEY",
    ):
        vector_store_module.PineconeVectorStore()
