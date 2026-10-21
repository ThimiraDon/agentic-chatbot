
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.rag.retrieval as retrieval_module


class FakeEmbeddingService:
    def embed_text(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def search(self, query_embedding, top_k, namespace):
        return [
            {
                "id": "chunk-1",
                "text": "Pump maintenance instructions.",
                "source": "manual.pdf",
                "page": 2,
                "score": 0.95,
            }
        ]


@pytest.fixture
def retrieval_service(monkeypatch):
    monkeypatch.setattr(
        retrieval_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        retrieval_module,
        "PineconeVectorStore",
        FakeVectorStore,
    )

    return retrieval_module.RetrievalService()


def test_retrieve_returns_documents(retrieval_service):
    documents = retrieval_service.retrieve(
        "How do I maintain a pump?"
    )

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Pump maintenance instructions."
    assert documents[0].metadata["source"] == "manual.pdf"
    assert documents[0].metadata["page"] == 2
    assert documents[0].metadata["score"] == 0.95


def test_retrieve_rejects_empty_question(retrieval_service):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieval_service.retrieve("   ")
