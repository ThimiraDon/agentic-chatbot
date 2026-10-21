
from pathlib import Path

import pytest
from langchain_core.documents import Document

import src.rag.ingestion as ingestion_module


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def upsert_documents(
        self,
        documents,
        embeddings,
        namespace,
    ):
        assert len(documents) == len(embeddings)
        assert namespace == "default"
        return len(documents)


@pytest.fixture
def ingestion_service(monkeypatch):
    # Simulate PDF extraction.
    monkeypatch.setattr(
        ingestion_module,
        "load_pdf",
        lambda path: [
            Document(
                page_content="Pump maintenance instructions.",
                metadata={"source": Path(path).name, "page": 0},
            )
        ],
    )

    # Simulate splitting into chunks.
    monkeypatch.setattr(
        ingestion_module,
        "split_documents",
        lambda documents: documents,
    )

    # Replace external services with test doubles.
    monkeypatch.setattr(
        ingestion_module,
        "EmbeddingService",
        FakeEmbeddingService,
    )
    monkeypatch.setattr(
        ingestion_module,
        "PineconeVectorStore",
        FakeVectorStore,
    )

    return ingestion_module.IngestionService()


def test_ingest_pdf_processes_and_stores_document(
    ingestion_service,
    tmp_path,
):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.touch()

    result = ingestion_service.ingest_pdf(pdf_path)

    assert result["source"] == "manual.pdf"
    assert result["pages_loaded"] == 1
    assert result["chunks_created"] == 1
    assert result["chunks_stored"] == 1
    assert result["namespace"] == "default"
