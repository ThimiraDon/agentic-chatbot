
import pytest

from langchain_core.documents import Document

from src.rag.text_splitter import split_documents


def test_split_documents_creates_multiple_chunks():
    """Long text should be divided into multiple chunks."""

    # Create a document long enough to require splitting.
    document = Document(
        page_content="Industrial pump maintenance. " * 100,
        metadata={"source": "manual.pdf", "page": 0},
    )

    # Split the document into smaller pieces.
    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    # Confirm that multiple chunks were created.
    assert len(chunks) > 1


def test_split_documents_preserves_metadata():
    """Each chunk should retain its original source metadata."""

    document = Document(
        page_content="Pump maintenance instructions. " * 20,
        metadata={"source": "manual.pdf", "page": 3},
    )

    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    # Every chunk should identify its original PDF and page.
    assert all(
        chunk.metadata["source"] == "manual.pdf"
        for chunk in chunks
    )

    assert all(
        chunk.metadata["page"] == 3
        for chunk in chunks
    )


def test_split_documents_adds_start_index():
    """Each chunk should record its starting character position."""

    document = Document(
        page_content="Pump maintenance instructions. " * 20,
        metadata={"source": "manual.pdf"},
    )

    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    # start_index is enabled in our splitter configuration.
    assert all(
        "start_index" in chunk.metadata
        for chunk in chunks
    )


def test_split_documents_rejects_invalid_chunk_size():
    """A chunk size of zero or less must be rejected."""

    document = Document(page_content="Pump maintenance.")

    with pytest.raises(ValueError):
        split_documents([document], chunk_size=0)


def test_split_documents_rejects_invalid_overlap():
    """Overlap must be non-negative and smaller than chunk size."""

    document = Document(page_content="Pump maintenance.")

    with pytest.raises(ValueError):
        split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=100,
        )


def test_split_documents_returns_empty_list_for_empty_input():
    """An empty input list should produce no chunks."""

    chunks = split_documents([])

    assert chunks == []
