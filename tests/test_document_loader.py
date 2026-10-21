
import pytest
from langchain_core.documents import Document

import src.rag.document_loader as document_loader


def test_load_pdf_returns_readable_pages(tmp_path, monkeypatch):
    """Readable pages should be returned with normalized source metadata."""

    # Create a temporary file so the loader's file checks pass.
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.touch()

    # Simulate the pages returned by PyPDFLoader.
    fake_documents = [
        Document(
            page_content="Pump maintenance instructions.",
            metadata={"source": "old/path/manual.pdf", "page": 0},
        ),
        Document(
            page_content="   ",
            metadata={"source": "old/path/manual.pdf", "page": 1},
        ),
    ]

    # Replace PyPDFLoader with a controlled test double.
    class FakePyPDFLoader:
        def __init__(self, file_path):
            assert file_path == str(pdf_path)

        def load(self):
            return fake_documents

    monkeypatch.setattr(
        document_loader,
        "PyPDFLoader",
        FakePyPDFLoader,
    )

    # Run the function being tested.
    documents = document_loader.load_pdf(pdf_path)

    # The blank page should be excluded.
    assert len(documents) == 1

    # The extracted text and metadata should be preserved correctly.
    assert documents[0].page_content == "Pump maintenance instructions."
    assert documents[0].metadata["source"] == "manual.pdf"
    assert documents[0].metadata["page"] == 0


def test_load_pdf_rejects_missing_file(tmp_path):
    """A nonexistent PDF path should raise FileNotFoundError."""

    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        document_loader.load_pdf(missing_path)


def test_load_pdf_rejects_non_pdf_file(tmp_path):
    """A file with the wrong extension should be rejected."""

    text_path = tmp_path / "manual.txt"
    text_path.touch()

    with pytest.raises(ValueError, match="must be a PDF"):
        document_loader.load_pdf(text_path)


def test_load_pdf_rejects_pdf_without_extractable_text(
    tmp_path,
    monkeypatch,
):
    """A PDF with no extractable text should raise ValueError."""

    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.touch()

    # Simulate a PDF loader that cannot extract any readable text.
    class FakePyPDFLoader:
        def __init__(self, file_path):
            pass

        def load(self):
            return [
                Document(page_content="   ", metadata={"page": 0})
            ]

    monkeypatch.setattr(
        document_loader,
        "PyPDFLoader",
        FakePyPDFLoader,
    )

    with pytest.raises(ValueError, match="No extractable text"):
        document_loader.load_pdf(pdf_path)
