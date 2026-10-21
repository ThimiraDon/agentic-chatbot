from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF and return its pages as LangChain Documents.

    Each Document contains:
    - page_content: Extracted text from a PDF page.
    - metadata: Source filename and page information.
    """

    # Convert the supplied path into a Path object.
    pdf_path = Path(file_path)

    # Make sure the PDF exists before attempting to load it.
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # Prevent passing an unsupported file type.
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    # Create a PyPDFLoader instance for the specified PDF.
    loader = PyPDFLoader(str(pdf_path))

    # Extract the text and metadata from each PDF page.
    # PyPDFLoader returns a list of LangChain Document objects.
    documents = loader.load()

    # Remove pages that contain no extractable text.
    documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    # Stop if no readable text was extracted.
    # Scanned PDFs may require an OCR pipeline.
    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be scanned and require OCR."
        )

    # Store a consistent source filename in each page's metadata.
    # PyPDFLoader already supplies page metadata.
    for document in documents:
        document.metadata["source"] = pdf_path.name

    # Return the extracted pages for the text-splitting stage.
    return documents
