
from pathlib import Path

from src.rag.document_loader import load_pdf
from src.rag.text_splitter import split_documents
from src.rag.vector_store import PineconeVectorStore


class IngestionService:
    """Process PDF documents and store them in Pinecone."""

    def __init__(self) -> None:
        # Initialize the Pinecone connection.
        self.vector_store = PineconeVectorStore()

    def ingest_pdf(
        self,
        file_path: str | Path,
        namespace: str = "default",
    ) -> dict:
        """
        Load, split, and store a PDF.

        Pinecone generates embeddings for the uploaded text.

        Args:
            file_path: Path to the PDF file.
            namespace: Pinecone namespace for the document.

        Returns:
            A summary of the ingestion operation.
        """

        # Step 1: Extract text from the PDF.
        documents = load_pdf(file_path)

        # Step 2: Divide the extracted text into smaller chunks.
        chunks = split_documents(documents)

        # Step 3: Upload the chunks to Pinecone.
        # Pinecone generates their embeddings automatically.
        stored_count = self.vector_store.upsert_documents(
            documents=chunks,
            namespace=namespace,
        )

        # Step 4: Return a summary of the ingestion operation.
        return {
            "source": Path(file_path).name,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
            "namespace": namespace,
        }
