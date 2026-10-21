
from pathlib import Path

from src.rag.ingestion import IngestionService


def main() -> None:
    # Path to the PDF we want to store in Pinecone.
    pdf_path = Path("data/climate.pdf")

    # Stop early if the PDF path is incorrect.
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    # Create the ingestion service.
    ingestion_service = IngestionService()

    # Load, split, embed, and store the PDF chunks.
    result = ingestion_service.ingest_pdf(
        file_path=pdf_path,
        namespace="default",
    )

    # Display the ingestion summary.
    print("PDF ingestion completed.")
    print(f"Source: {result['source']}")
    print(f"Pages loaded: {result['pages_loaded']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Chunks stored: {result['chunks_stored']}")
    print(f"Namespace: {result['namespace']}")


if __name__ == "__main__":
    main()
