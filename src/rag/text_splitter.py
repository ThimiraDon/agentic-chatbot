
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split loaded documents into smaller text chunks.

    Args:
        documents: Documents returned by the PDF loader.
        chunk_size: Maximum target size of each chunk, in characters.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list of smaller Document objects with their metadata preserved.
    """

    # Reject invalid chunk settings.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    # Create a splitter that tries to preserve natural text boundaries.
    # It prefers paragraphs, then lines, then sentences/words, and
    # finally individual characters when necessary.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    # Split each Document's text into smaller Documents.
    # Original metadata, such as source and page, is preserved.
    chunks = splitter.split_documents(documents)

    # Return chunks for embedding and storage in Pinecone.
    return chunks
