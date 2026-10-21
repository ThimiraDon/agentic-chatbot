
from langchain_core.documents import Document

from src.rag.vector_store import PineconeVectorStore


class RetrievalService:
    """Retrieve relevant document chunks from Pinecone."""

    def __init__(self) -> None:
        # Initialize the Pinecone connection.
        self.vector_store = PineconeVectorStore()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "default",
    ) -> list[Document]:
        """
        Retrieve the most relevant document chunks for a question.

        Pinecone generates the query embedding automatically.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Search Pinecone using the user's question.
        matches = self.vector_store.search(
            question=question,
            top_k=top_k,
            namespace=namespace,
        )

        # Convert search results into LangChain Documents.
        documents = []

        for match in matches:
            document = Document(
                page_content=match["text"],
                metadata={
                    "source": match["source"],
                    "page": match["page"],
                    "score": match["score"],
                    "id": match["id"],
                },
            )
            documents.append(document)

        return documents
