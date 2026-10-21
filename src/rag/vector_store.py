
import hashlib
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import EmbedQuery, Pinecone

load_dotenv()


class PineconeVectorStore:
    """Store and search document chunks using Pinecone."""

    def __init__(self) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if not api_key:
            raise ValueError("PINECONE_API_KEY is not configured.")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME is not configured.")

        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(index_name)

    def upsert_documents(
        self,
        documents: list[Document],
        namespace: str = "default",
    ) -> int:
        """Store document chunks. Pinecone generates their embeddings."""

        if not documents:
            return 0

        records = []

        for document in documents:
            text = document.page_content.strip()

            if not text:
                raise ValueError("Document chunks cannot be empty.")

            source = str(document.metadata.get("source", "unknown"))
            page = document.metadata.get("page")
            start_index = document.metadata.get("start_index")

            identity = (
                f"{source}|{page}|{start_index}|{text}"
            )
            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            record = {
                "_id": chunk_id,
                "chunk_text": text,
                "title": source,
                "source": source,
            }

            if isinstance(page, (int, float, str)):
                record["page"] = page

            if isinstance(start_index, (int, float, str)):
                record["start_index"] = start_index

            records.append(record)

        # Integrated-embedding indexes support up to 96
        # documents per upsert request.
        batch_size = 50

        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]

            self.index.documents.upsert(
                namespace=namespace,
                documents=batch,
            )

        return len(records)

    def search(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "default",
    ) -> list[dict]:
        """Find relevant chunks using integrated text embeddings."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        response = self.index.documents.search(
            namespace=namespace,
            top_k=top_k,
            score_by=[
                EmbedQuery(
                    field="chunk_text",
                    query=question,
                )
            ],
            include_fields=[
                "chunk_text",
                "title",
                "source",
                "page",
                "start_index",
            ],
        )

        results = []

        for match in response.matches:
            results.append({
                "id": match.id,
                "text": match.get("chunk_text", ""),
                "source": match.get("source", "unknown"),
                "page": match.get("page"),
                "score": match.score,
            })

        return results
