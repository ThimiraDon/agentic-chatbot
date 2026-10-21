
from src.rag.retrieval import RetrievalService


def main() -> None:
    retrieval_service = RetrievalService()

    documents = retrieval_service.retrieve(
        question="What is the main topic of this document?",
        top_k=3,
    )

    if not documents:
        print("No matching chunks found.")
        return

    for number, document in enumerate(documents, start=1):
        print(f"\n--- Result {number} ---")
        print(f"Source: {document.metadata['source']}")
        print(f"Page: {document.metadata['page']}")
        print(f"Content: {document.page_content[:300]}")


if __name__ == "__main__":
    main()
