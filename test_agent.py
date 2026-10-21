
from langchain_core.messages import HumanMessage

from src.checkpointer import create_checkpointer, pool
from src.graph import build_graph


def main() -> None:
    checkpointer = create_checkpointer()

    try:
        chatbot = build_graph(checkpointer=checkpointer)

        question = (
            "According to the climate.pdf document, "
            "what is its main topic? Use the knowledge base "
            "to find the answer."
        )

        response = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "configurable": {
                    "thread_id": "rag_test_climate_001"
                }
            },
        )

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(response.get("answer", "No answer returned."))

    finally:
        pool.close()


if __name__ == "__main__":
    main()
