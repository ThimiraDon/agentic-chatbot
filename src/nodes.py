import os

from src.state import ChatState
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4
)

def chatbot_node(state:ChatState)->dict:
    """
    Generate the answer using the LLM.
    """
    user_query=state["messages"]

    response=llm.invoke(user_query)

    return {
        "messages":[response]
    }


def response_node(state:ChatState)->dict:
    """
    Validate and prepare the answer for the final output
    """

    final_answer=state["messages"][-1].content

    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("The LLM returned an empty response.")

    return {
        "answer":final_answer.strip()
    }