import os

from langchain.messages import SystemMessage
from src.state import ChatState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.tools import tools

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

#creates a model configuration that knows which tools it can request.
llm_with_tools=llm.bind_tools(tools)

def chatbot_node(state:ChatState)->dict:
    
    system_message = SystemMessage(
        content="""
        YOU ARE A HELPFUL AI ASSISTANT.

        TOOL USAGE RULES:

        1. KNOWLEDGE BASE (RAG)
        - Use search_knowledge_base for questions about uploaded PDFs
        or documents stored in the knowledge base.
        - Always retrieve relevant document content before answering
        questions about those documents.
        - Base document-specific answers on the retrieved content.
        - If no relevant information is found, clearly tell the user.
        Do not invent information or pretend a document supports it.
        - If the document has not been uploaded or ingested, explain
        that it must be uploaded and processed first.

        2. CALCULATOR
        - Use calculator for every arithmetic calculation.
        - Never calculate arithmetic mentally or provide a numerical
        result before calling the calculator.

        3. WEB SEARCH
        - Use search_tool for current information or web research
        when the request requires it.

        4. WEATHER
        - Use get_weather for current weather information.

        GENERAL RULES:
        - Answer general questions directly when no tool is required.
        - After receiving a tool result, interpret it and provide a
        clear, helpful answer.
        - Do not invent facts or present unverified information as fact.
        - Do not claim to have searched a document unless you actually
        called the knowledge-base tool.
        """
    )


    recent_messages = state["messages"]#[-6:]

    response = llm_with_tools.invoke(
        [system_message, *recent_messages]
    )

    # user_query = [system_message, *state["messages"]]

    # response=llm_with_tools.invoke(user_query)

    return {
        "messages":[response]
    }


def response_node(state:ChatState)->dict:

    """
    Validate and prepare the final assistant response.

    Args:
        state: The current graph state containing conversation messages.

    Returns:
        A dictionary containing the cleaned final answer.
    """

    final_answer = state["messages"][-1].content

    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("The LLM returned an empty response.")

    return {
        "answer": final_answer.strip()
    }

