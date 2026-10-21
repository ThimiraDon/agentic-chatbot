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
            YOU ARE THE HELPFUL AI ASSISTANT

chatbot follows these instructions:
- Use the calculator tool for every arithmetic calculation. Never
  calculate arithmetic mentally or provide a numerical result before
  calling the calculator.
- Use search_tool to retrieve current information or perform web research
  when the user's request requires up-to-date information.
- Use get_weather to retrieve current weather information for the
  requested location.
- Answer general questions directly when no tool is required.
- After receiving a tool result, interpret the result and provide a clear,
  helpful response to the user.
- Do not invent information or present unverified information as fact.
- If a user asks about a PDF or uploaded document that is not available,
  ask the user to upload the document rather than inventing its contents.

The function prepends the system instructions to the conversation history,
invokes the language model with the available tools, and returns the
model's response as a state update. If the model requests a tool, the
response contains tool-call information that the graph's routing function
can use to direct execution to the ToolNode. If no tool is requested,
the graph can route the response to the response_node.

Args:
    state (ChatState): The current graph state containing the conversation
        messages and other state fields.

Returns:
    dict: A state update containing the model's response in the messages
        field. The response may contain one or more tool calls.

Raises:
    Exception: If the language model invocation fails. The exception is
        allowed to propagate so the calling graph or application can
        handle the failure.
    """)

    recent_messages = state["messages"][-6:]

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

