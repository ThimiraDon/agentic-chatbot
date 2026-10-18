from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    answer:str