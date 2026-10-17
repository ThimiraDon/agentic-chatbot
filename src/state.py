from typing import TypedDict

class ChatState(TypedDict):
    user_input:str
    messages:str
    answer:str