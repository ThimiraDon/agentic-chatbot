from pydantic import BaseModel, Field

#FastAPI uses these Pydantic models to validate incoming requests and define the JSON response.

class ChatRequest(BaseModel):
    question:str=Field(
        min_length=1,
        max_length=10000,
        description="user's input message"
    )
    thread_id:str

class ChatResponse(BaseModel):
    answer:str