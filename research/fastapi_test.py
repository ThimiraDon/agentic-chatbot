from fastapi import FastAPI, Query
from src.schemas import ChatResponse, ChatRequest
from src.graph import chatbot

app = FastAPI(
    title="Agentic AI Chatbot",
    version="0.0.0"
)

@app.get("/health")
def health_check():
    return {
        "Status":"OK"
    }
@app.post("/",response_model=ChatResponse)
def chat(request:ChatRequest)->ChatResponse:

    response=chatbot.invoke(
        {
            "user_input":request.question
        }
    )

    answer = response["answer"]

    return ChatResponse(answer=answer)


#================= Path parameters; conversation_id extract from the url
#braces indicate a variable part of the URL
@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id:int):
    return {
        "conversation_id":conversation_id,
        "message":"Conversation requested"
    }

#=============== Query parameter 
#The ? separates the path from the query parameters. 
# The limit is the parameter name, and 10 is its value.
@app.get("/conversation")
def get_conversation(
    limit:int=Query(default=10,gt=0,le=100)
):
    return {
        "requested_limit": limit
    }