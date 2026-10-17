from fastapi import FastAPI, HTTPException

from src.graph import chatbot
from src.schemas import ChatRequest, ChatResponse

app=FastAPI(
    title="Agentic AI Chatbot",
    description="chatbot with superpowers",
    version='0.1.0'
)

@app.get("/health")
def health_check():
    return {
        'status':'OK',
        'application':'agentic-chatbot'
    }

@app.post("/",response_model=ChatResponse)
def chat(request:ChatRequest)->ChatResponse:
    try:
        result = chatbot.invoke(
            {
                "user_input":request.question
            }
        )
        print(result)

        return ChatResponse(
            answer=result["messages"][-1].content
        )
    
    except Exception as e:
        print(f"Chat request failed: {type(e).__name__}: {e}")

        raise HTTPException(
            status_code=502,
            detail="The chatbot couldnt generate a response"
        ) from e
