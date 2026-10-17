from fastapi import FastAPI, HTTPException
from src.schemas import ChatResponse, ChatRequest
from src.graph import chatbot
import logging

logger = logging.getLogger(__name__)

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
    try:
        response=chatbot.invoke(
            {
                "user_input":request.question
            }
        )
        logger.info("Suuccessfully invoked")
        answer = response["answer"]

        return ChatResponse(answer=answer)

    except Exception as e:
        logger.info(f"Chat request failed: {type(e).__name__}:{e}")

        raise HTTPException(#frontend received this
            status_code=502,
            detail="chatbot could not generate a response"
        ) from e