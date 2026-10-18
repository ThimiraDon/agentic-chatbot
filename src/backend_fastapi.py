from fastapi import FastAPI, HTTPException
from src.schemas import ChatResponse, ChatRequest
from src.graph import chatbot
import logging
from langchain_core.messages import HumanMessage

#StreamingResponse is a FastAPI class that sends response content incrementally 
# instead of requiring the entire response to be ready first.
from fastapi.responses import StreamingResponse

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
                "messages":[
                    HumanMessage(content=request.question)
                ]
            }
        )
        logger.info("Successfully invoked")
        answer = response["answer"]

        return ChatResponse(answer=answer)

    except Exception as e:
        logger.info(f"Chat request failed: {type(e).__name__}:{e}")

        raise HTTPException(#frontend received this
            status_code=502,
            detail="chatbot could not generate a response"
        ) from e


@app.post("/tangi")
def chat_stream(request:ChatRequest):
    #generate() defines a generator function. 
    #A generator can produce values one at a time using the yield keyword.
    def generate():
        try:
            for chunk,metadata in chatbot.stream(
                {
                    "messages":[
                        HumanMessage(content=request.question)
                    ]
                },
                stream_mode="messages"
            ):
                if isinstance(chunk.content, str) and chunk.content:
                    yield chunk.content

        except Exception:
            logger.exception("Streaming request failed")
            raise

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )