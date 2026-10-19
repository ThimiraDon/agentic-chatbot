from fastapi import FastAPI, HTTPException
from src.schemas import ChatResponse, ChatRequest
from src.graph import build_graph
import logging
from langchain_core.messages import HumanMessage

from contextlib import asynccontextmanager
from src.checkpointer import create_checkpointer,pool
from src.conversation import create_conversation_table
from src.conversation import (list_conversation,
                              create_conversation,
                              update_conversation)

#StreamingResponse is a FastAPI class that sends response content incrementally 
# instead of requiring the entire response to be ready first.
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# config={
#     "configurable":{
#         "thread_id":"conversation_001"
#     }
# }

@asynccontextmanager
async def lifespan(app:FastAPI):
    #Initialize postgres checkpointer
    checkpointer=create_checkpointer()

    #create conversation table
    create_conversation_table()

    #build the graph
    #app.state is a place FastAPI provides to store application-wide objects and variables
    app.state.chatbot=build_graph(checkpointer=checkpointer)

    logger.info("Postgres checkpointer initialized")

    try:
        yield
    finally:
        pool.close()
        logger.info("PostgresSQL connection is closed")

app = FastAPI(
    title="Agentic AI Chatbot",
    version="0.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "Status":"OK"
    }
@app.post("/",response_model=ChatResponse)
def chat(request:ChatRequest)->ChatResponse:
    try:
        response=app.state.chatbot.invoke(
            {
                "messages":[
                    HumanMessage(content=request.question)
                ]
            },
            config={
                "configurable":{"thread_id":request.thread_id}
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

    # NEW: Create the conversation record if it doesn't exist.
    create_conversation(
        thread_id=request.thread_id,
        title=request.question[:40]
    )

    #generate() defines a generator function. 
    #A generator can produce values one at a time using the yield keyword.
    def generate():
        try:
            for chunk,metadata in app.state.chatbot.stream(
                {
                    "messages":[
                        HumanMessage(content=request.question)
                    ]
                },
                config={
                "configurable":{"thread_id":request.thread_id}
                },
                stream_mode="messages"
            ):
                if isinstance(chunk.content, str) and chunk.content:
                    yield chunk.content

        except Exception:
            logger.exception("Streaming request failed")
            raise
        finally:
            # NEW: Update the last-activity timestamp.
            update_conversation(request.thread_id)

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

#this endpoint returns conversation information, not the actual chat messages
@app.get("/conversations")
def get_conversations():
    return list_conversation()

@app.post("/conversations")
def add_conversations(thread_id:str,title:str="New Chat"):
    create_conversation(thread_id,title)
    return {
        "messages":"Conversation created!",
        "thread_id":thread_id
    }

@app.get("/conversations/{thread_id}/history")
def get_conversation_history(thread_id:str):

    config={
        "configurable":{
            "thread_id":thread_id
        }
    }

    state=app.state.chatbot.get_state(config)

    messages=state.values.get("messages",[])

    history=[]

    for message in messages:
        if message.type == "human":
            role="user"
        elif message.type == "ai":
            role="assistant"
        else:
            continue

        if isinstance(message.content,str):
            history.append({
                "role":role,
                "content":message.content
            })

    return {
        "thread_id":thread_id,
        "messages":history
    }