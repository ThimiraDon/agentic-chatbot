from fastapi import FastAPI, HTTPException
from src.schemas import ChatResponse, ChatRequest
from src.graph import build_graph
import logging
from langchain_core.messages import HumanMessage

import json
from contextlib import asynccontextmanager
from src.checkpointer import create_checkpointer,pool
from src.conversation import create_conversation_table
from src.conversation import (list_conversation,
                              create_conversation,
                              update_conversation)
from src.conversation import delete_conversation as delete_conversation_record

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

    app.state.checkpointer = checkpointer

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
def chat_stream(request: ChatRequest):
    create_conversation(
        thread_id=request.thread_id,
        title=request.question[:40]
    )

    def generate():
        try:
            for update in app.state.chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=request.question)
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": request.thread_id
                    }
                },
                stream_mode="updates"
            ):
                # 1. Detect when the AI requests a tool.
                chatbot_update = update.get("chatbot_node", {})
                messages = chatbot_update.get("messages", [])

                if messages:
                    ai_message = messages[-1]
                    tool_calls = getattr(ai_message, "tool_calls", [])

                    if tool_calls:
                        tool_names = ", ".join(
                            call["name"] for call in tool_calls
                        )

                        yield json.dumps({
                            "type": "status",
                            "message": f"Calling {tool_names}..."
                        }) + "\n"

                # 2. Notify the frontend after tool execution.
                if "tool_node" in update:
                    yield json.dumps({
                        "type": "status",
                        "message": "Tool finished. Preparing answer..."
                    }) + "\n"

                # 3. Send only the final answer.
                response_update = update.get("response_node")

                if response_update:
                    answer = response_update.get("answer")

                    if isinstance(answer, str) and answer:
                        yield json.dumps({
                            "type": "answer",
                            "content": answer
                        }) + "\n"

        except Exception as exc:
            logger.exception("Streaming request failed")

            yield json.dumps({
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}"
            }) + "\n"

        finally:
            update_conversation(request.thread_id)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
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

@app.delete("/conversations/{thread_id}")
def delete_conversation_endpoint(thread_id: str):
    try:
        # Delete the saved LangGraph state
        app.state.checkpointer.delete_thread(thread_id)

        # Delete the conversation metadata
        delete_conversation_record(thread_id)

        logger.info("Conversation %s deleted successfully", thread_id)

        return {
            "message": "Conversation deleted",
            "thread_id": thread_id
        }

    except Exception as exc:
        logger.exception("Failed to delete conversation %s", thread_id)

        raise HTTPException(
            status_code=500,
            detail="Could not delete conversation"
        ) from exc