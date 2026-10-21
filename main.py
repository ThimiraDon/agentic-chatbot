import requests
import streamlit as st
import uuid,json

API_URL="http://127.0.0.1:8000"

def get_conversations():
    try:
        response=requests.get(
            f"{API_URL}/conversations",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load conversations: {exc}")
        return []


def get_conversation_history(thread_id):
    response = requests.get(
        f"{API_URL}/conversations/{thread_id}/history",
        timeout=10
    )
    response.raise_for_status()
    return response.json()["messages"]

#Initialize thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id=str(uuid.uuid4())

# Store conversation history during the Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

#This remembers which conversation the user has selected for deletion.
if "confirm_delete_thread_id" not in st.session_state:
    st.session_state.confirm_delete_thread_id = None


#configures the browser page that Streamlit displays
st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


with st.sidebar:
    st.title("🤖 Agentic AI Chatbot")
    st.caption("Powered by FastAPI, LangGraph and Groq")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.thread_id=str(uuid.uuid4())
        st.session_state.messages=[]
        st.rerun()

    st.divider()
    st.subheader("Your Conversations")

    conversations=get_conversations()

    
    for con in conversations:
        title = con.get("title") or "New Chat"
        thread_id = con["thread_id"]

        # Show confirmation controls for the selected conversation
        if st.session_state.confirm_delete_thread_id == thread_id:
            st.caption(f"Delete '{title}'?")
            confirm_col, cancel_col = st.columns(2)

            if confirm_col.button(
                "Delete",
                key=f"confirm_delete_{thread_id}",
                use_container_width=True
            ):
                try:
                    response = requests.delete(
                        f"{API_URL}/conversations/{thread_id}",
                        timeout=10
                    )
                    response.raise_for_status()

                    # Reset the active chat if it was deleted
                    if st.session_state.thread_id == thread_id:
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.messages = []

                    st.session_state.confirm_delete_thread_id = None
                    st.rerun()

                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not delete conversation: {exc}")

            if cancel_col.button(
                "Cancel",
                key=f"cancel_delete_{thread_id}",
                use_container_width=True
            ):
                st.session_state.confirm_delete_thread_id = None
                st.rerun()

        else:
            chat_col, delete_col = st.columns([0.8, 0.2])

            if chat_col.button(
                title,
                key=f"conversation_{thread_id}",
                use_container_width=True
            ):
                try:
                    messages = get_conversation_history(thread_id)
                    st.session_state.thread_id = thread_id
                    st.session_state.messages = messages
                    st.rerun()

                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not load conversation: {exc}")

            if delete_col.button(
                "🗑️",
                key=f"delete_{thread_id}",
                help="Delete conversation",
                use_container_width=True
            ):
                st.session_state.confirm_delete_thread_id = thread_id
                st.rerun()






#Display previous messages
#message = {"role": "user", "content": "Hello"}
#   Take each message stored in the session.
#   Create a chat container based on its role.
#   Display that message's content inside the container.
#   Repeat for every stored message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
question = st.chat_input("Ask me anything...")

if question:
    # Display and save the user's message
    st.session_state.messages.append({
        "role":"user",
        "content":question
    })

    with st.chat_message("user"):
        st.markdown(question)

    
    # Send the question to FastAPI
    with st.chat_message("assistant"):
        status_box = st.status(
            "Thinking...",
            expanded=True
        )

        answer_placeholder = st.empty()
        answer = ""

        try:
            with requests.post(
                f"{API_URL}/tangi",
                json={
                    "question": question,
                    "thread_id": st.session_state.thread_id
                },
                stream=True,
                timeout=120
            ) as response:

                response.raise_for_status()

                for line in response.iter_lines(
                    decode_unicode=True
                ):
                    if not line:
                        continue

                    event = json.loads(line)

                    if event["type"] == "status":
                        status_box.update(
                            label=event["message"],
                            state="running"
                        )

                    elif event["type"] == "answer":
                        answer = event["content"]
                        answer_placeholder.markdown(answer)

                        status_box.update(
                            label="Response complete",
                            state="complete",
                            expanded=False
                        )

                    elif event["type"] == "error":
                        status_box.update(
                            label=event["message"],
                            state="error"
                        )

            if answer:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.rerun()

        except requests.exceptions.Timeout:
            status_box.update(
                label="Request timed out",
                state="error"
            )
            st.error("The request timed out. Please try again.")

        except requests.exceptions.ConnectionError:
            status_box.update(
                label="Connection failed",
                state="error"
            )
            st.error(
                "Cannot connect to FastAPI. "
                "Make sure the backend is running on port 8000."
            )

        except requests.exceptions.RequestException as exc:
            status_box.update(
                label="Request failed",
                state="error"
            )
            st.error(f"API request failed: {exc}")
