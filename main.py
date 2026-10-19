import requests
import streamlit as st
import uuid

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
        if st.button(
            con["title"],
            key=con["thread_id"],
            use_container_width=True
        ):
            thread_id = con["thread_id"]

            try:
                messages = get_conversation_history(thread_id)

                st.session_state.thread_id = thread_id
                st.session_state.messages = messages

                st.rerun()

            except requests.exceptions.RequestException as exc:
                st.error(f"Could not load conversation: {exc}")

#Initialize thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id=str(uuid.uuid4())

# Store conversation history during the Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        with st.spinner("thinking...."):
            try:
                #sends an HTTP POST request to a server
                #@app.post("/") decorator registers a POST endpoint at /.
                # Your frontend must send a POST request to that path.
                with requests.post(
                    f"{API_URL}/tangi",
                    json={"question":question,
                          "thread_id":st.session_state.thread_id},#question-backend expects a request matching its Pydantic schema:
                    stream=True,
                    timeout=120
                ) as response:

                    #check the backend status code continue if 200 OK otherwise raise an error
                    response.raise_for_status() 
                    ## Read and display chunks progressively
                    answer=st.write_stream(
                        response.iter_content(
                            chunk_size=None,
                            decode_unicode=True
                        )
                    )
                

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":answer
                })

                st.rerun()

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running on port 8000."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"API request failed: {exc}")