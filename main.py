import requests
import streamlit as st

API_URL="http://127.0.0.1:8000"


#configures the browser page that Streamlit displays
st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Chatbot")
st.caption("Powered by FastAPI, LangGraph and Groq")

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
                response = requests.post(
                    f"{API_URL}/",
                    json={"question":question},#question-backend expects a request matching its Pydantic schema:
                    timeout=120
                )

                #check the backend status code continue if 200 OK otherwise raise an error
                response.raise_for_status() 
                #from response object we get data through json body
                data = response.json()
                answer=data['answer']

                st.markdown(answer)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":answer
                })

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running on port 8000."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"API request failed: {exc}")

            except (ValueError, KeyError):
                st.error("The API returned an invalid response.")