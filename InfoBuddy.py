import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import groq
import speech_recognition as sr

st.set_page_config(
    page_title="InfoBuddy",
    page_icon="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 24 24'%3E%3Cpath fill='%23000' d='M17 2h-4V1h-2v1H7a3 3 0 0 0-3 3v3a5 5 0 0 0 5 5h6a5 5 0 0 0 5-5V5a3 3 0 0 0-3-3m-6 5.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0m5 0a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0M4 22a8 8 0 1 1 16 0z'/%3E%3C/svg%3E",
    layout="centered",
)


# Load environment variables
load_dotenv()
try:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
except KeyError as e:
    st.error(f"Environment variable {str(e)} is missing. Please check your .env file.")
    st.stop()

# Initialize Embeddings and Chroma for RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
persist_directory = "datastore_db_new"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 12})

# Initialize LLM and QA Chain
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)
output_parser = StrOutputParser()


# Helper function to create a prompt
def create_prompt(context, question):
    context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
    return [
        (
            "system",
            f"You are InfoBuddy, a College Query Chatbot. Your work is to provide factual information. Be friendly with the users and provide concise answers for their queries. Remember to check for related links for every question. Here is the conversation history:\n{context_str}",
        ),
        ("user", f"Question: {question}"),
    ]


# Generate a response using the LLM and RAG retriever
def generate_response(context, question):
    prompt_template = ChatPromptTemplate.from_messages(create_prompt(context, question))

    try:
        response = qa_chain.invoke({"query": question})
        result = response["result"]
        return f"{result}\n\n", result
    except groq.RateLimitError as e:
        return "Error: Rate limit reached. Please try again later.", None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"Error: {str(e)}", None


# Function to load external CSS
def load_css(file_path):
    with open(file_path, "r") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


# Function to handle voice input
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        try:
            # Recognize the speech and get the text
            text = r.recognize_google(audio)
            # Add the user's voice input to the chat history
            st.session_state.messages.append({"role": "user", "content": text})

            # Add typing animation while generating a response
            typing_placeholder = st.empty()
            with typing_placeholder.container():
                st.markdown(
                    """
                    <div class='chat-wrapper assistant'>
                        <div class='chat-icon assistant-icon'>
                            <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 24 24'%3E%3Cpath fill='%237ea3ed' d='M17 2h-4V1h-2v1H7a3 3 0 0 0-3 3v3a5 5 0 0 0 5 5h6a5 5 0 0 0 5-5V5a3 3 0 0 0-3-3m-6 5.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0m5 0a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0M4 22a8 8 0 1 1 16 0z'/%3E%3C/svg%3E" alt="Assistant Icon" style="width: 30px; height: 30px;"/>
                        </div>
                        <div class='typing-animation-container'>
                            <div class="typing-indicator">
                                <span></span><span></span><span></span>
                            </div>
                        </div>
                    </div>`
                    """,
                    unsafe_allow_html=True,
                )

            # Generate the assistant's response
            response, _ = generate_response(st.session_state.messages, text)

            # Remove the typing animation
            typing_placeholder.empty()

            # Add the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error(
                "Could not request results from Google Speech Recognition service."
            )


# Main function to handle the conversation
def handle_conversation():
    # Load external CSS
    load_css("styles.css")

    # Display page titles
    st.markdown("<h1 class='main-title'>InfoBuddy</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 class='sub-title'>Your College Query Chatbot</h3>",
        unsafe_allow_html=True,
    )

    # Initialize conversation state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome to InfoBuddy! How can I help you?",
            }
        ]

    # Display chat history with icons outside the chat bubble
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            # User message aligned to the right with icon on the far right
            st.markdown(
                f"""
                <div class='chat-wrapper user'>
                    <div class='chat-bubble user'><div class='chat-message'>{msg['content']}</div></div>
                    <div class='chat-icon user-icon'>
                        <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 50 50'%3E%3Cg fill='none' stroke-linecap='round' stroke-linejoin='round' stroke-width='2'%3E%3Cpath stroke='%23344054' d='M18.75 31.25h12.5a10.417 10.417 0 0 1 10.417 10.417a2.083 2.083 0 0 1-2.084 2.083H10.417a2.083 2.083 0 0 1-2.084-2.083A10.417 10.417 0 0 1 18.75 31.25'/%3E%3Cpath stroke='%23306cfe' d='M25 22.917A8.333 8.333 0 1 0 25 6.25a8.333 8.333 0 0 0 0 16.667'/%3E%3C/g%3E%3C/svg%3E" alt="User Icon" style="width: 30px; height: 30px;"/>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Assistant message aligned to the left with icon on the far left
            st.markdown(
                f"""
                <div class='chat-wrapper assistant'>
                    <div class='chat-icon assistant-icon'>
                        <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 24 24'%3E%3Cpath fill='%237ea3ed' d='M17 2h-4V1h-2v1H7a3 3 0 0 0-3 3v3a5 5 0 0 0 5 5h6a5 5 0 0 0 5-5V5a3 3 0 0 0-3-3m-6 5.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0m5 0a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0M4 22a8 8 0 1 1 16 0z'/%3E%3C/svg%3E" alt="Assistant Icon" style="width: 30px; height: 30px;"/>
                    </div>
                    <div class='chat-bubble assistant'><div class='chat-message'>{msg['content']}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Microphone button directly after each assistant's message
            if st.button("🎙️ Voice Input", key=f"mic_{idx}"):
                voice_text = voice_input()
                if voice_text:
                    prompt = voice_text
                    # Add the user's voice input to the chat history
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    # Generate and display a response from the assistant
                    response, _ = generate_response(st.session_state.messages, prompt)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

    # Handle user input
    prompt = st.chat_input(placeholder="Type your question here...")
    if prompt:
        # Add user input to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f"""
            <div class='chat-wrapper user'>
                <div class='chat-bubble user'><div class='chat-message'>{prompt}</div></div>
                <div class='chat-icon user-icon'>
                    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 50 50'%3E%3Cg fill='none' stroke-linecap='round' stroke-linejoin='round' stroke-width='2'%3E%3Cpath stroke='%23344054' d='M18.75 31.25h12.5a10.417 10.417 0 0 1 10.417 10.417a2.083 2.083 0 0 1-2.084 2.083H10.417a2.083 2.083 0 0 1-2.084-2.083A10.417 10.417 0 0 1 18.75 31.25'/%3E%3Cpath stroke='%23306cfe' d='M25 22.917A8.333 8.333 0 1 0 25 6.25a8.333 8.333 0 0 0 0 16.667'/%3E%3C/g%3E%3C/svg%3E" alt="User Icon" style="width: 30px; height: 30px;"/>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Add typing animation placeholder
        typing_placeholder = st.empty()
        with typing_placeholder.container():
            st.markdown(
                """
        <div class='chat-wrapper assistant'>
            <div class='chat-icon assistant-icon'>
                <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 24 24'%3E%3Cpath fill='%237ea3ed' d='M17 2h-4V1h-2v1H7a3 3 0 0 0-3 3v3a5 5 0 0 0 5 5h6a5 5 0 0 0 5-5V5a3 3 0 0 0-3-3m-6 5.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0m5 0a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0M4 22a8 8 0 1 1 16 0z'/%3E%3C/svg%3E" alt="Assistant Icon" style="width: 30px; height: 30px;"/>
            </div>
            <div class='typing-animation-container'>
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """,
                unsafe_allow_html=True,
            )

        # Generate and display the assistant's response
        response, _ = generate_response(st.session_state.messages, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Replace typing animation with the assistant's response
        typing_placeholder.empty()
        st.markdown(
            f"""
            <div class='chat-wrapper assistant'>
                <div class='chat-icon assistant-icon'>
                    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1em' height='1em' viewBox='0 0 24 24'%3E%3Cpath fill='%237ea3ed' d='M17 2h-4V1h-2v1H7a3 3 0 0 0-3 3v3a5 5 0 0 0 5 5h6a5 5 0 0 0 5-5V5a3 3 0 0 0-3-3m-6 5.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0m5 0a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0M4 22a8 8 0 1 1 16 0z'/%3E%3C/svg%3E" alt="Assistant Icon" style="width: 30px; height: 30px;"/>
                </div>
                <div class='chat-bubble assistant'><div class='chat-message'>{response}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.rerun()


# Run the conversation handler
if __name__ == "__main__":
    handle_conversation()
