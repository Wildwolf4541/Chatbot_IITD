import streamlit as st
import os
from dotenv import load_dotenv
from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Set the Google API Key from environment variables
# Ensure your .env file contains GOOGLE_API_KEY="your_api_key_here"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Caching document load and RAG chain for efficiency
# @st.cache_resource ensures that the function runs only once and its result is cached
@st.cache_resource
def load_rag_chain():
    """
    Loads the PDF document, splits it into chunks, creates embeddings,
    builds a vector store, and initializes the RetrievalQA chain.
    """
    # Check if the Google API Key is set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
        st.stop() # Stop the Streamlit app execution

    try:
        # Initialize the PDF loader with the document path
        # Make sure 'MaximalSquare.pdf' is in the same directory as your script
        loader = PyPDFLoader("MaximalSquare.pdf")
        docs = loader.load() # Load the document

        if not docs:
            st.error("No documents were loaded from MaximalSquare.pdf. Please ensure the file exists and is not empty.")
            st.stop()

        # Initialize the text splitter for breaking down documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs) # Split the loaded documents

        if not split_docs:
            st.error("No text was extracted or split from the document. The PDF might be empty or unreadable.")
            st.stop()

        # Initialize Google Generative AI Embeddings for creating vector representations of text
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create a Chroma vector store from the split documents and embeddings
        # This stores the document chunks and their embeddings, enabling efficient retrieval
        vectorstore = Chroma.from_documents(split_docs, embedding)

        # Convert the vector store into a retriever
        # The retriever fetches relevant document chunks based on a query
        retriever = vectorstore.as_retriever()

        # Initialize the RetrievalQA chain
        # This chain combines the language model (Gemini 1.5 Flash) with the retriever
        # to answer questions based on the retrieved document chunks.
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"), # Language Model
            retriever=retriever, # Document Retriever
            return_source_documents=False # Do not return the source documents in the response
        )
        return qa_chain
    except Exception as e:
        st.error(f"An error occurred during RAG chain setup: {e}")
        st.info("Please ensure your Google API Key is valid and has access to the Gemini Embeddings model. Also check if the 'MaximalSquare.pdf' file is valid and readable.")
        st.stop()


# Load the RAG chain when the application starts
qa_chain = load_rag_chain()

# Text-to-Speech function using gTTS
def speak_text(text, filename="response.mp3"):
    """
    Converts the given text to speech and plays it in the Streamlit app.
    Args:
        text (str): The text to convert to speech.
        filename (str): The name of the temporary audio file.
    """
    try:
        tts = gTTS(text=text) # Create a gTTS object
        tts.save(filename) # Save the speech to a file

        # Open and read the audio file bytes
        audio_file = open(filename, "rb")
        audio_bytes = audio_file.read()

        # Display the audio player in Streamlit
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Could not convert text to speech: {e}. Please check your internet connection or the length/content of the response.")


# --- Streamlit User Interface ---
# Set basic page configuration
st.set_page_config(page_title="🗣️ RAG Voice Bot", page_icon="🗣️")
st.title("🗣️ RAG Voice Bot with Gemini & TTS")

# Text input for the user's question
query = st.text_input("Ask a question from the document:")

# Button to trigger the query
if st.button("Ask"):
    # Check if the query input is empty
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Run the QA chain with the user's query
                response = qa_chain.run(query)
                st.success("Response:")
                st.write(response) # Display the text response
                speak_text(response) # Convert the response to speech and play it
            except Exception as e:
                st.error(f"An error occurred while getting a response: {e}. Please try again.")

