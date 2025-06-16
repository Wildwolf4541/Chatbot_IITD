from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from gtts import gTTS
import os

# ----------- 🔐 Load API Key (Google Generative AI) -----------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Make sure it's set in your .env or system env

# ----------- 📄 Load and split document -----------
loader = PyPDFLoader("data.pdf")  # Make sure 'data.pdf' is in the same directory
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs_split = splitter.split_documents(docs)

# ----------- 🤖 Embeddings + Vector Store -----------
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs_split, embedding)

# ----------- 🔍 RetrievalQA (RAG) Chain -----------
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="models/chat-bison-001"),  # You can also use "gemini-pro"
    retriever=retriever,
    return_source_documents=False
)

# ----------- 🔊 Text-to-Speech Function -----------
def speak_text(text, filename="response.mp3"):
    tts = gTTS(text=text)
    tts.save(filename)
    os.system(f"start {filename}")  # Windows. Use `afplay` (Mac) or `xdg-open` (Linux)

# ----------- 💬 Chat Interface Function -----------
def chat_with_bot(query):
    response = qa_chain.run(query)
    print("Bot:", response)
    speak_text(response)

# ----------- 🧪 Quick Test -----------
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        chat_with_bot(user_input)
