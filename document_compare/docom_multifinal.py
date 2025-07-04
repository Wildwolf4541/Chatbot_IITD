import os
import streamlit as st
import tempfile
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -- LOAD ENV VARIABLES --
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key or api_key.strip() == "":
    st.error("❌ GOOGLE_API_KEY is not set. Please add it to your .env file.")
    st.stop()

# -- CONFIGURE GEMINI & EMBEDDINGS --
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# -- UTILITIES --
@st.cache_data
def extract_text_from_pdf(file_contents):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_contents)
        tmp_path = tmp.name

    try:
        with pdfplumber.open(tmp_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    finally:
        os.remove(tmp_path)
    return text

def create_document_chunks(text, source_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]

def create_vector_db_from_chunks(chunks):
    with st.spinner(f"🔗 Creating vector store for {chunks[0].metadata['source']}..."):
        return FAISS.from_documents(chunks, embeddings)

def get_answer_from_gemini(question, context_docs, doc_names):
    # Group docs by source name
    grouped_docs = {name: [] for name in doc_names}
    for doc in context_docs:
        grouped_docs[doc.metadata["source"]].append(doc.page_content)

    # Build prompt dynamically for all uploaded PDFs
    contexts = ""
    for name in doc_names:
        ctx_text = "---".join(grouped_docs[name]) or "No relevant context found."
        contexts += f"\n---\n**Context from Document '{name}':**\n{ctx_text}\n"

    prompt = f"""
You are a highly skilled document comparison assistant. Your task is to answer the user's question by analyzing and comparing the provided text sections from multiple documents.

**User's Question:**
{question}

{contexts}

---
**Your Instructions:**
1. Carefully analyze changes across all versions.
2. Describe what changed, was added, or removed.
3. If insufficient context, say so clearly.
4. Be concise and precise.

**Answer:**
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"

# -- STREAMLIT APP CONFIG --
st.set_page_config(layout="wide", page_title="Compare Multiple PDFs")
st.title("📄 Ask Questions About Your PDF Versions")
st.markdown("Upload multiple versions of a PDF and ask Gemini to find differences across them.")

# -- SESSION STATE --
if "vector_dbs" not in st.session_state:
    st.session_state.vector_dbs = {}  # {filename: FAISS instance}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# -- SIDEBAR: Upload PDFs --
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload 2 or more PDFs", 
        type="pdf", 
        accept_multiple_files=True,
        key="multi_upload"
    )
    
    if st.button("Process Documents", type="primary", use_container_width=True):
        if uploaded_files and len(uploaded_files) >= 2:
            uploaded_filenames = [file.name for file in uploaded_files]
            if uploaded_filenames == st.session_state.processed_files:
                st.toast("✅ Documents already processed.")
            else:
                st.session_state.vector_dbs = {}  # Reset
                with st.status("⚙️ Processing PDFs...", expanded=True) as status:
                    for file in uploaded_files:
                        st.write(f"Extracting text from {file.name}...")
                        text = extract_text_from_pdf(file.getvalue())

                        st.write(f"Chunking {file.name}...")
                        chunks = create_document_chunks(text, file.name)

                        st.write(f"Creating vector store for {file.name}...")
                        vector_db = create_vector_db_from_chunks(chunks)
                        st.session_state.vector_dbs[file.name] = vector_db

                    st.session_state.processed_files = uploaded_filenames
                    status.update(label="✅ Processing Complete!", state="complete")
        else:
            st.warning("⚠️ Please upload at least 2 PDF files.")

# -- MAIN: Ask Questions --
st.header("2. Ask Your Question")

if st.session_state.vector_dbs:
    st.success(f"{len(st.session_state.vector_dbs)} documents processed! You can now ask questions.")

    question = st.text_input(
        "e.g., 'What changed between versions?'",
        placeholder="Ask about changes across your uploaded documents..."
    )

    if question:
        with st.spinner("🔍 Retrieving relevant context..."):
            combined_docs = []
            for name, vector_db in st.session_state.vector_dbs.items():
                retriever = vector_db.as_retriever(search_kwargs={"k": 4})
                docs = retriever.get_relevant_documents(question)
                combined_docs.extend(docs)

        with st.spinner("🤖 Gemini is analyzing..."):
            answer = get_answer_from_gemini(question, combined_docs, list(st.session_state.vector_dbs.keys()))
            st.info(f"**Answer:**\n\n{answer}")

        with st.expander("📚 View Retrieved Context"):
            for name, vector_db in st.session_state.vector_dbs.items():
                retriever = vector_db.as_retriever(search_kwargs={"k": 4})
                docs = retriever.get_relevant_documents(question)
                for doc in docs:
                    st.markdown(f"**Source: {name}**")
                    st.code(doc.page_content)
                    st.markdown("---")
else:
    st.info("Upload and process at least two PDF documents in the sidebar to begin.")
