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

# -- SAFEGUARD FALLBACK --
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

def get_answer_from_gemini(question, context_docs):
    original_docs = [doc.page_content for doc in context_docs if doc.metadata["source"] == "Original"]
    amended_docs = [doc.page_content for doc in context_docs if doc.metadata["source"] == "Amended"]

    prompt = f"""
You are a highly skilled document comparison assistant. Your task is to answer the user's question by analyzing and comparing the provided text sections from an 'Original' document and an 'Amended' document.

**User's Question:**
{question}

---
**Context from Original Document:**
{"---".join(original_docs) if original_docs else "No relevant context found in the original document."}
---
**Context from Amended Document:**
{"---".join(amended_docs) if amended_docs else "No relevant context found in the amended document."}
---

**Your Instructions:**
1. Carefully read the user's question and the provided context from both documents.
2. Directly answer the question based *only* on the information in the context.
3. Clearly state what has changed, what was added, or what was removed.
4. If the context is insufficient to answer the question, clearly state that you cannot answer based on the provided text.
5. Be concise and precise in your answer.

**Answer:**
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"

# -- STREAMLIT APP --
st.set_page_config(layout="wide", page_title="Ask a PDF")
st.title("📄 Ask Questions About Your PDF Changes")
st.markdown("Upload two versions of a PDF and ask Gemini to find the differences for you.")

# -- SESSION STATE INITIALIZATION --
if 'original_db' not in st.session_state:
    st.session_state.original_db = None
if 'amended_db' not in st.session_state:
    st.session_state.amended_db = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# -- SIDEBAR: Upload PDFs --
with st.sidebar:
    st.header("1. Upload Documents")
    file1 = st.file_uploader("Upload Original PDF", type="pdf", key="file1")
    file2 = st.file_uploader("Upload Amended PDF", type="pdf", key="file2")

    if st.button("Process Documents", type="primary", use_container_width=True):
        if file1 and file2:
            if [file1.name, file2.name] == st.session_state.processed_files:
                st.toast("✅ Documents are already processed and ready.")
            else:
                with st.status("⚙️ Processing PDFs...", expanded=True) as status:
                    st.write("Extracting text from documents...")
                    text1 = extract_text_from_pdf(file1.getvalue())
                    text2 = extract_text_from_pdf(file2.getvalue())

                    st.write("Chunking documents...")
                    chunks_v1 = create_document_chunks(text1, "Original")
                    chunks_v2 = create_document_chunks(text2, "Amended")

                    st.write("Creating separate vector stores...")
                    st.session_state.original_db = create_vector_db_from_chunks(chunks_v1)
                    st.session_state.amended_db = create_vector_db_from_chunks(chunks_v2)

                    st.session_state.processed_files = [file1.name, file2.name]
                    status.update(label="✅ Processing Complete!", state="complete")
        else:
            st.warning("⚠️ Please upload both PDF files.")

# -- MAIN: Ask Questions --
st.header("2. Ask Your Question")

if st.session_state.original_db and st.session_state.amended_db:
    st.success("Documents processed! You can now ask questions about the changes.")

    question = st.text_input(
        "e.g., 'What changed about the termination clause?' or 'Summarize the changes in section 5'",
        placeholder="Ask about changes between the two documents..."
    )

    if question:
        with st.spinner("🔍 Searching for relevant context in both documents..."):
            retriever_original = st.session_state.original_db.as_retriever(search_kwargs={"k": 4})
            retriever_amended = st.session_state.amended_db.as_retriever(search_kwargs={"k": 4})
            original_relevant = retriever_original.get_relevant_documents(question)
            amended_relevant = retriever_amended.get_relevant_documents(question)

        with st.spinner("🤖 Gemini is analyzing the differences..."):
            combined_context = original_relevant + amended_relevant
            answer = get_answer_from_gemini(question, combined_context)
            st.info(f"**Answer:**\n\n{answer}")

        with st.expander("📚 View Retrieved Context"):
            st.markdown("---")
            for doc in original_relevant:
                st.markdown(f"**Source: Original**")
                st.code(doc.page_content)
                st.markdown("---")
            for doc in amended_relevant:
                st.markdown(f"**Source: Amended**")
                st.code(doc.page_content)
                st.markdown("---")
else:
    st.info("Please upload and process two PDF documents in the sidebar to begin.")
