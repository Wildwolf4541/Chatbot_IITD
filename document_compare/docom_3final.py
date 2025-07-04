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

def get_answer_from_gemini(question, context_docs):
    grouped_docs = {"Original": [], "Amended": [], "Final": []}
    for doc in context_docs:
        grouped_docs[doc.metadata["source"]].append(doc.page_content)

    prompt = f"""
You are a highly skilled document comparison assistant. Your task is to answer the user's question by analyzing and comparing the provided text sections from an 'Original', 'Amended', and 'Final' document.

**User's Question:**
{question}

---
**Context from Original Document:**
{"---".join(grouped_docs["Original"]) or "No relevant context found."}

---
**Context from Amended Document:**
{"---".join(grouped_docs["Amended"]) or "No relevant context found."}

---
**Context from Final Document:**
{"---".join(grouped_docs["Final"]) or "No relevant context found."}

---
**Your Instructions:**
1. Carefully analyze changes across all three versions.
2. Describe what changed from Original to Amended, and from Amended to Final.
3. Point out new additions, removals, or reversals of changes.
4. If insufficient context, say so clearly.
5. Be concise and precise.

**Answer:**
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"

# -- STREAMLIT APP CONFIG --
st.set_page_config(layout="wide", page_title="Compare Three PDFs")
st.title("📄 Ask Questions About Your PDF Versions")
st.markdown("Upload three versions of a PDF and ask Gemini to find differences across them.")

# -- SESSION STATE --
for key in ['original_db', 'amended_db', 'final_db', 'processed_files']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'processed_files' else []

# -- SIDEBAR: Upload PDFs --
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload exactly 3 PDFs: Original, Amended, Final (in that order)", 
        type="pdf", 
        accept_multiple_files=True,
        key="multi_upload"
    )
    
    if st.button("Process Documents", type="primary", use_container_width=True):
        if uploaded_files and len(uploaded_files) == 3:
            file1, file2, file3 = uploaded_files
            
            if [file.name for file in uploaded_files] == st.session_state.processed_files:
                st.toast("✅ Documents already processed.")
            else:
                with st.status("⚙️ Processing PDFs...", expanded=True) as status:
                    st.write("Extracting text...")
                    text1 = extract_text_from_pdf(file1.getvalue())
                    text2 = extract_text_from_pdf(file2.getvalue())
                    text3 = extract_text_from_pdf(file3.getvalue())

                    st.write("Chunking documents...")
                    chunks1 = create_document_chunks(text1, "Original")
                    chunks2 = create_document_chunks(text2, "Amended")
                    chunks3 = create_document_chunks(text3, "Final")

                    st.write("Creating vector stores...")
                    st.session_state.original_db = create_vector_db_from_chunks(chunks1)
                    st.session_state.amended_db = create_vector_db_from_chunks(chunks2)
                    st.session_state.final_db = create_vector_db_from_chunks(chunks3)

                    st.session_state.processed_files = [file.name for file in uploaded_files]
                    status.update(label="✅ Processing Complete!", state="complete")
        else:
            st.warning("⚠️ Please upload exactly 3 PDF files.")

# -- MAIN: Ask Questions --
st.header("2. Ask Your Question")

if all([st.session_state.original_db, st.session_state.amended_db, st.session_state.final_db]):
    st.success("Documents processed! You can now ask questions.")

    question = st.text_input(
        "e.g., 'How did clause 5 change from original to final?'",
        placeholder="Ask about changes across all three documents..."
    )

    if question:
        with st.spinner("🔍 Retrieving relevant context..."):
            r1 = st.session_state.original_db.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(question)
            r2 = st.session_state.amended_db.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(question)
            r3 = st.session_state.final_db.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(question)
            combined = r1 + r2 + r3

        with st.spinner("🤖 Gemini is analyzing..."):
            answer = get_answer_from_gemini(question, combined)
            st.info(f"**Answer:**\n\n{answer}")

        with st.expander("📚 View Retrieved Context"):
            for label, docs in zip(["Original", "Amended", "Final"], [r1, r2, r3]):
                for doc in docs:
                    st.markdown(f"**Source: {label}**")
                    st.code(doc.page_content)
                    st.markdown("---")
else:
    st.info("Upload and process all three documents in the sidebar to begin.")
