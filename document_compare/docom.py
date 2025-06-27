import os
import streamlit as st
import difflib
import tempfile
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FakeEmbeddings

# -- LOAD ENV VARIABLES --
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -- SAFEGUARD FALLBACK --
if not api_key or api_key.strip() == "":
    st.error("❌ GOOGLE_API_KEY is not set. Please check your .env file.")
    st.stop()

# -- CONFIGURE GEMINI --
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# -- UTILITIES --
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def chunk_document(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def create_vector_db(chunks):
    embeddings = FakeEmbeddings(size=1536)  # Placeholder
    db = FAISS.from_documents(chunks, embeddings)
    return db

def token_diff_ratio(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()

def number_diff(a, b):
    import re
    return re.findall(r"\d+", a) != re.findall(r"\d+", b)

def compare_with_gemini(chunk1, chunk2):
    prompt = f"""
Compare these two text sections and identify all changes (word changes, number changes, additions, deletions, paraphrasing):

--- Original ---
{chunk1}

--- New ---
{chunk2}
"""
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"

# -- STREAMLIT APP --
st.set_page_config(layout="wide")
st.title("📄 Gemini-Powered PDF Document Comparator")

file1 = st.file_uploader("Upload Original PDF Document", type="pdf", key="file1")
file2 = st.file_uploader("Upload Amended PDF Document", type="pdf", key="file2")

if file1 and file2:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
        tmp1.write(file1.read())
        path1 = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp2.write(file2.read())
        path2 = tmp2.name

    with st.spinner("📚 Extracting and chunking PDF documents..."):
        text1 = extract_text_from_pdf(path1)
        text2 = extract_text_from_pdf(path2)

        chunks_v1 = chunk_document(text1)
        chunks_v2 = chunk_document(text2)
        db_v1 = create_vector_db(chunks_v1)
        db_v2 = create_vector_db(chunks_v2)
        retriever = db_v2.as_retriever(search_kwargs={"k": 1})

    st.success("✅ Vector stores created from PDFs.")
    st.markdown("---")

    for idx, doc in enumerate(chunks_v1):
        original_chunk = doc.page_content
        matched_chunk = retriever.get_relevant_documents(original_chunk)[0].page_content

        ratio = token_diff_ratio(original_chunk, matched_chunk)
        num_changed = number_diff(original_chunk, matched_chunk)

        if ratio < 0.999 or num_changed:
            diff = compare_with_gemini(original_chunk, matched_chunk)
        else:
            diff = "✅ No meaningful change detected."

        with st.expander(f"🔍 Chunk #{idx + 1} Comparison"):
            st.markdown("**📝 Original:**")
            st.code(original_chunk)
            st.markdown("**🆕 Amended:**")
            st.code(matched_chunk)
            st.markdown("**🧠 Detected Changes:**")
            st.success(diff)
