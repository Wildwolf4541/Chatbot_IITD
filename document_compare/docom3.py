import os
import streamlit as st
import difflib
import tempfile
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # ✅ updated import
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment and API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key or api_key.strip() == "":
    st.error("❌ GOOGLE_API_KEY is not set.")
    st.stop()

# Configure Gemini + Embeddings
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Constants
MODIFIED_THRESHOLD = 0.75
UNCHANGED_THRESHOLD = 0.99

# Functions
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def chunk_document(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk, metadata={"index": i}) for i, chunk in enumerate(splitter.split_text(text))]

def create_vector_db(chunks):
    return FAISS.from_documents(chunks, embeddings)

def token_diff_ratio(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()

def number_diff(a, b):
    import re
    return re.findall(r"\d+", a) != re.findall(r"\d+", b)

def compare_with_gemini(chunk1, chunk2):
    prompt = f"""
Compare these two text sections and identify **all changes** including word swaps, number edits, rephrasing, additions, or deletions:

--- Original ---
{chunk1}

--- New ---
{chunk2}

Give a concise summary and bullet-point list of all changes.
"""
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini Error: {e}"

# Streamlit UI
st.set_page_config(layout="wide", page_title="📄 Gemini-Powered PDF Comparator")
st.title("📄 PDF Comparator (LangChain + Gemini)")
st.markdown("Upload two PDFs and detect even the smallest changes using semantic similarity + LLMs.")

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload Original PDF", type="pdf", key="f1")
with col2:
    file2 = st.file_uploader("Upload Modified PDF", type="pdf", key="f2")

if file1 and file2:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
        tmp1.write(file1.read())
        path1 = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp2.write(file2.read())
        path2 = tmp2.name

    with st.spinner("🔍 Extracting & Chunking..."):
        text1 = extract_text_from_pdf(path1)
        text2 = extract_text_from_pdf(path2)

        chunks1 = chunk_document(text1)
        chunks2 = chunk_document(text2)

        db2 = create_vector_db(chunks2)

    st.success("✅ Document chunking & embedding complete.")
    st.markdown("---")

    for idx, doc1 in enumerate(chunks1):
        original = doc1.page_content
        sim_results = db2.similarity_search_with_relevance_scores(original, k=1)

        if sim_results:
            matched_doc, score = sim_results[0]
            amended = matched_doc.page_content

            # Decide whether to compare using LLM
            if score < UNCHANGED_THRESHOLD:
                with st.spinner(f"🔍 Gemini analyzing chunk #{idx+1}..."):
                    diff = compare_with_gemini(original, amended)
            else:
                diff = "✅ No meaningful change (very high similarity)."
        else:
            amended = "❌ No match found."
            diff = "This section may have been deleted."

        with st.expander(f"🧩 Chunk #{idx + 1}"):
            st.markdown("**📝 Original:**")
            st.code(original, language="text")

            st.markdown("**🆕 Amended:**")
            st.code(amended, language="text")

            st.markdown("**🧠 Gemini Analysis:**")
            st.success(diff)
