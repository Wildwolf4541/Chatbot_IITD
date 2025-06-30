import os
import re
import streamlit as st
import difflib
import tempfile
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# NEW: Import real embeddings from the LangChain Google integration
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -- LOAD ENV VARIABLES --
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -- SAFEGUARD FALLBACK --
if not api_key:
    st.error("❌ GOOGLE_API_KEY is not set. Please check your .env file or add it to your Streamlit secrets.")
    st.stop()

# -- CONFIGURE GEMINI & EMBEDDINGS --
try:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    # NEW: Use real embeddings for semantic similarity search
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to configure Google AI services: {e}")
    st.stop()

# -- CONSTANTS FOR THRESHOLDING (from your diagram) --
# NOTE: These thresholds are for FAISS's L2 distance. Lower is more similar.
# A score of 0.0 is a perfect match. You may need to tune these values.
UNCHANGED_THRESHOLD = 0.1  # Very similar, mark as unchanged
MODIFIED_THRESHOLD = 0.5   # Moderately similar, mark as modified

# -- UTILITIES --
def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def chunk_document(text):
    """Splits a long text into smaller document chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Increased chunk size for better context
        chunk_overlap=100
    )
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def create_vector_db(chunks):
    """Creates a FAISS vector database from document chunks using real embeddings."""
    if not chunks:
        return None
    try:
        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Failed to create vector database: {e}")
        return None

def compare_with_gemini(chunk1, chunk2):
    """Uses Gemini to get a detailed comparison of two text chunks."""
    prompt = f"""
As an expert legal and technical document analyst, your task is to provide a precise and concise comparison of the two text sections below.

Instructions:
1.  Identify ALL differences: word changes, numerical discrepancies, additions, deletions, and significant rephrasing.
2.  Summarize the core change in a single, clear sentence.
3.  Provide a bulleted list detailing each specific change.

---
### Original Text:
"{chunk1}"
---
### New Text:
"{chunk2}"
---

### Analysis:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"

# -- STREAMLIT APP --
st.set_page_config(layout="wide", page_title="Gemini PDF Comparator")
st.title("📄 Gemini-Powered PDF Document Comparator")
st.markdown(
    "Upload two versions of a PDF to see a detailed breakdown of **modified**, **added**, and **deleted** sections, analyzed by Gemini."
)

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("1️⃣ Upload Original PDF Document", type="pdf", key="file1")
with col2:
    file2 = st.file_uploader("2️⃣ Upload Amended PDF Document", type="pdf", key="file2")

if file1 and file2:
    with st.spinner("📚 Step 1/3: Extracting text from PDFs..."):
        text1 = extract_text_from_pdf(file1)
        text2 = extract_text_from_pdf(file2)

        if not text1 or not text2:
            st.error("Could not extract text from one or both PDFs. Please ensure they contain selectable text.")
            st.stop()

    with st.spinner("🧩 Step 2/3: Chunking and creating vector indexes..."):
        chunks_v1 = chunk_document(text1)
        chunks_v2 = chunk_document(text2)
        db_v1 = create_vector_db(chunks_v1)
        db_v2 = create_vector_db(chunks_v2)

        if not db_v1 or not db_v2:
            st.error("Failed to process documents into vector databases.")
            st.stop()

    with st.spinner("🔍 Step 3/3: Performing semantic comparison..."):
        modified_pairs = []
        deleted_chunks = []
        
        # Keep track of which chunks in v2 have been matched to a chunk in v1
        matched_v2_indices = set()

        # LOGIC: For each chunk in the original doc (v1), find its best match in the new doc (v2)
        for i, chunk1 in enumerate(chunks_v1):
            # Find the most similar chunk in db_v2 and get its content, index, and similarity score
            # We ask for k=1 to get the single best match.
            results = db_v2.similarity_search_with_score(chunk1.page_content, k=1)
            
            if not results:
                # This chunk from v1 has no reasonable match in v2, so it was deleted.
                deleted_chunks.append(chunk1.page_content)
                continue

            matched_doc, score = results[0]
            
            # The matched document in FAISS doesn't inherently know its original index.
            # We find it by comparing content. This is a bit inefficient but necessary.
            try:
                # Find the index of the matched document in the original list of chunks
                matched_index = [c.page_content for c in chunks_v2].index(matched_doc.page_content)
            except ValueError:
                # Should not happen if chunks are unique, but as a fallback:
                continue

            if score < UNCHANGED_THRESHOLD:
                # The chunks are almost identical. Mark as matched and continue.
                matched_v2_indices.add(matched_index)
            elif score < MODIFIED_THRESHOLD:
                # The chunks are similar but modified.
                modified_pairs.append({
                    "original": chunk1.page_content,
                    "amended": matched_doc.page_content
                })
                matched_v2_indices.add(matched_index)
            else:
                # The best match is still too different. Consider the original chunk deleted.
                deleted_chunks.append(chunk1.page_content)

        # LOGIC: Now find chunks that were added in v2
        # Any chunk in v2 that was NOT matched to a v1 chunk is an addition.
        added_chunks = [
            chunk2.page_content for i, chunk2 in enumerate(chunks_v2) 
            if i not in matched_v2_indices
        ]

    st.success("✅ Comparison complete!")
    st.markdown("---")

    # --- DISPLAY RESULTS ---
    st.header("📊 Comparison Summary")
    
    total_changes = len(modified_pairs) + len(added_chunks) + len(deleted_chunks)
    if total_changes == 0:
        st.success("✅ No significant differences were found between the two documents.")
    else:
        st.info(f"Found **{len(modified_pairs)}** modified section(s), **{len(added_chunks)}** added section(s), and **{len(deleted_chunks)}** deleted section(s).")

    if modified_pairs:
        with st.expander(f"✍️ Modified Sections ({len(modified_pairs)})", expanded=True):
            for i, pair in enumerate(modified_pairs):
                st.subheader(f"Modification #{i+1}")
                col_orig, col_amend = st.columns(2)
                with col_orig:
                    st.markdown("**📝 Original**")
                    st.info(pair["original"])
                with col_amend:
                    st.markdown("**🆕 Amended**")
                    st.warning(pair["amended"])
                
                with st.spinner(f"🧠 Gemini is analyzing modification #{i+1}..."):
                    diff_summary = compare_with_gemini(pair["original"], pair["amended"])
                
                st.markdown("---")
                st.markdown("**🤖 Gemini's Analysis:**")
                st.write(diff_summary)
                st.markdown("---")


    if added_chunks:
        with st.expander(f"➕ Added Sections ({len(added_chunks)})", expanded=False):
            for i, chunk in enumerate(added_chunks):
                st.subheader(f"Addition #{i+1}")
                st.success(chunk)
                st.markdown("---")

    if deleted_chunks:
        with st.expander(f"➖ Deleted Sections ({len(deleted_chunks)})", expanded=False):
            for i, chunk in enumerate(deleted_chunks):
                st.subheader(f"Deletion #{i+1}")
                st.error(chunk)
                st.markdown("---")