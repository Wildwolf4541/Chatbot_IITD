import os
import json
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util

# -- LOAD & CONFIGURE APIs --
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY is not set. Please check your .env file or add it to your Streamlit secrets.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"❌ Failed to configure services: {e}")
    st.stop()

# --- UTILITY FUNCTIONS (Unchanged from previous version) ---
def extract_text_from_pdf(uploaded_file):
    if uploaded_file is None: return ""
    try:
        all_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: all_text += page_text + "\n"
        return all_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}"); return None

def chunk_text_with_gemini(text: str):
    if not text: return []
    prompt = f"""
    You are an expert legal assistant specializing in document structuring.
    Your task is to parse the following document text and split it into a list of logical clauses or sections.

    Instructions:
    1.  Analyze the text and identify distinct clauses, sections, or paragraphs.
    2.  For each clause, extract its title or number (e.g., "1.1 Introduction", "Section A. Definitions"). If no title exists, create a concise, descriptive title from the first few words of the clause.
    3.  Format your output as a valid JSON array of objects.
    4.  Each JSON object must have two keys: "clause_title" and "clause_text".
    5.  Ensure the entire original text is included in the output, distributed among the clauses. Do not omit any part of the text.

    Here is the document text:
    ---
    {text}
    ---
    """
    try:
        response = gemini_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        clauses_data = json.loads(response.text)
        documents = []
        for i, clause in enumerate(clauses_data):
            doc = Document(page_content=clause.get("clause_text", ""), metadata={"clause_title": clause.get("clause_title", f"Clause {i+1}"), "clause_id": i})
            documents.append(doc)
        return documents
    except Exception as e:
        st.error(f"Error chunking text with Gemini: {e}"); st.warning("Falling back to simple paragraph splitting.")
        paragraphs = text.split('\n\n')
        return [Document(page_content=p, metadata={"clause_title": f"Paragraph {i+1}", "clause_id": i}) for i, p in enumerate(paragraphs) if p.strip()]

def align_clauses(docs1, docs2, threshold=0.6):
    if not docs1 or not docs2: return [], docs1, docs2
    embeddings1 = sbert_model.encode([d.page_content for d in docs1], convert_to_tensor=True)
    embeddings2 = sbert_model.encode([d.page_content for d in docs2], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    matched_pairs = []; unmatched_indices_2 = set(range(len(docs2)))
    for i in range(len(docs1)):
        best_score = -1; best_j = -1
        for j in list(unmatched_indices_2):
            score = cos_scores[i][j].item()
            if score > best_score: best_score = score; best_j = j
        if best_score >= threshold:
            matched_pairs.append((docs1[i], docs2[best_j], best_score))
            if best_j in unmatched_indices_2: unmatched_indices_2.remove(best_j)
    matched_indices_1 = {docs1.index(pair[0]) for pair in matched_pairs}
    deleted_docs = [doc for i, doc in enumerate(docs1) if i not in matched_indices_1]
    added_docs = [docs2[j] for j in unmatched_indices_2]
    return matched_pairs, added_docs, deleted_docs

def detect_conflict_gemini(clause1_text, clause2_text):
    prompt = f"""
    You are a meticulous legal document analysis assistant. Your task is to compare two clauses and provide a clear, concise summary of the differences.
    Instructions:
    1.  If the clauses are semantically identical (ignoring minor whitespace changes), respond with ONLY the phrase "No substantive changes detected."
    2.  If there are differences, summarize the core change in a single, clear sentence.
    3.  Then, provide a bulleted list detailing each specific modification, addition, or deletion.
    ---
    ### Original Clause:
    "{clause1_text}"
    ---
    ### Amended Clause:
    "{clause2_text}"
    ---
    ### Analysis:
    """
    try:
        response = gemini_model.generate_content(prompt); return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Gemini Clause Comparator")
st.title("📄 PDF Clause Conflict Detector")
st.markdown("This tool uses AI to intelligently segment documents into clauses, aligns them semantically, and uses Gemini to highlight differences.")

uploaded_file1 = st.file_uploader("1️⃣ Upload Original PDF", type="pdf")
uploaded_file2 = st.file_uploader("2️⃣ Upload Amended PDF", type="pdf")

if uploaded_file1 and uploaded_file2:
    text1 = extract_text_from_pdf(uploaded_file1)
    text2 = extract_text_from_pdf(uploaded_file2)

    if text1 and text2:
        with st.spinner("🤖 Intelligently chunking PDFs into clauses..."):
            docs1 = chunk_text_with_gemini(text1)
            docs2 = chunk_text_with_gemini(text2)
        st.success(f"Extracted {len(docs1)} clauses from PDF 1 and {len(docs2)} clauses from PDF 2.")

        with st.spinner("🧠 Aligning clauses using semantic similarity..."):
            aligned, added, deleted = align_clauses(docs1, docs2)
        
        # --- NEW LOGIC: Pre-sort clauses before displaying ---
        modified_pairs = []
        unchanged_pairs = []
        
        with st.spinner("🤖 Analyzing aligned clauses with Gemini..."):
            for doc1, doc2, sim in aligned:
                # Get the detailed diff from Gemini for every pair
                conflict_summary = detect_conflict_gemini(doc1.page_content, doc2.page_content)
                
                # Sort the pair into the correct bucket based on the analysis
                if "No substantive changes detected" in conflict_summary and sim > 0.98:
                    unchanged_pairs.append((doc1, doc2, sim))
                else:
                    modified_pairs.append((doc1, doc2, sim, conflict_summary))

        st.success(f"Analysis complete: Found {len(modified_pairs)} modified, {len(unchanged_pairs)} unchanged, {len(added)} added, and {len(deleted)} deleted clauses.")
        st.markdown("---")
        
        # --- NEW DISPLAY LOGIC: Use separate, non-nested expanders ---

        if modified_pairs:
            with st.expander(f"✍️ Modified Clauses ({len(modified_pairs)})", expanded=True):
                # Sort by similarity, lowest first to see biggest changes
                modified_pairs.sort(key=lambda x: x[2])
                for doc1, doc2, sim, conflict_summary in modified_pairs:
                    st.subheader(f"Clause: '{doc1.metadata.get('clause_title', 'N/A')}'")
                    st.caption(f"Semantic Similarity: {sim:.2f}")
                    
                    st.write(conflict_summary)
                    
                    col1, col2 = st.columns(2)
                    col1.info(f"**Original:**\n\n{doc1.page_content}")
                    col2.warning(f"**Amended:**\n\n{doc2.page_content}")
                    st.markdown("---")
        
        if added:
            with st.expander(f"➕ Added Clauses ({len(added)})"):
                 for doc in added:
                      st.subheader(f"Clause: '{doc.metadata.get('clause_title', 'N/A')}'")
                      st.success(doc.page_content)
                      st.markdown("---")

        if deleted:
            with st.expander(f"➖ Deleted Clauses ({len(deleted)})"):
                 for doc in deleted:
                      st.subheader(f"Clause: '{doc.metadata.get('clause_title', 'N/A')}'")
                      st.error(doc.page_content)
                      st.markdown("---")

        # if unchanged_pairs:
        #     with st.expander(f"✅ Unchanged Clauses ({len(unchanged_pairs)})"):
        #         for doc1, doc2, sim in unchanged_pairs:
        #             st.subheader(f"Clause: '{doc1.metadata.get('clause_title', 'N/A')}'")
        #             st.caption(f"Semantic Similarity: {sim:.2f}")
        #             # Here we just show the text directly, no need for another expander
        #             st.text(doc1.page_content)
        #             st.markdown("---")