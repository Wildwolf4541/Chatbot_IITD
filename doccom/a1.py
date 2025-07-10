import os
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-1.5-flash")

# Load sentence transformer
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

st.set_page_config(page_title="PDF Semantic Comparator", layout="wide")
st.title("📄 PDF Semantic Comparator (No Docling)")

# --- PDF text extraction ---
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    os.unlink(tmp_path)
    return text.strip()

# --- Clause splitting using LangChain ---
def split_into_clauses(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 30]

# --- Find most similar clause ---
def find_best_match(clause, other_clauses):
    emb1 = sbert_model.encode([clause], convert_to_tensor=True)
    emb2 = sbert_model.encode(other_clauses, convert_to_tensor=True)
    scores = util.cos_sim(emb1, emb2)[0]
    top_idx = scores.argmax().item()
    return other_clauses[top_idx], scores[top_idx].item()

# --- Use Gemini to detect difference ---
def is_major_difference(cl1, cl2):
    prompt = f"""
You're a legal policy conflict detector.

Compare the two clauses below. If they have a major semantic difference (like in eligibility, penalty, or obligation), explain it.
If there's no major difference, reply: "No major difference."

Clause 1:
\"\"\"{cl1}\"\"\"

Clause 2:
\"\"\"{cl2}\"\"\"
"""
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from LLM: {e}"

# --- Streamlit Interface ---
pdf1 = st.file_uploader("Upload PDF 1", type="pdf")
pdf2 = st.file_uploader("Upload PDF 2", type="pdf")

if pdf1 and pdf2 and st.button("🔍 Compare PDFs"):
    with st.spinner("Reading and processing..."):

        text1 = extract_text_from_pdf(pdf1)
        text2 = extract_text_from_pdf(pdf2)

        clauses1 = split_into_clauses(text1)
        clauses2 = split_into_clauses(text2)

        diffs = []

        for cl1 in clauses1:
            cl2, score = find_best_match(cl1, clauses2)
            if score > 0.75:
                verdict = is_major_difference(cl1, cl2)
                if "no major difference" not in verdict.lower():
                    diffs.append({
                        "clause1": cl1,
                        "clause2": cl2,
                        "score": round(score, 3),
                        "difference": verdict
                    })

    if diffs:
        st.success(f"Found {len(diffs)} major semantic differences.")
        for i, d in enumerate(diffs, 1):
            with st.expander(f"🧠 Difference #{i} (Score: {d['score']})"):
                st.markdown(f"**📄 PDF1 Clause:**\n\n{d['clause1']}")
                st.markdown(f"**📄 PDF2 Clause:**\n\n{d['clause2']}")
                st.markdown(f"**⚠️ Major Difference:**\n\n{d['difference']}")
    else:
        st.info("No major differences found.")
