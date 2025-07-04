import os
import streamlit as st
import tempfile
import pdfplumber
from dotenv import load_dotenv
import numpy as np
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
    # Increased chunk size and overlap for better context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]

def create_vector_db_from_chunks(chunks):
    with st.spinner(f"🔗 Creating vector store for {chunks[0].metadata['source']}..."):
        return FAISS.from_documents(chunks, embeddings)

def align_chunks_across_docs(docs_by_source):
    sources = list(docs_by_source.keys())
    all_docs = []
    all_embeddings = []

    for source in sources:
        docs = docs_by_source[source]
        for doc in docs:
            emb = embeddings.embed_query(doc.page_content)
            all_docs.append((source, doc))
            all_embeddings.append(emb)

    all_embeddings = np.array(all_embeddings)
    aligned_groups = []

    for source in sources:
        base_docs = docs_by_source[source]
        for doc in base_docs:
            emb = embeddings.embed_query(doc.page_content)
            similarities = np.dot(all_embeddings, emb) / (np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(emb) + 1e-10)

            best_matches = []
            used_sources = {source}
            top_indices = np.argsort(similarities)[::-1]
            for idx in top_indices:
                match_source, match_doc = all_docs[idx]
                if match_source not in used_sources:
                    best_matches.append(match_doc)
                    used_sources.add(match_source)
                if len(used_sources) == len(sources):
                    break

            aligned_group = [doc] + best_matches
            aligned_groups.append(aligned_group)

    return aligned_groups

def aligned_groups_to_hashable(aligned_groups):
    return tuple(
        tuple(doc.page_content for doc in group)
        for group in aligned_groups
    )

@st.cache_data
def summarize_aligned_groups(hashable_aligned_groups, sources):
    summaries = []
    for idx, group_texts in enumerate(hashable_aligned_groups):
        combined_text = "\n---\n".join(
            f"[{sources[i]}]: {text}" for i, text in enumerate(group_texts)
        )
        prompt = f"""
You are an expert document summarization assistant.

Given the following aligned sections from different versions of a document, create a clear, concise summary that:

- Highlights key differences between the versions
- Specifies any additions, removals, or modifications
- Mentions if content is unchanged where relevant
- Avoids repetition and focuses on what changed
- Ensure your summaries are consistent with each other.

Text to summarize:
{combined_text}

Provide the summary in bullet points or short paragraphs, clearly indicating changes.

Summary:
"""
        try:
            response = gemini_model.generate_content(prompt)
            summary = response.text.strip()
        except Exception as e:
            summary = f"❌ Error summarizing group {idx}: {e}"
        summaries.append(summary)
    return summaries

def generate_global_summary(summaries):
    combined = "\n\n".join(summaries)
    prompt = f"""
You are a document comparison expert.

Given the following summaries of aligned document sections from multiple document versions, produce a consistent and coherent global summary of changes across versions. Avoid contradictions and ensure clarity.

Summaries:
{combined}

Global Summary:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating global summary: {e}"

@st.cache_data(show_spinner=False)
def get_cached_answer(prompt: str) -> str:
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
    st.session_state.vector_dbs = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "aligned_groups" not in st.session_state:
    st.session_state.aligned_groups = None
if "summaries" not in st.session_state:
    st.session_state.summaries = None
if "global_summary" not in st.session_state:
    st.session_state.global_summary = None

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
                st.session_state.vector_dbs = {}
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
                    st.session_state.aligned_groups = None
                    st.session_state.summaries = None
                    st.session_state.global_summary = None
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
            docs_by_source = {}
            for name, vector_db in st.session_state.vector_dbs.items():
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # reduced k for quota
                docs = retriever.get_relevant_documents(question)
                docs_by_source[name] = docs

        with st.spinner("🔄 Aligning document chunks..."):
            aligned_groups = align_chunks_across_docs(docs_by_source)
            st.session_state.aligned_groups = aligned_groups
            sources = list(docs_by_source.keys())
        
        hashable_groups = aligned_groups_to_hashable(aligned_groups)

        with st.spinner("📝 Summarizing aligned chunks..."):
            summaries = summarize_aligned_groups(hashable_groups, sources)
            st.session_state.summaries = summaries

        with st.spinner("📝 Generating global summary..."):
            global_summary = generate_global_summary(summaries)
            st.session_state.global_summary = global_summary

        prompt = f"""
You are an expert document comparison assistant.

User's question: {question}

Here is a global summary of changes across the uploaded document versions:
{global_summary}

Based on this summary, answer the user's question clearly and concisely.
Use bullet points or numbered lists where helpful.

Answer:
"""
        with st.spinner("🤖 Gemini is generating the final answer..."):
            answer = get_cached_answer(prompt)

        st.info(f"**Answer:**\n\n{answer}")

        with st.expander("📚 View Retrieved Context (Aligned Groups)"):
            for i, group in enumerate(aligned_groups):
                st.markdown(f"### Aligned Group {i+1}")
                for doc in group:
                    st.markdown(f"**Source: {doc.metadata['source']}**")
                    st.code(doc.page_content)
                st.markdown("---")

        with st.expander("📝 Global Summary of Document Differences"):
            st.markdown(global_summary)

else:
    st.info("Upload and process at least two PDF documents in the sidebar to begin.")
