import streamlit as st
import pdfplumber
import difflib
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Document Comparison")

# --- CSS for Highlighting Differences ---
CSS = """
<style>
    .deleted {
        background-color: #ffc9c9; /* light red */
        text-decoration: line-through;
    }
    .inserted {
        background-color: #d4edda; /* light green */
    }
    .container {
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        white-space: pre-wrap;   /* Preserve whitespace and wrap text */
        word-wrap: break-word;   /* Break long words to prevent overflow */
        font-size: 14px;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- Utility Functions ---

@st.cache_data
def extract_text_from_pdf(file_contents):
    """
    Extracts text from the bytes of an uploaded PDF file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_contents)
        tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    finally:
        os.remove(tmp_path)
    return text

def highlight_word_diff(line1, line2):
    """
    Performs a word-level diff on two lines and returns HTML-formatted strings
    with highlights for changed words.
    """
    matcher = difflib.SequenceMatcher(None, line1.split(), line2.split())
    html1, html2 = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html1.append(" ".join(matcher.a[i1:i2]))
            html2.append(" ".join(matcher.b[j1:j2]))
        elif tag == 'delete':
            html1.append(f'<span class="deleted">{" ".join(matcher.a[i1:i2])}</span>')
        elif tag == 'insert':
            html2.append(f'<span class="inserted">{" ".join(matcher.b[j1:j2])}</span>')
        elif tag == 'replace':
            html1.append(f'<span class="deleted">{" ".join(matcher.a[i1:i2])}</span>')
            html2.append(f'<span class="inserted">{" ".join(matcher.b[j1:j2])}</span>')
    return " ".join(html1), " ".join(html2)

def generate_side_by_side_diff(text1, text2):
    """
    Generates two HTML strings representing a side-by-side comparison of two texts.
    """
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    html_lines1, html_lines2 = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for line in lines1[i1:i2]:
                html_lines1.append(line)
                html_lines2.append(line)
        else:
            block1, block2 = lines1[i1:i2], lines2[j1:j2]
            max_len = max(len(block1), len(block2))
            for i in range(max_len):
                line1 = block1[i] if i < len(block1) else ""
                line2 = block2[i] if i < len(block2) else ""
                if line1 == line2:
                    h1, h2 = line1, line2
                else:
                    h1, h2 = highlight_word_diff(line1, line2)
                html_lines1.append(h1 if line1 else ' ')
                html_lines2.append(h2 if line2 else ' ')

    return "<br>".join(html_lines1), "<br>".join(html_lines2)

# --- Streamlit App UI ---

st.title("📄 Visual Document Comparison")
st.markdown("Upload two versions of a PDF document to see the differences highlighted side-by-side.")

col1, col2 = st.columns(2)
with col1:
    st.header("Original Document")
    file1 = st.file_uploader("Upload the first PDF", type="pdf", key="file1")

with col2:
    st.header("Revised Document")
    file2 = st.file_uploader("Upload the second PDF", type="pdf", key="file2")

if file1 and file2:
    if st.button("Compare Documents", type="primary", use_container_width=True):
        with st.spinner("Analyzing documents..."):
            text1 = extract_text_from_pdf(file1.getvalue())
            text2 = extract_text_from_pdf(file2.getvalue())

            # --- NEW: Calculate and display similarity score ---
            # Create a SequenceMatcher instance with the full text of both documents
            matcher = difflib.SequenceMatcher(None, text1, text2)
            similarity_score = matcher.ratio()
            similarity_percentage_str = f"{similarity_score * 100:.1f}%"

            # Generate the detailed visual diff
            diff1, diff2 = generate_side_by_side_diff(text1, text2)

        # Display the overall score first
        st.header("Overall Similarity")
        st.metric(
            label="Document Similarity Score",
            value=similarity_percentage_str,
            help="This score represents how similar the extracted text from the two documents is. 100% means identical text."
        )
        st.progress(similarity_score)
        st.markdown("---") # Visual separator

        # Display the detailed comparison
        if text1 or text2:
            st.header("Detailed Comparison")
            display_col1, display_col2 = st.columns(2)
            
            with display_col1:
                st.markdown(f"**{file1.name}**")
                st.markdown(f'<div class="container">{diff1}</div>', unsafe_allow_html=True)
            
            with display_col2:
                st.markdown(f"**{file2.name}**")
                st.markdown(f'<div class="container">{diff2}</div>', unsafe_allow_html=True)
        else:
            st.error("Could not extract text from one or both PDFs. They might be image-based or empty.")
else:
    st.info("Please upload both documents to enable comparison.")