import streamlit as st
import os
import io
import fitz  # PyMuPDF
import difflib
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import html

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(layout="wide", page_title="PDF Comparison Tool")

# Modern UI Styling
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #F0F2F6;
    }
    
    /* Card-based results styling */
    .diff-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left-width: 6px;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    .diff-card-equal { border-left-color: #6c757d; }
    .diff-card-insert { border-left-color: #198754; } /* Green */
    .diff-card-delete { border-left-color: #dc3545; } /* Red */
    .diff-card-replace { border-left-color: #ffc107; } /* Yellow */
    
    /* Word-level highlighting */
    .del-word { text-decoration: line-through; background-color: #f8d7da; border-radius: 3px; padding: 2px 0; }
    .ins-word { background-color: #d1e7dd; border-radius: 3px; padding: 2px 0; }
    
    /* Result text font */
    pre {
        font-family: 'SF Mono', 'Courier New', monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.375rem;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. COMPARATOR ENGINE LOGIC (Identical core logic)
# ==============================================================================

@st.cache_data
def extract_text_from_pdf(file_contents: bytes) -> str:
    try:
        with fitz.open(stream=file_contents, filetype="pdf") as doc:
            return "".join(page.get_text("text", sort=True) for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF content: {e}")
        return ""

def _chunk_text_intelligently(text: str) -> list[str]:
    paragraphs = []
    current_paragraph = []
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph: paragraphs.append(" ".join(current_paragraph)); current_paragraph = []
            continue
        current_paragraph.append(line)
        if line.endswith('.') and not line.lower().endswith(('etc.', 'i.e.', 'e.g.')):
            if current_paragraph: paragraphs.append(" ".join(current_paragraph)); current_paragraph = []
    if current_paragraph: paragraphs.append(" ".join(current_paragraph))
    if len(paragraphs) < 5: paragraphs = [p.replace('\n', ' ').strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def normalize_chunk(text: str) -> str:
    text = text.lower()
    noise_patterns = [r'the gazette of india', r'extraordinary', r'part ii—', r'published by authority', r'\[\d+th.*?, \d{4}\]', r'registered no\. dl.*']
    for pattern in noise_patterns: text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text); text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_word_level_diff(text1: str, text2: str) -> tuple:
    words1 = re.split(r'(\s+)', text1); words2 = re.split(r'(\s+)', text2)
    matcher = difflib.SequenceMatcher(None, words1, words2, autojunk=False)
    diff1, diff2 = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk1, chunk2 = "".join(words1[i1:i2]), "".join(words2[j1:j2])
        if tag == 'equal': diff1.append(('equal', chunk1)); diff2.append(('equal', chunk2))
        elif tag == 'insert': diff2.append(('insert', chunk2))
        elif tag == 'delete': diff1.append(('delete', chunk1))
        elif tag == 'replace': diff1.append(('delete', chunk1)); diff2.append(('insert', chunk2))
    return diff1, diff2

@st.cache_data
def compare_text_content(text1: str, text2: str) -> list:
    raw_paras1 = _chunk_text_intelligently(text1); raw_paras2 = _chunk_text_intelligently(text2)
    cleaned_paras1 = [normalize_chunk(p) for p in raw_paras1]; cleaned_paras2 = [normalize_chunk(p) for p in raw_paras2]
    matcher = difflib.SequenceMatcher(None, cleaned_paras1, cleaned_paras2, autojunk=False)
    diff_result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2): diff_result.append(('equal', raw_paras1[i]))
        elif tag == 'insert':
            for i in range(j1, j2): diff_result.append(('insert', raw_paras2[i]))
        elif tag == 'delete':
            for i in range(i1, i2): diff_result.append(('delete', raw_paras1[i]))
        elif tag == 'replace':
            num_paras = max(i2 - i1, j2 - j1)
            for i in range(num_paras):
                para1 = raw_paras1[i1 + i] if (i1 + i) < i2 else ""; para2 = raw_paras2[j1 + i] if (j1 + i) < j2 else ""
                if not para1: diff_result.append(('insert', para2))
                elif not para2: diff_result.append(('delete', para1))
                else: diff_result.append(('replace', *get_word_level_diff(para1, para2)))
    return diff_result

def create_diff_pdf(diff_result: list, pdf1_name: str, pdf2_name: str, similarity_score: float) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin, gutter = 0.75 * inch, 0.25 * inch
    col_width = (width - 2 * margin - gutter) / 2
    base_style = ParagraphStyle('base', fontName='Courier', fontSize=8, leading=10, alignment=TA_LEFT)
    delete_style = "<font backColor='#fdb8c0'><strike>"; insert_style = "<font backColor='#a6f2b9'>"
    
    def setup_page(canvas_obj, y_pos, page_num):
        canvas_obj.setFont("Helvetica-Bold", 12)
        if page_num == 1:
            canvas_obj.drawCentredString(width / 2.0, y_pos, "PDF Comparison Report")
            y_pos -= 20
            canvas_obj.setFont("Helvetica", 10)
            canvas_obj.drawCentredString(width / 2.0, y_pos, f"Overall Similarity: {similarity_score:.2%}")
            y_pos -= 30
        canvas_obj.setFont("Helvetica-Bold", 10)
        canvas_obj.drawString(margin, y_pos, f"Original ({pdf1_name})")
        canvas_obj.drawString(margin + col_width + gutter, y_pos, f"Changed ({pdf2_name})")
        y_pos -= 20; canvas_obj.line(margin, y_pos + 5, width - margin, y_pos + 5); y_pos -= 5
        return y_pos
    
    y = height - margin; page_count = 1
    y = setup_page(c, y, page_count)

    for line_type, *content in diff_result:
        left_text, right_text = "", ""
        def escape_reportlab(text): return text.replace('&', '&').replace('<', '<').replace('>', '>')
        if line_type == 'equal': left_text = right_text = escape_reportlab(content[0] or '')
        elif line_type == 'delete': left_text = f"{delete_style}{escape_reportlab(content[0] or '')}</strike></font>"
        elif line_type == 'insert': right_text = f"{insert_style}{escape_reportlab(content[0] or '')}</font>"
        elif line_type == 'replace':
            for tag, text in content[0]: left_text += f"{delete_style}{escape_reportlab(text)}</strike></font>" if tag == 'delete' else escape_reportlab(text)
            for tag, text in content[1]: right_text += f"{insert_style}{escape_reportlab(text)}</font>" if tag == 'insert' else escape_reportlab(text)
        
        p_left = Paragraph(left_text, style=base_style); p_right = Paragraph(right_text, style=base_style)
        left_h, right_h = p_left.wrap(col_width, height)[1], p_right.wrap(col_width, height)[1]
        line_height = max(left_h, right_h, 10) + 4
        if y - line_height < margin:
            c.showPage(); page_count += 1; y = height - margin; y = setup_page(c, y, page_count)
        
        if line_type == 'delete': c.setFillColorRGB(1, 0.93, 0.94); c.rect(margin, y - line_height, col_width, line_height, stroke=0, fill=1)
        elif line_type == 'insert': c.setFillColorRGB(0.9, 1, 0.93); c.rect(margin + col_width + gutter, y - line_height, col_width, line_height, stroke=0, fill=1)
        p_left.drawOn(c, margin + 2, y - line_height + 2); p_right.drawOn(c, margin + col_width + gutter + 2, y - line_height + 2)
        y -= line_height + 2
    
    c.save(); buffer.seek(0); return buffer.getvalue()


# ==============================================================================
# 3. STREAMLIT APPLICATION UI AND LOGIC
# ==============================================================================

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("PDF Comparator")
    st.markdown("Upload two PDF files to start the comparison.")
    
    pdf1_file = st.file_uploader("1. Upload Original Document", type="pdf", key="pdf1")
    pdf2_file = st.file_uploader("2. Upload Changed Document", type="pdf", key="pdf2")

    # The button is only active when both files are uploaded
    compare_button = st.button(
        "🚀 Compare Documents", 
        type="primary", 
        use_container_width=True, 
        disabled=not (pdf1_file and pdf2_file)
    )

# --- Main Display Area ---
st.title("Intelligent Document Comparison")

if compare_button:
    st.cache_data.clear()
    
    with st.spinner("Analyzing documents, this may take a moment..."):
        text1 = extract_text_from_pdf(pdf1_file.getvalue())
        text2 = extract_text_from_pdf(pdf2_file.getvalue())

        if text1 is not None and text2 is not None:
            # NEW: Calculate overall similarity score on raw text
            matcher = difflib.SequenceMatcher(None, text1, text2)
            st.session_state.similarity_score = matcher.ratio()
            
            # Run the detailed comparison engine
            st.session_state.diff_result = compare_text_content(text1, text2)
            st.session_state.pdf1_name = pdf1_file.name
            st.session_state.pdf2_name = pdf2_file.name
        else:
            st.error("Failed to extract text from one or both files.")
            st.session_state.clear() # Clear state on error

# --- Results Display ---
if 'diff_result' in st.session_state:
    st.subheader("Overall Similarity")
    
    # Display the similarity score using st.metric and a progress bar
    score = st.session_state.similarity_score
    st.metric(label="Textual Similarity Score", value=f"{score:.2%}")
    st.progress(score)
    st.divider()

    # Header with Download Button
    results_header_cols = st.columns([0.75, 0.25])
    with results_header_cols[0]:
        st.subheader("Detailed Comparison")
    with results_header_cols[1]:
        pdf_bytes = create_diff_pdf(st.session_state.diff_result, st.session_state.pdf1_name, st.session_state.pdf2_name, score)
        st.download_button(
            label="⬇️ Download Report",
            data=pdf_bytes,
            file_name=f"comparison_report_{st.session_state.pdf1_name}_vs_{st.session_state.pdf2_name}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    if not st.session_state.diff_result:
        st.success("✅ The documents appear to have identical text content.")
    else:
        for item in st.session_state.diff_result:
            type, *content = item
            
            if type == 'equal':
                # Skip equal blocks to focus on changes
                continue
            
            left_html, right_html = "", ""
            if type == 'delete':
                left_html = f'<pre>{html.escape(content[0])}</pre>'
            elif type == 'insert':
                right_html = f'<pre>{html.escape(content[0])}</pre>'
            elif type == 'replace':
                for tag, text in content[0]: left_html += f'<span class="del-word">{html.escape(text)}</span>' if tag == 'delete' else html.escape(text)
                for tag, text in content[1]: right_html += f'<span class="ins-word">{html.escape(text)}</span>' if tag == 'insert' else html.escape(text)
                left_html = f"<pre>{left_html}</pre>"; right_html = f"<pre>{right_html}</pre>"
            
            st.markdown(f'<div class="diff-card diff-card-{type}">', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("<b>Original</b>", unsafe_allow_html=True)
                st.markdown(left_html, unsafe_allow_html=True)
            with res_col2:
                st.markdown("<b>Changed</b>", unsafe_allow_html=True)
                st.markdown(right_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Display a welcome message if no comparison has been run
    st.info("👋 Welcome! Please upload two PDF documents in the sidebar and click 'Compare' to see the magic happen.")