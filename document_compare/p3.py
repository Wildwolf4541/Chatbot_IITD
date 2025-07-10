import os
import uuid
import io
import fitz  # PyMuPDF
import difflib
import re
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from waitress import serve # Use a production-ready server

# ==============================================================================
# 1. FLASK APPLICATION SETUP
# ==============================================================================
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================================================================
# 2. COMPARATOR ENGINE LOGIC (No changes needed here)
# ==============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text("text", sort=True)
            return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def _chunk_text_intelligently(text: str) -> list[str]:
    paragraphs = []
    current_paragraph = []
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            continue
        current_paragraph.append(line)
        if line.endswith('.') and not line.lower().endswith(('etc.', 'i.e.', 'e.g.')):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    if len(paragraphs) < 5:
        paragraphs = [p.replace('\n', ' ').strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def normalize_chunk(text: str) -> str:
    text = text.lower()
    noise_patterns = [r'the gazette of india', r'extraordinary', r'part ii—', r'published by authority', r'\[\d+th.*?, \d{4}\]', r'registered no\. dl.*']
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_word_level_diff(text1: str, text2: str) -> tuple:
    words1 = re.split(r'(\s+)', text1)
    words2 = re.split(r'(\s+)', text2)
    matcher = difflib.SequenceMatcher(None, words1, words2, autojunk=False)
    diff1, diff2 = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk1, chunk2 = "".join(words1[i1:i2]), "".join(words2[j1:j2])
        if tag == 'equal':
            diff1.append(('equal', chunk1))
            diff2.append(('equal', chunk2))
        elif tag == 'insert':
            diff2.append(('insert', chunk2))
        elif tag == 'delete':
            diff1.append(('delete', chunk1))
        elif tag == 'replace':
            diff1.append(('delete', chunk1))
            diff2.append(('insert', chunk2))
    return diff1, diff2

def compare_pdfs(pdf_path1: str, pdf_path2: str) -> list:
    text1 = extract_text_from_pdf(pdf_path1)
    text2 = extract_text_from_pdf(pdf_path2)
    raw_paras1 = _chunk_text_intelligently(text1)
    raw_paras2 = _chunk_text_intelligently(text2)
    cleaned_paras1 = [normalize_chunk(p) for p in raw_paras1]
    cleaned_paras2 = [normalize_chunk(p) for p in raw_paras2]
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
                para1 = raw_paras1[i1 + i] if (i1 + i) < i2 else ""
                para2 = raw_paras2[j1 + i] if (j1 + i) < j2 else ""
                if not para1: diff_result.append(('insert', para2))
                elif not para2: diff_result.append(('delete', para1))
                else:
                    rich_diff1, rich_diff2 = get_word_level_diff(para1, para2)
                    diff_result.append(('replace', rich_diff1, rich_diff2))
    return diff_result

def create_diff_pdf(diff_result: list, pdf1_name: str, pdf2_name: str, output_filename: str):
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter
    margin, gutter = 0.75 * inch, 0.25 * inch
    col_width = (width - 2 * margin - gutter) / 2
    base_style = ParagraphStyle('base', fontName='Courier', fontSize=8, leading=10, alignment=TA_LEFT)
    delete_style = "<font backColor='#fdb8c0'><strike>"
    insert_style = "<font backColor='#a6f2b9'>"
    
    def setup_page(c, y):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"Original ({pdf1_name})")
        c.drawString(margin + col_width + gutter, y, f"Changed ({pdf2_name})")
        y -= 0.3 * inch
        c.line(margin, y + 0.1 * inch, width - margin, y + 0.1 * inch)
        return y
    
    y = height - margin
    y = setup_page(c, y)

    def draw_line_pair(left_text, right_text, line_type_for_bg):
        nonlocal y
        p_left = Paragraph(left_text, style=base_style)
        p_right = Paragraph(right_text, style=base_style)
        left_h, right_h = p_left.wrap(col_width, height)[1], p_right.wrap(col_width, height)[1]
        line_height = max(left_h, right_h, 10) + 4
        
        if y - line_height < margin:
            c.showPage()
            y = height - margin
            y = setup_page(c, y)

        if line_type_for_bg == 'delete': c.setFillColorRGB(1, 0.93, 0.94); c.rect(margin, y - line_height, col_width, line_height, stroke=0, fill=1)
        elif line_type_for_bg == 'insert': c.setFillColorRGB(0.9, 1, 0.93); c.rect(margin + col_width + gutter, y - line_height, col_width, line_height, stroke=0, fill=1)
        
        p_left.drawOn(c, margin + 2, y - line_height + 2)
        p_right.drawOn(c, margin + col_width + gutter + 2, y - line_height + 2)
        y -= line_height + 2

    for line_type, *content in diff_result:
        left_line, right_line = "", ""
        def escape(text): return text.replace('&', '&').replace('<', '<').replace('>', '>')
        
        if line_type == 'equal': left_line = right_line = escape(content[0] or '')
        elif line_type == 'delete': left_line = f"{delete_style}{escape(content[0] or '')}</strike></font>"
        elif line_type == 'insert': right_line = f"{insert_style}{escape(content[0] or '')}</font>"
        elif line_type == 'replace':
            for tag, text in content[0]: left_line += f"{delete_style}{escape(text)}</strike></font>" if tag == 'delete' else escape(text)
            for tag, text in content[1]: right_line += f"{insert_style}{escape(text)}</font>" if tag == 'insert' else escape(text)
        draw_line_pair(left_line, right_line, line_type)
    c.save()

# ==============================================================================
# 3. HTML, CSS, JAVASCRIPT FRONT-END (COMPLETELY REDESIGNED)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Comparison Tool</title>
    <style>
        :root {
            --primary-color: #0d6efd;
            --secondary-color: #6c757d;
            --success-color: #198754;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --border-radius: 0.375rem;
        }
        body { font-family: var(--font-family); line-height: 1.6; color: var(--dark-color); background-color: #e9ecef; }
        .main-container { max-width: 1400px; margin: 2rem auto; padding: 2rem; background-color: #fff; border-radius: var(--border-radius); box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.1); }
        header { text-align: center; margin-bottom: 2rem; }
        header h1 { font-size: 2.5rem; color: var(--dark-color); }
        header p { font-size: 1.1rem; color: var(--secondary-color); }
        .comparison-area { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem; }
        .upload-column { display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed #ced4da; border-radius: var(--border-radius); padding: 2rem; transition: border-color 0.2s; }
        .upload-column.has-file { border-color: var(--primary-color); background-color: #f0f8ff; }
        .upload-column h3 { margin-top: 0; color: var(--secondary-color); }
        .file-label { display: inline-block; background-color: var(--primary-color); color: white; padding: 0.75rem 1.5rem; border-radius: var(--border-radius); cursor: pointer; transition: background-color 0.2s; }
        .file-label:hover { background-color: #0b5ed7; }
        .file-name { margin-top: 1rem; color: var(--dark-color); font-weight: 500; }
        input[type="file"] { display: none; }
        .actions { text-align: center; }
        .compare-btn { background-color: var(--success-color); color: white; font-size: 1.2rem; padding: 0.8rem 2.5rem; border: none; border-radius: var(--border-radius); cursor: pointer; transition: background-color 0.2s; }
        .compare-btn:disabled { background-color: var(--secondary-color); cursor: not-allowed; }
        .compare-btn:not(:disabled):hover { background-color: #157347; }
        #status { text-align: center; margin-top: 2rem; }
        .loader { display: inline-block; border: 5px solid var(--light-color); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .hidden { display: none; }
        #results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
        #download-btn { background-color: var(--primary-color); color: white; padding: 0.6rem 1.2rem; border-radius: var(--border-radius); border: none; cursor: pointer; transition: background-color 0.2s; }
        #download-btn:hover { background-color: #0b5ed7; }
        .diff-card { margin-bottom: 1rem; border: 1px solid #dee2e6; border-left-width: 5px; border-radius: var(--border-radius); overflow: hidden; }
        .diff-card.equal { border-left-color: #ced4da; }
        .diff-card.insert { border-left-color: var(--success-color); }
        .diff-card.delete { border-left-color: var(--danger-color); }
        .diff-card.replace { border-left-color: var(--warning-color); }
        .diff-content { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem; font-family: 'SF Mono', 'Courier New', monospace; font-size: 0.9rem; line-height: 1.7; }
        .diff-col pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; }
        .del-word { text-decoration: line-through; background-color: #f8d7da; border-radius: 3px; }
        .ins-word { background-color: #d1e7dd; border-radius: 3px; }
    </style>
</head>
<body>
    <main class="main-container">
        <header>
            <h1>Intelligent PDF Document Comparator</h1>
            <p>Upload two PDF files to see a detailed, side-by-side comparison of their textual content.</p>
        </header>

        <form id="compare-form">
            <div class="comparison-area">
                <div class="upload-column" id="upload-col-1">
                    <h3>Original Document</h3>
                    <label for="pdf1" class="file-label">Choose File</label>
                    <input type="file" id="pdf1" name="pdf1" accept=".pdf" required>
                    <p class="file-name" id="file-name-1"></p>
                </div>
                <div class="upload-column" id="upload-col-2">
                    <h3>Revised Document</h3>
                    <label for="pdf2" class="file-label">Choose File</label>
                    <input type="file" id="pdf2" name="pdf2" accept=".pdf" required>
                    <p class="file-name" id="file-name-2"></p>
                </div>
            </div>
            <div class="actions">
                <button type="submit" id="compare-btn" class="compare-btn" disabled>Compare Documents</button>
            </div>
        </form>

        <div id="status" class="hidden">
            <div class="loader"></div>
            <p>Analyzing documents, please wait...</p>
        </div>
        
        <div id="results-area" class="hidden">
            <div id="results-header">
                <h2>Comparison Results</h2>
                <button id="download-btn">⬇️ Download Report</button>
            </div>
            <div id="diff-feed"></div>
        </div>
    </main>

    <script>
        const form = document.getElementById('compare-form');
        const pdf1Input = document.getElementById('pdf1');
        const pdf2Input = document.getElementById('pdf2');
        const fileName1 = document.getElementById('file-name-1');
        const fileName2 = document.getElementById('file-name-2');
        const uploadCol1 = document.getElementById('upload-col-1');
        const uploadCol2 = document.getElementById('upload-col-2');
        const compareBtn = document.getElementById('compare-btn');
        const statusDiv = document.getElementById('status');
        const resultsArea = document.getElementById('results-area');
        const diffFeed = document.getElementById('diff-feed');
        const downloadBtn = document.getElementById('download-btn');
        let diffData = null;
        let fileNames = {};

        function checkFiles() {
            const hasFile1 = pdf1Input.files.length > 0;
            const hasFile2 = pdf2Input.files.length > 0;
            compareBtn.disabled = !(hasFile1 && hasFile2);
        }

        pdf1Input.addEventListener('change', () => {
            fileName1.textContent = pdf1Input.files[0] ? pdf1Input.files[0].name : '';
            uploadCol1.classList.toggle('has-file', pdf1Input.files.length > 0);
            checkFiles();
        });

        pdf2Input.addEventListener('change', () => {
            fileName2.textContent = pdf2Input.files[0] ? pdf2Input.files[0].name : '';
            uploadCol2.classList.toggle('has-file', pdf2Input.files.length > 0);
            checkFiles();
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            statusDiv.classList.remove('hidden');
            resultsArea.classList.add('hidden');
            diffFeed.innerHTML = '';
            compareBtn.disabled = true;

            const formData = new FormData(form);
            fileNames = { pdf1: formData.get('pdf1').name, pdf2: formData.get('pdf2').name };
            
            try {
                const response = await fetch('/compare', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (result.status === 'success') {
                    diffData = result.diff;
                    displayDiff(diffData);
                    resultsArea.classList.remove('hidden');
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (error) {
                alert('An unexpected error occurred.');
                console.error(error);
            } finally {
                statusDiv.classList.add('hidden');
                checkFiles(); // Re-enable button if files are still selected
            }
        });

        downloadBtn.addEventListener('click', async () => {
            if (!diffData) return;
            downloadBtn.textContent = 'Generating...';
            downloadBtn.disabled = true;
            try {
                const response = await fetch('/generate_pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ diff: diffData, names: fileNames }),
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'comparison_report.pdf';
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    window.URL.revokeObjectURL(url);
                } else {
                    const errorResult = await response.json();
                    alert('Failed to generate PDF: ' + errorResult.message);
                }
            } catch (error) {
                alert('An unexpected error during PDF generation.');
                console.error(error);
            } finally {
                downloadBtn.textContent = '⬇️ Download Report';
                downloadBtn.disabled = false;
            }
        });

        function displayDiff(diff) {
            function escapeHtml(text) {
                return text.replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">");
            }

            diffFeed.innerHTML = ''; // Clear previous results
            diff.forEach(item => {
                const [type, ...content] = item;
                
                const card = document.createElement('div');
                card.className = `diff-card ${type}`;
                
                const cardContent = document.createElement('div');
                cardContent.className = 'diff-content';
                
                const col1 = document.createElement('div');
                col1.className = 'diff-col';
                const pre1 = document.createElement('pre');
                col1.appendChild(pre1);

                const col2 = document.createElement('div');
                col2.className = 'diff-col';
                const pre2 = document.createElement('pre');
                col2.appendChild(pre2);

                if (type === 'equal') {
                    pre1.innerHTML = pre2.innerHTML = escapeHtml(content[0] || '');
                } else if (type === 'delete') {
                    pre1.innerHTML = escapeHtml(content[0] || '');
                } else if (type === 'insert') {
                    pre2.innerHTML = escapeHtml(content[0] || '');
                } else if (type === 'replace') {
                    let leftHtml = '';
                    content[0].forEach(([tag, text]) => {
                        leftHtml += tag === 'delete' ? `<span class="del-word">${escapeHtml(text)}</span>` : escapeHtml(text);
                    });
                    pre1.innerHTML = leftHtml;

                    let rightHtml = '';
                    content[1].forEach(([tag, text]) => {
                        rightHtml += tag === 'insert' ? `<span class="ins-word">${escapeHtml(text)}</span>` : escapeHtml(text);
                    });
                    pre2.innerHTML = rightHtml;
                }
                cardContent.appendChild(col1);
                cardContent.appendChild(col2);
                card.appendChild(cardContent);
                diffFeed.appendChild(card);
            });
        }
    </script>
</body>
</html>
"""

# ==============================================================================
# 4. FLASK WEB ROUTES (Updated for new UI)
# ==============================================================================
@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/compare', methods=['POST'])
def compare():
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        return jsonify({'status': 'error', 'message': 'Both PDF files are required.'}), 400
    file1, file2 = request.files['pdf1'], request.files['pdf2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'status': 'error', 'message': 'Please select two files.'}), 400

    filename1 = secure_filename(f"{uuid.uuid4()}_{file1.filename}")
    filename2 = secure_filename(f"{uuid.uuid4()}_{file2.filename}")
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    file1.save(path1)
    file2.save(path2)

    try:
        diff_result = compare_pdfs(path1, path2)
        return jsonify({'status': 'success', 'diff': diff_result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500
    finally:
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    if not data or 'diff' not in data or 'names' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid data provided.'}), 400

    diff_result = data['diff']
    pdf1_name = data['names'].get('pdf1', 'Original.pdf')
    pdf2_name = data['names'].get('pdf2', 'Changed.pdf')
    
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{uuid.uuid4()}.pdf")
    
    buffer = None
    try:
        create_diff_pdf(diff_result, pdf1_name, pdf2_name, output_filename)
        with open(output_filename, 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to generate PDF: {str(e)}'}), 500
    finally:
        if os.path.exists(output_filename):
            os.remove(output_filename)

    if not buffer:
        return jsonify({'status': 'error', 'message': 'PDF buffer could not be created.'}), 500

    return send_file(
        buffer,
        as_attachment=True,
        download_name='comparison_report.pdf',
        mimetype='application/pdf'
    )

# ==============================================================================
# 5. RUN THE APPLICATION (Using Waitress for stability)
# ==============================================================================
if __name__ == '__main__':
    print("🚀 PDF Comparison Tool starting...")
    print("✅ Access it at http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000)