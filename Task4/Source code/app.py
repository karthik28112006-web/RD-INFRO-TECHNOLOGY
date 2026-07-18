import fitz
import spacy
import os
import json
import math
from collections import Counter
from flask import Flask, request, render_template_string, redirect, url_for, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# A secret key is required to safely store temporary session data across redirects
app.secret_key = 'ai_resume_screener_secret_key_123'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

nlp = spacy.load("en_core_web_sm")

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>AI Resume Screening System</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background-color: #fafafa; color: #333; }
.container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
table { border-collapse: collapse; width: 100%; margin-top: 20px; }
th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
th { background: #f4f6f8; font-weight: bold; }
tr:nth-child(even) { background-color: #f9f9f9; }
button { background: #007BFF; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #0056b3; }
.clear-btn { background: #dc3545; margin-left: 10px; }
.clear-btn:hover { background: #bd2130; }
input[type="text"], textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
label { font-weight: bold; display: block; margin-bottom: 5px; }
.form-group { margin-bottom: 20px; }
.error-msg { background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 4px; border: 1px solid #f5c6cb; margin-bottom: 20px; font-weight: bold; }
</style>
</head>
<body>
<div class="container">
<h2>AI-Based Resume Screening System</h2>
<p style="color: #666;">Automate resume screening for HR teams by extracting key information and ranking candidates.</p>
<hr><br>

{% if error %}
<div class="error-msg">
    ⚠ Error: {{ error }}
</div>
{% endif %}

<form method="POST" enctype="multipart/form-data">
<div class="form-group">
<label>Role / Role Title:</label>
<input type="text" name="role_title" value="{{ role_title or '' }}" placeholder="e.g., Python Developer" required>
</div>

<div class="form-group">
<label>Target Skills & Keywords (Paste Job Description text here):</label>
<textarea name="job_description" rows="6" placeholder="You must paste the job requirements here so the AI can score the candidates..." required>{{ job_description or '' }}</textarea>
</div>

<div class="form-group">
<label>Upload Resumes (PDF, multiple allowed):</label>
<input type="file" name="resumes" multiple accept=".pdf" required><br><br>
</div>

<button type="submit">Screen & Rank Candidates</button>
<a href="{{ url_for('clear') }}" style="text-decoration: none;"><button type="button" class="clear-btn">Clear Page</button></a>
</form>

{% if results %}
<h3>Ranked Results for: <span style="color: #007BFF;">{{ role_title }}</span></h3>
<table>
<tr><th>Rank</th><th>Candidate File Name</th><th>Screened Role</th><th>Match Score</th></tr>
{% for r in results %}
<tr><td>{{ loop.index }}</td><td>{{ r.filename }}</td><td>{{ r.role }}</td><td><strong>{{ r.score }}%</strong></td></tr>
{% endfor %}
</table>
{% endif %}
</div>
</body>
</html>
"""

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() or ""
    doc.close()
    return text

def clean_text_list(text):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]

def calculate_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

def score_resume(resume_path, jd_tokens, jd_vector):
    raw_text = extract_text(resume_path)
    resume_tokens_list = clean_text_list(raw_text)
    
    resume_vector = Counter(resume_tokens_list)
    similarity = calculate_cosine_similarity(resume_vector, jd_vector)
    
    resume_tokens_set = set(resume_tokens_list)
    overlap = len(jd_tokens & resume_tokens_set) / len(jd_tokens) if jd_tokens else 0
    
    final_score = (0.6 * similarity + 0.4 * overlap) * 100
    return round(final_score, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        role_title = request.form.get("role_title", "").strip()
        job_description = request.form.get("job_description", "").strip()
        files = request.files.getlist("resumes")

        if not job_description or job_description.isspace():
            session['error'] = "Job Description content cannot be empty! Please paste skills or requirements to screen against."
            session['role_title'] = role_title
            session['job_description'] = job_description
            return redirect(url_for('index'))

        jd_tokens_list = clean_text_list(job_description)
        jd_tokens = set(jd_tokens_list)
        jd_vector = Counter(jd_tokens_list)

        results = []
        for file in files:
            if file and file.filename.lower().endswith(".pdf"):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                try:
                    s = score_resume(file_path, jd_tokens, jd_vector)
                    results.append({"filename": filename, "score": s, "role": role_title})
                except Exception as e:
                    results.append({"filename": filename, "score": 0, "role": role_title})
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)

        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Save output data to session temporary storage memory block 
        session['results'] = results
        session['role_title'] = role_title
        session['job_description'] = job_description
        session['error'] = None
        
        # Perform clean redirect loop to strip out browser form cache history
        return redirect(url_for('index'))

    # GET request handler: Safely read session items and then discard them instantly
    results = session.get('results', None)
    role_title = session.get('role_title', None)
    job_description = session.get('job_description', None)
    error = session.get('error', None)
    
    return render_template_string(TEMPLATE, results=results, job_description=job_description, role_title=role_title, error=error)

@app.route("/clear")
def clear():
    # Dedicated clean routine route to fully wipe browser session variables
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
