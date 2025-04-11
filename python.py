import re
import fitz  # PyMuPDF for PDF text extraction
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# ----------------- Resume Parser -----------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        if len(line.split()) >= 2 and line[0].isupper():
            return line.strip()
    return "Not found"

def extract_skills(text):
    skill_keywords = ['python', 'java', 'c++', 'sql', 'html', 'css', 'javascript',
                      'machine learning', 'deep learning', 'nlp', 'flask', 'django',
                      'pandas', 'numpy']
    found_skills = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill)
    return list(set(found_skills))

def extract_education(text):
    edu_keywords = ['bachelor', 'master', 'b.sc', 'm.sc', 'b.tech', 'm.tech', 'phd']
    return [line for line in text.lower().split('\n') if any(edu in line for edu in edu_keywords)]

def extract_experience(text):
    exp_keywords = ['experience', 'intern', 'worked', 'developed']
    return [line for line in text.lower().split('\n') if any(exp in line for exp in exp_keywords)]

def extract_projects(text):
    proj_keywords = ['project', 'developed', 'implemented', 'built']
    return [line for line in text.lower().split('\n') if any(proj in line for proj in proj_keywords)]

# ----------------- Job Matching -----------------
sample_jobs = [
    {"title": "Data Scientist", "description": "python, pandas, machine learning, data analysis"},
    {"title": "Web Developer", "description": "html, css, javascript, react, django"},
    {"title": "Software Engineer", "description": "java, c++, sql, system design"},
    {"title": "AI Engineer", "description": "machine learning, deep learning, python, nlp"},
    {"title": "Frontend Developer", "description": "react, javascript, html, css"},
    {"title": "Backend Developer", "description": "python, flask, django, apis, databases"},
    {"title": "Data Analyst", "description": "sql, excel, data visualization, pandas"},
    {"title": "DevOps Engineer", "description": "docker, kubernetes, ci/cd, aws"},
    {"title": "ML Engineer", "description": "machine learning, python, scikit-learn, numpy"},
    {"title": "Cybersecurity Analyst", "description": "network security, python, risk assessment"}
]

def match_jobs(resume_skills):
    job_texts = [job['description'] for job in sample_jobs]
    all_texts = [" ".join(resume_skills)] + job_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    matched_jobs = []
    for i, score in enumerate(similarities):
        job_skills = sample_jobs[i]['description'].lower().split(', ')
        missing = [skill for skill in job_skills if skill not in resume_skills]
        matched_jobs.append({
            "title": sample_jobs[i]['title'],
            "score": round(score * 100, 2),
            "missing_skills": missing
        })
    top_matches = sorted(matched_jobs, key=lambda x: x['score'], reverse=True)[:3]
    return top_matches

# ----------------- GUI -----------------
def run_analyzer(file_path):
    print("\n[INFO] Reading resume and extracting data...\n")
    text = extract_text_from_pdf(file_path)
    name = extract_name(text)
    email = extract_email(text)
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)
    projects = extract_projects(text)
    top_jobs = match_jobs(skills)

    print("\n--- Parsed Resume Details ---")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Skills: {', '.join(skills) if skills else 'No skills found'}")
    print(f"Education: {education if education else 'Not found'}")
    print(f"Experience: {experience if experience else 'Not found'}")
    print(f"Projects: {projects if projects else 'Not found'}")

    print("\n--- Top 3 Job Matches ---")
    for job in top_jobs:
        print(f"{job['title']} - Match: {job['score']}%")
        print(f"Missing Skills: {', '.join(job['missing_skills']) if job['missing_skills'] else 'None'}\n")

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        run_analyzer(file_path)

# GUI Setup
root = tk.Tk()
root.title("Resume Analyser & Job Matcher")
root.geometry("400x200")
btn = tk.Button(root, text="Select Resume PDF", command=select_file, font=('Arial', 14))
btn.pack(pady=50)
root.mainloop()

