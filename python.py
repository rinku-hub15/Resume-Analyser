import os
import re
import fitz  # PyMuPDF
import spacy
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from sentence_transformers import SentenceTransformer, util
from tkinter import simpledialog
import csv
import pytesseract
from PIL import Image
import io

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample job roles
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

# ------------------ Helper Functions ------------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()

        if not text or len(text) < 50:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img)

        full_text += text + "\n"

    return full_text

def extract_sections(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    sections = {"education": [], "projects": [], "experience": []}
    current_section = None

    section_keywords = {
        "education": ["education", "academic"],
        "projects": ["project", "key projects", "personal projects"],
        "experience": ["experience", "work", "employment"]
    }

    for line in lines:
        lower_line = line.lower()
        found_section = False
        for key, keywords in section_keywords.items():
            if any(k in lower_line for k in keywords):
                current_section = key
                found_section = True
                break

        if not found_section and current_section:
            if any(kw in line.lower() for kw in ["skills", "certification", "hobby", "award"]):
                current_section = None
            else:
                sections[current_section].append(line)

    return sections

def extract_entities(text):
    lines = text.split('\n')

    name = "Not found"
    for line in lines[:10]:
        words = line.strip().split()
        if (
            1 < len(words) <= 4
            and not any(char.isdigit() for char in line)
            and '@' not in line
            and all(w[0].isalpha() and w[0].isupper() for w in words[:2])
        ):
            name = line.strip()
            break

    sections = extract_sections(text)
    education = list(set(sections['education']))
    experience = list(set(sections['experience']))
    projects = list(set(sections['projects']))

    return name, education, experience, projects

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "Not found"

def extract_skills(text):
    skills = ['python', 'java', 'c++', 'sql', 'html', 'css', 'javascript',
              'machine learning', 'deep learning', 'nlp', 'flask', 'django',
              'pandas', 'numpy', 'react', 'excel', 'scikit-learn', 'kubernetes',
              'docker', 'aws', 'data visualization', 'system design']
    found = []
    text_lower = text.lower()
    for skill in skills:
        if skill in text_lower:
            found.append(skill)
    return list(set(found))

def match_jobs_semantically(resume_skills):
    resume_str = ", ".join(resume_skills)
    resume_embedding = model.encode(resume_str, convert_to_tensor=True)
    matched = []
    for job in sample_jobs:
        job_embedding = model.encode(job['description'], convert_to_tensor=True)
        score = util.cos_sim(resume_embedding, job_embedding).item() * 100
        job_skills = [s.strip() for s in job['description'].split(',')]
        missing = [skill for skill in job_skills if skill not in resume_skills]
        matched.append({
            "title": job['title'],
            "score": round(score, 2),
            "missing_skills": missing
        })
    return sorted(matched, key=lambda x: x['score'], reverse=True)[:3]

def process_resume(file_path):
    text = extract_text_from_pdf(file_path)
    name, education, experience, projects = extract_entities(text)
    email = extract_email(text)
    skills = extract_skills(text)
    job_matches = match_jobs_semantically(skills)
    return {
        "name": name,
        "email": email,
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "matches": job_matches
    }

# ------------------ GUI ------------------
root = tk.Tk()
root.title("Resume Analyzer: Top 3 Matches & Details")
root.geometry("1200x650")

frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

tree = ttk.Treeview(frame, columns=("Name", "Email", "Top Match 1", "Score 1", "Top Match 2", "Score 2", "Top Match 3", "Score 3"), show="headings")
for col in tree["columns"]:
    tree.heading(col, text=col)
    tree.column(col, width=150)
tree.pack(fill=tk.BOTH, expand=True)

def show_details(event):
    item = tree.focus()
    if not item:
        return
    data = tree.item(item)['values']
    name = data[0]
    resume_data = next((r for r in root.data_list if r['name'] == name), None)
    if not resume_data:
        return

    detail = tk.Toplevel(root)
    detail.title(f"Details for {name}")
    text = tk.Text(detail, wrap=tk.WORD)
    text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    text.insert(tk.END, f"Name: {resume_data['name']}\n")
    text.insert(tk.END, f"Email: {resume_data['email']}\n")
    text.insert(tk.END, f"\nSkills: {', '.join(resume_data['skills'])}\n")
    text.insert(tk.END, f"\nEducation:\n  - " + '\n  - '.join(resume_data['education']) + "\n")
    text.insert(tk.END, f"\nExperience:\n  - " + '\n  - '.join(resume_data['experience']) + "\n")
    text.insert(tk.END, f"\nProjects:\n  - " + '\n  - '.join(resume_data['projects']) + "\n")
    text.insert(tk.END, f"\nTop Job Matches:\n")
    for job in resume_data['matches']:
        text.insert(tk.END, f"- {job['title']} ({job['score']}%)\n  Missing Skills: {', '.join(job['missing_skills'])}\n")

    text.config(state=tk.DISABLED)

def load_multiple():
    file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    if file_paths:
        tree.delete(*tree.get_children())
        root.data_list = []
        for path in file_paths:
            data = process_resume(path)
            root.data_list.append(data)
            top = data['matches']
            tree.insert("", "end", values=(
                data['name'], data['email'],
                top[0]['title'], top[0]['score'],
                top[1]['title'], top[1]['score'],
                top[2]['title'], top[2]['score']
            ))

def export_results():
    filename = filedialog.asksaveasfilename(defaultextension=".csv")
    if filename:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Email", "Top Match 1", "Score 1", "Top Match 2", "Score 2", "Top Match 3", "Score 3"])
            for item in tree.get_children():
                writer.writerow(tree.item(item)['values'])
        messagebox.showinfo("Export", "Export successful!")

btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)

multi_btn = ttk.Button(btn_frame, text="Upload Multiple Resumes", command=load_multiple)
multi_btn.pack(side=tk.LEFT, padx=10)

export_btn = ttk.Button(btn_frame, text="Export Table as CSV", command=export_results)
export_btn.pack(side=tk.LEFT, padx=10)

tree.bind("<Double-1>", show_details)

root.mainloop()
