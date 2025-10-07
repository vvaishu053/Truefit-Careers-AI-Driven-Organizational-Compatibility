from flask import Flask, request, jsonify, session, redirect, url_for
import psycopg2
from flask_cors import CORS
from flask_bcrypt import Bcrypt
import PyPDF2
import docx
import re
from transformers import pipeline,GPT2Tokenizer, GPT2LMHeadModel
import pickle
from psycopg2.extras import DictCursor
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
import os
import subprocess
import signal
import time

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Session config
app.config["SESSION_COOKIE_NAME"] = "my_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # The crucial fix for local development
app.config["SESSION_COOKIE_SECURE"] = False

# Allow React frontend
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

# Initialize Bcrypt
bcrypt = Bcrypt(app)

# Initialize LLM summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Database Connection ---
def get_db_connection():
    return psycopg2.connect(
        dbname="major",
        user="postgres",
        password="shivanirao1710",
        host="localhost",
        port="5432"
    )

# --- Resume Parsing & Job Matching Functions ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def generate_insights(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def extract_section(text, keywords):
    for keyword in keywords:
        if keyword.lower() in text.lower():
            start_index = text.lower().find(keyword.lower())
            end_index = find_next_section(text, start_index)
            return text[start_index:end_index].strip()
    return ""

def find_next_section(text, start_index):
    section_headers = ["experience", "education", "projects", "skills", "certifications", "achievements"]
    next_index = len(text)
    for header in section_headers:
        header_index = text.lower().find(header, start_index + 1)
        if header_index != -1 and header_index < next_index:
            next_index = header_index
    return next_index

skill_variations = {
    "nodejs": "node.js", "react.js": "reactjs", "next.js": "nextjs", "javascript": "js", "typescript": "ts", "c++": "cpp",
    "c#": "csharp", "html5": "html", "css3": "css", "postgresql": "postgres", "mongodb": "mongo", "aws": "amazon web services",
    "gcp": "google cloud platform", "azure": "microsoft azure", "kubernetes": "k8s", "python3": "python", "python2": "python",
}
skills_list = [
    "python", "java", "c++", "c#", "ruby", "sql", "html", "css", "javascript", "java swings", "typescript", "php",
    "flask", "django", "nodejs", "expressjs", "react", "reactjs", "nextjs", "angular", "vue.js", "jquery", "bootstrap", "spring", "asp.net",
    "swift", "kotlin", "objective-c", "go", "scala", "perl", "shell scripting", "bash", "powershell", "rust",
    "haskell", "sql server", "postgresql", "mongodb", "mysql", "oracle", "redis", "firebase", "sqlite",
    "hadoop", "spark", "kafka", "elasticsearch", "cassandra", "bigquery", "aws", "azure", "google cloud", "gcp",
    "terraform", "docker", "kubernetes", "ansible", "puppet", "chef", "jenkins", "git", "gitlab", "github", "bitbucket",
    "vagrant", "virtualbox", "jenkins", "ci/cd", "maven", "gradle", "npm", "yarn", "bower", "nginx", "apache",
    "webpack", "graphql", "rest api", "soap", "json", "xml", "protobuf", "swagger", "microservices", "devops",
    "cloudformation", "azure devops", "cloud storage", "cloud architecture", "containerization", "serverless",
    "elastic beanstalk", "lambda", "cloudwatch", "docker swarm", "nginx", "apache kafka", "fluentd", "prometheus",
    "grafana", "openstack", "vagrant", "selenium", "pytest", "junit", "mocha", "chai", "karma", "jasmine",
    "testng", "jupyter", "pandas", "matplotlib", "seaborn", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "nltk", "spaCy", "openCV", "d3.js", "tableau", "power bi", "matlab", "sas", "r", "spss", "stata",
    "excel", "rds", "spark sql", "sas", "apache flink", "databricks", "etl", "business intelligence", "data mining",
    "data engineering", "data scientist", "etl pipelines", "data lakes", "deep learning", "machine learning",
    "computer vision", "natural language processing", "predictive analytics", "data visualization", "statistics",
    "blockchain", "cryptocurrency", "bitcoin", "ethereum", "iot", "iot protocols", "home automation", "arduino",
    "raspberry pi", "mqtt", "zigbee", "smart contracts", "solidity", "ethereum", "docker", "pytorch", "keras",
    "tensorflow", "scipy", "data wrangling", "jupyter notebooks", "tableau", "google analytics", "splunk",
    "elasticsearch", "salesforce", "service now", "aws lambda", "apache spark", "cloud computing", "cloud migration",
    "blockchain", "nfc", "qr codes", "tcp/ip", "vpn", "pentesting", "ethical hacking", "penetration testing",
    "security", "open security", "ssl", "tls", "http", "oauth", "network security", "firewall", "siem", "firewall",
    "authentication", "authorization", "ssh", "sftp", "ssl", "keycloak", "data encryption", "cybersecurity", "risk management",
    "communication", "teamwork", "leadership", "problem-solving", "creativity", "critical thinking", "time management",
    "adaptability", "collaboration", "conflict resolution", "empathy", "active listening", "negotiation", "presentation",
    "public speaking", "decision making", "attention to detail", "interpersonal skills", "self-motivation", "work ethic",
    "confidentiality", "organizational skills", "stress management", "self-learning", "positive attitude", "customer service",
    "accountability", "delegation", "mentorship", "project management", "resource management", "goal setting",
    "strategic thinking", "analytical thinking", "emotional intelligence", "networking", "team building", "influencing",
    "persuasion", "flexibility", "confidentiality", "coaching", "facilitation", "mindfulness", "decision-making",
    "adaptability", "learning agility", "self-awareness", "conflict management", "collaboration skills", "relationship-building"
]
def normalize_skill(skill):
    return skill_variations.get(skill.lower(), skill.lower())
def normalize_text(text):
    for variation, canonical in skill_variations.items():
        text = re.sub(rf"\b{re.escape(variation)}\b", canonical, text, flags=re.IGNORECASE)
    return text

def parse_resume(text):
    data = {
        "name": "", "email": "", "phone": "", "skills": "", "experience": "",
        "education": "", "projects": "", "insights": ""
    }
    text = normalize_text(text)
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text, re.IGNORECASE)
    if emails:
        data["email"] = emails[0]
    phones = re.findall(r"\+?\d[\d -]{8,12}\d", text)
    if phones:
        data["phone"] = phones[0]
    data["name"] = text.split("\n")[0].strip()
    normalized_skills_list = [normalize_skill(skill) for skill in skills_list]
    found_skills = [skill for skill in normalized_skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    data["skills"] = ", ".join(found_skills)
    experience_keywords = ["experience", "work history", "employment", "professional experience"]
    data["experience"] = extract_section(text, experience_keywords)
    education_keywords = ["education", "academic background", "degrees", "qualifications"]
    data["education"] = extract_section(text, education_keywords)
    projects_keywords = ["projects", "personal projects", "project experience"]
    data["projects"] = extract_section(text, projects_keywords)
    data["insights"] = generate_insights(text)
    return data

def get_resume_data(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT resume_id, name, email, phone, skills, experience, education, projects, insights, behavioral_tag
            FROM resumes
            WHERE user_id = %s
            ORDER BY resume_id DESC
            LIMIT 1
        """
        cur.execute(query, (user_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            return {
                'resume_id': result[0], 'name': result[1], 'email': result[2], 'phone': result[3],
                'skills': result[4], 'experience': result[5], 'education': result[6],
                'projects': result[7], 'insights': result[8], 'behavioral_tag': result[9]
            }
        return None
    except Exception as e:
        print("Error fetching data:", e)
        return None

# --- User Authentication Routes ---
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if user:
            return jsonify({"error": "Username already exists"}), 400
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        return jsonify({"message": "Registration successful"}), 201
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id, password FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        if row and bcrypt.check_password_hash(row[1], password):
            session["user_id"] = row[0]
            return jsonify({"message": "Login successful", "user_id": row[0]}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401
    finally:
        conn.close()

@app.route("/home", methods=["GET"])
def home():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT username FROM users WHERE user_id = %s", (session["user_id"],))
        username = cur.fetchone()[0]
        return jsonify({"message": f"Hello {username}! Welcome to Truefit Careers"}), 200
    except (psycopg2.Error, TypeError) as e:
        return jsonify({"error": "User not found or database error"}), 500
    finally:
        conn.close()

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200

# --- Resume & Job API Routes ---
@app.route("/api/upload_resume", methods=["POST"])
def upload_resume():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith((".docx", ".doc")):
        text = extract_text_from_docx(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 415
    parsed_data = parse_resume(text)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM resumes WHERE user_id = %s", (session['user_id'],))
        cur.execute(
            """
            INSERT INTO resumes (user_id, name, email, phone, skills, experience, education, projects, file_name, insights)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (session['user_id'], parsed_data["name"], parsed_data["email"], parsed_data["phone"],
             parsed_data["skills"], parsed_data["experience"], parsed_data["education"],
             parsed_data["projects"], file.filename, parsed_data["insights"])
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "Resume uploaded and parsed successfully"}), 200
    except Exception as e:
        print("Database error:", e)
        return jsonify({"error": "Error saving resume to database"}), 500

@app.route("/api/get_resume", methods=["GET"])
def get_resume():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    resume_data = get_resume_data(session['user_id'])
    if resume_data:
        return jsonify(resume_data), 200
    return jsonify({"error": "No resume data found"}), 404

@app.route("/api/get_jobs", methods=["GET"])
def get_jobs():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT skills, experience, projects FROM resumes WHERE user_id = %s ORDER BY resume_id DESC LIMIT 1", (session['user_id'],))
        resume_data = cur.fetchone()
        if not resume_data:
            return jsonify({"jobs": []}), 200
        combined_skills = set()
        for skills_field in resume_data:
            if skills_field:
                combined_skills.update([normalize_skill(skill.strip()) for skill in skills_field.split(",")])
        cur.execute("SELECT job_role, company_name, company_type, skills FROM job_roles")
        jobs = cur.fetchall()
        cur.close()
        conn.close()
        if not jobs:
            return jsonify({"jobs": []}), 200
        recommended_jobs = []
        for job in jobs:
            job_role, company_name, company_type, job_skills = job
            job_skills_set = set([normalize_skill(s.strip()) for s in job_skills.lower().split(",")])
            matched_skills = combined_skills.intersection(job_skills_set)
            missing_skills = job_skills_set.difference(combined_skills)
            if matched_skills:
                relevance_score = (len(matched_skills) / len(job_skills_set)) * 100 if job_skills_set else 0
                recommended_jobs.append({
                    "job_role": job_role, "company_name": company_name, "company_type": company_type,
                    "skills": job_skills, "matched_skills": ", ".join(matched_skills),
                    "missing_skills": ", ".join(missing_skills), "relevance_score": round(relevance_score, 2)
                })
        ranked_jobs = sorted(recommended_jobs, key=lambda x: x["relevance_score"], reverse=True)[:5]
        return jsonify({"jobs": ranked_jobs}), 200
    except Exception as e:
        print("Error fetching job recommendations:", e)
        return jsonify({"jobs": []}), 500

# --- NEW PROFILE ROUTE FOR REACT ---
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    resume_data = get_resume_data(user_id)
    
    if resume_data:
        return jsonify({"data": resume_data}), 200
    else:
        return jsonify({"error": "To update your profile, please upload your resume"}), 404


# --- BEHAVIOURAL ANALYZER ---
# Load Random Forest model
model = pickle.load(open("rf_model.pkl", "rb"))

# Fetch questions from DB
@app.route("/api/behaviour", methods=["GET"])
def get_questions():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, question_text FROM questions ORDER BY id")
        questions = [{"id": row[0], "question_text": row[1]} for row in cur.fetchall()]
        cur.close()
        conn.close()
        return jsonify({"questions": questions}), 200
    except Exception as e:
        print("Error fetching questions:", e)
        return jsonify({"error": "Failed to fetch questions"}), 500


# Predict behavioural scores & save responses
@app.route("/api/submit_behaviour", methods=["POST"])
def submit_behaviour():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json  # Expecting {"answers": {"1": 4, "2": 3, ...}}
    answers_dict = data.get("answers")
    if not answers_dict or len(answers_dict) == 0:
        return jsonify({"error": "No answers provided"}), 400

    user_id = session['user_id']
    answers = [int(answers_dict[str(i)]) for i in range(1, len(answers_dict)+1)]

    # Step 1: Predict Big Five scores
    prediction = model.predict([answers])[0]
    predicted_scores = {
        "openness": float(prediction[0]),
        "conscientiousness": float(prediction[1]),
        "extraversion": float(prediction[2]),
        "agreeableness": float(prediction[3]),
        "neuroticism": float(prediction[4])
    }

    # Step 2: Determine behavioural tag
    max_trait = max(predicted_scores, key=predicted_scores.get)
    tag_mapping = {
        "openness": "Creative",
        "conscientiousness": "Organized",
        "extraversion": "Extroverted",
        "agreeableness": "Compassionate",
        "neuroticism": "Neurotic"
    }
    behavioral_tag = tag_mapping[max_trait]

    # Step 3: Save responses & results in DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete previous responses/results
        cur.execute("DELETE FROM responses WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM results WHERE user_id = %s", (user_id,))

        # Save responses (assuming q1...q50 columns)
        placeholders = ','.join([f"q{i+1}" for i in range(len(answers))])
        values_placeholders = ','.join(['%s']*len(answers))
        cur.execute(
            f"INSERT INTO responses (user_id, {placeholders}) VALUES (%s, {values_placeholders})",
            [user_id] + answers
        )

        # Save results
        cur.execute("""
            INSERT INTO results (user_id, openness, conscientiousness, extraversion, agreeableness, neuroticism, behavioral_tag)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            predicted_scores["openness"],
            predicted_scores["conscientiousness"],
            predicted_scores["extraversion"],
            predicted_scores["agreeableness"],
            predicted_scores["neuroticism"],
            behavioral_tag
        ))

        # Update resumes table
        cur.execute("UPDATE resumes SET behavioral_tag=%s WHERE user_id=%s", (behavioral_tag, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "scores": predicted_scores,
            "behavioral_tag": behavioral_tag
        }), 200

    except Exception as e:
        print("Error saving behavioural data:", e)
        return jsonify({"error": "Failed to save behavioural data"}), 500


# Get latest behavioural result
@app.route("/api/result_behaviour", methods=["GET"])
def get_result_behaviour():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT openness, conscientiousness, extraversion, agreeableness, neuroticism, behavioral_tag
            FROM results
            WHERE user_id=%s
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            scores = {
                "openness": row[0],
                "conscientiousness": row[1],
                "extraversion": row[2],
                "agreeableness": row[3],
                "neuroticism": row[4]
            }
            behavioral_tag = row[5]
            return jsonify({"scores": scores, "behavioral_tag": behavioral_tag}), 200
        else:
            return jsonify({"error": "No results found"}), 404

    except Exception as e:
        print("Error fetching behavioural result:", e)
        return jsonify({"error": "Failed to fetch results"}), 500

#JOB RECOMMENDATION
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# fetch job data
def get_job_data_from_postgresql():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features, embedding FROM job_data_cleaned")
        job_data = cursor.fetchall()
        cursor.close()
        conn.close()
        return job_data
    except Exception as err:
        print(f"Error: {err}")
        return []

def prepare_faiss_index(job_data):
    job_embeddings = []
    job_titles = []
    for job in job_data:
        embedding = job.get('embedding', '')
        if embedding:
            try:
                embedding = json.loads(embedding)
                job_embeddings.append(embedding)
                job_titles.append(job['job_role'])
            except:
                continue
    job_embeddings = np.array(job_embeddings).astype('float32')
    faiss_index = faiss.IndexFlatL2(job_embeddings.shape[1])
    faiss_index.add(job_embeddings)
    return faiss_index, job_titles

def find_job_roles_by_skills(skills, top_n=5):
    skills_query = [s.strip() for s in skills.lower().split(",")]
    query = " ".join(skills_query)
    query_embedding = sentence_model.encode([query])[0]
    job_data = get_job_data_from_postgresql()
    faiss_index, job_titles = prepare_faiss_index(job_data)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)
    recommended_jobs = [job_data[i] for i in indices[0] if i < len(job_data)]
    return recommended_jobs

def find_job_roles_by_job_role(job_role, top_n=5):
    job_role = job_role.lower().strip()
    query_embedding = sentence_model.encode([job_role])[0]
    job_data = get_job_data_from_postgresql()
    faiss_index, job_titles = prepare_faiss_index(job_data)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)
    recommended_jobs = [job_data[i] for i in indices[0] if i < len(job_data)]
    return recommended_jobs

def find_job_roles_by_company(company_name):
    company_name = company_name.lower().strip()
    job_data = get_job_data_from_postgresql()
    filtered_jobs = [job for job in job_data if job['company_name'].lower() == company_name]
    return filtered_jobs

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    search_type = data.get("search_type", "skills")
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter skills, job role, or company name."}), 400

    if search_type == "skills":
        recommendations = find_job_roles_by_skills(query)
    elif search_type == "job_role":
        recommendations = find_job_roles_by_job_role(query)
    elif search_type == "company_name":
        recommendations = find_job_roles_by_company(query)
    else:
        recommendations = []

    # Convert jobs to simple dicts for JSON
    rec_list = []
    for job in recommendations:
        rec_list.append({
            "job_role": job.get("job_role"),
            "company_name": job.get("company_name"),
            "company_type": job.get("company_type"),
            "knowledge_cleaned": job.get("knowledge_cleaned"),
            "skills_cleaned": job.get("skills_cleaned")
        })

    return jsonify({"recommendations": rec_list})

from flask import jsonify, request
import json

@app.route("/api/job_details", methods=["GET"])
def job_details_api():
    job_role = request.args.get("job_role", "").strip().lower()
    company_name = request.args.get("company_name", "").strip().lower()

    try:
        job_data = get_job_data_from_postgresql()  # returns list of dicts

        # Convert DictRow to regular dict if needed
        job_data = [dict(job) if not isinstance(job, dict) else job for job in job_data]

        # Find matching job
        job_details = next(
            (
                job for job in job_data
                if job.get('job_role', '').strip().lower() == job_role
                and job.get('company_name', '').strip().lower() == company_name
            ),
            None
        )

        if job_details:
            return jsonify({"job": job_details}), 200
        else:
            return jsonify({"error": "Job not found"}), 404

    except Exception as e:
        print("Error fetching job details:", e)
        return jsonify({"error": "Failed to fetch job details"}), 500

def get_user_name():
    """Fetch logged-in user's name from resumes table"""
    user_id = session.get("user_id")  # assume user_id stored in session

    if not user_id:
        return "there"  # fallback if no login

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT name FROM resumes WHERE user_id = %s LIMIT 1;", (user_id,))
            result = cursor.fetchone()
            return result[0] if result else "there"
    except Exception as e:
        print("Error fetching name:", e)
        return "there"
    finally:
        conn.close()

# ==============================
# 4. GPT MODELS
# ==============================
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Hugging Face DialoGPT pipeline
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# ==============================
# 5. GRAMMAR FIX + SHORT RESPONSES
# ==============================
def correct_grammar_and_generate_response(text):
    """Fix grammar using GPT-2 and generate short, meaningful responses."""
    inputs = gpt_tokenizer.encode(text, return_tensors="pt")

    outputs = gpt_model.generate(
        inputs,
        max_new_tokens=30,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.85,
        temperature=0.6
    )

    generated_response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # keep only first sentence
    cleaned_response = re.split(r"[.!?]", generated_response)[0].strip() + "."

    return cleaned_response

@app.route("/chatbot", methods=["POST"])
def chatbot_route():
    user_name = get_user_name()
    user_input = request.json.get("user_input", "").strip()

    if not user_input:
        return jsonify({
            "response": f"Hello {user_name}, please ask me something!",
            "user_name": user_name
        })

    job_info = get_job_data_from_postgresql()
    response = ""

    # ✅ Skills query
    match = re.search(r"(?:skills needed for|skills for|technology required for)\s+(.+)", user_input, re.IGNORECASE)
    if match:
        job_role = match.group(1).strip()
        matching_jobs = [job for job in job_info if job_role.lower() in job["job_role"].lower()]

        if matching_jobs:
            response = f"Here are the key skills needed for **{job_role}**, {user_name}:\n\n"
            response += "\n".join(
                f"- **{job['job_role']}** at {job['company_name']} (Skills: {job['skills_cleaned']})"
                for job in matching_jobs[:3]
            )
        else:
            response = f"Sorry, {user_name}, I couldn't find skills for **'{job_role}'**."

    # ✅ Job roles query
    elif re.search(r"(?:job roles for|roles for|positions for|careers in)\s+(.+)", user_input, re.IGNORECASE):
        job_role = re.search(r"(?:job roles for|roles for|positions for|careers in)\s+(.+)", user_input, re.IGNORECASE).group(1).strip()
        matching_jobs = [job for job in job_info if job_role.lower() in job["job_role"].lower()]

        if matching_jobs:
            response = f"Here are some job roles related to **{job_role}**, {user_name}:\n\n"
            response += "\n".join(
                f"- **{job['job_role']}** at {job['company_name']} (Skills: {job['skills_cleaned']})"
                for job in matching_jobs[:3]
            )
        else:
            response = f"Sorry, {user_name}, I couldn't find roles for **'{job_role}'**."

    # ✅ Jobs at company query
    elif re.search(r"jobs at\s+(.+)", user_input, re.IGNORECASE):
        company_name = re.search(r"jobs at\s+(.+)", user_input, re.IGNORECASE).group(1).strip()
        matching_jobs = [job for job in job_info if company_name.lower() in job["company_name"].lower()]

        if matching_jobs:
            response = f"Here are some job roles available at **{company_name}**, {user_name}:\n\n"
            response += "\n".join(
                f"- **{job['job_role']}** (Skills: {job['skills_cleaned']})"
                for job in matching_jobs[:3]
            )
        else:
            response = f"Sorry, {user_name}, I couldn't find any jobs at **{company_name}**."

    # ✅ General queries → GPT
    else:
        corrected_input = correct_grammar_and_generate_response(user_input)
        bot_response = chatbot(corrected_input, max_new_tokens=50, pad_token_id=gpt_tokenizer.eos_token_id)[0]["generated_text"]
        response = re.split(r"[\n]", bot_response)[0].strip()

    return jsonify({"response": response, "user_name": user_name})



@app.route('/start-interview', methods=['POST'])
def start_interview():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    # Fetch actual name from DB (your helper)
    username = get_user_name()
    if not username:
        return jsonify({"error": "No username found"}), 400

    env = os.environ.copy()
    env['LOGGED_IN_USER'] = username  # Pass logged-in user's name

    try:
        # Start the live_interview.py process
        process = subprocess.Popen(["python", "interview/live_interview.py"], env=env)
        session['interview_pid'] = process.pid
        print(f"▶️ Started live_interview.py for {username} (PID={process.pid})")
    except Exception as e:
        return jsonify({"error": f"Failed to start interview: {str(e)}"}), 500

    return jsonify({
        "message": f"Interview started for {username}",
        "username": username
    })


# -------------------------
# Get Questions
# -------------------------
@app.route('/get-questions', methods=['GET'])
def getquestions():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM interview_questions")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify({"questions": [{"id": r[0], "question": r[1]} for r in rows]})


# -------------------------
# Submit Interview
# -------------------------
@app.route('/submit-interview', methods=['POST'])
def submit_interview():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    try:
        data = request.json
        answers = data.get("answers", {})

        username = get_user_name()
        if not username:
            return jsonify({"error": "No username found"}), 400

        conn = get_db_connection()
        cur = conn.cursor()

        # Delete old answers for this user
        cur.execute("DELETE FROM interview_answers WHERE username = %s", (username,))

        # Insert new answers
        for question_id, answer in answers.items():
            cur.execute("""
                INSERT INTO interview_answers (question, answer, username)
                SELECT question, %s, %s FROM interview_questions WHERE id = %s
            """, (answer, username, question_id))

        conn.commit()
        cur.close()
        conn.close()

        # Kill the live interview process if running
        pid = session.get("interview_pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"🛑 Stopped live_interview.py (PID={pid})")
                session.pop("interview_pid", None)
            except ProcessLookupError:
                print(f"⚠️ No process found with PID={pid}")
                session.pop("interview_pid", None)

        return jsonify({"message": "Interview submitted successfully", "answers": answers})

    except Exception as e:
        print(f"❌ Error in submit_interview: {e}")
        return jsonify({"error": f"Submit failed: {str(e)}"}), 500

# -------------------------
# Results Data
# -------------------------
@app.route('/results-data', methods=['GET'])
def results_data():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    # use the correct helper function
    username = get_user_name()
    if not username:
        return jsonify({"error": "No username found"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Fetch emotions
        cur.execute("""
            SELECT timestamp, emotion 
            FROM emotion_logs 
            WHERE username = %s 
            ORDER BY id
        """, (username,))
        emotions = [{"timestamp": e[0], "emotion": e[1]} for e in cur.fetchall()]

        # Fetch interview answers
        cur.execute("""
            SELECT question, answer 
            FROM interview_answers 
            WHERE username = %s 
            ORDER BY id
        """, (username,))
        answers = [{"question": a[0], "answer": a[1]} for a in cur.fetchall()]

        return jsonify({"emotions": emotions, "answers": answers})

    except Exception as e:
        print(f"❌ Error in /results-data: {e}")
        return jsonify({"error": "Failed to fetch results"}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/generate-report', methods=['POST'])
def generate_report():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = get_user_name()  # ✅ use your helper function
    if not username or username == "there":
        return jsonify({"error": "Username not found"}), 401

    # Set username in environment variable for subprocess
    env = os.environ.copy()
    env['LOGGED_IN_USER'] = username

    try:
        # Run the report generator script
        subprocess.run(["python", "generate_report.py"], env=env, check=True)
        return jsonify({"status": "success", "message": "Report generated successfully"})
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": "Report generation failed",
            "details": str(e)
        }), 500
@app.route('/interview-report', methods=['GET'])
def interview_report():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = get_user_name()  # ✅ use your helper function
    if not username or username == "there":
        return jsonify({"error": "Username not found"}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT question, answer, analysis FROM soft_skill_analysis WHERE username = %s",
        (username,)
    )
    rows = cur.fetchall()
    conn.close()

    return jsonify({
        "username": username,
        "report": [
            {"question": r[0], "answer": r[1], "analysis": r[2]} for r in rows
        ]
    })



@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user_id"]
    user_name = get_user_name()
    if not user_name:
        return jsonify({"error": "User profile not found"}), 404

    try:
        # --- Step 1: Get job data ---
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("""
            SELECT job_role, company_name, company_type, embedding, knowledge_cleaned, skills_cleaned 
            FROM job_data_cleaned
        """)
        job_data = cur.fetchall()
        cur.close()
        conn.close()

        job_embeddings, job_info = [], []
        for job in job_data:
            embedding = job.get("embedding")
            if embedding:
                try:
                    embedding = json.loads(embedding)
                    job_embeddings.append(embedding)
                    job_info.append(job)
                except:
                    continue

        if not job_embeddings:
            return jsonify({"error": "No job embeddings found"}), 500

        # --- Step 2: Prepare FAISS index ---
        job_embeddings = np.array(job_embeddings).astype("float32")
        faiss_index = faiss.IndexFlatL2(job_embeddings.shape[1])
        faiss_index.add(job_embeddings)

        # --- Step 3: Get user profile ---
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)

        # Resume (latest)
        cur.execute(
            "SELECT skills, insights FROM resumes WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1",
            (user_id,)
        )
        resume = cur.fetchone()

        # Soft skills (by username from resumes table)
        cur.execute(
            "SELECT analysis FROM soft_skill_analysis WHERE username = %s",
            (user_name,)
        )
        soft_skills = cur.fetchall()
        cur.close()
        conn.close()

        if not resume:
            return jsonify({"error": "No resume found"}), 404

        user_profile = f"""
        Name: {user_name}
        Resume Skills: {resume['skills'] or ''}
        Resume Insights: {resume['insights'] or ''}
        Behavioral Analysis: {" ".join([row['analysis'] for row in soft_skills]) if soft_skills else ""}
        """

        # --- Step 4: Embed and search ---
        user_embedding = sentence_model.encode([user_profile])[0].astype("float32")
        D, I = faiss_index.search(np.array([user_embedding]), k=6)

        recommendations = []
        for idx, distance in zip(I[0], D[0]):
            job = job_info[idx]

    # Convert L2 distance to similarity percentage
    # You can adjust max_distance according to your embeddings
            max_distance = 10.0  # estimated max L2 distance
            similarity_percentage = max(0, 100 - (float(distance) / max_distance * 100))

            recommendations.append({
                "job_role": job["job_role"],
                "company_name": job["company_name"],
                "company_type": job["company_type"],
                "knowledge": job.get("knowledge_cleaned", ""),
                "skills": job.get("skills_cleaned", ""),
                "similarity_score": round(similarity_percentage, 2)  # rounded to 2 decimals
            })


        return jsonify({"recommendations": recommendations, "user_name": user_name})

    except Exception as e:
        print("Recommendation error:", e)
        return jsonify({"error": "Internal server error"}), 500




if __name__ == "__main__":
    app.run(debug=True)




