import os
import re
import json
import html
from collections import Counter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq

# ------------------- CONFIG -------------------
st.set_page_config(page_title="ResuMate", layout="centered")
st.title("ResuMate")
st.caption("Compare your resume with any job description and get actionable feedback.")

# ------------------- ENV & LLM -------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------- HELPERS -------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_skill_db():
    try:
        with open("skills.json", "r", encoding="utf-8") as f:
            skill_db = json.load(f)
        all_skills = set(skill.lower() for cat in skill_db.values() for skill in cat)
        return all_skills
    except FileNotFoundError:
        st.error("❌ skills.json not found! Please ensure it's in the app directory.")
        return set()

model = load_model()
ALL_SKILLS = load_skill_db()

def extract_text_from_pdf(uploaded_file):
    try:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\-\.\#]', ' ', text)
    return text

def clean_tokens(text):
    return list(set(re.findall(r'\b[a-zA-Z0-9\-\+\.\#]+\b', text.lower())))

def get_resume_details(resume_text):
    resume_text_norm = normalize_text(resume_text)
    tokens = clean_tokens(resume_text_norm)

    matched_skills = {
        skill for skill in ALL_SKILLS
        if re.search(rf"\b{re.escape(skill)}\b", resume_text_norm)
    }

    return {
        "raw_text": resume_text_norm,
        "tokens": tokens,
        "skills": list(matched_skills),
        "word_freq": Counter(tokens)
    }

def compare_resume_with_jd(resume_details, jd_text):
    jd_clean = normalize_text(jd_text)

    jd_skills = {
        skill for skill in ALL_SKILLS
        if re.search(rf"\b{re.escape(skill)}\b", jd_clean)
    }

    if not jd_skills:
        return {"matched_skills": [], "missing_skills": [], "keyword_similarity": 0}

    matched_skills = []
    if resume_details["skills"]:
        resume_embeddings = model.encode(resume_details["skills"], convert_to_tensor=True)
        jd_embeddings = model.encode(list(jd_skills), convert_to_tensor=True)
        cos_scores = util.cos_sim(jd_embeddings, resume_embeddings)

        for i, jd_skill in enumerate(jd_skills):
            if max(cos_scores[i]) > 0.75:
                matched_skills.append(jd_skill)

    missing_skills = list(jd_skills - set(matched_skills))

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_details["raw_text"], jd_clean])
    keyword_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "keyword_similarity": round(keyword_sim * 100, 2)
    }

def get_final_score_and_suggestions(analysis):
    if analysis["matched_skills"] or analysis["missing_skills"]:
        skill_score = (
            len(analysis["matched_skills"]) /
            (len(analysis["matched_skills"]) + len(analysis["missing_skills"])) * 70
        )
    else:
        skill_score = 0

    keyword_score = (analysis["keyword_similarity"] / 100) * 30
    return round(skill_score + keyword_score, 2), round(keyword_score, 2)

def chipify(items, kind="good"):
    if not items:
        return "<span>No items</span>"
    bg = "#e6ffed" if kind == "good" else "#fff4e6"
    color = "#067d3e" if kind == "good" else "#8a4b00"
    return " ".join(
        [
            f"<span style='display:inline-block;padding:6px 10px;margin:4px 6px;"
            f"border-radius:16px;background:{bg};color:{color};font-size:13px;"
            f"border:1px solid rgba(0,0,0,0.05)'>{html.escape(s)}</span>"
            for s in items
        ]
    )

def analyze_with_llm(resume_text, jd_text):
    prompt = f"""
    You are a recruitment expert. Compare the following resume and job description,
    and provide feedback on strengths, weaknesses, and improvement suggestions.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}
    """
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM analysis failed: {e}"

# ------------------- UI -------------------
st.markdown("### 📂 Upload Resume & 📄 Job Description")

col_left, spacer, col_right = st.columns([1, 0.05, 1])

with col_left:
    upload_option = st.radio("Choose Resume Input Method:", ["Upload PDF", "Paste Text"])
    resume_text = ""
    if upload_option == "Upload PDF":
        resume_file = st.file_uploader("📁 Upload Resume (PDF)", type=["pdf"])
        if resume_file:
            resume_text = extract_text_from_pdf(resume_file)
    elif upload_option == "Paste Text":
        resume_text = st.text_area("✏️ Paste Resume Text Here", height=300)

with col_right:
    jd_input = st.text_area("📄 Paste Job Description", height=300)

# Analyze Button
if st.button("🚀 Analyze Resume", use_container_width=True):
    if not resume_text.strip() or not jd_input.strip():
        st.error("⚠️ Please upload a resume and paste a job description.")
    else:
        with st.spinner("🔍 Analyzing Resume with Agentic Intelligence..."):
            resume_details = get_resume_details(resume_text)
            analysis = compare_resume_with_jd(resume_details, jd_input)
            score, keyword_match = get_final_score_and_suggestions(analysis)
            llm_insight = analyze_with_llm(resume_text, jd_input)

        # Results Section
        st.markdown("---")
        st.subheader("📊 Scores")
        col1, col2 = st.columns(2)
        col1.metric("Similarity Score", f"{score}%")
        col2.metric("Keyword Match Score", f"{keyword_match}%")

        # Skills Overview in two columns
        st.markdown("### 🛠 Skills Overview")
        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            st.markdown("✅ **Matched Skills**")
            st.markdown(chipify(analysis["matched_skills"], kind="good"), unsafe_allow_html=True)
        with skill_col2:
            st.markdown("❌ **Missing Skills**")
            st.markdown(chipify(analysis["missing_skills"], kind="bad"), unsafe_allow_html=True)

        # Styled Feedback Section
        st.markdown("### 🤖 LLM-Based Feedback")
        st.markdown(
            f"""
            <div style='padding:15px; background-color:#f9f9f9; border-radius:10px; border:1px solid #ddd;'>
                <strong>Feedback Summary:</strong><br>
                {llm_insight.replace("\n", "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )
