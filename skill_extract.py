import re
import json
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
SEMANTIC_MATCH_THRESHOLD = 0.75


# Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load skill database only once
@st.cache_data
def load_skill_db():
    with open("skills.json", "r") as f:
        skill_db = json.load(f)
    all_skills = set(skill.lower() for cat in skill_db.values() for skill in cat)
    return all_skills

ALL_SKILLS = load_skill_db()

# ---------------- TEXT CLEANING ----------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\-\.\#]', ' ', text)
    return text

def clean_tokens(text):
    tokens = re.findall(r'\b[a-zA-Z0-9\-\+\.\#]+\b', text.lower())
    return list(set(tokens))

# ---------------- RESUME EXTRACTION ----------------
def get_resume_details(resume_text):
    clean_text = normalize_text(resume_text)
    tokens = clean_tokens(clean_text)
    skills = set(token for token in tokens if token in ALL_SKILLS)
    return {"skills": list(skills), "raw_text": clean_text}

# ---------------- COMPARISON LOGIC ----------------
def compare_resume_with_jd(resume_details, jd_text):
    jd_clean = normalize_text(jd_text)
    jd_tokens = clean_tokens(jd_clean)
    jd_skills = set(token for token in jd_tokens if token in ALL_SKILLS)
    resume_skills = set(resume_details["skills"])

    # Handle case where either JD or Resume has no matching skills
    if not jd_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "keyword_similarity": 0
        }

    # Embeddings for skills
    resume_embeddings = model.encode(list(resume_skills), convert_to_tensor=True) if resume_skills else None
    jd_embeddings = model.encode(list(jd_skills), convert_to_tensor=True) if jd_skills else None

    matched_skills = []
    if resume_embeddings is not None and jd_embeddings is not None:
        cos_scores = util.cos_sim(jd_embeddings, resume_embeddings)
        for i, jd_skill in enumerate(jd_skills):
            if max(cos_scores[i]) > 0.75:  # semantic match threshold
                matched_skills.append(jd_skill)

    # Skills in JD but not matched
    missing_skills = list(jd_skills - set(matched_skills))

    # TF-IDF similarity between full resume and JD
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_details["raw_text"], jd_clean])
    keyword_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "keyword_similarity": round(keyword_sim * 100, 2)
    }

# ---------------- FINAL SCORE ----------------
def get_final_score_and_suggestions(analysis):
    skill_score = (
        len(analysis["matched_skills"]) /
        (len(analysis["matched_skills"]) + len(analysis["missing_skills"])) * 70
    ) if (analysis["matched_skills"] or analysis["missing_skills"]) else 0

    keyword_score = (analysis["keyword_similarity"] / 100) * 30
    final_score = round(skill_score + keyword_score, 2)

    return final_score, round(keyword_score, 2)
