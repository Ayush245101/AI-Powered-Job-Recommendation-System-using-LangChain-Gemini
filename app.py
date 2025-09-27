import streamlit as st
from pathlib import Path
from src.utils.settings import get_settings
from src.ingestion.load_jobs import load_jobs
from src.ingestion.resume_parser import parse_resume, parse_pdf
from src.retrieval.vector_store import VectorStore
from src.pipeline.recommend import recommend

st.set_page_config(page_title="AI Job Recommender", layout="wide")

settings = get_settings()

@st.cache_resource(show_spinner="Loading vector store ...")
def get_vector_store(jobs):
    vs = VectorStore()
    vs.ensure(jobs)
    return vs

def main():
    st.title("🎯 AI-Powered Job Recommendation System")
    st.markdown("Upload your resume or enter skills to get personalized job matches.")

    with st.sidebar:
        st.header("User Input")
        uploaded = st.file_uploader("Upload Resume (txt/pdf)", type=["txt", "pdf"])  # PDF parsing added
        manual_skills = st.text_area("Skills (comma-separated)")
        location_pref = st.text_input("Preferred Location (optional)")
        job_type_pref = st.selectbox("Job Type Preference", ["", "Internship", "Full-Time"])  # simple filter
        submit = st.button("Get Recommendations")

    jobs = load_jobs()
    vector_store = get_vector_store(jobs)

    user_profile = {"skills": []}

    if uploaded:
        raw_bytes = uploaded.read()
        if uploaded.type == "application/pdf" or uploaded.name.lower().endswith('.pdf'):
            pdf_text = parse_pdf(raw_bytes)
            if pdf_text:
                parsed = parse_resume(pdf_text)
                user_profile["skills"].extend(parsed["skills"])
            else:
                st.warning("PDF parsing returned no text; consider uploading a TXT version.")
        else:
            text = raw_bytes.decode("utf-8", errors="ignore")
            parsed = parse_resume(text)
            user_profile["skills"].extend(parsed["skills"])
    if manual_skills:
        user_profile["skills"].extend([s.strip().lower() for s in manual_skills.split(',') if s.strip()])

    user_profile["skills"] = list(dict.fromkeys(user_profile["skills"]))  # dedupe

    if submit:
        if not user_profile["skills"]:
            st.warning("Please provide skills via resume or manual input.")
            return
        query = " ".join(user_profile["skills"]) + f" {location_pref} {job_type_pref}".strip()
        retrieved = vector_store.search(query, k=12)
        with st.spinner("Generating recommendations with Gemini..."):
            try:
                results = recommend(user_profile, retrieved, top_n=5)
            except Exception as e:
                st.error(f"Recommendation pipeline error: {e}")
                return
        if not results:
            st.error("No recommendations produced.")
        else:
            st.subheader("Top Recommendations")
            for i, r in enumerate(results, start=1):
                st.markdown(f"**{i}. {r.get('title','?')} @ {r.get('company','?')}**")
                st.caption(r.get('reason',''))
            if not get_settings().gemini_api_key:
                st.info("Using heuristic fallback (no GEMINI_API_KEY set). Add key to enable LLM reasoning.")
    else:
        st.info("Enter your details and press 'Get Recommendations'.")

if __name__ == "__main__":
    main()
