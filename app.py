import streamlit as st
from datetime import date, datetime
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
        st.divider()
        st.subheader("Filters")
        hide_closed = st.checkbox("Hide closed roles", value=False)
        only_closing_soon = st.checkbox("Only show closing soon (≤ 7 days)", value=False)
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
            # helper to compute deadline state
            def _deadline_state(r: dict):
                expired = False; closing_soon = False; days_left = None
                raw_deadline = (r.get('apply_by') or '').strip()
                if raw_deadline:
                    try:
                        dl = datetime.strptime(raw_deadline, "%Y-%m-%d").date()
                        today = date.today()
                        if dl < today:
                            expired = True
                        else:
                            days_left = (dl - today).days
                            if days_left <= 7:
                                closing_soon = True
                    except Exception:
                        pass
                return expired, closing_soon, days_left

            # apply filters
            filtered = []
            for r in results:
                expired, closing_soon, days_left = _deadline_state(r)
                if hide_closed and expired:
                    continue
                if only_closing_soon and not (closing_soon and not expired):
                    continue
                # store computed flags for reuse in rendering
                r['_expired'] = expired; r['_closing_soon'] = closing_soon; r['_days_left'] = days_left
                filtered.append(r)

            if not filtered:
                st.info("No results after applying filters.")
                return

            for i, r in enumerate(filtered, start=1):
                title = r.get('title','?'); company = r.get('company','?')
                st.markdown(f"**{i}. {title} @ {company}**")
                # meta line with location/type and optional apply_by
                meta_bits = []
                if r.get('location'): meta_bits.append(r['location'])
                if r.get('type'): meta_bits.append(r['type'])
                # compute deadline state
                expired = r.get('_expired', False); closing_soon = r.get('_closing_soon', False); days_left = r.get('_days_left')
                raw_deadline = (r.get('apply_by') or '').strip()
                if raw_deadline and r.get('_days_left') is None and not expired:
                    # unknown format; show raw
                    meta_bits.append(f"Apply by: {raw_deadline}")
                if raw_deadline and not expired and days_left is not None:
                    meta_bits.append(f"Apply by: {raw_deadline} ({days_left} days left)")
                if expired:
                    meta_bits.append("Applications closed")
                if meta_bits:
                    st.caption(" • ".join(meta_bits))
                # reason and apply link
                if r.get('reason'):
                    st.write(r['reason'])
                if r.get('apply_url') and not expired:
                    st.link_button("Apply", r['apply_url'], help="Opens the application link in your browser")
            if not get_settings().gemini_api_key:
                st.info("Using heuristic fallback (no GEMINI_API_KEY set). Add key to enable LLM reasoning.")
    else:
        st.info("Enter your details and press 'Get Recommendations'.")

if __name__ == "__main__":
    main()
