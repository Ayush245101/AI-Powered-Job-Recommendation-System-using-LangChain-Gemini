# AI-Powered Job Recommendation System (Streamlit + Gemini)

![CI](https://github.com/Ayush245101/AI-Powered-Job-Recommendation-System-using-LangChain-Gemini/actions/workflows/ci.yml/badge.svg)

## Overview
A Streamlit web application that recommends personalized jobs or internships to students by combining user input (skills, experience, preferences) and resume parsing with a Retrieval-Augmented Generation (RAG) pipeline powered by a lightweight vector store and Google's Gemini.

## Key Features
- Upload resume (PDF/text) or manually enter skills & experience
- Skill extraction and normalization
- Lightweight job postings ingestion (sample dataset + pluggable live scrapers placeholder)
- Lightweight vector store using hashing-based embeddings (no FAISS needed)
- Semantic retrieval over job descriptions and skills
- Gemini LLM (google-generativeai) for personalized recommendation reasoning & concise explanations with safe fallback
- Application links and deadlines: shows "Apply" button when available, displays "Apply by" date, flags expired roles, and indicates days left (<= 7 days as "closing soon").
- Streamlit UI with interactive filtering & export

## Project Structure
```
app.py
src/
  ingestion/
    load_jobs.py
    resume_parser.py
  retrieval/
    vector_store.py
  pipeline/
    recommend.py
  utils/
    settings.py
    logging.py
data/
  sample_jobs.csv
notebooks/
  exploration.ipynb (placeholder)
docs/
  REPORT_TEMPLATE.md
.env.example
requirements.txt
```

## Getting Started
### 1. Clone & Install
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your Gemini API key.

Environment variables:
- `GEMINI_API_KEY` (required for LLM ranking)
- `EMBEDDING_MODEL` (optional)
- `VECTOR_STORE_PATH` (optional)
- `SERP_API_KEY` (optional, reserved for future job-source integration)

Note: `.env` is git-ignored; never commit secrets.

### 3. Build Vector Store
The first app launch will automatically build a lightweight vector store from `data/sample_jobs.csv` if it doesn't exist.

### 4. Run App
```bash
streamlit run app.py
```

Or via VS Code task: open Command Palette -> Run Task -> "Run Streamlit App".

If `GEMINI_API_KEY` is not set the app will still work using a heuristic fallback (skill overlap + similarity scores) with an info banner.

## Deployments

### Option A: GitHub Container Registry (GHCR)
This repo is configured to build and push a Docker image to GHCR on pushes to `main` (and tags `v*.*.*`).

- Image name: `ghcr.io/<owner>/<repo>:<tag>` (e.g., `ghcr.io/Ayush245101/AI-Powered-Job-Recommendation-System-using-LangChain-Gemini:latest`)
- Workflow file: `.github/workflows/docker-publish.yml`

Run locally with Docker:
```powershell
docker run --rm -p 8501:8501 -e GEMINI_API_KEY=$Env:GEMINI_API_KEY ghcr.io/Ayush245101/AI-Powered-Job-Recommendation-System-using-LangChain-Gemini:latest
```

### Option B: Streamlit Community Cloud
1) Fork this repo
2) Create a new app in Streamlit Cloud pointing to `app.py`
3) Add a secret called `GEMINI_API_KEY` (if you want LLM ranking)

### Option C: Any VM / Render / Fly.io
- Build with Dockerfile provided or run via Python directly
- Expose port 8501; set `GEMINI_API_KEY` for LLM mode

## Sample Input
- Skills: Python, Machine Learning, NLP
- Preferences: Remote, Internship, Location: "India"

## Sample Output (Excerpt)
```
1. ML Research Intern @ Alpha Labs
Reason: Matches ML + NLP focus; internship; remote-friendly; aligns with your Python + transformer experience.
Confidence: High
 Meta: Remote • Internship • Apply by: 2025-11-15 (12 days left)
```

## RAG Flow
1. Ingest job postings -> clean -> embed -> store locally (hashing-based)
2. Parse resume / user input -> extract skills & preferences
3. Form retrieval query & metadata filters
4. Retrieve top-K similar job vectors
5. Re-rank & summarize with Gemini using prompt template (or heuristic fallback)
6. Streamlit displays ranked recommendations

## Extending (Next Steps)
- Add real job source scrapers (LinkedIn/Google Jobs) via official APIs or compliant scraping
- Add feedback loop (thumbs up/down) -> store in simple SQLite + fine-tune rerank logic
- Add caching layer (Redis) for repeated queries
- Implement multilingual support

## Disclaimer
Live scraping may violate ToS; integrate only with proper permissions or official APIs. Provided code ships with a static sample dataset.

## License
MIT
