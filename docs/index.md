# AI-Powered Job Recommendation System

Welcome! This is the public landing page for the project.

What this is: A Streamlit app that recommends jobs based on your skills and resume. The app needs a running Python/Streamlit server, so GitHub Pages hosts this static overview with quick run options below.

## Run Options

- Run in GitHub Codespaces (interactive, in browser)
  1. Open Codespaces for this repo:
     - https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=Ayush245101/AI-Powered-Job-Recommendation-System-using-LangChain-Gemini
  2. In the terminal (optional): `export GEMINI_API_KEY=your_key`
  3. Start: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`
  4. Click the forwarded port prompt to open the app

- Streamlit Community Cloud (hosted from your GitHub fork)
  1. Fork the repo to your GitHub
  2. Create an app pointing to `app.py` on `main`
  3. Add a secret `GEMINI_API_KEY="your_key"` (optional, enables LLM ranking)

- Docker (run locally)
  1. Install Docker Desktop
  2. Pull image (after GHCR publish completes):
     - `ghcr.io/ayush245101/ai-powered-job-recommendation-system-using-langchain-gemini:latest`
  3. Run:
     - `docker run --rm -p 8501:8501 -e GEMINI_API_KEY=$GEMINI_API_KEY ghcr.io/ayush245101/ai-powered-job-recommendation-system-using-langchain-gemini:latest`

## Notes
- Without GEMINI_API_KEY, the app still works using a heuristic fallback.
- GitHub Pages cannot host the Streamlit server; use Codespaces or Streamlit Cloud to interact with the app online.
- Source repo: https://github.com/Ayush245101/AI-Powered-Job-Recommendation-System-using-LangChain-Gemini
