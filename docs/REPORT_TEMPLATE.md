# Project Report: AI-Powered Job Recommendation System

## 1. Problem Explanation
(Describe the challenge students face discovering relevant jobs/internships and why personalization matters.)

## 2. Solution Overview
(Brief how RAG + Gemini + vector retrieval addresses the problem.)

## 3. Architecture Diagram
(Insert or describe: Ingestion -> Embeddings -> Vector Store -> Retrieval -> LLM Re-Ranking -> UI)

## 4. Data Sources
- Sample CSV provided
- (Planned) Official APIs / approved sources for live postings

## 5. RAG Components
| Component | Implementation | Notes |
|----------|----------------|-------|
| Document Source | Job CSV | Placeholder for live ingestion |
| Embeddings | SentenceTransformer | Cosine similarity via FAISS |
| Vector Store | FAISS | ID + in-memory metadata |
| LLM | Gemini 1.5 Flash | Ranking + reasoning |
| Prompt | System + user compiled | JSON output enforced |

## 6. Prompt Strategy
(Explain system prompt, JSON enforcement, truncation, few-shot potential.)

## 7. Key Modules
- `ingestion/` loading & parsing
- `retrieval/` vector store management
- `pipeline/` LLM ranking logic
- `app.py` Streamlit UI

## 8. Challenges & Resolutions
| Challenge | Resolution |
|-----------|-----------|
| Missing live data | Added pluggable ingestion abstraction |
| Parsing messy resumes | Started with regex; plan ML-based NER |
| JSON parsing errors | Regex bracket extraction fallback |
| Cold start vector store | Auto-build on first launch |

## 9. Evaluation Ideas
- Precision@K vs user accepted jobs
- Feedback loop storing likes
- Skill coverage vs job requirements

## 10. Future Improvements
(Feedback learning, re-ranker, structured resume parsing, multi-lingual, fairness filters.)

## 11. Setup & Run Summary
(Reference README: pip install, env, streamlit run.)

## 12. Learnings / Reflections
(Insert personal reflections.)

## 13. Appendix
- Sample prompts
- Example outputs
