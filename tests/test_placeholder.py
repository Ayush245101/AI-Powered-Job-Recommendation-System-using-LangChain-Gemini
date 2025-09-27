import os
import pytest

# Ensure tests never download large models
os.environ.setdefault("LIGHTWEIGHT_EMBEDDINGS", "1")
from src.ingestion.load_jobs import load_jobs
from src.retrieval.vector_store import VectorStore
from src.pipeline.recommend import recommend


@pytest.mark.timeout(60)
def test_vector_retrieval_and_fallback_recommendation():
    # Keep k small to avoid heavy downloads/timeouts
    jobs = load_jobs()
    vs = VectorStore(); vs.ensure(jobs)
    results = vs.search('python internship machine learning', k=2)
    assert results, 'Expected at least one retrieval result'
    user_profile = {'skills': ['python','machine','learning']}
    recs = recommend(user_profile, results, top_n=2)
    assert recs, 'Expected fallback recommendations or LLM results'
    assert 'title' in recs[0]
