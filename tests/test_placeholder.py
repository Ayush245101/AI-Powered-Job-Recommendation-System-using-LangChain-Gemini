from src.ingestion.load_jobs import load_jobs
from src.retrieval.vector_store import VectorStore
from src.pipeline.recommend import recommend


def test_vector_retrieval_and_fallback_recommendation():
    jobs = load_jobs()
    vs = VectorStore(); vs.ensure(jobs)
    results = vs.search('python internship machine learning', k=5)
    assert results, 'Expected at least one retrieval result'
    user_profile = {'skills': ['python','machine','learning']}
    recs = recommend(user_profile, results, top_n=3)
    assert recs, 'Expected fallback recommendations'
    assert 'title' in recs[0]
