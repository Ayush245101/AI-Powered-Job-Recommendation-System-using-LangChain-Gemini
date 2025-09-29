from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from ..utils.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger()

class VectorStore:
    """Simple hashing-embedding vector store: zero external deps, fast, portable."""
    def __init__(self, dim: int = 128):
        self.settings = get_settings()
        self.dim = dim
        self.jobs: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, jobs: List[Dict]):
        texts = [self._job_text(j) for j in jobs]
        embeddings = self._hash_embed(texts, self.dim)
        self.embeddings = embeddings
        self.jobs = jobs
        logger.info(f"Vector store built with {len(jobs)} items (hash dim={self.dim})")
        self._persist()

    def _persist(self):
        path = Path(self.settings.vector_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path) + "_jobs.npy", np.array(self.jobs, dtype=object), allow_pickle=True)
        if self.embeddings is not None:
            np.save(str(path) + "_emb.npy", self.embeddings)

    def load(self):
        path = Path(self.settings.vector_store_path)
        jobs_file = Path(str(path) + "_jobs.npy")
        emb_file = Path(str(path) + "_emb.npy")
        if not jobs_file.exists() or not emb_file.exists():
            raise FileNotFoundError("Vector store not found; build first.")
        self.jobs = list(np.load(str(jobs_file), allow_pickle=True))
        self.embeddings = np.load(str(emb_file))
        logger.info("Vector store loaded (jobs: %d)" % len(self.jobs))

    def ensure(self, jobs: List[Dict]):
        if self.embeddings is None:
            try:
                self.load()
            except FileNotFoundError:
                self.build(jobs)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.embeddings is None:
            raise RuntimeError("Index not initialized")
        q = self._hash_embed([query], self.dim)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims = (self.embeddings @ q_norm.T).squeeze(1)
        idxs_sorted = np.argsort(-sims)[:k]
        results = []
        for i in idxs_sorted:
            job = self.jobs[int(i)]
            results.append({"job": job, "score": float(sims[int(i)])})
        return results

    @staticmethod
    def _job_text(job: Dict) -> str:
        return f"{job['title']} {job['company']} {job['location']} {job['type']} {' '.join(job.get('skills_list', []))} {job['description']}"

    @staticmethod
    def _hash_embed(texts: List[str], dim: int = 128) -> np.ndarray:
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = (hash(tok) % dim)
                vecs[i, h] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms
