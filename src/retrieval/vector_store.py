from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import os
from ..utils.settings import get_settings
from ..utils.logging import get_logger

try:  # runtime optional: Windows may struggle with faiss-cpu
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

logger = get_logger()

class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self._light = bool(os.environ.get("LIGHTWEIGHT_EMBEDDINGS"))
        self.model = None
        if not self._light:
            # Lazy import to avoid heavy dependency during fast tests
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(self.settings.embedding_model)
        self.index = None
        self.jobs: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None  # fallback store

    def build(self, jobs: List[Dict]):
        texts = [self._job_text(j) for j in jobs]
        if self._light:
            embeddings = self._hash_embed(texts)
            self.embeddings = embeddings
            self.index = None
        else:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if not self._light and _FAISS_AVAILABLE:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            # normalize for cosine manually
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            embeddings = embeddings / norms
            self.embeddings = embeddings
        self.jobs = jobs
        logger.info(f"Vector store built with {len(jobs)} items")
        self._persist(embeddings)

    def _persist(self, embeddings: np.ndarray):
        # Always persist job metadata even if FAISS missing (store JSON-ish npy)
        path = Path(self.settings.vector_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array(self.jobs, dtype=object)
        np.save(str(path) + "_jobs.npy", meta, allow_pickle=True)
        if self._light or (not _FAISS_AVAILABLE) or self.index is None:
            return  # skip FAISS index persistence
        faiss.write_index(self.index, str(path))

    def load(self):
        path = Path(self.settings.vector_store_path)
        jobs_file = Path(str(path) + "_jobs.npy")
        if not jobs_file.exists():
            raise FileNotFoundError("Metadata not found; rebuild store.")
        try:
            stored_jobs = np.load(str(jobs_file), allow_pickle=True)
            self.jobs = list(stored_jobs)
        except Exception:
            self.jobs = []
        if _FAISS_AVAILABLE and path.exists():
            try:
                self.index = faiss.read_index(str(path))
            except Exception:
                self.index = None
        else:
            self.index = None
        logger.info("Vector store loaded (jobs: %d, faiss: %s)" % (len(self.jobs), bool(self.index)))

    def ensure(self, jobs: List[Dict]):
        if self.index is None:
            try:
                self.load()
            except FileNotFoundError:
                self.build(jobs)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None and self.embeddings is None:
            raise RuntimeError("Index not initialized")
            if self._light:
                q_emb = self._hash_embed([query])
            else:
                q_emb = self.model.encode([query], convert_to_numpy=True)
            if not self._light and _FAISS_AVAILABLE and self.index is not None:
            faiss.normalize_L2(q_emb)
            scores, idxs = self.index.search(q_emb, k)
            pairs = list(zip(scores[0], idxs[0]))
        else:
            # manual cosine
            q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            sims = (self.embeddings @ q_norm.T).squeeze(1)
            idxs_sorted = np.argsort(-sims)[:k]
            pairs = [(sims[i], i) for i in idxs_sorted]
        results = []
        for score, i in pairs:
            if i == -1:
                continue
            job = self.jobs[i]
            results.append({"job": job, "score": float(score)})
        return results

    @staticmethod
    def _job_text(job: Dict) -> str:
        return f"{job['title']} {job['company']} {job['location']} {job['type']} {' '.join(job.get('skills_list', []))} {job['description']}"

    @staticmethod
    def _hash_embed(texts: List[str], dim: int = 128) -> np.ndarray:
        """Simple, fast hashing-based embedding for tests/CI."""
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = (hash(tok) % dim)
                vecs[i, h] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms
