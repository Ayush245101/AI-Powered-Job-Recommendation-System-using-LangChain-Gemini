from typing import List, Dict, Any
import json
from ..utils.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger()

SYSTEM_PROMPT = (
    "You are a helpful assistant ranking job postings for a candidate. "
    "Return ONLY a JSON list of objects with keys: job_id, score (0-1), reason. "
    "Consider skill overlap, role fit, and description relevance."
)

def _init_gemini():
    settings = get_settings()
    if not settings.gemini_api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        model_name = getattr(settings, "gemini_model", None) or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:  # pragma: no cover
        logger.warning(f"Gemini init failed: {e}")
        return None

def _heuristic_rank(user_profile: Dict[str, Any], retrieved: List[Dict], top_n: int = 5) -> List[Dict]:
    skills = set(map(str.lower, user_profile.get("skills", [])))
    ranked = []
    for item in retrieved:
        job = item["job"]
        job_skills = set(map(str.lower, job.get("skills_list", [])))
        overlap = len(skills & job_skills)
        sim = float(item.get("score", 0.0))
        score = 0.6 * (overlap / (len(skills) + 1e-6)) + 0.4 * sim
        ranked.append(_flatten(job, score, f"Overlap {overlap} skills; retrieval sim {sim:.2f}"))
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]

def recommend(user_profile: Dict[str, Any], retrieved: List[Dict], top_n: int = 5) -> List[Dict]:
    model = _init_gemini()
    if model is None:
        return _heuristic_rank(user_profile, retrieved, top_n)
    try:
        jobs_slim = [
            {"id": j["job"].get("id", i), "title": j["job"]["title"], "skills": j["job"].get("skills_list", [])}
            for i, j in enumerate(retrieved)
        ]
        user_payload = {
            "skills": user_profile.get("skills", []),
            "preferences": user_profile.get("preferences", {})
        }
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"User profile JSON: {json.dumps(user_payload)}\n"
            f"Jobs JSON: {json.dumps(jobs_slim)}\n"
            "Return JSON only."
        )
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
        try:
            data = json.loads(text)
        except Exception:
            # try to extract JSON substring
            import re
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                raise ValueError("No JSON in response")
            data = json.loads(m.group(0))
        id_to_job = {jobs_slim[i]["id"]: retrieved[i]["job"] for i in range(len(jobs_slim))}
        scored = []
        for it in data:
            jid = it.get("job_id")
            if jid in id_to_job:
                scored.append(_flatten(id_to_job[jid], float(it.get("score", 0.0)), it.get("reason", "")))
        if not scored:
            return _heuristic_rank(user_profile, retrieved, top_n)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]
    except Exception as e:
        logger.warning(f"LLM ranking failed; falling back. Error: {e}")
        return _heuristic_rank(user_profile, retrieved, top_n)

def _flatten(job: Dict[str, Any], score: float, reason: str) -> Dict[str, Any]:
    return {
        "id": job.get("id"),
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "type": job.get("type"),
        "skills_list": job.get("skills_list", []),
        "description": job.get("description"),
        "apply_url": job.get("apply_url"),
        "apply_by": job.get("apply_by"),
        "score": score,
        "reason": reason,
    }
