from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from ..utils.settings import get_settings
from ..utils.logging import get_logger
import math

logger = get_logger()

SYSTEM_PROMPT = """You are an assistant that ranks job postings for a student. Given user skills and retrieved job postings, produce a ranked list with:
- title
- company
- short reason (<=25 words)
Return JSON list with keys: title, company, reason, score (0-1).
"""

def llm_client():
    s = get_settings()
    if not s.gemini_api_key:
        return None
    try:
        # Use a widely available model alias
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=s.gemini_api_key, temperature=0.3)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to init Gemini client: {e}")
        return None


def recommend(user_profile: Dict, retrieved: List[Dict], top_n: int = 5) -> List[Dict]:
    llm = llm_client()
    if not retrieved:
        return []

    if llm is None:
        # Offline heuristic fallback: score overlap of skills
        user_skills = set(user_profile.get("skills", []))
        scored = []
        for r in retrieved:
            job = r["job"]
            job_skills = set(job.get("skills_list", []))
            inter = user_skills.intersection(job_skills)
            score = (len(inter) / (len(user_skills) + 1e-6)) * 0.7 + r.get("score", 0) * 0.3
            scored.append({
                "title": job["title"],
                "company": job["company"],
                "reason": f"Skill overlap: {', '.join(list(inter)[:4])}",
                "score": round(float(score), 3)
            })
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]

    # LLM path
    skills = ", ".join(user_profile.get("skills", []))
    jobs_block = []
    for r in retrieved:
        j = r["job"]
        jobs_block.append(f"- {j['title']} @ {j['company']} | skills: {', '.join(j.get('skills_list', []))} | desc: {j['description'][:120]}")
    content = f"User skills: {skills}\nJobs:\n" + "\n".join(jobs_block)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content)
    ]
    try:
        resp = llm.invoke(messages)
        text = resp.content
        import json, re
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            logger.warning("LLM response didn't contain JSON list; returning empty")
            return []
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data[:top_n]
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            return []
    except Exception as e:
        logger.warning(f"LLM call failed, falling back to heuristic: {e}")
        # fallback to heuristic
        user_skills = set(user_profile.get("skills", []))
        scored = []
        for r in retrieved:
            job = r["job"]
            job_skills = set(job.get("skills_list", []))
            inter = user_skills.intersection(job_skills)
            score = (len(inter) / (len(user_skills) + 1e-6)) * 0.7 + r.get("score", 0) * 0.3
            scored.append({
                "title": job["title"],
                "company": job["company"],
                "reason": f"Skill overlap: {', '.join(list(inter)[:4])}",
                "score": round(float(score), 3)
            })
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]
