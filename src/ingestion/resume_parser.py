from typing import List, Dict
import re
from io import BytesIO

try:
    from pypdf import PdfReader  # lightweight PDF text extraction
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

BULLET_SPLIT = re.compile(r"[\n\r]+[\-\*\u2022]\s+")
SKILL_TOKENIZER = re.compile(r"[^a-zA-Z0-9+#]+")

COMMON_SKILL_ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
}

def normalize_skill(token: str) -> str:
    t = token.strip().lower()
    return COMMON_SKILL_ALIASES.get(t, t)

def extract_skills(text: str) -> List[str]:
    tokens = [normalize_skill(t) for t in SKILL_TOKENIZER.split(text) if t.strip()]
    uniq = []
    for t in tokens:
        if len(t) < 2:  # skip single letters
            continue
        if t not in uniq:
            uniq.append(t)
    return uniq[:50]

def parse_resume(raw: str) -> Dict:
    # Simple heuristic extraction
    skills = extract_skills(raw)
    return {
        "skills": skills,
        "raw_length": len(raw),
        "skill_count": len(skills)
    }

def parse_pdf(file_bytes: bytes) -> str:
    """Parse PDF bytes to text. Returns empty string if PDF parsing unavailable."""
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        texts = []
        for page in reader.pages[:20]:  # safety limit
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)
    except Exception:
        return ""
