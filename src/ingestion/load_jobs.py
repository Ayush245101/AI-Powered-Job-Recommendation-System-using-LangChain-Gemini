import pandas as pd
from pathlib import Path
from typing import List, Dict
from ..utils.logging import get_logger

logger = get_logger()

REQUIRED_COLUMNS = ["id","title","company","location","type","skills","description"]

def load_jobs(csv_path: str = "data/sample_jobs.csv") -> List[Dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Job dataset not found at {csv_path}")
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df = df.fillna("")
    # Normalize skills to list
    df["skills_list"] = df["skills"].apply(lambda s: [x.strip().lower() for x in str(s).split(';') if x.strip()])
    records = df.to_dict(orient="records")
    logger.info(f"Loaded {len(records)} job records")
    return records
