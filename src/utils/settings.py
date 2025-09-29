from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    vector_store_path: str = Field(default="data/vector_store", alias="VECTOR_STORE_PATH")
    top_k: int = 6
    serp_api_key: str = Field(default="", alias="SERP_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
