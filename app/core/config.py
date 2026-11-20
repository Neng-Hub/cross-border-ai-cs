from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "none"
    OPENAI_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    LLM_MODEL: str = "ep-placeholder"
    EMBEDDING_API_KEY: str = "none"
    EMBEDDING_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    EMBEDDING_MODEL: str = "ep-placeholder"

    class Config:
        env_file = ".env"

settings = Settings()