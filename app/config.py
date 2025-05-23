from pydantic import BaseModel
from typing import Optional
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI API Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model_name: str = "gpt-4"
    temperature: float = 0.1
    
    # Together AI Configuration
    together_api_key: Optional[str] = os.getenv("TOGETHER_API_KEY")
    
    # Retriever Configuration
    index_path: Optional[str] = "./data/index.faiss"
    embedding_model: str = "dmis-lab/biobert-base-cased-v1.2"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Data Sources
    pubmed_api_key: str = ""
    orphanet_api_key: Optional[str] = os.getenv("ORPHANET_API_KEY")
    pubmed_email: str = "nvpatil@usc.edu"
    pubmed_tool_name: str = "RAG_Medical_Assistant"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings() 