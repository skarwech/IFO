"""
API configuration and settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, Union
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """API settings with environment variable support."""
    
    # API Settings
    API_TITLE: str
    API_VERSION: str
    API_DESCRIPTION: str = "Intelligent Flood Optimization - Backend API for AquaOptAI"
    
    # CORS Settings
    CORS_ORIGINS: Union[list[str], str]
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # Server Settings
    HOST: str
    PORT: int
    RELOAD: bool = True
    
    # IFO Configuration
    IFO_CONFIG_PATH: str
    DATA_DIR: str
    RESULTS_DIR: str
    
    # Optimization Settings
    DEFAULT_HORIZON: int
    MAX_HORIZON: int
    MIN_TIMESTEP: float = 0.25  # 15 minutes
    
    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL: int
    WS_MAX_CONNECTIONS: int
    
    # Database/Cache (optional for future)
    REDIS_URL: Optional[str] = None
    DATABASE_URL: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security (for production)
    API_KEY: Optional[str] = None
    SECRET_KEY: Optional[str] = None
    ENABLE_AUTH: bool
    ENABLE_RATE_LIMIT: bool
    
    # Gemini API for AquaBot chatbot
    GEMINI_API_KEY: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields from .env
    )


# Singleton settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
