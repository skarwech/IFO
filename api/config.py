"""
API configuration and settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """API settings with environment variable support."""
    
    # API Settings
    API_TITLE: str = "IFO Backend API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Intelligent Flood Optimization - Backend API for AquaOptAI"
    
    # CORS Settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://aquaoptai.vercel.app",  # Production frontend
    ]
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # IFO Configuration
    IFO_CONFIG_PATH: str = "./config.yaml"
    DATA_DIR: str = "./data"
    RESULTS_DIR: str = "./results"
    
    # Optimization Settings
    DEFAULT_HORIZON: int = 96
    MAX_HORIZON: int = 288
    MIN_TIMESTEP: float = 0.25  # 15 minutes
    
    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds
    WS_MAX_CONNECTIONS: int = 100
    
    # Database/Cache (optional for future)
    REDIS_URL: Optional[str] = None
    DATABASE_URL: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security (for production)
    API_KEY: Optional[str] = None
    SECRET_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
