"""API authentication and security middleware."""
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import secrets
import hashlib
import time
from functools import lru_cache

from api.config import settings


# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_id: [(timestamp, count)]}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                (ts, count) for ts, count in self.requests[client_id]
                if ts > window_start
            ]
        
        # Count requests in window
        current_count = sum(count for _, count in self.requests.get(client_id, []))
        
        if current_count >= self.max_requests:
            return False
        
        # Add new request
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id].append((now, 1))
        
        return True


# Global rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Verified API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    # If no API key configured, allow all requests
    if not settings.API_KEY:
        return "anonymous"
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )
    
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return api_key


def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """
    Check rate limit for client.
    
    Args:
        api_key: Verified API key
        
    Returns:
        API key if allowed
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Use API key as client identifier
    client_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )
    
    return api_key


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


# Optional: JWT token authentication (for future use)
security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> dict:
    """
    Verify JWT token (placeholder for future implementation).
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        Token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token"
        )
    
    # TODO: Implement JWT verification with python-jose
    # For now, simple placeholder
    token = credentials.credentials
    
    if not token:
        raise HTTPException(
            status_code=403,
            detail="Invalid token"
        )
    
    return {"sub": "user", "token": token}
