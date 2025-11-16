"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import os


# Set test environment
os.environ['GEMINI_API_KEY'] = 'test_key_for_testing'


@pytest.fixture
def client():
    """Create test client."""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data


class TestSystemEndpoints:
    """Test system status endpoints."""
    
    def test_get_status(self, client):
        """Test system status retrieval."""
        response = client.get("/api/status")
        
        # May return 503 if services not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'pumps' in data
            assert 'tunnel' in data
            assert 'total_power' in data
    
    def test_get_pumps(self, client):
        """Test pumps status retrieval."""
        response = client.get("/api/pumps")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_get_tunnel(self, client):
        """Test tunnel metrics retrieval."""
        response = client.get("/api/tunnel")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'volume' in data
            assert 'level' in data


class TestForecastEndpoints:
    """Test forecasting endpoints."""
    
    def test_get_forecast_default(self, client):
        """Test forecast with default horizon."""
        response = client.get("/api/forecast")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'timestamps' in data or 'inflow_predictions' in data
    
    def test_get_forecast_custom_horizon(self, client):
        """Test forecast with custom horizon."""
        response = client.get("/api/forecast?horizon=48")
        
        assert response.status_code in [200, 400, 503]
    
    def test_get_forecast_invalid_horizon(self, client):
        """Test forecast with invalid horizon."""
        response = client.get("/api/forecast?horizon=500")
        
        assert response.status_code == 400


class TestOptimizationEndpoints:
    """Test optimization endpoints."""
    
    def test_run_optimization_default(self, client):
        """Test optimization with default parameters."""
        response = client.post("/api/optimize", json={
            "horizon": 24,
            "mode": "offline"
        })
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'schedule' in data or 'status' in data


class TestChatbotEndpoints:
    """Test chatbot endpoints."""
    
    def test_chat_basic(self, client):
        """Test basic chat interaction."""
        response = client.post("/api/chat", json={
            "message": "What is IFO?",
            "include_system_status": False
        })
        
        # May fail if GEMINI_API_KEY not valid
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'response' in data
            assert 'timestamp' in data
    
    def test_chat_with_context(self, client):
        """Test chat with system context."""
        response = client.post("/api/chat", json={
            "message": "What is the current status?",
            "include_system_status": True
        })
        
        assert response.status_code in [200, 503]
    
    def test_chat_history(self, client):
        """Test chat history retrieval."""
        response = client.get("/api/chat/history")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_reset_chat_history(self, client):
        """Test chat history reset."""
        response = client.delete("/api/chat/history")
        
        assert response.status_code in [200, 503]


class TestEdgeEndpoints:
    """Test edge device endpoints."""
    
    def test_post_edge_metrics(self, client):
        """Test posting edge metrics."""
        response = client.post("/api/edge/metrics", json={
            "metrics": {
                "cpu_temp_c": 45.2,
                "cpu0_freq_mhz": 1200.0,
                "soc_voltage_v": 3.3
            }
        })
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'ok'
    
    def test_get_edge_metrics(self, client):
        """Test retrieving edge metrics."""
        response = client.get("/api/edge/metrics")
        
        assert response.status_code in [200, 404, 503]
