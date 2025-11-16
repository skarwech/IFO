"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'timestep_minutes': 15,
        'horizon_steps': 96,
        'tunnel': {
            'initial_volume': 5000.0,
            'max_capacity': 10000.0,
            'min_level': 0.0,
            'max_level': 8.0,
        },
        'optimization': {
            'horizon': 96,
            'objective': 'minimize_cost',
            'solver_timeout': 10,
            'solver_gap': 0.02,
        },
        'forecasting': {
            'model': 'lstm',
            'lookback': 32,
            'epochs': 50,
        }
    }


@pytest.fixture
def sample_inflow_data():
    """Sample inflow data for testing."""
    timestamps = pd.date_range(start='2024-01-01', periods=200, freq='15min')
    inflow = 100 + 50 * np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 10
    inflow = np.maximum(inflow, 0)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'F1_m3h': inflow
    })


@pytest.fixture
def sample_volume_data():
    """Sample volume data for testing."""
    levels = np.linspace(0, 8, 20)
    volumes = 1000 * levels  # Linear approximation
    
    return pd.DataFrame({
        'Level_m': levels,
        'Volume_m3': volumes
    })


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create temporary config file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return str(config_path)


@pytest.fixture
def mock_opc_server():
    """Mock OPC UA server for testing."""
    class MockOPCServer:
        def __init__(self):
            self.values = {
                'Level_m': 2.5,
                'Volume_m3': 5000.0,
                'F1_m3h': 100.0,
                'F2_m3h': 100.0,
                'Pump1_Hz': 48.0,
                'Pump2_Hz': 48.0,
                'Pump3_Hz': 48.0,
                'Pump4_Hz': 48.0,
            }
        
        def read(self, node_id):
            return self.values.get(node_id, 0.0)
        
        def write(self, node_id, value):
            self.values[node_id] = value
    
    return MockOPCServer()
