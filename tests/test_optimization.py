"""Tests for MPC optimization."""
import pytest
import numpy as np
from src.optimize import MPCOptimizer


class TestMPCOptimizer:
    """Test MPC optimization engine."""
    
    def test_initialization(self, sample_config):
        """Test optimizer initialization."""
        optimizer = MPCOptimizer(sample_config)
        assert optimizer.horizon == 96
        assert optimizer.timestep == 0.25
    
    def test_optimization_feasibility(self, sample_config, sample_inflow_data):
        """Test that optimization produces feasible solutions."""
        optimizer = MPCOptimizer(sample_config)
        
        # Run optimization
        initial_volume = 5000.0
        inflow_forecast = np.ones(96) * 100.0  # Constant inflow
        
        result = optimizer.optimize(
            initial_volume=initial_volume,
            inflow_forecast=inflow_forecast,
            max_iterations=1
        )
        
        # Check solution exists
        assert result is not None
        assert 'frequencies' in result
        assert 'volumes' in result
        assert 'status' in result
    
    def test_volume_constraints(self, sample_config):
        """Test volume constraint compliance."""
        optimizer = MPCOptimizer(sample_config)
        
        initial_volume = 5000.0
        inflow_forecast = np.ones(96) * 150.0
        
        result = optimizer.optimize(
            initial_volume=initial_volume,
            inflow_forecast=inflow_forecast,
            max_iterations=1
        )
        
        if result and 'volumes' in result:
            volumes = result['volumes']
            
            # Check all volumes within bounds
            max_capacity = sample_config['tunnel']['max_capacity']
            assert all(0 <= v <= max_capacity for v in volumes)
    
    def test_discrete_frequencies(self, sample_config):
        """Test that frequencies are discrete."""
        optimizer = MPCOptimizer(sample_config)
        
        initial_volume = 5000.0
        inflow_forecast = np.ones(96) * 100.0
        
        result = optimizer.optimize(
            initial_volume=initial_volume,
            inflow_forecast=inflow_forecast,
            max_iterations=1
        )
        
        if result and 'frequencies' in result:
            frequencies = np.array(result['frequencies'])
            
            # Check frequencies are in valid set
            valid_freqs = {0.0, 48.0, 49.0, 50.0}
            for freq_set in frequencies:
                for freq in freq_set:
                    assert freq in valid_freqs or abs(freq) < 1e-6
    
    def test_energy_calculation(self, sample_config):
        """Test energy cost calculation."""
        optimizer = MPCOptimizer(sample_config)
        
        # Known power profile
        power_profile = np.ones(96) * 400.0  # 400 kW constant
        timestep = 0.25  # 15 minutes
        
        energy = optimizer.calculate_energy(power_profile, timestep)
        
        # Expected: 400 kW * 0.25 h * 96 steps = 9600 kWh
        expected_energy = 400.0 * 0.25 * 96
        
        assert abs(energy - expected_energy) < 1.0
    
    def test_solver_timeout_handling(self, sample_config):
        """Test graceful handling of solver timeout."""
        # Set very short timeout
        config = sample_config.copy()
        config['optimization']['solver_timeout'] = 0.001
        
        optimizer = MPCOptimizer(config)
        
        initial_volume = 5000.0
        inflow_forecast = np.ones(96) * 100.0
        
        # Should not crash, may return suboptimal or None
        result = optimizer.optimize(
            initial_volume=initial_volume,
            inflow_forecast=inflow_forecast,
            max_iterations=1
        )
        
        # Just verify it doesn't crash
        assert result is None or isinstance(result, dict)
