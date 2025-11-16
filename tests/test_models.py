"""Tests for pump models and volume calculations."""
import pytest
import numpy as np
from src.model import PumpFleet, VolumeModel, DigitizedPump


class TestDigitizedPump:
    """Test digitized pump curves."""
    
    def test_initialization(self):
        """Test pump initialization."""
        pump = DigitizedPump()
        assert pump.nom_freq == 50.0
        assert len(pump.flow_points) > 0
        assert len(pump.head_points) > 0
    
    def test_affinity_laws_flow(self):
        """Test flow scaling with affinity laws."""
        pump = DigitizedPump()
        base_flow = pump.get_flow(50.0)
        scaled_flow = pump.get_flow(48.0)
        
        # Q ∝ f
        expected_ratio = 48.0 / 50.0
        actual_ratio = scaled_flow / base_flow
        
        assert abs(actual_ratio - expected_ratio) < 0.01
    
    def test_affinity_laws_power(self):
        """Test power scaling with affinity laws."""
        pump = DigitizedPump()
        base_power = pump.get_power(50.0)
        scaled_power = pump.get_power(48.0)
        
        # P ∝ f³
        expected_ratio = (48.0 / 50.0) ** 3
        actual_ratio = scaled_power / base_power
        
        assert abs(actual_ratio - expected_ratio) < 0.02
    
    def test_frequency_bounds(self):
        """Test frequency bounds enforcement."""
        pump = DigitizedPump()
        
        # Should handle frequencies within valid range
        flow_48 = pump.get_flow(48.0)
        flow_50 = pump.get_flow(50.0)
        
        assert flow_48 > 0
        assert flow_50 > 0
        assert flow_50 > flow_48


class TestPumpFleet:
    """Test pump fleet operations."""
    
    def test_initialization(self):
        """Test fleet initialization."""
        fleet = PumpFleet(small_count=4, large_count=0)
        assert len(fleet.small_pumps) == 4
        assert len(fleet.large_pumps) == 0
    
    def test_total_flow_calculation(self):
        """Test total flow from multiple pumps."""
        fleet = PumpFleet(small_count=4, large_count=0)
        frequencies = np.array([48.0, 48.0, 48.0, 48.0])
        
        total_flow = fleet.get_total_flow(frequencies)
        single_flow = fleet.small_pumps[0].get_flow(48.0)
        
        assert abs(total_flow - 4 * single_flow) < 0.1
    
    def test_total_power_calculation(self):
        """Test total power from multiple pumps."""
        fleet = PumpFleet(small_count=4, large_count=0)
        frequencies = np.array([48.0, 49.0, 50.0, 48.0])
        
        total_power = fleet.get_total_power(frequencies)
        
        # Should be sum of individual powers
        expected_power = sum(
            pump.get_power(freq) 
            for pump, freq in zip(fleet.small_pumps, frequencies)
        )
        
        assert abs(total_power - expected_power) < 0.1


class TestVolumeModel:
    """Test volume-level conversion."""
    
    def test_initialization(self, sample_volume_data):
        """Test volume model initialization."""
        model = VolumeModel(sample_volume_data)
        assert model.volume_table is not None
        assert len(model.volume_table) > 0
    
    def test_volume_to_level_conversion(self, sample_volume_data):
        """Test volume to level interpolation."""
        model = VolumeModel(sample_volume_data)
        
        # Test exact point
        level_1 = model.volume_to_level(1000.0)
        assert abs(level_1 - 1.0) < 0.1
        
        # Test interpolation
        level_2 = model.volume_to_level(1500.0)
        assert 1.0 < level_2 < 2.0
    
    def test_level_to_volume_conversion(self, sample_volume_data):
        """Test level to volume interpolation."""
        model = VolumeModel(sample_volume_data)
        
        # Test roundtrip
        original_volume = 5000.0
        level = model.volume_to_level(original_volume)
        recovered_volume = model.level_to_volume(level)
        
        assert abs(recovered_volume - original_volume) < 10.0
    
    def test_bounds_handling(self, sample_volume_data):
        """Test handling of out-of-bounds values."""
        model = VolumeModel(sample_volume_data)
        
        # Test below minimum
        level_low = model.volume_to_level(-100.0)
        assert level_low >= 0.0
        
        # Test above maximum
        level_high = model.volume_to_level(1e6)
        assert level_high >= 0.0
