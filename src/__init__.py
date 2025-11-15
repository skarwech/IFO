"""
Intelligent Pump Scheduler (IPS) for Blominmäki WWTP
Multi-agent AI system for wastewater pumping optimization with digitized pump curves.
"""

__version__ = "2.0.0"
__author__ = "IFO Team"

# Data utilities
from .data_utils import VolumeCalculator, DataLoader, create_sample_data

# Pump models (enhanced with digitized curves)
from .pump_models import (
    PumpModel,
    PumpStation,
    EnhancedPumpStation,
    PumpCharacteristics
)

# Enhanced pump models with digitized curves
try:
    from .enhanced_pump_models import (
        EnhancedPumpModel,
        DigitizedCurves,
        create_small_pump_curves,
        create_large_pump_curves
    )
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False

# Forecasting
from .forecasting import InflowForecaster

# Simulation
from .simulation import (
    TunnelSimulator,
    BaselineController,
    EnergyCalculator
)

# Optimization
from .optimization import MPCOptimizer

# Coordination
from .coordinator import (
    PumpCoordinator,
    PriceForecaster,
    StorageManager
)

# Dashboard
from .dashboard import (
    PumpDashboard,
    create_static_report
)

__all__ = [
    # Data
    'VolumeCalculator',
    'DataLoader',
    'create_sample_data',
    
    # Pumps
    'PumpModel',
    'PumpStation',
    'EnhancedPumpStation',
    'PumpCharacteristics',
    
    # Forecasting
    'InflowForecaster',
    
    # Simulation
    'TunnelSimulator',
    'BaselineController',
    'EnergyCalculator',
    
    # Optimization
    'MPCOptimizer',
    
    # Coordination
    'PumpCoordinator',
    'PriceForecaster',
    'StorageManager',
    
    # Dashboard
    'PumpDashboard',
    'create_static_report',
]

# Conditional exports for enhanced models
if ENHANCED_MODELS_AVAILABLE:
    __all__.extend([
        'EnhancedPumpModel',
        'DigitizedCurves',
        'create_small_pump_curves',
        'create_large_pump_curves'
    ])
