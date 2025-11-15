"""
Pump models with backward compatibility wrapper for enhanced models.
Uses digitized performance curves when available, falls back to legacy models.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import enhanced models
try:
    from .enhanced_pump_models import (
        EnhancedPumpStation as _EnhancedStation,
        EnhancedPumpModel as _EnhancedModel,
        create_small_pump_curves,
        create_large_pump_curves
    )
    ENHANCED_AVAILABLE = True
    logger.info("Enhanced pump models with digitized curves loaded")
except ImportError as e:
    ENHANCED_AVAILABLE = False
    logger.warning(f"Enhanced models not available: {e}")


@dataclass
class PumpCharacteristics:
    """Pump characteristics at nominal operating point."""
    name: str
    rated_power_kw: float
    nominal_rpm: float
    nominal_hz: float
    impeller_mm: float
    q_nominal_ls: float
    h_nominal_m: float
    eta_nominal: float
    p2_nominal_kw: float
    p1_nominal_kw: float
    curve_a: Optional[float] = None
    curve_b: Optional[float] = None
    curve_c: Optional[float] = None


class PumpModel:
    """
    Legacy pump model using simplified affinity laws.
    Used as fallback when enhanced models are not available.
    """
    
    def __init__(self, characteristics: PumpCharacteristics):
        self.char = characteristics
        self._setup_curves()
    
    def _setup_curves(self):
        """Setup simplified pump curves."""
        if all([self.char.curve_a, self.char.curve_b, self.char.curve_c]):
            self.has_curve = True
        else:
            # Approximate quadratic curve from nominal point
            q_nom = self.char.q_nominal_ls
            h_nom = self.char.h_nominal_m
            h_shutoff = h_nom * 1.2
            q_max = q_nom * 1.5
            
            self.char.curve_a = h_shutoff
            self.char.curve_b = -(h_shutoff - h_nom) / q_nom
            self.char.curve_c = -(h_shutoff - h_nom) / (q_nom ** 2)
            self.has_curve = True
    
    def calculate_head(self, flow_ls: float, frequency_hz: float = 50.0) -> float:
        """Calculate head at given flow and frequency."""
        f_ratio = frequency_hz / self.char.nominal_hz
        flow_nominal = flow_ls / f_ratio
        
        # H = a + b*Q + c*Q²
        head_nominal = (
            self.char.curve_a +
            self.char.curve_b * flow_nominal +
            self.char.curve_c * (flow_nominal ** 2)
        )
        
        head = head_nominal * (f_ratio ** 2)
        return max(0.0, head)
    
    def calculate_flow(self, head_m: float, frequency_hz: float = 50.0) -> float:
        """Calculate flow at given head (inverse of head curve)."""
        f_ratio = frequency_hz / self.char.nominal_hz
        head_nominal = head_m / (f_ratio ** 2)
        
        # Solve quadratic: c*Q² + b*Q + (a - H) = 0
        a_coef = self.char.curve_c
        b_coef = self.char.curve_b
        c_coef = self.char.curve_a - head_nominal
        
        discriminant = b_coef**2 - 4*a_coef*c_coef
        if discriminant < 0:
            return 0.0
        
        q1 = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
        q2 = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)
        
        flow_nominal = max(0, min(q1, q2, key=lambda x: abs(x - self.char.q_nominal_ls)))
        flow = flow_nominal * f_ratio
        
        return max(0.0, flow)
    
    def calculate_power(
        self,
        flow_ls: float,
        head_m: float,
        frequency_hz: float = 50.0
    ) -> Tuple[float, float]:
        """Calculate efficiency and power."""
        # Simplified efficiency curve (peaks at nominal point)
        q_ratio = flow_ls / self.char.q_nominal_ls if self.char.q_nominal_ls > 0 else 0
        eta = self.char.eta_nominal * (1 - 0.3 * ((q_ratio - 1.0) ** 2))
        eta = np.clip(eta, 0.3, 0.95)
        
        # Power from hydraulic formula
        power_hydraulic = (flow_ls / 1000.0) * head_m * 9.81  # kW
        power = power_hydraulic / eta if eta > 0 else 0
        
        # Limit to rated power
        power = min(power, self.char.rated_power_kw)
        
        return eta, max(0.0, power)


class PumpStation:
    """
    Legacy pump station model.
    Used as fallback when enhanced models are not available.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.small_pumps = []
        self.large_pumps = []
        self.all_pumps = []
        
        # Create small pumps
        small_config = config['system']['pumps']['small']
        small_char = PumpCharacteristics(
            name="Small",
            rated_power_kw=small_config['rated_power_kw'],
            nominal_rpm=991,
            nominal_hz=50.0,
            impeller_mm=534,
            q_nominal_ls=464,
            h_nominal_m=31.5,
            eta_nominal=0.816,
            p2_nominal_kw=175.6,
            p1_nominal_kw=188.7
        )
        
        for _ in range(len(small_config['labels'])):
            pump = PumpModel(small_char)
            self.small_pumps.append(pump)
            self.all_pumps.append(pump)
        
        # Create large pumps
        large_config = config['system']['pumps']['large']
        large_char = PumpCharacteristics(
            name="Large",
            rated_power_kw=large_config['rated_power_kw'],
            nominal_rpm=743,
            nominal_hz=50.0,
            impeller_mm=749,
            q_nominal_ls=925,
            h_nominal_m=31.5,
            eta_nominal=0.848,
            p2_nominal_kw=336.6,
            p1_nominal_kw=358.1
        )
        
        for _ in range(len(large_config['labels'])):
            pump = PumpModel(large_char)
            self.large_pumps.append(pump)
            self.all_pumps.append(pump)
        
        logger.info(f"Legacy pump station: {len(self.all_pumps)} pumps")
    
    def calculate_total_flow(
        self,
        pump_states: Dict[int, Dict[str, float]],
        system_head_m: float
    ) -> Dict[str, float]:
        """Calculate total flow and power."""
        total_flow_ls = 0.0
        total_power_kw = 0.0
        active_pumps = 0
        
        for i, state in pump_states.items():
            if state['on'] and i < len(self.all_pumps):
                pump = self.all_pumps[i]
                freq = state['frequency_hz']
                
                flow = pump.calculate_flow(system_head_m, freq)
                _, power = pump.calculate_power(flow, system_head_m, freq)
                
                total_flow_ls += flow
                total_power_kw += power
                active_pumps += 1
        
        return {
            'total_flow_ls': total_flow_ls,
            'total_flow_lps': total_flow_ls,
            'total_flow_m3h': total_flow_ls * 3.6,
            'total_power_kw': total_power_kw,
            'active_pumps': active_pumps,
            'system_head_m': system_head_m
        }
    
    def get_num_pumps(self) -> int:
        return len(self.all_pumps)
    
    def get_pump(self, index: int):
        return self.all_pumps[index]


class EnhancedPumpStation:
    """
    Wrapper that uses enhanced models when available, falls back to legacy.
    This is the recommended class to use in applications.
    """
    
    def __init__(self, config: Dict):
        """Initialize pump station with best available model."""
        self.config = config
        
        if ENHANCED_AVAILABLE:
            self._station = _EnhancedStation(config)
            self._use_enhanced = True
            logger.info("Using enhanced pump models with digitized curves")
        else:
            self._station = PumpStation(config)
            self._use_enhanced = False
            logger.info("Using legacy pump models")
    
    def calculate_total_flow(
        self,
        pump_states: Dict[int, Dict[str, float]],
        system_head_m: float
    ) -> Dict[str, float]:
        """Calculate total flow and power for pump configuration."""
        return self._station.calculate_total_flow(pump_states, system_head_m)
    
    def get_num_pumps(self) -> int:
        """Get total number of pumps."""
        return self._station.get_num_pumps()
    
    def get_pump(self, index: int):
        """Get pump by index."""
        return self._station.get_pump(index)
    
    @property
    def using_enhanced_models(self) -> bool:
        """Check if using enhanced digitized curve models."""
        return self._use_enhanced
