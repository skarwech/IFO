"""
Enhanced pump models with digitized performance curves from manufacturer data.
Implements accurate interpolation for Q(H,f), P(Q,f), η(H,f) using affinity laws.
"""

import numpy as np
from scipy.interpolate import interp1d, griddata
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DigitizedCurves:
    """Digitized pump performance curve data points."""
    # H-Q curve at 50Hz (l/s, m)
    q_hq: np.ndarray
    h_hq: np.ndarray
    
    # P-Q curve at 50Hz (l/s, kW)
    q_pq: np.ndarray
    p_pq: np.ndarray
    
    # NPSHr-Q curve (l/s, m)
    q_npsh: np.ndarray
    npsh_r: np.ndarray
    
    # Efficiency-Head curves for multiple frequencies (Hz: {H: [m], eta: [%]})
    eta_h_curves: Dict[float, Dict[str, np.ndarray]]
    
    # Nominal operating point
    q_nom: float  # l/s
    h_nom: float  # m
    eta_nom: float  # fraction (0-1)
    p2_nom: float  # kW (shaft power)
    p1_nom: float  # kW (input power)


class EnhancedPumpModel:
    """
    Enhanced pump model with manufacturer curve data and affinity laws.
    
    Affinity Laws:
    - Q ∝ f  (flow proportional to frequency)
    - H ∝ f² (head proportional to frequency squared)
    - P ∝ f³ (power proportional to frequency cubed)
    """
    
    def __init__(self, name: str, curves: DigitizedCurves, rated_power_kw: float):
        """
        Initialize enhanced pump model.
        
        Args:
            name: Pump identifier
            curves: Digitized performance curves
            rated_power_kw: Maximum rated power
        """
        self.name = name
        self.curves = curves
        self.rated_power_kw = rated_power_kw
        self.f_nominal = 50.0  # Nominal frequency
        
        # Create interpolation functions
        self._setup_interpolators()
        
        logger.info(f"Enhanced pump '{name}' initialized with digitized curves")
    
    def _setup_interpolators(self):
        """Setup interpolation functions for all curves."""
        # H-Q curve at 50Hz (cubic for smoothness)
        self.interp_h_q = interp1d(
            self.curves.q_hq,
            self.curves.h_hq,
            kind='cubic',
            bounds_error=False,
            fill_value=(self.curves.h_hq[0], self.curves.h_hq[-1])
        )
        
        # P-Q curve at 50Hz
        self.interp_p_q = interp1d(
            self.curves.q_pq,
            self.curves.p_pq,
            kind='cubic',
            bounds_error=False,
            fill_value=(self.curves.p_pq[0], self.curves.p_pq[-1])
        )
        
        # NPSHr-Q curve
        self.interp_npsh_q = interp1d(
            self.curves.q_npsh,
            self.curves.npsh_r,
            kind='linear',
            bounds_error=False,
            fill_value=(self.curves.npsh_r[0], self.curves.npsh_r[-1])
        )
        
        # Efficiency-Head interpolators for each frequency
        self.interp_eta_h = {}
        for freq, data in self.curves.eta_h_curves.items():
            self.interp_eta_h[freq] = interp1d(
                data['H'],
                data['eta'],
                kind='cubic',
                bounds_error=False,
                fill_value=(min(data['eta']), min(data['eta']))
            )
    
    def get_head_at_flow(self, flow_lps: float, frequency_hz: float = 50.0) -> float:
        """
        Calculate pump head for given flow and frequency.
        
        Args:
            flow_lps: Flow rate in l/s
            frequency_hz: Operating frequency in Hz
            
        Returns:
            Head in meters
        """
        # Apply affinity law: Q ∝ f
        f_ratio = frequency_hz / self.f_nominal
        flow_50hz = flow_lps / f_ratio
        
        # Get head at 50Hz from curve
        head_50hz = float(self.interp_h_q(flow_50hz))
        
        # Apply affinity law: H ∝ f²
        head = head_50hz * (f_ratio ** 2)
        
        return max(0.0, head)
    
    def get_flow_at_head(self, head_m: float, frequency_hz: float = 50.0) -> float:
        """
        Calculate flow for given head and frequency (inverse H-Q).
        
        Args:
            head_m: Required head in meters
            frequency_hz: Operating frequency in Hz
            
        Returns:
            Flow rate in l/s
        """
        # Apply affinity law to get equivalent 50Hz head
        f_ratio = frequency_hz / self.f_nominal
        head_50hz = head_m / (f_ratio ** 2)
        
        # Invert H-Q curve numerically
        q_samples = np.linspace(
            self.curves.q_hq[0],
            self.curves.q_hq[-1] * 1.2,
            500
        )
        h_samples = self.interp_h_q(q_samples)
        
        # Find flow that gives closest head match
        idx = np.argmin(np.abs(h_samples - head_50hz))
        flow_50hz = q_samples[idx]
        
        # Apply affinity law: Q ∝ f
        flow = flow_50hz * f_ratio
        
        return max(0.0, flow)
    
    def get_power(self, flow_lps: float, frequency_hz: float = 50.0) -> float:
        """
        Calculate power consumption for given flow and frequency.
        
        Args:
            flow_lps: Flow rate in l/s
            frequency_hz: Operating frequency in Hz
            
        Returns:
            Power in kW
        """
        # Apply affinity law to get 50Hz equivalent flow
        f_ratio = frequency_hz / self.f_nominal
        flow_50hz = flow_lps / f_ratio
        
        # Get power at 50Hz from curve
        power_50hz = float(self.interp_p_q(flow_50hz))
        
        # Apply affinity law: P ∝ f³
        power = power_50hz * (f_ratio ** 3)
        
        # Limit to rated power
        power = min(power, self.rated_power_kw)
        
        return max(0.0, power)
    
    def get_efficiency(self, head_m: float, frequency_hz: float = 50.0) -> float:
        """
        Calculate efficiency for given head and frequency.
        
        Args:
            head_m: Operating head in meters
            frequency_hz: Operating frequency in Hz
            
        Returns:
            Efficiency as fraction (0-1)
        """
        # Apply affinity law to get 50Hz equivalent head
        f_ratio = frequency_hz / self.f_nominal
        head_50hz = head_m / (f_ratio ** 2)
        
        # Find closest available frequency in efficiency curves
        available_freqs = sorted(self.interp_eta_h.keys())
        closest_freq = min(available_freqs, key=lambda x: abs(x - frequency_hz))
        
        # Get efficiency from curve (in %)
        eta_pct = float(self.interp_eta_h[closest_freq](head_50hz))
        
        # Apply small penalty for non-exact frequency match
        freq_diff = abs(frequency_hz - closest_freq)
        penalty_factor = 1.0 - 0.005 * freq_diff  # 0.5% per Hz difference
        
        eta = (eta_pct / 100.0) * penalty_factor
        
        # Bound efficiency to reasonable range
        return np.clip(eta, 0.3, 0.95)
    
    def get_npsh_required(self, flow_lps: float, frequency_hz: float = 50.0) -> float:
        """
        Calculate NPSHr for given flow and frequency.
        
        Args:
            flow_lps: Flow rate in l/s
            frequency_hz: Operating frequency in Hz
            
        Returns:
            NPSHr in meters
        """
        # Apply affinity law
        f_ratio = frequency_hz / self.f_nominal
        flow_50hz = flow_lps / f_ratio
        
        # Get NPSHr at 50Hz
        npsh_50hz = float(self.interp_npsh_q(flow_50hz))
        
        # NPSHr scales like head: NPSHr ∝ f²
        npsh = npsh_50hz * (f_ratio ** 2)
        
        return max(1.0, npsh)
    
    def get_operating_point(
        self,
        system_head_m: float,
        frequency_hz: float = 50.0
    ) -> Dict[str, float]:
        """
        Calculate complete operating point where pump curve intersects system head.
        
        Args:
            system_head_m: System head requirement
            frequency_hz: Operating frequency
            
        Returns:
            Dictionary with all operating parameters
        """
        # Get flow at intersection
        flow_lps = self.get_flow_at_head(system_head_m, frequency_hz)
        
        # Calculate all parameters at this flow
        pump_head = self.get_head_at_flow(flow_lps, frequency_hz)
        power = self.get_power(flow_lps, frequency_hz)
        efficiency = self.get_efficiency(system_head_m, frequency_hz)
        npsh_req = self.get_npsh_required(flow_lps, frequency_hz)
        
        return {
            'flow_lps': flow_lps,
            'flow_m3h': flow_lps * 3.6,
            'head_m': pump_head,
            'system_head_m': system_head_m,
            'power_kw': power,
            'efficiency': efficiency,
            'npsh_req_m': npsh_req,
            'frequency_hz': frequency_hz,
            'head_margin_m': pump_head - system_head_m
        }


def create_small_pump_curves() -> DigitizedCurves:
    """
    Create digitized curves for small pumps (S3.120.500.2500.6.74M.H.Z).
    Data from Grundfos performance curves dated 16.04.2018.
    """
    return DigitizedCurves(
        # H vs Q at 50Hz (digitized from curve)
        q_hq=np.array([0, 240, 320, 400, 480, 560, 640, 720, 800]),
        h_hq=np.array([48, 48, 46, 43, 38, 32, 25, 16, 5]),
        
        # P vs Q at 50Hz
        q_pq=np.array([240, 400, 560, 720, 800]),
        p_pq=np.array([225, 200, 175, 180, 190]),
        
        # NPSHr vs Q
        q_npsh=np.array([240, 400, 560, 720, 800]),
        npsh_r=np.array([2, 4, 6, 8, 10]),
        
        # System efficiency vs Head for different frequencies
        eta_h_curves={
            50.0: {
                'H': np.array([35, 30, 25, 20, 15, 10, 5, 0]),
                'eta': np.array([76, 81, 82, 80, 76, 70, 60, 0])
            },
            47.5: {
                'H': np.array([32, 27, 22, 18, 14, 9, 4, 0]),
                'eta': np.array([74, 79, 80, 78, 74, 68, 58, 0])
            },
            45.0: {
                'H': np.array([29, 24, 20, 16, 12, 8, 4, 0]),
                'eta': np.array([70, 75, 76, 74, 70, 64, 54, 0])
            },
            40.0: {
                'H': np.array([23, 19, 16, 13, 10, 6, 3, 0]),
                'eta': np.array([62, 67, 68, 66, 62, 56, 46, 0])
            },
            35.0: {
                'H': np.array([17, 14, 12, 9, 7, 4, 2, 0]),
                'eta': np.array([54, 59, 60, 58, 54, 48, 38, 0])
            },
            30.0: {
                'H': np.array([13, 10, 8, 6, 5, 3, 1, 0]),
                'eta': np.array([46, 51, 52, 50, 46, 40, 30, 0])
            },
            25.0: {
                'H': np.array([9, 7, 6, 4, 3, 2, 1, 0]),
                'eta': np.array([38, 43, 44, 42, 38, 32, 22, 0])
            }
        },
        
        # Nominal point (from datasheet)
        q_nom=464.0,  # l/s
        h_nom=31.5,   # m
        eta_nom=0.816,  # 81.6%
        p2_nom=175.6,  # kW
        p1_nom=188.7   # kW
    )


def create_large_pump_curves() -> DigitizedCurves:
    """
    Create digitized curves for large pumps (S3.145.500.4000.8.78L.H.Z).
    Data from Grundfos performance curves dated 16.04.2018.
    """
    return DigitizedCurves(
        # H vs Q at 50Hz (digitized from curve)
        q_hq=np.array([0, 400, 600, 800, 1000, 1200, 1400, 1600]),
        h_hq=np.array([56, 56, 50, 42, 32, 22, 10, 0]),
        
        # P vs Q at 50Hz
        q_pq=np.array([400, 800, 1200, 1600]),
        p_pq=np.array([400, 380, 350, 300]),
        
        # NPSHr vs Q
        q_npsh=np.array([400, 800, 1200, 1600]),
        npsh_r=np.array([2, 4, 7, 8]),
        
        # System efficiency vs Head for different frequencies
        eta_h_curves={
            50.0: {
                'H': np.array([40, 35, 30, 25, 20, 15, 10, 5, 0]),
                'eta': np.array([70, 85, 84, 82, 78, 72, 64, 50, 0])
            },
            47.5: {
                'H': np.array([38, 32, 27, 22, 18, 14, 9, 4, 0]),
                'eta': np.array([68, 83, 82, 80, 76, 70, 62, 48, 0])
            },
            45.0: {
                'H': np.array([35, 29, 24, 20, 16, 12, 8, 4, 0]),
                'eta': np.array([64, 79, 78, 76, 72, 66, 58, 44, 0])
            },
            40.0: {
                'H': np.array([30, 23, 19, 16, 13, 10, 6, 3, 0]),
                'eta': np.array([56, 71, 70, 68, 64, 58, 50, 36, 0])
            },
            35.0: {
                'H': np.array([24, 17, 14, 12, 9, 7, 4, 2, 0]),
                'eta': np.array([48, 63, 62, 60, 56, 50, 42, 28, 0])
            },
            30.0: {
                'H': np.array([18, 13, 10, 8, 6, 5, 3, 1, 0]),
                'eta': np.array([40, 55, 54, 52, 48, 42, 34, 20, 0])
            },
            25.0: {
                'H': np.array([13, 9, 7, 6, 4, 3, 2, 1, 0]),
                'eta': np.array([32, 47, 46, 44, 40, 34, 26, 12, 0])
            }
        },
        
        # Nominal point (from datasheet)
        q_nom=925.0,  # l/s
        h_nom=31.5,   # m
        eta_nom=0.848,  # 84.8%
        p2_nom=336.6,  # kW
        p1_nom=358.1   # kW
    )


class EnhancedPumpStation:
    """
    Enhanced pump station with 4 small and 4 large pumps using digitized curves.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize enhanced pump station.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.small_pumps = []
        self.large_pumps = []
        self.all_pumps = []
        
        # Create small pumps (1.1-1.4)
        small_curves = create_small_pump_curves()
        small_config = config['system']['pumps']['small']
        
        for label in small_config['labels']:
            pump = EnhancedPumpModel(
                name=f"Small_{label}",
                curves=small_curves,
                rated_power_kw=small_config['rated_power_kw']
            )
            self.small_pumps.append(pump)
            self.all_pumps.append(pump)
        
        # Create large pumps (2.1-2.4)
        large_curves = create_large_pump_curves()
        large_config = config['system']['pumps']['large']
        
        for label in large_config['labels']:
            pump = EnhancedPumpModel(
                name=f"Large_{label}",
                curves=large_curves,
                rated_power_kw=large_config['rated_power_kw']
            )
            self.large_pumps.append(pump)
            self.all_pumps.append(pump)
        
        logger.info(
            f"Enhanced pump station initialized: "
            f"{len(self.small_pumps)} small + {len(self.large_pumps)} large pumps"
        )
    
    def calculate_total_flow(
        self,
        pump_states: Dict[int, Dict[str, float]],
        system_head_m: float
    ) -> Dict[str, float]:
        """
        Calculate total flow and power for given pump configuration.
        
        Args:
            pump_states: {pump_index: {'on': bool, 'frequency_hz': float}}
            system_head_m: System head requirement (H = 30 - L1)
            
        Returns:
            Dictionary with total flow, power, and pump details
        """
        total_flow_lps = 0.0
        total_power_kw = 0.0
        active_pumps = 0
        pump_details = []
        
        for pump_idx, state in pump_states.items():
            if state['on'] and pump_idx < len(self.all_pumps):
                pump = self.all_pumps[pump_idx]
                freq = state['frequency_hz']
                
                # Get operating point
                op = pump.get_operating_point(system_head_m, freq)
                
                total_flow_lps += op['flow_lps']
                total_power_kw += op['power_kw']
                active_pumps += 1
                
                pump_details.append({
                    'pump_id': pump_idx,
                    'pump_name': pump.name,
                    'frequency_hz': freq,
                    'flow_lps': op['flow_lps'],
                    'flow_m3h': op['flow_m3h'],
                    'power_kw': op['power_kw'],
                    'efficiency': op['efficiency'],
                    'head_m': op['head_m']
                })
        
        return {
            'total_flow_lps': total_flow_lps,
            'total_flow_ls': total_flow_lps,  # Backward compatibility
            'total_flow_m3h': total_flow_lps * 3.6,
            'total_power_kw': total_power_kw,
            'active_pumps': active_pumps,
            'system_head_m': system_head_m,
            'pump_details': pump_details,
            'avg_efficiency': (
                np.mean([p['efficiency'] for p in pump_details])
                if pump_details else 0.0
            )
        }
    
    def get_num_pumps(self) -> int:
        """Get total number of pumps."""
        return len(self.all_pumps)
    
    def get_pump(self, index: int) -> EnhancedPumpModel:
        """Get pump by index."""
        if 0 <= index < len(self.all_pumps):
            return self.all_pumps[index]
        raise IndexError(f"Pump index {index} out of range")
