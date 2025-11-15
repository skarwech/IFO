"""
Simulation model for wastewater tunnel dynamics and baseline control.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from .data_utils import VolumeCalculator
from .pump_models import EnhancedPumpStation

logger = logging.getLogger(__name__)


class TunnelSimulator:
    """
    Simulate tunnel water level and volume dynamics.
    
    Dynamics: V(t+1) = V(t) + (F1(t) - F2(t)) * Δt
    where F1 = inflow, F2 = pumped flow
    """
    
    def __init__(
        self,
        config: Dict,
        volume_calculator: VolumeCalculator,
        pump_station: EnhancedPumpStation
    ):
        """
        Initialize simulator.
        
        Args:
            config: Configuration dictionary
            volume_calculator: Volume-level converter
            pump_station: Pump station model
        """
        self.config = config
        self.vol_calc = volume_calculator
        self.pump_station = pump_station
        
        self.dt_hours = config['simulation']['timestep_hours']
        
        # Constraints
        self.l1_min = config['constraints']['level']['min_m']
        self.l1_max = config['constraints']['level']['max_m']
        self.l2_fixed = config['system']['tunnel'].get('l2_fixed', 30.0)
        
        logger.info("TunnelSimulator initialized")
    
    def step(
        self,
        current_volume_m3: float,
        inflow_m3h: float,
        pump_states: Dict[int, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Simulate one time step.
        
        Args:
            current_volume_m3: Current volume
            inflow_m3h: Inflow rate in m³/h
            pump_states: Pump configuration
            
        Returns:
            Dict with next state and metrics
        """
        # Current level
        current_level = self.vol_calc.get_level(current_volume_m3)
        
        # System head (L2 fixed, L1 varies)
        system_head_m = self.l2_fixed - current_level
        
        # Calculate pumped flow
        pump_result = self.pump_station.calculate_total_flow(
            pump_states,
            system_head_m
        )
        
        outflow_m3h = pump_result['total_flow_m3h']
        power_kw = pump_result['total_power_kw']
        
        # Update volume: V(t+1) = V(t) + (F1 - F2) * Δt
        net_flow_m3h = inflow_m3h - outflow_m3h
        delta_volume = net_flow_m3h * self.dt_hours
        next_volume = current_volume_m3 + delta_volume
        
        # Apply constraints
        next_volume = np.clip(
            next_volume,
            self.vol_calc.get_min_volume(),
            self.vol_calc.get_max_volume()
        )
        
        next_level = self.vol_calc.get_level(next_volume)
        
        return {
            'volume_m3': next_volume,
            'level_m': next_level,
            'inflow_m3h': inflow_m3h,
            'outflow_m3h': outflow_m3h,
            'power_kw': power_kw,
            'system_head_m': system_head_m,
            'active_pumps': pump_result['active_pumps'],
            'net_flow_m3h': net_flow_m3h
        }
    
    def simulate(
        self,
        initial_volume_m3: float,
        inflow_series_m3h: np.ndarray,
        pump_schedule: List[Dict[int, Dict[str, float]]],
        price_series: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Run full simulation.
        
        Args:
            initial_volume_m3: Starting volume
            inflow_series_m3h: Inflow forecast
            pump_schedule: Pump states for each timestep
            price_series: Electricity prices (optional)
            
        Returns:
            DataFrame with simulation results
        """
        n_steps = len(inflow_series_m3h)
        
        if len(pump_schedule) != n_steps:
            raise ValueError("Pump schedule length must match inflow series")
        
        results = []
        volume = initial_volume_m3
        
        for t in range(n_steps):
            state = self.step(
                volume,
                inflow_series_m3h[t],
                pump_schedule[t]
            )
            
            # Calculate cost
            if price_series is not None:
                cost = state['power_kw'] * price_series[t] * self.dt_hours
            else:
                cost = 0.0
            
            results.append({
                'timestep': t,
                'volume_m3': state['volume_m3'],
                'level_m': state['level_m'],
                'inflow_m3h': state['inflow_m3h'],
                'outflow_m3h': state['outflow_m3h'],
                'power_kw': state['power_kw'],
                'cost_eur': cost,
                'system_head_m': state['system_head_m'],
                'active_pumps': state['active_pumps']
            })
            
            volume = state['volume_m3']
        
        return pd.DataFrame(results)


class BaselineController:
    """
    Baseline pump control strategy (rule-based).
    Maintains level within range using simple logic.
    """
    
    def __init__(self, config: Dict, pump_station: EnhancedPumpStation):
        """
        Initialize baseline controller.
        
        Args:
            config: Configuration dictionary
            pump_station: Pump station model
        """
        self.config = config
        self.pump_station = pump_station
        
        # Control thresholds
        self.l1_min = config['constraints']['level']['min_m']
        self.l1_max = config['constraints']['level']['max_m']
        self.l1_target = (self.l1_min + self.l1_max) / 2
        
        # Frequency settings
        self.freq_min = config['constraints']['pump_frequency']['min_hz']
        self.freq_max = config['constraints']['pump_frequency']['max_hz']
        
        self.num_pumps = pump_station.get_num_pumps()
        
        logger.info("BaselineController initialized")
    
    def get_action(
        self,
        current_level_m: float,
        current_inflow_m3h: float
    ) -> Dict[int, Dict[str, float]]:
        """
        Determine pump states based on current conditions.
        
        Simple logic:
        - High level: activate more pumps
        - Low level: reduce pumps
        - Adjust frequency based on deviation from target
        
        Args:
            current_level_m: Current water level
            current_inflow_m3h: Current inflow rate
            
        Returns:
            Pump states dictionary
        """
        # Determine number of pumps needed based on level
        level_error = current_level_m - self.l1_target
        
        if level_error > 1.0:
            # High level: use more pumps
            n_active = min(6, self.num_pumps)
        elif level_error > 0.5:
            n_active = min(4, self.num_pumps)
        elif level_error > -0.5:
            n_active = min(3, self.num_pumps)
        else:
            # Low level: minimal pumping
            n_active = max(1, min(2, self.num_pumps))
        
        # Determine frequency based on level error
        if abs(level_error) > 1.0:
            frequency = self.freq_max
        elif abs(level_error) > 0.5:
            frequency = (self.freq_min + self.freq_max) / 2
        else:
            frequency = self.freq_min
        
        # Create pump states (activate largest pumps first for efficiency)
        pump_states = {}
        for i in range(self.num_pumps):
            if i < n_active:
                pump_states[i] = {'on': True, 'frequency_hz': frequency}
            else:
                pump_states[i] = {'on': False, 'frequency_hz': self.freq_min}
        
        return pump_states
    
    def simulate(
        self,
        simulator: TunnelSimulator,
        initial_volume_m3: float,
        inflow_series_m3h: np.ndarray,
        price_series: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Run baseline control simulation.
        
        Args:
            simulator: Tunnel simulator
            initial_volume_m3: Starting volume
            inflow_series_m3h: Inflow forecast
            price_series: Electricity prices
            
        Returns:
            Simulation results DataFrame
        """
        n_steps = len(inflow_series_m3h)
        
        # Generate pump schedule using baseline logic
        pump_schedule = []
        volume = initial_volume_m3
        
        for t in range(n_steps):
            level = simulator.vol_calc.get_level(volume)
            pump_states = self.get_action(level, inflow_series_m3h[t])
            pump_schedule.append(pump_states)
            
            # Simulate step to update volume for next iteration
            state = simulator.step(volume, inflow_series_m3h[t], pump_states)
            volume = state['volume_m3']
        
        # Run full simulation with generated schedule
        return simulator.simulate(
            initial_volume_m3,
            inflow_series_m3h,
            pump_schedule,
            price_series
        )


class EnergyCalculator:
    """Calculate energy metrics and costs."""
    
    @staticmethod
    def calculate_total_cost(
        power_series_kw: np.ndarray,
        price_series_eur_kwh: np.ndarray,
        dt_hours: float = 0.25
    ) -> float:
        """
        Calculate total electricity cost.
        
        Args:
            power_series_kw: Power consumption series
            price_series_eur_kwh: Electricity price series
            dt_hours: Timestep in hours
            
        Returns:
            Total cost in EUR
        """
        energy_kwh = power_series_kw * dt_hours
        cost = np.sum(energy_kwh * price_series_eur_kwh)
        return float(cost)
    
    @staticmethod
    def calculate_total_energy(
        power_series_kw: np.ndarray,
        dt_hours: float = 0.25
    ) -> float:
        """
        Calculate total energy consumption.
        
        Args:
            power_series_kw: Power consumption series
            dt_hours: Timestep in hours
            
        Returns:
            Total energy in kWh
        """
        energy_kwh = np.sum(power_series_kw * dt_hours)
        return float(energy_kwh)
    
    @staticmethod
    def calculate_savings(
        baseline_cost_eur: float,
        optimized_cost_eur: float
    ) -> Dict[str, float]:
        """
        Calculate cost savings.
        
        Args:
            baseline_cost_eur: Baseline cost
            optimized_cost_eur: Optimized cost
            
        Returns:
            Dictionary with savings metrics
        """
        absolute_savings = baseline_cost_eur - optimized_cost_eur
        relative_savings = (
            absolute_savings / baseline_cost_eur * 100
            if baseline_cost_eur > 0 else 0.0
        )
        
        return {
            'baseline_cost_eur': baseline_cost_eur,
            'optimized_cost_eur': optimized_cost_eur,
            'absolute_savings_eur': absolute_savings,
            'relative_savings_pct': relative_savings
        }
