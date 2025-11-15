"""
Multi-agent coordinator for intelligent pump scheduling.
Integrates forecasting, storage management, and MPC optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .forecasting import InflowForecaster
from .optimization import MPCOptimizer
from .simulation import TunnelSimulator, EnergyCalculator
from .data_utils import VolumeCalculator
from .pump_models import EnhancedPumpStation

logger = logging.getLogger(__name__)


class PriceForecaster:
    """Agent for electricity price forecasting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.historical_prices = None
    
    def fit(self, price_series: pd.Series):
        """Learn price patterns from historical data."""
        self.historical_prices = price_series.copy()
        logger.info("Price forecaster fitted")
    
    def forecast(
        self,
        current_time: pd.Timestamp,
        horizon_hours: int = 24
    ) -> np.ndarray:
        """
        Forecast electricity prices.
        
        Simple approach: use historical patterns from same time of day.
        """
        if self.historical_prices is None:
            # Default constant price
            return np.ones(horizon_hours) * 0.10
        
        # Extract hour-of-day pattern
        prices_by_hour = self.historical_prices.groupby(
            self.historical_prices.index.hour
        ).mean()
        
        # Generate forecast
        forecast = []
        for h in range(horizon_hours):
            hour = (current_time.hour + h) % 24
            if hour in prices_by_hour.index:
                forecast.append(prices_by_hour[hour])
            else:
                forecast.append(prices_by_hour.mean())
        
        return np.array(forecast)


class StorageManager:
    """Agent for tunnel storage management."""
    
    def __init__(self, config: Dict, volume_calculator: VolumeCalculator):
        self.config = config
        self.vol_calc = volume_calculator
        
        self.l1_min = config['constraints']['level']['min_m']
        self.l1_max = config['constraints']['level']['max_m']
        self.l1_operational_max = config['constraints']['level']['operational_max_m']
    
    def get_storage_recommendation(
        self,
        current_level_m: float,
        inflow_forecast: np.ndarray,
        price_forecast: np.ndarray
    ) -> Dict[str, float]:
        """
        Recommend storage strategy based on forecasts.
        
        Returns target levels to optimize cost while maintaining safety.
        """
        # Identify low-price periods
        price_threshold = np.median(price_forecast)
        low_price_periods = price_forecast < price_threshold
        
        # Calculate available capacity
        current_volume = self.vol_calc.get_volume(current_level_m)
        max_volume = self.vol_calc.get_volume(self.l1_operational_max)
        available_capacity_m3 = max_volume - current_volume
        
        # Cumulative inflow forecast
        total_inflow = np.sum(inflow_forecast)
        
        # Storage strategy
        if current_level_m > self.l1_operational_max - 0.5:
            # Near capacity: must pump
            target_level = self.l1_operational_max - 1.0
            urgency = 'high'
        elif current_level_m < self.l1_min + 0.5:
            # Low level: minimize pumping
            target_level = self.l1_min + 1.0
            urgency = 'low'
        elif np.any(low_price_periods):
            # Low prices ahead: can store water
            target_level = min(
                current_level_m + 1.0,
                self.l1_operational_max
            )
            urgency = 'medium'
        else:
            # High prices: pump down
            target_level = self.l1_min + 1.5
            urgency = 'medium'
        
        return {
            'target_level_m': target_level,
            'current_level_m': current_level_m,
            'urgency': urgency,
            'available_capacity_m3': available_capacity_m3
        }


class PumpCoordinator:
    """
    Multi-agent coordinator for intelligent pump scheduling.
    Integrates all agents and orchestrates optimization.
    """
    
    def __init__(
        self,
        config: Dict,
        volume_calculator: VolumeCalculator,
        pump_station: EnhancedPumpStation
    ):
        """
        Initialize coordinator.
        
        Args:
            config: Configuration dictionary
            volume_calculator: Volume calculator
            pump_station: Pump station model
        """
        self.config = config
        self.vol_calc = volume_calculator
        self.pump_station = pump_station
        
        # Initialize agents
        logger.info("Initializing multi-agent system...")
        
        self.inflow_forecaster = InflowForecaster(config)
        self.price_forecaster = PriceForecaster(config)
        self.storage_manager = StorageManager(config, volume_calculator)
        self.mpc_optimizer = MPCOptimizer(config, volume_calculator, pump_station)
        
        # Simulation for validation
        self.simulator = TunnelSimulator(config, volume_calculator, pump_station)
        
        # State tracking
        self.current_state = None
        self.optimization_history = []
        
        logger.info("Multi-agent coordinator initialized")
    
    def fit(
        self,
        historical_data: pd.DataFrame,
        inflow_column: str = 'F1',
        price_column: str = 'price_normal'
    ):
        """
        Train all agents on historical data.
        
        Args:
            historical_data: Historical data with inflow and prices
            inflow_column: Column name for inflow data
            price_column: Column name for price data
        """
        logger.info("Training forecasting agents...")
        
        # Train inflow forecaster
        if inflow_column in historical_data.columns:
            inflow_series = historical_data[inflow_column]
            self.inflow_forecaster.fit(inflow_series.values)
            logger.info("Inflow forecaster trained")
        
        # Train price forecaster
        if price_column in historical_data.columns:
            price_series = historical_data[price_column]
            if 'timestamp' in historical_data.columns:
                price_series.index = historical_data['timestamp']
            self.price_forecaster.fit(price_series)
            logger.info("Price forecaster trained")
        
        logger.info("All agents trained successfully")
    
    def optimize_schedule(
        self,
        current_volume_m3: float,
        current_time: pd.Timestamp,
        horizon_hours: int = 24
    ) -> Dict:
        """
        Generate optimal pump schedule for forecast horizon.
        
        Args:
            current_volume_m3: Current tunnel volume
            current_time: Current timestamp
            horizon_hours: Optimization horizon
            
        Returns:
            Dictionary with optimized schedule and forecasts
        """
        logger.info(f"Optimizing schedule for {horizon_hours}h horizon...")
        
        # Get forecasts
        inflow_forecast = self.inflow_forecaster.forecast(
            current_volume_m3,
            horizon_hours
        )
        
        price_forecast = self.price_forecaster.forecast(
            current_time,
            horizon_hours
        )
        
        # Get storage recommendation
        current_level = self.vol_calc.get_level(current_volume_m3)
        storage_rec = self.storage_manager.get_storage_recommendation(
            current_level,
            inflow_forecast,
            price_forecast
        )
        
        # Run MPC optimization
        try:
            solution = self.mpc_optimizer.optimize(
                initial_volume_m3=current_volume_m3,
                inflow_forecast_m3h=inflow_forecast,
                price_forecast_eur_kwh=price_forecast,
                target_level_m=storage_rec['target_level_m']
            )
            
            if solution['status'] == 'optimal':
                logger.info(
                    f"Optimization successful: "
                    f"cost={solution['total_cost_eur']:.2f} EUR"
                )
            else:
                logger.warning(f"Optimization status: {solution['status']}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fallback to baseline
            solution = {
                'status': 'failed',
                'pump_schedule': None
            }
        
        return {
            'solution': solution,
            'inflow_forecast': inflow_forecast,
            'price_forecast': price_forecast,
            'storage_recommendation': storage_rec,
            'current_level_m': current_level,
            'current_volume_m3': current_volume_m3,
            'timestamp': current_time
        }
    
    def execute_step(
        self,
        optimization_result: Dict,
        step_index: int = 0
    ) -> Dict[int, Dict[str, float]]:
        """
        Extract pump states for current timestep from optimization result.
        
        Args:
            optimization_result: Result from optimize_schedule
            step_index: Which timestep to execute (default: first step)
            
        Returns:
            Pump states for current timestep
        """
        solution = optimization_result['solution']
        
        if solution['status'] == 'optimal' and solution['pump_schedule']:
            # Return first step of optimized schedule
            if step_index < len(solution['pump_schedule']):
                return solution['pump_schedule'][step_index]
        
        # Fallback: use baseline control
        logger.warning("Using fallback baseline control")
        from .simulation import BaselineController
        baseline = BaselineController(self.config, self.pump_station)
        
        return baseline.get_action(
            optimization_result['current_level_m'],
            optimization_result['inflow_forecast'][0]
        )
    
    def run_closed_loop(
        self,
        initial_volume_m3: float,
        start_time: pd.Timestamp,
        duration_hours: int,
        actual_inflow_m3h: np.ndarray,
        actual_prices_eur_kwh: np.ndarray,
        reoptimize_interval_hours: int = 6
    ) -> pd.DataFrame:
        """
        Run closed-loop MPC simulation with periodic reoptimization.
        
        Args:
            initial_volume_m3: Starting volume
            start_time: Start timestamp
            duration_hours: Total simulation duration
            actual_inflow_m3h: Actual inflow realization
            actual_prices_eur_kwh: Actual prices
            reoptimize_interval_hours: How often to reoptimize
            
        Returns:
            DataFrame with closed-loop results
        """
        logger.info(f"Running {duration_hours}h closed-loop simulation...")
        
        dt_hours = self.config['simulation']['timestep_hours']
        n_steps = int(duration_hours / dt_hours)
        reopt_steps = int(reoptimize_interval_hours / dt_hours)
        
        results = []
        volume = initial_volume_m3
        current_time = start_time
        current_schedule = None
        schedule_step = 0
        
        for t in range(n_steps):
            # Reoptimize if needed
            if t % reopt_steps == 0 or current_schedule is None:
                logger.info(f"Step {t}/{n_steps}: Reoptimizing...")
                opt_result = self.optimize_schedule(
                    volume,
                    current_time,
                    horizon_hours=24
                )
                current_schedule = opt_result
                schedule_step = 0
            
            # Get pump states for current step
            pump_states = self.execute_step(current_schedule, schedule_step)
            schedule_step += 1
            
            # Simulate step
            state = self.simulator.step(
                volume,
                actual_inflow_m3h[t],
                pump_states
            )
            
            # Calculate cost
            cost = state['power_kw'] * actual_prices_eur_kwh[t] * dt_hours
            
            results.append({
                'timestep': t,
                'timestamp': current_time,
                'volume_m3': state['volume_m3'],
                'level_m': state['level_m'],
                'inflow_m3h': state['inflow_m3h'],
                'outflow_m3h': state['outflow_m3h'],
                'power_kw': state['power_kw'],
                'cost_eur': cost,
                'price_eur_kwh': actual_prices_eur_kwh[t],
                'active_pumps': state['active_pumps']
            })
            
            # Update state
            volume = state['volume_m3']
            current_time += timedelta(hours=dt_hours)
        
        logger.info("Closed-loop simulation complete")
        return pd.DataFrame(results)
