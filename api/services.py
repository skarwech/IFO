"""
Business logic services for API endpoints.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import (
    ForecastAgent,
    PlannerAgent,
    ExecutorAgent,
    SupervisorAgent
)
from src.optimize import MPCOptimizer
from src.model import PumpFleet
from api.models import (
    PumpStatus,
    TunnelMetrics,
    SystemStatus,
    ForecastData,
    OptimizationResult,
    KPIMetrics,
    PumpMode
)


class IFOService:
    """Core IFO system service."""
    
    def __init__(self, config_path: str = "./config.yaml"):
        """Initialize IFO service with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize multi-agent system
        self.forecast_agent = None
        self.planner_agent = None
        self.executor_agent = None
        self.supervisor = None
        
        # Initialize optimizer
        self.optimizer = None
        
        # Initialize pump fleet for flow calculations
        self.pump_fleet = PumpFleet(small_count=4, large_count=0)  # 4 small pumps
        
        # Current state - use .get() with defaults to handle missing keys
        tunnel_config = self.config.get('tunnel', {})
        self.current_frequencies = np.array([48.0, 48.0, 48.0, 48.0])
        self.current_volume = tunnel_config.get('initial_volume', 5000.0)
        self.current_inflow = 0.0
        self.last_update = datetime.now()
        self.optimization_active = False
        self.last_optimization_time = None
        # Edge device last metrics/heartbeat
        self.last_edge_metrics: Dict[str, Any] = {}
        self.last_edge_heartbeat: Optional[datetime] = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # Default minimal config
            return {
                'timestep_minutes': 15,
                'tunnel': {
                    'initial_volume': 5000.0,
                    'max_capacity': 10000.0,
                    'min_level': 0.0,
                    'max_level': 5.0,
                },
                'optimization': {
                    'horizon': 96,
                    'objective': 'minimize_cost',
                },
                'forecasting': {
                    'model': 'lstm',
                    'lookback': 24,
                }
            }
    
    def initialize_agents(self):
        """Initialize multi-agent system."""
        try:
            self.forecast_agent = ForecastAgent(self.config)
            self.planner_agent = PlannerAgent(self.config)
            self.executor_agent = ExecutorAgent(self.config)
            self.supervisor = SupervisorAgent(self.config)
            
            # Initialize optimizer
            self.optimizer = MPCOptimizer(
                small_count=4,
                large_count=4,
                time_limit_s=self.config.get('solver_timeout', 10),
                mip_gap=self.config.get('solver_gap', 0.02)
            )
        except Exception as e:
            print(f"Warning: Could not initialize agents: {e}")
            # Initialize optimizer even if agents fail
            try:
                self.optimizer = MPCOptimizer(small_count=4, large_count=4)
            except Exception as opt_err:
                print(f"Warning: Could not initialize optimizer: {opt_err}")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        pumps = []
        total_power = 0.0
        
        for i, freq in enumerate(self.current_frequencies):
            # Calculate flow using pump fleet model
            flow = self.pump_fleet.small.flow_at_frequency(freq) if freq > 0 else 0.0
            # Convert from lps to m³/h
            flow_m3h = flow * 3.6
            
            # Calculate power (simplified cubic model)
            power = self.pump_fleet.small.power_at_frequency(freq) if freq > 0 else 0.0
            
            pumps.append(PumpStatus(
                id=i+1,
                frequency=float(freq),
                flow=float(flow_m3h),
                power=float(power),
                mode=PumpMode.AUTO,
                is_running=freq > 0
            ))
            total_power += power
        
        tunnel = TunnelMetrics(
            volume=float(self.current_volume),
            level=float(self.current_volume / 2000.0),  # Simplified
            inflow_rate=float(self.current_inflow),
            outflow_rate=float(sum(p.flow for p in pumps)),
            timestamp=self.last_update
        )
        
        return SystemStatus(
            pumps=pumps,
            tunnel=tunnel,
            total_power=float(total_power),
            optimization_active=self.optimization_active,
            last_optimization=self.last_optimization_time,
            system_mode="hybrid"
        )
    
    def get_forecast(self, horizon: int = 96) -> ForecastData:
        """Get inflow forecast."""
        try:
            # Use LSTM forecasting if available
            timestamps = [
                datetime.now() + timedelta(minutes=15*i)
                for i in range(horizon)
            ]
            
            # Placeholder forecast (replace with actual LSTM)
            inflow = np.random.uniform(50, 150, horizon).tolist()
            
            return ForecastData(
                timestamps=timestamps,
                inflow_predictions=inflow,
                model_type="LSTM"
            )
        except Exception as e:
            # Fallback to dummy data
            timestamps = [
                datetime.now() + timedelta(minutes=15*i)
                for i in range(horizon)
            ]
            return ForecastData(
                timestamps=timestamps,
                inflow_predictions=[100.0] * horizon,
                model_type="fallback"
            )
    
    def run_optimization(self, horizon: int = 96, mode: str = "hybrid") -> OptimizationResult:
        """Run optimization and return results."""
        start_time = datetime.now()
        
        try:
            # Get forecast
            forecast = self.get_forecast(horizon)
            
            # Initialize optimizer if not already done
            if self.optimizer is None:
                self.optimizer = MPCOptimizer(small_count=4, large_count=4)
            
            # Create price forecast (simplified - could be time-based)
            price_forecast = [0.07] * horizon  # EUR/kWh
            
            # Run optimization using MPCOptimizer
            result = self.optimizer.optimize(
                initial_volume_m3=self.current_volume,
                inflow_forecast_m3_per_15min=pd.Series(forecast.inflow_predictions),
                price_forecast_eur_kwh=pd.Series(price_forecast)
            )
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results from optimizer output
            schedule = result.get('schedule', [])
            volume_trajectory = result.get('volume_trajectory', [])
            kpis = result.get('kpis', {})
            status = result.get('status', 'Unknown')
            
            # Calculate power from schedule
            power = []
            if schedule:
                for timestep_freqs in schedule:
                    timestep_power = 0.0
                    for freq in timestep_freqs:
                        # Simplified cubic power model
                        timestep_power += 0.001 * (freq / 48.0) ** 3 * 100
                    power.append(timestep_power)
            
            total_energy = kpis.get('total_energy_kwh', 0.0)
            
            self.last_optimization_time = datetime.now()
            
            return OptimizationResult(
                schedule=schedule if schedule else [[48.0]*4 for _ in range(horizon)],
                predicted_power=power if power else [100.0]*horizon,
                predicted_volume=volume_trajectory if volume_trajectory else [self.current_volume]*horizon,
                total_energy=float(total_energy),
                cost_savings=kpis.get('savings_pct', 0.0),
                computation_time=computation_time,
                status=status
            )
        except Exception as e:
            print(f"Optimization error: {e}")
            # Return fallback result
            return OptimizationResult(
                schedule=[[48.0]*4 for _ in range(horizon)],
                predicted_power=[100.0]*horizon,
                predicted_volume=[self.current_volume]*horizon,
                total_energy=100.0 * horizon * 0.25,
                computation_time=(datetime.now() - start_time).total_seconds(),
                status="error"
            )
    
    def calculate_kpis(self, start: datetime, end: datetime) -> KPIMetrics:
        """Calculate KPIs for a time period."""
        # Placeholder implementation
        return KPIMetrics(
            total_energy_consumed=2500.0,
            average_power=400.0,
            peak_power=600.0,
            energy_cost=375.0,
            cost_savings_vs_baseline=15.5,
            volume_compliance=98.5,
            avg_tunnel_level=2.5,
            period_start=start,
            period_end=end
        )
    
    def update_pump_frequencies(self, frequencies: List[float]):
        """Update pump operating frequencies."""
        self.current_frequencies = np.array(frequencies)
        self.last_update = datetime.now()

    def update_edge_metrics(self, metrics: Dict[str, Any]):
        """Store last seen edge device metrics and heartbeat."""
        self.last_edge_metrics = metrics or {}
        self.last_edge_heartbeat = datetime.now()

    def get_edge_metrics(self) -> Dict[str, Any]:
        """Return last edge metrics with timestamp if available."""
        return {
            "metrics": self.last_edge_metrics,
            "heartbeat": self.last_edge_heartbeat.isoformat() if self.last_edge_heartbeat else None,
        }


# Singleton service instance
_ifo_service = None


def get_ifo_service() -> IFOService:
    """Get or create IFO service singleton."""
    global _ifo_service
    if _ifo_service is None:
        _ifo_service = IFOService()
    return _ifo_service
