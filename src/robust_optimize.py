"""Enhanced optimizer with fallback strategies and robust error handling."""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Optimization result status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"
    FALLBACK = "fallback"


class GreedyFallback:
    """Greedy heuristic fallback when MILP solver fails."""
    
    def __init__(self, pump_fleet, volume_model):
        self.pump_fleet = pump_fleet
        self.volume_model = volume_model
        self.freq_choices = [48.0, 49.0, 50.0]
    
    def solve(
        self,
        initial_volume: float,
        inflow_forecast: np.ndarray,
        horizon: int,
        timestep: float = 0.25
    ) -> Dict[str, Any]:
        """
        Greedy heuristic: maintain volume near midpoint by adjusting pump frequencies.
        
        Strategy:
        - If volume rising: increase pump frequency
        - If volume falling: decrease pump frequency
        - Always keep at least one pump running
        """
        logger.warning("Using greedy fallback optimizer")
        
        target_volume = (self.volume_model.level_to_volume(0.0) + 
                        self.volume_model.level_to_volume(8.0)) / 2
        
        volumes = [initial_volume]
        frequencies_schedule = []
        
        for t in range(horizon):
            current_volume = volumes[-1]
            inflow = inflow_forecast[t] if t < len(inflow_forecast) else 100.0
            
            # Decide frequency based on volume deviation
            volume_error = current_volume - target_volume
            
            if volume_error > 2000:  # Too much water
                freq = 50.0  # Max pumping
            elif volume_error > 1000:
                freq = 49.0
            elif volume_error < -1000:
                freq = 48.0  # Min pumping
            else:
                freq = 49.0  # Nominal
            
            # All pumps at same frequency (simple strategy)
            pump_count = len(self.pump_fleet.small_pumps)
            frequencies = np.full(pump_count, freq)
            frequencies_schedule.append(frequencies.tolist())
            
            # Calculate outflow
            outflow = self.pump_fleet.get_total_flow(frequencies)
            
            # Update volume
            net_flow = inflow - outflow
            next_volume = current_volume + net_flow * timestep
            
            # Clamp to bounds
            min_vol = self.volume_model.level_to_volume(0.0)
            max_vol = self.volume_model.level_to_volume(8.0)
            next_volume = np.clip(next_volume, min_vol, max_vol)
            
            volumes.append(next_volume)
        
        # Calculate power
        power_profile = []
        for freqs in frequencies_schedule:
            power = self.pump_fleet.get_total_power(np.array(freqs))
            power_profile.append(power)
        
        return {
            'frequencies': frequencies_schedule,
            'volumes': volumes,
            'power': power_profile,
            'status': OptimizationStatus.FALLBACK.value,
            'total_energy': sum(power_profile) * timestep,
            'message': 'Greedy heuristic solution (MILP solver failed)'
        }


class RobustOptimizer:
    """Wrapper for MPCOptimizer with fallback and retry logic."""
    
    def __init__(self, mpc_optimizer, max_retries: int = 2, enable_fallback: bool = True):
        """
        Initialize robust optimizer.
        
        Args:
            mpc_optimizer: Base MPCOptimizer instance
            max_retries: Number of solver retries with relaxed parameters
            enable_fallback: Whether to use greedy fallback on total failure
        """
        self.mpc_optimizer = mpc_optimizer
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        
        # Create fallback solver
        if hasattr(mpc_optimizer, 'fleet') and hasattr(mpc_optimizer, 'volume_model'):
            self.fallback = GreedyFallback(
                mpc_optimizer.fleet,
                mpc_optimizer.volume_model
            )
        else:
            self.fallback = None
            logger.warning("Fallback solver not available (missing fleet or volume_model)")
    
    def optimize(
        self,
        initial_volume: float,
        inflow_forecast: np.ndarray,
        price_forecast: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize with retry and fallback logic.
        
        Returns:
            Optimization result dict with status field
        """
        horizon = len(inflow_forecast)
        
        # Try main optimizer with progressively relaxed parameters
        for attempt in range(self.max_retries + 1):
            try:
                # Adjust timeout and gap for retries
                if attempt > 0:
                    timeout_multiplier = 2 ** attempt
                    gap_multiplier = 2 ** attempt
                    
                    original_timeout = getattr(self.mpc_optimizer, 'time_limit_s', 10)
                    original_gap = getattr(self.mpc_optimizer, 'mip_gap', 0.02)
                    
                    self.mpc_optimizer.time_limit_s = original_timeout * timeout_multiplier
                    self.mpc_optimizer.mip_gap = original_gap * gap_multiplier
                    
                    logger.info(
                        f"Retry {attempt}: timeout={self.mpc_optimizer.time_limit_s}s, "
                        f"gap={self.mpc_optimizer.mip_gap}"
                    )
                
                # Call base optimizer
                if price_forecast is None:
                    price_forecast = np.ones(horizon) * 0.1  # Default price
                
                result = self.mpc_optimizer.optimize(
                    initial_volume_m3=initial_volume,
                    inflow_forecast_m3_per_15min=inflow_forecast / 4.0,  # Convert m3/h to per 15min
                    price_forecast_eur_kwh=price_forecast
                )
                
                if result and result.get('status') in ['optimal', 'feasible']:
                    logger.info(f"Optimization successful on attempt {attempt + 1}")
                    return result
                
                logger.warning(f"Attempt {attempt + 1} failed: {result.get('status') if result else 'None'}")
                
            except Exception as e:
                logger.error(f"Optimization attempt {attempt + 1} error: {e}")
                
                if attempt == self.max_retries:
                    # Last attempt failed, try fallback
                    break
        
        # All retries failed, use fallback if enabled
        if self.enable_fallback and self.fallback:
            logger.warning("All optimization attempts failed, using fallback")
            return self.fallback.solve(
                initial_volume=initial_volume,
                inflow_forecast=inflow_forecast,
                horizon=horizon
            )
        
        # Total failure
        logger.error("Optimization completely failed with no fallback")
        return {
            'status': OptimizationStatus.ERROR.value,
            'message': 'Optimization failed and no fallback available',
            'frequencies': None,
            'volumes': None,
            'power': None
        }


def create_robust_optimizer(config: Dict[str, Any]) -> RobustOptimizer:
    """
    Factory function to create robust optimizer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RobustOptimizer instance
    """
    from src.optimize import MPCOptimizer
    
    # Extract optimization config
    opt_config = config.get('optimization', {})
    timeout = opt_config.get('solver_timeout', 10)
    gap = opt_config.get('solver_gap', 0.02)
    max_retries = opt_config.get('max_retries', 2)
    enable_fallback = opt_config.get('enable_fallback', True)
    
    # Create base optimizer
    mpc = MPCOptimizer(time_limit_s=timeout, mip_gap=gap)
    
    # Wrap with robust logic
    return RobustOptimizer(
        mpc_optimizer=mpc,
        max_retries=max_retries,
        enable_fallback=enable_fallback
    )
