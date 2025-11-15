"""
Model Predictive Control (MPC) optimization engine using MILP.
"""

import numpy as np
import pandas as pd
from pulp import *
from typing import Dict, List, Tuple, Optional
import logging

from .data_utils import VolumeCalculator
from .pump_models import PumpStation

logger = logging.getLogger(__name__)


class MPCOptimizer:
    """
    MPC-based pump scheduler using Mixed Integer Linear Programming.
    """
    
    def __init__(
        self,
        config: Dict,
        volume_calculator: VolumeCalculator,
        pump_station: PumpStation
    ):
        """
        Initialize MPC optimizer.
        
        Args:
            config: Configuration dictionary
            volume_calculator: Volume-level converter
            pump_station: Pump station model
        """
        self.config = config
        self.volume_calc = volume_calculator
        self.pump_station = pump_station
        
        # MPC parameters
        mpc_config = config['mpc']
        self.horizon_hours = mpc_config['horizon_hours']
        self.timestep_minutes = config['simulation']['timestep_minutes']
        self.timestep_hours = self.timestep_minutes / 60.0
        self.horizon_steps = int((self.horizon_hours * 60) / self.timestep_minutes)
        
        # Objective weights
        self.weight_energy = mpc_config['objective_weights']['energy_cost']
        self.weight_smoothness = mpc_config['objective_weights']['flow_smoothness']
        self.weight_efficiency = mpc_config['objective_weights']['efficiency_penalty']
        self.weight_emptying = mpc_config['objective_weights']['emptying_reward']
        
        # System constraints
        self.l1_min = config['system']['tunnel']['l1_min']
        self.l1_max = config['system']['tunnel']['l1_max']
        self.l2_fixed = config['system']['tunnel']['l2_fixed']
        self.emptying_threshold = config['system']['tunnel']['emptying_threshold']
        
        # Constraint parameters
        constraints_config = config['constraints']
        self.freq_min = constraints_config['frequency']['min_hz']
        self.freq_max = constraints_config['frequency']['max_hz']
        self.freq_step = constraints_config['frequency']['step_hz']
        self.max_ramp = constraints_config['flow']['max_ramp_m3h']
        self.min_on_time = int(constraints_config['pumps']['min_on_time_hours'] * 60 / self.timestep_minutes)
        self.min_off_time = int(constraints_config['pumps']['min_off_time_hours'] * 60 / self.timestep_minutes)
        self.min_active_pumps = constraints_config['pumps']['min_active_pumps']
        
        # Discretized frequencies
        self.frequencies = np.arange(self.freq_min, self.freq_max + self.freq_step, self.freq_step)
        
        # Number of pumps
        self.num_pumps = pump_station.get_num_pumps()
        
        logger.info(
            f"MPC initialized: horizon={self.horizon_steps} steps "
            f"({self.horizon_hours}h), {self.num_pumps} pumps"
        )
    
    def _linearize_pump_curves(self) -> Dict:
        """
        Pre-compute linearized pump performance for each frequency.
        
        Returns:
            Dictionary with pump performance data
        """
        pump_data = {}
        
        for i in range(self.num_pumps):
            pump = self.pump_station.get_pump(i)
            pump_data[i] = {}
            
            for freq in self.frequencies:
                # For typical head range
                head_range = np.linspace(20, 35, 10)
                flows = []
                powers = []
                
                for head in head_range:
                    flow = pump.calculate_flow(head, freq)
                    _, power = pump.calculate_power(flow, head, freq)
                    flows.append(flow * 3.6)  # Convert to m³/h
                    powers.append(power)
                
                # Store average values (simplified)
                pump_data[i][freq] = {
                    'avg_flow_m3h': np.mean(flows),
                    'avg_power_kw': np.mean(powers)
                }
        
        return pump_data
    
    def optimize(
        self,
        initial_state: Dict[str, float],
        inflow_forecast: np.ndarray,
        price_forecast: np.ndarray,
        previous_pump_states: Optional[Dict[int, Dict[str, float]]] = None,
        require_emptying: bool = False
    ) -> Dict:
        """
        Solve MPC optimization problem.
        
        Args:
            initial_state: Current system state {'level', 'volume'}
            inflow_forecast: Forecasted inflow for horizon (m³ per timestep)
            price_forecast: Forecasted electricity prices (EUR/kWh)
            previous_pump_states: Previous pump states for min on/off time constraints
            require_emptying: Whether to enforce emptying constraint
            
        Returns:
            Dictionary with optimal solution
        """
        logger.info("Starting MPC optimization...")
        
        # Ensure forecast length matches horizon
        if len(inflow_forecast) < self.horizon_steps:
            # Pad with last value
            inflow_forecast = np.pad(
                inflow_forecast,
                (0, self.horizon_steps - len(inflow_forecast)),
                mode='edge'
            )
        inflow_forecast = inflow_forecast[:self.horizon_steps]
        
        if len(price_forecast) < self.horizon_steps:
            price_forecast = np.pad(
                price_forecast,
                (0, self.horizon_steps - len(price_forecast)),
                mode='edge'
            )
        price_forecast = price_forecast[:self.horizon_steps]
        
        # Pre-compute pump data
        pump_data = self._linearize_pump_curves()
        
        # Create optimization problem
        prob = LpProblem("Pump_Scheduling", LpMinimize)
        
        # Decision variables
        # Binary: pump on/off
        pump_on = LpVariable.dicts(
            "pump_on",
            ((t, i) for t in range(self.horizon_steps) for i in range(self.num_pumps)),
            cat='Binary'
        )
        
        # Binary: frequency selection (one-hot encoding)
        freq_select = LpVariable.dicts(
            "freq_select",
            ((t, i, f) for t in range(self.horizon_steps) 
             for i in range(self.num_pumps) 
             for f in range(len(self.frequencies))),
            cat='Binary'
        )
        
        # Continuous: volumes
        volume = LpVariable.dicts(
            "volume",
            range(self.horizon_steps + 1),
            lowBound=self.volume_calc.volume_from_level(self.l1_min),
            upBound=self.volume_calc.volume_from_level(self.l1_max)
        )
        
        # Continuous: total flow at each timestep
        total_flow = LpVariable.dicts(
            "total_flow",
            range(self.horizon_steps),
            lowBound=0
        )
        
        # Continuous: power at each timestep
        power = LpVariable.dicts(
            "power",
            range(self.horizon_steps),
            lowBound=0
        )
        
        # Initial volume
        prob += volume[0] == initial_state['volume']
        
        # Constraints for each timestep
        for t in range(self.horizon_steps):
            # Volume dynamics: V(t+1) = V(t) + F1(t) - F2(t)
            # F2(t) in m³ = total_flow(t) in m³/h * timestep_hours
            prob += (
                volume[t + 1] == volume[t] + inflow_forecast[t] - total_flow[t] * self.timestep_hours
            )
            
            # Each pump: select exactly one frequency if on
            for i in range(self.num_pumps):
                prob += lpSum([freq_select[t, i, f] for f in range(len(self.frequencies))]) == pump_on[t, i]
            
            # Total flow = sum of individual pump flows
            flow_expr = lpSum([
                lpSum([
                    pump_data[i][self.frequencies[f]]['avg_flow_m3h'] * freq_select[t, i, f]
                    for f in range(len(self.frequencies))
                ])
                for i in range(self.num_pumps)
            ])
            prob += total_flow[t] == flow_expr
            
            # Total power
            power_expr = lpSum([
                lpSum([
                    pump_data[i][self.frequencies[f]]['avg_power_kw'] * freq_select[t, i, f]
                    for f in range(len(self.frequencies))
                ])
                for i in range(self.num_pumps)
            ])
            prob += power[t] == power_expr
            
            # Minimum active pumps
            prob += lpSum([pump_on[t, i] for i in range(self.num_pumps)]) >= self.min_active_pumps
            
            # Flow smoothness (ramp rate)
            if t > 0:
                prob += total_flow[t] - total_flow[t-1] <= self.max_ramp
                prob += total_flow[t-1] - total_flow[t] <= self.max_ramp
        
        # Emptying constraint
        if require_emptying:
            # At least one timestep should have volume close to minimum
            emptying_volume = self.volume_calc.volume_from_level(self.emptying_threshold)
            # Allow some margin
            margin = 1000  # m³
            
            # At least one timestep in last quarter of horizon
            emptying_window_start = int(0.75 * self.horizon_steps)
            emptying_met = LpVariable.dicts(
                "emptying_met",
                range(emptying_window_start, self.horizon_steps),
                cat='Binary'
            )
            
            for t in range(emptying_window_start, self.horizon_steps):
                # If emptying_met[t] = 1, then volume[t] <= emptying_volume + margin
                M = self.volume_calc.volume_from_level(self.l1_max)
                prob += volume[t] <= emptying_volume + margin + M * (1 - emptying_met[t])
            
            # At least one timestep must meet emptying
            prob += lpSum([emptying_met[t] for t in range(emptying_window_start, self.horizon_steps)]) >= 1
        
        # Objective function
        # 1. Energy cost
        energy_cost = lpSum([
            power[t] * self.timestep_hours * price_forecast[t]
            for t in range(self.horizon_steps)
        ])
        
        # 2. Flow smoothness (minimize variance approximation)
        avg_flow = lpSum([total_flow[t] for t in range(self.horizon_steps)]) / self.horizon_steps
        flow_variance = lpSum([
            (total_flow[t] - avg_flow) ** 2
            for t in range(self.horizon_steps)
        ])
        
        # 3. Efficiency penalty (prefer high frequencies near nominal)
        efficiency_penalty = lpSum([
            lpSum([
                lpSum([
                    # Penalize frequencies far from max
                    (1 - f / (len(self.frequencies) - 1)) * freq_select[t, i, f]
                    for f in range(len(self.frequencies))
                ])
                for i in range(self.num_pumps)
            ])
            for t in range(self.horizon_steps)
        ])
        
        # Combine objectives
        prob += (
            self.weight_energy * energy_cost +
            self.weight_smoothness * flow_variance / 1e6 +  # Scale down
            self.weight_efficiency * efficiency_penalty
        )
        
        # Solve
        solver = PULP_CBC_CMD(msg=0, timeLimit=30)
        status = prob.solve(solver)
        
        if status != LpStatusOptimal:
            logger.warning(f"Optimization status: {LpStatus[status]}")
            return self._get_fallback_solution(initial_state, inflow_forecast)
        
        # Extract solution
        solution = self._extract_solution(
            pump_on, freq_select, volume, total_flow, power,
            inflow_forecast, price_forecast
        )
        
        logger.info(
            f"Optimization complete: cost={solution['total_cost']:.2f} EUR, "
            f"energy={solution['total_energy']:.2f} kWh"
        )
        
        return solution
    
    def _extract_solution(
        self,
        pump_on: Dict,
        freq_select: Dict,
        volume: Dict,
        total_flow: Dict,
        power: Dict,
        inflow_forecast: np.ndarray,
        price_forecast: np.ndarray
    ) -> Dict:
        """Extract solution from optimization variables."""
        # Pump schedules
        pump_schedules = []
        for t in range(self.horizon_steps):
            schedule = {}
            for i in range(self.num_pumps):
                is_on = value(pump_on[t, i]) > 0.5
                
                # Find selected frequency
                freq_hz = 0.0
                if is_on:
                    for f in range(len(self.frequencies)):
                        if value(freq_select[t, i, f]) > 0.5:
                            freq_hz = self.frequencies[f]
                            break
                
                schedule[i] = {
                    'on': is_on,
                    'frequency_hz': freq_hz
                }
            pump_schedules.append(schedule)
        
        # Trajectories
        volumes = [value(volume[t]) for t in range(self.horizon_steps + 1)]
        levels = [self.volume_calc.level_from_volume(v) for v in volumes]
        flows = [value(total_flow[t]) for t in range(self.horizon_steps)]
        powers = [value(power[t]) for t in range(self.horizon_steps)]
        
        # Costs
        costs = [
            powers[t] * self.timestep_hours * price_forecast[t]
            for t in range(self.horizon_steps)
        ]
        
        return {
            'pump_schedules': pump_schedules,
            'volumes': volumes,
            'levels': levels,
            'flows': flows,
            'powers': powers,
            'costs': costs,
            'total_cost': sum(costs),
            'total_energy': sum([p * self.timestep_hours for p in powers]),
            'inflow_forecast': inflow_forecast.tolist(),
            'price_forecast': price_forecast.tolist()
        }
    
    def _get_fallback_solution(
        self,
        initial_state: Dict[str, float],
        inflow_forecast: np.ndarray
    ) -> Dict:
        """
        Generate fallback solution if optimization fails.
        Simple heuristic: run constant number of pumps at full speed.
        """
        logger.warning("Using fallback solution")
        
        # Run 2 large pumps at full speed
        pump_schedules = []
        for t in range(self.horizon_steps):
            schedule = {}
            for i in range(self.num_pumps):
                if i < 2:  # First 2 pumps (large)
                    schedule[i] = {'on': True, 'frequency_hz': 50.0}
                else:
                    schedule[i] = {'on': False, 'frequency_hz': 0.0}
            pump_schedules.append(schedule)
        
        return {
            'pump_schedules': pump_schedules,
            'volumes': [],
            'levels': [],
            'flows': [],
            'powers': [],
            'costs': [],
            'total_cost': 0,
            'total_energy': 0,
            'inflow_forecast': inflow_forecast.tolist(),
            'price_forecast': []
        }
    
    def get_next_control(self, solution: Dict) -> Dict[int, Dict[str, float]]:
        """
        Get control action for next timestep (first step of horizon).
        
        Args:
            solution: MPC solution dictionary
            
        Returns:
            Pump states for next timestep
        """
        if solution['pump_schedules']:
            return solution['pump_schedules'][0]
        else:
            # Default: one pump on
            return {i: {'on': i == 0, 'frequency_hz': 50.0 if i == 0 else 0.0} 
                   for i in range(self.num_pumps)}
