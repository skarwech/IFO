"""
Multi-Agent Framework for Intelligent Pump Scheduling
Agents: Forecast, Planner, Executor, Supervisor
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.forecast import InflowForecaster
from src.optimize import MPCOptimizer
from src.model import VolumeModel, PumpFleet

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, digital_twin_client=None):
        self.name = name
        self.client = digital_twin_client
        self.state = {}
        self.history = []
        logger.info(f"Agent '{name}' initialized")
    
    @abstractmethod
    def perceive(self) -> Dict[str, Any]:
        """Gather information from environment."""
        pass
    
    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decisions based on perception."""
        pass
    
    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> bool:
        """Execute decision."""
        pass
    
    def step(self) -> Dict[str, Any]:
        """Execute one agent cycle: perceive -> decide -> act."""
        perception = self.perceive()
        decision = self.decide(perception)
        success = self.act(decision)
        
        result = {
            'agent': self.name,
            'perception': perception,
            'decision': decision,
            'success': success,
            'timestamp': pd.Timestamp.now()
        }
        
        self.history.append(result)
        return result


class ForecastAgent(BaseAgent):
    """
    Agent responsible for forecasting inflows and energy prices.
    Uses LSTM for F1 prediction, simple patterns for prices.
    """
    
    def __init__(self, digital_twin_client=None, horizon_steps: int = 96):
        super().__init__("ForecastAgent", digital_twin_client)
        self.horizon_steps = horizon_steps
        self.inflow_forecaster = InflowForecaster(
            lookback_steps=32,
            hidden_size=64,
            num_layers=2
        )
        self.trained = False
        self.recent_inflows = []
        self.recent_prices = []
    
    def train(self, historical_inflows: pd.Series, epochs: int = 50):
        """Train LSTM forecaster on historical data."""
        logger.info(f"Training {self.name} on {len(historical_inflows)} samples")
        self.inflow_forecaster.fit(historical_inflows, epochs=epochs)
        self.trained = True
    
    def perceive(self) -> Dict[str, Any]:
        """Read recent inflows and prices from Digital Twin."""
        perception = {}
        
        if self.client:
            try:
                state = self.client.get_system_state()
                perception['current_F1'] = state.get('InflowF1 m3 15min', 0.0)
                perception['price_high'] = state.get('PriceHigh EUR kWh', 0.0)
                perception['price_normal'] = state.get('PriceNormal EUR kWh', 0.0)
                
                # Update recent history
                self.recent_inflows.append(perception['current_F1'])
                if len(self.recent_inflows) > 96:
                    self.recent_inflows = self.recent_inflows[-96:]
                
            except Exception as e:
                logger.error(f"{self.name} perception error: {e}")
        
        perception['recent_inflows'] = self.recent_inflows.copy()
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasts for inflows and prices."""
        decision = {
            'F1_forecast': [],
            'price_forecast': []
        }
        
        # Inflow forecast
        if self.trained and len(perception['recent_inflows']) >= 24:
            recent_series = pd.Series(perception['recent_inflows'][-24:])
            try:
                forecast = self.inflow_forecaster.forecast(
                    recent_series,
                    horizon_steps=self.horizon_steps
                )
                decision['F1_forecast'] = forecast
            except Exception as e:
                logger.warning(f"LSTM forecast failed: {e}. Using persistence.")
                avg = np.mean(perception['recent_inflows'][-24:])
                decision['F1_forecast'] = [avg] * self.horizon_steps
        else:
            # Simple persistence model
            if perception['recent_inflows']:
                avg = np.mean(perception['recent_inflows'][-min(24, len(perception['recent_inflows'])):])
                decision['F1_forecast'] = [avg] * self.horizon_steps
            else:
                decision['F1_forecast'] = [1000.0] * self.horizon_steps
        
        # Price forecast (simple pattern: high during day, normal at night)
        base_high = perception.get('price_high', 0.10)
        base_normal = perception.get('price_normal', 0.05)
        
        price_pattern = []
        for h in range(self.horizon_steps):
            # 15-min steps, 4 per hour
            hour = (h // 4) % 24
            if 8 <= hour < 22:  # Day hours
                price_pattern.append(base_high)
            else:  # Night hours
                price_pattern.append(base_normal)
        
        decision['price_forecast'] = price_pattern
        
        logger.debug(f"{self.name} forecasted F1 mean: {np.mean(decision['F1_forecast']):.1f} m3/15min")
        return decision
    
    def act(self, decision: Dict[str, Any]) -> bool:
        """Store forecasts for other agents to access."""
        self.state['latest_F1_forecast'] = decision['F1_forecast']
        self.state['latest_price_forecast'] = decision['price_forecast']
        return True


class PlannerAgent(BaseAgent):
    """
    Agent that creates optimal pump schedules using MPC.
    Uses forecasts from ForecastAgent.
    """
    
    def __init__(self, digital_twin_client=None, config: Dict[str, Any] = None):
        super().__init__("PlannerAgent", digital_twin_client)
        self.config = config or {}
        
        # Initialize optimizer (uses internal VolumeModel and PumpFleet)
        self.optimizer = MPCOptimizer(
            small_count=4,
            large_count=4,
            time_limit_s=self.config.get('solver_timeout', 10),
            mip_gap=self.config.get('solver_gap', 0.02)
        )
        
        self.current_schedule = None
    
    def perceive(self) -> Dict[str, Any]:
        """Read current state and forecasts."""
        perception = {}
        
        if self.client:
            try:
                state = self.client.get_system_state()
                perception['current_V'] = state.get('Volume V m3', 0.0)
                perception['current_L1'] = state.get('WaterLevel L1 m', 0.0)
            except Exception as e:
                logger.error(f"{self.name} perception error: {e}")
                perception['current_V'] = 15000.0  # Default
        
        # Get forecasts from other agents (would use shared memory or OPC UA in production)
        perception['F1_forecast'] = getattr(self, '_shared_F1_forecast', [1000.0] * 96)
        perception['price_forecast'] = getattr(self, '_shared_price_forecast', [0.07] * 96)
        
        return perception
    
    def set_forecasts(self, F1_forecast: List[float], price_forecast: List[float]):
        """Receive forecasts from ForecastAgent (interface method)."""
        self._shared_F1_forecast = F1_forecast
        self._shared_price_forecast = price_forecast
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Run MPC optimization to generate pump schedule."""
        logger.info(f"{self.name} running optimization...")
        
        try:
            # Respect configured planning horizon to reduce MILP size
            F1_fc = perception['F1_forecast']
            price_fc = perception['price_forecast']
            horizon = self.config.get('horizon_steps', len(F1_fc))
            if horizon < len(F1_fc):
                F1_fc = F1_fc[:horizon]
                price_fc = price_fc[:horizon]
            result = self.optimizer.optimize(
                initial_volume_m3=perception['current_V'],
                inflow_forecast_m3_per_15min=pd.Series(F1_fc),
                price_forecast_eur_kwh=pd.Series(price_fc)
            )
            
            schedule = result.get('schedule')
            kpis = result.get('kpis', {})
            success = result.get('status') in ('Optimal', 'FeasibleWithinTime') and schedule is not None
            
            if not success:
                raise RuntimeError(f"Optimization status: {result.get('status')}")
            
            decision = {
                'schedule': schedule,
                'kpis': kpis,
                'success': True
            }
            
            logger.info(f"{self.name} optimization complete. Total cost: {kpis.get('total_cost_eur', 0):.2f} EUR")
            
        except Exception as e:
            logger.error(f"{self.name} optimization failed: {e}")
            # Fallback: constant moderate pumping
            decision = {
                'schedule': self._create_fallback_schedule(len(perception['F1_forecast'])),
                'kpis': {},
                'success': False
            }
        
        return decision
    
    def _create_fallback_schedule(self, horizon: int) -> pd.DataFrame:
        """Create simple fallback schedule if optimization fails."""
        schedule = pd.DataFrame({
            'timestep': range(horizon),
            'F2_total': [2000.0] * horizon,
        })
        
        # Simple split between pumps
        for section in [1, 2]:
            for num in range(1, 5):
                pump_id = f"{section}.{num}"
                schedule[f'F2_{pump_id}'] = 0.0
        
        # Assign to 2 large pumps
        schedule['F2_2.1'] = 1000.0
        schedule['F2_2.2'] = 1000.0
        
        return schedule
    
    def act(self, decision: Dict[str, Any]) -> bool:
        """Write schedule to Digital Twin for ExecutorAgent."""
        self.current_schedule = decision['schedule']
        self.state['latest_kpis'] = decision.get('kpis', {})
        
        if self.client and decision.get('success', False):
            try:
                # Write first step target to OPC UA
                schedule = decision['schedule']
                if 'flow_m3h' in schedule.columns:
                    first_F2 = float(schedule['flow_m3h'].iloc[0])
                elif 'F2_total' in schedule.columns:
                    first_F2 = float(schedule['F2_total'].iloc[0])
                else:
                    first_F2 = 2000.0
                self.client.write_control_plan(first_F2, "Scheduled")
                return True
            except Exception as e:
                logger.error(f"{self.name} failed to write plan: {e}")
                return False
        
        return decision.get('success', False)


class ExecutorAgent(BaseAgent):
    """
    Agent that executes pump schedules and simulates tunnel dynamics.
    Reads plan from PlannerAgent, applies pumping, updates volume.
    """
    
    def __init__(self, digital_twin_client=None, config: Dict[str, Any] = None):
        super().__init__("ExecutorAgent", digital_twin_client)
        self.config = config or {}
        self.volume_model = VolumeModel()
        self.current_step = 0
    
    def perceive(self) -> Dict[str, Any]:
        """Read current state and control plan."""
        perception = {}
        
        if self.client:
            try:
                state = self.client.get_system_state()
                perception['current_V'] = state.get('Volume V m3', 15000.0)
                perception['current_F1'] = state.get('InflowF1 m3 15min', 1000.0)
                perception['target_F2'] = self.client.read_variable("Control/TargetF2_m3h")
            except Exception as e:
                logger.error(f"{self.name} perception error: {e}")
        
        perception['timestep'] = self.current_step
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate new volume based on mass balance."""
        V_current = perception.get('current_V', 15000.0)
        F1 = perception.get('current_F1', 1000.0)  # m3/15min
        F2_target = perception.get('target_F2', 2000.0)  # m3/h
        
        timestep_hours = self.config.get('timestep_hours', 0.25)
        
        # Convert F2 to m3/15min
        F2_15min = F2_target * timestep_hours
        
        # Mass balance: V(t+1) = V(t) + F1 - F2
        V_next = V_current + F1 - F2_15min
        
        # Enforce constraints
        V_min = self.config.get('v_min', 5000.0)
        V_max = self.config.get('v_max', 35000.0)
        V_next = np.clip(V_next, V_min, V_max)
        
        # Calculate new level
        try:
            L_next = self.volume_model.volume_to_level(V_next)
        except:
            L_next = V_next / 2500.0  # Approximate
        
        decision = {
            'V_next': V_next,
            'L_next': L_next,
            'F2_actual': F2_15min,
            'success': True
        }
        
        return decision
    
    def act(self, decision: Dict[str, Any]) -> bool:
        """Update Digital Twin state with new volume and level."""
        if self.client and decision.get('success', False):
            try:
                self.client.write_variable("Tunnel/Volume_V_m3", decision['V_next'])
                self.client.write_variable("Tunnel/WaterLevel_L1_m", decision['L_next'])
                self.client.write_variable("Control/OptimizationStatus", "Executing")
                
                self.current_step += 1
                logger.debug(f"{self.name} updated state: V={decision['V_next']:.0f} m3, L={decision['L_next']:.2f} m")
                return True
            except Exception as e:
                logger.error(f"{self.name} failed to update state: {e}")
                return False
        
        return False


class SupervisorAgent(BaseAgent):
    """
    Coordinating agent that manages other agents.
    Monitors performance, calculates savings, aggregates metrics.
    """
    
    def __init__(self, digital_twin_client=None, agents: Dict[str, BaseAgent] = None):
        super().__init__("SupervisorAgent", digital_twin_client)
        self.agents = agents or {}
        self.metrics = {
            'total_cost': 0.0,
            'total_energy': 0.0,
            'constraint_violations': 0,
            'steps_executed': 0
        }
        self.baseline_cost = 0.0
    
    def perceive(self) -> Dict[str, Any]:
        """Monitor all agents and system state."""
        perception = {
            'agent_states': {},
            'system_health': 'OK'
        }
        
        for name, agent in self.agents.items():
            perception['agent_states'][name] = {
                'state': agent.state,
                'history_length': len(agent.history)
            }
        
        if self.client:
            try:
                state = self.client.get_system_state()
                perception['current_L1'] = state.get('WaterLevel L1 m', 0.0)
                perception['current_V'] = state.get('Volume V m3', 0.0)
            except Exception as e:
                logger.error(f"{self.name} perception error: {e}")
        
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance and decide on interventions."""
        decision = {
            'actions': [],
            'metrics': {},
            'alerts': []
        }
        
        # Check for constraint violations
        L1 = perception.get('current_L1', 10.0)
        if L1 < 2.0:
            decision['alerts'].append("CRITICAL: Water level too low")
            decision['actions'].append('increase_pumping')
        elif L1 > 13.0:
            decision['alerts'].append("WARNING: Water level high")
            decision['actions'].append('decrease_pumping')
        
        # Aggregate metrics from PlannerAgent
        if 'PlannerAgent' in self.agents:
            planner = self.agents['PlannerAgent']
            kpis = planner.state.get('latest_kpis', {})
            decision['metrics'] = kpis
            
            # Calculate savings
            if self.baseline_cost > 0:
                optimized_cost = kpis.get('total_cost_eur', 0)
                savings_pct = (self.baseline_cost - optimized_cost) / self.baseline_cost * 100
                decision['metrics']['savings_percent'] = savings_pct
                logger.info(f"{self.name} calculated savings: {savings_pct:.1f}%")
        
        return decision
    
    def act(self, decision: Dict[str, Any]) -> bool:
        """Execute supervisory actions (alerts, coordination)."""
        # Log alerts
        for alert in decision.get('alerts', []):
            logger.warning(f"{self.name} ALERT: {alert}")
        
        # Update metrics
        self.metrics.update(decision.get('metrics', {}))
        self.metrics['steps_executed'] += 1
        
        return True
    
    def set_baseline(self, baseline_cost: float):
        """Set baseline cost for savings calculation."""
        self.baseline_cost = baseline_cost
        logger.info(f"{self.name} baseline cost set to {baseline_cost:.2f} EUR")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_steps': self.metrics['steps_executed'],
            'total_cost': self.metrics.get('total_cost_eur', 0),
            'baseline_cost': self.baseline_cost,
            'savings_percent': self.metrics.get('savings_percent', 0),
            'agent_count': len(self.agents),
            'alerts_count': sum(len(h.get('decision', {}).get('alerts', [])) 
                              for h in self.history)
        }


def create_agent_system(digital_twin_client=None, config: Dict[str, Any] = None) -> Dict[str, BaseAgent]:
    """
    Factory function to create all agents.
    """
    agents = {}
    
    # Create agents
    agents['forecast'] = ForecastAgent(digital_twin_client)
    agents['planner'] = PlannerAgent(digital_twin_client, config)
    agents['executor'] = ExecutorAgent(digital_twin_client, config)
    
    # Create supervisor with reference to all agents
    agents['supervisor'] = SupervisorAgent(digital_twin_client, agents)
    
    logger.info(f"Created agent system with {len(agents)} agents")
    return agents
