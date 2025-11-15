"""
Multi-Agent Wastewater System with OPC UA Digital Twin
Main orchestrator for Junction 2025 Challenge
"""

import logging
import argparse
import time
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from src.digital_twin import DigitalTwinServer, DigitalTwinClient, OPCUA_AVAILABLE
from src.agents import create_agent_system
from src.dashboard_multiagent import generate_static_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent system with OPC UA Digital Twin.
    Coordinates: Forecast -> Planner -> Executor -> Supervisor cycle.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.digital_twin_server = None
        self.digital_twin_client = None
        self.agents = None
        self.historical_data = None
        self.simulation_mode = config.get('simulation_mode', 'offline')
    
    def setup_digital_twin(self, historical_data: pd.DataFrame = None):
        """Initialize OPC UA Digital Twin server and client."""
        if not OPCUA_AVAILABLE:
            logger.warning("OPC UA not available. Running in simulation-only mode.")
            return False
        
        try:
            # Start server
            endpoint = self.config.get('opcua_endpoint', 'opc.tcp://0.0.0.0:4840/freeopcua/server/')
            self.digital_twin_server = DigitalTwinServer(endpoint)
            
            # Start with historical data replay
            replay = self.config.get('replay_historical', False)
            step_delay = self.config.get('replay_step_delay', 0.1)
            
            self.digital_twin_server.start(
                historical_data=historical_data,
                replay=replay,
                step_delay=step_delay
            )
            
            # Give server time to start
            time.sleep(2)
            
            # Connect client
            client_endpoint = endpoint.replace('0.0.0.0', 'localhost')
            self.digital_twin_client = DigitalTwinClient(client_endpoint)
            self.digital_twin_client.connect()
            
            logger.info("Digital Twin OPC UA server and client initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Digital Twin: {e}")
            return False
    
    def setup_agents(self):
        """Create and initialize all agents."""
        self.agents = create_agent_system(
            digital_twin_client=self.digital_twin_client,
            config=self.config
        )
        
        # Train ForecastAgent if historical data available
        if self.historical_data is not None:
            logger.info("Training ForecastAgent on historical data...")
            inflow_col = 'Inflow to tunnel F1 (m³/15 min)'
            if inflow_col in self.historical_data.columns:
                inflow_series = self.historical_data[inflow_col].dropna()
                epochs = self.config.get('forecast_epochs', 30)
                self.agents['forecast'].train(inflow_series, epochs=epochs)
                logger.info("ForecastAgent training complete")
    
    def run_offline_simulation(self, num_steps: int = 96):
        """
        Run offline simulation without OPC UA.
        Uses historical data to simulate 24 hours of operation.
        """
        logger.info(f"Starting offline simulation for {num_steps} steps (24 hours)")
        
        # Initialize from historical data
        if self.historical_data is not None and len(self.historical_data) > 0:
            initial_row = self.historical_data.iloc[0]
            initial_V = initial_row.get('Water volume in tunnel V (m³)', 15000.0)
            logger.info(f"Initial volume: {initial_V:.0f} m3")
        else:
            initial_V = 15000.0
        
        # Storage for results and incremental cost
        results = []
        cumulative_cost = 0.0
        current_V = initial_V
        
        for step in range(num_steps):
            logger.info(f"\n=== STEP {step + 1}/{num_steps} ===")
            
            # 1. Forecast Agent
            forecast_result = self.agents['forecast'].step()
            F1_forecast = self.agents['forecast'].state.get('latest_F1_forecast', [1000.0] * 96)
            price_forecast = self.agents['forecast'].state.get('latest_price_forecast', [0.07] * 96)
            
            # 2. Planner Agent
            self.agents['planner'].set_forecasts(F1_forecast, price_forecast)
            self.agents['planner']._shared_F1_forecast = F1_forecast
            self.agents['planner']._shared_price_forecast = price_forecast
            
            # Update planner perception manually for offline mode
            planner_perception = {
                'current_V': current_V,
                'current_L1': current_V / 2500.0,  # Approximate
                'F1_forecast': F1_forecast,
                'price_forecast': price_forecast
            }
            planner_decision = self.agents['planner'].decide(planner_perception)
            self.agents['planner'].act(planner_decision)
            
            # 3. Executor Agent - apply first step of plan
            if planner_decision.get('success', False):
                schedule = planner_decision['schedule']
                if 'flow_m3h' in schedule.columns:
                    target_F2 = float(schedule['flow_m3h'].iloc[0])
                elif 'F2_total' in schedule.columns:
                    target_F2 = float(schedule['F2_total'].iloc[0])
                else:
                    target_F2 = 2000.0
                
                # Get actual F1 from historical data if available
                if self.historical_data is not None and step < len(self.historical_data):
                    actual_F1 = self.historical_data.iloc[step].get(
                        'Inflow to tunnel F1 (m³/15 min)', 1000.0)
                else:
                    actual_F1 = F1_forecast[0]
                
                # Compute step cost using only the applied first step
                if 'cost_eur' in schedule.columns:
                    step_cost = float(schedule['cost_eur'].iloc[0])
                else:
                    step_power = float(schedule['power_kw'].iloc[0]) if 'power_kw' in schedule.columns else 0.0
                    step_price = float(price_forecast[0])
                    step_cost = 0.25 * step_power * step_price
                cumulative_cost += step_cost
                
                # Execute dynamics
                executor_perception = {
                    'current_V': current_V,
                    'current_F1': actual_F1,
                    'target_F2': target_F2,
                    'timestep': step
                }
                executor_decision = self.agents['executor'].decide(executor_perception)
                
                # Update state
                current_V = executor_decision['V_next']
                current_L1 = executor_decision['L_next']
                
                logger.info(f"Volume: {current_V:.0f} m3, Level: {current_L1:.2f} m, "
                          f"F1: {actual_F1:.0f}, F2: {target_F2:.0f} m3/h")
            
            # 4. Supervisor Agent
            supervisor_perception = {
                'current_V': current_V,
                'current_L1': current_V / 2500.0,
                'agent_states': {name: agent.state for name, agent in self.agents.items()}
            }
            supervisor_decision = self.agents['supervisor'].decide(supervisor_perception)
            self.agents['supervisor'].act(supervisor_decision)
            
            # Store results
            results.append({
                'step': step,
                'V': current_V,
                'L1': current_L1,
                'F1': actual_F1 if 'actual_F1' in locals() else F1_forecast[0],
                'F2': target_F2 if 'target_F2' in locals() else 0.0,
                'cost': step_cost if 'step_cost' in locals() else 0.0,
                'alerts': len(supervisor_decision.get('alerts', []))
            })
        
        # Generate summary
        results_df = pd.DataFrame(results)
        summary = self.agents['supervisor'].get_summary()
        
        logger.info("\n" + "="*60)
        logger.info("SIMULATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total steps: {summary['total_steps']}")
        logger.info(f"Final volume: {current_V:.0f} m3")
        logger.info(f"Alerts raised: {summary['alerts_count']}")
        
        if 'kpis' in planner_decision:
            kpis = planner_decision['kpis']
            logger.info(f"\nFinal KPIs:")
            for key, value in kpis.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.2f}")
        
        return results_df
    
    def run_realtime_simulation(self, duration_steps: int = 96):
        """
        Run real-time simulation with OPC UA Digital Twin.
        Server replays historical data, agents react in real-time.
        """
        if not OPCUA_AVAILABLE or self.digital_twin_client is None:
            logger.error("OPC UA not available. Cannot run real-time simulation.")
            return None
        
        logger.info(f"Starting real-time simulation for {duration_steps} steps")
        
        results = []
        
        for step in range(duration_steps):
            logger.info(f"\n=== REALTIME STEP {step + 1}/{duration_steps} ===")
            
            # Agents perceive through OPC UA
            forecast_result = self.agents['forecast'].step()
            
            # Share forecasts
            F1_forecast = self.agents['forecast'].state.get('latest_F1_forecast', [])
            price_forecast = self.agents['forecast'].state.get('latest_price_forecast', [])
            
            if F1_forecast:
                self.agents['planner'].set_forecasts(F1_forecast, price_forecast)
                planner_result = self.agents['planner'].step()
                executor_result = self.agents['executor'].step()
                supervisor_result = self.agents['supervisor'].step()
                
                # Read current state from Digital Twin
                state = self.digital_twin_client.get_system_state()
                
                results.append({
                    'step': step,
                    'V': state.get('Volume V m3', 0),
                    'L1': state.get('WaterLevel L1 m', 0),
                    'planner_success': planner_result.get('success', False),
                    'executor_success': executor_result.get('success', False)
                })
            
            # Wait for next timestep (adjustable speed)
            time.sleep(self.config.get('realtime_step_delay', 0.5))
        
        results_df = pd.DataFrame(results)
        logger.info("Real-time simulation complete")
        
        return results_df
    
    def shutdown(self):
        """Cleanup resources."""
        if self.digital_twin_client:
            self.digital_twin_client.disconnect()
        
        if self.digital_twin_server:
            self.digital_twin_server.stop()
        
        logger.info("Multi-Agent system shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Wastewater System with OPC UA Digital Twin"
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/test_data.csv',
                       help='Path to historical data')
    parser.add_argument('--mode', type=str, default='offline',
                       choices=['offline', 'realtime'],
                       help='Simulation mode')
    parser.add_argument('--steps', type=int, default=96,
                       help='Number of simulation steps (default 96 = 24 hours)')
    parser.add_argument('--output', type=str, default='results/multi_agent_results.csv',
                       help='Output results file')
    parser.add_argument('--report', action='store_true',
                       help='Generate HTML report alongside CSV output')
    parser.add_argument('--baseline', type=float, default=None,
                       help='Baseline cost (EUR) for savings calculation in report/summary')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        config = {
            'timestep_hours': 0.25,
            'forecast_epochs': 30,
            'simulation_mode': args.mode,
            'opcua_endpoint': 'opc.tcp://0.0.0.0:4840/freeopcua/server/',
            'replay_historical': False,
            'replay_step_delay': 0.1,
            'realtime_step_delay': 0.5
        }
    
    # Override with CLI args
    config['simulation_mode'] = args.mode
    
    # Load historical data
    data_path = Path(args.data)
    if data_path.exists():
        logger.info(f"Loading historical data from {data_path}")
        
        if data_path.suffix == '.csv':
            historical_data = pd.read_csv(data_path, skiprows=[1])
        else:
            historical_data = pd.read_excel(data_path, skiprows=[1])
        
        logger.info(f"Loaded {len(historical_data)} rows of historical data")
    else:
        logger.error(f"Data file {args.data} not found")
        return
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(config)
    orchestrator.historical_data = historical_data
    
    try:
        # Setup Digital Twin (if in realtime mode)
        if args.mode == 'realtime':
            success = orchestrator.setup_digital_twin(historical_data)
            if not success:
                logger.warning("Falling back to offline mode")
                args.mode = 'offline'
        
        # Setup agents
        orchestrator.setup_agents()
        
        # Run simulation
        if args.mode == 'offline':
            results = orchestrator.run_offline_simulation(num_steps=args.steps)
        else:
            results = orchestrator.run_realtime_simulation(duration_steps=args.steps)
        
        # Save results
        if results is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Generate HTML report if requested
            if args.report:
                report_path = str(output_path).replace('.csv', '_report.html')
                generate_static_report(results, baseline_cost=args.baseline, output_path=report_path)
                logger.info(f"HTML report saved to {report_path}")
            
            # Display summary statistics
            logger.info("\n" + "="*60)
            logger.info("RESULTS SUMMARY")
            logger.info("="*60)
            logger.info(f"Volume - Min: {results['V'].min():.0f}, "
                       f"Max: {results['V'].max():.0f}, "
                       f"Avg: {results['V'].mean():.0f} m3")
            logger.info(f"Level - Min: {results['L1'].min():.2f}, "
                       f"Max: {results['L1'].max():.2f}, "
                       f"Avg: {results['L1'].mean():.2f} m")
            
            if 'cost' in results.columns:
                total_cost = float(results['cost'].sum())
                logger.info(f"Total cost: {total_cost:.2f} EUR")
                
                if args.baseline is not None:
                    # Scale baseline by simulated horizon (assuming baseline is for 24h = 96 steps)
                    baseline_total = float(args.baseline) * (float(args.steps) / 96.0)
                    savings_pct = (baseline_total - total_cost) / baseline_total * 100 if baseline_total > 0 else 0.0
                    logger.info(f"Baseline cost (scaled): {baseline_total:.2f} EUR")
                    logger.info(f"Savings: {savings_pct:.1f}% ({baseline_total - total_cost:.2f} EUR)")
    
    finally:
        # Cleanup
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
