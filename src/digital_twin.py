"""
Digital Twin OPC UA Server for Wastewater System
Replays historical data and exposes real-time state to agents.
"""

import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from opcua import Server, ua
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False
    logging.warning("opcua library not available. Install with: pip install opcua")

logger = logging.getLogger(__name__)


class DigitalTwinServer:
    """
    OPC UA Server that replays historical wastewater data.
    Exposes variables: L1, V, F1, F2, pump flows/powers/frequencies, prices.
    """
    
    def __init__(self, endpoint: str = "opc.tcp://0.0.0.0:4840/freeopcua/server/"):
        if not OPCUA_AVAILABLE:
            raise ImportError("opcua library required. Install: pip install opcua")
        
        self.endpoint = endpoint
        self.server = Server()
        self.server.set_endpoint(endpoint)
        
        # Setup namespaces
        self.uri = "http://hsy.wastewater.digitaltwin"
        self.idx = self.server.register_namespace(self.uri)
        
        # Node references
        self.nodes = {}
        self.running = False
        self.replay_thread = None
        self.current_step = 0
        self.historical_data = None
        
        logger.info(f"Digital Twin OPC UA Server initialized at {endpoint}")
    
    def setup_nodes(self):
        """Create OPC UA nodes for all variables."""
        objects = self.server.get_objects_node()
        
        # Create main folder
        twin_folder = objects.add_folder(self.idx, "WastewaterSystem")
        
        # Tunnel state variables
        tunnel = twin_folder.add_folder(self.idx, "Tunnel")
        self.nodes['L1'] = tunnel.add_variable(self.idx, "WaterLevel_L1_m", 0.0)
        self.nodes['V'] = tunnel.add_variable(self.idx, "Volume_V_m3", 0.0)
        self.nodes['F1'] = tunnel.add_variable(self.idx, "InflowF1_m3_15min", 0.0)
        self.nodes['F2'] = tunnel.add_variable(self.idx, "OutflowF2_m3h", 0.0)
        
        # Make writable for agents to update plans
        for key in ['L1', 'V', 'F1', 'F2']:
            self.nodes[key].set_writable()
        
        # Pump variables (1.1-1.4 small, 2.1-2.4 large)
        pumps = twin_folder.add_folder(self.idx, "Pumps")
        for section in [1, 2]:
            for num in range(1, 5):
                pump_id = f"{section}.{num}"
                pump_folder = pumps.add_folder(self.idx, f"Pump_{pump_id}")
                
                self.nodes[f'pump_flow_{pump_id}'] = pump_folder.add_variable(
                    self.idx, f"Flow_m3h", 0.0)
                self.nodes[f'pump_power_{pump_id}'] = pump_folder.add_variable(
                    self.idx, f"Power_kW", 0.0)
                self.nodes[f'pump_freq_{pump_id}'] = pump_folder.add_variable(
                    self.idx, f"Frequency_Hz", 0.0)
                
                # Make writable for control
                self.nodes[f'pump_flow_{pump_id}'].set_writable()
                self.nodes[f'pump_power_{pump_id}'].set_writable()
                self.nodes[f'pump_freq_{pump_id}'].set_writable()
        
        # Energy prices
        energy = twin_folder.add_folder(self.idx, "Energy")
        self.nodes['price_high'] = energy.add_variable(self.idx, "PriceHigh_EUR_kWh", 0.0)
        self.nodes['price_normal'] = energy.add_variable(self.idx, "PriceNormal_EUR_kWh", 0.0)
        self.nodes['price_high'].set_writable()
        self.nodes['price_normal'].set_writable()
        
        # Control variables (for agents to write plans)
        control = twin_folder.add_folder(self.idx, "Control")
        self.nodes['target_F2'] = control.add_variable(self.idx, "TargetF2_m3h", 0.0)
        self.nodes['optimization_status'] = control.add_variable(self.idx, "OptimizationStatus", "Idle")
        self.nodes['target_F2'].set_writable()
        self.nodes['optimization_status'].set_writable()
        
        # Timestamp
        self.nodes['timestamp'] = twin_folder.add_variable(self.idx, "Timestamp", datetime.now())
        self.nodes['timestamp'].set_writable()
        
        logger.info(f"Created {len(self.nodes)} OPC UA nodes")
    
    def load_historical_data(self, df: pd.DataFrame):
        """Load historical data for replay."""
        self.historical_data = df.copy()
        self.current_step = 0
        logger.info(f"Loaded {len(df)} rows of historical data for replay")
    
    def update_from_row(self, row: pd.Series):
        """Update OPC UA variables from a data row."""
        try:
            # Map column names to node keys
            column_map = {
                'Water level in tunnel L2 (m)': 'L1',  # Note: L2 in data is actually L1
                'Water volume in tunnel V (m³)': 'V',
                'Inflow to tunnel F1 (m³/15 min)': 'F1',
                'Sum of pumped flow to WWTP F2 (m³/h)': 'F2',
                'Electricity price 1: high': 'price_high',
                'Electricity price 2: normal': 'price_normal',
            }
            
            for col, node_key in column_map.items():
                if col in row.index and node_key in self.nodes:
                    value = float(row[col]) if pd.notna(row[col]) else 0.0
                    self.nodes[node_key].set_value(value)
            
            # Update pump data
            for section in [1, 2]:
                for num in range(1, 5):
                    pump_id = f"{section}.{num}"
                    
                    flow_col = f'Pump flow {pump_id}'
                    power_col = f'pump power intake {pump_id}'
                    freq_col = f'Pump frequency {pump_id}'
                    
                    if flow_col in row.index:
                        val = float(row[flow_col]) if pd.notna(row[flow_col]) else 0.0
                        self.nodes[f'pump_flow_{pump_id}'].set_value(val)
                    
                    if power_col in row.index:
                        val = float(row[power_col]) if pd.notna(row[power_col]) else 0.0
                        self.nodes[f'pump_power_{pump_id}'].set_value(val)
                    
                    if freq_col in row.index:
                        val = float(row[freq_col]) if pd.notna(row[freq_col]) else 0.0
                        self.nodes[f'pump_freq_{pump_id}'].set_value(val)
            
            # Update timestamp
            if 'Time stamp' in row.index:
                self.nodes['timestamp'].set_value(datetime.now())
            
        except Exception as e:
            logger.error(f"Error updating OPC UA nodes: {e}")
    
    def replay_loop(self, step_delay: float = 0.1):
        """Replay historical data in loop."""
        logger.info(f"Starting replay loop with {len(self.historical_data)} steps")
        
        while self.running and self.current_step < len(self.historical_data):
            row = self.historical_data.iloc[self.current_step]
            self.update_from_row(row)
            self.current_step += 1
            
            if self.current_step % 100 == 0:
                logger.info(f"Replay progress: {self.current_step}/{len(self.historical_data)}")
            
            time.sleep(step_delay)
        
        logger.info("Replay completed")
        self.running = False
    
    def start(self, historical_data: Optional[pd.DataFrame] = None, replay: bool = False, step_delay: float = 0.1):
        """Start OPC UA server."""
        self.server.start()
        logger.info(f"OPC UA Server started at {self.endpoint}")
        
        self.setup_nodes()
        
        if historical_data is not None:
            self.load_historical_data(historical_data)
        
        if replay and self.historical_data is not None:
            self.running = True
            self.replay_thread = threading.Thread(
                target=self.replay_loop,
                args=(step_delay,),
                daemon=True
            )
            self.replay_thread.start()
            logger.info("Started replay thread")
    
    def stop(self):
        """Stop OPC UA server."""
        self.running = False
        if self.replay_thread:
            self.replay_thread.join(timeout=5)
        self.server.stop()
        logger.info("OPC UA Server stopped")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state from OPC UA nodes."""
        state = {}
        try:
            state['L1'] = self.nodes['L1'].get_value()
            state['V'] = self.nodes['V'].get_value()
            state['F1'] = self.nodes['F1'].get_value()
            state['F2'] = self.nodes['F2'].get_value()
            state['price_high'] = self.nodes['price_high'].get_value()
            state['price_normal'] = self.nodes['price_normal'].get_value()
            state['timestamp'] = self.nodes['timestamp'].get_value()
        except Exception as e:
            logger.error(f"Error reading state: {e}")
        return state


class DigitalTwinClient:
    """
    OPC UA Client for agents to connect to Digital Twin.
    """
    
    def __init__(self, endpoint: str = "opc.tcp://localhost:4840/freeopcua/server/"):
        if not OPCUA_AVAILABLE:
            raise ImportError("opcua library required")
        
        from opcua import Client
        self.client = Client(endpoint)
        self.endpoint = endpoint
        self.connected = False
        self.nodes_cache = {}
        logger.info(f"Digital Twin Client initialized for {endpoint}")
    
    def connect(self):
        """Connect to OPC UA server."""
        try:
            self.client.connect()
            self.connected = True
            logger.info(f"Connected to OPC UA server at {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from server."""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from OPC UA server")
    
    def browse_nodes(self, node_path: str = "Objects"):
        """Browse available nodes."""
        try:
            root = self.client.get_node(f"ns=0;i=85")  # Objects folder
            children = root.get_children()
            return [str(child) for child in children]
        except Exception as e:
            logger.error(f"Browse error: {e}")
            return []
    
    def read_variable(self, node_path: str) -> Any:
        """Read value from OPC UA variable."""
        try:
            if node_path not in self.nodes_cache:
                # Browse to find node (simplified - in production use proper path resolution)
                root = self.client.get_objects_node()
                self.nodes_cache[node_path] = root.get_child([
                    "2:WastewaterSystem", f"2:{node_path}"
                ])
            
            return self.nodes_cache[node_path].get_value()
        except Exception as e:
            logger.debug(f"Read error for {node_path}: {e}")
            return None
    
    def write_variable(self, node_path: str, value: Any):
        """Write value to OPC UA variable."""
        try:
            if node_path not in self.nodes_cache:
                root = self.client.get_objects_node()
                self.nodes_cache[node_path] = root.get_child([
                    "2:WastewaterSystem", f"2:{node_path}"
                ])
            
            self.nodes_cache[node_path].set_value(value)
            logger.debug(f"Wrote {value} to {node_path}")
        except Exception as e:
            logger.error(f"Write error for {node_path}: {e}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        state = {}
        variables = ['Tunnel/WaterLevel_L1_m', 'Tunnel/Volume_V_m3', 
                    'Tunnel/InflowF1_m3_15min', 'Tunnel/OutflowF2_m3h',
                    'Energy/PriceHigh_EUR_kWh', 'Energy/PriceNormal_EUR_kWh']
        
        for var in variables:
            val = self.read_variable(var)
            if val is not None:
                key = var.split('/')[-1].replace('_', ' ')
                state[key] = val
        
        return state
    
    def write_control_plan(self, target_F2: float, status: str = "Optimizing"):
        """Write control plan to server."""
        self.write_variable("Control/TargetF2_m3h", target_F2)
        self.write_variable("Control/OptimizationStatus", status)
