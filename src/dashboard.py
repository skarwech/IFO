"""
Plotly/Dash web dashboard for system visualization.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PumpDashboard:
    """
    Interactive web dashboard for pump control visualization.
    """
    
    def __init__(
        self,
        config: Dict,
        simulation_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize dashboard.
        
        Args:
            config: Configuration dictionary
            simulation_data: Optional simulation results to display
        """
        self.config = config
        self.simulation_data = simulation_data
        
        dashboard_config = config['dashboard']
        self.host = dashboard_config['host']
        self.port = dashboard_config['port']
        self.update_interval = dashboard_config['update_interval_seconds']
        
        # Create Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1(
                "Blominmäki WWTP - Intelligent Pump Scheduler",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}
            ),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3("Current Level", style={'color': '#3498db'}),
                    html.H2(id='current-level', children="-- m", style={'color': '#2c3e50'})
                ], className='summary-card'),
                
                html.Div([
                    html.H3("Active Pumps", style={'color': '#e74c3c'}),
                    html.H2(id='active-pumps', children="--", style={'color': '#2c3e50'})
                ], className='summary-card'),
                
                html.Div([
                    html.H3("Power", style={'color': '#f39c12'}),
                    html.H2(id='current-power', children="-- kW", style={'color': '#2c3e50'})
                ], className='summary-card'),
                
                html.Div([
                    html.H3("Cost Today", style={'color': '#27ae60'}),
                    html.H2(id='cost-today', children="-- EUR", style={'color': '#2c3e50'})
                ], className='summary-card'),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
            
            # Main plots
            dcc.Graph(id='level-flow-plot', style={'height': '400px'}),
            dcc.Graph(id='power-cost-plot', style={'height': '350px'}),
            dcc.Graph(id='pump-status-plot', style={'height': '300px'}),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # milliseconds
                n_intervals=0
            ),
            
            # Hidden div to store data
            html.Div(id='hidden-data', style={'display': 'none'})
            
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('current-level', 'children'),
             Output('active-pumps', 'children'),
             Output('current-power', 'children'),
             Output('cost-today', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_summary(n):
            """Update summary cards."""
            if self.simulation_data is None or len(self.simulation_data) == 0:
                return "-- m", "--", "-- kW", "-- EUR"
            
            # Get latest data
            latest = self.simulation_data.iloc[-1]
            
            level = f"{latest['L1']:.2f} m"
            
            active = int(latest['active_pumps']) if 'active_pumps' in latest else 0
            active_text = f"{active}"
            
            power = f"{latest['power_kw']:.1f} kW" if 'power_kw' in latest else "-- kW"
            
            # Calculate cost for last 24h
            if len(self.simulation_data) >= 96:  # 24h of 15min data
                recent = self.simulation_data.iloc[-96:]
                if 'power_kw' in recent.columns:
                    # Assume average price for now
                    avg_price = 0.05  # EUR/kWh
                    timestep_h = 0.25
                    cost = (recent['power_kw'] * timestep_h * avg_price).sum()
                    cost_text = f"{cost:.2f} EUR"
                else:
                    cost_text = "-- EUR"
            else:
                cost_text = "-- EUR"
            
            return level, active_text, power, cost_text
        
        @self.app.callback(
            Output('level-flow-plot', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_level_flow(n):
            """Update level and flow plots."""
            if self.simulation_data is None or len(self.simulation_data) == 0:
                return go.Figure()
            
            df = self.simulation_data.tail(384)  # Last 4 days (96 steps/day)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Water Level', 'Flow Rates'),
                vertical_spacing=0.12
            )
            
            # Level plot
            if 'L1' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df))),
                        y=df['L1'],
                        mode='lines',
                        name='Water Level',
                        line=dict(color='#3498db', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add constraint lines
                l1_min = self.config['system']['tunnel']['l1_min']
                l1_max = self.config['system']['tunnel']['l1_max']
                
                fig.add_hline(
                    y=l1_min, line_dash="dash", line_color="red",
                    annotation_text="Min Level", row=1, col=1
                )
                fig.add_hline(
                    y=l1_max, line_dash="dash", line_color="red",
                    annotation_text="Max Level", row=1, col=1
                )
            
            # Flow plot
            if 'F1_m3' in df.columns:
                # Convert to m³/h for visualization
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df))),
                        y=df['F1_m3'] * 4,  # Convert from m³/15min to m³/h
                        mode='lines',
                        name='Inflow F1',
                        line=dict(color='#9b59b6', width=1.5)
                    ),
                    row=2, col=1
                )
            
            if 'F2_m3h' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df))),
                        y=df['F2_m3h'],
                        mode='lines',
                        name='Outflow F2',
                        line=dict(color='#e74c3c', width=2)
                    ),
                    row=2, col=1
                )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Level (m)", row=1, col=1)
            fig.update_yaxes(title_text="Flow (m³/h)", row=2, col=1)
            
            fig.update_layout(
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('power-cost-plot', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_power_cost(n):
            """Update power and cost plots."""
            if self.simulation_data is None or len(self.simulation_data) == 0:
                return go.Figure()
            
            df = self.simulation_data.tail(384)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Power Consumption', 'System Head'),
                horizontal_spacing=0.12
            )
            
            # Power plot
            if 'power_kw' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df))),
                        y=df['power_kw'],
                        mode='lines',
                        name='Power',
                        line=dict(color='#f39c12', width=2),
                        fill='tozeroy'
                    ),
                    row=1, col=1
                )
            
            # System head plot
            if 'system_head_m' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df))),
                        y=df['system_head_m'],
                        mode='lines',
                        name='Head',
                        line=dict(color='#16a085', width=2)
                    ),
                    row=1, col=2
                )
            
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Head (m)", row=1, col=2)
            
            fig.update_layout(
                height=350,
                showlegend=False,
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('pump-status-plot', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_pump_status(n):
            """Update pump status heatmap."""
            if self.simulation_data is None or len(self.simulation_data) == 0:
                return go.Figure()
            
            df = self.simulation_data.tail(384)
            
            # Find pump status columns
            pump_on_cols = [col for col in df.columns if col.startswith('pump_') and col.endswith('_on')]
            
            if not pump_on_cols:
                return go.Figure()
            
            # Create matrix for heatmap
            num_pumps = len(pump_on_cols)
            pump_matrix = np.zeros((num_pumps, len(df)))
            
            for i, col in enumerate(pump_on_cols):
                pump_matrix[i, :] = df[col].values
            
            fig = go.Figure(data=go.Heatmap(
                z=pump_matrix,
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df))),
                y=[f"Pump {i}" for i in range(num_pumps)],
                colorscale=[[0, '#ecf0f1'], [1, '#27ae60']],
                showscale=False
            ))
            
            fig.update_layout(
                title="Pump Status (On/Off)",
                height=300,
                xaxis_title="Time",
                yaxis_title="Pump",
                template='plotly_white'
            )
            
            return fig
    
    def update_data(self, new_data: pd.DataFrame):
        """
        Update simulation data.
        
        Args:
            new_data: New simulation data
        """
        self.simulation_data = new_data
    
    def run(self, debug: bool = False):
        """
        Run the dashboard server.
        
        Args:
            debug: Run in debug mode
        """
        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=debug)


def create_static_report(
    simulation_data: pd.DataFrame,
    config: Dict,
    output_file: str = "report.html"
):
    """
    Create static HTML report with plots.
    
    Args:
        simulation_data: Simulation results
        config: Configuration
        output_file: Output HTML file path
    """
    from plotly.subplots import make_subplots
    import plotly.io as pio
    
    # Create comprehensive figure
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Water Level', 'Volume',
            'Inflow vs Outflow', 'Power Consumption',
            'System Head', 'Active Pumps',
            'Pump Status Heatmap', 'Cost Accumulation'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"colspan": 2}, None],
        ]
    )
    
    x_axis = simulation_data.index if isinstance(simulation_data.index, pd.DatetimeIndex) else list(range(len(simulation_data)))
    
    # 1. Water level
    if 'L1' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['L1'], mode='lines', name='Level'),
            row=1, col=1
        )
    
    # 2. Volume
    if 'V' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['V'], mode='lines', name='Volume'),
            row=1, col=2
        )
    
    # 3. Flows
    if 'F1_m3' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['F1_m3']*4, mode='lines', name='Inflow'),
            row=2, col=1
        )
    if 'F2_m3h' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['F2_m3h'], mode='lines', name='Outflow'),
            row=2, col=1
        )
    
    # 4. Power
    if 'power_kw' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['power_kw'], mode='lines', 
                      name='Power', fill='tozeroy'),
            row=2, col=2
        )
    
    # 5. System head
    if 'system_head_m' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['system_head_m'], mode='lines', name='Head'),
            row=3, col=1
        )
    
    # 6. Active pumps
    if 'active_pumps' in simulation_data.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=simulation_data['active_pumps'], mode='lines', name='Active'),
            row=3, col=2
        )
    
    # 7. Pump status heatmap
    pump_on_cols = [col for col in simulation_data.columns if col.startswith('pump_') and col.endswith('_on')]
    if pump_on_cols:
        pump_matrix = np.array([simulation_data[col].values for col in pump_on_cols])
        fig.add_trace(
            go.Heatmap(
                z=pump_matrix,
                x=x_axis,
                y=[f"P{i}" for i in range(len(pump_on_cols))],
                colorscale='Greens'
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1400,
        title_text="Pump Control Simulation Report",
        showlegend=True,
        template='plotly_white'
    )
    
    # Save to HTML
    pio.write_html(fig, output_file)
    logger.info(f"Static report saved to {output_file}")
