"""
Simplified dashboard adapter for new schedule format.
"""

import dash
from dash import dcc, html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_schedule_dashboard(schedule_df: pd.DataFrame, kpis: dict, port: int = 8050):
    """
    Create and launch a simple dashboard from optimization schedule.
    
    Args:
        schedule_df: Schedule DataFrame with columns from MPCOptimizer
        kpis: KPI dictionary from optimization result
        port: Port to run dashboard on
    """
    app = dash.Dash(__name__)
    
    # Create plots
    fig_level_flow = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Tunnel Level', 'Pumped Flow'),
        vertical_spacing=0.12
    )
    
    fig_level_flow.add_trace(
        go.Scatter(
            x=schedule_df['t'],
            y=schedule_df['level_m'],
            name='Level',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )
    
    fig_level_flow.add_trace(
        go.Scatter(
            x=schedule_df['t'],
            y=schedule_df['flow_m3h'],
            name='Flow',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    fig_level_flow.update_xaxes(title_text="Time Step (15 min)", row=2, col=1)
    fig_level_flow.update_yaxes(title_text="Level (m)", row=1, col=1)
    fig_level_flow.update_yaxes(title_text="Flow (m³/h)", row=2, col=1)
    fig_level_flow.update_layout(height=600, showlegend=True)
    
    # Power and cost
    fig_power_cost = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Power Consumption', 'Cumulative Cost'),
        vertical_spacing=0.12
    )
    
    fig_power_cost.add_trace(
        go.Bar(
            x=schedule_df['t'],
            y=schedule_df['power_kw'],
            name='Power',
            marker_color='orange'
        ),
        row=1, col=1
    )
    
    cumulative_cost = schedule_df['cost_eur'].cumsum()
    fig_power_cost.add_trace(
        go.Scatter(
            x=schedule_df['t'],
            y=cumulative_cost,
            name='Cumulative Cost',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    fig_power_cost.update_xaxes(title_text="Time Step (15 min)", row=2, col=1)
    fig_power_cost.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig_power_cost.update_yaxes(title_text="Cost (EUR)", row=2, col=1)
    fig_power_cost.update_layout(height=600, showlegend=True)
    
    # Pump activation heatmap
    pump_data = []
    for idx, row in schedule_df.iterrows():
        pump_data.append({
            't': row['t'],
            'Small Pumps': row['small_active'],
            'Large Pumps': row['large_active'],
        })
    pump_df = pd.DataFrame(pump_data)
    
    fig_pumps = go.Figure()
    fig_pumps.add_trace(go.Bar(
        x=pump_df['t'],
        y=pump_df['Small Pumps'],
        name='Small Pumps',
        marker_color='lightblue'
    ))
    fig_pumps.add_trace(go.Bar(
        x=pump_df['t'],
        y=pump_df['Large Pumps'],
        name='Large Pumps',
        marker_color='darkblue'
    ))
    fig_pumps.update_layout(
        barmode='stack',
        title='Active Pumps Over Time',
        xaxis_title='Time Step (15 min)',
        yaxis_title='Number of Active Pumps',
        height=400
    )
    
    # Layout
    app.layout = html.Div([
        html.H1("IPS Optimization Results Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                html.H3("Total Cost", style={'color': '#27ae60'}),
                html.H2(f"{kpis.get('total_cost_eur', 0):.2f} EUR", style={'color': '#2c3e50'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Total Energy", style={'color': '#e74c3c'}),
                html.H2(f"{kpis.get('total_energy_kwh', 0):.1f} kWh", style={'color': '#2c3e50'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Avg Power", style={'color': '#f39c12'}),
                html.H2(f"{kpis.get('avg_power_kw', 0):.1f} kW", style={'color': '#2c3e50'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Avg Efficiency", style={'color': '#3498db'}),
                html.H2(f"{kpis.get('avg_efficiency_pct', 0):.1f}%", style={'color': '#2c3e50'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Violations", style={'color': '#9b59b6'}),
                html.H2(f"{kpis.get('ramp_violations', 0) + kpis.get('level_violations', 0)}", style={'color': '#2c3e50'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
        ], style={'marginBottom': '30px'}),
        
        dcc.Graph(figure=fig_level_flow),
        dcc.Graph(figure=fig_power_cost),
        dcc.Graph(figure=fig_pumps),
        
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
    
    logger.info(f"Starting dashboard on http://127.0.0.1:{port}")
    app.run_server(debug=False, port=port)
