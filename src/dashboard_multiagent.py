"""
Multi-Agent Dashboard for Wastewater System Visualization
Real-time monitoring of agent states and system performance.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

logger = logging.getLogger(__name__)


def create_tunnel_cross_section(water_level: float, max_level: float = 13.0) -> go.Figure:
    """
    Create cross-section view of tunnel with water level.
    """
    fig = go.Figure()
    
    # Tunnel outline (simplified rectangular)
    tunnel_width = 10  # meters
    tunnel_height = max_level
    
    fig.add_trace(go.Scatter(
        x=[0, 0, tunnel_width, tunnel_width, 0],
        y=[0, tunnel_height, tunnel_height, 0, 0],
        mode='lines',
        line=dict(color='gray', width=3),
        fill=None,
        name='Tunnel'
    ))
    
    # Water level
    fig.add_trace(go.Scatter(
        x=[0, tunnel_width, tunnel_width, 0, 0],
        y=[0, 0, water_level, water_level, 0],
        mode='lines',
        line=dict(color='blue', width=2),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.3)',
        name='Water'
    ))
    
    # Add level annotation
    fig.add_annotation(
        x=tunnel_width/2,
        y=water_level,
        text=f"L1 = {water_level:.2f} m",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40
    )
    
    # Add constraint lines
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                  annotation_text="Min Level", annotation_position="right")
    fig.add_hline(y=13.0, line_dash="dash", line_color="red",
                  annotation_text="Max Level", annotation_position="right")
    
    fig.update_layout(
        title="Tunnel Cross-Section",
        xaxis_title="Width (m)",
        yaxis_title="Height (m)",
        showlegend=True,
        height=400,
        yaxis=dict(range=[0, tunnel_height + 1])
    )
    
    return fig


def create_agent_status_chart(agent_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create timeline chart showing agent activity.
    """
    fig = go.Figure()
    
    agents = ['ForecastAgent', 'PlannerAgent', 'ExecutorAgent', 'SupervisorAgent']
    colors = {'ForecastAgent': 'blue', 'PlannerAgent': 'green', 
              'ExecutorAgent': 'orange', 'SupervisorAgent': 'purple'}
    
    for agent in agents:
        # Filter history for this agent
        agent_events = [h for h in agent_history if h.get('agent') == agent]
        
        if agent_events:
            steps = [e.get('step', i) for i, e in enumerate(agent_events)]
            success = [1 if e.get('success', False) else 0 for e in agent_events]
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=success,
                mode='markers+lines',
                name=agent,
                marker=dict(size=8, color=colors.get(agent, 'gray'))
            ))
    
    fig.update_layout(
        title="Agent Activity Timeline",
        xaxis_title="Simulation Step",
        yaxis_title="Success (1=Yes, 0=No)",
        height=300,
        yaxis=dict(range=[-0.1, 1.1])
    )
    
    return fig


def create_cost_comparison_chart(results: pd.DataFrame, baseline_cost: float = None) -> go.Figure:
    """
    Create chart comparing optimized vs baseline costs.
    """
    fig = go.Figure()
    
    if 'cost' in results.columns:
        cumulative_cost = results['cost'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=results['step'],
            y=cumulative_cost,
            mode='lines',
            name='Optimized (Multi-Agent)',
            line=dict(color='green', width=3)
        ))
        
        if baseline_cost is not None:
            # Scale baseline total to simulation length (assuming provided for 24h = 96 steps)
            steps = len(results)
            scaled_baseline_total = float(baseline_cost) * (steps / 96.0)
            # Assume baseline accrues at a constant rate over steps
            baseline_cumulative = [scaled_baseline_total * (i+1) / steps for i in range(steps)]
            
            fig.add_trace(go.Scatter(
                x=results['step'],
                y=baseline_cumulative,
                mode='lines',
                name='Baseline (Historical)',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            # Calculate savings
            final_optimized = float(cumulative_cost.iloc[-1])
            final_baseline = float(baseline_cumulative[-1])
            savings_pct = (final_baseline - final_optimized) / final_baseline * 100
            
            fig.add_annotation(
                x=len(results) * 0.7,
                y=max(final_baseline, final_optimized) * 0.5,
                text=f"Savings: {savings_pct:.1f}%<br>({final_baseline - final_optimized:.2f} EUR)",
                showarrow=True,
                arrowhead=2,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2
            )
    
    fig.update_layout(
        title="Cumulative Energy Cost Comparison",
        xaxis_title="Simulation Step",
        yaxis_title="Cumulative Cost (EUR)",
        height=400
    )
    
    return fig


def create_volume_level_chart(results: pd.DataFrame) -> go.Figure:
    """
    Create dual-axis chart for volume and level.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if 'V' in results.columns:
        fig.add_trace(
            go.Scatter(x=results['step'], y=results['V'], 
                      name='Volume', line=dict(color='blue')),
            secondary_y=False
        )
    
    if 'L1' in results.columns:
        fig.add_trace(
            go.Scatter(x=results['step'], y=results['L1'],
                      name='Water Level', line=dict(color='cyan')),
            secondary_y=True
        )
    
    # Add constraint zones
    fig.add_hrect(y0=0, y1=5000, line_width=0, fillcolor="red", opacity=0.1,
                  secondary_y=False, annotation_text="Low Volume")
    fig.add_hrect(y0=35000, y1=40000, line_width=0, fillcolor="red", opacity=0.1,
                  secondary_y=False, annotation_text="High Volume")
    
    fig.update_xaxes(title_text="Simulation Step")
    fig.update_yaxes(title_text="Volume (m³)", secondary_y=False)
    fig.update_yaxes(title_text="Water Level (m)", secondary_y=True)
    
    fig.update_layout(title="Tunnel Volume and Water Level", height=400)
    
    return fig


def create_flow_chart(results: pd.DataFrame) -> go.Figure:
    """
    Create chart showing inflow (F1) and outflow (F2).
    """
    fig = go.Figure()
    
    if 'F1' in results.columns:
        fig.add_trace(go.Scatter(
            x=results['step'],
            y=results['F1'],
            name='Inflow F1',
            line=dict(color='blue', width=2)
        ))
    
    if 'F2' in results.columns:
        fig.add_trace(go.Scatter(
            x=results['step'],
            y=results['F2'],
            name='Outflow F2 (Planned)',
            line=dict(color='green', width=2)
        ))
    
    fig.update_layout(
        title="Flow Rates",
        xaxis_title="Simulation Step",
        yaxis_title="Flow (m³/h or m³/15min)",
        height=400
    )
    
    return fig


def create_multi_agent_dashboard(results: pd.DataFrame, 
                                 agent_history: List[Dict[str, Any]] = None,
                                 baseline_cost: float = None,
                                 current_level: float = None) -> dash.Dash:
    """
    Create comprehensive Dash dashboard for multi-agent system.
    
    Args:
        results: Simulation results DataFrame
        agent_history: List of agent step results
        baseline_cost: Baseline cost for comparison
        current_level: Current water level for cross-section
    
    Returns:
        Dash application object
    """
    app = dash.Dash(__name__)
    
    # Create figures
    cost_fig = create_cost_comparison_chart(results, baseline_cost)
    volume_fig = create_volume_level_chart(results)
    flow_fig = create_flow_chart(results)
    
    if current_level is None and 'L1' in results.columns:
        current_level = results['L1'].iloc[-1]
    
    cross_section_fig = create_tunnel_cross_section(current_level or 6.0)
    
    if agent_history:
        agent_status_fig = create_agent_status_chart(agent_history)
    else:
        agent_status_fig = go.Figure()
    
    # Create layout
    app.layout = html.Div([
        html.H1("Multi-Agent Wastewater System Dashboard",
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=cross_section_fig)
            ], className='six columns'),
            
            html.Div([
                html.H3("System Status", style={'textAlign': 'center'}),
                html.Div(id='status-display', children=[
                    html.P(f"Current Volume: {results['V'].iloc[-1]:.0f} m³" if 'V' in results.columns else "N/A"),
                    html.P(f"Current Level: {results['L1'].iloc[-1]:.2f} m" if 'L1' in results.columns else "N/A"),
                    html.P(f"Total Steps: {len(results)}"),
                    html.P(f"Alerts: {results['alerts'].sum() if 'alerts' in results.columns else 0}")
                ], style={'fontSize': '18px', 'padding': '20px'})
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            dcc.Graph(figure=cost_fig)
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=volume_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=flow_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            dcc.Graph(figure=agent_status_fig)
        ], className='row'),
        
        html.Hr(),
        
        html.Div([
            html.H3("Performance Metrics", style={'textAlign': 'center'}),
            html.Div(id='metrics-display', children=[
                html.Table([
                    html.Tr([html.Th("Metric"), html.Th("Value")]),
                    html.Tr([html.Td("Final Volume"), html.Td(f"{results['V'].iloc[-1]:.0f} m³" if 'V' in results.columns else "N/A")]),
                    html.Tr([html.Td("Final Level"), html.Td(f"{results['L1'].iloc[-1]:.2f} m" if 'L1' in results.columns else "N/A")]),
                    html.Tr([html.Td("Total Cost"), html.Td(f"{results['cost'].sum():.2f} EUR" if 'cost' in results.columns else "N/A")]),
                    html.Tr([html.Td("Avg F1"), html.Td(f"{results['F1'].mean():.0f} m³/15min" if 'F1' in results.columns else "N/A")]),
                    html.Tr([html.Td("Avg F2"), html.Td(f"{results['F2'].mean():.0f} m³/h" if 'F2' in results.columns else "N/A")]),
                ], style={'width': '100%', 'border': '1px solid black', 'textAlign': 'center'})
            ])
        ], className='row', style={'padding': '20px'})
    ])
    
    return app


def generate_static_report(results: pd.DataFrame, 
                          baseline_cost: float = None,
                          output_path: str = "results/multi_agent_report.html") -> str:
    """
    Generate static HTML report with all charts.
    
    Args:
        results: Simulation results
        baseline_cost: Baseline cost for comparison
        output_path: Path to save HTML report
    
    Returns:
        Path to generated report
    """
    from pathlib import Path
    
    # Create figures
    cost_fig = create_cost_comparison_chart(results, baseline_cost)
    volume_fig = create_volume_level_chart(results)
    flow_fig = create_flow_chart(results)
    
    current_level = results['L1'].iloc[-1] if 'L1' in results.columns else 6.0
    cross_section_fig = create_tunnel_cross_section(current_level)
    
    # Compute baseline metrics (scale to simulation length if provided)
    steps = len(results)
    scaled_baseline = None
    savings_text = "N/A"
    total_cost_val = float(results['cost'].sum()) if 'cost' in results.columns else 0.0
    if baseline_cost is not None and steps > 0:
        scaled_baseline = float(baseline_cost) * (steps / 96.0)
        if scaled_baseline > 0:
            savings_pct = (scaled_baseline - total_cost_val) / scaled_baseline * 100.0
            savings_text = f"{savings_pct:.1f}%"
        else:
            savings_text = "0.0%"

    # Combine into single HTML
    html_content = f"""
    <html>
    <head>
        <title>Multi-Agent System Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ text-align: center; color: #2c3e50; }}
            .chart {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Multi-Agent Wastewater System Report</h1>
        
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Simulation Steps</td><td>{len(results)}</td></tr>
            <tr><td>Final Volume</td><td>{results['V'].iloc[-1]:.0f} m³</td></tr>
            <tr><td>Final Level</td><td>{results['L1'].iloc[-1]:.2f} m</td></tr>
            <tr><td>Total Cost</td><td>{total_cost_val:.2f} EUR</td></tr>
            <tr><td>Baseline (scaled)</td><td>{(scaled_baseline if scaled_baseline is not None else 'N/A')}</td></tr>
            <tr><td>Savings</td><td>{savings_text}</td></tr>
        </table>
        
        <div class="chart">
            {cost_fig.to_html(include_plotlyjs='cdn')}
        </div>
        
        <div class="chart">
            {volume_fig.to_html(include_plotlyjs=False)}
        </div>
        
        <div class="chart">
            {flow_fig.to_html(include_plotlyjs=False)}
        </div>
        
        <div class="chart">
            {cross_section_fig.to_html(include_plotlyjs=False)}
        </div>
        
        <p style="text-align: center; margin-top: 40px; color: #7f8c8d;">
            Generated by Multi-Agent Wastewater System | Junction 2025
        </p>
    </body>
    </html>
    """
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Static report saved to {output_file}")
    return str(output_file)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample results
    steps = 96
    results = pd.DataFrame({
        'step': range(steps),
        'V': 15000 + 5000 * np.sin(np.linspace(0, 4*np.pi, steps)),
        'L1': 6 + 2 * np.sin(np.linspace(0, 4*np.pi, steps)),
        'F1': 1000 + 200 * np.random.randn(steps),
        'F2': 2000 + 100 * np.random.randn(steps),
        'cost': 10 + 2 * np.random.rand(steps),
        'alerts': np.random.randint(0, 2, steps)
    })
    
    # Generate static report
    report_path = generate_static_report(results, baseline_cost=1200.0)
    print(f"Report generated: {report_path}")
    
    # Launch dashboard
    app = create_multi_agent_dashboard(results, baseline_cost=1200.0)
    app.run_server(debug=True, port=8051)
