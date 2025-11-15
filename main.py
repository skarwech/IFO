"""Simplified main entrypoint for IPS using new modules.

Usage:
    python main.py Hackathon_HSY_data.xlsx "Volume of tunnel vs level Blominmäki.xlsx" --horizon 48

Positional arguments:
    data_file   Historical dataset (.xlsx or .csv)
    volume_file Volume-level table (.xlsx or .csv)
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.forecast import InflowForecaster
from src.model import VolumeModel, combine_price
from src.optimize import MPCOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ips-main")


def read_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        logger.error(f"Data file not found: {path}")
        sys.exit(1)
    if p.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(p, sheet_name='Taul1', skiprows=[1])
    return pd.read_csv(p, skiprows=[1])


def read_volume_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        logger.error(f"Volume file not found: {path}")
        sys.exit(1)
    if p.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(p, sheet_name='Taul1')
    return pd.read_csv(p)


def extract_inflow(df: pd.DataFrame) -> pd.Series:
    # Column name variants
    candidates = [c for c in df.columns if 'Inflow' in c or c.strip().lower() in ('f1', 'inflow')]
    if not candidates:
        logger.error("Inflow column not found (expected 'Inflow to tunnel F1')")
        sys.exit(1)
    col = candidates[0]
    s = df[col].astype(float)
    # If units m3/15min convert nothing; ensure later conversion
    return s


def extract_price(df: pd.DataFrame) -> pd.Series:
    try:
        return combine_price(df)
    except Exception:
        logger.warning("Price columns not found, using constant price 0.1 EUR/kWh")
        return pd.Series(np.full(len(df), 0.1))


def initial_volume(df: pd.DataFrame, vol_model: VolumeModel) -> float:
    vol_col = [c for c in df.columns if c.strip().lower() in ('v', 'volume')]
    lvl_col = [c for c in df.columns if 'Water level' in c or c.strip().lower() in ('l1', 'level')]
    if vol_col:
        return float(df[vol_col[0]].iloc[0])
    if lvl_col:
        return vol_model.level_to_volume(float(df[lvl_col[0]].iloc[0]))
    logger.info("No volume/level column, defaulting to level 2.0 m")
    return vol_model.level_to_volume(2.0)


def run_legacy(args):
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    df = read_data(args.data_file)
    vol_table = read_volume_table(args.volume_file)
    logger.info(f"Loaded data rows={len(df)} cols={len(df.columns)}")

    inflow_series = extract_inflow(df)
    price_series = extract_price(df)
    vol_model = VolumeModel(volume_table_path=args.volume_file)
    init_vol = initial_volume(df, vol_model)
    logger.info(f"Initial volume: {init_vol:.2f} m3")

    # Train forecaster on first 75% of data
    split = int(0.75 * len(inflow_series))
    train_series = inflow_series.iloc[:split]
    recent_series = inflow_series.iloc[-args.lookback:]
    forecaster = InflowForecaster(lookback_steps=args.lookback)
    logger.info("Training LSTM inflow forecaster...")
    forecaster.fit(train_series, epochs=args.epochs)
    horizon_steps = forecaster.horizon_from_hours(args.horizon)
    logger.info(f"Forecasting horizon steps: {horizon_steps}")
    inflow_forecast = forecaster.forecast(recent_series, horizon_steps)

    # Price forecast: simple persistence/mean blend
    price_window = price_series.iloc[-96:]
    mean_price = price_window.mean()
    last_price = price_window.iloc[-1]
    price_forecast = pd.Series([
        0.7 * last_price + 0.3 * mean_price for _ in range(horizon_steps)
    ])

    optimizer = MPCOptimizer(small_count=4, large_count=4)
    logger.info("Solving MILP optimization...")
    result = optimizer.optimize(init_vol, inflow_forecast, price_forecast)
    logger.info(f"Optimization status: {result['status']}")
    if result['status'] not in ('Optimal','FeasibleWithinTime'):
        logger.error("Optimization failed to find solution")
        sys.exit(2)

    schedule = result['schedule']
    schedule.to_csv(out_dir / 'schedule.csv', index=False)
    logger.info(f"Saved schedule to {out_dir / 'schedule.csv'}")
    
    # Display KPIs
    kpis = result.get('kpis', {})
    logger.info("\n" + "="*60)
    logger.info("KEY PERFORMANCE INDICATORS")
    logger.info("="*60)
    logger.info(f"Total Cost:          {kpis.get('total_cost_eur', 0):.2f} EUR")
    logger.info(f"Total Energy:        {kpis.get('total_energy_kwh', 0):.1f} kWh")
    logger.info(f"Average Power:       {kpis.get('avg_power_kw', 0):.1f} kW")
    logger.info(f"Peak Power:          {kpis.get('peak_power_kw', 0):.1f} kW")
    logger.info(f"Total Pumped:        {kpis.get('total_pumped_m3', 0):.0f} m³")
    logger.info(f"Average Level:       {kpis.get('avg_level_m', 0):.2f} m")
    logger.info(f"Level Range:         {kpis.get('min_level_m', 0):.2f} - {kpis.get('max_level_m', 0):.2f} m")
    logger.info(f"Avg Efficiency:      {kpis.get('avg_efficiency_pct', 0):.1f}%")
    logger.info(f"Ramp Violations:     {kpis.get('ramp_violations', 0)}")
    logger.info(f"Level Violations:    {kpis.get('level_violations', 0)}")
    logger.info("="*60)
    logger.info("\nSchedule Preview:")
    print(schedule.head(10))
    
    # Launch dashboard if requested
    if args.dashboard:
        from src.dashboard_simple import create_schedule_dashboard
        logger.info("\nLaunching interactive dashboard...")
        create_schedule_dashboard(schedule, kpis, port=args.port)


def run_hybrid(args):
    """Hybrid RL+MPC orchestrator (minimal offline demo)."""
    from src.agents import ForecastAgent, PlannerAgent, ExecutorAgent, SupervisorAgent
    
    df = read_data(args.data_file)
    inflow = extract_inflow(df).tolist()
    price = extract_price(df).tolist()
    vol_model = VolumeModel(volume_table_path=args.volume_file)
    V0 = initial_volume(df, vol_model)
    
    forecast = ForecastAgent(horizon_steps=args.horizon*4)
    forecast.train(pd.Series(inflow), epochs=args.epochs)
    planner = PlannerAgent(config={
        'horizon_steps': args.horizon*4,
        'solver_timeout': 10
    })
    executor = ExecutorAgent()
    supervisor = SupervisorAgent()

    # Minimal RL stub for demo confidence gating
    class RLPlannerStub:
        def act(self, state):
            # returns (action, confidence)
            return 0, 0.0
    planner_rl = RLPlannerStub()
    
    # One offline planning cycle: use RL if confident, else MPC
    # Seed recent inflows for forecasting without Digital Twin
    forecast.recent_inflows = inflow[-96:]
    f_decision = forecast.decide(forecast.perceive())
    f1 = f_decision['F1_forecast']
    pr = f_decision['price_forecast']
    # Dummy state for RL, confidence check
    import numpy as np
    state = np.zeros(80, dtype=np.float32)
    action, conf = planner_rl.act(state)
    if conf < 0.85:
        # Use existing PlannerAgent (MPC) path
        decision = planner.decide({'current_V': V0, 'F1_forecast': f1, 'price_forecast': pr})
        schedule = decision.get('schedule')
        if schedule is not None:
            schedule.to_csv(Path(args.output)/'hybrid_schedule.csv', index=False)
            print("Hybrid: MPC fallback used; schedule saved to results")
    else:
        print("Hybrid: RL selected (demo stub)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IPS / Hybrid RL+MPC')
    parser.add_argument('data_file', type=str, help='Historical data file (.xlsx/.csv)')
    parser.add_argument('volume_file', type=str, help='Volume table file (.xlsx/.csv)')
    parser.add_argument('--horizon', type=int, default=24, help='Horizon in hours')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='LSTM training epochs')
    parser.add_argument('--lookback', type=int, default=32, help='LSTM lookback steps (15-min)')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard after optimization')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--hybrid', action='store_true', help='Run Hybrid RL+MPC orchestrator')
    args = parser.parse_args()
    (run_hybrid if args.hybrid else run_legacy)(args)
