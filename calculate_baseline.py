"""
Calculate baseline cost from historical HSY data.
This represents the "business-as-usual" operation cost that our AI system will beat.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_baseline_cost(data_path: str, steps: int = 96) -> dict:
    """
    Calculate baseline cost from historical data.
    
    Args:
        data_path: Path to CSV/Excel with historical data
        steps: Number of steps to analyze (default 96 = 24 hours)
    
    Returns:
        Dictionary with baseline metrics
    """
    # Load data
    path = Path(data_path)
    if path.suffix == '.csv':
        df = pd.read_csv(path, skiprows=[1])
    else:
        df = pd.read_excel(path, sheet_name='Taul1', skiprows=[1])
    
    # Limit to requested steps
    df = df.head(steps)
    
    # Identify columns
    power_cols = [c for c in df.columns if 'Power' in c or 'power' in c]
    price_high_col = [c for c in df.columns if 'Electricity price 1' in c or 'high' in c][0]
    price_normal_col = [c for c in df.columns if 'Electricity price 2' in c or 'normal' in c][0]
    
    # Total power per timestep (sum of all pump powers)
    if power_cols:
        total_power = df[power_cols].fillna(0.0).sum(axis=1)
    else:
        # Fallback: estimate from inflow F1 assuming baseline pumps continuously
        # Typical baseline: pump at average inflow rate with 50% efficiency
        # Hydraulic power = ρ × g × Q × H, where H ≈ 25m, Q from F1
        # Assuming F1 in m³/15min, convert to m³/h: F1 × 4
        # Power (kW) ≈ 1000 kg/m³ × 9.81 m/s² × (Q m³/h / 3600) × H / efficiency
        # Simplified: P ≈ 0.07 kW per (m³/h) for H=25m, η=0.5
        flow_col = [c for c in df.columns if 'Inflow' in c or 'F1' in c]
        if flow_col:
            F1_m3_per_15min = df[flow_col[0]].fillna(1000.0)
            F1_m3h = F1_m3_per_15min * 4.0
            total_power = F1_m3h * 0.07  # Rough baseline power estimate
        else:
            # Last resort: assume constant 2000 m³/h baseline flow
            total_power = pd.Series([2000 * 0.07] * len(df))
    
    # Use the higher price (conservative baseline)
    price = df[[price_high_col, price_normal_col]].max(axis=1)
    
    # Energy cost per timestep (kW * EUR/kWh * 0.25h)
    timestep_hours = 0.25
    cost_per_step = total_power * price * timestep_hours
    
    # Aggregate metrics
    total_cost = float(cost_per_step.sum())
    total_energy = float((total_power * timestep_hours).sum())
    avg_power = float(total_power.mean())
    peak_power = float(total_power.max())
    
    # Cost breakdown by time of day
    if 'Time stamp' in df.columns or 'Timestamp' in df.columns:
        ts_col = 'Time stamp' if 'Time stamp' in df.columns else 'Timestamp'
        # Assume 15-min intervals starting from midnight
        hours = np.arange(len(df)) * 0.25
        df['hour'] = hours % 24
        
        night_cost = float(cost_per_step[df['hour'] < 6].sum())
        day_cost = float(cost_per_step[(df['hour'] >= 6) & (df['hour'] < 22)].sum())
        evening_cost = float(cost_per_step[df['hour'] >= 22].sum())
    else:
        night_cost = day_cost = evening_cost = 0.0
    
    results = {
        'total_cost_eur': total_cost,
        'total_energy_kwh': total_energy,
        'avg_power_kw': avg_power,
        'peak_power_kw': peak_power,
        'night_cost_eur': night_cost,
        'day_cost_eur': day_cost,
        'evening_cost_eur': evening_cost,
        'num_steps': len(df),
        'duration_hours': len(df) * timestep_hours
    }
    
    return results


def print_baseline_report(results: dict):
    """Print formatted baseline report."""
    print("\n" + "="*60)
    print("BASELINE COST CALCULATION (Historical Operation)")
    print("="*60)
    print(f"Analysis Period: {results['num_steps']} steps ({results['duration_hours']:.1f} hours)")
    print(f"\nTotal Energy Cost: {results['total_cost_eur']:.2f} EUR")
    print(f"Total Energy Consumption: {results['total_energy_kwh']:.2f} kWh")
    print(f"Average Power: {results['avg_power_kw']:.1f} kW")
    print(f"Peak Power: {results['peak_power_kw']:.1f} kW")
    
    if results['night_cost_eur'] > 0:
        print(f"\nCost Breakdown:")
        print(f"  Night (00:00-06:00): {results['night_cost_eur']:.2f} EUR")
        print(f"  Day (06:00-22:00):   {results['day_cost_eur']:.2f} EUR")
        print(f"  Evening (22:00-00:00): {results['evening_cost_eur']:.2f} EUR")
    
    print("\nThis baseline will be used for savings calculation:")
    print(f"  Savings (%) = (Baseline - Optimized) / Baseline × 100")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate baseline cost from historical data')
    parser.add_argument('--data', type=str, default='data/test_data.csv',
                       help='Path to historical data CSV/Excel')
    parser.add_argument('--steps', type=int, default=96,
                       help='Number of steps to analyze (default 96 = 24h)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: save results to JSON file')
    
    args = parser.parse_args()
    
    # Calculate
    results = calculate_baseline_cost(args.data, args.steps)
    
    # Print report
    print_baseline_report(results)
    
    # Save if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
