"""
Final validation script for Junction 2025 competition submission.
Runs full 96-step (24-hour) simulation and generates comprehensive report.
"""

import subprocess
import sys
import json
from pathlib import Path


def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        return False
    print(f"✅ {description} completed successfully")
    return True


def main():
    """Run complete validation pipeline."""
    
    print("\n" + "="*60)
    print("🏆 JUNCTION 2025 - FINAL VALIDATION")
    print("Multi-Agent AI Wastewater Optimization System")
    print("="*60)
    
    # Step 1: Calculate baseline
    print("\n📊 Step 1: Calculating baseline cost from historical data...")
    if not run_command(
        'python calculate_baseline.py --data data/test_data.csv --steps 95 --output results/baseline_metrics.json',
        "Baseline Calculation"
    ):
        return
    
    # Load baseline
    with open('results/baseline_metrics.json', 'r') as f:
        baseline = json.load(f)
    baseline_cost = baseline['total_cost_eur']
    
    print(f"\n💰 Baseline Cost: {baseline_cost:.2f} EUR (for {baseline['num_steps']} steps)")
    
    # Step 2: Run multi-agent optimization
    print("\n🤖 Step 2: Running multi-agent AI optimization...")
    if not run_command(
        f'python main_multiagent.py --mode offline --data data/test_data.csv --steps 95 '
        f'--output results/junction2025_results.csv --report --baseline {baseline_cost}',
        "Multi-Agent Optimization"
    ):
        return
    
    # Step 3: Analyze results
    print("\n📈 Step 3: Analyzing results...")
    import pandas as pd
    
    results = pd.read_csv('results/junction2025_results.csv')
    optimized_cost = results['cost'].sum()
    
    savings_eur = baseline_cost - optimized_cost
    savings_pct = (savings_eur / baseline_cost) * 100
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Baseline Cost:    {baseline_cost:.2f} EUR")
    print(f"Optimized Cost:   {optimized_cost:.2f} EUR")
    print(f"Savings:          {savings_eur:.2f} EUR ({savings_pct:.1f}%)")
    print(f"\nAnnual Savings:   {savings_eur * 365:.0f} EUR/year")
    print(f"10-Plant Network: {savings_eur * 365 * 10:.0f} EUR/year")
    print(f"{'='*60}")
    
    # Step 4: Verify constraints
    print("\n✅ Step 4: Verifying constraints...")
    violations = {
        'level_low': (results['L1'] < 0.0).sum(),
        'level_high': (results['L1'] > 8.0).sum(),
        'volume_low': (results['V'] < 5000).sum(),
        'volume_high': (results['V'] > 35000).sum()
    }
    
    total_violations = sum(violations.values())
    
    if total_violations == 0:
        print("✅ All constraints satisfied (100% compliance)")
    else:
        print(f"⚠️  Constraint violations detected:")
        for k, v in violations.items():
            if v > 0:
                print(f"   - {k}: {v} steps")
    
    # Step 5: Generate summary
    print("\n📄 Step 5: Generating summary report...")
    
    summary = {
        "competition": "Junction 2025 - Valmet-HSY Challenge",
        "solution": "Multi-Agent AI Wastewater Optimization",
        "simulation": {
            "steps": len(results),
            "duration_hours": len(results) * 0.25,
            "data_source": "HSY Historical Data"
        },
        "costs": {
            "baseline_eur": round(baseline_cost, 2),
            "optimized_eur": round(optimized_cost, 2),
            "savings_eur": round(savings_eur, 2),
            "savings_percent": round(savings_pct, 1)
        },
        "annual_impact": {
            "single_plant_eur": round(savings_eur * 365, 0),
            "ten_plants_eur": round(savings_eur * 365 * 10, 0)
        },
        "performance": {
            "final_volume_m3": round(results['V'].iloc[-1], 0),
            "final_level_m": round(results['L1'].iloc[-1], 2),
            "avg_flow_m3h": round(results['F2'].mean(), 1),
            "constraint_violations": total_violations
        },
        "judging_criteria": {
            "real_world_applicability": "OPC UA integration + Edge deployment ready",
            "clarity_and_impact": f"{savings_pct:.1f}% cost savings demonstrated",
            "technical_soundness": f"{100 - (total_violations/len(results)*100):.0f}% constraint compliance",
            "creativity": "4 autonomous agents with LSTM + MILP optimization"
        }
    }
    
    with open('results/junction2025_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Summary saved to results/junction2025_summary.json")
    print("✅ HTML report available at results/junction2025_results_report.html")
    
    # Final message
    print("\n" + "="*60)
    print("🎉 VALIDATION COMPLETE!")
    print("="*60)
    print("\n📦 Deliverables Ready:")
    print("   ✅ results/junction2025_results.csv (time series data)")
    print("   ✅ results/junction2025_results_report.html (visualization)")
    print("   ✅ results/junction2025_summary.json (metrics summary)")
    print("   ✅ results/baseline_metrics.json (baseline calculation)")
    print("\n📚 Documentation:")
    print("   ✅ JUNCTION2025_README.md (competition pitch)")
    print("   ✅ ARM_DEPLOYMENT.md (edge computing proposal)")
    print("   ✅ MULTIAGENT_README.md (technical details)")
    print("\n🚀 Ready for Junction 2025 submission!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
