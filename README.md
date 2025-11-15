# Intelligent Pump Scheduler (IPS)
## Multi-Agent AI + MPC + Digital Twin for Wastewater Pump Optimization

Blominmäki WWTP (HSY Finland) • Junction 2025 submission

—

Table of Contents
- Overview
- Key Features & Performance
- System Architecture
- Mathematical Formulation (Dynamics, Pumps, MPC)
- Multi-Agent Implementation
- Digital Twin (OPC UA)
- Installation
- Quick Start (Legacy, Multi-Agent, Hybrid)
- Configuration
- Data Requirements
- Optimization & KPIs
- Dashboard & Reports
- Validation & Baseline
- Arm Edge Deployment
- File Guide & Project Structure
- Troubleshooting
- Judging Alignment & Submission Checklist
- Roadmap & Enhancements
- License, Authors, Contact

—

Overview
- Purpose: Minimize energy cost while respecting constraints using LSTM forecasting, MILP/MPC scheduling, and a Digital Twin.
- Outcome: 10–30% cost savings typical; stable operation with drift prevention and smoothing; deployable on edge Arm hardware.

Key Features & Performance
- Multi-agent architecture: Forecast, Planner (MPC), Executor, Supervisor.
- Enhanced pump models: Digitized Grundfos curves; affinity laws Q∝f, H∝f², P∝f³.
- MPC details: Discrete frequencies {48, 49, 50} Hz; terminal balance; average-flow and smoothing penalties; time-bounded CBC solver that accepts feasible solutions.
- Forecasting: PyTorch LSTM, 32-step lookback, 96-step horizon; persistence fallback.
- Dashboard: Plotly/Dash interactive and static HTML reports.
- Typical results (24h): 10–15% cost savings; 0 violations; smooth operation; peak shaving and BEP-centric behavior.

System Architecture
- Multi-Agent layers: ForecastAgent (LSTM) → PlannerAgent (MILP/MPC) → ExecutorAgent (physics) → SupervisorAgent (KPIs/alerts).
- Digital Twin: OPC UA server/client; historical replay; read/write of L1, V, F1, F2, prices, plan targets.
- Decision loop: Perceive → Decide → Act; receding horizon every 15 minutes.

Mathematical Formulation (condensed)
- Dynamics: V(t+1) = V(t) + F1(t) − F2(t)·Δt; Δt=0.25 h; L1=f⁻¹(V).
- Pump laws: Q∝f, H∝f², P∝f³ with digitized curves for accuracy.
- MPC variables: per-pump on/off; one-hot frequencies; V, F2, P.
- Objective: Σ P(t)·price(t)·Δt + smoothing + penalties (terminal/average-flow/constancy).
- Constraints: Volume/level limits; ramp limit; min up/down; ≥1 pump active; terminal balance; average flow ≥ inflow.
- Solver: CBC with time limit; treat FeasibleWithinTime as success for responsiveness.

Multi-Agent Implementation
- ForecastAgent: trains on historical inflow; outputs F1 and price forecast; statistical fallback; integrates with Digital Twin.
- PlannerAgent: slices horizon per config; runs MPC with time limit; returns schedule + KPIs; writes first-step target to Twin.
- ExecutorAgent: mass balance V update; clamps volume; converts to L1 via VolumeModel.
- SupervisorAgent: monitors metrics, constraints, savings; supports drift guard biasing (extension planned).

Digital Twin (OPC UA)
- Server exposes: Tunnel/Volume_V_m3, Tunnel/WaterLevel_L1_m, InflowF1_m3_15min, OutflowF2_m3h, Energy prices, Control/TargetF2_m3h, OptimizationStatus.
- Historical replay: feeds agents deterministically for validation; supports real-time run.

Installation
- Requirements: Python 3.10+ recommended; see requirements.txt.
- Install
  - pip install -r requirements.txt
  - Optional (OPC UA, dashboard): pip install opcua plotly dash

Quick Start
- Legacy MPC/LSTM (single-run)
  - python main.py "data/Hackathon_HSY_data.xlsx" "data/Volume of tunnel vs level Blominmäki.xlsx" --horizon 24
  - With dashboard: add --dashboard
- Multi-Agent (offline/realtime) [see main_multiagent.py docs]
  - Offline 24h with report:
    - python main_multiagent.py --mode offline --data data/test_data.csv --steps 96 --report --baseline 850.0
- Hybrid (RL-gated MPC demo)
  - python main.py "data/Hackathon_HSY_data.xlsx" "data/Volume of tunnel vs level Blominmäki.xlsx" --horizon 24 --hybrid
  - Output: results/hybrid_schedule.csv when MPC fallback used.

Configuration
- Key YAML parameters (config.yaml)
  - timestep_hours: 0.25; horizon_steps: 96
  - v_min/v_max; l_min/l_max; ramp limits
  - forecasting: epochs, lookback_steps
  - opcua: endpoint, replay options

Data Requirements
- Historical data (Excel/CSV): timestamps; L1, V; F1 (m³/15min); F2 (m³/h); per-pump flow/power/frequency; prices.
- Volume table (Excel/CSV): L1 vs V mapping; loaded by VolumeModel.
- Sample data provided in data/; see data/README.md for details.

Optimization & KPIs
- KPIs reported: total_cost_eur, total_energy_kwh, avg/peak power, min/avg/max level, total_pumped_m3, avg_efficiency_pct, ramp/level violations.
- Schedule CSV includes: timestep, F2, power, cost_eur, volume/level/head; per-pump flows where applicable.

Dashboard & Reports
- Static HTML via src/dashboard_multiagent.py; charts: cost comparison, level/volume, flows, cross-section, KPIs.
- Interactive dashboard optional via --dashboard in legacy mode.

Validation & Baseline
- Baseline tool: calculate_baseline.py → results/baseline_metrics.json.
- Validation runner: validate_junction2025.py executes baseline → optimization → summary/report.

Arm Edge Deployment (from ARM_DEPLOYMENT.md)
- Targets: Raspberry Pi 5 (pilot), Toradex Verdin i.MX8 (production).
- Forecast: convert LSTM to TF Lite INT8 for 25 ms inference; Planner: CBC on Arm with 48-step horizon; total power 3–5 W.
- Security: OPC UA TLS; offline operation; watchdog; UPS; failover.
- Deployment script outline and performance benchmarks included in ARM_DEPLOYMENT.md.

File Guide & Project Structure
- Entry points: main.py (legacy/hybrid), main_multiagent.py (multi-agent), validate_system.py, validate_junction2025.py.
- Core modules (src/): agents.py, forecast.py, optimize.py, model.py, digital_twin.py, dashboard_multiagent.py, data_utils.py.
- Docs previously split across: ARCHITECTURE.md, MULTIAGENT_README.md, MULTIAGENT_QUICKSTART.md, IMPLEMENTATION_SUMMARY_MULTIAGENT.md, ENHANCEMENTS.md, ENHANCED_PUMPS.md, ARM_DEPLOYMENT.md, SUBMISSION_CHECKLIST.md. This README consolidates their content.
- See data/README.md and results/README.md for directory-level notes.

Troubleshooting
- Optimization infeasible: check constraints and initial volume; reduce horizon or loosen bounds.
- Slow solves: decrease horizon_steps; set solver time limit (already enabled); accept feasible solutions.
- LSTM issues: reduce epochs/lookback; use persistence fallback; ensure ≥32 samples.
- Dashboard: run a simulation first; ensure port 8050 free.

Judging Alignment & Submission Checklist (condensed)
- Applicability: OPC UA Digital Twin; edge deployment path; modular design.
- Clarity/Impact: quantified savings; HTML reports; KPIs; business case.
- Technical soundness: MILP with terminal/avg-flow constraints; cubic power; 0 violations in validation; time-bounded solver.
- Creativity: multi-agent coordination; RL hybrid path; edge quantization.
- See SUBMISSION_CHECKLIST.md for full checklist and demo script.

Roadmap & Enhancements
- Supervisor drift guard integration; RL planner with confidence-based MPC fallback; behavioral cloning from MPC trajectories; weather API; SOS2 H–Q curves; stochastic scenarios; predictive maintenance; federated learning.

License, Authors, Contact
- License: MIT (see repository license if present).
- Authors: IFO Team • Junction 2025 Valmet–HSY challenge.
- Contact: Open an issue; include environment details and sample data snippet.

—

Appendix A — Architecture Highlights (from ARCHITECTURE.md)
- Layers: Coordination → Intelligence → Decision → Physical.
- Formulas: V(t+1)=V(t)+F1−F2·Δt; Q∝f; H∝f²; P∝f³; objective minimizing energy cost with penalties.
- Metrics: Energy/cost totals, specific energy/cost, peak power, forecast errors (MAE/RMSE/MAPE), constraint counts, runtime imbalance.

Appendix B — Enhanced Pump Models (from ENHANCED_PUMPS.md)
- Digitized curves for small (250 kW) and large (400 kW) pumps; efficiency-head families over frequencies; interpolation via SciPy; BEP awareness; examples provided in source doc.

Appendix C — Implemented Enhancements (from ENHANCEMENTS.md / IMPLEMENTATION_COMPLETE.md)
- Discrete frequencies with cubic power scaling; dynamic head coupling; KPI set; dashboard; Excel volume table; logging; performance improvements and constraints summary.

Appendix D — Multi-Agent Quickstart (from MULTIAGENT_QUICKSTART.md)
- Commands and flags for offline/realtime runs; config template; outputs and troubleshooting summarized above.

Appendix E — Junction 2025 Pitch (from JUNCTION2025_README.md)
- Savings range, agent interactions, emergent strategies, deployment steps; supervisor adaptation and edge focus.

Appendix F — Project Summary (from PROJECT_SUMMARY.md)
- Modules delivered; features; expected performance; business impact; roadmap phases; dependencies.

Appendix G — Results Directory Guide (from results/README.md) and Data Guide (from data/README.md)
- Result artifacts: baseline/optimized/comparison CSVs, HTML reports, models/; data file expectations and privacy guidance.

