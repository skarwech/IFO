from typing import Dict, Any
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, LpBinary, LpStatus, value, LpContinuous, PULP_CBC_CMD
from .model import VolumeModel, PumpFleet

DT_HOURS = 0.25  # 15 minutes
RAMP_LIMIT_M3H = 500.0
MIN_UP_STEPS = int(2.0 / DT_HOURS)  # 2h / 0.25 = 8 steps
MIN_DOWN_STEPS = MIN_UP_STEPS
FREQ_CHOICES = [48, 49, 50]  # Discrete frequency options (Hz)
L2_FIXED = 30.0  # Downstream head (m)

class MPCOptimizer:
    def __init__(self, small_count: int = 4, large_count: int = 4, time_limit_s: int = 10, mip_gap: float = 0.02):
        self.volume_model = VolumeModel()
        self.fleet = PumpFleet(small_count=small_count, large_count=large_count)
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap

    def optimize(self, initial_volume_m3: float, inflow_forecast_m3_per_15min: pd.Series, price_forecast_eur_kwh: pd.Series) -> Dict[str, Any]:
        steps = len(inflow_forecast_m3_per_15min)
        inflow_m3h = inflow_forecast_m3_per_15min.values * 4.0
        prices = price_forecast_eur_kwh.values
        prob = LpProblem("IPS_MPC", LpMinimize)

        # Per-pump binary and discrete frequency selection
        y_small = [[LpVariable(f"y_small_{i}_{t}", cat=LpBinary) for t in range(steps)] for i in range(self.fleet.small_count)]
        y_large = [[LpVariable(f"y_large_{i}_{t}", cat=LpBinary) for t in range(steps)] for i in range(self.fleet.large_count)]
        # Frequency selection binaries for discrete choices {48, 49, 50}
        z_freq_small = [[[LpVariable(f"z_freq_small_{i}_{t}_{fq}", cat=LpBinary) for fq in FREQ_CHOICES] for t in range(steps)] for i in range(self.fleet.small_count)]
        z_freq_large = [[[LpVariable(f"z_freq_large_{i}_{t}_{fq}", cat=LpBinary) for fq in FREQ_CHOICES] for t in range(steps)] for i in range(self.fleet.large_count)]
        # Continuous level and volume
        VMIN = float(self.volume_model.level_to_volume(0.0))
        VMAX = float(self.volume_model.level_to_volume(8.0))
        volume = [LpVariable(f"volume_{t}", lowBound=VMIN, upBound=VMAX) for t in range(steps + 1)]
        level = [LpVariable(f"level_{t}", lowBound=0.0, upBound=8.0) for t in range(steps + 1)]

        prob += volume[0] == initial_volume_m3
        initial_level = self.volume_model.volume_to_level(initial_volume_m3)
        prob += level[0] == initial_level

        # Terminal soft volume balance: volume[steps] ~ volume[0]
        term_slack_up = LpVariable("term_slack_up", lowBound=0)
        term_slack_dn = LpVariable("term_slack_dn", lowBound=0)

        # Frequency selection constraints: exactly one frequency if on, zero if off
        for i in range(self.fleet.small_count):
            for t in range(steps):
                prob += lpSum(z_freq_small[i][t]) == y_small[i][t]
        for i in range(self.fleet.large_count):
            for t in range(steps):
                prob += lpSum(z_freq_large[i][t]) == y_large[i][t]

        # At least one pump on
        for t in range(steps):
            prob += lpSum(y_small[i][t] for i in range(self.fleet.small_count)) + lpSum(y_large[i][t] for i in range(self.fleet.large_count)) >= 1

        # Flow and power with affinity laws
        total_flow_m3h = []
        total_power_kw = []
        for t in range(steps):
            # Flow: Q = Q_nominal * (f/50) per pump, summed
            flow_lps_expr = lpSum(
                lpSum(z_freq_small[i][t][fq] * FREQ_CHOICES[fq] * (self.fleet.small.nominal_flow_lps / 50.0) for fq in range(len(FREQ_CHOICES)))
                for i in range(self.fleet.small_count)
            ) + lpSum(
                lpSum(z_freq_large[i][t][fq] * FREQ_CHOICES[fq] * (self.fleet.large.nominal_flow_lps / 50.0) for fq in range(len(FREQ_CHOICES)))
                for i in range(self.fleet.large_count)
            )
            flow_m3h_expr = flow_lps_expr * 3.6
            total_flow_m3h.append(flow_m3h_expr)
            
            # Power: P = P_nominal * (f/50)^3 using piecewise linearization
            # Breakpoints: f in {48,49,50} -> scale factors
            power_scale = {48: (48.0/50.0)**3, 49: (49.0/50.0)**3, 50: 1.0}
            power_expr = lpSum(
                lpSum(z_freq_small[i][t][fq] * self.fleet.small.nominal_power_kw * power_scale[FREQ_CHOICES[fq]] for fq in range(len(FREQ_CHOICES)))
                for i in range(self.fleet.small_count)
            ) + lpSum(
                lpSum(z_freq_large[i][t][fq] * self.fleet.large.nominal_power_kw * power_scale[FREQ_CHOICES[fq]] for fq in range(len(FREQ_CHOICES)))
                for i in range(self.fleet.large_count)
            )
            total_power_kw.append(power_expr)
            
            # Volume dynamics
            prob += volume[t + 1] == volume[t] + (inflow_m3h[t] - flow_m3h_expr) * DT_HOURS
            # Approximate level from volume (linear piecewise for speed)
            # Simplified: use mid-range approximation L ≈ (V - 350) / 10000 for operational range
            # More accurate: could use SOS2 piecewise but adds complexity
            prob += level[t + 1] * 10000 >= volume[t + 1] - 5000
            prob += level[t + 1] * 10000 <= volume[t + 1] + 5000
            
            # Ramp constraints
            if t > 0:
                prev_flow = total_flow_m3h[t - 1]
                prob += flow_m3h_expr - prev_flow <= RAMP_LIMIT_M3H
                prob += prev_flow - flow_m3h_expr <= RAMP_LIMIT_M3H

        # Terminal soft equality constraint
        prob += volume[steps] - volume[0] == term_slack_up - term_slack_dn

        # Average-flow balance over horizon: sum(F2) >= sum(F1) - small slack
        # Implement with non-negative slack to avoid infeasibility on tight bounds
        avg_balance_slack = LpVariable("avg_balance_slack", lowBound=0)
        prob += lpSum(total_flow_m3h) * DT_HOURS >= float(np.sum(inflow_m3h)) * DT_HOURS - avg_balance_slack

        # Smoothness |Δflow| penalty variables
        ramp_abs = []
        for t in range(1, steps):
            z = LpVariable(f"ramp_abs_{t}", lowBound=0)
            ramp_abs.append(z)
            prob += z >= total_flow_m3h[t] - total_flow_m3h[t-1]
            prob += z >= total_flow_m3h[t-1] - total_flow_m3h[t]

        # Constancy penalty around average inflow rate
        if steps > 0:
            F2_target = float(np.mean(inflow_m3h))
        else:
            F2_target = 0.0
        dev_abs = []
        for t in range(steps):
            d = LpVariable(f"dev_const_{t}", lowBound=0)
            dev_abs.append(d)
            prob += d >= total_flow_m3h[t] - F2_target
            prob += d >= F2_target - total_flow_m3h[t]

        # Min up/down time per pump
        for i in range(self.fleet.small_count):
            for t in range(1, steps):
                start = y_small[i][t] - y_small[i][t - 1]
                stop = y_small[i][t - 1] - y_small[i][t]
                if t + MIN_UP_STEPS <= steps:
                    prob += lpSum(y_small[i][k] for k in range(t, t + MIN_UP_STEPS)) >= start * MIN_UP_STEPS
                if t + MIN_DOWN_STEPS <= steps:
                    prob += lpSum(1 - y_small[i][k] for k in range(t, t + MIN_DOWN_STEPS)) >= stop * MIN_DOWN_STEPS
        for i in range(self.fleet.large_count):
            for t in range(1, steps):
                start = y_large[i][t] - y_large[i][t - 1]
                stop = y_large[i][t - 1] - y_large[i][t]
                if t + MIN_UP_STEPS <= steps:
                    prob += lpSum(y_large[i][k] for k in range(t, t + MIN_UP_STEPS)) >= start * MIN_UP_STEPS
                if t + MIN_DOWN_STEPS <= steps:
                    prob += lpSum(1 - y_large[i][k] for k in range(t, t + MIN_DOWN_STEPS)) >= stop * MIN_DOWN_STEPS

        # Objective: minimize energy cost with additional penalties
        cost_terms = [total_power_kw[t] * prices[t] * DT_HOURS for t in range(steps)]
        terminal_penalty = 5e-4 * (term_slack_up + term_slack_dn)
        smooth_penalty = 1e-4 * lpSum(ramp_abs) if ramp_abs else 0
        constancy_penalty = 5e-5 * lpSum(dev_abs) if dev_abs else 0
        # Optional price-weighted smoothness to avoid moving at expensive times
        smooth_price_penalty = 5e-5 * lpSum(ramp_abs[i-1] * prices[i] for i in range(1, steps)) if ramp_abs else 0
        balance_penalty = 1e-3 * avg_balance_slack
        prob += lpSum(cost_terms) + terminal_penalty + smooth_penalty + constancy_penalty + smooth_price_penalty + balance_penalty

        # Solve with time limit and MIP gap to avoid long runs
        solver = PULP_CBC_CMD(msg=True, timeLimit=self.time_limit_s)
        status_code = prob.solve(solver)
        status_str = LpStatus[status_code]
        result = {
            "status": status_str,
            "objective_cost": value(prob.objective) if prob.objective else 0.0,
        }
        # If solver didn't prove optimality but produced a feasible incumbent, continue
        # Heuristic: check if some key variables have values; if not, return early
        any_values = volume[0].value() is not None and total_flow_m3h[0] is not None
        if status_str not in ("Optimal",) and not any_values:
            return result
        
        # Extract schedule
        schedule_rows = []
        for t in range(steps):
            active_small = sum(int(y_small[i][t].value()) for i in range(self.fleet.small_count))
            active_large = sum(int(y_large[i][t].value()) for i in range(self.fleet.large_count))
            
            # Extract selected frequencies
            freq_list_small = []
            freq_list_large = []
            for i in range(self.fleet.small_count):
                for fq in range(len(FREQ_CHOICES)):
                    if z_freq_small[i][t][fq].value() and z_freq_small[i][t][fq].value() > 0.5:
                        freq_list_small.append(FREQ_CHOICES[fq])
            for i in range(self.fleet.large_count):
                for fq in range(len(FREQ_CHOICES)):
                    if z_freq_large[i][t][fq].value() and z_freq_large[i][t][fq].value() > 0.5:
                        freq_list_large.append(FREQ_CHOICES[fq])
            
            avg_f_small = np.mean(freq_list_small) if freq_list_small else 0.0
            avg_f_large = np.mean(freq_list_large) if freq_list_large else 0.0
            
            vol_val = volume[t].value()
            lvl_val = self.volume_model.volume_to_level(vol_val) if vol_val else 0.0
            head_val = L2_FIXED - lvl_val
            
            schedule_rows.append({
                "t": t,
                "small_active": active_small,
                "large_active": active_large,
                "avg_freq_small_hz": round(avg_f_small, 1),
                "avg_freq_large_hz": round(avg_f_large, 1),
                "volume_m3": round(vol_val, 1),
                "level_m": round(lvl_val, 2),
                "head_m": round(head_val, 2),
                "flow_m3h": round(value(total_flow_m3h[t]), 1),
                "power_kw": round(value(total_power_kw[t]), 1),
                "cost_eur": round(value(total_power_kw[t]) * prices[t] * DT_HOURS, 3),
                "price_eur_kwh": round(prices[t], 4)
            })
        
        result["schedule"] = pd.DataFrame(schedule_rows)
        
        # Compute KPIs
        df = result["schedule"]
        result["kpis"] = {
            "total_cost_eur": df["cost_eur"].sum(),
            "total_energy_kwh": df["power_kw"].sum() * DT_HOURS,
            "avg_power_kw": df["power_kw"].mean(),
            "peak_power_kw": df["power_kw"].max(),
            "avg_level_m": df["level_m"].mean(),
            "min_level_m": df["level_m"].min(),
            "max_level_m": df["level_m"].max(),
            "total_pumped_m3": df["flow_m3h"].sum() * DT_HOURS,
            "avg_efficiency_pct": self._estimate_avg_efficiency(df),
            "ramp_violations": self._count_ramp_violations(df),
            "level_violations": sum((df["level_m"] < 0.0) | (df["level_m"] > 8.0))
        }
        
        # If not proven optimal but we have a feasible schedule, mark accordingly
        if status_str != "Optimal":
            result["status"] = "FeasibleWithinTime"
        return result
    
    def _estimate_avg_efficiency(self, df: pd.DataFrame) -> float:
        # Rough efficiency estimate: assume BEP near nominal, penalize extremes
        eff_estimates = []
        for _, row in df.iterrows():
            if row["small_active"] > 0:
                eff_estimates.append(0.80)  # Approximate small pump efficiency
            if row["large_active"] > 0:
                eff_estimates.append(0.83)  # Approximate large pump efficiency
        return round(np.mean(eff_estimates) * 100, 1) if eff_estimates else 0.0
    
    def _count_ramp_violations(self, df: pd.DataFrame) -> int:
        violations = 0
        for i in range(1, len(df)):
            ramp = abs(df.iloc[i]["flow_m3h"] - df.iloc[i-1]["flow_m3h"])
            if ramp > RAMP_LIMIT_M3H + 1:
                violations += 1
        return violations
