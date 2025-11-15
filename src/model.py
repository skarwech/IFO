from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------- Volume / Level Model -----------------

class TunnelGeometry:
    """Implements piecewise formulas for tunnel volume vs level.
    Based on provided quadratic / linear / inverted quadratic segments.
    Level L (m) in [0, 14.1]; operational range 0-8 m.
    """
    def __init__(self):
        pass

    @staticmethod
    def level_to_volume(L: float) -> float:
        if L < 0.4:
            return 350.0
        if L <= 5.9:
            Vbx = L - 0.4
            return (((1000.0 * (Vbx ** 2)) / 2.0) * 5.0) + 350.0
        if L <= 8.6:
            Vcx = L - 5.9
            return (5500.0 * Vcx * 5.0) + 75975.0
        if L <= 14.1:
            Vdx = L - 8.6
            return (((5.5 * 5500.0 / 2.0) - ((5.5 - Vdx) ** 2 * 1000.0 / 2.0)) * 5.0) + 150225.0
        # Above design max
        return 225850.0

    @staticmethod
    def build_table(step: float = 0.01) -> pd.DataFrame:
        levels = np.arange(0.0, 14.11, step)
        volumes = [TunnelGeometry.level_to_volume(l) for l in levels]
        return pd.DataFrame({"level_m": levels, "volume_m3": volumes})

class VolumeModel:
    def __init__(self, volume_table_path: str = None):
        if volume_table_path:
            table = self._load_excel_table(volume_table_path)
        else:
            table = TunnelGeometry.build_table(step=0.01)
        self._l2v = interp1d(table["level_m"], table["volume_m3"], kind="linear", fill_value="extrapolate")
        self._v2l = interp1d(table["volume_m3"], table["level_m"], kind="linear", fill_value="extrapolate")
    
    def _load_excel_table(self, path: str) -> pd.DataFrame:
        from pathlib import Path
        p = Path(path)
        if p.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(p, sheet_name='Taul1')
        else:
            df = pd.read_csv(p)
        # Standardize column names
        level_col = [c for c in df.columns if 'Level' in c or 'level' in c or 'L1' in c][0]
        volume_col = [c for c in df.columns if 'Volume' in c or 'volume' in c or 'V' in c][0]
        return pd.DataFrame({
            "level_m": df[level_col].astype(float),
            "volume_m3": df[volume_col].astype(float)
        }).sort_values("level_m").reset_index(drop=True)

    def level_to_volume(self, level_m: float) -> float:
        return float(self._l2v(level_m))

    def volume_to_level(self, volume_m3: float) -> float:
        return float(self._v2l(volume_m3))

# ---------------- Pump Curve Model -----------------

@dataclass
class PumpCurve:
    frequency_hz: float
    h_q_points: List[tuple]  # (Q_lps, H_m)
    p_q_points: List[tuple]  # (Q_lps, P_kW)
    eta_q_points: List[tuple]  # (Q_lps, efficiency_fraction)
    npsh_q_points: List[tuple]  # (Q_lps, NPSHr_m)

class DigitizedPump:
    def __init__(self, curves: Dict[float, PumpCurve], name: str, nominal_flow_lps: float, nominal_power_kw: float):
        self.curves = curves
        self.name = name
        self.nominal_flow_lps = nominal_flow_lps
        self.nominal_power_kw = nominal_power_kw
        # Build interpolators for 50 Hz reference
        ref = curves[50.0]
        self._h_of_q_ref = interp1d([q for q, _ in ref.h_q_points], [h for _, h in ref.h_q_points], kind="linear", fill_value="extrapolate")
        self._eta_of_q_ref = interp1d([q for q, _ in ref.eta_q_points], [e for _, e in ref.eta_q_points], kind="linear", fill_value="extrapolate")

    def flow_at_frequency(self, freq_hz: float) -> float:
        return self.nominal_flow_lps * (freq_hz / 50.0)

    def power_at_frequency(self, freq_hz: float) -> float:
        return self.nominal_power_kw * (freq_hz / 50.0)  # linearized approximation

    def efficiency_at_head(self, head_m: float, freq_hz: float) -> float:
        # Approximate by shifting head according to affinity (H ∝ f^2)
        head_at_ref = head_m / ((freq_hz / 50.0) ** 2)
        # Estimate best efficiency at similar flow region: use head mapping via ref curve shape
        # Inverse usage: sample Q grid
        q_grid = np.linspace(0, self.nominal_flow_lps * 1.2, 50)
        h_vals = self._h_of_q_ref(q_grid)
        idx = (np.abs(h_vals - head_at_ref)).argmin()
        eta = float(self._eta_of_q_ref(q_grid[idx]))
        return max(0.0, min(eta, 0.90))

class PumpFleet:
    def __init__(self, small_count: int, large_count: int):
        # Small pump curves (50 Hz)
        small_curve_50 = PumpCurve(
            frequency_hz=50.0,
            h_q_points=[(0,48),(240,48),(320,46),(400,43),(480,38),(560,32),(640,25),(720,16),(800,5)],
            p_q_points=[(240,225),(400,200),(560,175),(720,180),(800,190)],
            eta_q_points=[(240,0.76),(320,0.79),(400,0.81),(480,0.82),(560,0.81),(640,0.79),(720,0.75)],
            npsh_q_points=[(240,2),(400,4),(560,6),(720,8),(800,10)]
        )
        large_curve_50 = PumpCurve(
            frequency_hz=50.0,
            h_q_points=[(0,56),(400,56),(600,50),(800,42),(1000,32),(1200,22),(1400,10),(1600,0)],
            p_q_points=[(400,400),(800,380),(1200,350),(1600,300)],
            eta_q_points=[(400,0.70),(600,0.78),(800,0.83),(1000,0.85),(1200,0.84),(1400,0.80)],
            npsh_q_points=[(400,2),(800,4),(1200,7),(1600,8)]
        )
        self.small = DigitizedPump({50.0: small_curve_50}, "small", nominal_flow_lps=464, nominal_power_kw=188.7)
        self.large = DigitizedPump({50.0: large_curve_50}, "large", nominal_flow_lps=925, nominal_power_kw=358.1)
        self.small_count = small_count
        self.large_count = large_count

    def total_flow_lps(self, small_on: int, large_on: int, freq_map: Dict[str, float]) -> float:
        f_small = freq_map.get("small", 50.0)
        f_large = freq_map.get("large", 50.0)
        return small_on * self.small.flow_at_frequency(f_small) + large_on * self.large.flow_at_frequency(f_large)

    def total_power_kw(self, small_on: int, large_on: int, freq_map: Dict[str, float]) -> float:
        f_small = freq_map.get("small", 50.0)
        f_large = freq_map.get("large", 50.0)
        return small_on * self.small.power_at_frequency(f_small) + large_on * self.large.power_at_frequency(f_large)

# Utility to derive price from two columns

def combine_price(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if "Electricity price" in c]
    if not cols:
        raise ValueError("No electricity price columns found")
    temp = df[cols].fillna(0.0)
    return temp.max(axis=1)
