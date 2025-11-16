import os
from typing import Optional, Dict, Any


def _read_file(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None


def read_arm_metrics() -> Dict[str, Any]:
    """Read simple device metrics from Linux sysfs (Raspberry Pi friendly).
    Returns values if available; keys may be None when not present.
    """
    temp = _read_file("/sys/class/thermal/thermal_zone0/temp")
    if temp is not None:
        temp = temp / 1000.0
    cpu0_freq = _read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if cpu0_freq is not None:
        cpu0_freq = cpu0_freq / 1000.0
    volts = _read_file("/sys/class/hwmon/hwmon0/in0_input")
    if volts is not None:
        volts = volts / 1000.0
    return {
        "cpu_temp_c": temp,
        "cpu0_freq_mhz": cpu0_freq,
        "soc_voltage_v": volts,
    }
