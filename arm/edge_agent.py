import argparse
import time
import json
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import requests

try:
    import torch  # optional on edge
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from .metrics import read_arm_metrics
from .opcua_edge import OPCUAEdgeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("edge-agent")


class EdgePolicy:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.ready = False
        self.model = None
        if TORCH_OK and model_path:
            try:
                self.model = torch.jit.load(model_path, map_location="cpu")
                self.model.eval()
                self.ready = True
                log.info("Loaded TorchScript policy.")
            except Exception as e:
                log.warning(f"Policy load failed: {e}")

    def act(self, obs: np.ndarray, pump_count: int = 4) -> Dict[str, Any]:
        if self.ready and TORCH_OK:
            try:
                with torch.no_grad():
                    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                    y = self.model(x).squeeze(0).cpu().numpy()
                hz = np.clip(y[:pump_count], 0.0, 52.0).tolist()
                return {"pump_hz": hz, "source": "policy"}
            except Exception as e:
                log.warning(f"Policy inference failed: {e}")
        # Fallback: minimal safe frequency
        return {"pump_hz": [48.0] * pump_count, "source": "fallback"}


def build_observation(state: Dict[str, Any], pump_count: int = 4) -> np.ndarray:
    feats: List[float] = [
        float(state.get("level_m", 2.0) or 2.0),
        float(state.get("volume_m3", 8000.0) or 8000.0),
        float(state.get("inflow_m3h", 600.0) or 600.0),
        float(state.get("outflow_m3h", 600.0) or 600.0),
    ]
    pumps = state.get("pump_hz", [48.0] * pump_count) or [48.0] * pump_count
    feats.extend([float(h or 48.0) for h in pumps][:pump_count])
    return np.array(feats, dtype=np.float32)


def schedule_to_action(step: List[float], pump_count: int = 4) -> Dict[str, Any]:
    hz = [float(f) for f in (step or [])][:pump_count]
    if len(hz) < pump_count:
        hz += [48.0] * (pump_count - len(hz))
    return {"pump_hz": hz}


def run_edge(args):
    backend = args.backend.rstrip("/")
    opc = OPCUAEdgeClient(args.opcua)
    opc.connect()

    policy = EdgePolicy(model_path=args.policy) if args.policy else EdgePolicy()
    log.info(f"Edge agent started. Backend={backend} OPCUA={args.opcua}")

    pump_count = 4

    try:
        while True:
            # Read current state
            state = opc.read_state()
            obs = build_observation(state, pump_count=pump_count)

            action = policy.act(obs, pump_count=pump_count)

            # Optional hybrid: call backend MPC and use first-step schedule
            if args.hybrid:
                try:
                    r = requests.post(
                        f"{backend}/api/optimize",
                        json={"horizon": args.horizon, "mode": "hybrid"},
                        timeout=10,
                    )
                    if r.ok:
                        sched = r.json().get("schedule")
                        if isinstance(sched, list) and sched:
                            action = schedule_to_action(sched[0], pump_count=pump_count)
                            action["source"] = "mpc-backend"
                except Exception as e:
                    log.warning(f"Backend MPC request failed: {e}")

            # Apply pump commands
            opc.write_pumps(action["pump_hz"])

            # Report edge metrics to backend (best-effort)
            try:
                metrics = read_arm_metrics()
                requests.post(
                    f"{backend}/api/edge/metrics",
                    json={"metrics": metrics},
                    timeout=2,
                )
            except Exception:
                pass

            src = action.get("source", "?")
            log.info(f"Applied {src}: hz={action['pump_hz']}")
            time.sleep(args.period)
    finally:
        opc.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="IFO ARM Edge Agent")
    p.add_argument("--backend", default="http://localhost:8000", help="Backend API base URL")
    p.add_argument("--opcua", default="opc.tcp://127.0.0.1:4840", help="OPC UA endpoint")
    p.add_argument("--policy", default=None, help="TorchScript .pt path (optional)")
    p.add_argument("--hybrid", action="store_true", help="Use backend MPC each loop")
    p.add_argument("--horizon", type=int, default=24, help="Optimization horizon for backend MPC")
    p.add_argument("--period", type=float, default=5.0, help="Control loop period in seconds")
    args = p.parse_args()
    run_edge(args)
