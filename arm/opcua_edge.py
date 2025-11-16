from typing import Dict, Any, List
from opcua import Client


class OPCUAEdgeClient:
    """Minimal OPC UA client wrapper for edge operations.
    Adjust NodeIds to match your server model.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.client = Client(endpoint)

    def connect(self):
        self.client.connect()

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass

    def read_state(self) -> Dict[str, Any]:
        # Replace with actual NodeIds for your server.
        return {
            "level_m": self._read("Level_m"),
            "volume_m3": self._read("Volume_m3"),
            "inflow_m3h": self._read("F1_m3h"),
            "outflow_m3h": self._read("F2_m3h"),
            "pump_hz": [self._read(f"Pump{i}_Hz") for i in range(1, 5)],
        }

    def write_pumps(self, hz: List[float]):
        for i, f in enumerate(hz, start=1):
            self._write(f"Pump{i}_Hz", float(f))

    def _read(self, name: str):
        try:
            node = self.client.get_node(f"ns=2;s={name}")
            return node.get_value()
        except Exception:
            return None

    def _write(self, name: str, value: float):
        try:
            node = self.client.get_node(f"ns=2;s={name}")
            node.set_value(value)
        except Exception:
            pass
