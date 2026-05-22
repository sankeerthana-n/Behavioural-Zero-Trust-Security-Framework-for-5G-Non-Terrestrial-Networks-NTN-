"""
NTN Satellite Simulator -- Module 1
===================================
Simulates 4 satellite layers (LEO, MEO, HAPS, GEO) emitting UE telemetry
every second. Supports automatic and manual attack injection.

Layers:
  LEO  -- Low Earth Orbit   (550 km,  ~7800 m/s)
  MEO  -- Medium Earth Orbit (8000 km, ~3900 m/s)
  HAPS -- High Altitude PS   (20 km,   ~0-30 m/s)
  GEO  -- Geostationary      (35786 km,~3100 m/s)
"""

import asyncio
import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Fixed seed for reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Physical constants per layer
# ---------------------------------------------------------------------------
LAYER_CONFIG = {
    "HAPS": {
        "alt_km": 20,
        "velocity_ms": 15,        # nominal
        "velocity_range": (0, 50),
        "doppler_hz": 0,
        "doppler_range": (-500, 500),
        "rssi_base": -65,
        "rtt_ms": 15,
    },
    "LEO": {
        "alt_km": 550,
        "velocity_ms": 7800,
        "velocity_range": (7500, 8000),
        "doppler_hz": 0,
        "doppler_range": (-50000, 50000),
        "rssi_base": -85,
        "rtt_ms": 20,
    },
    "MEO": {
        "alt_km": 8000,
        "velocity_ms": 3900,
        "velocity_range": (3700, 4100),
        "doppler_hz": 0,
        "doppler_range": (-10000, 10000),
        "rssi_base": -95,
        "rtt_ms": 100,
    },
    "GEO": {
        "alt_km": 35786,
        "velocity_ms": 3100,
        "velocity_range": (3000, 3200),
        "doppler_hz": 0,
        "doppler_range": (-1000, 1000),
        "rssi_base": -105,
        "rtt_ms": 600,
    },
}


@dataclass
class SatelliteNode:
    """Represents a single satellite / HAPS node."""
    node_id: str
    layer: str
    base_lat: float = 0.0
    base_lon: float = 0.0
    tick: int = 0

    def config(self) -> Dict:
        return LAYER_CONFIG[self.layer]

    def generate_telemetry(self) -> Dict[str, Any]:
        """Generate one telemetry record with realistic orbital motion."""
        cfg = self.config()
        self.tick += 1

        # Simulate orbital motion (simplified circular orbit)
        angular_rate = cfg["velocity_ms"] / ((6371 + cfg["alt_km"]) * 1000)  # rad/s
        elapsed = self.tick  # seconds since start
        lat = self.base_lat + 20.0 * math.sin(angular_rate * elapsed)
        lon = self.base_lon + 20.0 * math.cos(angular_rate * elapsed)

        # Clamp lat to valid range
        lat = max(-90, min(90, lat))
        lon = ((lon + 180) % 360) - 180

        velocity = cfg["velocity_ms"] + random.uniform(-10, 10)
        doppler = random.uniform(*cfg["doppler_range"])
        rssi = cfg["rssi_base"] + random.uniform(-3, 3)
        traffic = random.choices(
            ["normal", "burst", "anomalous"],
            weights=[0.85, 0.12, 0.03],
            k=1,
        )[0]

        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "timestamp": int(time.time() * 1000),
            "gps": {"lat": round(lat, 6), "lon": round(lon, 6), "alt_km": cfg["alt_km"]},
            "velocity_ms": round(velocity, 2),
            "doppler_hz": round(doppler, 2),
            "rssi_dbm": round(rssi, 2),
            "traffic_pattern": traffic,
            "is_spoofed": False,
        }


class NTNSimulator:
    """
    Manages multiple satellite nodes and injects attack scenarios.
    """

    def __init__(self):
        self.start_time: float = time.time()
        self.tick: int = 0
        self.nodes: Dict[str, SatelliteNode] = {
            "LEO_01": SatelliteNode("LEO_01", "LEO", base_lat=10.0, base_lon=30.0),
            "MEO_01": SatelliteNode("MEO_01", "MEO", base_lat=-20.0, base_lon=60.0),
            "HAPS_01": SatelliteNode("HAPS_01", "HAPS", base_lat=35.0, base_lon=-5.0),
            "GEO_01": SatelliteNode("GEO_01", "GEO", base_lat=0.0, base_lon=100.0),
        }
        self._attack_queue: List[str] = []
        self._attack_log: List[Dict] = []
        self._last_meo_telemetry: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Attack injection
    # ------------------------------------------------------------------
    def queue_attack(self, attack_type: str) -> Dict:
        """Queue a manual attack for next tick."""
        valid = ("velocity_spoof", "replay", "impersonation")
        if attack_type not in valid:
            return {"error": f"Unknown type. Use: {valid}"}
        self._attack_queue.append(attack_type)
        return {"queued": attack_type}

    def _inject_velocity_spoof(self, telemetry: Dict) -> Dict:
        """LEO node reports HAPS-level velocity (~20 m/s)."""
        spoofed = copy.deepcopy(telemetry)
        spoofed["velocity_ms"] = round(random.uniform(10, 30), 2)
        spoofed["is_spoofed"] = True
        spoofed["traffic_pattern"] = "anomalous"
        self._attack_log.append({
            "tick": self.tick,
            "type": "velocity_spoof",
            "node_id": spoofed["node_id"],
            "detail": f"LEO reporting velocity {spoofed['velocity_ms']} m/s (HAPS-level)",
            "timestamp": int(time.time() * 1000),
        })
        return spoofed

    def _inject_replay(self, telemetry: Dict) -> List[Dict]:
        """MEO node repeats identical timestamp 5 times."""
        if self._last_meo_telemetry is None:
            self._last_meo_telemetry = copy.deepcopy(telemetry)

        replayed = []
        for i in range(5):
            rec = copy.deepcopy(self._last_meo_telemetry)
            rec["is_spoofed"] = True
            rec["traffic_pattern"] = "anomalous"
            replayed.append(rec)

        self._attack_log.append({
            "tick": self.tick,
            "type": "replay",
            "node_id": telemetry["node_id"],
            "detail": f"MEO replaying timestamp {self._last_meo_telemetry['timestamp']} x5",
            "timestamp": int(time.time() * 1000),
        })
        return replayed

    def _inject_impersonation(self, telemetry: Dict) -> Dict:
        """New node claims existing node_id with wrong GPS."""
        spoofed = copy.deepcopy(telemetry)
        # Shift GPS by 30-50 degrees (thousands of km off)
        spoofed["gps"]["lat"] += random.uniform(30, 50)
        spoofed["gps"]["lon"] += random.uniform(30, 50)
        spoofed["gps"]["lat"] = max(-90, min(90, spoofed["gps"]["lat"]))
        spoofed["gps"]["lon"] = ((spoofed["gps"]["lon"] + 180) % 360) - 180
        spoofed["is_spoofed"] = True
        spoofed["traffic_pattern"] = "anomalous"
        self._attack_log.append({
            "tick": self.tick,
            "type": "impersonation",
            "node_id": spoofed["node_id"],
            "detail": (
                f"Impersonator at ({spoofed['gps']['lat']:.1f}, "
                f"{spoofed['gps']['lon']:.1f}) claiming {spoofed['node_id']}"
            ),
            "timestamp": int(time.time() * 1000),
        })
        return spoofed

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def generate_tick(self) -> List[Dict]:
        """Generate one second of telemetry from all nodes."""
        self.tick += 1
        records: List[Dict] = []

        for node in self.nodes.values():
            t = node.generate_telemetry()
            records.append(t)
            # Save latest MEO telemetry for replay
            if node.layer == "MEO":
                self._last_meo_telemetry = copy.deepcopy(t)

        # --- Auto-inject attacks on schedule ---
        cycle = self.tick % 30
        if cycle == 10:
            self._attack_queue.append("velocity_spoof")
        elif cycle == 20:
            self._attack_queue.append("replay")
        elif cycle == 0 and self.tick > 0:
            self._attack_queue.append("impersonation")

        # --- Process queued attacks ---
        processed: List[Dict] = []
        for attack in self._attack_queue:
            if attack == "velocity_spoof":
                leo_rec = next((r for r in records if r["layer"] == "LEO"), None)
                if leo_rec:
                    idx = records.index(leo_rec)
                    records[idx] = self._inject_velocity_spoof(leo_rec)
            elif attack == "replay":
                meo_rec = next((r for r in records if r["layer"] == "MEO"), None)
                if meo_rec:
                    idx = records.index(meo_rec)
                    replayed = self._inject_replay(meo_rec)
                    records[idx] = replayed[0]
                    records.extend(replayed[1:])
            elif attack == "impersonation":
                # Impersonate a random node
                target = random.choice(list(self.nodes.values()))
                imp = self._inject_impersonation(
                    target.generate_telemetry()
                )
                records.append(imp)
        self._attack_queue.clear()

        return records

    @property
    def attack_log(self) -> List[Dict]:
        return list(self._attack_log)
