"""
Behavioural Analytics Engine -- Module 2
========================================
Trust = w1(Location) + w2(Velocity) + w3(Identity)

Layer-specific weights:
  HAPS: w1=0.5, w2=0.2, w3=0.3  (GPS most reliable)
  LEO:  w1=0.3, w2=0.5, w3=0.2  (velocity cross-check critical)
  MEO:  w1=0.2, w2=0.3, w3=0.5  (identity/pattern primary)
  GEO:  w1=0.2, w2=0.2, w3=0.6  (traffic pattern dominates)
"""

import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from ntn_simulator import LAYER_CONFIG, SatelliteNode

# ---------------------------------------------------------------------------
# Layer-specific trust weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "HAPS": (0.5, 0.2, 0.3),
    "LEO":  (0.3, 0.5, 0.2),
    "MEO":  (0.2, 0.3, 0.5),
    "GEO":  (0.2, 0.2, 0.6),
}

# Decision thresholds
ALLOW_THRESHOLD = 70
CHALLENGE_THRESHOLD = 40


class BehaviouralAnalyticsEngine:
    """
    Evaluates each telemetry record and produces a trust score + decision.
    """

    def __init__(self, nodes: Dict[str, SatelliteNode], start_time: float):
        self.nodes = nodes
        self.start_time = start_time
        # Track timestamps seen per node (for replay detection)
        self._seen_timestamps: Dict[str, set] = defaultdict(set)
        # Track GPS positions per node (for impersonation detection)
        self._gps_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        # Track traffic patterns per node (moving average)
        self._traffic_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        # Total evaluation counters
        self.total_evaluated: int = 0
        self.threats_detected: int = 0
        self.spoofed_caught: int = 0
        self.spoofed_total: int = 0

    # ------------------------------------------------------------------
    # Location score
    # ------------------------------------------------------------------
    def _location_score(
        self, telemetry: Dict, node: Optional[SatelliteNode]
    ) -> Tuple[float, List[str]]:
        """
        Compare reported GPS vs orbital-physics-expected position.
        Return (score 0.0-1.0, list_of_threats).
        """
        threats: List[str] = []
        if node is None:
            threats.append("unknown_node_location")
            return 0.0, threats

        cfg = LAYER_CONFIG[telemetry["layer"]]
        reported = telemetry["gps"]

        # Compute expected position using simplified orbital mechanics
        elapsed = time.time() - self.start_time
        angular_rate = cfg["velocity_ms"] / ((6371 + cfg["alt_km"]) * 1000)
        expected_lat = node.base_lat + 20.0 * math.sin(angular_rate * elapsed)
        expected_lon = node.base_lon + 20.0 * math.cos(angular_rate * elapsed)
        expected_lat = max(-90, min(90, expected_lat))
        expected_lon = ((expected_lon + 180) % 360) - 180

        # Haversine-lite: approximate distance in km
        dlat = math.radians(reported["lat"] - expected_lat)
        dlon = math.radians(reported["lon"] - expected_lon)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(expected_lat))
            * math.cos(math.radians(reported["lat"]))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = 6371 * c

        if distance_km > 200:
            threats.append("location_anomaly_gt_200km")
            return 0.0, threats
        elif distance_km > 100:
            threats.append("location_drift_gt_100km")
            return 0.3, threats
        elif distance_km > 50:
            return 0.7, threats
        else:
            return 1.0, threats

    # ------------------------------------------------------------------
    # Velocity score
    # ------------------------------------------------------------------
    def _velocity_score(self, telemetry: Dict) -> Tuple[float, List[str]]:
        """
        Compare reported velocity vs layer physical bounds.
        """
        threats: List[str] = []
        layer = telemetry["layer"]
        cfg = LAYER_CONFIG[layer]
        vmin, vmax = cfg["velocity_range"]
        v = telemetry["velocity_ms"]

        if v < vmin or v > vmax:
            threats.append(
                f"impossible_physics_velocity_{v:.0f}ms_expected_{vmin}-{vmax}"
            )
            # Score based on how far out of bounds
            if v < vmin:
                deviation = (vmin - v) / vmin
            else:
                deviation = (v - vmax) / vmax
            score = max(0.0, 1.0 - deviation * 2)
            return score, threats

        # Within bounds -- perfect score with small noise penalty
        centre = (vmin + vmax) / 2
        spread = (vmax - vmin) / 2
        normalised = abs(v - centre) / spread if spread > 0 else 0
        return max(0.8, 1.0 - normalised * 0.2), threats

    # ------------------------------------------------------------------
    # Identity score
    # ------------------------------------------------------------------
    def _identity_score(self, telemetry: Dict) -> Tuple[float, List[str]]:
        """
        Detect replay, duplicate node_id from different GPS,
        traffic pattern deviation.
        """
        threats: List[str] = []
        subs: List[float] = []
        nid = telemetry["node_id"]
        ts = telemetry["timestamp"]
        gps = (telemetry["gps"]["lat"], telemetry["gps"]["lon"])
        traffic = telemetry["traffic_pattern"]

        # --- Replay detection ---
        if ts in self._seen_timestamps[nid]:
            threats.append("replay_attack_duplicate_timestamp")
            subs.append(0.0)
        else:
            subs.append(1.0)
        self._seen_timestamps[nid].add(ts)
        # Keep set bounded
        if len(self._seen_timestamps[nid]) > 1000:
            oldest = sorted(self._seen_timestamps[nid])[:500]
            self._seen_timestamps[nid] -= set(oldest)

        # --- Impersonation detection (GPS jump) ---
        history = self._gps_history[nid]
        if len(history) >= 2:
            prev = history[-1]
            dlat = abs(gps[0] - prev[0])
            dlon = abs(gps[1] - prev[1])
            jump_deg = math.sqrt(dlat ** 2 + dlon ** 2)
            if jump_deg > 5.0:  # > ~500 km jump in 1 second
                threats.append(f"impersonation_gps_jump_{jump_deg:.1f}deg")
                subs.append(0.0)
            elif jump_deg > 2.0:
                threats.append(f"suspicious_gps_jump_{jump_deg:.1f}deg")
                subs.append(0.4)
            else:
                subs.append(1.0)
        else:
            subs.append(1.0)
        history.append(gps)

        # --- Traffic pattern deviation ---
        th = self._traffic_history[nid]
        if len(th) >= 5:
            anomalous_count = sum(1 for t in th if t == "anomalous")
            anomalous_ratio = anomalous_count / len(th)
            if traffic == "anomalous":
                if anomalous_ratio > 0.3:
                    subs.append(0.2)
                    threats.append("persistent_anomalous_traffic")
                else:
                    subs.append(0.5)
                    threats.append("traffic_pattern_anomalous")
            elif traffic == "burst":
                subs.append(0.8)
            else:
                subs.append(1.0)
        else:
            # Short history -- still flag anomalous traffic
            if traffic == "anomalous":
                subs.append(0.5)
                threats.append("traffic_pattern_anomalous")
            else:
                subs.append(1.0)
        th.append(traffic)

        score = sum(subs) / len(subs) if subs else 1.0
        return score, threats

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(self, telemetry: Dict) -> Dict[str, Any]:
        """
        Evaluate a single telemetry record and return decision dict.
        """
        t0 = time.perf_counter()
        layer = telemetry["layer"]
        nid = telemetry["node_id"]
        is_spoofed = telemetry.get("is_spoofed", False)

        # Get the known node (if any)
        node = self.nodes.get(nid)

        w1, w2, w3 = WEIGHTS.get(layer, (0.33, 0.33, 0.34))

        loc_score, loc_threats = self._location_score(telemetry, node)
        vel_score, vel_threats = self._velocity_score(telemetry)
        id_score, id_threats = self._identity_score(telemetry)

        all_threats = loc_threats + vel_threats + id_threats
        raw = w1 * loc_score + w2 * vel_score + w3 * id_score
        trust_score = round(raw * 100, 2)

        if trust_score >= ALLOW_THRESHOLD:
            decision = "ALLOW"
        elif trust_score >= CHALLENGE_THRESHOLD:
            decision = "CHALLENGE"
        else:
            decision = "DENY"

        threat_detected = len(all_threats) > 0
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        # Update counters
        self.total_evaluated += 1
        if is_spoofed:
            self.spoofed_total += 1
        if threat_detected:
            self.threats_detected += 1
            if is_spoofed:
                self.spoofed_caught += 1

        return {
            "node_id": nid,
            "layer": layer,
            "trust_score": trust_score,
            "decision": decision,
            "threat_detected": threat_detected,
            "threat_type": "; ".join(all_threats) if all_threats else None,
            "latency_ms": latency_ms,
            "location_score": round(loc_score, 3),
            "velocity_score": round(vel_score, 3),
            "identity_score": round(id_score, 3),
            "is_spoofed": is_spoofed,
            "timestamp": telemetry["timestamp"],
        }

    @property
    def detection_rate(self) -> float:
        if self.spoofed_total == 0:
            return 100.0
        return round(self.spoofed_caught / self.spoofed_total * 100, 2)
