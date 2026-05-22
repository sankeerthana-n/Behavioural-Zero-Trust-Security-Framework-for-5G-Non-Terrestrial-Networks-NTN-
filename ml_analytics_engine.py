"""
ML-Enhanced Behavioural Analytics Engine -- Module 2B
=====================================================
Hybrid trust scoring combining Isolation Forest ML model with
physics-based rules.

  final_trust = 0.6 * ML_trust + 0.4 * Physics_trust

The ML model handles subtle statistical anomalies while the physics
rules catch obvious impossible-velocity/GPS attacks.
"""

import math
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

from ntn_simulator import LAYER_CONFIG, SatelliteNode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")

# ---------------------------------------------------------------------------
# Same layer weights as the original engine (for the physics component)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "HAPS": (0.5, 0.2, 0.3),
    "LEO":  (0.3, 0.5, 0.2),
    "MEO":  (0.2, 0.3, 0.5),
    "GEO":  (0.2, 0.2, 0.6),
}

# Feature column order (must match training)
FEATURE_COLS = [
    "velocity_ms", "altitude_km", "doppler_hz", "rssi_dbm",
    "velocity_deviation", "gps_distance_km", "timestamp_delta_ms",
    "traffic_encoded", "layer_encoded",
]

LAYER_TO_INT = {"HAPS": 0, "LEO": 1, "MEO": 2, "GEO": 3}
TRAFFIC_TO_INT = {"normal": 0, "burst": 1, "anomalous": 2}

# Decision thresholds (same as original)
ALLOW_THRESHOLD = 70
CHALLENGE_THRESHOLD = 40


class MLAnalyticsEngine:
    """
    ML-enhanced analytics engine that combines Isolation Forest predictions
    with physics-based orbital verification.
    """

    def __init__(self, nodes: Dict[str, SatelliteNode], start_time: float):
        self.nodes = nodes
        self.start_time = start_time

        # Load trained ML model and scaler
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        # State tracking (same as original engine)
        self._seen_timestamps: Dict[str, set] = defaultdict(set)
        self._gps_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._traffic_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self._last_timestamp: Dict[str, int] = {}

        # Counters
        self.total_evaluated: int = 0
        self.threats_detected: int = 0
        self.spoofed_caught: int = 0
        self.spoofed_total: int = 0

        # ML-specific counters
        self.ml_catches: int = 0
        self.physics_catches: int = 0
        self.hybrid_catches: int = 0

        # Store model info for API
        self.model_info = {
            "algorithm": "Isolation Forest",
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination,
            "max_samples": self.model.max_samples,
            "features": FEATURE_COLS,
            "model_path": MODEL_PATH,
        }

    # ------------------------------------------------------------------
    # Feature extraction (same logic as generate_dataset.py)
    # ------------------------------------------------------------------
    def _extract_features(self, telemetry: Dict) -> np.ndarray:
        """Extract the 9 ML features from a telemetry record."""
        layer = telemetry["layer"]
        cfg = LAYER_CONFIG[layer]
        nid = telemetry["node_id"]

        velocity = telemetry["velocity_ms"]
        velocity_deviation = abs(velocity - cfg["velocity_ms"])

        # GPS distance from expected position
        node = self.nodes.get(nid)
        if node:
            elapsed = time.time() - self.start_time
            angular_rate = cfg["velocity_ms"] / ((6371 + cfg["alt_km"]) * 1000)
            exp_lat = node.base_lat + 20.0 * math.sin(angular_rate * elapsed)
            exp_lon = node.base_lon + 20.0 * math.cos(angular_rate * elapsed)
            exp_lat = max(-90, min(90, exp_lat))
            exp_lon = ((exp_lon + 180) % 360) - 180

            rep_lat = telemetry["gps"]["lat"]
            rep_lon = telemetry["gps"]["lon"]
            dlat = math.radians(rep_lat - exp_lat)
            dlon = math.radians(rep_lon - exp_lon)
            a = (math.sin(dlat / 2) ** 2
                 + math.cos(math.radians(exp_lat))
                 * math.cos(math.radians(rep_lat))
                 * math.sin(dlon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            gps_distance_km = 6371 * c
        else:
            gps_distance_km = 9999.0  # Unknown node = max anomaly

        # Timestamp delta
        ts = telemetry["timestamp"]
        last_ts = self._last_timestamp.get(nid, ts - 1000)
        timestamp_delta_ms = ts - last_ts
        self._last_timestamp[nid] = ts

        traffic_encoded = TRAFFIC_TO_INT.get(telemetry["traffic_pattern"], 2)
        layer_encoded = LAYER_TO_INT.get(layer, 0)

        features = np.array([[
            velocity,
            cfg["alt_km"],
            telemetry["doppler_hz"],
            telemetry["rssi_dbm"],
            velocity_deviation,
            gps_distance_km,
            timestamp_delta_ms,
            traffic_encoded,
            layer_encoded,
        ]], dtype=np.float64)

        return features, gps_distance_km, velocity_deviation

    # ------------------------------------------------------------------
    # Physics-based scoring (simplified from original engine)
    # ------------------------------------------------------------------
    def _physics_trust(self, telemetry: Dict, gps_dist_km: float,
                       vel_dev: float) -> Tuple[float, List[str]]:
        """
        Compute physics-based trust score (0-1) and threat list.
        Same logic as the original analytics_engine.py.
        """
        layer = telemetry["layer"]
        cfg = LAYER_CONFIG[layer]
        nid = telemetry["node_id"]
        threats: List[str] = []

        # --- Location score ---
        if gps_dist_km > 200:
            loc_score = 0.0
            threats.append("location_anomaly_gt_200km")
        elif gps_dist_km > 100:
            loc_score = 0.3
            threats.append("location_drift_gt_100km")
        elif gps_dist_km > 50:
            loc_score = 0.7
        else:
            loc_score = 1.0

        # --- Velocity score ---
        vmin, vmax = cfg["velocity_range"]
        v = telemetry["velocity_ms"]
        if v < vmin or v > vmax:
            deviation = (vmin - v) / vmin if v < vmin else (v - vmax) / vmax
            vel_score = max(0.0, 1.0 - deviation * 2)
            threats.append(f"impossible_physics_velocity_{v:.0f}ms")
        else:
            centre = (vmin + vmax) / 2
            spread = (vmax - vmin) / 2
            normalised = abs(v - centre) / spread if spread > 0 else 0
            vel_score = max(0.8, 1.0 - normalised * 0.2)

        # --- Identity score ---
        ts = telemetry["timestamp"]
        gps = (telemetry["gps"]["lat"], telemetry["gps"]["lon"])
        traffic = telemetry["traffic_pattern"]
        id_subs: List[float] = []

        # Replay detection
        if ts in self._seen_timestamps[nid]:
            threats.append("replay_attack_duplicate_timestamp")
            id_subs.append(0.0)
        else:
            id_subs.append(1.0)
        self._seen_timestamps[nid].add(ts)
        if len(self._seen_timestamps[nid]) > 1000:
            oldest = sorted(self._seen_timestamps[nid])[:500]
            self._seen_timestamps[nid] -= set(oldest)

        # GPS jump detection
        history = self._gps_history[nid]
        if len(history) >= 2:
            prev = history[-1]
            jump_deg = math.sqrt(
                (gps[0] - prev[0]) ** 2 + (gps[1] - prev[1]) ** 2
            )
            if jump_deg > 5.0:
                threats.append(f"impersonation_gps_jump_{jump_deg:.1f}deg")
                id_subs.append(0.0)
            elif jump_deg > 2.0:
                threats.append(f"suspicious_gps_jump_{jump_deg:.1f}deg")
                id_subs.append(0.4)
            else:
                id_subs.append(1.0)
        else:
            id_subs.append(1.0)
        history.append(gps)

        # Traffic pattern
        th = self._traffic_history[nid]
        if traffic == "anomalous":
            id_subs.append(0.5)
            threats.append("traffic_pattern_anomalous")
        elif traffic == "burst":
            id_subs.append(0.8)
        else:
            id_subs.append(1.0)
        th.append(traffic)

        id_score = sum(id_subs) / len(id_subs) if id_subs else 1.0

        # Weighted combination
        w1, w2, w3 = WEIGHTS.get(layer, (0.33, 0.33, 0.34))
        physics_trust = w1 * loc_score + w2 * vel_score + w3 * id_score

        return physics_trust, threats

    # ------------------------------------------------------------------
    # ML prediction
    # ------------------------------------------------------------------
    def _ml_predict(self, features: np.ndarray) -> Tuple[float, str]:
        """
        Run Isolation Forest prediction.
        Returns (ml_trust_score_0to1, prediction_label, raw_score).
        """
        features_scaled = self.scaler.transform(features)

        # decision_function: higher = more normal, lower = more anomalous
        raw_score = self.model.decision_function(features_scaled)[0]

        # Use a strong threshold instead of predict() to reduce false positives.
        # predict() uses the contamination parameter (0.15) which is calibrated
        # for the training set ratio, but in live mode almost all traffic is normal.
        # A score below -0.15 is a strong anomaly signal.
        ML_ANOMALY_THRESHOLD = -0.15
        is_anomaly = raw_score < ML_ANOMALY_THRESHOLD

        # Normalize decision score to 0-1 trust range.
        # Typical live normal scores are in [0.0, 0.15] range.
        # Attack scores are in [-0.3, -0.05] range.
        # Map so that 0.0 -> 0.5 (neutral), positive -> higher trust, negative -> lower.
        ml_trust = max(0.0, min(1.0, (raw_score + 0.2) / 0.4))

        label = "anomaly" if is_anomaly else "normal"
        return ml_trust, label, raw_score

    # ------------------------------------------------------------------
    # Main evaluation (hybrid ML + physics)
    # ------------------------------------------------------------------
    def evaluate(self, telemetry: Dict) -> Dict[str, Any]:
        """
        Evaluate telemetry using hybrid ML + physics scoring.
        """
        t0 = time.perf_counter()
        layer = telemetry["layer"]
        nid = telemetry["node_id"]
        is_spoofed = telemetry.get("is_spoofed", False)

        # Extract features for ML
        features, gps_dist_km, vel_dev = self._extract_features(telemetry)

        # Get ML prediction
        ml_trust, ml_label, ml_raw_score = self._ml_predict(features)

        # Get physics-based trust
        physics_trust, physics_threats = self._physics_trust(
            telemetry, gps_dist_km, vel_dev
        )

        # --- Hybrid combination ---
        # 60% ML + 40% Physics
        hybrid_trust = 0.6 * ml_trust + 0.4 * physics_trust
        trust_score = round(hybrid_trust * 100, 2)

        # Determine detection method
        ml_flagged = (ml_label == "anomaly")
        physics_flagged = len(physics_threats) > 0

        if ml_flagged and physics_flagged:
            detection_method = "hybrid"
            self.hybrid_catches += 1
        elif ml_flagged:
            detection_method = "ml"
            self.ml_catches += 1
        elif physics_flagged:
            detection_method = "physics"
            self.physics_catches += 1
        else:
            detection_method = "none"

        # Build threat list
        all_threats = list(physics_threats)
        if ml_flagged:
            all_threats.insert(0, f"ml_anomaly_detected(score={ml_raw_score:.3f})")

        # Decision
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
            "location_score": round(physics_trust, 3),  # reuse field name for compat
            "velocity_score": round(ml_trust, 3),        # reuse field name for compat
            "identity_score": round(hybrid_trust, 3),     # reuse field name for compat
            "is_spoofed": is_spoofed,
            "timestamp": telemetry["timestamp"],
            # ML-specific fields
            "ml_anomaly_score": round(ml_raw_score, 4),
            "ml_prediction": ml_label,
            "detection_method": detection_method,
        }

    @property
    def detection_rate(self) -> float:
        if self.spoofed_total == 0:
            return 100.0
        return round(self.spoofed_caught / self.spoofed_total * 100, 2)
