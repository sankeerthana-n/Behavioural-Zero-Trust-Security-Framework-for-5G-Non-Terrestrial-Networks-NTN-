"""
Dataset Generator for ML-Based Anomaly Detection
=================================================
Generates a labeled CSV dataset of NTN satellite telemetry for training
an Isolation Forest anomaly detection model.

Output: dataset/ntn_telemetry_dataset.csv  (~50,000 rows)

Features extracted:
  1. velocity_ms          - reported satellite velocity
  2. altitude_km          - reported altitude
  3. doppler_hz           - Doppler shift
  4. rssi_dbm             - signal strength
  5. velocity_deviation   - |reported - expected| velocity for layer
  6. gps_distance_km      - haversine distance from expected orbital position
  7. timestamp_delta_ms   - time since last telemetry from same node (0=replay)
  8. traffic_encoded      - normal=0, burst=1, anomalous=2
  9. layer_encoded         - HAPS=0, LEO=1, MEO=2, GEO=3

Label:
  is_spoofed             - 0 (normal) or 1 (attack) -- ground truth

Usage:
  python generate_dataset.py
"""

import csv
import math
import os
import random
import time
from collections import defaultdict

# ---- Deterministic seed everywhere ----
random.seed(42)

# ---- Import layer config from existing simulator ----
from ntn_simulator import LAYER_CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAYERS = ["HAPS", "LEO", "MEO", "GEO"]
LAYER_TO_INT = {"HAPS": 0, "LEO": 1, "MEO": 2, "GEO": 3}
TRAFFIC_TO_INT = {"normal": 0, "burst": 1, "anomalous": 2}

NODE_BASES = {
    "HAPS_01": {"layer": "HAPS", "lat": 35.0, "lon": -5.0},
    "LEO_01":  {"layer": "LEO",  "lat": 10.0, "lon": 30.0},
    "MEO_01":  {"layer": "MEO",  "lat": -20.0, "lon": 60.0},
    "GEO_01":  {"layer": "GEO",  "lat": 0.0,  "lon": 100.0},
}

TOTAL_NORMAL = 40000
TOTAL_ATTACK = 10000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ntn_telemetry_dataset.csv")

FEATURE_COLUMNS = [
    "node_id", "layer", "velocity_ms", "altitude_km", "doppler_hz",
    "rssi_dbm", "velocity_deviation", "gps_distance_km",
    "timestamp_delta_ms", "traffic_encoded", "layer_encoded", "is_spoofed",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km between two GPS coordinates."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371 * c


def expected_position(layer, base_lat, base_lon, tick):
    """Compute expected orbital position at a given tick."""
    cfg = LAYER_CONFIG[layer]
    angular_rate = cfg["velocity_ms"] / ((6371 + cfg["alt_km"]) * 1000)
    lat = base_lat + 20.0 * math.sin(angular_rate * tick)
    lon = base_lon + 20.0 * math.cos(angular_rate * tick)
    lat = max(-90, min(90, lat))
    lon = ((lon + 180) % 360) - 180
    return lat, lon


def generate_normal_record(node_id, layer, base_lat, base_lon, tick, last_ts):
    """Generate a single normal (non-spoofed) telemetry feature row."""
    cfg = LAYER_CONFIG[layer]

    # Realistic values within physical bounds
    velocity = cfg["velocity_ms"] + random.uniform(-10, 10)
    doppler = random.uniform(*cfg["doppler_range"])
    rssi = cfg["rssi_base"] + random.uniform(-3, 3)
    traffic = random.choices(["normal", "burst", "anomalous"], weights=[0.88, 0.10, 0.02], k=1)[0]

    # Expected position
    exp_lat, exp_lon = expected_position(layer, base_lat, base_lon, tick)
    # Reported position: close to expected (small noise)
    rep_lat = exp_lat + random.uniform(-0.5, 0.5)
    rep_lon = exp_lon + random.uniform(-0.5, 0.5)

    gps_dist = haversine_km(exp_lat, exp_lon, rep_lat, rep_lon)
    vel_dev = abs(velocity - cfg["velocity_ms"])

    # Timestamp: monotonically increasing with ~1000ms intervals + jitter
    ts = last_ts + 1000 + random.randint(-50, 50)
    ts_delta = ts - last_ts

    return {
        "node_id": node_id,
        "layer": layer,
        "velocity_ms": round(velocity, 2),
        "altitude_km": cfg["alt_km"],
        "doppler_hz": round(doppler, 2),
        "rssi_dbm": round(rssi, 2),
        "velocity_deviation": round(vel_dev, 2),
        "gps_distance_km": round(gps_dist, 4),
        "timestamp_delta_ms": ts_delta,
        "traffic_encoded": TRAFFIC_TO_INT[traffic],
        "layer_encoded": LAYER_TO_INT[layer],
        "is_spoofed": 0,
    }, ts


def generate_velocity_spoof(node_id, layer, base_lat, base_lon, tick, last_ts):
    """LEO claims HAPS-level velocity."""
    cfg = LAYER_CONFIG[layer]

    # Spoofed velocity: way out of bounds for the layer
    if layer == "LEO":
        velocity = random.uniform(5, 40)   # HAPS-level, impossible for LEO
    elif layer == "MEO":
        velocity = random.uniform(7500, 8200)  # LEO-level, impossible for MEO
    else:
        velocity = cfg["velocity_ms"] * random.uniform(0.1, 0.3)  # drastically low

    doppler = random.uniform(*cfg["doppler_range"])
    rssi = cfg["rssi_base"] + random.uniform(-5, 5)

    exp_lat, exp_lon = expected_position(layer, base_lat, base_lon, tick)
    rep_lat = exp_lat + random.uniform(-1, 1)
    rep_lon = exp_lon + random.uniform(-1, 1)
    gps_dist = haversine_km(exp_lat, exp_lon, rep_lat, rep_lon)
    vel_dev = abs(velocity - cfg["velocity_ms"])

    ts = last_ts + 1000 + random.randint(-50, 50)
    ts_delta = ts - last_ts

    return {
        "node_id": node_id,
        "layer": layer,
        "velocity_ms": round(velocity, 2),
        "altitude_km": cfg["alt_km"],
        "doppler_hz": round(doppler, 2),
        "rssi_dbm": round(rssi, 2),
        "velocity_deviation": round(vel_dev, 2),
        "gps_distance_km": round(gps_dist, 4),
        "timestamp_delta_ms": ts_delta,
        "traffic_encoded": TRAFFIC_TO_INT["anomalous"],
        "layer_encoded": LAYER_TO_INT[layer],
        "is_spoofed": 1,
    }, ts


def generate_replay(node_id, layer, base_lat, base_lon, tick, last_ts):
    """Replay attack: timestamp_delta = 0 (duplicate timestamp)."""
    cfg = LAYER_CONFIG[layer]
    velocity = cfg["velocity_ms"] + random.uniform(-10, 10)
    doppler = random.uniform(*cfg["doppler_range"])
    rssi = cfg["rssi_base"] + random.uniform(-3, 3)

    exp_lat, exp_lon = expected_position(layer, base_lat, base_lon, tick)
    rep_lat = exp_lat + random.uniform(-0.5, 0.5)
    rep_lon = exp_lon + random.uniform(-0.5, 0.5)
    gps_dist = haversine_km(exp_lat, exp_lon, rep_lat, rep_lon)
    vel_dev = abs(velocity - cfg["velocity_ms"])

    # Key: timestamp delta is 0 (exact duplicate)
    ts_delta = 0

    return {
        "node_id": node_id,
        "layer": layer,
        "velocity_ms": round(velocity, 2),
        "altitude_km": cfg["alt_km"],
        "doppler_hz": round(doppler, 2),
        "rssi_dbm": round(rssi, 2),
        "velocity_deviation": round(vel_dev, 2),
        "gps_distance_km": round(gps_dist, 4),
        "timestamp_delta_ms": ts_delta,
        "traffic_encoded": TRAFFIC_TO_INT["anomalous"],
        "layer_encoded": LAYER_TO_INT[layer],
        "is_spoofed": 1,
    }, last_ts  # timestamp doesn't advance


def generate_impersonation(node_id, layer, base_lat, base_lon, tick, last_ts):
    """Impersonation: GPS is wildly off from expected position."""
    cfg = LAYER_CONFIG[layer]
    velocity = cfg["velocity_ms"] + random.uniform(-15, 15)
    doppler = random.uniform(*cfg["doppler_range"])
    rssi = cfg["rssi_base"] + random.uniform(-5, 5)

    exp_lat, exp_lon = expected_position(layer, base_lat, base_lon, tick)
    # GPS shifted by 30-50 degrees (thousands of km off)
    rep_lat = exp_lat + random.uniform(30, 50)
    rep_lon = exp_lon + random.uniform(30, 50)
    rep_lat = max(-90, min(90, rep_lat))
    rep_lon = ((rep_lon + 180) % 360) - 180
    gps_dist = haversine_km(exp_lat, exp_lon, rep_lat, rep_lon)
    vel_dev = abs(velocity - cfg["velocity_ms"])

    ts = last_ts + 1000 + random.randint(-50, 50)
    ts_delta = ts - last_ts

    return {
        "node_id": node_id,
        "layer": layer,
        "velocity_ms": round(velocity, 2),
        "altitude_km": cfg["alt_km"],
        "doppler_hz": round(doppler, 2),
        "rssi_dbm": round(rssi, 2),
        "velocity_deviation": round(vel_dev, 2),
        "gps_distance_km": round(gps_dist, 4),
        "timestamp_delta_ms": ts_delta,
        "traffic_encoded": TRAFFIC_TO_INT["anomalous"],
        "layer_encoded": LAYER_TO_INT[layer],
        "is_spoofed": 1,
    }, ts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    node_ids = list(NODE_BASES.keys())

    # ---- Generate 40,000 normal records ----
    print("[1/3] Generating 40,000 normal telemetry records...")
    records_per_node = TOTAL_NORMAL // len(node_ids)
    for nid in node_ids:
        info = NODE_BASES[nid]
        ts = int(time.time() * 1000)
        for tick in range(1, records_per_node + 1):
            rec, ts = generate_normal_record(
                nid, info["layer"], info["lat"], info["lon"], tick, ts
            )
            rows.append(rec)

    # ---- Generate 10,000 attack records ----
    print("[2/3] Generating 10,000 attack telemetry records...")
    attack_types = [generate_velocity_spoof, generate_replay, generate_impersonation]
    attacks_per_type = TOTAL_ATTACK // len(attack_types)

    for attack_fn in attack_types:
        for i in range(attacks_per_type):
            nid = random.choice(node_ids)
            info = NODE_BASES[nid]
            ts = int(time.time() * 1000) + i * 1000
            tick = random.randint(1, 10000)
            rec, _ = attack_fn(nid, info["layer"], info["lat"], info["lon"], tick, ts)
            rows.append(rec)

    # Remaining attacks to reach exactly 10,000
    remaining = TOTAL_ATTACK - (attacks_per_type * len(attack_types))
    for i in range(remaining):
        nid = random.choice(node_ids)
        info = NODE_BASES[nid]
        ts = int(time.time() * 1000) + i * 1000
        tick = random.randint(1, 10000)
        rec, _ = random.choice(attack_types)(
            nid, info["layer"], info["lat"], info["lon"], tick, ts
        )
        rows.append(rec)

    # ---- Shuffle with deterministic seed ----
    random.shuffle(rows)

    # ---- Write CSV ----
    print(f"[3/3] Writing {len(rows)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Summary stats ----
    normal_count = sum(1 for r in rows if r["is_spoofed"] == 0)
    attack_count = sum(1 for r in rows if r["is_spoofed"] == 1)
    print(f"\nDataset generated successfully!")
    print(f"  Total records:  {len(rows)}")
    print(f"  Normal records: {normal_count}")
    print(f"  Attack records: {attack_count}")
    print(f"  Saved to:       {OUTPUT_FILE}")
    print(f"  Columns:        {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
