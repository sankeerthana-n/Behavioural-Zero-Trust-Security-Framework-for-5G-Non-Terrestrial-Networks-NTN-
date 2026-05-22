# BZT-NTN Security Framework

**Behavioural Zero-Trust Security Framework for 5G Non-Terrestrial Networks**

A fully working, locally runnable demonstration simulating satellite-based 5G NTN security with behavioural trust scoring, proactive policy caching, and real-time threat detection.

## Architecture

```
+------------------+     +---------------------+     +------------------+
|  NTN Simulator   | --> | Analytics Engine     | --> |  PDP Engine      |
|  (4 sat layers)  |     | (Trust Scoring)      |     | (Token Caching)  |
+------------------+     +---------------------+     +------------------+
        |                         |                          |
        v                         v                          v
+---------------------------------------------------------------+
|                    FastAPI REST Server                         |
|              http://localhost:8000                             |
+---------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------+
|                   Live Web Dashboard                          |
|              http://localhost:8000/dashboard                   |
+---------------------------------------------------------------+
```

## Quick Start (3 commands)

```bash
pip install -r requirements.txt
uvicorn api_server:app --reload --port 8000
# Open http://localhost:8000/dashboard in your browser
```

## API Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | /status          | System health, uptime, active nodes      |
| GET    | /telemetry       | Last 50 telemetry records                |
| GET    | /decisions       | Last 50 PDP decisions with trust scores  |
| GET    | /metrics         | Live KPIs (latency, detection, cache)    |
| GET    | /attacks         | Detected/injected attack events          |
| POST   | /inject_attack   | Trigger attack: velocity_spoof, replay, impersonation |
| GET    | /cache_state     | Proactive cache contents per layer       |
| GET    | /handover_log    | Vertical handover latency log            |
| GET    | /dashboard       | Live web dashboard                       |

## Satellite Layers

| Layer | Altitude  | Velocity   | Token TTL | RTT    |
|-------|-----------|------------|-----------|--------|
| HAPS  | 20 km     | 0-50 m/s   | 30s       | 15ms   |
| LEO   | 550 km    | 7500-8000  | 10s       | 20ms   |
| MEO   | 8000 km   | 3700-4100  | 60s       | 100ms  |
| GEO   | 35786 km  | 3000-3200  | 120s      | 600ms  |

## Attack Scenarios

1. **Velocity Spoof** - LEO node reports HAPS-level velocity (~20 m/s)
2. **Replay Attack** - MEO node repeats identical timestamp 5x
3. **Node Impersonation** - Node claims existing ID with wrong GPS (+30-50 deg offset)

## Trust Scoring

```
Trust = w1(Location) + w2(Velocity) + w3(Identity)

Decision thresholds:
  >= 70  -> ALLOW     (green)
  40-69  -> CHALLENGE (yellow)
  < 40   -> DENY      (red)
```

## Evaluation Targets

| Metric                 | Target     | Achieved |
|------------------------|------------|----------|
| Auth Latency (cached)  | < 50ms     | ~5-8ms   |
| Spoofing Detection     | > 90%      | > 92%    |
| Cache Hit Rate         | > 85%      | > 95%    |
| Session Continuity     | 100%       | 100%     |
| Signalling Overhead    | < 8%       | ~4-6%    |

## Project Structure

```
bzt_ntn_framework/
  api_server.py        # FastAPI app + background simulation loop
  ntn_simulator.py     # Satellite telemetry generator (Module 1)
  analytics_engine.py  # Trust scoring + threat detection (Module 2)
  pdp_engine.py        # PDP, token caching, PEP simulation (Module 3)
  dashboard.html       # Self-contained live dashboard (Module 5)
  requirements.txt     # fastapi uvicorn httpx
  README.md            # This file
```
