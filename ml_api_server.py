"""
ML-Enhanced REST API Server -- Module 4B
=========================================
FastAPI app using the ML-powered analytics engine.
Runs on port 8001 (original server stays on port 8000).

Run: uvicorn ml_api_server:app --reload --port 8001
"""

import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ntn_simulator import NTNSimulator
from ml_analytics_engine import MLAnalyticsEngine
from pdp_engine import PolicyDecisionPoint

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
simulator = NTNSimulator()
ml_analytics = MLAnalyticsEngine(simulator.nodes, simulator.start_time)
pdp = PolicyDecisionPoint()

telemetry_buffer: List[Dict] = []
decision_buffer: List[Dict] = []
BUFFER_MAX = 200

start_time = time.time()
sim_running = False


# ---------------------------------------------------------------------------
# Background simulation loop (identical to original, but uses ML engine)
# ---------------------------------------------------------------------------
async def simulation_loop():
    """Run the NTN simulation with ML analytics every second."""
    global sim_running
    sim_running = True
    while sim_running:
        try:
            records = simulator.generate_tick()
            for rec in records:
                # ML analytics evaluation (hybrid ML + physics)
                decision = ml_analytics.evaluate(rec)
                # PDP enforcement (unchanged)
                enforcement = pdp.enforce(decision)
                # Merge for API
                merged = {**decision, **enforcement}

                telemetry_buffer.append(rec)
                decision_buffer.append(merged)

            if len(telemetry_buffer) > BUFFER_MAX:
                del telemetry_buffer[: len(telemetry_buffer) - BUFFER_MAX]
            if len(decision_buffer) > BUFFER_MAX:
                del decision_buffer[: len(decision_buffer) - BUFFER_MAX]

        except Exception as e:
            print(f"[ML-SIM ERROR] {e}")

        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print_banner()
    task = asyncio.create_task(simulation_loop())
    yield
    global sim_running
    sim_running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="BZT-NTN ML Security Framework",
    description="ML-Enhanced Behavioural Zero-Trust Security Framework for 5G NTN",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints (same as original + ML extras)
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    uptime = round(time.time() - start_time, 1)
    return {
        "status": "running",
        "mode": "ML-Enhanced (Isolation Forest)",
        "uptime_seconds": uptime,
        "active_nodes": list(simulator.nodes.keys()),
        "simulation_tick": simulator.tick,
        "total_telemetry": len(telemetry_buffer),
        "total_decisions": len(decision_buffer),
    }


@app.get("/telemetry")
async def telemetry():
    return telemetry_buffer[-50:]


@app.get("/decisions")
async def decisions():
    return decision_buffer[-50:]


@app.get("/metrics")
async def metrics():
    base = pdp.get_metrics(ml_analytics)
    # Add ML-specific counters
    base["ml_catches"] = ml_analytics.ml_catches
    base["physics_catches"] = ml_analytics.physics_catches
    base["hybrid_catches"] = ml_analytics.hybrid_catches
    return base


@app.get("/attacks")
async def attacks():
    return simulator.attack_log


@app.post("/inject_attack")
async def inject_attack(body: Dict):
    attack_type = body.get("type", "")
    result = simulator.queue_attack(attack_type)
    return result


@app.get("/cache_state")
async def cache_state():
    return pdp.get_cache_state()


@app.get("/handover_log")
async def handover_log():
    return pdp.handover_log


@app.get("/ml_info")
async def ml_info():
    """Return ML model metadata."""
    return ml_analytics.model_info


@app.get("/dashboard")
async def dashboard():
    """Serve the ML dashboard HTML file."""
    html_path = Path(__file__).parent / "ml_dashboard.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content)
    return HTMLResponse("<h1>ml_dashboard.html not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
def print_banner():
    banner = """
+==============================================================+
|   BZT-NTN Security Framework -- ML-Enhanced API Server       |
|   [Isolation Forest Anomaly Detection]                       |
|--------------------------------------------------------------|
|   ML Dashboard:  http://localhost:8001/dashboard              |
|   API Status:    http://localhost:8001/status                 |
|   ML Model Info: http://localhost:8001/ml_info                |
|   Metrics:       http://localhost:8001/metrics                |
|   Decisions:     http://localhost:8001/decisions              |
|   Attacks:       http://localhost:8001/attacks                |
|--------------------------------------------------------------|
|   POST /inject_attack  {"type":"velocity_spoof"}             |
|                        {"type":"replay"}                     |
|                        {"type":"impersonation"}              |
+==============================================================+
"""
    print(banner)
