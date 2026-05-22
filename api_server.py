"""
REST API Server -- Module 4
===========================
FastAPI app with background simulation loop.
Run: uvicorn api_server:app --reload --port 8000
"""

import asyncio
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from ntn_simulator import NTNSimulator
from analytics_engine import BehaviouralAnalyticsEngine
from pdp_engine import PolicyDecisionPoint

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
simulator = NTNSimulator()
analytics = BehaviouralAnalyticsEngine(simulator.nodes, simulator.start_time)
pdp = PolicyDecisionPoint()

telemetry_buffer: List[Dict] = []
decision_buffer: List[Dict] = []
BUFFER_MAX = 200

start_time = time.time()
sim_running = False


# ---------------------------------------------------------------------------
# Background simulation loop
# ---------------------------------------------------------------------------
async def simulation_loop():
    """Run the NTN simulation, analytics, and PDP enforcement every second."""
    global sim_running
    sim_running = True
    while sim_running:
        try:
            records = simulator.generate_tick()
            for rec in records:
                # Analytics evaluation
                decision = analytics.evaluate(rec)
                # PDP enforcement
                enforcement = pdp.enforce(decision)
                # Merge for API
                merged = {**decision, **enforcement}

                # Buffer management
                telemetry_buffer.append(rec)
                decision_buffer.append(merged)

            # Keep buffers bounded
            if len(telemetry_buffer) > BUFFER_MAX:
                del telemetry_buffer[: len(telemetry_buffer) - BUFFER_MAX]
            if len(decision_buffer) > BUFFER_MAX:
                del decision_buffer[: len(decision_buffer) - BUFFER_MAX]

        except Exception as e:
            print(f"[SIM ERROR] {e}")

        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background simulation on startup."""
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
    title="BZT-NTN Security Framework",
    description="Behavioural Zero-Trust Security Framework for 5G NTN",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS -- allow dashboard.html from file:// and any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    uptime = round(time.time() - start_time, 1)
    return {
        "status": "running",
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
    return pdp.get_metrics(analytics)


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


@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard HTML file."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content)
    return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
def print_banner():
    banner = """
+==============================================================+
|   BZT-NTN Security Framework -- API Server                   |
|--------------------------------------------------------------|
|   Dashboard:   http://localhost:8000/dashboard                |
|   API Status:  http://localhost:8000/status                   |
|   Telemetry:   http://localhost:8000/telemetry                |
|   Decisions:   http://localhost:8000/decisions                |
|   Metrics:     http://localhost:8000/metrics                  |
|   Attacks:     http://localhost:8000/attacks                  |
|   Cache State: http://localhost:8000/cache_state              |
|   Handover:    http://localhost:8000/handover_log             |
|--------------------------------------------------------------|
|   POST /inject_attack  {"type":"velocity_spoof"}             |
|                        {"type":"replay"}                     |
|                        {"type":"impersonation"}              |
+==============================================================+
"""
    print(banner)
