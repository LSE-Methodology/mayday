"""
FastAPI server for the Predict-Ship-Compete workshop.

Endpoints:
  - GET  /                     → landing page
  - GET  /sql                  → SQL query interface
  - GET  /dashboard            → live A/B test dashboard
  - POST /api/sql              → execute SQL query (read-only)
  - GET  /api/schema           → get database schema
  - POST /api/teams/{name}/register → register a team
  - POST /api/teams/{name}/model   → upload a model (pickle)
  - GET  /api/teams            → list all teams
  - GET  /api/leaderboard      → live leaderboard
  - POST /api/simulation/start → start the A/B test
  - POST /api/simulation/stop  → stop the A/B test
  - GET  /api/simulation/status → simulation status & metrics
"""

import asyncio
import cloudpickle
import io
import pickle
import sqlite3
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simulator import ABTestSimulator

DB_PATH = Path(__file__).parent / "data" / "workshop.db"
STATIC_DIR = Path(__file__).parent / "static"

# --- State ---
teams: dict[str, dict[str, Any]] = {}
simulator: ABTestSimulator | None = None

# Bound how many SQL queries can hit SQLite in parallel. Each query runs in a
# thread (asyncio.to_thread), so we get real concurrent disk reads (sqlite3
# releases the GIL during I/O) without flooding the threadpool — leaves CPU
# headroom for the simulation loop and other handlers.
_SQL_CONCURRENCY = 4
_sql_sem: asyncio.Semaphore | None = None

# Cache results for repeated identical queries. Workshop students all run the
# same canned training-pull from the notebook, so first student pays the cost
# and the rest get a near-instant response. Capped to bound memory.
_SQL_CACHE_MAX_ENTRIES = 8
_sql_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
_sql_cache_lock = threading.Lock()

# Per-key locks prevent thundering herd: 20 students hitting the same query
# simultaneously would otherwise all see the cache miss before any of them
# can populate, and each end up doing the full work. With this, only the
# first request actually computes; the others block briefly on the same lock,
# then read the result from the cache.
_sql_pending: dict[tuple[str, int], threading.Lock] = {}
_sql_pending_meta_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulator, _sql_sem
    _sql_sem = asyncio.Semaphore(_SQL_CONCURRENCY)
    if not DB_PATH.exists():
        print("ERROR: Database not found. Run `python generate_data.py` first.")
    else:
        simulator = ABTestSimulator(DB_PATH, teams)
        print(f"Server ready. Database: {DB_PATH}")
    yield
    if simulator and simulator.running:
        simulator.stop()


app = FastAPI(title="Predict-Ship-Compete", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Page routes ---

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return (STATIC_DIR / "index.html").read_text()

@app.get("/sql", response_class=HTMLResponse)
async def sql_page():
    return (STATIC_DIR / "sql.html").read_text()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    return (STATIC_DIR / "dashboard.html").read_text()

@app.get("/slides", response_class=HTMLResponse)
async def slides_page():
    return (STATIC_DIR / "slides.html").read_text()

@app.get("/hints/{phase}", response_class=HTMLResponse)
async def hints_page(phase: str):
    allowed = {"explore", "build", "optimise", "deploy"}
    if phase not in allowed:
        raise HTTPException(404, "Page not found")
    return (STATIC_DIR / "hints" / f"{phase}.html").read_text()


# --- SQL API ---

class SQLQuery(BaseModel):
    query: str
    limit: int = 1000

@app.post("/api/sql")
async def execute_sql(body: SQLQuery):
    """Execute a read-only SQL query against the workshop database."""
    query = body.query.strip().rstrip(";")
    if not query:
        raise HTTPException(400, "Empty query")

    # Block writes and dangerous operations
    first_word = query.split()[0].upper() if query.split() else ""
    blocked = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
               "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX"}
    if first_word in blocked:
        raise HTTPException(403, "Write operations are not allowed")

    # Block access to hidden tables
    if "_users_full" in query.lower():
        raise HTTPException(403, "Access denied to internal tables")

    try:
        async with _sql_sem:
            return await asyncio.to_thread(_run_sql_cached, query, body.limit)
    except sqlite3.OperationalError as e:
        raise HTTPException(400, f"SQL error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")


def _run_sql_cached(query: str, limit: int) -> dict[str, Any]:
    """Cached wrapper around _run_sql_blocking.

    SQLite is read-only and the data never changes during a workshop, so the
    same (query, limit) is guaranteed to return the same result. Multiple
    students hitting the same canonical pull get the cached response.
    Per-key locks prevent thundering herd on the first miss.
    """
    key = (query, limit)
    with _sql_cache_lock:
        cached = _sql_cache.get(key)
        if cached is not None:
            _sql_cache.move_to_end(key)
            return cached

    with _sql_pending_meta_lock:
        key_lock = _sql_pending.setdefault(key, threading.Lock())

    with key_lock:
        with _sql_cache_lock:
            cached = _sql_cache.get(key)
            if cached is not None:
                return cached

        result = _run_sql_blocking(query, limit)

        with _sql_cache_lock:
            _sql_cache[key] = result
            _sql_cache.move_to_end(key)
            while len(_sql_cache) > _SQL_CACHE_MAX_ENTRIES:
                _sql_cache.popitem(last=False)
        return result


def _run_sql_blocking(query: str, limit: int) -> dict[str, Any]:
    """Synchronous SQL execution — runs in a worker thread.

    Rows are returned as a list of tuples (not dicts) — much smaller JSON
    payload and faster to serialise. Both `pd.DataFrame(rows, columns=cols)`
    and the SQL Explorer (sql.html) consume them positionally.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM ({query}) LIMIT {limit}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()
    return {"columns": columns, "rows": rows, "count": len(rows),
            "truncated": len(rows) == limit}


@app.get("/api/schema")
async def get_schema():
    """Return the database schema (tables and columns)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get all public tables
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'
        ORDER BY name
    """)
    tables = {}
    for (table_name,) in cursor.fetchall():
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        tables[table_name] = {"columns": cols, "row_count": count}

    conn.close()
    return {"tables": tables}


# --- Team management ---

class TeamRegister(BaseModel):
    members: list[str] = []

@app.post("/api/teams/{team_name}/register")
async def register_team(team_name: str, body: TeamRegister):
    """Register a new team."""
    if team_name in teams:
        raise HTTPException(409, f"Team '{team_name}' already exists")
    if len(team_name) > 30 or not team_name.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "Team name must be alphanumeric (with _ or -), max 30 chars")
    if len(teams) >= 20:
        raise HTTPException(400, "Maximum 20 teams reached")

    teams[team_name] = {
        "members": body.members,
        "model": None,
        "model_uploaded_at": None,
        "model_type": None,
        "metrics": {
            "impressions": 0,
            "clicks": 0,
            "conversions": 0,
            "revenue": 0.0,
            "total_latency_ms": 0.0,
            "latency_violations": 0,
            "errors": 0,
        },
        "history": [],  # time-series for the dashboard
    }
    return {"status": "registered", "team": team_name}


@app.get("/api/teams")
async def list_teams():
    """List all registered teams."""
    return {
        name: {
            "members": t["members"],
            "has_model": t["model"] is not None,
            "model_type": t["model_type"],
            "model_uploaded_at": t["model_uploaded_at"],
        }
        for name, t in teams.items()
    }


@app.post("/api/teams/{team_name}/model")
async def upload_model(team_name: str, model: UploadFile = File(...)):
    """Upload a pickled sklearn model for a team."""
    if team_name not in teams:
        raise HTTPException(404, f"Team '{team_name}' not found. Register first.")

    content = await model.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(400, "Model too large (max 50MB)")

    try:
        model_obj = cloudpickle.loads(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to unpickle model: {e}")

    # Validate the model has a predict method
    if not hasattr(model_obj, "predict"):
        raise HTTPException(400, "Model must have a .predict() method")

    # Quick validation: try scoring a dummy row using the simulator's actual features
    try:
        dummy = simulator.build_validation_dummy()
        start = time.perf_counter()
        preds = model_obj.predict(dummy)
        elapsed = (time.perf_counter() - start) * 1000
        pred_value = float(preds[0])
        # NaN/inf would slip past `float()` and then break JSON serialisation,
        # so flag them as a validation failure with a clear message.
        if not np.isfinite(pred_value):
            raise ValueError(
                f"predict() returned a non-finite value ({pred_value!r}). "
                "Check for NaN/inf in your training data or features.")
    except Exception as e:
        raise HTTPException(400, f"Model validation failed: {e}\n\n"
                           "Your model must accept a DataFrame with the feature columns "
                           "described in the student notebook. Check your feature names.")

    model_type = type(model_obj).__name__
    teams[team_name]["model"] = model_obj
    teams[team_name]["model_uploaded_at"] = pd.Timestamp.now().isoformat()
    teams[team_name]["model_type"] = model_type

    # Reset metrics for this team
    teams[team_name]["metrics"] = {
        "impressions": 0, "clicks": 0, "conversions": 0,
        "revenue": 0.0, "total_latency_ms": 0.0,
        "latency_violations": 0, "errors": 0,
    }
    teams[team_name]["history"] = []

    return {
        "status": "uploaded",
        "team": team_name,
        "model_type": model_type,
        "validation_prediction": pred_value,
        "validation_latency_ms": round(elapsed, 2),
    }


# --- Simulation control ---

@app.post("/api/simulation/start")
async def start_simulation(requests_per_second: int = 20, latency_budget_ms: int = 50):
    """Start the A/B test simulation."""
    if simulator is None:
        raise HTTPException(500, "Simulator not initialized")

    teams_with_models = [t for t in teams if teams[t]["model"] is not None]
    if len(teams_with_models) < 1:
        raise HTTPException(400, "At least 1 team must have a model uploaded")

    simulator.latency_budget_ms = latency_budget_ms
    simulator.requests_per_second = requests_per_second
    asyncio.create_task(simulator.run())

    return {"status": "started", "teams": teams_with_models,
            "requests_per_second": requests_per_second,
            "latency_budget_ms": latency_budget_ms}


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop the A/B test simulation."""
    if simulator:
        simulator.stop()
    return {"status": "stopped"}


@app.get("/api/simulation/status")
async def simulation_status():
    """Get current simulation status and metrics for all teams."""
    leaderboard = []
    for name, team in teams.items():
        m = team["metrics"]
        leaderboard.append({
            "team": name,
            "has_model": team["model"] is not None,
            "model_type": team["model_type"],
            "impressions": m["impressions"],
            "clicks": m["clicks"],
            "conversions": m["conversions"],
            "revenue": round(m["revenue"], 2),
            "ctr": round(m["clicks"] / max(m["impressions"], 1), 4),
            "cvr": round(m["conversions"] / max(m["clicks"], 1), 4),
            "revenue_per_impression": round(m["revenue"] / max(m["impressions"], 1), 6),
            "avg_latency_ms": round(m["total_latency_ms"] / max(m["impressions"], 1), 2),
            "latency_violations": m["latency_violations"],
            "errors": m["errors"],
            "history": team["history"][-120:],  # last 120 data points
        })

    # Sort by revenue per impression (the true metric)
    leaderboard.sort(key=lambda x: x["revenue_per_impression"], reverse=True)

    return {
        "running": simulator.running if simulator else False,
        "total_requests": simulator.total_requests if simulator else 0,
        "leaderboard": leaderboard,
    }


@app.get("/api/leaderboard")
async def leaderboard():
    """Shortcut for dashboard polling."""
    return await simulation_status()


if __name__ == "__main__":
    import os
    import uvicorn
    # PaaS hosts (Render, Fly, Heroku, Railway) inject $PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
