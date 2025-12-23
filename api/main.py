"""Minimal FastAPI app wiring the Jac walkers and byLLM stubs.

Run: pip install fastapi uvicorn pydantic httpx
Start: uvicorn api.main:app --reload --port 8000
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone

from backend.byllm.interpret import interpret_input
from backend.byllm.craft import craft_response

# Jac engine integration
from backend.jac.engine import init_engine, load_walkers_module, log_mood, trend_analyzer
from jaclang.runtimelib.runtime import JacRuntime

app = FastAPI(title="Mind-Mate API")

# initialize Jac runtime on startup
@app.on_event("startup")
def startup():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "jac"))
    init_engine(base_path=base_path)
    load_walkers_module(base_path)

# In-memory store for prototype purposes
DB = {"journal": [], "events": []}


class EntryIn(BaseModel):
    user_id: str
    score: Optional[float] = None
    tags: Optional[List[str]] = []
    text: Optional[str] = None


class ResponseOut(BaseModel):
    message: str
    suggestions: List[dict]
    journal_id: Optional[str] = None


@app.post("/entry", response_model=ResponseOut)
def post_entry(entry: EntryIn):
    # 1. store entry in local DB with timezone-aware timestamp
    data = entry.dict()
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    DB["journal"].append(data)

    # Ensure Jac runtime is initialized (idempotent; covers test client case)
    if JacRuntime.exec_ctx is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "jac"))
        init_engine(base_path=base_path)
        load_walkers_module(base_path)

    # 2. persist to Jac runtime (engine) and obtain interpretation
    try:
        log_res = log_mood(data)  # persists JournalEntry and Emotion nodes in Jac
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Jac engine error: {e}")

    # 3. run trend analyzer to update Emotion.trend properties
    trends = trend_analyzer()

    # 4. interpret input and craft response
    interpretation = interpret_input(entry.text or "")
    context = {
        "emotions": interpretation.get("primary_emotions", []),
        "safety_flags": interpretation.get("safety_flags", {}),
        "trends": trends.get("current", {}),
    }

    resp = craft_response({**context, **{"entry": data}})
    return ResponseOut(
        message=resp["message"],
        suggestions=resp["suggestions"],
        journal_id=log_res.get("journal_id"),
    )


@app.get("/trends")
def get_trends():
    # Ensure Jac runtime available and return simple trend summary
    if JacRuntime.exec_ctx is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "jac"))
        init_engine(base_path=base_path)
        load_walkers_module(base_path)
    return {"status": "ok", "trends": trend_analyzer().get("current", {})}