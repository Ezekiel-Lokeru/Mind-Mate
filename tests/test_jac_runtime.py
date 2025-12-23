# tests/test_jac_runtime.py
import os
from datetime import datetime, timedelta, timezone

from backend.jac.engine import init_engine, load_walkers_module, log_mood, trend_analyzer
from jaclang.runtimelib.runtime import JacRuntime

BASE = os.path.join(os.path.dirname(__file__), "..", "backend", "jac")

def test_init_and_load():
    init_engine(base_path=BASE)
    mod = load_walkers_module(BASE)
    assert mod is not None
    assert "walkers" in JacRuntime.loaded_modules

def test_log_mood_creates_nodes():
    init_engine(base_path=BASE)
    load_walkers_module(BASE)
    entry = {
        "user_id": "u1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": 0.5,
        "tags": ["anxiety", "sadness"],
        "text": "I'm stressed about a deadline.",
    }
    res = log_mood(entry)
    assert "journal_id" in res
    assert isinstance(res["created_emotions"], list)

def test_trend_analyzer_sets_trend():
    init_engine(base_path=BASE)
    load_walkers_module(BASE)
    now = datetime.now(timezone.utc)
    e1 = {"user_id": "u2", "timestamp": now.isoformat(), "tags": ["joy"], "text": "I had a good walk"}
    log_mood(e1)
    out = trend_analyzer(window_days=1, compare_days=1)
    assert isinstance(out.get("current"), dict)

def test_trend_detection_rising():
    init_engine(base_path=BASE)
    load_walkers_module(BASE)
    now = datetime.now(timezone.utc)
    d0 = (now - timedelta(days=2)).isoformat()
    d1 = (now - timedelta(days=1)).isoformat()
    d2 = now.isoformat()

    # day 0: 1 entry, day1: 2 entries, day2: 4 entries -> rising
    log_mood({"user_id":"u", "timestamp":d0, "tags":["anxiety"], "text":""})
    log_mood({"user_id":"u", "timestamp":d1, "tags":["anxiety","anxiety"], "text":""})
    log_mood({"user_id":"u", "timestamp":d2, "tags":["anxiety","anxiety","anxiety","anxiety"], "text":""})

    out = trend_analyzer(window_days=3, compare_days=3)
    assert "anxiety" in out["stats"]
    assert out["stats"]["anxiety"]["trend"] == "rising"