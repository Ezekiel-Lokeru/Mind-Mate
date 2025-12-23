# backend/jac/engine.py
"""Simple Jac runtime helpers for Mind-Mate (Python / jaclang)."""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from jaclang.compiler.program import JacProgram
from jaclang.runtimelib.runtime import JacRuntimeInterface as Jac, JacRuntime

from backend.byllm.interpret import interpret_input


BASE_MODULE = "walkers"


def init_engine(base_path: Optional[str] = None, session_file: Optional[str] = None):
    """Initialize Jac runtime and context."""
    if base_path:
        JacRuntime.set_base_path(base_path)
    Jac.attach_program(JacProgram())
    ctx = Jac.create_j_context(session=session_file)
    JacRuntime.set_context(ctx)


def load_walkers_module(base_path: Optional[str] = None):
    """Import the Jac `walkers.jac` module into the runtime."""
    if base_path is None:
        base_path = os.path.dirname(__file__)  # backend/jac
    result = Jac.jac_import(target=BASE_MODULE, base_path=base_path, lng="jac")
    return result[0] if result else None


def _ensure_module():
    """Make sure the walkers module is loaded."""
    if BASE_MODULE not in JacRuntime.loaded_modules:
        load_walkers_module(JacRuntime.base_path_dir)


def log_mood(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a mood entry.

    entry keys: user_id, timestamp (ISO str), score, tags (List[str]), text (str)
    """
    _ensure_module()
    module_name = BASE_MODULE

    # make JournalEntry node
    journal_props = {
        "id": entry.get("id") or f"je:{entry.get('timestamp')}",
        "timestamp": entry.get("timestamp"),
        "text": entry.get("text", ""),
        "moods_detected": entry.get("tags", []),
        "score": entry.get("score"),
        "user_id": entry.get("user_id"),
    }

    je_obj = Jac.spawn_node("JournalEntry", journal_props, module_name)
    je_anchor = je_obj.__jac__
    je_anchor.persistent = True
    JacRuntime.exec_ctx.mem.set(je_anchor)

    # Ensure Emotion nodes exist & update last_seen
    created_emotions = []
    for tag in entry.get("tags", []):
        found = None
        for anc in JacRuntime.exec_ctx.mem.query(lambda a: getattr(a, "archetype", None) is not None):
            try:
                arch = anc.archetype
                if arch.__class__.__name__ == "Emotion" and getattr(arch, "name", None) == tag:
                    found = anc
                    break
            except Exception:
                continue

        if not found:
            e_obj = Jac.spawn_node(
                "Emotion",
                {"name": tag, "valence": 0.0, "intensity": 0.0, "last_seen": entry.get("timestamp")},
                module_name,
            )
            e_anchor = e_obj.__jac__
            e_anchor.persistent = True
            JacRuntime.exec_ctx.mem.set(e_anchor)
            created_emotions.append(tag)
        else:
            found.archetype.last_seen = entry.get("timestamp")
            JacRuntime.exec_ctx.mem.set(found)

    # interpret free text via byLLM
    interp = {}
    if entry.get("text"):
        interp = interpret_input(entry["text"])

        for t in interp.get("triggers", []):
            trg = Jac.spawn_node("Trigger", {"name": t}, module_name)
            trg_anchor = trg.__jac__
            trg_anchor.persistent = True
            JacRuntime.exec_ctx.mem.set(trg_anchor)

    return {"journal_id": je_anchor.id.hex, "created_emotions": created_emotions, "interpretation": interp}


# --- Improved trend analyzer (Step 3) ---
def _linear_slope(y: List[float]) -> float:
    """Compute slope for y-values using simple least squares over indices [0..n-1]."""
    n = len(y)
    if n == 0:
        return 0.0
    xs = list(range(n))
    sx = sum(xs)
    sy = sum(y)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * yy for x, yy in zip(xs, y))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    slope = (n * sxy - sx * sy) / denom
    return slope


def trend_analyzer(window_days: int = 7, compare_days: int = 7):
    """
    Compute per-day counts and derive trend via linear slope normalization.

    Returns:
      { "current": {emotion: [counts per day]...},
        "prev": {...},
        "stats": { emotion: {slope, mean, norm_slope, trend}, ... } }
    """
    # Ensure engine/context present
    if JacRuntime.exec_ctx is None:
        return {"current": {}, "prev": {}, "stats": {}}

    now = datetime.now(timezone.utc)
    # Align start to beginning of day (UTC)
    window_start = (now - timedelta(days=window_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    compare_start = (window_start - timedelta(days=compare_days)).replace(hour=0, minute=0, second=0, microsecond=0)

    def counts_by_emotion(start: datetime, end: datetime, days: int) -> Dict[str, List[int]]:
        """Return a mapping emotion -> list of counts per day (length == days)."""
        counts_map: Dict[str, List[int]] = {}

        # pre-seed with existing emotion archetypes so all emotions appear
        for anc in JacRuntime.exec_ctx.mem.query(lambda a: getattr(a, "archetype", None) is not None):
            try:
                arch = anc.archetype
                if arch.__class__.__name__ == "Emotion":
                    counts_map.setdefault(getattr(arch, "name", None), [0] * days)
            except Exception:
                continue

        # iterate JournalEntry anchors and bin mood occurrences
        for anc in JacRuntime.exec_ctx.mem.query(lambda a: getattr(a, "archetype", None) is not None):
            try:
                arch = anc.archetype
                if arch.__class__.__name__ != "JournalEntry":
                    continue
                ts_raw = getattr(arch, "timestamp", None) if not isinstance(arch, dict) else arch.get("timestamp")
                if not ts_raw:
                    continue
                try:
                    t = datetime.fromisoformat(ts_raw)
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    t = t.astimezone(timezone.utc)
                except Exception:
                    continue
                if not (start <= t <= end):
                    continue
                day_idx = (t.replace(hour=0, minute=0, second=0, microsecond=0) - start).days
                if day_idx < 0 or day_idx >= days:
                    continue
                moods = getattr(arch, "moods_detected", None) or (arch.get("moods_detected") if isinstance(arch, dict) else [])
                for m in moods:
                    if not isinstance(m, str):
                        continue
                    counts_map.setdefault(m, [0] * days)[day_idx] += 1
            except Exception:
                continue
        return counts_map

    # gather counts
    current_counts = counts_by_emotion(window_start, now, window_days)
    prev_counts = counts_by_emotion(compare_start, window_start - timedelta(seconds=1), compare_days)

    stats: Dict[str, Dict[str, Any]] = {}
    for emotion, counts in current_counts.items():
        slope = _linear_slope(counts)
        mean = sum(counts) / max(1, len(counts))
        norm_slope = slope / max(1.0, mean)
        trend = "stable"
        if norm_slope > 0.15:
            trend = "rising"
        elif norm_slope < -0.15:
            trend = "falling"
        stats[emotion] = {"slope": slope, "mean": mean, "norm_slope": norm_slope, "trend": trend}

        # update Emotion node anchor with trend and score
        for anc in JacRuntime.exec_ctx.mem.query(lambda a: getattr(a, "archetype", None) is not None):
            try:
                arch = anc.archetype
                if arch.__class__.__name__ == "Emotion" and getattr(arch, "name", None) == emotion:
                    arch.trend = trend
                    arch.trend_score = norm_slope
                    JacRuntime.exec_ctx.mem.set(anc)
                    break
            except Exception:
                continue

    return {"current": current_counts, "prev": prev_counts, "stats": stats}