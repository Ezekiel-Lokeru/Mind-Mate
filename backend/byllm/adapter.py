from typing import Dict, Any, Optional
import os
import time
import threading
import collections
import json
import re
import logging


SYSTEM_PROMPT = (
    "You are a compassionate assistant. Given the user's short journal entry, "
    "respond with a JSON object only (no additional text) with the following keys:\n"
    "- primary_emotions: array of strings (primary emotions detected)\n"
    "- triggers: array of strings (situational triggers)\n"
    "- intensity: number between 0.0 and 1.0\n"
    "- safety_flags: object with boolean flags, e.g., {\"self_harm_risk\": true}\n"
    "Return compact JSON only."
)


class GoogleGeminiClient:
    def __init__(self, model: str = "gemini-1.5", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    def _extract_json_block(self, content: str) -> Optional[str]:
        # Find the first balanced {...} block
        start = content.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(content)):
            c = content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]
        # Fallback: regex that finds first {...} (best-effort)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        return m.group(0) if m else None

    def interpret(self, text: str) -> dict:
        prompt = SYSTEM_PROMPT + "\n\nUser: " + text
        try:
            import google.generativeai as genai  # optional dependency, imported lazily
            if self.api_key:
                genai.configure(api_key=self.api_key)
            resp = genai.generate_text(model=self.model, prompt=prompt)
            content = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:
            logging.warning("Gemini client error: %s", e)
            # fallback to local deterministic LLM
            return LocalLLM().interpret(text)

        json_text = self._extract_json_block(content)
        if not json_text:
            logging.warning("Could not find JSON block in Gemini response; falling back to LocalLLM.")
            return LocalLLM().interpret(text)

        try:
            out = json.loads(json_text)
        except Exception as e:
            logging.warning("Failed to parse JSON from Gemini response: %s", e)
            return LocalLLM().interpret(text)

        # Ensure keys with safe defaults and merge safety flags with local heuristics
        out_primary = out.get("primary_emotions") or ["neutral"]
        out_triggers = out.get("triggers") or []
        intensity = float(out.get("intensity") or 0.5)
        model_safety = dict(out.get("safety_flags") or {})
        local_safety = LocalLLM().interpret(text).get("safety_flags", {})

        # Merge safety flags (logical OR)
        merged_safety = {}
        for key in set(local_safety) | set(model_safety):
            merged_safety[key] = bool(local_safety.get(key, False)) or bool(model_safety.get(key, False))

        return {
            "primary_emotions": out_primary,
            "triggers": out_triggers,
            "intensity": intensity,
            "safety_flags": merged_safety,
        }

class LocalLLM:
    """Deterministic local fallback LLM for tests/CI."""

    def interpret(self, text: str) -> Dict[str, Any]:
        lower = (text or "").lower()
        primary = []
        triggers = []
        intensity = 0.5
        safety_flags = {"self_harm_risk": False}

        if "stressed" in lower or "anx" in lower or "worried" in lower:
            primary.append("anxiety")
        if "happy" in lower or "great" in lower or "joy" in lower:
            primary.append("joy")
        if "deadline" in lower or "due" in lower:
            triggers.append("deadline")

        # Expanded self-harm pattern checks
        danger_patterns = [
            "suicide",
            "kill myself",
            "want to die",
            "i want to die",
            "end my life",
            "hurt myself",
            "self harm",
        ]
        if any(p in lower for p in danger_patterns):
            safety_flags["self_harm_risk"] = True

        return {
            "primary_emotions": primary or ["neutral"],
            "triggers": triggers,
            "intensity": intensity,
            "safety_flags": safety_flags,
        }

# RateLimiter: simple sliding window per-minute
class RateLimiter:
    def __init__(self, per_min: int):
        self.per_min = int(per_min)
        self._req_times = collections.deque()
        self._lock = threading.Lock()
    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            cutoff = now - 60
            while self._req_times and self._req_times[0] < cutoff:
                self._req_times.popleft()
            if len(self._req_times) < self.per_min:
                self._req_times.append(now)
                return True
            return False

class LLMAdapter:
    def __init__(self, client=None, rate_per_min=None):
        # choose provider by env var
        if client:
            self.client = client
        else:
            provider = os.getenv("BYLLM_PROVIDER", "local")
            if provider == "google_gemini":
                self.client = GoogleGeminiClient(model=os.getenv("BYLLM_GEMINI_MODEL", "gemini-1.5"),
                                                 api_key=os.getenv("GOOGLE_API_KEY"))
            else:
                self.client = LocalLLM()
        rate = int(rate_per_min) if rate_per_min is not None else int(os.getenv("BYLLM_RATE_PER_MIN", "30"))
        self.rate_limiter = RateLimiter(rate)
    def interpret(self, text: str) -> dict:
        if not self.rate_limiter.allow():
            raise RuntimeError("Rate limit exceeded for byLLM provider")
        try:
            res = self.client.interpret(text)
        except Exception:
            res = LocalLLM().interpret(text)
        # ensure safety flags merged etc.
        return res