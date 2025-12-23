import sys
import types
from backend.byllm.interpret import interpret_input
from backend.byllm.adapter import LLMAdapter
from backend.byllm.adapter import GoogleGeminiClient

def test_gemini_client_parses_json(monkeypatch):
    # Simulate a genai module (minimal)
    class Resp:
        def __init__(self, text):
            self.text = text

    def generate_text(model, prompt):
        # Content with extra text and JSON object
        return Resp("Assistant analysis:\n{\"primary_emotions\": [\"joy\"], \"triggers\": [], \"intensity\": 0.6, \"safety_flags\": {\"self_harm_risk\": false}}")

import types as _types

dummy = _types.SimpleNamespace(generate_text=generate_text, configure=lambda api_key: None)

# Create a fake 'google' package and attach the 'generativeai' submodule
google_pkg = _types.ModuleType("google")
google_pkg.generativeai = dummy

monkeypatch.setitem(sys.modules, "google", google_pkg)
monkeypatch.setitem(sys.modules, "google.generativeai", dummy)

client = GoogleGeminiClient(model="test", api_key=None)
out = client.interpret("I felt joy today")

assert out["primary_emotions"] == ["joy"]
assert out["intensity"] == 0.6
assert out["safety_flags"].get("self_harm_risk") is False

def test_adapter_detects_anxiety_and_trigger():
    res = interpret_input("I'm stressed about a deadline")
    assert "anxiety" in res["primary_emotions"]
    assert "deadline" in res["triggers"]

def test_adapter_safety_flags():
    res = interpret_input("I want to kill myself")
    assert res["safety_flags"]["self_harm_risk"] is True

def test_rate_limiter_init():
    a = LLMAdapter(rate_per_min=5)
    assert a.rate_limiter.per_min == 5

def test_rate_limiter_env_fallback(monkeypatch):
    monkeypatch.setenv("BYLLM_RATE_PER_MIN", "7")
    a = LLMAdapter()
    assert a.rate_limiter.per_min == 7