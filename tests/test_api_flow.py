import json
from backend.byllm.interpret import interpret_input
from backend.byllm.craft import craft_response


def test_interpret_simple():
    res = interpret_input("I'm feeling really stressed about a deadline")
    assert "anxiety" in res["primary_emotions"]
    assert "deadline" in res["triggers"]


def test_craft_anxiety():
    context = {"emotions": ["anxiety"], "safety_flags": {}}
    resp = craft_response(context)
    assert "anxiety" in resp["message"].lower() or len(resp["suggestions"]) > 0
