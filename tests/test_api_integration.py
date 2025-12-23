from fastapi.testclient import TestClient
from api.main import app
from jaclang.runtimelib.runtime import JacRuntime

client = TestClient(app)

def test_entry_endpoint_creates_journal_and_returns_response():
    entry = {
        "user_id": "integration_user",
        "score": 0.45,
        "tags": ["anxiety"],
        "text": "I'm stressed about an upcoming deadline.",
    }

    r = client.post("/entry", json=entry)
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert isinstance(body.get("suggestions"), list)
    assert "journal_id" in body and isinstance(body["journal_id"], str) and body["journal_id"]

    # confirm JournalEntry persisted in Jac memory
    found = False
    for anc in JacRuntime.exec_ctx.mem.query(lambda a: getattr(a, "archetype", None) is not None):
        try:
            arch = anc.archetype
            if arch.__class__.__name__ == "JournalEntry":
                found = True
                break
        except Exception:
            continue
    assert found, "No JournalEntry anchor found in Jac memory after /entry"