"""The HTTP contract between the console and its UI."""

from __future__ import annotations

import json

QUESTION = "my millet seedlings have pale twisted leaves"


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """-> [(event name, payload), ...] in the order the browser receives them."""
    out = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        name = data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if name and data is not None:
            out.append((name, json.loads(data)))
    return out


def test_status_shape(client):
    s = client.get("/api/status").json()

    assert set(s) == {
        "model",
        "model_size_mb",
        "threads",
        "library",
        "offline_enforced",
        "strict_offline",
        "internet_reachable",
    }
    assert s["model"].endswith(".gguf")
    assert s["library"]["documents"] == 1 and s["library"]["chunks"] > 0
    assert s["offline_enforced"] is False
    # Never claim a network state that was not probed.
    assert s["internet_reachable"] is None


def test_library_lists_documents(client):
    docs = client.get("/api/library").json()["documents"]

    assert [d["title"] for d in docs] == ["Millet Diseases"]


def test_search_returns_citations(client):
    body = client.post("/api/search", json={"question": QUESTION, "k": 4}).json()

    assert body["citations"], "the indexed millet manual should match"
    first = body["citations"][0]
    assert first["title"] == "Millet Diseases" and first["org"] == "ICRISAT"
    assert "downy mildew" in first["text"]
    assert isinstance(body["sufficient"], bool)


def test_ask_streams_citations_tokens_contract_then_done(client):
    resp = client.post("/api/ask", json={"question": QUESTION, "k": 4})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    names = [n for n, _ in events]

    assert names[0] == "citations"
    assert names[-1] == "done"
    assert names[-2] == "contract"
    assert set(names[1:-2]) == {"token"}, "tokens must sit between citations and the contract"

    citations = events[0][1]["citations"]
    assert citations and citations[0]["title"] == "Millet Diseases"

    contract = events[-2][1]
    assert contract["status"] == "ANSWERED"
    assert contract["structured"] is True and contract["repaired"] is False
    assert contract["actions"] == ["Uproot and burn the infected plants. [1]"]
    assert contract["sources"] == [1]
    assert contract["caution"].startswith("Call the extension agent")

    done = events[-1][1]
    assert done["tokens"] > 0 and done["seconds"] >= 0 and "tps" in done


def test_ask_reports_a_dead_backend_as_an_error_event(client, backend):
    def explode(prompt, **opts):
        raise RuntimeError("llama-server died")

    backend.stream = explode

    events = parse_sse(client.post("/api/ask", json={"question": QUESTION, "k": 4}).text)

    assert [n for n, _ in events] == ["citations", "error"]
    assert "llama-server died" in events[-1][1]["message"]


def test_ask_validates_the_question(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": QUESTION, "k": 99}).status_code == 422


def test_index_serves_the_single_file_ui(client):
    html = client.get("/").text

    assert "Sahel Sage" in html
    assert "http://" not in html and "https://" not in html, "no CDN, no remote font"
