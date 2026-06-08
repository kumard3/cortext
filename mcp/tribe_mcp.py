#!/usr/bin/env python3
"""MCP server exposing the local Cortext / TRIBE Scorer API as agent tools.

Wraps a running Cortext (the desktop app, or `./run.sh`) so any MCP client
(Claude Desktop / Claude Code / Cursor) can score and rank text/audio/video by
predicted fMRI brain response.

Tools: get_status, score_text, rank, best_of_n, compare, explain, score_media,
list_results.

Discovery:
  - TRIBE_API_URL env (e.g. http://127.0.0.1:8011) wins if set.
  - else the macOS desktop app's config.json is read for host/port.
  - else falls back to http://127.0.0.1:8011.
Auth:
  - TRIBE_API_KEY env is sent as X-API-Key (loopback usually needs none).

Run (stdio):  python tribe_mcp.py
Deps:         pip install "mcp[cli]"   (or: uv pip install mcp)

HONEST NOTE: TRIBE predicts neural salience to passively-consumed media, NOT
likes/upvotes. Rank by peak_activation / per_vertex_variance_mean (length
independent), never total_activation (~0.97 correlated with length). Treat every
result as a salience tiebreaker, not an engagement oracle.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP


def _discover_base() -> str:
    env = os.environ.get("TRIBE_API_URL")
    if env:
        return env.rstrip("/")
    cfg = Path.home() / "Library/Application Support/co.kumard3.cortext/app/config.json"
    try:
        d = json.loads(cfg.read_text())
        host = d.get("host", "127.0.0.1")
        port = int(d.get("port", 8011))
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{port}"
    except Exception:
        return "http://127.0.0.1:8011"


BASE = _discover_base()
API_KEY = os.environ.get("TRIBE_API_KEY", "")
NOTE = "Salience (peak_activation), not engagement. Tiebreaker, not an oracle."

mcp = FastMCP("cortext")


# ----------------------------------------------------------------------- http
def _headers(extra: dict | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    if extra:
        h.update(extra)
    return h


def _get(path: str):
    req = urllib.request.Request(BASE + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _post(path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _post_file(path: str):
    boundary = "----cortext" + uuid.uuid4().hex
    fname = os.path.basename(path)
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        filedata = f.read()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    body = head + filedata + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(BASE + "/api/score/file", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


# -------------------------------------------------------------------- helpers
def _slim(r: dict) -> dict:
    return {
        "text": r.get("text"),
        "modality": r.get("modality"),
        "chars": r.get("chars"),
        "peak_activation": r.get("peak_activation"),
        "per_vertex_variance_mean": r.get("per_vertex_variance_mean"),
        "total_activation": r.get("total_activation"),
        "id": r.get("id"),
    }


def _wait_for_new(before_ids: set, need: int, timeout_seconds: int) -> list[dict]:
    """Poll /api/results until `need` items with ids not in before_ids appear."""
    deadline = time.time() + max(60, int(timeout_seconds))
    new: list[dict] = []
    while time.time() < deadline:
        time.sleep(4)
        try:
            results = _get("/api/results")
        except urllib.error.URLError:
            continue
        new = [r for r in results if r.get("id") not in before_ids]
        if len(new) >= need:
            break
    return new


def _score_collect(texts: list[str], timeout_seconds: int) -> tuple[list[dict], dict, bool]:
    """Submit texts, wait for their results. Returns (new_results, submit_resp, timed_out)."""
    before = {r["id"] for r in _get("/api/results")}
    sub = _post("/api/score", {"texts": texts})
    new = _wait_for_new(before, len(texts), timeout_seconds)
    return new, sub, len(new) < len(texts)


# ---------------------------------------------------------------------- tools
@mcp.tool()
def get_status() -> dict:
    """Server + model status (status: loading|ready|error, device, #results,
    queued). Call this first to confirm the model is ready before scoring."""
    try:
        return _get("/api/status")
    except urllib.error.URLError as e:
        return {"error": f"cannot reach Cortext at {BASE}: {e}. Is the app running?"}


@mcp.tool()
def score_text(texts: list[str], timeout_seconds: int = 1800) -> dict:
    """Score one or more text drafts by predicted fMRI brain response and return
    them ranked best-first by peak_activation. Blocks until done (minutes/draft
    on CPU). See module note on interpretation."""
    return rank(texts, timeout_seconds)


@mcp.tool()
def rank(texts: list[str], timeout_seconds: int = 1800) -> dict:
    """Rank text drafts best-first by predicted neural salience (peak_activation).
    Returns the ordering plus a one-line 'why' for the winner."""
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return {"error": "no non-empty texts"}
    try:
        new, sub, timed_out = _score_collect(texts, timeout_seconds)
    except urllib.error.URLError as e:
        return {"error": f"cannot reach Cortext at {BASE}: {e}"}
    ordered = sorted(new, key=lambda r: r.get("peak_activation", 0.0), reverse=True)
    ranking = [{"rank": i + 1, **_slim(r)} for i, r in enumerate(ordered)]
    why = None
    if ranking:
        top = ranking[0]
        why = (
            f"#{1} has the highest peak_activation ({top['peak_activation']}). "
            "Higher peak = stronger predicted salience spike, independent of length."
        )
    return {"submitted": sub, "completed": len(new), "expected": len(texts),
            "timed_out": timed_out, "ranking": ranking, "why": why, "note": NOTE}


@mcp.tool()
def best_of_n(texts: list[str], timeout_seconds: int = 1800) -> dict:
    """Score candidate drafts and return ONLY the single best one by peak_activation,
    plus the full ranking. Use this for best-of-N selection: have your model
    generate N variants, pass them here, post/use the winner."""
    res = rank(texts, timeout_seconds)
    if "ranking" in res and res["ranking"]:
        res["best"] = res["ranking"][0]
    return res


@mcp.tool()
def compare(a: str, b: str, timeout_seconds: int = 1800) -> dict:
    """Pairwise: score two drafts and say which has stronger predicted salience,
    and by how much (percent margin on peak_activation)."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return {"error": "both a and b must be non-empty"}
    try:
        new, _, timed_out = _score_collect([a, b], timeout_seconds)
    except urllib.error.URLError as e:
        return {"error": f"cannot reach Cortext at {BASE}: {e}"}
    by_text = {r.get("text"): r for r in new}
    ra, rb = by_text.get(a), by_text.get(b)
    if not ra or not rb:
        return {"error": "scoring did not complete for both", "timed_out": timed_out,
                "got": [_slim(r) for r in new]}
    pa = ra.get("peak_activation", 0.0) or 0.0
    pb = rb.get("peak_activation", 0.0) or 0.0
    winner = "a" if pa >= pb else "b"
    lo, hi = sorted((pa, pb))
    margin = round((hi - lo) / lo * 100, 1) if lo else None
    return {"a": _slim(ra), "b": _slim(rb), "winner": winner,
            "margin_pct_on_peak": margin, "note": NOTE}


@mcp.tool()
def explain(text: str, timeout_seconds: int = 1800) -> dict:
    """Score one draft and return WHY: the per-timestep activation profile (the
    salience curve over the stimulus), when it peaks, and summary metrics. Useful
    to see whether a draft builds a sharp spike vs. stays flat."""
    text = text.strip()
    if not text:
        return {"error": "empty text"}
    try:
        new, _, _ = _score_collect([text], timeout_seconds)
        if not new:
            return {"error": "scoring did not complete in time"}
        r = new[0]
        raw = _get(f"/api/result/{r['id']}/raw?fmt=json")  # {id, shape, preds}
    except urllib.error.URLError as e:
        return {"error": f"cannot reach Cortext at {BASE}: {e}"}
    preds = raw.get("preds") or []
    per_step = [float(sum(row)) for row in preds]
    peak_ts = max(range(len(per_step)), key=lambda i: per_step[i]) if per_step else -1
    return {
        "text": text,
        "metrics": _slim(r),
        "n_timesteps": len(per_step),
        "n_vertices": (raw.get("shape") or [None, None])[1],
        "temporal_profile": [round(v, 3) for v in per_step],
        "peak_timestep": peak_ts,
        "interpretation": "A sharp single peak = a strong salience moment; a flat "
                          "profile = low/steady response. Peak height drives the rank.",
        "note": NOTE,
    }


@mcp.tool()
def score_media(path: str, timeout_seconds: int = 1800) -> dict:
    """Score a local audio or video file (e.g. a podcast intro, ad cut, video hook)
    by predicted brain response. Returns the metrics for that clip."""
    if not os.path.isfile(path):
        return {"error": f"file not found: {path}"}
    try:
        before = {r["id"] for r in _get("/api/results")}
        sub = _post_file(path)
        new = _wait_for_new(before, 1, timeout_seconds)
    except urllib.error.URLError as e:
        return {"error": f"cannot reach Cortext at {BASE}: {e}"}
    except OSError as e:
        return {"error": f"could not read file: {e}"}
    if not new:
        return {"error": "scoring did not complete in time", "submitted": sub}
    return {"submitted": sub, "result": _slim(new[0]), "note": NOTE}


@mcp.tool()
def list_results(limit: int = 20) -> list[dict]:
    """Return the most recent scored results (newest last)."""
    try:
        res = _get("/api/results")
    except urllib.error.URLError as e:
        return [{"error": f"cannot reach Cortext at {BASE}: {e}"}]
    return [_slim(r) for r in res[-max(1, limit):]]


if __name__ == "__main__":
    mcp.run()
