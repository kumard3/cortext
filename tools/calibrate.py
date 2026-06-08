#!/usr/bin/env python3
"""Calibrate Cortext / TRIBE against YOUR real engagement.

Scores a CSV of your past posts and reports whether TRIBE's predicted-salience
metrics actually correlate with the engagement those posts got. This is the
honest test: TRIBE predicts neural salience to passive media, not likes. This
tells you, on your own data, whether that salience is worth anything as a
proxy for you specifically.

Usage:
    python tools/calibrate.py posts.csv
    python tools/calibrate.py posts.csv --text-col tweet --engagement-col likes
    python tools/calibrate.py posts.csv --base http://127.0.0.1:8011

CSV: a header row with a text column and a numeric engagement column.
Defaults: text column "text", engagement column auto-detected from
{engagement, likes, faves, score, impressions, views, upvotes}.

Stdlib only. Note: each row takes minutes to score on CPU, so keep N modest
(20-50) or be patient.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENGAGEMENT_GUESSES = ["engagement", "likes", "faves", "favorites", "score",
                      "impressions", "views", "upvotes", "reactions"]


def discover_base() -> str:
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


def http_json(base: str, path: str, body=None):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("TRIBE_API_KEY")
    if key:
        headers["X-API-Key"] = key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def pearson(xs: list[float], ys: list[float]):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def strength(r: float) -> str:
    a = abs(r)
    if a >= 0.6:
        return "strong"
    if a >= 0.35:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "negligible"


def load_csv(path: str, text_col: str, eng_col: str | None):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if text_col not in cols:
            sys.exit(f"text column '{text_col}' not in CSV columns {cols}")
        if eng_col is None:
            for g in ENGAGEMENT_GUESSES:
                if g in cols:
                    eng_col = g
                    break
        if eng_col is None or eng_col not in cols:
            sys.exit(f"engagement column not found; pass --engagement-col (have {cols})")
        rows = []
        for row in reader:
            text = (row.get(text_col) or "").strip()
            raw = (row.get(eng_col) or "").replace(",", "").strip()
            if not text or not raw:
                continue
            try:
                rows.append((text, float(raw)))
            except ValueError:
                continue
    return rows, eng_col


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate TRIBE salience vs your engagement.")
    ap.add_argument("csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--engagement-col", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--timeout", type=int, default=0, help="overall wait seconds (0 = auto)")
    args = ap.parse_args()

    base = (args.base or discover_base()).rstrip("/")
    rows, eng_col = load_csv(args.csv, args.text_col, args.engagement_col)
    if len(rows) < 3:
        sys.exit(f"need >=3 usable rows, got {len(rows)}")
    texts = [t for t, _ in rows]
    eng_by_text = {t: e for t, e in rows}
    print(f"Cortext: {base}")
    print(f"Scoring {len(texts)} posts (engagement column: '{eng_col}'). This takes a while...")

    try:
        before = {r["id"] for r in http_json(base, "/api/results")}
        http_json(base, "/api/score", {"texts": texts})
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Cortext at {base}: {e}. Is the app running?")

    need = len(set(texts))
    timeout = args.timeout or max(600, need * 300)
    deadline = time.time() + timeout
    new = []
    while time.time() < deadline:
        time.sleep(6)
        try:
            results = http_json(base, "/api/results")
        except urllib.error.URLError:
            continue
        new = [r for r in results if r.get("id") not in before]
        done = len({r.get("text") for r in new})
        print(f"  scored {done}/{need}", end="\r", flush=True)
        if done >= need:
            break
    print()

    by_text = {}
    for r in new:
        by_text.setdefault(r.get("text"), r)  # first occurrence
    matched = [(eng_by_text[t], by_text[t]) for t in texts if t in by_text]
    if len(matched) < 3:
        sys.exit(f"only {len(matched)} posts scored in time; re-run to resume (cached) or raise --timeout")

    eng = [m[0] for m in matched]
    metrics = {
        "peak_activation": [m[1].get("peak_activation", 0.0) for m in matched],
        "per_vertex_variance_mean": [m[1].get("per_vertex_variance_mean", 0.0) for m in matched],
        "total_activation": [m[1].get("total_activation", 0.0) for m in matched],
        "chars": [float(m[1].get("chars", 0)) for m in matched],
    }

    print(f"\n=== Calibration on {len(matched)} posts ===")
    print(f"{'metric':<28}{'pearson r':>11}   strength")
    results_r = {}
    for name, vals in metrics.items():
        r = pearson(vals, eng)
        results_r[name] = r
        rtxt = "n/a" if r is None else f"{r:+.3f}"
        stxt = "" if r is None else strength(r)
        print(f"{name:<28}{rtxt:>11}   {stxt}")

    print("\n--- read this ---")
    salience = {k: results_r[k] for k in ("peak_activation", "per_vertex_variance_mean")
                if results_r[k] is not None}
    best = max(salience, key=lambda k: abs(salience[k])) if salience else None
    if best is None:
        print("Could not compute correlations (no variance). Need more varied data.")
    else:
        r = salience[best]
        print(f"Best length-independent predictor for you: {best} (r={r:+.3f}, {strength(r)}).")
        if abs(r) < 0.2:
            print("That's negligible: TRIBE salience does NOT predict your engagement on this sample.")
            print("Use it as a curiosity, not a posting guide.")
        elif abs(r) < 0.35:
            print("Weak signal: maybe a faint tiebreaker for you, nothing to lean on.")
        else:
            print("There's a real correlation for you. Worth using as ONE input among others.")
    tot = results_r.get("chars")
    if tot is not None and abs(tot) > 0.4:
        print(f"(Note: char count alone correlates {tot:+.3f} with your engagement, "
              "so watch out for length confounds.)")
    print("\nCaveat: correlation != causation, small samples are noisy, and this is your "
          "data only. Re-run as you gather more posts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
