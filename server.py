"""TRIBE Scorer — hostable API + web UI.

Loads TRIBE v2 once and keeps it resident. Scores text, audio, or video by
predicted fMRI brain response. Configurable at runtime (device, text model,
layers). Streams live progress over SSE. Lets you export the raw
(timesteps x 20,484 vertices) predictions per item.

Run:        ./run.sh                      (local, http://127.0.0.1:8011)
Host:       ./serve.sh                     (local + Cloudflare tunnel)
Configure:  edit config.json or use the Settings panel in the UI

HONESTY: TRIBE predicts brain response to passively consumed media, NOT
likes/upvotes. total_activation ~ text length; rank by peak / variance.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

# Let unsupported MPS (Apple GPU) ops fall back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from tribev2.demo_utils import TribeModel

HERE = Path(__file__).parent
CACHE = HERE / "cache"
UPLOADS = CACHE / "uploads"
PREDS = CACHE / "preds"
for d in (CACHE, UPLOADS, PREDS):
    d.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = HERE / "web_results.json"
CONFIG_FILE = HERE / "config.json"
INDEX = HERE / "web" / "index.html"

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

DEFAULT_CONFIG = {
    "host": "127.0.0.1",                     # loopback by default; serve.sh proxies via the tunnel
    "device": "auto",                       # auto | cpu | cuda | mps  (main model + default)
    "text_device": "",                       # override for the heavy LLaMA text encoder ("" = follow device)
    "audio_device": "",                      # override for the audio encoder ("" = follow device)
    "text_model": "unsloth/Llama-3.2-3B",   # non-gated mirror of meta-llama/Llama-3.2-3B
    "text_layers": [0.5, 0.75, 1.0],
    "save_raw": True,                        # persist raw predictions for export
    "api_key": "",                           # if set, required on writes
    "port": 8011,
}

# Only these repos may be loaded as the text encoder. Loading an arbitrary HF
# repo from an unauthenticated request is an RCE/abuse vector, so it is gated.
ALLOWED_TEXT_MODELS = {
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
}

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def resolve_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if cuda else "cpu"
    if pref == "mps":
        return "mps" if mps else "cpu"
    # auto: CUDA if present, else CPU. MPS is opt-in (best-effort; some ops fall
    # back to CPU) so "auto" stays reliable on Macs.
    if cuda:
        return "cuda"
    return "cpu"


def resolve_devices(cfg: dict) -> dict:
    main = resolve_device(cfg.get("device", "auto"))
    return {
        "main": main,
        "text": resolve_device(cfg.get("text_device") or cfg.get("device", "auto")),
        "audio": resolve_device(cfg.get("audio_device") or cfg.get("device", "auto")),
    }


CONFIG = load_config()
if not CONFIG_FILE.exists():
    save_config(CONFIG)

app = FastAPI()

STATE = {"status": "loading", "error": None, "model": None,
         "device": resolve_devices(CONFIG)["main"], "devices": resolve_devices(CONFIG)}
JOBS: "queue.Queue[dict]" = queue.Queue()
RESULTS: list[dict] = []
SUBSCRIBERS: set[asyncio.Queue] = set()
LOOP: asyncio.AbstractEventLoop | None = None


def emit(ev: dict) -> None:
    ev = {**ev, "t": time.time()}
    if LOOP is None:
        return
    for q in list(SUBSCRIBERS):
        try:
            LOOP.call_soon_threadsafe(q.put_nowait, ev)
        except Exception:
            pass


class EmitLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if msg.strip():
            emit({"type": "log", "msg": msg[:300]})


def config_update_from(cfg: dict, devs: dict) -> dict:
    # text_feature/audio_feature only accept auto|cpu|cuda|accelerate (NOT mps),
    # so clamp mps -> cpu for the extractors. The main fMRI model can still use mps.
    # video_feature has no `device` field (extra keys rejected), so it's omitted.
    td = "cpu" if devs["text"] == "mps" else devs["text"]
    ad = "cpu" if devs["audio"] == "mps" else devs["audio"]
    return {
        "data.text_feature.model_name": cfg["text_model"],
        "data.text_feature.layers": list(cfg["text_layers"]),
        "data.text_feature.device": td,
        "data.audio_feature.device": ad,
    }


def _load_model() -> None:
    devs = resolve_devices(CONFIG)
    STATE["device"] = devs["main"]
    STATE["devices"] = devs
    STATE["status"] = "loading"
    emit({"type": "status", "status": "loading", "device": devs["main"]})
    STATE["model"] = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=CACHE, device=devs["main"],
        config_update=config_update_from(CONFIG, devs),
    )
    STATE["status"] = "ready"
    STATE["error"] = None
    emit({"type": "status", "status": "ready", "device": devs["main"]})


def safe_name(text: str, i: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-")
    return f"{i}_{slug or 'item'}"


def compute_metrics(label: str, modality: str, preds) -> dict:
    preds = np.asarray(preds)
    per_step = preds.sum(axis=1)
    rid = uuid.uuid4().hex[:8]
    m = {
        "id": rid,
        "text": label,
        "modality": modality,
        "chars": len(label),
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "total_activation": float(preds.sum()),
        "mean_activation": float(preds.mean()),
        "peak_activation": float(preds.max()),
        "per_vertex_variance_mean": float(preds.var(axis=0).mean()),
        "time_to_peak_sec": int(per_step.argmax()),
        "ts": time.time(),
    }
    if CONFIG.get("save_raw"):
        np.save(PREDS / f"{rid}.npy", preds.astype(np.float32))
        m["raw"] = f"{rid}.npy"
    return m


def _score(label: str, modality: str, **events_kw) -> None:
    t0 = time.time()
    emit({"type": "stage", "stage": "extracting events"})
    df = STATE["model"].get_events_dataframe(**events_kw)
    emit({"type": "stage", "stage": "predicting fMRI brain response"})
    preds, _ = STATE["model"].predict(events=df)
    m = compute_metrics(label, modality, preds)
    m["seconds"] = round(time.time() - t0, 1)
    RESULTS.append(m)
    RESULTS_FILE.write_text(json.dumps(RESULTS, indent=2))
    emit({"type": "result", "metrics": m})


def worker() -> None:
    logging.getLogger().addHandler(EmitLogHandler(level=logging.INFO))
    try:
        _load_model()
    except Exception as e:  # noqa: BLE001
        STATE["status"] = "error"
        STATE["error"] = str(e)
        emit({"type": "status", "status": "error", "error": str(e)})
        return

    while True:
        job = JOBS.get()
        try:
            if job["type"] == "reload":
                _load_model()
                continue

            emit({"type": "job_start", "job": job["id"], "n": job.get("n", 1)})
            if job["type"] == "score_text":
                for i, text in enumerate(job["texts"]):
                    emit({"type": "draft_start", "index": i, "text": text})
                    try:
                        p = CACHE / f"text_{safe_name(text, i)}.txt"
                        p.write_text(text)
                        _score(text, "text", text_path=p)
                    except Exception as e:  # noqa: BLE001
                        emit({"type": "draft_error", "index": i, "error": str(e)[:300]})
            elif job["type"] == "score_file":
                label, path, modality = job["label"], Path(job["path"]), job["modality"]
                emit({"type": "draft_start", "index": 0, "text": f"[{modality}] {label}"})
                try:
                    kw = {"audio_path": path} if modality == "audio" else {"video_path": path}
                    _score(label, modality, **kw)
                except Exception as e:  # noqa: BLE001
                    emit({"type": "draft_error", "index": 0, "error": str(e)[:300]})
            emit({"type": "job_done", "job": job["id"]})
        except Exception as e:  # noqa: BLE001
            emit({"type": "status", "status": STATE["status"], "error": str(e)[:300]})


# ----------------------------------------------------------------------------- API

def require_key(x_api_key: str | None) -> None:
    key = CONFIG.get("api_key") or ""
    public = CONFIG.get("host", "127.0.0.1") not in LOOPBACK_HOSTS
    if public and not key:
        # Fail closed: never allow unauthenticated writes on a public bind.
        raise HTTPException(status_code=503, detail="server is public but no api_key is set")
    if key and not (x_api_key and hmac.compare_digest(str(x_api_key), str(key))):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class ScoreReq(BaseModel):
    texts: list[str]


class ConfigReq(BaseModel):
    device: str | None = None
    text_device: str | None = None
    audio_device: str | None = None
    text_model: str | None = None
    text_layers: list[float] | None = None
    save_raw: bool | None = None
    api_key: str | None = None


@app.on_event("startup")
async def _startup() -> None:
    global LOOP
    LOOP = asyncio.get_running_loop()
    if RESULTS_FILE.exists():
        try:
            RESULTS.extend(json.loads(RESULTS_FILE.read_text()))
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX.read_text()


@app.get("/api/status")
def status() -> dict:
    return {
        "status": STATE["status"], "error": STATE["error"],
        "device": STATE["device"], "devices": STATE.get("devices"),
        "results": len(RESULTS), "queued": JOBS.qsize(),
    }


@app.get("/api/config")
def get_config() -> dict:
    safe = {k: v for k, v in CONFIG.items() if k != "api_key"}
    safe["api_key_set"] = bool(CONFIG.get("api_key"))
    safe["resolved_devices"] = STATE.get("devices")
    safe["available"] = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
    }
    return safe


@app.post("/api/config")
def set_config(req: ConfigReq, x_api_key: str | None = Header(default=None)) -> dict:
    require_key(x_api_key)
    if req.text_model is not None and req.text_model not in ALLOWED_TEXT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"text_model not allowed. allowed: {sorted(ALLOWED_TEXT_MODELS)}",
        )
    changed_model = False
    for field in ("device", "text_device", "audio_device", "text_model", "text_layers"):
        val = getattr(req, field)
        if val is not None and val != CONFIG.get(field):
            CONFIG[field] = val
            changed_model = True
    if req.save_raw is not None:
        CONFIG["save_raw"] = req.save_raw
    if req.api_key is not None:
        CONFIG["api_key"] = req.api_key
    save_config(CONFIG)
    if changed_model:
        JOBS.put({"type": "reload"})
    return {"ok": True, "reloading": changed_model}


@app.get("/api/results")
def results() -> list[dict]:
    return RESULTS


@app.get("/api/result/{rid}/raw")
def result_raw(rid: str, fmt: str = "npy"):
    item = next((r for r in RESULTS if r.get("id") == rid), None)
    if not item or not item.get("raw"):
        raise HTTPException(status_code=404, detail="no raw predictions for this id")
    npy = PREDS / item["raw"]
    if not npy.exists():
        raise HTTPException(status_code=404, detail="raw file missing")
    if fmt == "json":
        arr = np.load(npy)
        return JSONResponse({"id": rid, "shape": list(arr.shape), "preds": arr.tolist()})
    return FileResponse(npy, media_type="application/octet-stream", filename=f"tribe_{rid}.npy")


def _render_brainmap(npy_path: Path, out_path: Path) -> None:
    # Lazy imports: only pulled when a brain map is actually requested.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    arr = np.asarray(np.load(npy_path))     # (timesteps, 20484) on fsaverage5
    vec = arr.mean(axis=0)                   # mean predicted activation per vertex
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    half = vec.shape[0] // 2                 # fsaverage5: 10242 vertices per hemisphere
    lh, rh = vec[:half], vec[half:]
    vmax = float(np.abs(vec).max()) or 1.0
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(11, 5))
    plotting.plot_surf_stat_map(fs.infl_left, lh, hemi="left", bg_map=fs.sulc_left,
                                vmax=vmax, axes=axes[0], colorbar=False, title="left")
    plotting.plot_surf_stat_map(fs.infl_right, rh, hemi="right", bg_map=fs.sulc_right,
                                vmax=vmax, axes=axes[1], colorbar=True, title="right")
    fig.suptitle("Predicted cortical response (mean over time)")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@app.get("/api/result/{rid}/brainmap")
def result_brainmap(rid: str):
    item = next((r for r in RESULTS if r.get("id") == rid), None)
    if not item or not item.get("raw"):
        raise HTTPException(status_code=404, detail="no raw predictions for this id (enable save_raw)")
    npy = PREDS / item["raw"]
    if not npy.exists():
        raise HTTPException(status_code=404, detail="raw file missing")
    png = PREDS / f"{rid}_brainmap.png"
    if not png.exists():
        try:
            _render_brainmap(npy, png)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"brain map render failed: {e}")
    return FileResponse(png, media_type="image/png", filename=f"cortext_{rid}.png")


@app.post("/api/score")
def score_text(req: ScoreReq, x_api_key: str | None = Header(default=None)) -> dict:
    require_key(x_api_key)
    texts = [t.strip() for t in req.texts if t.strip()]
    job_id = uuid.uuid4().hex[:8]
    JOBS.put({"type": "score_text", "id": job_id, "texts": texts, "n": len(texts)})
    return {"job": job_id, "n": len(texts), "queued": JOBS.qsize()}


@app.post("/api/score/file")
async def score_file(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    require_key(x_api_key)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        modality = "audio"
    elif suffix in VIDEO_SUFFIXES:
        modality = "video"
    else:
        raise HTTPException(status_code=400, detail=f"unsupported file type '{suffix}'")
    dest = UPLOADS / f"{uuid.uuid4().hex[:8]}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    job_id = uuid.uuid4().hex[:8]
    JOBS.put({"type": "score_file", "id": job_id, "label": file.filename or dest.name,
              "path": str(dest), "modality": modality})
    return {"job": job_id, "modality": modality, "queued": JOBS.qsize()}


@app.delete("/api/results")
def clear_results(x_api_key: str | None = Header(default=None)) -> dict:
    require_key(x_api_key)
    RESULTS.clear()
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    for f in PREDS.glob("*.npy"):
        f.unlink()
    return {"ok": True}


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"


@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue()
    SUBSCRIBERS.add(q)

    async def gen():
        try:
            yield _sse({"type": "status", "status": STATE["status"],
                        "error": STATE["error"], "device": STATE["device"]})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield _sse(ev)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            SUBSCRIBERS.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = CONFIG.get("host", "127.0.0.1")
    port = int(CONFIG.get("port", 8011))

    # Fail closed: never expose a public bind without an API key. Generate one.
    if host not in LOOPBACK_HOSTS and not CONFIG.get("api_key"):
        CONFIG["api_key"] = secrets.token_urlsafe(32)
        save_config(CONFIG)
        print("=" * 64)
        print(f"PUBLIC BIND ({host}) with no api_key. Generated one (saved to config.json):")
        print(f"  X-API-Key: {CONFIG['api_key']}")
        print("Send it as the X-API-Key header on writes (Settings tab in the UI).")
        print("=" * 64)

    print(f"TRIBE Scorer on http://{host}:{port}  (device={STATE['device']})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
