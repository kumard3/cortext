# Cortext

Score your text drafts (tweets, posts, headlines) by **predicted brain response**,
locally. Cortext wraps Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2)
fMRI foundation model so you can rank drafts before you publish — as a native
macOS app, a CLI, a web UI, or an MCP tool that AI agents can call.

> **Cortext** is the app; **TRIBE v2** is the model underneath (Meta, CC BY-NC).

## Pick your way in

- **Desktop app (macOS, Apple Silicon)** — open `Cortext.dmg`; it builds its own
  runtime and downloads the models on first launch. No terminal. See [`desktop/`](desktop/).
- **CLI / web (macOS or Linux)** — `./install.sh && ./run.sh` (below).
- **MCP server (for AI agents)** — [`mcp/`](mcp/) exposes scoring tools to Claude
  Code, Cursor, and other MCP clients.
- **Calibrate it on your own data** — [`tools/calibrate.py`](tools/) tells you
  whether the score predicts *your* engagement.

## Quickstart (CLI + web)

```bash
git clone https://github.com/kumard3/cortext
cd cortext
./install.sh      # uv, Python 3.12 venv, tribev2, pinned deps, applies fixes
./run.sh          # web UI at http://127.0.0.1:8011
```

Paste drafts (one per line), hit Score, watch the leaderboard fill in. Each row
exports the raw predictions (`.npy` / `json`) and renders a cortical heat-map
(**🧠 map**).

## Desktop app

The native macOS app (in [`desktop/`](desktop/)) is a Tauri 2 shell. On first
launch it sets up a local Python runtime from a **locked** dependency set, runs
the backend as a managed sidecar, and points the window at the web UI. Models
download on first score. Build it:

```bash
cd desktop
npm install
npm run tauri icon <path/to/icon-1024.png>
npm run build      # app + .dmg (or: npm run dev)
```

## MCP server (agent tools)

[`mcp/tribe_mcp.py`](mcp/) exposes the running scorer to MCP clients. Tools:
`get_status`, `score_text`, `rank`, `best_of_n`, `compare`, `explain`,
`score_media`, `list_results`. So an agent can generate N variants and call
`best_of_n` to pick the most salient. Setup in [`mcp/README.md`](mcp/README.md).

## Calibrate against your own engagement

```bash
python tools/calibrate.py posts.csv --engagement-col likes
```

Scores your past posts and reports whether TRIBE's salience metrics correlate
with the engagement they actually got — the honest test of whether it predicts
anything for *you*.

## What it actually does

For each draft it runs the real TRIBE text pipeline:

```
text -> gTTS speech -> whisperx transcription -> LLaMA + audio features -> predicted fMRI
```

The model outputs an array of shape `(timesteps x 20,484 cortical vertices)` on
the `fsaverage5` brain mesh: predicted activation across the whole cortex over
time. The UI reduces that to a few summary metrics and ranks drafts.

## Read this before trusting a number

- TRIBE predicts **brain response to passively consumed media**. It is **not** a
  validated proxy for likes, upvotes, or clicks. Treat the score as a salience
  tiebreaker, not an engagement oracle.
- `total_activation` correlates ~**0.97 with text length**, so it just rewards
  longer text. Rank by **peak** (salience spike) and **variance** (dynamics),
  which are length-independent.
- Empirical pattern in testing: concrete, specific, number-heavy text scores
  higher than vague or feature-list text.

## CLI

```bash
./score.py "your draft here"
./score.py "draft one" "draft two"
./score.py --file drafts.txt        # one per line
```

## Requirements

- macOS or Linux (desktop app: macOS Apple Silicon)
- `git`, `curl`, and **`ffmpeg`** (`brew install ffmpeg` / `apt-get install ffmpeg`)
- ~16 GB free disk and ideally 16 GB+ RAM
- First scored draft downloads ~**12 GB** of models (TRIBE weights, whisper
  large-v3, wav2vec2, the LLaMA text encoder). One time, then cached.
- GPU optional. Runs on CPU otherwise, a few minutes per item.

## Running on GPU (and CPU/GPU hybrid)

Set the device in the **Settings** tab or in `config.json`. `GET /api/config`
reports what's detected.

- **NVIDIA (CUDA):** `device: "auto"` uses it automatically.
- **Apple Silicon (MPS):** set `device: "mps"`. Best-effort: a few ops fall back
  to CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`). whisperx stays on CPU (ctranslate2).
- **Hybrid:** put just the heavy LLaMA encoder on the GPU — `device: "cpu"`,
  `text_device: "mps"` (or `"cuda"`); everything else stays on CPU.

## The fixes (running TRIBE on a fresh machine)

TRIBE out of the box hits several breakages. `patch_tribe.py` + the installer
+ the server config handle all of them, and dependencies are **pinned to a
lockfile** so they can never drift:

1. whisperx via uvx pulls torchaudio >= 2.9, which removed `list_audio_backends`
   that pyannote still calls. Pinned to torch/torchaudio 2.8 on Python 3.12.
2. pyannote's VAD checkpoint fails torch 2.6+ `weights_only` pickling. Switched
   to silero VAD.
3. CPU has no float16 compute type. Uses int8 on CPU.
4. The text encoder `meta-llama/Llama-3.2-3B` is gated. Points to a bit-identical
   non-gated mirror (`unsloth/Llama-3.2-3B`). No HuggingFace token needed.
5. `neuralset` needs `exca.steps.base.NoValue`, which newer `exca` removed.
   Pinned `exca==0.5.20` (the floor it was built against).
6. gTTS auto-detects the draft language with langdetect, which misfires on short
   English text (returns e.g. `so`) and crashes TTS. The pipeline is English-only,
   so the TTS language is forced to `en`.
7. `transformers` 5.x references `torch.float8_e8m0fnu` (needs torch >= 2.7) and
   breaks LLaMA loading on torch 2.6. Pinned `transformers==4.57.6`.

## Security (read before hosting)

- **Binds `127.0.0.1` by default.** `./run.sh` is local-only.
- **Public binds fail closed.** A non-loopback `host` with no `api_key` makes the
  server generate one on launch and print it; writes then require `X-API-Key`.
- **The text model is allowlisted** (`ALLOWED_TEXT_MODELS` in `server.py`), so the
  config endpoint can't load an arbitrary model.
- Mutating routes honor `X-API-Key`; read routes are open. Host multi-user behind
  your own auth proxy.

## License / attribution

This wrapper is provided as-is. **TRIBE v2 and its weights are licensed CC BY-NC
by Meta** (non-commercial). Use this for research and personal content work, not
commercial products. See the upstream repo for terms.
