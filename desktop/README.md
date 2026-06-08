# TRIBE Scorer — desktop app (macOS, Apple Silicon)

A Tauri 2 shell around the existing TRIBE Scorer backend. On first launch it
builds a local Python runtime (via `uv`), then runs `server.py` as a managed
sidecar and points the window at the local web UI. Models (~12 GB) download on
the first score, into the Hugging Face cache. Everything runs on the machine.

## How it works

```
Tauri shell (Rust)
 ├─ first run: copies bundled payload -> ~/Library/Application Support/co.kumard3.tribescorer/app/
 │             runs bootstrap.sh (uv venv + vendored tribev2 + deps + the 4 fixes)
 ├─ spawns:   <app>/.venv/bin/python server.py   (FastAPI on a free 127.0.0.1 port)
 ├─ waits for the port, then navigates the window to http://127.0.0.1:<port>
 └─ on quit:  kills the server process
```

The frontend in `src/` is only the first-run/setup screen. Once the backend is
up, the same window loads the existing `web/index.html` served by FastAPI, so
its relative `/api/*` calls work unchanged.

## Prerequisites (build machine)

- macOS on Apple Silicon
- Xcode Command Line Tools: `xcode-select --install`
- Rust: `curl https://sh.rustup.rs -sSf | sh`
- Node 18+ and npm
- `ffmpeg` for runtime (`brew install ffmpeg`) — the one dependency the app
  cannot vendor; the setup screen tells the user if it is missing.

## Develop

```bash
cd desktop
npm install
npm run dev        # runs scripts/sync-resources.sh, then `tauri dev`
```

`sync-resources.sh` copies `server.py`, `patch_tribe.py`, `requirements-extra.txt`,
and `web/` from the repo root into `src-tauri/resources/payload/` (single source
of truth). `bootstrap.sh` and `config.default.json` live in the payload dir.

## Build a .dmg

```bash
cd desktop
# one-time: generate app icons from a 1024x1024 png
npm run tauri icon ../path/to/icon-1024.png
npm run build      # -> src-tauri/target/release/bundle/dmg/*.dmg
```

### Signing & notarization (for distribution)

Unsigned apps trigger Gatekeeper warnings. To distribute:

- Apple Developer account ($99/yr), a "Developer ID Application" certificate.
- Set the Tauri signing env vars before `npm run build`:
  `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`,
  `APPLE_ID`, `APPLE_PASSWORD` (app-specific), `APPLE_TEAM_ID`.
- Tauri then signs and notarizes the `.dmg` during the build.

## Data locations

- App runtime + venv + vendored tribev2: `~/Library/Application Support/co.kumard3.tribescorer/app/`
- Model weights: Hugging Face cache (`~/.cache/huggingface`) + `<app>/cache/`
- Config: `<app>/config.json` (port is rewritten to a free port each launch)

## Notes / limits

- **License:** TRIBE weights are CC BY-NC. Keep the app free, research/personal
  use. Do not sell or use commercially.
- **Footprint:** ~16 GB disk, 16 GB+ RAM recommended; first run needs internet.
- Windows/Linux are out of scope for v1 (would add CUDA/CPU torch variants).
- This codebase was authored without a local Tauri build; compile it on a Mac
  with the prerequisites above. If `cargo`/`tauri` flags drift between minor
  versions, `npm run tauri --help` shows the current surface.
