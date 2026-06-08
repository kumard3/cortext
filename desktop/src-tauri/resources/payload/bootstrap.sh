#!/usr/bin/env bash
# First-run setup for Cortext. Idempotent: safe to re-run.
# Usage: bootstrap.sh <APP_DIR>
#
# Self-contained: uv/uvx are bundled in $APP_DIR/bin, uv brings its own standalone
# Python (the system Python is never used), and ffmpeg comes from the imageio-ffmpeg
# dependency. So the host needs no Python, no uv, and no system ffmpeg.
set -euo pipefail

APP_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
cd "$APP_DIR"
# Bundled uv/uvx first, then common locations as a fallback.
export PATH="$APP_DIR/bin:$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "==> Cortext first-run setup in $APP_DIR"

# 1. uv (bundled). Only install if somehow missing.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not bundled, installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv unavailable"; exit 2; }

# 2. Python 3.12 venv. uv downloads a standalone CPython if needed; the system
#    Python is never touched.
echo "==> creating Python 3.12 environment"
# Prefer uv's own managed Python (host-independent). If it can't be fetched
# (GitHub down / offline) and a system Python 3.12 exists, fall back to that so
# setup still succeeds.
uv venv --python 3.12 --python-preference only-managed .venv \
  || uv venv --python 3.12 .venv

# 3. tribev2 source (editable, so the inference patch can apply)
if [ ! -d vendor/tribev2 ]; then
  echo "==> downloading TRIBE v2 source"
  git clone --depth 1 https://github.com/facebookresearch/tribev2.git vendor/tribev2
fi

# 4. the EXACT locked dependency set (deterministic, proven end-to-end). Includes
#    torch, transformers==4.57.6, exca==0.5.20, and imageio-ffmpeg.
echo "==> installing pinned dependencies (locked set, longest step)"
uv pip install --python .venv -r requirements.lock

# 5. tribev2 itself (deps already locked above)
echo "==> installing TRIBE v2 (deps already locked)"
uv pip install --python .venv -e "./vendor/tribev2[plotting]" --no-deps

# 6. spaCy English model, pinned to the exact release wheel
echo "==> installing spaCy en_core_web_sm (pinned)"
uv pip install --python .venv "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# 7. apply the inference fixes (whisperx pins, silero VAD, int8, gTTS English)
echo "==> applying inference fixes"
.venv/bin/python patch_tribe.py

# 8. ffmpeg from imageio-ffmpeg, so no system ffmpeg is required. whisperx (run via
#    uvx) finds it because the app puts $APP_DIR/.venv/bin on PATH.
echo "==> linking ffmpeg (from imageio-ffmpeg)"
FFMPEG="$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG" .venv/bin/ffmpeg

# Mark complete only after every step above succeeded (set -e guards this).
touch "$APP_DIR/.setup_complete"
echo "==> setup complete (models download on first score, ~12GB, one time)"
