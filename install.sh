#!/usr/bin/env bash
# One-shot installer for the TRIBE Scorer. Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> TRIBE Scorer install"

# 1. uv (Python package/venv manager + uvx, used to run whisperx)
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH. Add \$HOME/.local/bin to PATH and re-run."; exit 1; }

# 2. ffmpeg (whisperx decodes audio with it)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install it, then re-run ./install.sh"
  echo "  macOS:  brew install ffmpeg"
  echo "  Debian: sudo apt-get install -y ffmpeg"
  exit 1
fi

# 3. virtualenv (Python 3.12 required: whisperx + the torch pins resolve cleanly there)
uv venv --python 3.12 .venv

# 4. tribev2 itself, vendored + editable so the inference patch can be applied.
#    [plotting] pulls nilearn/nibabel/matplotlib for the brain maps.
if [ ! -d vendor/tribev2 ]; then
  echo "==> cloning facebookresearch/tribev2"
  git clone --depth 1 https://github.com/facebookresearch/tribev2.git vendor/tribev2
fi
uv pip install --python .venv -e "./vendor/tribev2[plotting]"

# 5. web UI deps
uv pip install --python .venv -r requirements-extra.txt

# 6. spaCy English model (used during event extraction)
.venv/bin/python -m spacy download en_core_web_sm

# 7. apply the 4 inference fixes to the vendored tribev2 (see patch_tribe.py)
.venv/bin/python patch_tribe.py

# 8. cloudflared for `./serve.sh` (best effort, non-fatal)
if ! command -v cloudflared >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install cloudflared || echo "(skip) install cloudflared yourself for ./serve.sh"
  else
    echo "(optional) install cloudflared for ./serve.sh: https://pkg.cloudflare.com"
  fi
fi

echo ""
echo "==> Done."
echo "    Local:   ./run.sh        then open http://127.0.0.1:8011"
echo "    Hosted:  ./serve.sh      (local + Cloudflare tunnel URL)"
echo "    NOTE: the first scored item downloads ~12GB of models (one time)."
echo "          video also downloads V-JEPA2 on first use (large)."
