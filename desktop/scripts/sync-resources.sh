#!/usr/bin/env bash
# Copy the backend payload (single-sourced from the repo root) into the Tauri
# bundle resources before a dev run or build. bootstrap.sh and
# config.default.json are authored directly in the payload dir and left alone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                 # tribe-scorer repo root
PAYLOAD="$(cd "$HERE/.." && pwd)/src-tauri/resources/payload"

mkdir -p "$PAYLOAD"
cp "$ROOT/server.py"               "$PAYLOAD/server.py"
cp "$ROOT/patch_tribe.py"          "$PAYLOAD/patch_tribe.py"
cp "$ROOT/requirements-extra.txt"  "$PAYLOAD/requirements-extra.txt"
rm -rf "$PAYLOAD/web"
cp -R "$ROOT/web" "$PAYLOAD/web"

# Bundle uv + uvx so the host needs no install tools / no system Python.
mkdir -p "$PAYLOAD/bin"
for b in uv uvx; do
  src="$(command -v "$b" || true)"
  if [ -n "$src" ]; then cp "$src" "$PAYLOAD/bin/$b"; chmod +x "$PAYLOAD/bin/$b"; else
    echo "WARNING: $b not found on PATH; the app won't be self-contained without it"; fi
done

echo "synced backend payload -> $PAYLOAD"
ls -1 "$PAYLOAD"
