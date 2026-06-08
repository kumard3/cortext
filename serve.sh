#!/usr/bin/env bash
# Run the API + expose it over a Cloudflare tunnel.
#   ./serve.sh                 -> quick tunnel (ephemeral *.trycloudflare.com URL)
#   ./serve.sh <tunnel-name>   -> named tunnel (stable domain; requires `cloudflared login` first)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: not installed. Run ./install.sh first."; exit 1
fi
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found."
  echo "  macOS:  brew install cloudflared"
  echo "  Linux:  https://pkg.cloudflare.com  (or re-run ./install.sh)"
  exit 1
fi

# The tunnel exposes the API publicly, so require an API key. Generate one if
# missing (the server binds loopback, so it won't self-detect the public exposure).
KEY=$(.venv/bin/python - <<'PY'
import json, os, secrets
p = "config.json"
cfg = json.load(open(p)) if os.path.exists(p) else {}
if not cfg.get("api_key"):
    cfg["api_key"] = secrets.token_urlsafe(32)
    json.dump(cfg, open(p, "w"), indent=2)
print(cfg["api_key"])
PY
)
echo "================================================================"
echo "Tunnel is public. Writes require this header:"
echo "  X-API-Key: $KEY"
echo "In the web UI: Settings > 'API key for your requests'."
echo "================================================================"

PORT=$(.venv/bin/python -c "import json;print(json.load(open('config.json')).get('port',8011))" 2>/dev/null || echo 8011)

echo "==> starting API on :$PORT"
.venv/bin/python server.py &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT
sleep 4

if [ "$#" -ge 1 ]; then
  echo "==> named tunnel '$1' -> http://localhost:$PORT"
  exec cloudflared tunnel run --url "http://localhost:$PORT" "$1"
else
  echo "==> quick tunnel (grab the https://*.trycloudflare.com URL below)"
  exec cloudflared tunnel --url "http://localhost:$PORT"
fi
