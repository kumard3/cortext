# TRIBE Scorer — local API

The backend (`server.py`) is a FastAPI app. Build apps and agents on top of it.
The desktop app runs it on a random free `127.0.0.1` port (see
`~/Library/Application Support/co.kumard3.tribescorer/app/config.json`);
`./run.sh` runs it on `127.0.0.1:8011`.

## Auth

- On a loopback bind (`127.0.0.1`), writes need no key.
- On a public bind, an API key is required and auto-generated if unset; send it
  as the `X-API-Key` header on writes. Reads are open.

## Scoring is asynchronous

`POST /api/score` enqueues a job and returns immediately. Results arrive later
on `GET /api/results` (and live on `GET /api/stream`). Each item takes a few
minutes on CPU. Poll results, or subscribe to the SSE stream.

## Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/` | — | the web UI |
| GET | `/api/status` | — | `{status: loading\|ready\|error, device, devices, results, queued}` |
| GET | `/api/config` | — | current config (no key); `available`/`resolved_devices` |
| POST | `/api/config` | `{device?, text_device?, audio_device?, text_model?, text_layers?, save_raw?, api_key?}` | changing model fields triggers a reload. `text_model` must be in the allowlist |
| GET | `/api/results` | — | array of scored items |
| GET | `/api/result/{id}/raw?fmt=npy\|json` | — | raw `(timesteps x 20484)` predictions |
| POST | `/api/score` | `{texts: string[]}` | `{job, n, queued}` |
| POST | `/api/score/file` | multipart `file` | audio/video; `{job, modality, queued}` |
| DELETE | `/api/results` | — | clears results + raw files |
| GET | `/api/stream` | — | SSE: `status`, `stage`, `draft_start`, `result`, `job_done`, `log` |

## Result item

```json
{
  "id": "ab12cd34",
  "text": "your draft",
  "modality": "text",
  "chars": 142,
  "n_timesteps": 12,
  "n_vertices": 20484,
  "total_activation": 1234.5,
  "mean_activation": 0.05,
  "peak_activation": 9.8,
  "per_vertex_variance_mean": 0.12,
  "time_to_peak_sec": 3,
  "seconds": 91.2,
  "raw": "ab12cd34.npy"
}
```

## Ranking guidance

Rank by **`peak_activation`** and **`per_vertex_variance_mean`** (length-independent).
**Do not** rank by `total_activation` — it correlates ~0.97 with text length.
TRIBE predicts brain response to passively consumed media, **not** likes/upvotes:
treat any ranking as a salience tiebreaker, not an engagement oracle.

## Examples

```bash
BASE=http://127.0.0.1:8011

# submit
curl -s -X POST $BASE/api/score -H 'Content-Type: application/json' \
  -d '{"texts":["draft one","draft two"]}'

# poll
curl -s $BASE/api/results | jq 'sort_by(-.peak_activation) | .[] | {text, peak_activation}'

# raw predictions for one item
curl -s "$BASE/api/result/ab12cd34/raw?fmt=npy" -o tribe_ab12cd34.npy
```
