# TRIBE Scorer — MCP server

Exposes a running TRIBE Scorer (desktop app or `./run.sh`) to MCP clients
(Claude Desktop / Claude Code / Cursor) as tools: `get_status`, `score_text`,
`list_results`.

## Install

```bash
# in a venv of your choice
uv pip install mcp          # or: pip install "mcp[cli]"
```

## Point it at your server

- If the **desktop app** is running, the server auto-discovers its port from
  `~/Library/Application Support/co.kumard3.tribescorer/app/config.json`.
- Otherwise set `TRIBE_API_URL` (default `http://127.0.0.1:8011`).
- If the server has an API key set, set `TRIBE_API_KEY` (loopback usually does not).

## Add to Claude Code

```bash
claude mcp add tribe-scorer --scope user \
  --env TRIBE_API_URL=http://127.0.0.1:8011 \
  -- /absolute/path/to/venv/bin/python /absolute/path/to/mcp/tribe_mcp.py
```

For Claude Desktop, add an equivalent stdio entry to its MCP config pointing at
the same `python tribe_mcp.py` command.

## Tools

- `get_status()` — model/server status; call before scoring.
- `score_text(texts: string[], timeout_seconds=1800)` — scores drafts, returns
  them ranked by `peak_activation`. Blocks (minutes per draft on CPU).
- `list_results(limit=20)` — recent scored results.

## Honest note

`score_text` ranks by `peak_activation` (length-independent salience), not
`total_activation` (which tracks text length). TRIBE predicts brain response to
passive media, not likes/upvotes — treat the ranking as a tiebreaker.
