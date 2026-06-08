// Setup screen controller. Listens to the Rust "setup" events, renders progress,
// and kicks off the backend. On success the Rust side navigates this window to
// the running web UI, so there is nothing to do for "ready" beyond a final note.

const stageEl = document.getElementById("stage");
const logEl = document.getElementById("log");
const barFill = document.getElementById("bar-fill");
const errorBox = document.getElementById("error");
const errorMsg = document.getElementById("error-msg");
const retry = document.getElementById("retry");

const STAGE_PCT = { preparing: 8, installing: 38, starting: 80, connecting: 92, ready: 100 };
let lines = [];

function appendLog(msg) {
  lines.push(msg);
  if (lines.length > 400) lines = lines.slice(-400);
  logEl.textContent = lines.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

function setStage(stage, message) {
  stageEl.textContent = message || stage;
  const pct = STAGE_PCT[stage];
  if (pct) {
    barFill.classList.add("determinate");
    barFill.style.width = pct + "%";
  }
}

function showError(message) {
  errorMsg.textContent = message;
  errorBox.hidden = false;
}

async function begin() {
  errorBox.hidden = true;
  try {
    await window.__TAURI__.core.invoke("start");
  } catch (e) {
    showError(String(e));
  }
}

function onEvent(e) {
  const p = (e && e.payload) || {};
  if (p.kind === "stage") setStage(p.stage, p.message);
  else if (p.kind === "log") appendLog(p.message);
  else if (p.kind === "ready") setStage("ready", "Opening Cortext…");
  else if (p.kind === "error") showError(p.message);
}

async function init() {
  if (!window.__TAURI__) {
    stageEl.textContent = "Please open the TRIBE Scorer app.";
    return;
  }
  await window.__TAURI__.event.listen("setup", onEvent);
  retry.addEventListener("click", begin);
  begin();
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
