// TRIBE Scorer desktop shell (macOS, Apple Silicon).
//
// Flow: setup.html loads -> JS attaches listeners -> invoke("start") ->
// background thread copies the payload, runs first-run bootstrap (uv venv +
// vendored tribev2 + deps + the 4 fixes), launches server.py as a managed
// sidecar, waits for the local port, then navigates the window to the running
// web UI. The Python server downloads the ~12 GB of models itself on first
// model load; the web UI shows that progress via /api/status.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use tauri::{AppHandle, Emitter, Manager, RunEvent};

struct ServerState {
    child: Arc<Mutex<Option<Child>>>,
    started: Arc<Mutex<bool>>,
}

const EVENT: &str = "setup";

fn emit_stage(app: &AppHandle, stage: &str, message: &str) {
    let _ = app.emit(EVENT, json!({"kind": "stage", "stage": stage, "message": message}));
}
fn emit_log(app: &AppHandle, line: &str) {
    let _ = app.emit(EVENT, json!({"kind": "log", "message": line}));
}
fn emit_error(app: &AppHandle, message: &str) {
    let _ = app.emit(EVENT, json!({"kind": "error", "message": message}));
}
fn emit_ready(app: &AppHandle, url: &str) {
    let _ = app.emit(EVENT, json!({"kind": "ready", "url": url}));
}

/// PATH with the common homebrew + uv locations prepended, so uv/ffmpeg resolve.
fn augmented_path() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let extra = format!(
        "{home}/.local/bin:{home}/.cargo/bin:/opt/homebrew/bin:/usr/local/bin",
        home = home
    );
    match std::env::var("PATH") {
        Ok(p) => format!("{extra}:{p}"),
        Err(_) => extra,
    }
}

/// PATH with the app's bundled bin (uv/uvx) and the venv bin (ffmpeg) in front,
/// so the host needs no uv / ffmpeg / system Python at runtime.
fn run_path(app_dir: &Path) -> String {
    format!(
        "{ad}/bin:{ad}/.venv/bin:{base}",
        ad = app_dir.display(),
        base = augmented_path()
    )
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Copy server.py / web / patch_tribe.py / requirements / bootstrap.sh / config
/// from the bundled payload into the writable app dir. Runtime dirs (.venv,
/// vendor, cache, config.json) are never overwritten once they exist.
fn ensure_payload(app: &AppHandle, app_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(app_dir).map_err(|e| e.to_string())?;
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resource dir: {e}"))?;
    let payload = resource_dir.join("payload");
    if !payload.exists() {
        return Err(format!("bundled payload not found at {}", payload.display()));
    }
    let version = app.package_info().version.to_string();
    let marker = app_dir.join(".payload_version");
    let up_to_date = std::fs::read_to_string(&marker).map(|v| v.trim() == version).unwrap_or(false);
    if up_to_date {
        return Ok(());
    }
    // Copy code/UI files (overwrite), leave runtime dirs alone.
    for name in ["server.py", "patch_tribe.py", "requirements-extra.txt", "requirements.lock", "bootstrap.sh"] {
        let from = payload.join(name);
        if from.exists() {
            std::fs::copy(&from, app_dir.join(name)).map_err(|e| e.to_string())?;
        }
    }
    let web_src = payload.join("web");
    if web_src.exists() {
        copy_dir_recursive(&web_src, &app_dir.join("web")).map_err(|e| e.to_string())?;
    }
    // Bundled uv/uvx binaries -> app dir, made executable.
    let bin_src = payload.join("bin");
    if bin_src.exists() {
        copy_dir_recursive(&bin_src, &app_dir.join("bin")).map_err(|e| e.to_string())?;
        use std::os::unix::fs::PermissionsExt;
        for exe in ["uv", "uvx"] {
            let p = app_dir.join("bin").join(exe);
            if p.exists() {
                let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755));
            }
        }
    }
    // config.json: seed from default only if absent (preserve user settings).
    let cfg = app_dir.join("config.json");
    if !cfg.exists() {
        let def = payload.join("config.default.json");
        if def.exists() {
            std::fs::copy(&def, &cfg).map_err(|e| e.to_string())?;
        }
    }
    std::fs::write(&marker, &version).map_err(|e| e.to_string())?;
    Ok(())
}

fn is_installed(app_dir: &Path) -> bool {
    // Marker written only after bootstrap fully succeeds. Checking for the venv /
    // vendor dirs alone gives false positives when an install is interrupted.
    app_dir.join(".setup_complete").exists()
}

/// Run a command, streaming stdout+stderr to the UI as log events. Err on non-zero exit.
fn run_streamed(app: &AppHandle, cmd: &mut Command) -> Result<(), String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    if let Some(out) = child.stdout.take() {
        let app2 = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                emit_log(&app2, &line);
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        let app2 = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                emit_log(&app2, &line);
            }
        });
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("step exited with {}", status.code().unwrap_or(-1)))
    }
}

fn free_port() -> Result<u16, String> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    Ok(l.local_addr().map_err(|e| e.to_string())?.port())
}

/// Force loopback host + the chosen port into the app's config.json.
fn write_port(app_dir: &Path, port: u16) -> Result<(), String> {
    let cfg_path = app_dir.join("config.json");
    let mut cfg: serde_json::Value = std::fs::read_to_string(&cfg_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    cfg["host"] = json!("127.0.0.1");
    cfg["port"] = json!(port);
    if cfg.get("device").is_none() {
        cfg["device"] = json!("auto");
    }
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap())
        .map_err(|e| e.to_string())
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(addr) = addr.parse() {
            if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok() {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(600));
    }
    false
}

fn run_setup(app: &AppHandle, child_slot: &Arc<Mutex<Option<Child>>>) -> Result<String, String> {
    let app_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data dir: {e}"))?
        .join("app");

    emit_stage(app, "preparing", "Preparing application files");
    ensure_payload(app, &app_dir)?;

    if !is_installed(&app_dir) {
        emit_stage(
            app,
            "installing",
            "First-run setup: building the runtime and dependencies (a few minutes)",
        );
        let mut cmd = Command::new("bash");
        cmd.arg(app_dir.join("bootstrap.sh"))
            .arg(&app_dir)
            .current_dir(&app_dir)
            .env("PATH", run_path(&app_dir));
        run_streamed(app, &mut cmd).map_err(|e| {
            format!("setup failed: {e}. If it mentions ffmpeg, install it (brew install ffmpeg) and reopen.")
        })?;
    }

    let port = free_port()?;
    write_port(&app_dir, port)?;

    emit_stage(app, "starting", "Starting the scoring engine");
    let mut server = Command::new(app_dir.join(".venv/bin/python"));
    server
        .arg("server.py")
        .current_dir(&app_dir)
        .env("PATH", augmented_path())
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut server = server.spawn().map_err(|e| format!("could not start server: {e}"))?;
    // Forward the server log to the UI until we navigate away.
    if let Some(out) = server.stdout.take() {
        let app2 = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                emit_log(&app2, &line);
            }
        });
    }
    if let Some(err) = server.stderr.take() {
        let app2 = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                emit_log(&app2, &line);
            }
        });
    }
    *child_slot.lock().unwrap() = Some(server);

    emit_stage(app, "connecting", "Waiting for the local server");
    if !wait_for_port(port, Duration::from_secs(120)) {
        return Err("server did not come up in time".into());
    }
    Ok(format!("http://127.0.0.1:{port}"))
}

fn do_setup(app: AppHandle, child_slot: Arc<Mutex<Option<Child>>>, started: Arc<Mutex<bool>>) {
    match run_setup(&app, &child_slot) {
        Ok(url) => {
            emit_ready(&app, &url);
            if let Some(win) = app.get_webview_window("main") {
                if let Ok(u) = url.parse() {
                    let _ = win.navigate(u);
                }
            }
        }
        Err(e) => {
            emit_error(&app, &e);
            *started.lock().unwrap() = false; // allow the user to retry
        }
    }
}

#[tauri::command]
fn start(app: AppHandle, state: tauri::State<ServerState>) {
    {
        let mut started = state.started.lock().unwrap();
        if *started {
            return;
        }
        *started = true;
    }
    let child_slot = state.child.clone();
    let started = state.started.clone();
    std::thread::spawn(move || do_setup(app, child_slot, started));
}

fn main() {
    let state = ServerState {
        child: Arc::new(Mutex::new(None)),
        started: Arc::new(Mutex::new(false)),
    };
    let child_for_exit = state.child.clone();

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![start])
        .build(tauri::generate_context!())
        .expect("error building TRIBE Scorer")
        .run(move |_app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = child_for_exit.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
