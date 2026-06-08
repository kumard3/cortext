"""Apply the inference fixes to the vendored tribev2.

Upstream runs whisperx via `uvx whisperx`, which resolves the latest
torch/torchaudio. torchaudio >= 2.9 dropped `list_audio_backends`, which
pyannote (whisperx's default VAD) still calls, so transcription crashes.
This patch:

  1. pins py3.12 + torch/torchaudio 2.8 + whisperx 3.7.2 in the uvx call
  2. switches the VAD to silero (pyannote's checkpoint fails torch 2.6+
     weights_only pickling)
  3. uses int8 compute on CPU (float16 is not a valid CPU compute type)
  6. forces gTTS to English: demo_utils auto-detects the draft language with
     langdetect, which misfires on short English text (e.g. returns 'so'), and
     gTTS then raises "Language not supported". The pipeline is English-only
     (transcription is hard-coded to "english"), so we pin the TTS lang too.

The 4th fix (gated meta-llama/Llama-3.2-3B -> non-gated mirror) lives in
server.py / score.py via config_update. The 5th fix (exca==0.5.20 pin) lives in
the installer, not here.

Idempotent: re-running is a no-op. Patches run independently.
"""

import re
import sys
from pathlib import Path

EVENTS = Path("vendor/tribev2/tribev2/eventstransforms.py")
DEMO = Path("vendor/tribev2/tribev2/demo_utils.py")


def patch_events() -> None:
    if not EVENTS.exists():
        raise SystemExit(f"ERROR: {EVENTS} not found. Run install (it clones tribev2) first.")
    s = EVENTS.read_text()
    if "--vad_method" in s and "--from" in s and 'else "int8"' in s:
        print("eventstransforms: already patched")
        return

    s = s.replace(
        'compute_type = "float16"',
        'compute_type = "float16" if device == "cuda" else "int8"',
        1,
    )
    s = re.sub(
        r'"uvx",\s*\n\s*"whisperx",',
        '"uvx",\n'
        '                "--python", "3.12",\n'
        '                "--with", "torch==2.8.0",\n'
        '                "--with", "torchaudio==2.8.0",\n'
        '                "--from", "whisperx==3.7.2",\n'
        '                "whisperx",',
        s,
        count=1,
    )
    s = s.replace(
        '"--align_model",',
        '"--vad_method",\n                "silero",\n                "--align_model",',
        1,
    )
    EVENTS.write_text(s)
    print(f"patched {EVENTS}")


def patch_demo() -> None:
    if not DEMO.exists():
        raise SystemExit(f"ERROR: {DEMO} not found. Run install (it clones tribev2) first.")
    s = DEMO.read_text()
    if 'lang = "en"' in s:
        print("demo_utils: already patched")
        return
    if "lang = detect(self.text)" not in s:
        print("demo_utils: nothing to patch (upstream changed?)")
        return
    s = s.replace(
        "lang = detect(self.text)",
        'lang = "en"  # forced: langdetect misfires on short English drafts',
        1,
    )
    DEMO.write_text(s)
    print(f"patched {DEMO}")


def main() -> int:
    patch_events()
    patch_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
