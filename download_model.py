#!/usr/bin/env python3
"""
Download Piper TTS German voice models from HuggingFace.
Run this once before using pdf_to_audio.py.

Usage:
    python download_model.py                          # downloads default (thorsten-high)
    python download_model.py --model thorsten_emotional
    python download_model.py --list                   # show all available German models
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
HF_BASE    = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# ── Available German models ────────────────────────────────────────────────────
# Quality:  low (~10 MB) | medium (~30 MB) | high (~65 MB)
MODELS: dict[str, dict] = {
    "de_DE-thorsten-high": {
        "path":        "de/de_DE/thorsten/high",
        "description": "Thorsten · male · high quality ✨ (recommended)",
        "size_mb":     65,
        "sample_rate": 22050,
    },
    "de_DE-thorsten_emotional-medium": {
        "path":        "de/de_DE/thorsten_emotional/medium",
        "description": "Thorsten Emotional · male · styles: neutral happy angry disgusted fearful sad surprised",
        "size_mb":     65,
        "sample_rate": 22050,
    },
    "de_DE-thorsten-medium": {
        "path":        "de/de_DE/thorsten/medium",
        "description": "Thorsten · male · medium quality · faster/lighter",
        "size_mb":     30,
        "sample_rate": 22050,
    },
    "de_DE-kerstin-low": {
        "path":        "de/de_DE/kerstin/low",
        "description": "Kerstin · female · low quality · smallest/fastest",
        "size_mb":     10,
        "sample_rate": 16000,
    },
    "de_DE-eva_k-x_low": {
        "path":        "de/de_DE/eva_k/x_low",
        "description": "Eva K · female · extra-low quality · minimal resources",
        "size_mb":     5,
        "sample_rate": 16000,
    },
    "de_DE-pavoque-low": {
        "path":        "de/de_DE/pavoque/low",
        "description": "Pavoque · male · low quality",
        "size_mb":     10,
        "sample_rate": 16000,
    },
    "de_DE-ramona-low": {
        "path":        "de/de_DE/ramona/low",
        "description": "Ramona · female · low quality",
        "size_mb":     10,
        "sample_rate": 16000,
    },
    "de_DE-stefanie-medium": {
        "path":        "de/de_DE/stefanie/medium",
        "description": "Stefanie · female · medium quality",
        "size_mb":     30,
        "sample_rate": 22050,
    },
}

DEFAULT_MODEL = "de_DE-thorsten-high"


# ── Download helpers ──────────────────────────────────────────────────────────

class _ProgressBar:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._last = -1

    def __call__(self, block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(block * block_size * 100 / total))
        if pct == self._last:
            return
        self._last = pct
        filled = pct // 2
        bar    = "█" * filled + "░" * (50 - filled)
        mb_done = block * block_size / 1_048_576
        mb_total = total / 1_048_576
        print(f"\r   [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB",
              end="", flush=True)


def download_file(url: str, dest: Path) -> None:
    """Download a single file with a progress bar."""
    print(f"\n  ⬇️   {dest.name}")
    print(f"       {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_ProgressBar(dest.name))
        print()  # newline after progress bar
    except urllib.error.HTTPError as exc:
        print(f"\n  ❌  HTTP {exc.code}: {url}")
        raise
    except urllib.error.URLError as exc:
        print(f"\n  ❌  Network error: {exc.reason}")
        raise


def download_model(model_key: str) -> None:
    """Download the .onnx and .onnx.json files for a model."""
    if model_key not in MODELS:
        print(f"❌  Unknown model: {model_key}")
        print("   Run with --list to see available models.")
        sys.exit(1)

    meta     = MODELS[model_key]
    hf_path  = meta["path"]
    onnx_url = f"{HF_BASE}/{hf_path}/{model_key}.onnx"
    json_url = f"{HF_BASE}/{hf_path}/{model_key}.onnx.json"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    onnx_dest = MODELS_DIR / f"{model_key}.onnx"
    json_dest = MODELS_DIR / f"{model_key}.onnx.json"

    # Skip if already downloaded
    if onnx_dest.exists() and json_dest.exists():
        size_mb = onnx_dest.stat().st_size / 1_048_576
        print(f"\n✅  Already downloaded: {model_key}  ({size_mb:.0f} MB)")
        print(f"   Location: {MODELS_DIR}")
        return

    print(f"\n📦  Downloading: {model_key}")
    print(f"   {meta['description']}")
    print(f"   ~{meta['size_mb']} MB")

    download_file(onnx_url, onnx_dest)
    download_file(json_url, json_dest)

    # Quick sanity check — load and parse the JSON config
    try:
        cfg = json.loads(json_dest.read_text())
        sr  = cfg.get("audio", {}).get("sample_rate", "?")
        print(f"\n✅  Model ready  (sample rate: {sr} Hz)")
    except Exception:
        print("\n⚠️   Could not parse config JSON — model may be corrupt.")

    print(f"   Location: {MODELS_DIR}\n")


def list_models() -> None:
    print("\nAvailable German Piper TTS models:\n")
    print(f"  {'Model key':<45} {'MB':>4}  Description")
    print(f"  {'─'*45} {'─'*4}  {'─'*52}")
    for key, meta in MODELS.items():
        marker = " ← default" if key == DEFAULT_MODEL else ""
        print(f"  {key:<45} {meta['size_mb']:>4}  {meta['description']}{marker}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Download Piper TTS German voice models",
    )
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   help=f"Model key to download (default: {DEFAULT_MODEL})")
    p.add_argument("--list", "-l", action="store_true",
                   help="List all available German models and exit")
    p.add_argument("--all", action="store_true",
                   help="Download ALL available German models")
    args = p.parse_args()

    if args.list:
        list_models()
        return

    if args.all:
        for key in MODELS:
            download_model(key)
        return

    download_model(args.model)


if __name__ == "__main__":
    main()