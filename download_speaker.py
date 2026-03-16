#!/usr/bin/env python3
"""
Download a reference speaker WAV for XTTS v2 voice cloning.

XTTS v2 needs 6+ seconds of clean speech from the target speaker.
This script downloads a high-quality Thorsten voice sample from his
official dataset on HuggingFace — the same voice used in the DDC model,
so the output sounds consistent regardless of which model you use.

Usage:
    uv run python download_speaker.py               # saves speaker_thorsten.wav
    uv run python download_speaker.py --out my.wav  # custom output path
    uv run python download_speaker.py --list        # show all available samples
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Output path that pdf_to_audio.py looks for by default
DEFAULT_OUT = Path(__file__).parent / "speaker_thorsten.wav"

# Thorsten's CC0 dataset on HuggingFace — long, clean, studio-quality recordings.
# We pick samples that are 8–15 seconds (ideal for XTTS v2 voice cloning).
HF_BASE = "https://huggingface.co/datasets/thorsten-voice/thorsten-neutral/resolve/main/wavs"

SAMPLES = {
    "thorsten_01": {
        "url":  f"{HF_BASE}/thorsten-neutral_00001.wav",
        "desc": "Neutral tone, ~10 s — recommended default",
    },
    "thorsten_02": {
        "url":  f"{HF_BASE}/thorsten-neutral_00002.wav",
        "desc": "Neutral tone, ~8 s",
    },
    "thorsten_03": {
        "url":  f"{HF_BASE}/thorsten-neutral_00003.wav",
        "desc": "Neutral tone, ~12 s",
    },
}

DEFAULT_SAMPLE = "thorsten_01"


class _Progress:
    def __call__(self, block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct   = min(100, int(block * block_size * 100 / total))
        filled = pct // 2
        bar   = "█" * filled + "░" * (50 - filled)
        kb    = block * block_size // 1024
        print(f"\r  [{bar}] {pct:3d}%  {kb} KB", end="", flush=True)


def download(sample_key: str, out_path: Path) -> None:
    meta = SAMPLES.get(sample_key)
    if not meta:
        print(f"❌  Unknown sample: {sample_key}")
        print("   Run with --list to see available samples.")
        sys.exit(1)

    print(f"\n📥  Downloading speaker sample: {sample_key}")
    print(f"   {meta['desc']}")
    print(f"   {meta['url']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(meta["url"], out_path, reporthook=_Progress())
        print(f"\n✅  Saved to: {out_path}")
    except urllib.error.HTTPError as e:
        print(f"\n❌  HTTP {e.code} — sample not found at that URL.")
        print("   The HuggingFace dataset layout may have changed.")
        print("   You can supply any 6+ second German WAV with --speaker-wav.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n❌  Network error: {e.reason}")
        sys.exit(1)

    print(f"\nUsage:")
    print(f"  uv run python pdf_to_audio.py doc.pdf \\")
    print(f"      --model tts_models/multilingual/multi-dataset/xtts_v2 \\")
    print(f"      --speaker-wav {out_path}")
    print(f"\n  Or just run without --speaker-wav if you saved to the default location:")
    print(f"  uv run python pdf_to_audio.py doc.pdf \\")
    print(f"      --model tts_models/multilingual/multi-dataset/xtts_v2")


def list_samples() -> None:
    print("\nAvailable Thorsten speaker samples:\n")
    for key, meta in SAMPLES.items():
        marker = " ← default" if key == DEFAULT_SAMPLE else ""
        print(f"  {key:<20} {meta['desc']}{marker}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Download Thorsten speaker WAV for XTTS v2")
    p.add_argument("--sample", default=DEFAULT_SAMPLE,
                   help=f"Which sample to download (default: {DEFAULT_SAMPLE})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output path (default: {DEFAULT_OUT})")
    p.add_argument("--list", "-l", action="store_true",
                   help="List available samples and exit")
    args = p.parse_args()

    if args.list:
        list_samples()
        return

    if args.out.exists():
        print(f"\n✅  Already exists: {args.out}  ({args.out.stat().st_size // 1024} KB)")
        print("   Delete it and re-run to replace.")
        return

    download(args.sample, args.out)


if __name__ == "__main__":
    main()
