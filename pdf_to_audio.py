#!/usr/bin/env python3
"""
PDF → Audio Converter  (German)
================================
Uses Piper TTS with the Thorsten German voice for fast, high-quality
offline text-to-speech.  Optimised for Apple Silicon (M4 Pro).

Usage
-----
    python pdf_to_audio.py document.pdf
    python pdf_to_audio.py *.pdf --output-dir ./audio --speed 1.1
    python pdf_to_audio.py doc.pdf --model de_DE-thorsten_emotional-medium --style happy
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

# ── Dependency check (friendly messages before hard crash) ────────────────────
MISSING: list[str] = []
try:
    import fitz  # PyMuPDF
except ImportError:
    MISSING.append("pymupdf")

try:
    from piper import PiperVoice
except ImportError:
    MISSING.append("piper-tts")

try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")

try:
    import soundfile as sf
except ImportError:
    MISSING.append("soundfile")

if MISSING:
    print("❌  Missing dependencies. Install them with:\n")
    print(f"    pip install {' '.join(MISSING)}\n")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL = "de_DE-thorsten-high"

# Silence durations (milliseconds)
SILENCE_PARAGRAPH_MS = 800   # pause between paragraphs
SILENCE_SENTENCE_MS  = 250   # pause between sentences / chunks

# Maximum characters sent to TTS in a single call.
# Piper handles long strings fine, but chunking at sentence boundaries
# gives more natural prosody and avoids any edge-case timeouts.
MAX_CHUNK_CHARS = 400

# Available speaking styles for thorsten_emotional model
EMOTIONAL_STYLES = ["neutral", "happy", "angry", "disgusted",
                    "fearful", "sad", "surprised"]


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    """Return cleaned plain text from all pages of a PDF."""
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def clean_german_text(raw: str) -> str:
    """
    Clean PDF-extracted German text for TTS consumption.

    Handles:
    - Hyphenated line-breaks  (Binde-\\nstrich → Bindestrich)
    - Standalone page numbers
    - Excessive whitespace / form feeds
    - Common PDF extraction artefacts
    """
    text = raw

    # Merge hyphenated words split across lines (German is hyphen-heavy)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace form-feed characters with paragraph breaks
    text = re.sub(r"\f", "\n\n", text)

    # Remove lines that are just a page number (1–4 digits, optional whitespace)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Collapse more than 2 consecutive newlines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [l.rstrip() for l in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ── Text chunking ─────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> List[str]:
    """Split a paragraph into sentences using punctuation heuristics."""
    # Split on . ! ? followed by whitespace/end, but keep abbreviations mostly intact
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[Optional[str]]:
    """
    Split text into TTS chunks.

    Returns a list where:
      - A non-empty string  → synthesise this
      - None                → insert a paragraph-level silence
    """
    paragraphs = re.split(r"\n\n+", text)
    result: list[Optional[str]] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            result.append(para)
        else:
            sentences = _sentence_split(para)
            current = ""
            for sent in sentences:
                if not sent:
                    continue
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        result.append(current)
                    # If a single sentence is too long, split at comma
                    if len(sent) > max_chars:
                        parts = re.split(r"(?<=,)\s+", sent)
                        bucket = ""
                        for part in parts:
                            if len(bucket) + len(part) + 1 <= max_chars:
                                bucket = (bucket + " " + part).strip()
                            else:
                                if bucket:
                                    result.append(bucket)
                                bucket = part
                        if bucket:
                            result.append(bucket)
                    else:
                        current = sent
            if current:
                result.append(current)

        result.append(None)  # paragraph pause after each paragraph

    # Remove trailing None
    while result and result[-1] is None:
        result.pop()

    return result


# ── Audio helpers ─────────────────────────────────────────────────────────────

def silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Return an array of zeros (silence) of the given duration."""
    return np.zeros(int(sample_rate * duration_ms / 1000.0), dtype=np.float32)


def synth_chunk(
    voice: PiperVoice,
    text: str,
    speaker_id: Optional[int] = None,
    length_scale: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Synthesise one text chunk.  Returns (float32 audio array, sample_rate).

    Real piper-tts API (confirmed from source):
      voice.synthesize(text, syn_config) -> Iterable[AudioChunk]
      AudioChunk.audio_float_array       -> np.ndarray float32 in [-1, 1]
      AudioChunk.sample_rate             -> int
    SynthesisConfig(length_scale=...)    controls speaking speed.
    """
    from piper.config import SynthesisConfig

    syn_config = SynthesisConfig(length_scale=length_scale)

    arrays: list[np.ndarray] = []
    sample_rate: int = 22050  # fallback; overwritten by first chunk

    for chunk in voice.synthesize(text, syn_config=syn_config):
        sample_rate = chunk.sample_rate
        arrays.append(chunk.audio_float_array)

    pcm = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    return pcm, sample_rate


# ── Core pipeline ─────────────────────────────────────────────────────────────

def convert(
    pdf_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    models_dir: Path = MODELS_DIR,
    speed: float = 1.0,
    speaker_id: Optional[int] = None,
) -> None:
    """Full PDF → audio pipeline for a single file."""

    # 1. Extract text
    bar("📄", f"Reading  {pdf_path.name}")
    raw_text = extract_text(pdf_path)
    clean    = clean_german_text(raw_text)
    chunks   = chunk_text(clean)

    n_chunks = sum(1 for c in chunks if c is not None)
    bar("✂️ ", f"Chunked into {n_chunks} segments  ({len(clean):,} chars)")

    # 2. Load model
    onnx_path   = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"

    if not onnx_path.exists():
        print(f"\n❌  Model not found: {onnx_path}")
        print(    "   Run:  python download_model.py\n")
        sys.exit(1)

    bar("🧠", f"Loading  {model_name}")
    # length_scale = 1/speed  (Piper convention: >1 = slower)
    length_scale = 1.0 / max(0.5, min(speed, 2.0))
    voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))

    # 3. Synthesise
    bar("🎙️ ", "Synthesising …")
    segments: list[np.ndarray] = []
    sample_rate: Optional[int] = None
    done = 0

    t_synth_start = time.time()
    for chunk in chunks:
        if chunk is None:
            # Paragraph pause
            if sample_rate:
                segments.append(silence(SILENCE_PARAGRAPH_MS, sample_rate))
            continue

        audio, sr = synth_chunk(voice, chunk, speaker_id=speaker_id,
                                length_scale=length_scale)
        if sample_rate is None:
            sample_rate = sr

        segments.append(audio)
        segments.append(silence(SILENCE_SENTENCE_MS, sample_rate))

        done += 1
        pct = int(done / n_chunks * 40)
        bar_str = "█" * pct + "░" * (40 - pct)
        print(f"\r   [{bar_str}] {done}/{n_chunks}", end="", flush=True)

    print()  # newline after progress bar
    synth_time = time.time() - t_synth_start

    # 4. Concatenate & save
    bar("💾", f"Writing  {output_path.name}")
    final = np.concatenate(segments)
    sf.write(str(output_path), final, sample_rate)

    duration_s = len(final) / sample_rate
    ratio = duration_s / synth_time if synth_time > 0 else 0
    bar("✅", f"Done!  duration={fmt_time(duration_s)}  "
            f"synth={synth_time:.1f}s  ({ratio:.0f}× real-time)")


# ── Utilities ─────────────────────────────────────────────────────────────────

def bar(icon: str, msg: str) -> None:
    print(f"\n  {icon}  {msg}")


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert German PDF files to audio (Piper TTS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_audio.py buch.pdf
  python pdf_to_audio.py *.pdf --output-dir ./hoerbuecher
  python pdf_to_audio.py doc.pdf --model de_DE-thorsten_emotional-medium --style happy
  python pdf_to_audio.py doc.pdf --speed 1.15 --format mp3
        """,
    )
    p.add_argument("pdfs", nargs="+", type=Path, metavar="PDF",
                   help="Input PDF file(s)")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                   help="Output directory (default: same folder as each PDF)")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   help=f"Piper model name (default: {DEFAULT_MODEL})")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                   help=f"Folder containing .onnx model files (default: {MODELS_DIR})")
    p.add_argument("--style", choices=EMOTIONAL_STYLES, default=None,
                   help="Speaking style for thorsten_emotional model")
    p.add_argument("--speed", type=float, default=1.0, metavar="RATE",
                   help="Speaking rate: 1.0=normal, 1.1=slightly faster (default: 1.0)")
    p.add_argument("--format", choices=["wav", "flac", "ogg"], default="wav",
                   help="Output audio format (default: wav)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Resolve emotional speaker_id from style name
    speaker_id: Optional[int] = None
    if args.style:
        if "emotional" not in args.model:
            print("⚠️   --style requires the thorsten_emotional model. Ignoring.")
        else:
            speaker_id = EMOTIONAL_STYLES.index(args.style)
            print(f"  🎭  Style: {args.style} (speaker_id={speaker_id})")

    errors: list[str] = []
    for pdf_path in args.pdfs:
        if not pdf_path.is_file():
            print(f"\n⚠️   Not found: {pdf_path} — skipping")
            continue

        out_dir = args.output_dir if args.output_dir else pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{pdf_path.stem}.{args.format}"

        t0 = time.time()
        try:
            convert(
                pdf_path   = pdf_path,
                output_path= output_path,
                model_name = args.model,
                models_dir = args.models_dir,
                speed      = args.speed,
                speaker_id = speaker_id,
            )
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            print(f"\n❌  Failed: {pdf_path.name}: {msg}")
            print("\n--- traceback ---")
            traceback.print_exc()
            print("---\n")
            errors.append(str(pdf_path))

        print(f"  ⏱️   Wall time: {fmt_time(time.time() - t0)}")

    if errors:
        print(f"\n⚠️   {len(errors)} file(s) failed: {', '.join(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()