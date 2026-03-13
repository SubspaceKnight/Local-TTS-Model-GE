# PDF → Audio Converter (German TTS)

Converts German PDF files to spoken audio using **Piper TTS** and the
**Thorsten** German voice — a high-quality, fully offline neural TTS model
that runs at 100× real-time on Apple Silicon.

---

## Requirements

- macOS (Apple Silicon M-series recommended)
- Python 3.10 or later
- ~100 MB disk for the default voice model

---

## Setup (one-time)

```bash
# 1. Create venv and install all dependencies in one shot
uv sync

# 2. Download the German voice model (~65 MB, from HuggingFace)
uv run python download_model.py

# 3. (Optional) Install ffmpeg for MP3 conversion
brew install ffmpeg
```

---

## Usage

### Convert a single PDF
```bash
uv run python pdf_to_audio.py mein_buch.pdf
# → mein_buch.wav  (same folder)
```

### Convert multiple PDFs to a specific output folder
```bash
uv run python pdf_to_audio.py *.pdf --output-dir ./hoerbuecher
```

### Adjust speaking speed
```bash
uv run python pdf_to_audio.py doc.pdf --speed 1.15   # 15 % faster
uv run python pdf_to_audio.py doc.pdf --speed 0.9    # slightly slower
```

### Use the Emotional model with a speaking style
```bash
# First download the emotional model
uv run python download_model.py --model de_DE-thorsten_emotional-medium

# Then use it — available styles: neutral happy angry disgusted fearful sad surprised
uv run python pdf_to_audio.py doc.pdf \
    --model de_DE-thorsten_emotional-medium \
    --style neutral
```

### Convert to MP3 (via ffmpeg)
```bash
uv run python pdf_to_audio.py doc.pdf
ffmpeg -i doc.wav -q:a 4 doc.mp3 && rm doc.wav
```

### Save as FLAC or OGG directly
```bash
uv run python pdf_to_audio.py doc.pdf --format flac
uv run python pdf_to_audio.py doc.pdf --format ogg
```

---

## Available German voices

```bash
uv run python download_model.py --list
```

| Model key                          | MB  | Voice              |
|------------------------------------|-----|--------------------|
| `de_DE-thorsten-high` ✨ default   | 65  | Thorsten (male)    |
| `de_DE-thorsten_emotional-medium`  | 65  | Thorsten + styles  |
| `de_DE-thorsten-medium`            | 30  | Thorsten (lighter) |
| `de_DE-stefanie-medium`            | 30  | Stefanie (female)  |
| `de_DE-kerstin-low`                | 10  | Kerstin (female)   |

---

## How it works

```
PDF file
  │
  ▼  PyMuPDF
Plain text
  │
  ▼  clean_german_text()
      • merge hyphenated line-breaks (Binde-\nstrich → Bindestrich)
      • strip page numbers & artefacts
      • normalise whitespace
  │
  ▼  chunk_text()
      • split at sentence boundaries (≤400 chars per chunk)
      • paragraph markers → silence gaps
  │
  ▼  Piper TTS (ONNX, runs on CPU via ONNX Runtime)
      • ~100× real-time on M4 Pro
  │
  ▼  NumPy concatenation
      • 250 ms silence between sentences
      • 800 ms silence between paragraphs
  │
  ▼  soundfile
Audio file (WAV / FLAC / OGG)
```

---

## Performance on M4 Pro (24 GB)

| Document            | Pages | Text chars | Synth time | Audio length | Speed  |
|---------------------|-------|------------|------------|--------------|--------|
| Short article       | 5     | ~12 000    | ~3 s       | ~5 min       | ~100×  |
| Business report     | 30    | ~70 000    | ~18 s      | ~30 min      | ~100×  |
| Novel chapter       | 50    | ~120 000   | ~30 s      | ~50 min      | ~100×  |

---

## Troubleshooting

**`piper-tts` install fails on Apple Silicon**

```bash
# Make sure you have the Xcode command-line tools
xcode-select --install
uv sync
```

**`soundfile` can't write OGG**

OGG requires `libsndfile` with OGG/Vorbis support.  Easiest fix:

```bash
brew install libsndfile
uv sync --reinstall-package soundfile
```

**Audio sounds "robotic" or cuts off mid-word**

Try reducing `--speed` slightly (e.g. `--speed 0.95`) or switch to
`de_DE-thorsten-high` if you're using a low-quality model.