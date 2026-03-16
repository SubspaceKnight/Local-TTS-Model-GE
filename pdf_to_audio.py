#!/usr/bin/env python3
"""
PDF → Audio Converter  (German)
================================
Supports two TTS backends, auto-selected by model name:

  Piper  (fast, ~100× real-time on Apple Silicon)
    --model de_DE-thorsten-high              ← default
    --model de_DE-thorsten_emotional-medium

  Coqui TTS  (clearer articulation, ~2–5× real-time on Apple Silicon CPU)
    --model tts_models/de/thorsten/tacotron2-DDC   ← recommended for clarity
    --model tts_models/de/thorsten/vits

  If the model name contains "/" → Coqui backend.
  Otherwise → Piper backend.

Usage
-----
    python pdf_to_audio.py document.pdf
    python pdf_to_audio.py document.pdf --model tts_models/de/thorsten/tacotron2-DDC
    python pdf_to_audio.py *.pdf --output-dir ./audio --speed 0.95
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
    import numpy as np
except ImportError:
    MISSING.append("numpy")

try:
    import soundfile as sf
except ImportError:
    MISSING.append("soundfile")

if MISSING:
    print("❌  Missing core dependencies. Install them with:\n")
    print(f"    uv add {' '.join(MISSING)}\n")
    sys.exit(1)

# Backend imports are deferred — only the one actually used is required.
# Piper:  pip install piper-tts
# Coqui:  pip install TTS
_piper_available = False
_coqui_available = False
try:
    from piper import PiperVoice
    _piper_available = True
except ImportError:
    pass
try:
    # coqui-tts (Idiap fork) installs the same internal TTS module as the
    # original coqui-ai/TTS — the import path is identical.
    from TTS.api import TTS as CoquiTTS
    _coqui_available = True
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL      = "de_DE-thorsten-high"          # Piper — fast
DDC_MODEL          = "tts_models/de/thorsten/tacotron2-DDC"  # Coqui — clearer
VITS_MODEL         = "tts_models/de/thorsten/vits"           # Coqui — alternative

def _is_coqui_model(model_name: str) -> bool:
    """Coqui model names contain '/' (e.g. tts_models/de/thorsten/tacotron2-DDC)."""
    return "/" in model_name

# Silence durations (milliseconds)
SILENCE_HEADING_MS   = 1400  # pause before a chapter/section heading
SILENCE_PARAGRAPH_MS = 900   # pause between paragraphs
SILENCE_SENTENCE_MS  = 320   # pause between sentences / chunks
                              # (was 250 — extra headroom prevents word-swallow)

# Chunk sizes per backend.
# Piper: short chunks (≤300 chars) prevent attention drift in the ONNX model.
# Coqui DDC: DDC's double-decoder consistency handles whole paragraphs cleanly;
#   passing longer text gives the model full sentence context and better prosody.
#   We cap at 800 chars to stay safe on very long German compound sentences.
MAX_CHUNK_CHARS_PIPER = 300
MAX_CHUNK_CHARS_DDC   = 800
MAX_CHUNK_CHARS = MAX_CHUNK_CHARS_PIPER  # default; overridden in convert()

# Audio post-processing
FADE_MS        = 8    # fade-in/out applied to every chunk (prevents clicks)
PEAK_HEADROOM  = 0.92 # normalise each chunk so peaks don't exceed this level

# Available speaking styles for thorsten_emotional model
EMOTIONAL_STYLES = ["neutral", "happy", "angry", "disgusted",
                    "fearful", "sad", "surprised"]


# ── PDF text extraction ─────────────────────────────────────────────────────

# Heading patterns for German academic/textbook PDFs.
# Matches:  "7.1", "7.1.2", "7.1.2.3"  optionally followed by title text.
# Also matches label-only lines like "Wichtig", "Merke", "Fazit", "Beispiel".
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.)+\d*"                    # numbered:  7.  7.1  7.1.2
    r"|"
    r"(?:Wichtig|Merke|Fazit|Beispiel|Zusammenfassung|Definition|Hinweis|Achtung)"
    r")\b",
    re.IGNORECASE,
)

# Figure / table caption patterns — these blocks are skipped entirely.
#
# Handles three cases seen in German textbook PDFs:
#
#  1. Caption with leading arrow/bullet symbol:
#       "▶ Abb. 7.2  Schematische Darstellung …"
#       "● Abb. 7.2  …"
#     → strip any leading non-alphanumeric chars before matching
#
#  2. Standard caption starts:
#       "Abb. 7.1"  "Abbildung 3:"  "Tab. 2.1"  "Quelle:"  "Legende:"
#
#  3. Figure-internal numbered call-out boxes extracted as text blocks:
#       "6. Enzyme, die sich im …"   "7. Die Wiederaufnahme …"
#     These are SHORT blocks (≤ 120 chars) starting with a single digit
#     followed by ". " — distinct from section headings like "7.1" which
#     contain a second dot.  We filter them by shape (short + single-digit
#     bullet) in _is_figure_callout(), not via regex, to avoid false positives.

_FIGURE_RE = re.compile(
    r"(?:"                               # no ^ anchor — we strip prefix first
    r"Abb(?:ildung)?\.?\s*\d"            # Abb. / Abbildung + digit
    r"|Tab(?:elle)?\.?\s*\d"             # Tab. / Tabelle + digit
    r"|Fig(?:ure)?\.?\s*\d"              # Fig. / Figure + digit
    r"|Quelle\s*:"                       # Quelle:
    r"|Legende\s*:"                      # Legende:
    r"|\((?:aus|nach|mod\. nach|©)"      # (aus:  (nach:  (©
    r"|©"                                # bare copyright symbol
    r"|Aus\s+[A-ZÄÖÜ]\w+"               # "Aus Ehlert 2003" / "Aus Müller …"
    r")",
    re.IGNORECASE,
)

# Matches the leading decoration chars that German textbooks put before captions
# e.g. "▶ Abb. …"  "● Abb. …"  "■ Abb. …"  "◀ Abb. …"
_CAPTION_PREFIX_RE = re.compile(r"^[^\w\(©]+")


# Minimum characters of descriptive text that must follow the figure identifier
# for a line to be treated as a caption rather than an inline reference.
#
# Real caption:    "Abb. 7.2  Schematische Darstellung einer Synapse …"  (long)
# Inline ref:      "Abb. 7.2, Schritt 5"                                 (short)
#
# We measure the length of whatever comes AFTER the leading identifier+number,
# e.g. after "Abb. 7.2 " we need at least 20 chars of title text.
_MIN_CAPTION_TITLE_CHARS = 20


def _is_figure_line(line: str) -> bool:
    """
    True only if this line is a standalone figure/table caption.

    Guards against false positives on inline references like:
        "… passiv aus (▶ Abb. 7.2, Schritt 5). Die Feuerrate …"
    where PyMuPDF might extract "(▶ Abb. 7.2, Schritt 5)" as a tiny block.

    A real caption has a descriptive title after the identifier; an inline
    reference snippet does not.  We enforce a minimum title length.
    """
    stripped = _CAPTION_PREFIX_RE.sub("", line.strip())   # drop ▶ ● etc.
    if not _FIGURE_RE.match(stripped):
        return False

    # Find where the identifier+number ends (e.g. after "Abb. 7.2 " or "Tab. 3:")
    # and measure how much descriptive text follows.
    m = re.match(
        r"(?:Abb(?:ildung)?\.?\s*[\d.]+|Tab(?:elle)?\.?\s*[\d.]+|"
        r"Fig(?:ure)?\.?\s*[\d.]+)\s*[:\-–]?\s*",
        stripped, re.IGNORECASE
    )
    if m:
        title_part = stripped[m.end():]
        return len(title_part) >= _MIN_CAPTION_TITLE_CHARS

    # For non-numbered patterns (Quelle:, Legende:, Aus …, ©) always match —
    # these never appear as inline references.
    return True


def _is_figure_callout(line: str) -> bool:
    """
    True for short figure-internal numbered call-out boxes like:
        "6. Enzyme, die sich im extrazellulären …"
        "7. Die Wiederaufnahme von Transmittern …"
    Heuristic: single digit + ". " at start, total length ≤ 160 chars,
    and the digit is NOT followed by another dot (that would be a section heading).
    """
    m = re.match(r"^(\d)\.\s", line.strip())
    if not m:
        return False
    # Reject "7.1" style headings — they have a dot after the digit group
    if re.match(r"^\d+\.\d", line.strip()):
        return False
    return len(line.strip()) <= 160


# A line is a "paragraph break signal" if it is blank OR starts a heading.
def _is_para_break(line: str) -> bool:
    return line.strip() == "" or bool(_HEADING_RE.match(line.strip()))


def _extract_page_columns(page) -> str:
    """
    Extract text from a page that may have 1 or 2 columns.

    Strategy: retrieve text blocks with their bounding boxes, detect whether
    the page has a gutter (two-column layout), then sort blocks into
    left-column-first, top-to-bottom order.
    """
    blocks = page.get_text("blocks")  # [(x0,y0,x1,y1, text, …), …]
    if not blocks:
        return ""

    page_w = page.rect.width
    midpoint = page_w / 2

    # Heuristic: if >20 % of blocks have their right edge below the midpoint
    # → two-column layout.
    left_blocks = [b for b in blocks if b[2] < midpoint * 1.15]
    two_col = len(left_blocks) / max(len(blocks), 1) > 0.20

    if two_col:
        col_left  = sorted([b for b in blocks if b[0] < midpoint], key=lambda b: b[1])
        col_right = sorted([b for b in blocks if b[0] >= midpoint], key=lambda b: b[1])
        ordered   = col_left + col_right
    else:
        ordered = sorted(blocks, key=lambda b: b[1])

    return "\n".join(b[4] for b in ordered if isinstance(b[4], str))


def extract_text(pdf_path: Path) -> str:
    """
    Extract text from all pages, handling two-column layouts.
    Returns raw multi-page text; cleaning happens in clean_and_fuse_lines().
    """
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(_extract_page_columns(page))
    doc.close()
    return "\f".join(pages)   # form-feed as page separator


def clean_and_fuse_lines(raw: str) -> str:
    """
    Turn raw PDF-extracted text into clean, TTS-ready prose.

    Problems addressed (all visible in the screenshot):
    ─────────────────────────────────────────────────────
    1. Soft hyphens:  "Diagnos-\\n  tik"  →  "Diagnostik"
    2. Hard line wraps inside a paragraph: join continuation lines
    3. Page numbers (isolated digits) removed
    4. Form-feed page separators → paragraph break
    5. Collapse excessive whitespace
    6. Preserve heading lines as their own paragraph
    7. Strip artefact characters (bullets "●", tab stops, etc.)

    Algorithm
    ─────────
    Process line-by-line, carrying a "current paragraph" buffer.
    - If a line ends with a hyphen AND the next non-blank line starts with
      a lowercase letter → remove hyphen, join directly.
    - If a line is a soft-wrap continuation (no sentence-final punctuation,
      next line starts lowercase) → join with a space.
    - Blank lines and heading lines flush the buffer and start a new para.
    """
    # ── Phase 1: normalise page separators and tabs ──────────────────────────
    text = raw.replace("\f", "\n\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

    lines = text.splitlines()

    # ── Phase 2: remove isolated page numbers ────────────────────────────────
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Pure page number: 1-4 digits, possibly prefixed by chapter ref like "234"
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        cleaned.append(line)
    lines = cleaned

    # ── Phase 3: fuse hyphenated and soft-wrapped lines ──────────────────────
    # Build a look-ahead list of non-empty stripped lines for context.
    fused: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            fused.append("")
            i += 1
            continue

        # Look ahead: find next non-empty line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        next_line = lines[j].strip() if j < len(lines) else ""

        # ── Rule A: soft hyphen join ─────────────────────────────────────────
        # "Diagnos-"  +  "tik und …"  →  "Diagnostik und …"
        if (stripped.endswith("-")
                and next_line
                and next_line[0].islower()
                and not _HEADING_RE.match(next_line)):
            fused.append(stripped[:-1])   # drop hyphen, no space
            # swallow the blank lines between i and j, then merge next_line
            # by NOT appending a newline — handled in phase 4
            lines[j] = fused.pop() + next_line   # merge directly into j
            i += 1
            continue

        # ── Rule B: soft-wrap join ───────────────────────────────────────────
        # Current line does NOT end with sentence-final punctuation (.!?:"),
        # and the next line starts with a lowercase letter (continuation).
        sentence_final  = re.search(r'[.!?:""»]$', stripped)
        is_heading_now  = _HEADING_RE.match(stripped)
        is_heading_next = _HEADING_RE.match(next_line) if next_line else False

        # ── Rule C: heading title continuation ──────────────────────────────
        # Handles multi-line headings like:
        #   "7.2 Anatomie und Funktion des"   ← matches _HEADING_RE
        #   "Nervensystems"                   ← continuation on next block
        # PyMuPDF often extracts the two parts as separate blocks with blank
        # lines between them, so we do NOT require j == i+1 — we look ahead
        # past any blank lines to find the continuation.
        # Guards: continuation must be short (≤ 60 chars), not itself a
        # heading, not a figure line, and the heading must not end with
        # sentence-final punctuation.
        if (is_heading_now
                and next_line
                and not is_heading_next
                and not sentence_final
                and not _is_figure_line(next_line)
                and len(next_line) <= 60):
            # Emit the merged heading directly into fused — do NOT put it
            # back into lines[], which would cause Rule C to fire again on
            # the already-merged result and eat the following body line.
            fused.append(stripped + " " + next_line)
            i = j + 1   # skip past the continuation line
            continue

        if (not sentence_final
                and not is_heading_now
                and not is_heading_next
                and next_line
                and next_line[0].islower()
                and j == i + 1):   # only join if truly adjacent (no blank gap)
            # Join: replace the next line with merged content
            lines[j] = stripped + " " + next_line
            i += 1
            continue

        fused.append(stripped)
        i += 1

    # ── Phase 4: rebuild paragraphs ──────────────────────────────────────────
    # Collapse multiple blank lines; ensure headings get their own paragraph.
    # Skip figure/table caption paragraphs entirely.
    paragraphs: list[str] = []
    current: list[str] = []
    _skip_figure_block = False   # True while inside a caption block

    for line in fused:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            _skip_figure_block = False   # blank line ends any caption block
        elif _is_figure_line(line) or _is_figure_callout(line):
            # Flush whatever we had, then enter skip mode.
            # Covers: "▶ Abb. 7.2 …", "Abb. 7.2 …", and numbered call-outs
            # like "6. Enzyme, die sich …" extracted from figure boxes.
            if current:
                paragraphs.append(" ".join(current))
                current = []
            _skip_figure_block = True    # skip this line and continuations
        elif _skip_figure_block:
            # Caption continuation lines: short, often no sentence-final punct.
            # Keep skipping until a blank line or heading resets state.
            pass
        elif _HEADING_RE.match(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            _skip_figure_block = False
            paragraphs.append(line)      # heading as its own paragraph
        else:
            _skip_figure_block = False
            current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(p for p in paragraphs if p.strip())



# ── German text normalisation for TTS ────────────────────────────────────────
#
# The single biggest intelligibility improvement for academic German TTS.
# The model has no built-in abbreviation expander; it reads "z.B." as
# "z Punkt B Punkt" and numbers like "7,5 mg" as "sieben Komma fünf mg".
# Expanding these to spoken form before synthesis is far more effective
# than any audio post-processing.

# Ordered list of (pattern, replacement) tuples.
# Order matters: longer/more-specific patterns first.
_ABBREV_RULES: list[tuple] = [
    # ── Sentence-safe abbreviations (always expand, even mid-sentence) ───────
    # Each entry: (regex_pattern, spoken_replacement)

    # Common German academic abbreviations
    (r"\bz\.B\.",           "zum Beispiel"),
    (r"\bz\. B\.",          "zum Beispiel"),
    (r"\bd\.h\.",           "das heißt"),
    (r"\bd\. h\.",          "das heißt"),
    (r"\bu\.a\.",           "unter anderem"),
    (r"\bu\. a\.",          "unter anderem"),
    (r"\busw\.",            "und so weiter"),
    (r"\bbzw\.",            "beziehungsweise"),
    (r"\betc\.",            "et cetera"),
    (r"\bggf\.",            "gegebenenfalls"),
    (r"\bsog\.",            "sogenannte"),
    (r"\bvgl\.",            "vergleiche"),
    (r"\bebd\.",            "ebenda"),
    (r"\bca\.",             "circa"),
    (r"\bCa\.",             "circa"),
    (r"\binkl\.",           "inklusive"),
    (r"\bexkl\.",           "exklusive"),
    (r"\bmax\.",            "maximal"),
    (r"\bmin\.",            "minimal"),
    (r"\babs\.",            "absolut"),
    (r"\brel\.",            "relativ"),
    (r"\bev\.",             "eventuell"),
    (r"\bevtl\.",           "eventuell"),
    (r"\bsog\.",            "sogenannte"),
    (r"\bHrsg\.",           "Herausgeber"),
    (r"\bHrsg\.\s+v\.",     "herausgegeben von"),
    (r"\bAufl\.",           "Auflage"),
    (r"\bBd\.",             "Band"),
    (r"\bNr\.",             "Nummer"),
    (r"\bS\.\s*(?=\d)",     "Seite "),     # "S. 47" → "Seite 47"
    (r"\bAbb\.\s*(?=\d)",   "Abbildung "), # "Abb. 7" → "Abbildung 7"
    (r"\bTab\.\s*(?=\d)",   "Tabelle "),   # "Tab. 3" → "Tabelle 3"
    (r"\bKap\.\s*(?=\d)",   "Kapitel "),
    (r"\bAbs\.\s*(?=\d)",   "Absatz "),
    (r"\bArt\.\s*(?=\d)",   "Artikel "),
    (r"\bDr\.",             "Doktor"),
    (r"\bProf\.",           "Professor"),
    (r"\bDipl\.",           "Diplom"),

    # Units — expand when preceded by a digit
    (r"(?<=\d)\s*mg\b",     " Milligramm"),
    (r"(?<=\d)\s*µg\b",     " Mikrogramm"),
    (r"(?<=\d)\s*ng\b",     " Nanogramm"),
    (r"(?<=\d)\s*ml\b",     " Milliliter"),
    (r"(?<=\d)\s*µl\b",     " Mikroliter"),
    (r"(?<=\d)\s*l\b",      " Liter"),
    (r"(?<=\d)\s*kg\b",     " Kilogramm"),
    (r"(?<=\d)\s*g\b",      " Gramm"),
    (r"(?<=\d)\s*km\b",     " Kilometer"),
    (r"(?<=\d)\s*m\b",      " Meter"),
    (r"(?<=\d)\s*cm\b",     " Zentimeter"),
    (r"(?<=\d)\s*mm\b",     " Millimeter"),
    (r"(?<=\d)\s*Hz\b",     " Hertz"),
    (r"(?<=\d)\s*kHz\b",    " Kilohertz"),
    (r"(?<=\d)\s*MHz\b",    " Megahertz"),
    (r"(?<=\d)\s*ms\b",     " Millisekunden"),
    (r"(?<=\d)\s*s\b",      " Sekunden"),
    (r"(?<=\d)\s*min\b",    " Minuten"),
    (r"(?<=\d)\s*h\b",      " Stunden"),
    (r"(?<=\d)\s*%",        " Prozent"),
    (r"(?<=\d)\s*°C\b",     " Grad Celsius"),
    (r"(?<=\d)\s*°\b",      " Grad"),
    (r"(?<=\d)\s*µm\b",     " Mikrometer"),
    (r"(?<=\d)\s*nm\b",     " Nanometer"),
    (r"(?<=\d)\s*mV\b",     " Millivolt"),
    (r"(?<=\d)\s*µV\b",     " Mikrovolt"),
    (r"(?<=\d)\s*nA\b",     " Nanoampere"),

    # German decimal comma: "7,5" → "sieben Komma fünf"
    # We only expand single-decimal cases to avoid over-engineering;
    # keep the numeric form for longer numbers (model handles digits fine).
    # Actually, leave decimal numbers as-is — the model reads German decimals
    # with comma reasonably well.  Only fix the "," → " Komma " when it's
    # clearly a decimal (digit,digit) to avoid eating list commas.
    # Disabled: too many edge cases.  Left as comment for reference.
    # (r"(\d),(\d)", r"\1 Komma \2"),

    # Parenthetical references: strip them to reduce model confusion
    # e.g. "(▶ Abb. 7.2, Schritt 5)" → ""   — already handled upstream,
    # but catch any that slipped through
    # Note: we expand Abb./Tab. above so these are now safe to leave in.
]

# Compile all patterns once at module load
_ABBREV_COMPILED: list[tuple] = [
    (re.compile(pat), repl) for pat, repl in _ABBREV_RULES
]


def expand_abbreviations(text: str) -> str:
    """
    Expand German abbreviations and units to their spoken forms.
    Applied per-chunk just before TTS synthesis.
    """
    for pattern, replacement in _ABBREV_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def ensure_terminal_punctuation(text: str) -> str:
    """
    Append a full stop if the chunk has no sentence-final punctuation.
    This tells the TTS model to close its prosody arc (falling intonation)
    rather than leaving it open, which causes the 'rushed/swallowed ending'.
    """
    t = text.rstrip()
    if t and t[-1] not in ".!?:;\"'»":
        t += "."
    return t


# ── Text chunking — chapter-aware ────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    """
    Split a paragraph into sentences.
    Avoids splitting on common German abbreviations (z.B., d.h., bzw., etc.)
    """
    # Protect common abbreviations by replacing their dots temporarily
    abbrevs = ["z.B", "d.h", "bzw", "etc", "usw", "sog", "ggf", "ca", "Dr", "Prof",
               "Abb", "Tab", "vgl", "ebd", "Hrsg", "Nr", "St", "vs", "Inc", "Ltd"]
    protected = text
    for abbr in abbrevs:
        protected = protected.replace(abbr + ".", abbr + "##DOT##")

    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„»])", protected)
    return [p.replace("##DOT##", ".").strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Optional[str]]:
    """
    Split clean text into TTS render units.

    Chunking strategy (in priority order):
      1. Chapter/section headings  → long pause before + heading spoken alone
      2. Paragraph boundaries      → medium pause
      3. Sentence boundaries       → chunk boundary if paragraph too long
      4. Comma boundaries          → last resort for run-on sentences

    Returns a list where:
      None  → insert a silence gap (duration depends on context)
      str   → synthesise this text
    """
    PARA_PAUSE   = None          # signals paragraph-level silence
    HEADING_PAUSE = "§HEADING§"  # signals heading-level (longer) silence

    paragraphs = re.split(r"\n\n+", text)
    result: list[Optional[str]] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # ── Heading paragraph ────────────────────────────────────────────────
        if _HEADING_RE.match(para):
            result.append(HEADING_PAUSE)
            result.append(para)
            result.append(PARA_PAUSE)
            continue

        # ── Normal paragraph: split into sentences, then bin into chunks ─────
        sentences = _sentence_split(para)
        current = ""

        for sent in sentences:
            if not sent:
                continue
            candidate = (current + " " + sent).strip() if current else sent

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                # Sentence itself too long → split at commas
                if len(sent) > max_chars:
                    sub_parts = re.split(r"(?<=,)\s+", sent)
                    bucket = ""
                    for part in sub_parts:
                        trial = (bucket + " " + part).strip() if bucket else part
                        if len(trial) <= max_chars:
                            bucket = trial
                        else:
                            if bucket:
                                result.append(bucket)
                            bucket = part
                    if bucket:
                        result.append(bucket)
                    current = ""
                else:
                    current = sent

        if current:
            result.append(current)

        result.append(PARA_PAUSE)

    # Remove trailing pauses
    while result and result[-1] in (PARA_PAUSE, HEADING_PAUSE):
        result.pop()

    return result


# ── Audio helpers ─────────────────────────────────────────────────────────────

def silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Return an array of zeros (silence) of the given duration."""
    return np.zeros(int(sample_rate * duration_ms / 1000.0), dtype=np.float32)


def _apply_fade(pcm: np.ndarray, sample_rate: int, fade_ms: int) -> np.ndarray:
    """Apply a linear fade-in and fade-out to avoid clicks at chunk boundaries."""
    fade_samples = int(sample_rate * fade_ms / 1000)
    if len(pcm) < fade_samples * 2:
        return pcm
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    pcm = pcm.copy()
    pcm[:fade_samples]  *= ramp          # fade-in
    pcm[-fade_samples:] *= ramp[::-1]    # fade-out
    return pcm


def _normalise(pcm: np.ndarray, headroom: float) -> np.ndarray:
    """Peak-normalise so the loudest sample sits at `headroom` (0–1)."""
    peak = np.max(np.abs(pcm))
    if peak > 1e-6:
        pcm = pcm * (headroom / peak)
    return pcm


def _post_process(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Shared post-processing: normalise peak level + short fade at boundaries."""
    pcm = _normalise(pcm, PEAK_HEADROOM)
    pcm = _apply_fade(pcm, sample_rate, FADE_MS)
    return pcm


def _prepare_text(text: str) -> str:
    """Shared text prep: expand abbreviations + ensure terminal punctuation."""
    text = expand_abbreviations(text)
    text = ensure_terminal_punctuation(text)
    return text


def synth_chunk_piper(
    voice: "PiperVoice",
    text: str,
    length_scale: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Synthesise one chunk using the Piper backend."""
    from piper.config import SynthesisConfig
    syn_config = SynthesisConfig(length_scale=length_scale)
    text = _prepare_text(text)
    arrays: list[np.ndarray] = []
    sample_rate: int = 22050
    for chunk in voice.synthesize(text, syn_config=syn_config):
        sample_rate = chunk.sample_rate
        arrays.append(chunk.audio_float_array)
    pcm = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    return _post_process(pcm, sample_rate), sample_rate


def synth_chunk_coqui(
    tts: "CoquiTTS",
    text: str,
    speed: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Synthesise one chunk using the Coqui TTS backend (DDC / VITS).

    CoquiTTS.tts() returns a Python list of float32 samples at the model's
    native sample rate (22050 Hz for Thorsten DDC/VITS).
    Speed control: Coqui does not expose length_scale directly for Tacotron2;
    we resample the output instead — simple and side-effect-free.
    """
    text = _prepare_text(text)
    wav_list = tts.tts(text=text)
    pcm = np.array(wav_list, dtype=np.float32)
    sample_rate: int = tts.synthesizer.output_sample_rate

    # Speed control via resampling (only when not 1.0, costs negligible CPU)
    if abs(speed - 1.0) > 0.02:
        target_len = int(len(pcm) / speed)
        indices = np.linspace(0, len(pcm) - 1, target_len)
        pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

    return _post_process(pcm, sample_rate), sample_rate


# Keep a unified entry point used by convert()
def synth_chunk(
    voice,       # PiperVoice | CoquiTTS
    text: str,
    is_coqui: bool = False,
    speed: float = 1.0,
    length_scale: float = 1.0,
) -> Tuple[np.ndarray, int]:
    if is_coqui:
        return synth_chunk_coqui(voice, text, speed=speed)
    else:
        return synth_chunk_piper(voice, text, length_scale=length_scale)


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
    clean     = clean_and_fuse_lines(raw_text)
    max_chars = MAX_CHUNK_CHARS_DDC if _is_coqui_model(model_name) else MAX_CHUNK_CHARS_PIPER
    chunks    = chunk_text(clean, max_chars=max_chars)

    n_chunks = sum(1 for c in chunks if c is not None and c != "§HEADING§")
    bar("✂️ ", f"Chunked into {n_chunks} segments  ({len(clean):,} chars)")

    # 2. Load model — auto-select Piper or Coqui backend
    use_coqui    = _is_coqui_model(model_name)
    # length_scale for Piper: >1 = slower  (= 1/speed)
    length_scale = 1.0 / max(0.5, min(speed, 2.0))

    if use_coqui:
        if not _coqui_available:
            print("\n❌  Coqui TTS not installed. Run:")
            print("    uv sync --extra coqui\n")
            sys.exit(1)
        bar("🧠", f"Loading Coqui model  {model_name}")
        print("   (first run downloads ~950 MB: Tacotron2-DDC model + WaveGrad vocoder")
        print("    cached to ~/.local/share/tts/ — only happens once)")
        # Device selection for Coqui/PyTorch:
        #   MPS  = Apple Silicon GPU (fastest on M-series)
        #   CUDA = Nvidia GPU
        #   CPU  = fallback
        # CoquiTTS only accepts a boolean gpu= flag (True = CUDA).
        # For MPS we set gpu=False but move the model to MPS manually after load.
        import torch
        if torch.backends.mps.is_available():
            bar("⚡", "MPS (Apple Silicon GPU) detected — will move model to MPS after load")
            _use_mps = True
        else:
            _use_mps = False

        voice = CoquiTTS(model_name=model_name, progress_bar=True, gpu=False)

        if _use_mps:
            try:
                # Move the synthesiser's TTS model and vocoder to MPS
                if hasattr(voice, "synthesizer") and voice.synthesizer is not None:
                    if hasattr(voice.synthesizer, "tts_model") and voice.synthesizer.tts_model is not None:
                        voice.synthesizer.tts_model = voice.synthesizer.tts_model.to("mps")
                    if hasattr(voice.synthesizer, "vocoder_model") and voice.synthesizer.vocoder_model is not None:
                        voice.synthesizer.vocoder_model = voice.synthesizer.vocoder_model.to("mps")
                bar("⚡", "Model moved to MPS — GPU inference active")
            except Exception as e:
                bar("⚠️ ", f"MPS move failed ({e}) — falling back to CPU")
    else:
        if not _piper_available:
            print("\n❌  Piper TTS not installed. Run:")
            print("    uv add piper-tts\n")
            sys.exit(1)
        onnx_path   = models_dir / f"{model_name}.onnx"
        config_path = models_dir / f"{model_name}.onnx.json"
        if not onnx_path.exists():
            print(f"\n❌  Piper model not found: {onnx_path}")
            print(    "   Run:  python download_model.py\n")
            sys.exit(1)
        bar("🧠", f"Loading Piper model  {model_name}")
        voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))

    # 3. Synthesise — streaming directly to disk
    #
    # Previous approach accumulated all chunks in a list[np.ndarray] and then
    # concatenated them at the end — keeping the full audio (200-300 MB for a
    # long document) in RAM alongside the loaded model (~500 MB).
    # Now we open a SoundFile in write mode upfront and flush each chunk to
    # disk immediately after synthesis, then del the array. Peak RAM stays
    # flat at: model weights + one chunk at a time.
    bar("\U0001f399\ufe0f ", f"Synthesising \u2026  ({'Coqui DDC' if use_coqui else 'Piper'})")

    sample_rate: Optional[int] = None
    done          = 0
    total_samples = 0
    t_synth_start = time.time()
    writer        = None   # sf.SoundFile, opened on first real audio chunk

    # We defer silence between chunks: write it BEFORE the next audio chunk,
    # not after — this prevents a trailing silence gap at end of file.
    pending_silence_ms: int = 0

    def _open_writer(sr: int) -> "sf.SoundFile":
        fmt = output_path.suffix.lstrip(".").upper()
        if fmt not in ("WAV", "FLAC", "OGG"):
            fmt = "WAV"
        return sf.SoundFile(str(output_path), mode="w",
                            samplerate=sr, channels=1, format=fmt)

    for chunk in chunks:
        # ── Silence sentinels ────────────────────────────────────────────────
        if chunk is None:
            pending_silence_ms += SILENCE_PARAGRAPH_MS
            continue
        if chunk == "\u00a7HEADING\u00a7":
            pending_silence_ms += SILENCE_HEADING_MS
            continue

        # ── Synthesise one text chunk ─────────────────────────────────────────
        audio, sr = synth_chunk(voice, chunk,
                                is_coqui=use_coqui,
                                speed=speed,
                                length_scale=length_scale)

        # Open the file on first chunk (now we know sample_rate)
        if writer is None:
            sample_rate = sr
            writer = _open_writer(sr)

        # Write pending silence, then the audio, then queue inter-sentence gap
        if pending_silence_ms > 0:
            sil = silence(pending_silence_ms, sr)
            writer.write(sil)
            total_samples += len(sil)
            pending_silence_ms = 0

        writer.write(audio)
        total_samples += len(audio)
        pending_silence_ms = SILENCE_SENTENCE_MS  # deferred until next chunk

        del audio   # release immediately — GC can reclaim RAM now

        done += 1
        pct     = int(done / n_chunks * 40)
        bar_str = "\u2588" * pct + "\u2591" * (40 - pct)
        print(f"\r   [{bar_str}] {done}/{n_chunks}", end="", flush=True)

    print()
    synth_time = time.time() - t_synth_start

    # 4. Close the file
    bar("\U0001f4be", f"Finalising  {output_path.name}")
    if writer is None:
        print("\u26a0\ufe0f  No audio was written — all chunks were empty.")
        return
    writer.close()

    duration_s = total_samples / sample_rate
    ratio      = duration_s / synth_time if synth_time > 0 else 0
    bar("\u2705", f"Done!  duration={fmt_time(duration_s)}  "
            f"synth={synth_time:.1f}s  ({ratio:.0f}\u00d7 real-time)")

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
  python pdf_to_audio.py buch.pdf
  python pdf_to_audio.py buch.pdf --model tts_models/de/thorsten/tacotron2-DDC
  python pdf_to_audio.py *.pdf --output-dir ./hoerbuecher
  python pdf_to_audio.py doc.pdf --speed 0.92
        """,
    )
    p.add_argument("pdfs", nargs="+", type=Path, metavar="PDF",
                   help="Input PDF file(s)")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                   help="Output directory (default: same folder as each PDF)")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   help=(
                       f"Model name (default: {DEFAULT_MODEL}). "
                       "Piper: de_DE-thorsten-high | de_DE-thorsten_emotional-medium. "
                       f"Coqui DDC (clearer): {DDC_MODEL}. "
                       f"Coqui VITS: {VITS_MODEL}"
                   ))
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                   help=f"Folder containing .onnx model files (default: {MODELS_DIR})")
    p.add_argument("--style", choices=EMOTIONAL_STYLES, default=None,
                   help="Speaking style for thorsten_emotional model")
    p.add_argument("--speed", type=float, default=0.95, metavar="RATE",
                   help="Speaking rate: 1.0=normal, 0.95=slightly slower for clarity (default: 0.95)")
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