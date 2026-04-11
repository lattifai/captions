# lattifai-captions

**The universal caption toolkit.** Read, write, convert, and validate 25+ subtitle formats with word-level precision — from YouTube to Netflix to Final Cut Pro.

[![PyPI](https://img.shields.io/pypi/v/lattifai-captions)](https://pypi.org/project/lattifai-captions/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## Why lattifai-captions?

| | |
|---|---|
| **25+ formats** | SRT, VTT, ASS, TTML, SRV3, TextGrid, FCPXML, Premiere XML, and more |
| **Word-level timing** | Preserve millisecond-precision word timestamps across format conversions |
| **Broadcast-ready** | Netflix, BBC, and EBU compliance validation out of the box |
| **NLE integration** | Direct export to Avid, Final Cut Pro, Premiere Pro, and Pro Tools |
| **Lightweight** | Zero dependency on PyTorch, TensorFlow, or any ML framework |

## Installation

```bash
pip install lattifai-captions --extra-index-url https://lattifai.github.io/pypi/simple/

# With AI-powered sentence splitting
pip install lattifai-captions[splitting] --extra-index-url https://lattifai.github.io/pypi/simple/
```

Or configure pip globally:

```ini
# ~/.pip/pip.conf
[global]
extra-index-url = https://lattifai.github.io/pypi/simple/
```

## Quick Start

```python
from lattifai.caption import Caption

# Read any format — auto-detected
caption = Caption.read("input.srt")

# Convert to any format
caption.write("output.vtt")
caption.write("output.ass")
caption.write("output.ttml")

# Get as string
vtt_content = caption.to_string("vtt")
```

## Supported Formats

### Read & Write

| Format | Extensions | Highlights |
|--------|------------|------------|
| **SRT** | `.srt` | Industry standard subtitle format |
| **WebVTT** | `.vtt` | Web standard; auto-detects YouTube word-level timestamps |
| **ASS / SSA** | `.ass` `.ssa` | Styled subtitles with karaoke support |
| **SRV3** | `.srv3` `.ytt` | YouTube Timed Text v3 — millisecond word timing |
| **SBV** | `.sbv` | YouTube SubViewer |
| **SUB** | `.sub` | MicroDVD |
| **SAMI** | `.sami` `.smi` | SAMI subtitle format |
| **JSON** | `.json` | Structured data with full word-level arrays |
| **CSV / TSV** | `.csv` `.tsv` | Tabular export for data analysis |
| **TextGrid** | `.textgrid` | Praat — phonetics and linguistics research |
| **LRC** | `.lrc` | Lyrics with word-level timestamps |
| **Gemini** | `.md` | Gemini AI transcript markdown |

### Write-Only — Professional Post-Production

| Format | Target | Use Case |
|--------|--------|----------|
| **TTML** | W3C standard | Streaming platforms |
| **IMSC1** | Netflix / streaming | Netflix Timed Text profile |
| **EBU-TT-D** | European broadcast | EBU broadcast delivery |
| **Avid DS** | Avid Media Composer | SubCap import |
| **FCPXML** | Final Cut Pro | Native timeline import |
| **Premiere XML** | Adobe Premiere Pro | Graphic clip subtitles |
| **Audition CSV** | Adobe Audition | Marker-based editing |
| **EdiMarker CSV** | Pro Tools | Session markers |

## Word-Level Timing

Preserve and convert word-by-word timestamps across formats:

```python
from lattifai.caption import Caption
from lattifai.caption.config import RenderConfig, ASSConfig

caption = Caption.read("video.srv3")  # YouTube SRV3 with word timing

# Inspect word-level data
for seg in caption.supervisions:
    for word in seg.alignment.get("word", []):
        print(f"  {word.symbol}: {word.start:.3f}s ({word.duration:.3f}s)")

# Export with word timing preserved (per-format opt-in)
caption.write("output.srt", render=RenderConfig(word_level=True))   # one cue per word
caption.write("output.lrc", render=RenderConfig(word_level=True))   # enhanced LRC inline
caption.write("output.vtt", render=RenderConfig(word_level=True))   # YouTube VTT inline

# Karaoke-style ASS output (karaoke_effect alone is enough)
caption.write("output.ass", format_config=ASSConfig(karaoke_effect="sweep"))

# JSON / TextGrid keep word data automatically (lossless)
caption.write("output.json")
```

### `RenderConfig.word_level` — Tri-State Semantics

A single `Optional[bool]` controls word-level output across every writer.
**Most users never need to set it** — the default (`None`) is the right
behavior for every format.

| Value | Meaning |
|-------|---------|
| `None` *(default)* | Smart per-format default. Renderers stay segment-level; lossless serializers (JSON/TextGrid) preserve word data when present. |
| `True` | Force word-level output. Renderers emit per-word cues / inline timestamps; if word alignment is missing, a warning is logged and the writer degrades to segment output. |
| `False` | Force segment-level output. Lossless serializers **drop** their word data; ASS karaoke is disabled even when `karaoke_effect` is set. |

#### Per-Format Behavior Matrix

| Format | `None` *(default)* | `True` | `False` |
|--------|---|---|---|
| **SRT / SBV / SUB** | one cue per segment | one cue per **word** (warns if no alignment) | same as None |
| **VTT** | standard segment VTT | YouTube VTT with inline `<00:00:01.000>` markers | same as None |
| **LRC** | standard `[mm:ss.xx]` lines | enhanced LRC with inline `<mm:ss.xx>` per word | same as None |
| **TTML / IMSC1 / EBU-TT-D** | one `<p>` per segment | iTunes namespace + per-word `<span>` | same as None |
| **SRV3** | segment paragraphs | per-word `<s t="">` offsets | same as None |
| **ASS / SSA** | segment Dialogue lines; honors `ASSConfig.karaoke_effect` to emit `{\k}` tags automatically | force word level — with `karaoke_effect` → `{\k}` tags; without → one Dialogue line per word | force segment; **disables `karaoke_effect` with a warning** |
| **JSON** | preserves `words` array when alignment is present (lossless) | same as None | **strips `words` array** even when present |
| **TextGrid** | emits `words` tier when alignment is present (lossless) | same as None | **omits `words` tier** even when present |
| **Premiere XML** | one clip per segment | one clip per word | same as None |
| **FCPXML** | one element per segment | one element per word | same as None |
| **CSV / TSV / TXT / AUD / Avid / Audition / EdiMarker / Markdown** | segment-level only (format has no word concept) | same as None (warning is *not* emitted; format is incapable) | same as None |

#### Key Design Notes

- **ASS karaoke is triggered by `ASSConfig.karaoke_effect` alone** — you no longer
  need to combine it with `word_level=True`. Setting `word_level=False` while
  `karaoke_effect` is set will explicitly disable karaoke (with a warning).
- **Lossless serializers (JSON / TextGrid) ignore `True`** — they always
  preserve word data when it exists. The only state that changes their
  output is `False`, which strips the word data.
- **Degradation is logged, not fatal** — when `word_level=True` is requested
  but no supervision has word alignment, a single `WARNING` is logged via
  the `lattifai.caption` logger and the writer falls back to segment output.
  This matches the "best-effort batch" model used by transcription pipelines.

## Broadcast Standardization

Enforce Netflix, BBC, or custom broadcast guidelines:

```python
from lattifai.caption import Caption, CaptionStandardizer, CaptionValidator

# Standardize
standardizer = CaptionStandardizer(
    min_duration=0.7,       # Minimum segment duration (seconds)
    max_duration=7.0,       # Maximum segment duration
    min_gap=0.08,           # 80ms gap to prevent flicker
    max_lines=2,            # Lines per segment
    max_chars_per_line=42,  # Auto-adjusts to 21 for CJK
)
caption = Caption.read("input.srt")
standardized = standardizer.process(caption.supervisions)

# Validate
validator = CaptionValidator(min_duration=0.7, max_duration=7.0, max_chars_per_line=42)
result = validator.validate(caption.supervisions)
print(f"Valid: {result.valid} | CPS: {result.avg_cps:.1f} | Warnings: {len(result.warnings)}")
```

## NLE Export

Direct export to professional editing software:

```python
from lattifai.caption import Caption, FCPXMLConfig, FCPXMLStyle

caption = Caption.read("input.srt")

# Final Cut Pro
caption.write("timeline.fcpxml")

# Avid Media Composer
caption.write("avid.txt", format="avid_ds")

# Adobe Premiere Pro
caption.write("premiere.xml", format="premiere_xml")
```

## Sentence Splitting

AI-powered sentence segmentation using [wtpsplit](https://github.com/segment-any-text/wtpsplit) (requires `[splitting]` extra):

```python
caption = Caption.read("input.srt")
split_caption = caption.split_sentences()
```

## Time Operations

```python
# Shift all timestamps
shifted = caption.shift_time(seconds=2.5)

# Adjust word-level margins (prevents cut-off words)
adjusted = caption.with_margins(start_margin=0.05, end_margin=0.15)

# Resolve overlapping segments
from lattifai.caption import resolve_overlaps, CollisionMode
resolved = resolve_overlaps(caption.supervisions, mode=CollisionMode.TRIM)
```

## API Reference

### Caption

```python
from lattifai.caption import Caption

# Read
caption = Caption.read("file.srt")                    # Auto-detect format
caption = Caption.read("file.txt", format="srt")      # Explicit format
caption = Caption.from_string(content, format="vtt")   # From string

# Write
caption.write("output.vtt")
caption.write(
    "output.ass",
    format_config=ASSConfig(karaoke_effect="sweep"),  # karaoke_effect alone triggers ASS karaoke
)
content = caption.to_string("srt")

# Transform
caption.shift_time(seconds=1.0)
caption.split_sentences()
caption.with_margins(start_margin=0.05, end_margin=0.15)

# Properties
caption.supervisions    # List[Supervision]
caption.duration        # Total duration in seconds
caption.language        # Language code
caption.source_format   # Original format detected
```

### Supervision

```python
from lattifai.caption import Supervision

sup = Supervision(start=0.0, duration=2.5, text="Hello world", speaker="Alice")

sup.end          # 2.5 (start + duration)
sup.alignment    # {"word": [AlignmentItem(symbol, start, duration, score), ...]}
```

## License

Apache-2.0
