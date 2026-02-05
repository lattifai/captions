# lattifai-captions

**The universal caption toolkit.** Read, write, convert, and validate 25+ subtitle formats with word-level precision — from YouTube to Netflix to Final Cut Pro.

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
from lattifai.caption import Caption, KaraokeConfig

caption = Caption.read("video.srv3")  # YouTube SRV3 with word timing

# Inspect word-level data
for seg in caption.supervisions:
    for word in seg.alignment.get("word", []):
        print(f"  {word.symbol}: {word.start:.3f}s ({word.duration:.3f}s)")

# Export with word timing preserved
caption.write("output.json", word_level=True)
caption.write("output.lrc", word_level=True)

# Karaoke-style ASS output
caption.write("output.ass", word_level=True, karaoke_config=KaraokeConfig(enabled=True))
```

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
caption.write("output.ass", word_level=True, karaoke_config=KaraokeConfig(enabled=True))
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
