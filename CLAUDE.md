# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`lattifai-captions` is a Python library for caption/subtitle processing with comprehensive format support. It's part of the LattifAI ecosystem and serves as the caption I/O layer for the alignment and transcription pipelines.

## Common Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/caption/test_formats.py

# Run specific test
pytest tests/caption/test_formats.py::test_name -v

# Run with coverage
pytest --cov=src tests/
```

## Architecture

### Core Data Model

```
Caption (caption.py)
├── supervisions: List[Supervision]  # Main caption segments
├── transcription: List[Supervision] # ASR results (if from transcription)
├── alignments: List[Supervision]    # Post-alignment results
├── audio_events: TextGrid           # Audio event detection
└── speaker_diarization              # Speaker diarization output
```

**Supervision** (`supervision.py`) extends Lhotse's `SupervisionSegment`:
- Core fields: `text`, `start`, `duration`, `speaker`, `id`
- Word-level alignment stored in `alignment["word"]` as `List[AlignmentItem]`

### Format Registry System

Located in `formats/__init__.py`. Uses decorator-based registration:

```python
@register_format("srt")      # Both read/write
@register_reader("gemini")   # Read-only
@register_writer("avid_ds")  # Write-only
```

**Base classes** (`formats/base.py`):
- `FormatReader` - Must implement `read()`, `can_read()`, `extract_metadata()`
- `FormatWriter` - Must implement `write()`, `to_bytes()`
- `FormatHandler` - Combined reader/writer

**Format categories**:
| Type | Formats | Notes |
|------|---------|-------|
| Standard | srt, vtt, ass, ssa, sub, sbv | Via pysubs2 |
| Tabular | csv, tsv, aud, json | Custom parsers |
| Specialized | textgrid, gemini, lrc, ttml, srv3 | Format-specific |
| NLE (write-only) | avid_ds, fcpxml, premiere_xml, audition_csv | Professional video |

### Key Modules

- **config.py**: `InputCaptionFormat`, `OutputCaptionFormat` Literal types; `CaptionConfig`, `KaraokeConfig`, `StandardizationConfig`
- **standardize.py**: Netflix/BBC broadcast compliance; `CaptionStandardizer`, `CaptionValidator`
- **sentence_splitter.py**: Uses wtpsplit for intelligent sentence segmentation (optional dependency)
- **utils.py**: Timecode operations, overlap detection/resolution, SRT generation helpers

### Data Flow

**Reading**: `Caption.read(path)` → `detect_format()` → `get_reader(fmt).read()` → `Caption`

**Writing**: `caption.write(path)` → `get_writer(fmt).write(supervisions)` → file

**Alignment priority** (for output): `alignments` > `supervisions` > `transcription`

## Dependencies

- **lhotse**: Core data structures (`SupervisionSegment`, `AlignmentItem`)
- **pysubs2**: SRT/ASS/SSA/SUB format handling
- **praatio** + **tgt**: Praat TextGrid support
- **wtpsplit** (optional): Sentence splitting with `[splitting]` extra

## Testing

Tests are in `tests/caption/`. Key test files:
- `test_formats.py` - Format roundtrip tests
- `test_standardize.py` - Broadcast compliance
- `test_*_format.py` - Individual format tests

Test data in `tests/data/captions/`.
