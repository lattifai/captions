# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`lattifai-captions` is a Python library for caption/subtitle processing with 25+ format support. It serves as the caption I/O layer for the LattifAI alignment and transcription pipelines. Pure Python — no PyTorch/TensorFlow dependency.

## Common Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file / single test
pytest tests/caption/test_formats.py
pytest tests/caption/test_formats.py::test_name -v

# Run with coverage
pytest --cov=src tests/

# Code formatting (pre-commit runs these automatically)
black src/ tests/ --line-length=120
isort src/ tests/ --profile=black --line-length=120
flake8 src/ tests/
```

## Code Style

- **Line length**: 120 characters (black + flake8)
- **Import sorting**: isort with black profile
- **Pre-commit hooks**: black, flake8, isort, trailing whitespace, end-of-file fixer
- `formats/__init__.py` allows `E401, F401` (wildcard imports for re-exports)
- `tests/` has relaxed linting (`E501, F541, F401, F841` ignored)

## Multilingual Text Convention

**Canonical source of truth is the original text string, never reconstructed
from word symbols.** This repo stores user-facing text on ``Supervision.text``
(verbatim, exactly as the user provided it) and stores per-word timing
separately on ``Supervision.alignment["word"]`` as ``AlignmentItem``s. Any
pipeline that needs the written text must read ``sup.text`` — it must not
rebuild it by joining ``w.symbol`` values.

Rationale (pulled from how the core tokenizer and backend handle this):

- ``lattifai_core.tokenization.tokenize_multilingual_text`` splits CJK
  (``\u4e00-\u9fff``) **per character**, Latin letter sequences per word,
  and keeps punctuation/spaces as separate tokens. The inverse operation
  (detokenize) is deliberately ``raise NotImplementedError`` in
  ``LatticeTokenizer.detokenize`` — reconstructing text from tokens is
  considered ambiguous and the codebase refuses to do it.
- ``lattifai-backend/app/api/v1/alignment.py:finalize_alignments`` builds
  supervisions with ``text=transcript`` (the exact client-supplied text) —
  word alignment lives on ``supervision.alignment["word"]`` only. Clients
  rely on the text being untouched for display and roundtrip.
- The only places in ``lattifai_core`` that do ``" ".join(w.symbol ...)``
  (``tokenization/overlap.py``, ``diarization/stages/a2b_resolve.py``) feed
  the result *back into the aligner*, whose tokenizer then re-splits CJK
  per character. So those paths are internal pipeline stages, not
  user-facing output.

**Rule for this repo (lattifai-captions):**

- When you split a supervision that has word alignment, slice the
  **original** ``sup.text`` at the char positions that correspond to the
  first/last word of each group — do not ``" ".join(w.symbol)``. Joining
  produces ``"我 觉 得"`` for Chinese, breaks ``"Terry Tao"``'s expected
  spacing in mixed-language content, and destroys user-authored whitespace
  and punctuation.
- Mapping words → char positions: walk ``sup.text`` and ``words`` in order,
  using ``text.find(word.symbol, cursor)`` to advance a cursor. Word order
  in the alignment matches text order; this is O(n).
- If you legitimately need a roundtrippable text from a word list with no
  original text (rare — only when the list is synthesized from scratch),
  call ``tokenize_multilingual_text`` to re-tokenize and concatenate: CJK
  chars concatenate directly, Latin/punctuation tokens keep their
  separators. Never hard-code ``" ".join``.
- Tests covering multilingual text must include at least one pure-CJK case
  and one CJK+Latin mixed case.

## Timing Precision Convention

All caption/subtitle times (seconds) are rounded to **4 decimal places** (100μs
resolution) whenever they are written out or stored after arithmetic. Rationale:

- 4 decimals is finer than every caption format we emit — SRT/VTT are
  millisecond (3dp), ASS is centisecond (2dp), word-level alignment in JSON is
  also 4dp — so we never lose resolution on serialization.
- 4 decimals (1e-4) safely side-steps IEEE-754 float drift from arithmetic
  chains like ``start + (ratio * duration) + gap`` which would otherwise turn
  a perfect 0.08s gap into 0.07999999…s and fail validator checks.
- Fewer decimals (e.g., 3dp) are NOT enough: a gap of exactly 0.08 rounded to
  3dp still surfaces drift when the two operands don't share the same rounding
  error. Use 4dp everywhere to keep arithmetic idempotent.

**Rule:** any time you compute a new ``start``, ``end``, ``duration``, or
``gap`` inside ``standardize.py`` / ``utils.py`` / format writers, wrap it in
``round(value, 4)``. Tests asserting on split timings should use
``abs(a - b) < 1e-3`` tolerance rather than exact equality.

## Publishing

Two channels — both triggered by `v*.*.*` tag push or `workflow_dispatch`:

| Workflow | Target | Notes |
|----------|--------|-------|
| `publish-wheels.yml` | PyPI (Trusted Publishing) | sdist only, OIDC auth |
| `publish-to-pages.yml` | GitHub Pages PyPI | sdist + wheel, SSH deploy key to `lattifai/pypi` repo |

Install from GitHub Pages: `pip install lattifai-captions --extra-index-url https://lattifai.github.io/pypi/simple/`

## Architecture

### Core Data Model

```
Caption (caption.py)
├── supervisions: List[Supervision]  # Caption segments with timing + text
├── language: Optional[str]          # Source language code
├── target_lang: Optional[str]       # Translation target language
├── kind: Optional[str]              # 'captions' / 'subtitles' / 'descriptions'
├── source_format: Optional[str]     # Original format (e.g., 'vtt', 'ass')
├── source_path: Optional[str]       # Path to source file
└── metadata: Dict[str, Any]         # Format-specific metadata (ass_info, ass_styles, etc.)
```

**Supervision** (`supervision.py`) — local copy of Lhotse's `SupervisionSegment` to avoid heavy dependencies:
- Core fields: `text`, `start`, `duration`, `speaker`, `id`
- Word-level alignment: `alignment["word"]` → `List[AlignmentItem(symbol, start, duration, score)]`
- Utility: `fastcopy()` for efficient dataclass copies, `_add_durations()` for float-safe time arithmetic (48kHz sampling)

### Format Registry System

Decorator-based registration in `formats/__init__.py`:

```python
@register_format("srt")      # Both read/write
@register_reader("gemini")   # Read-only
@register_writer("avid_ds")  # Write-only
```

**Base classes** (`formats/base.py`):
- `FormatReader` — implement `read()`, `can_read()`, `extract_metadata()`
- `FormatWriter` — implement `write()`, `to_bytes()`
- `FormatHandler` — combined reader/writer

**Format categories**:
| Type | Formats | Implementation |
|------|---------|----------------|
| Standard | srt, vtt, ass, ssa, sub, sbv | pysubs2-based |
| Tabular | csv, tsv, aud, json | Custom parsers |
| Specialized | textgrid, markdown, lrc, srv3 | Format-specific |
| NLE (write-only) | avid_ds, fcpxml, premiere_xml, audition_csv | Professional video editors |
| Broadcast (write-only) | ttml, imsc1, ebu_tt_d | Streaming/broadcast delivery |

**Data flow**:
- Read: `Caption.read(path)` → `detect_format()` → `get_reader(fmt).read()` → `Caption`
- Write: `caption.write(path)` → `get_writer(fmt).write(supervisions)` → file

### Config System (v0.4.0)

```
RenderConfig          — Cross-format: include_speaker, word_level, translation_first
ASSConfig             — ASS-specific: font, colors, margins, karaoke_effect, speaker_color
LRCConfig             — LRC-specific: precision, metadata
TTMLConfig            — TTML-specific: profile, region, timing_mode
StandardizationConfig — Broadcast compliance: duration, CPS, line limits
```

`FormatConfig` type alias = `Union[ASSConfig, LRCConfig, TTMLConfig, ...]`

### Key Modules

- **config.py**: Format type literals; `RenderConfig`, `ASSConfig`, `LRCConfig`, `StandardizationConfig`
- **colors.py**: 12 karaoke color schemes, 10-color speaker palette, `resolve_speaker_color()`
- **standardize.py**: Netflix/BBC broadcast compliance; `CaptionStandardizer`, `CaptionValidator`
- **sentence_splitter.py**: wtpsplit integration for sentence segmentation (optional `[splitting]` extra)
- **utils.py**: Timecode operations, `resolve_overlaps(CollisionMode.TRIM|REMOVE|EXTEND)`, SRT generation

## Dependencies

Only 3 core dependencies:
- **pysubs2**: SRT/ASS/SSA/SUB format handling
- **praatio** + **tgt**: Praat TextGrid support

Optional `[splitting]` extra adds: wtpsplit, onnxruntime, huggingface_hub, modelscope

## Testing

CI matrix: Ubuntu/macOS/Windows × Python 3.10–3.13 (12 combinations).

Test data in `tests/data/` and `tests/data/captions/`. Key test files:
- `test_formats.py` — Format roundtrip tests
- `test_ass_format.py` — ASS read/write, karaoke, styles
- `test_karaoke.py` — Karaoke effects and color schemes
- `test_speaker_color.py` — Speaker color in ASS output
- `test_speaker_roundtrip.py` — Speaker label roundtrip across formats
- `test_standardize.py` — Broadcast compliance
- `test_markdown.py` — Markdown/bilingual format
- `test_srv3_format.py` — YouTube SRV3 word-level timing
- `test_professional_formats.py` — NLE export (FCPXML/Premiere/Avid)
- `test_lrc_format.py` — LRC lyric format with LRCConfig
- `test_ttml_reader.py` — TTML/IMSC1/EBU-TT-D profiles
