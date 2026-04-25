# Changelog

## 0.4.8 - 2026-04-25

### Fixes (Windows)
- **SBV reader**: normalize CRLF → LF before splitting cue blocks. Previously the `\n\n` blank-line delimiter never matched on Windows-authored files (`write_text` writes CRLF), so the entire file collapsed into a single supervision
- **`Caption.from_string`**: detect dominant `line_terminator` (CRLF / LF) on string input, mirroring what `Caption.read` does for file input. Without this, `read → to_bytes → from_string → to_bytes` produced different bytes on the second pass — first round preserved CRLF from the source file, second round defaulted to LF
- **`.gitattributes`**: force LF for all `tests/data/**` fixtures and source files. Windows checkouts with `core.autocrlf=true` were silently rewriting fixtures to CRLF, propagating through `line_terminator` detection and breaking byte-level assertions

## 0.4.7 - 2026-04-25

### Features
- **ASS byte-level roundtrip fidelity** for subtitle-group files: preserve original Script Info, Styles section, comment lines, and field ordering so `read → write` is bit-identical on hand-authored ASS sources
- **SRT byte-level roundtrip fidelity** for subtitle-group files: preserve original timestamp formatting, blank-line conventions, and trailing whitespace

### Fixes
- ASS reader: tolerate malformed timestamps and colors in hand-edited files instead of failing to parse
- VTT/SRT/JSON readers: convert pysubs2's internal `\N` marker to actual `\n` newlines (was leaking the escape sequence)
- VTT/SRT/JSON readers: emit `\n` instead of a space when an inline `\N` line break is present
- `split_sentences`: preserve per-supervision fields (speaker, language, custom metadata) across sentence splits

### Internal
- Drop redundant `from __future__ import annotations` across the package — `requires-python = ">=3.10"` already provides PEP 604 union syntax and PEP 585 built-in generics

## 0.4.6 - 2026-04-12

### Features
- **WebVTT W3C standard compliance**: cue settings (align/line/position/size/vertical/region), `<v>` voice tags, inline formatting (`<b>/<i>/<u>/<c>/<ruby>/<lang>`), REGION/STYLE/NOTE blocks, `VTTConfig`
- SRT: speaker color support via `<font color="#RRGGBB">` HTML tags with `SRTConfig.speaker_color`
- `RenderConfig.speaker_color`: cross-format speaker coloring fallback (ASS/SRT/VTT)
- Deduplicate consecutive speaker labels in ASS/markdown writer output

### Fixes
- VTT: `_format_timestamp()` boundary carry (59.9996s no longer emits `:60.000`)
- VTT: speaker roundtrip fidelity for `voice_tag` and `speaker_color` modes
- SRT: speaker roundtrip for `speaker_color` output (parse `<font>` wrapped prefixes)
- SRT: disable speaker dedup so every cue retains prefix for roundtrip
- Validate hex digits in speaker color palette (`_resolve_palette` fail-fast)
- Avid DS: fix `SpeakerTracker` import path in `nle/avid.py`

## 0.4.5 - 2026-04-12

### Features
- ASS: add `borderstyle`, `scalex`, `scaley`, `spacing`, `angle`, `underline`, `strikeout` fields to `ASSConfig`

### Fixes
- Kinetic: use `\rKaraoke` to isolate word-scope animations from parent style bleed
- Bump default ASS font size from 48 to 64

## 0.4.1 - 2026-04-07

### BREAKING CHANGES
- **`KaraokeConfig` deleted** — karaoke fields moved into format-specific configs:
  - `ASSConfig.karaoke_effect` / `karaoke_color_scheme` (was `KaraokeConfig.effect` / `color_scheme`)
  - `LRCConfig` new dataclass with `precision` / `metadata` (was `KaraokeConfig.lrc_precision` / `lrc_metadata`)
  - `TTMLConfig.timing_mode` new field (was `KaraokeConfig.ttml_timing_mode`)
- **`OutputBehavior` renamed to `RenderConfig`** — parameter `behavior=` renamed to `render=` across all write APIs
- **Writer APIs require explicit `RenderConfig`** — loose kwargs no longer accepted:
  - `word_level=True` → `render=RenderConfig(word_level=True)`
  - `include_speaker=False` → `render=RenderConfig(include_speaker_in_text=False)`
- **Caption config system redesigned** — `ASSConfig` is now self-contained with PlayRes, wrap_style, and all visual style fields; `FormatConfig` type alias added for `format_config` parameter

### Migration

```python
# Before (0.2.x)
from lattifai.caption.config import KaraokeConfig, OutputBehavior
cap.write(path, karaoke=KaraokeConfig(enabled=True),
          behavior=OutputBehavior(include_speaker_in_text=False))

# After (0.4.0)
from lattifai.caption.config import ASSConfig, RenderConfig
cap.write(path, format_config=ASSConfig(karaoke_effect="sweep"),
          render=RenderConfig(word_level=True, include_speaker_in_text=False))
```

### Features
- `>>` speaker change marker support across all caption formats
- 12 karaoke color presets + 10-color speaker palette
- Speaker color support in ASS output (`auto` / explicit `#RRGGBB` / per-speaker list)
- Background color / opaque box support (`ASSConfig.background_color`, borderstyle=3)
- Standardization: word-boundary splitting, alignment-based timing, oversized segment handling
- JSON v2 document-level output format

### Fixes
- VTT: use `OutlineColour` (not `BackColour`) as CSS background-color for borderstyle=3
- ASS: convert `\n` to `\N` for inline line breaks in output
- Karaoke: gap-aware timing, original-text separators, reorder speaker palette
- Standardize: prevent under-filled segments and orphans in splitting

## 0.2.9 - 2026-03-31

### Fixes
- Patch xlm-roberta special tokens for transformers 5.x compatibility
- Register "md" extension alias for markdown format

## 0.2.8 - 2026-03-31

### Features
- Bilingual subtitle support with translation fields
- Auto-detect title-case speaker names in subtitle parsing
- Markdown format: bilingual read/write, YAML frontmatter, dialogue segments, time range support
- `translation_first` option to swap bilingual text order
- Group utterances by speaker in TextGrid format

### Refactor
- Rename gemini format to markdown with backward compatibility
- Remove podcast_transcript format (superseded by markdown)

### Fixes
- Guard against negative duration from overlapping VTT cues
- Preserve space after CJK colon in sentence splitter
- Split inline event markers from mixed text
- Filter unknown fields in `Supervision.from_dict()` for forward-compatibility

## 0.2.7 - 2026-02-28

### Fixes
- Split multi-event text like `[Laughter] [Applause]` into separate events in both Gemini parser and sentence splitter

## 0.2.6 - 2026-02-27

### Fixes
- Default outline width to 0 (no outline) instead of 2
- Remove unused imports in config module

## 0.2.5 - 2026-02-27

### Fixes
- Respect metadata `ass_styles` for karaoke style instead of overwriting with defaults
- Change default ASS font size from 128 to 20 (matching Aegisub/ASS convention)

## 0.2.4 - 2026-02-27

### Features
- VTT STYLE block generation from ASS style metadata (`::cue` with font, color, background, outline)

### Fixes
- Round start/end times in JSON output, include score in word alignments

## 0.2.2 - 2026-02-26

### Fixes
- Always include speaker field in JSON output
- Handle "auto" format in `Caption.read()` to trigger auto-detection
- Skip space insertion after CJK colon in sentence splitting

## 0.1.8 - 2026-02-05

### Fixes
- Rename `_fastcopy` to `fastcopy` and update references

## 0.1.7 - 2026-02-04

### Fixes
- Accept extra kwargs in Gemini `write_aligned_transcript`

## 0.1.5 - 2026-01-21

### Features
- TextGrid: separate `[event]` into dedicated tiers with overlap handling
- Supervision: add dict-style and attribute access to custom fields
- Gemini: support `[START] text [END]` timestamp format and millisecond timestamps
- Gemini: skip YAML front matter and `<thinking>` blocks
- Gemini: filter thinking/meta blocks in SRT/VTT parsing
- ASS: PlayResX/PlayResY support in metadata

### Refactor
- Remove `SupervisionSegment` alias, use `Supervision` everywhere
- Remove `CaptionConfig` (moved to lattifai-python)
- Remove transcription/alignment/diarization fields from Caption
- Simplify sentence_splitter.py

### Fixes
- Preserve event supervisions as separate segments in sentence splitting
- Infer start time from previous segment for end-only timestamps
- Register JSON format and fix karaoke test assertion
- Auto-convert numpy types in custom fields
