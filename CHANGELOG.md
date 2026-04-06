# Changelog

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
