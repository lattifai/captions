# Changelog

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
