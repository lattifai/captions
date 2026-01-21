# lattifai-captions

Caption/subtitle processing library with comprehensive format support.

## Features

- **Multi-format support**: SRT, VTT, ASS, SSA, TTML, TextGrid, LRC, SRV3, and more
- **YouTube formats**: SRV3 (YTT v3), YouTube VTT with word-level timestamps
- **Professional NLE formats**: Avid DS, Final Cut Pro XML, Premiere Pro XML, Adobe Audition
- **Word-level timing**: Karaoke-style word-by-word timestamps
- **Standardization**: Netflix/BBC broadcast guidelines compliance
- **Sentence splitting**: AI-powered intelligent sentence segmentation
- **Zero dependencies on heavy ML frameworks**: Lightweight and fast

## Installation

```bash
# Basic installation
pip install lattifai-captions

# With sentence splitting support
pip install lattifai-captions[splitting]
```

## Quick Start

```python
from lattifai.caption import Caption

# Read a caption file
caption = Caption.read("input.srt")

# Write to different format
caption.write("output.vtt")

# Convert to string
vtt_content = caption.to_string("vtt")

# Access segments
for segment in caption.supervisions:
    print(f"{segment.start:.2f} - {segment.end:.2f}: {segment.text}")
```

## Supported Formats

### Input/Output (Read & Write)

| Format | Extensions | Description |
|--------|------------|-------------|
| **SRT** | `.srt` | SubRip subtitle format |
| **VTT** | `.vtt` | WebVTT, includes YouTube VTT with word-level timestamps |
| **ASS/SSA** | `.ass`, `.ssa` | Advanced SubStation Alpha |
| **SRV3** | `.srv3`, `.ytt` | YouTube Timed Text v3 with word-level timing |
| **SBV** | `.sbv` | YouTube SubViewer format |
| **SUB** | `.sub` | MicroDVD subtitle format |
| **SAMI** | `.sami`, `.smi` | SAMI subtitle format |
| **JSON** | `.json` | Structured data with word-level support |
| **CSV/TSV** | `.csv`, `.tsv` | Tabular formats |
| **TextGrid** | `.textgrid` | Praat TextGrid format |
| **LRC** | `.lrc` | Lyrics format with word-level timestamps |
| **Gemini** | `.md` | Gemini AI transcript markdown |

### Output Only

| Format | Extensions | Description |
|--------|------------|-------------|
| **TTML** | `.ttml` | Timed Text Markup Language (W3C standard) |
| **IMSC1** | `.ttml` | Netflix/streaming TTML profile |
| **EBU-TT-D** | `.ttml` | European broadcast TTML profile |
| **Avid DS** | `.txt` | Avid Media Composer SubCap |
| **FCPXML** | `.fcpxml` | Final Cut Pro XML |
| **Premiere XML** | `.xml` | Adobe Premiere Pro XML |
| **Audition CSV** | `.csv` | Adobe Audition markers |
| **EdiMarker CSV** | `.csv` | Pro Tools markers |

## Word-Level Timing

Many formats support word-level timing for karaoke-style output:

```python
from lattifai.caption import Caption

caption = Caption.read("input.srv3")  # SRV3 has built-in word timing

# Access word-level alignment
for segment in caption.supervisions:
    if segment.alignment and "word" in segment.alignment:
        for word in segment.alignment["word"]:
            print(f"  {word.symbol}: {word.start:.3f}s - {word.end:.3f}s")

# Export with word-level timing
caption.write("output.json", word_level=True)  # JSON preserves words array
caption.write("output.ass", word_level=True, karaoke_config=KaraokeConfig(enabled=True))
```

## YouTube SRV3 Format

SRV3 is YouTube's proprietary timed text format with millisecond-precision word timing:

```python
from lattifai.caption import Caption

# Read SRV3 (automatically extracts word-level timing)
caption = Caption.read("video.srv3")

# Convert to other formats
caption.write("output.srt")  # Standard SRT
caption.write("output.vtt", word_level=True)  # VTT with word timing
caption.write("output.srv3", word_level=True)  # Back to SRV3
```

SRV3 structure example:
```xml
<timedtext format="3">
  <body>
    <p t="240" d="6559" w="1">
      <s ac="0">Does</s>
      <s t="320" ac="0"> fast</s>
      <s t="560" ac="0"> charging</s>
    </p>
  </body>
</timedtext>
```

## Sentence Splitting

Split captions into natural sentences (requires `[splitting]` extra):

```python
from lattifai.caption import Caption, SentenceSplitter

# Using Caption method
caption = Caption.read("input.srt")
split_caption = caption.split_sentences()

# Or use SentenceSplitter directly
splitter = SentenceSplitter()
split_supervisions = splitter.split_sentences(caption.supervisions)
```

## Format Conversion

```python
from lattifai.caption import Caption

# Read any format
caption = Caption.read("input.srt")

# Write to any supported format
caption.write("output.vtt")
caption.write("output.ass")
caption.write("output.json")
caption.write("output.srv3", word_level=True)
caption.write("output.ttml")

# Or get as string
srt_content = caption.to_string("srt")
json_content = caption.to_string("json", word_level=True)
```

## Standardization

Apply broadcast standards to captions:

```python
from lattifai.caption import Caption, CaptionStandardizer

standardizer = CaptionStandardizer(
    min_duration=0.7,      # Minimum segment duration
    max_duration=7.0,      # Maximum segment duration
    min_gap=0.08,          # Minimum gap between segments
    max_lines=2,           # Maximum lines per segment
    max_chars_per_line=42, # Maximum characters per line
)

caption = Caption.read("input.srt")
standardized = standardizer.process(caption.supervisions)
```

## Validation

Check captions against quality standards:

```python
from lattifai.caption import Caption, CaptionValidator

validator = CaptionValidator(
    min_duration=0.7,
    max_duration=7.0,
    min_gap=0.08,
    max_chars_per_line=42,
)

caption = Caption.read("input.srt")
result = validator.validate(caption.supervisions)

print(f"Valid: {result.valid}")
print(f"Average CPS: {result.avg_cps:.1f}")
print(f"Max CPL: {result.max_cpl}")
print(f"Warnings: {result.warnings}")
```

## API Reference

### Caption Class

```python
from lattifai.caption import Caption

# Class methods
Caption.read(path, format=None, normalize_text=True)
Caption.from_string(content, format)
Caption.from_supervisions(supervisions, language=None, metadata=None)

# Instance methods
caption.write(path, include_speaker=True, word_level=False, karaoke_config=None)
caption.to_string(format, include_speaker=True, word_level=False, karaoke_config=None)
caption.split_sentences()
caption.shift_time(seconds)

# Properties
caption.supervisions  # List[Supervision]
caption.duration      # Total duration in seconds
caption.language      # Language code
caption.source_format # Original format
```

### Supervision Class

```python
from lattifai.caption import Supervision

sup = Supervision(
    start=0.0,           # Start time in seconds
    duration=2.5,        # Duration in seconds
    text="Hello world",  # Caption text
    speaker="Alice",     # Optional speaker label
    alignment=None,      # Optional word-level alignment
)

# Properties
sup.end       # start + duration
sup.text      # Caption text
sup.speaker   # Speaker label
sup.alignment # Dict with "word" key containing AlignmentItem list
```

## License

Apache-2.0
