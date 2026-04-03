# Subtitle Background Color — Design Document

## Problem

ASS format conflates "shadow color" and "background box color" into a single
field `BackColour`, switching behavior via `BorderStyle` (1 = outline+shadow,
3 = opaque box). Our `CaptionStyle.back_color` inherits this ambiguity — the
docstring says "Shadow color" but the field name suggests "background".

Multiple output formats support background boxes independently (TTML, FCPXML,
VTT/CSS), each with their own color format and alpha handling. There is no
unified abstraction to control this across formats.

## Visual Layer Model

Subtitle text rendering has four visual layers, from bottom to top:

```
┌─────────────────────────────────────┐
│  background_color  (background box) │   ← NEW: colored rectangle behind text
│  ┌───────────────────────────────┐  │
│  │  back_color + shadow_depth    │  │   ← existing: drop shadow (legacy)
│  │  ┌─────────────────────────┐  │  │
│  │  │  outline_color + width  │  │  │   ← existing: text stroke
│  │  │  ┌───────────────────┐  │  │  │
│  │  │  │  primary_color    │  │  │  │   ← existing: text fill
│  │  │  └───────────────────┘  │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Design

### CaptionStyle Change

Add one field to `CaptionStyle`. No renames, no breaking changes.

```python
@dataclass
class CaptionStyle:
    # --- Text colors (existing, unchanged) ---
    primary_color: str = "#FFFFFF"       # Main text fill
    secondary_color: str = "#00FFFF"     # Karaoke highlight / secondary
    outline_color: str = "#000000"       # Text outline stroke
    back_color: str = "#000000"          # Shadow color (legacy ASS BackColour in borderstyle=1)

    # --- Text effects (existing, unchanged) ---
    outline_width: float = 0             # Outline thickness
    shadow_depth: float = 1.0            # Shadow distance

    # --- Background box (NEW) ---
    background_color: str = ""
    """Subtitle background box color.
    - "":           no background box (default — text floats on video)
    - "#RRGGBB":    solid opaque background box
    - "#RRGGBBAA":  semi-transparent background box (e.g., "#00000080" = 50% black)
    """
```

### Field Semantics

| Field | Role | When active |
|-------|------|-------------|
| `primary_color` | Text fill color | Always |
| `secondary_color` | Karaoke highlight sweep color | Karaoke mode |
| `outline_color` | Text outline/stroke color | When `outline_width > 0` |
| `back_color` | Drop shadow color (legacy) | When `shadow_depth > 0` and no background box |
| `shadow_depth` | Shadow distance in pixels | When no background box |
| `outline_width` | Outline thickness in pixels | Always (including box mode) |
| **`background_color`** | **Background box fill** | **When non-empty** |

### Interaction Rules

- `background_color=""` (default): text has outline + shadow, no box.
- `background_color="#RRGGBB"` or `"#RRGGBBAA"`: text has background box.
  In ASS, this switches to `borderstyle=3`; `back_color` is repurposed as the
  box color (ASS limitation). Shadow is disabled in box mode per ASS spec.
  `outline_width` still applies (acts as padding between text and box edge).
- When both `background_color` and `back_color` are set, `background_color`
  takes precedence for formats that distinguish them (TTML, FCPXML).
  For ASS (which only has one BackColour), `background_color` wins.

### Top-Level CLI Parameter

`background_color` is a **top-level** CLI parameter (like `speaker_color`),
not nested under `karaoke.style.*`. This is a general caption styling feature,
not karaoke-specific.

## Format Mapping

### ASS/SSA

```
background_color=""       →  BorderStyle=1, BackColour=back_color, Shadow=shadow_depth
background_color="#HEX"   →  BorderStyle=3, BackColour=background_color, Shadow=0
```

ASS `borderstyle=3` uses `BackColour` as the opaque box color and ignores shadow.

**Alpha inversion (CRITICAL)**: ASS alpha values are inverted from standard hex.
- Standard hex: `FF` = fully opaque, `00` = fully transparent
- ASS alpha:    `00` = fully opaque, `FF` = fully transparent
- Conversion: `ass_alpha = 255 - standard_alpha`
- Example: `#00000080` (50% opaque) → `&H7F000000` in ASS (127 = 255 - 128)

### TTML / IMSC1 / EBU-TT-D

```
background_color=""       →  (no tts:backgroundColor attribute)
background_color="#HEX"   →  tts:backgroundColor="#RRGGBB" or "#RRGGBBAA"
```

Note: TTML natively supports both 6-digit `#RRGGBB` (assumes 100% opacity)
and 8-digit `#RRGGBBAA`. Pass as-is; do NOT append `FF` to 6-digit values
to avoid compatibility issues with stricter parsers.

**Existing behavior**: TTML writer already has `TTMLStyle.background_color`
defaulting to `#000000C0`. The `CaptionStyle.background_color` parameter
overrides this when explicitly set; TTML's own default applies when empty.

### FCPXML (Final Cut Pro)

```
background_color=""       →  (no backgroundColor attribute)
background_color="#HEX"   →  backgroundColor="R G B A" (0.0–1.0 floats)
```

Convert `#RRGGBBAA` to space-separated normalized floats.

**Existing behavior**: FCPXML already has `FCPXMLStyle.background_color`.
The `CaptionStyle.background_color` parameter is bridged via kwargs.

### VTT (WebVTT)

```
background_color=""       →  (no background styling, or transparent)
background_color="#HEX"   →  ::cue { background-color: rgba(R, G, B, A); }
```

Convert `#RRGGBBAA` to CSS `rgba()`. The VTT STYLE block must be placed
at the top of the file (WebVTT does not support inline background-color).

**Current limitation**: VTT writer only emits STYLE blocks when ASS metadata
is present. Direct CaptionStyle → VTT styling is a future enhancement.

### Formats Without Background Support

SRT, LRC, JSON, Markdown, TextGrid, Premiere XML, Avid DS, Audition CSV —
`background_color` is silently ignored. No error raised.

## CLI Usage

```bash
# No background (default)
lai caption convert input.json output.ass

# Solid black background box
lai caption convert input.json output.ass background_color="#000000"

# Semi-transparent dark blue background
lai caption convert input.json output.ass background_color="#0A1628C0"

# Combined with karaoke color scheme
lai caption convert input.json output.ass \
    word_level=true \
    karaoke.enabled=true \
    karaoke.color_scheme=azure-gold \
    background_color="#00000080"

# Combined with speaker colors
lai caption convert input.json output.ass \
    include_speaker_in_text=true \
    speaker_color=auto \
    background_color="#00000080"
```

## Color Scheme Integration

Karaoke color schemes (in `KARAOKE_COLOR_SCHEMES`) already have a `back_color`
field used for shadow. To add background box support:

- Add optional `background_color` key to scheme dicts
- `__post_init__` applies it if present
- Schemes without `background_color` keep current behavior (no box)

```python
# Example: scheme with background box
"azure-gold": {
    "primary_color": "#FFFFFF",
    "secondary_color": "#FFC209",
    "outline_color": "#1387C0",
    "back_color": "#0A3D5C",
    "outline_width": 2.0,
    "background_color": "#0A3D5C80",   # optional, semi-transparent
}
```

## Implementation Status

- [x] Add `background_color: str = ""` to `CaptionStyle`
- [x] ASS writer: set `borderstyle=3` and map `background_color` to `backcolor`
- [x] ASS writer: handle `#RRGGBBAA` → inverted alpha conversion (255 - alpha)
- [x] ASS writer: non-karaoke Default style also supports `background_color`
- [x] `_create_karaoke_style()`: pass `borderstyle` based on `background_color`
- [x] Top-level `background_color` param in CLI (`lai caption convert`)
- [x] Thread through `Caption.write()` → writer chain
- [x] KaraokeConfig `__post_init__`: apply `background_color` from scheme
- [x] Tests: alpha inversion, borderstyle switching, karaoke+background, non-karaoke
- [ ] TTML writer: bridge `CaptionStyle.background_color` to `TTMLStyle`
- [ ] FCPXML writer: bridge `CaptionStyle.background_color` to `FCPXMLStyle`
- [ ] VTT writer: direct `CaptionStyle` → CSS `background-color` (without ASS metadata)
- [ ] Input validation: reject malformed color values early
