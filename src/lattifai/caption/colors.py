"""Unified color management for LattifAI caption styles.

This module centralizes all color palettes, presets, and conversion utilities
used across caption formats (ASS, VTT, TTML, etc.).

Color format conventions:
    - User-facing / config: #RRGGBB  (e.g. "#1387C0")
    - ASS inline override:  BBGGRR   (e.g. "C08713")
    - ASS style field:      &HAABBGGRR
"""

from typing import Dict, List, Optional

# =============================================================================
# Karaoke Color Schemes
# =============================================================================

# Each scheme: primary (unsung text), secondary (highlight sweep), outline, back (shadow)
KARAOKE_COLOR_SCHEMES: Dict[str, Dict[str, str]] = {
    "azure-gold": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#FFC209",  # 金柠暖阳
        "outline_color": "#1387C0",  # 晴空海蓝
        "back_color": "#0A3D5C",
        "outline_width": 2.0,
    },
    "sakura-purple": {
        "primary_color": "#F7C3D9",  # 柔樱粉
        "secondary_color": "#7953B1",  # 紫鸢深紫
        "outline_color": "#063C85",  # 深海藏蓝
        "back_color": "#1A1A2E",
        "outline_width": 2.0,
    },
    "mint-ocean": {
        "primary_color": "#A1FEEF",  # 薄荷冰青
        "secondary_color": "#658AE4",  # 柔空蓝
        "outline_color": "#28314E",  # 深海夜蓝
        "back_color": "#0A0A1A",
        "outline_width": 2.0,
    },
    "gardenia-green": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#9DC92A",  # 苹果绿
        "outline_color": "#77964A",  # 碧山
        "back_color": "#1C2B1A",
        "outline_width": 2.0,
    },
    "sunset-warm": {
        "primary_color": "#FAEDD1",  # 奶油米白
        "secondary_color": "#F4520D",  # 暖橙光
        "outline_color": "#1387C0",  # 晴空海蓝
        "back_color": "#0A1628",
        "outline_width": 2.0,
    },
    "prussian-elegant": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#FBC03D",  # 栀子黄
        "outline_color": "#003153",  # 普鲁士蓝
        "back_color": "#001A2C",
        "outline_width": 2.5,
    },
    "burgundy-classic": {
        "primary_color": "#F7F2DF",  # 宣纸白
        "secondary_color": "#CC5D84",  # 琅玕紫
        "outline_color": "#800020",  # 勃艮第红
        "back_color": "#2A000D",
        "outline_width": 2.0,
    },
    "langgan-spring": {
        "primary_color": "#C1D796",  # 春辰 (unsung text)
        "secondary_color": "#CC5D84",  # 琅玕紫 (highlight sweep)
        "outline_color": "#8A3A5A",  # 琅玕紫暗化
        "back_color": "#2A1020",
        "outline_width": 2.0,
    },
    "mars-teal": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#008C8C",  # 马尔斯绿
        "outline_color": "#003153",  # 普鲁士蓝
        "back_color": "#001A1A",
        "outline_width": 2.0,
    },
    "spring-field": {
        "primary_color": "#FBFFF2",  # 荔枝白
        "secondary_color": "#46B065",  # Spring Fields 中绿
        "outline_color": "#008E6B",  # Spring Fields 深绿
        "back_color": "#0A2A1A",
        "outline_width": 2.0,
    },
    "navy-pink": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#F7C3D9",  # 柔樱粉
        "outline_color": "#063C85",  # 深海藏蓝
        "back_color": "#021A3A",
        "outline_width": 2.0,
    },
    "apricot-dark": {
        "primary_color": "#FEA72E",  # 杏黄
        "secondary_color": "#F7F2DF",  # 宣纸白
        "outline_color": "#3A3C50",  # 玄青
        "back_color": "#1A1A28",
        "outline_width": 2.0,
    },
}


def resolve_karaoke_color_scheme(name: str) -> Optional[Dict]:
    """Resolve a karaoke color scheme name to style dict. Returns None if not found."""
    return KARAOKE_COLOR_SCHEMES.get(name.lower().strip())


# =============================================================================
# Speaker Color Palette
# =============================================================================

# 10-color palette for auto speaker coloring (BBGGRR format for ASS)
SPEAKER_PALETTE: List[str] = [
    "C08713",  # 晴空海蓝 Azure         (#1387C0)
    "09C2FF",  # 金柠暖阳 Warm Yellow   (#FFC209)
    "D9C3F7",  # 柔樱粉 Soft Pink       (#F7C3D9)
    "2AC99D",  # 苹果绿 Apple Green     (#9DC92A)
    "EFFEA1",  # 薄荷冰青 Mint Ice      (#A1FEEF)
    "0D52F4",  # 暖橙光 Warm Orange     (#F4520D)
    "E48A65",  # 柔空蓝 Sky Blue        (#658AE4)
    "3DC0FB",  # 栀子黄 Gardenia Yellow (#FBC03D)
    "845DCC",  # 琅玕紫 Langgan Purple  (#CC5D84)
    "8C8C00",  # 马尔斯绿 Mars Green    (#008C8C)
]


# =============================================================================
# Color Conversion Utilities
# =============================================================================


def hex_rgb_to_bgr(hex_color: str) -> str:
    """Convert #RRGGBB to BBGGRR (ASS inline override format).

    Args:
        hex_color: Color string in #RRGGBB format

    Returns:
        BBGGRR string (no prefix)

    Example:
        >>> hex_rgb_to_bgr("#1387C0")
        'C08713'
    """
    c = hex_color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Expected #RRGGBB, got '{hex_color}'")
    try:
        int(c, 16)
    except ValueError:
        raise ValueError(f"Expected #RRGGBB with hex digits, got '{hex_color}'")
    return f"{c[4:6]}{c[2:4]}{c[0:2]}".upper()


def bgr_to_hex_rgb(bgr_color: str) -> str:
    """Convert BBGGRR to #RRGGBB.

    Args:
        bgr_color: Color string in BBGGRR format (no prefix)

    Returns:
        #RRGGBB string

    Example:
        >>> bgr_to_hex_rgb("C08713")
        '#1387C0'
    """
    if len(bgr_color) != 6:
        raise ValueError(f"Expected 6-char BBGGRR, got '{bgr_color}'")
    try:
        int(bgr_color, 16)
    except ValueError:
        raise ValueError(f"Expected BBGGRR with hex digits, got '{bgr_color}'")
    return f"#{bgr_color[4:6]}{bgr_color[2:4]}{bgr_color[0:2]}".upper()


def resolve_speaker_color(speaker: str, speaker_color_spec: str, cache: dict) -> str:
    """Resolve ASS BBGGRR color string for a speaker.

    Args:
        speaker: Speaker name
        speaker_color_spec: Color spec string:
            - "": no coloring
            - "auto": use built-in SPEAKER_PALETTE
            - "#RRGGBB,#00BFFF,...": comma-separated, auto-assigned per speaker
        cache: Mutable dict tracking assigned colors {speaker_name: "BBGGRR"}

    Returns:
        BBGGRR color string for ASS inline override, or "" if no coloring
    """
    if not speaker_color_spec:
        return ""

    if speaker in cache:
        return cache[speaker]

    # Parse palette: "auto" uses built-in, comma-separated uses user-provided
    if speaker_color_spec == "auto":
        palette = SPEAKER_PALETTE
    else:
        palette = []
        for c in speaker_color_spec.split(","):
            c = c.strip()
            if not c.startswith("#"):
                c = f"#{c}"
            try:
                palette.append(hex_rgb_to_bgr(c))
            except ValueError:
                continue
        if not palette:
            return ""

    # Assign next color from palette (cycle if more speakers than colors)
    color = palette[len(cache) % len(palette)]
    cache[speaker] = color
    return color
