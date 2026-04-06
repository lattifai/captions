"""Tests for auto-detecting caption format from pasted content.

Scenario: user pastes raw caption text into a textarea (no file extension).
Caption.from_string() should auto-detect the format when format is omitted.

Each format has at least 2 detection samples exercising different edge cases.
"""

import pytest

from lattifai.caption import Caption
from lattifai.caption.formats import detect_format_from_content

# =============================================================================
# Sample content per format — at least 2 variants each
# =============================================================================

# -- VTT -----------------------------------------------------------------------
VTT_STANDARD = """\
WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world

00:00:02.500 --> 00:00:05.000
This is pasted VTT content
"""

VTT_WITH_METADATA = """\
WEBVTT
Kind: captions
Language: zh-Hans

00:00:00.000 --> 00:00:03.000
你好世界
"""

VTT_WITH_NOTE_AND_POSITION = """\
WEBVTT

NOTE This is a comment

00:01:30.500 --> 00:01:33.200 position:10% align:start
こんにちは世界
"""

VTT_HLS_TIMESTAMP_MAP = """\
WEBVTT
X-TIMESTAMP-MAP=MPEGTS:900000,LOCAL:00:00:00.000

00:00:00.000 --> 00:00:02.000
<c.colorE5E5E5>Text with inline color class</c>
"""

# -- SRT -----------------------------------------------------------------------
SRT_STANDARD = """\
1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is pasted SRT content
"""

SRT_CHINESE = """\
1
00:00:00,000 --> 00:00:03,000
LattifAI 是专为音视频内容资产结构化而设计的高性能引擎。

2
00:00:03,500 --> 00:00:06,000
它完全在本地计算运行，确保您的数据安全。
"""

SRT_ZERO_PADDED_INDEX = """\
0001
00:00:00,000 --> 00:00:02,000
Zero-padded index.

0002
00:00:02,500 --> 00:00:05,000
Second line.
"""

SRT_WITH_BOM = "\ufeff1\n00:00:00,000 --> 00:00:02,000\nBOM prefix\n"

# -- ASS -----------------------------------------------------------------------
ASS_STANDARD = """\
[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000088EF,&H00000000,&H00666666,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello world
"""

ASS_1080P_CJK = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans CJK SC,36,&H00FFFFFF,&H000088EF,&H00000000,&H00666666,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,,1920x1080 中文测试
"""

ASS_MINIMAL_WITH_COMMENT = """\
[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,This is a comment line
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Actual dialogue
"""

# -- SSA -----------------------------------------------------------------------
SSA_STANDARD = """\
[Script Info]
Title: Test
ScriptType: v4.00

[V4 Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,20

[Events]
Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: Marked=0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello SSA
"""

SSA_MINIMAL = """\
[Script Info]
ScriptType: v4.00

[Events]
Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: Marked=0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Minimal SSA
"""

# -- LRC -----------------------------------------------------------------------
LRC_STANDARD = """\
[00:00.00]Hello world
[00:02.50]This is pasted LRC content
[00:05.00]Third line
"""

LRC_WITH_METADATA = """\
[ti:夜曲]
[ar:周杰伦]
[al:十一月的肖邦]
[00:12.34]一盏黄黄旧旧的灯
[00:16.78]时间在旁闷不吭声
"""

LRC_MULTIPLE_TAGS = """\
[00:10.50][00:12.50]Chorus line repeated
[00:15.00]Next line
"""

# -- JSON ----------------------------------------------------------------------
JSON_ARRAY = """\
[
  {"text": "Hello world", "start": 0.0, "duration": 2.0},
  {"text": "This is pasted JSON", "start": 2.5, "duration": 2.5}
]
"""

JSON_OBJECT_WRAPPER = """\
{
  "supervisions": [
    {"text": "日本語テスト", "start": 0.0, "duration": 1.5, "speaker": "田中"},
    {"text": "второй сегмент", "start": 2.0, "duration": 2.0, "speaker": "Иван"}
  ]
}
"""

# -- SBV -----------------------------------------------------------------------
SBV_STANDARD = """\
0:00:00.000,0:00:02.000
Hello world

0:00:02.500,0:00:05.000
This is SBV content
"""

SBV_SPANISH = """\
0:00:00.000,0:00:01.500
Mínimo contenido SBV

0:00:02.000,0:00:04.000
Segunda línea en español
"""

SBV_MULTILINE_CUE = """\
0:00:00.000,0:00:05.000
Hello, world!
This is a second line, with commas.

0:00:06.000,0:00:10.000
Another block.
"""

# -- Markdown ------------------------------------------------------------------
MD_WITH_TIMESTAMPS = """\
[00:00] **Host:** Welcome to the show.

[00:05] **Guest:** Thanks for having me.

[00:10] Let's talk about AI.
"""

MD_SPEAKER_ONLY = """\
**Alice:** I think this approach is better.

**Bob:** I disagree, let me explain why.
"""

MD_CHINESE_MEETING = """\
# Meeting Transcript

[00:00:00] **张总:** 大家好，今天讨论一下Q3的目标。

[00:00:15] **李经理:** 我们的KPI需要调整。

[00:00:30] 具体数字稍后发邮件。
"""

MD_LIST_FORMAT = """\
* [00:01:00] **Interviewer:** Can you tell us more?
* [00:01:05] **Guest:** Absolutely, let me explain.
"""

# -- CSV -----------------------------------------------------------------------
CSV_STANDARD = """\
speaker,start,end,text
Alice,0.0,2.0,Hello world
Bob,2.5,5.0,This is CSV content
"""

CSV_EXTRA_COLUMNS = """\
ID,Start,End,Speaker,Text,Confidence
1,0.0,2.0,Alice,"Hello, world!",0.99
2,2.5,5.0,Bob,"How are you?",0.85
"""

# -- TSV -----------------------------------------------------------------------
TSV_STANDARD = "speaker\tstart\tend\ttext\nAlice\t0.0\t2.0\tHello world\nBob\t2.5\t5.0\tThis is TSV\n"

TSV_JAPANESE = "speaker\tstart\tend\ttext\n田中\t0.0\t1.5\t日本語テスト\n"

# -- Audacity (AUD) ------------------------------------------------------------
AUD_STANDARD = """\
0.000000\t2.000000\tHello world
2.500000\t5.000000\tSecond segment
"""

AUD_INTEGER_SECONDS = """\
0\t1.5\tFirst label
1.5\t3\tSecond label — integer seconds
"""

AUD_EMPTY_LABEL = """\
0.500000\t1.500000\t
2.000000\t3.000000\tLabel 2
"""

# -- TextGrid ------------------------------------------------------------------
TEXTGRID_STANDARD = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 5.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 5.0
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 2.0
            text = "Hello world"
"""

TEXTGRID_JAPANESE = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "transcript"
        xmin = 0
        xmax = 2.0
        intervals: size = 2
        intervals [1]:
            xmin = 0.0
            xmax = 1.0
            text = "はい"
        intervals [2]:
            xmin = 1.0
            xmax = 2.0
            text = "どうも"
"""

# -- MicroDVD (SUB) ------------------------------------------------------------
SUB_STANDARD = """\
{0}{50}Hello world
{60}{120}This is MicroDVD
"""

SUB_GERMAN_WITH_FPS = """\
{1}{1}29.97
{0}{50}Erste Zeile auf Deutsch
{60}{120}Zweite Zeile mit Ü
"""

# -- SAMI ----------------------------------------------------------------------
SAMI_STANDARD = """\
<SAMI>
<HEAD><TITLE>Test</TITLE></HEAD>
<BODY>
<SYNC Start=0><P Class=ENCC>Hello world</P></SYNC>
<SYNC Start=2500><P Class=ENCC>SAMI content</P></SYNC>
</BODY>
</SAMI>
"""

SAMI_KOREAN_LOWERCASE = """\
<sami>
<head>
<style type="text/css">p { font-family: sans-serif; }</style>
</head>
<body>
<sync start="0"><p class="KRCC">안녕하세요</p></sync>
<sync start="2000"><p class="KRCC">테스트입니다</p></sync>
</body>
</sami>
"""

# -- SRV3 (YouTube) ------------------------------------------------------------
SRV3_STANDARD = """\
<?xml version="1.0" encoding="utf-8"?>
<timedtext format="3">
<body>
<p t="0" d="2000">Hello world</p>
<p t="2500" d="2500">SRV3 content</p>
</body>
</timedtext>
"""

SRV3_WITH_HEAD = """\
<?xml version="1.0" encoding="utf-8"?>
<timedtext format="3">
<head><wp id="1" ap="7"/></head>
<body>
<p t="500" d="1500" w="1">Segment with attributes</p>
</body>
</timedtext>
"""

SRV3_MINIMAL = """\
<timedtext format="3">
<body>
<p t="0" d="1000">Minimal</p>
</body>
</timedtext>
"""

# -- TTML ----------------------------------------------------------------------
TTML_STANDARD = """\
<?xml version="1.0" encoding="UTF-8"?>
<tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml">
<body>
<div>
<p begin="00:00:00.000" end="00:00:02.000">Hello world</p>
</div>
</body>
</tt>
"""

TTML_NAMESPACED = """\
<?xml version="1.0" encoding="UTF-8"?>
<tt:tt xml:lang="ja" xmlns:tt="http://www.w3.org/ns/ttml"
    xmlns:tts="http://www.w3.org/ns/ttml#styling">
<tt:body>
  <tt:div>
    <tt:p begin="0s" end="2s">日本語テスト</tt:p>
  </tt:div>
</tt:body>
</tt:tt>
"""

# -- FCPXML --------------------------------------------------------------------
FCPXML_STANDARD = """\
<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
<library>
<event name="test"><project name="test"/></event>
</library>
</fcpxml>
"""

FCPXML_MINIMAL = """\
<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.8">
<project name="Test"/>
</fcpxml>
"""

# -- Premiere XML --------------------------------------------------------------
PREMIERE_STANDARD = """\
<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
<sequence><name>test</name></sequence>
</xmeml>
"""

PREMIERE_WITH_DOCTYPE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
<sequence id="sequence-1"/>
</xmeml>
"""

# -- Plain text (TXT fallback) ------------------------------------------------
TXT_ENGLISH = "Today we're going to talk about artificial intelligence."

TXT_CHINESE = "今天我们来聊一聊人工智能的发展。机器学习已经改变了很多行业。"

TXT_MULTILINE = """\
Welcome to our show.

Today's guest is a scientist working on climate change.

He has published over 50 papers on the topic."""

TXT_MISLEADING_NUMBERS = """\
In the year 2023, we saw a 50.5% increase.
The event started at 14:00 but there was no transcript time.
Section 1.2: The beginning.
"""

TXT_EMOJI = "🎬 This video covers the basics of machine learning 🤖"

TXT_WHITESPACE_PADDED = "  \n  Just whitespace-padded text with leading blanks.\n  "


# =============================================================================
# Parametrized detection tests
# =============================================================================

DETECT_CASES = [
    # -- VTT (4) --
    (VTT_STANDARD, "vtt", "standard"),
    (VTT_WITH_METADATA, "vtt", "with Kind/Language metadata"),
    (VTT_WITH_NOTE_AND_POSITION, "vtt", "NOTE comment + position cues"),
    (VTT_HLS_TIMESTAMP_MAP, "vtt", "HLS X-TIMESTAMP-MAP + inline styles"),
    # -- SRT (4) --
    (SRT_STANDARD, "srt", "standard"),
    (SRT_CHINESE, "srt", "Chinese text"),
    (SRT_ZERO_PADDED_INDEX, "srt", "zero-padded index numbers"),
    (SRT_WITH_BOM, "srt", "UTF-8 BOM prefix"),
    # -- ASS (3) --
    (ASS_STANDARD, "ass", "standard"),
    (ASS_1080P_CJK, "ass", "1080p PlayRes with CJK font"),
    (ASS_MINIMAL_WITH_COMMENT, "ass", "minimal with comment event"),
    # -- SSA (2) --
    (SSA_STANDARD, "ssa", "standard v4.00"),
    (SSA_MINIMAL, "ssa", "minimal"),
    # -- LRC (3) --
    (LRC_STANDARD, "lrc", "standard"),
    (LRC_WITH_METADATA, "lrc", "with [ti:] [ar:] [al:] metadata"),
    (LRC_MULTIPLE_TAGS, "lrc", "multiple time tags per line"),
    # -- JSON (2) --
    (JSON_ARRAY, "json", "flat array"),
    (JSON_OBJECT_WRAPPER, "json", "object with CJK/Cyrillic"),
    # -- SBV (3) --
    (SBV_STANDARD, "sbv", "standard"),
    (SBV_SPANISH, "sbv", "Spanish text"),
    (SBV_MULTILINE_CUE, "sbv", "multi-line cue with commas"),
    # -- Markdown (4) --
    (MD_WITH_TIMESTAMPS, "markdown", "timestamps + speakers"),
    (MD_SPEAKER_ONLY, "markdown", "speaker labels only"),
    (MD_CHINESE_MEETING, "markdown", "Chinese meeting transcript"),
    (MD_LIST_FORMAT, "markdown", "unordered list format"),
    # -- CSV (2) --
    (CSV_STANDARD, "csv", "standard"),
    (CSV_EXTRA_COLUMNS, "csv", "extra columns + quoted fields"),
    # -- TSV (2) --
    (TSV_STANDARD, "tsv", "standard"),
    (TSV_JAPANESE, "tsv", "Japanese text"),
    # -- AUD (3) --
    (AUD_STANDARD, "aud", "standard"),
    (AUD_INTEGER_SECONDS, "aud", "integer seconds"),
    (AUD_EMPTY_LABEL, "aud", "empty label text"),
    # -- TextGrid (2) --
    (TEXTGRID_STANDARD, "textgrid", "standard"),
    (TEXTGRID_JAPANESE, "textgrid", "Japanese intervals"),
    # -- SUB (2) --
    (SUB_STANDARD, "sub", "standard MicroDVD"),
    (SUB_GERMAN_WITH_FPS, "sub", "German with FPS header"),
    # -- SAMI (2) --
    (SAMI_STANDARD, "sami", "standard uppercase"),
    (SAMI_KOREAN_LOWERCASE, "sami", "Korean with lowercase tags"),
    # -- SRV3 (3) --
    (SRV3_STANDARD, "srv3", "standard"),
    (SRV3_WITH_HEAD, "srv3", "with head/wp attributes"),
    (SRV3_MINIMAL, "srv3", "minimal without XML declaration"),
    # -- TTML (2) --
    (TTML_STANDARD, "ttml", "standard"),
    (TTML_NAMESPACED, "ttml", "tt: namespace prefix"),
    # -- FCPXML (2) --
    (FCPXML_STANDARD, "fcpxml", "standard v1.9"),
    (FCPXML_MINIMAL, "fcpxml", "minimal v1.8"),
    # -- Premiere XML (2) --
    (PREMIERE_STANDARD, "premiere_xml", "standard"),
    (PREMIERE_WITH_DOCTYPE, "premiere_xml", "with DOCTYPE"),
    # -- TXT fallback (6) --
    (TXT_ENGLISH, "txt", "English sentence"),
    (TXT_CHINESE, "txt", "Chinese sentence"),
    (TXT_MULTILINE, "txt", "multi-paragraph"),
    (TXT_MISLEADING_NUMBERS, "txt", "misleading numbers that look like times"),
    (TXT_EMOJI, "txt", "emoji content"),
    (TXT_WHITESPACE_PADDED, "txt", "whitespace-padded"),
]


def _case_id(val):
    """Generate readable test ID from (content, format, desc) tuple."""
    if isinstance(val, tuple) and len(val) == 3:
        return f"{val[1]}-{val[2].replace(' ', '_')}"
    return None


class TestDetectFormatFromContent:
    """Parametrized: detect_format_from_content() returns correct format ID."""

    @pytest.mark.parametrize("content,expected,desc", DETECT_CASES, ids=[f"{c[1]}-{c[2]}" for c in DETECT_CASES])
    def test_detect(self, content, expected, desc):
        assert detect_format_from_content(content) == expected, f"Failed for: {desc}"


# =============================================================================
# Integration tests — from_string() auto-detect end-to-end
# =============================================================================

PARSE_CASES = [
    # (content, expected_format, min_segments, text_substring)
    (VTT_STANDARD, "vtt", 2, "Hello world"),
    (SRT_STANDARD, "srt", 2, "Hello world"),
    (SRT_CHINESE, "srt", 2, "高性能引擎"),
    (ASS_STANDARD, "ass", 1, "Hello world"),
    (LRC_STANDARD, "lrc", 2, "Hello world"),
    (JSON_ARRAY, "json", 2, "Hello world"),
    (SBV_STANDARD, "sbv", 2, "Hello world"),
    (MD_WITH_TIMESTAMPS, "markdown", 2, "Welcome"),
    (CSV_STANDARD, "csv", 2, "Hello world"),
    (AUD_STANDARD, "aud", 1, "Hello world"),
]


class TestFromStringAutoDetect:
    """Parametrized: from_string() auto-detects and parses correctly."""

    @pytest.mark.parametrize(
        "content,expected_fmt,min_segs,text_sub",
        PARSE_CASES,
        ids=[c[1] for c in PARSE_CASES],
    )
    def test_auto_detect_parse(self, content, expected_fmt, min_segs, text_sub):
        caption = Caption.from_string(content)
        assert caption.source_format == expected_fmt
        assert len(caption) >= min_segs
        full_text = " ".join(s.text for s in caption)
        assert text_sub in full_text

    def test_explicit_format_overrides(self):
        """Explicit format parameter bypasses auto-detection."""
        caption = Caption.from_string(SRT_STANDARD, format="srt")
        assert len(caption) == 2
        assert caption.source_format == "srt"

    def test_plain_text_english(self):
        caption = Caption.from_string(TXT_ENGLISH)
        assert len(caption) >= 1
        assert "artificial intelligence" in " ".join(s.text for s in caption)

    def test_plain_text_chinese(self):
        caption = Caption.from_string(TXT_CHINESE)
        assert len(caption) >= 1
        assert "人工智能" in " ".join(s.text for s in caption)

    def test_vtt_metadata_preserved(self):
        caption = Caption.from_string(VTT_WITH_METADATA)
        assert caption.source_format == "vtt"
        assert caption.language == "zh-Hans"
