"""P0-1: Encoding detection tests for ASS/SRT reader.

Verifies that caption files in various encodings (UTF-16, GBK, UTF-8-sig,
GB18030, Latin-1) can be read correctly without UnicodeDecodeError.
"""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import Caption

# Minimal ASS content template with CJK text for encoding tests
ASS_TEMPLATE = """\
[Script Info]
Title: Encoding Test
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,他决不会放弃塔伯特
Dialogue: 0,0:00:05.00,0:00:10.00,Default,,0,0,0,,He would never give up on Talbot
"""

SRT_TEMPLATE = """\
1
00:00:01,000 --> 00:00:05,000
他决不会放弃塔伯特

2
00:00:05,000 --> 00:00:10,000
He would never give up on Talbot
"""


def _write_file(content: str, encoding: str, suffix: str, tmpdir: str, bom: bytes = b"") -> Path:
    """Write content with specific encoding and optional BOM."""
    path = Path(tmpdir) / f"test{suffix}"
    raw = bom + content.encode(encoding)
    path.write_bytes(raw)
    return path


class TestEncodingDetection:
    """Caption reader must handle diverse encodings (P0-1)."""

    def test_utf8_sig_ass(self, tmp_path):
        """UTF-8 with BOM (42% of YYeTS files)."""
        path = _write_file(ASS_TEMPLATE, "utf-8-sig", ".ass", str(tmp_path))
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_utf16_le_ass(self, tmp_path):
        """UTF-16 LE with BOM (41% of YYeTS files — currently all fail)."""
        path = _write_file(ASS_TEMPLATE, "utf-16-le", ".ass", str(tmp_path), bom=b"\xff\xfe")
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_utf16_be_ass(self, tmp_path):
        """UTF-16 BE with BOM."""
        path = _write_file(ASS_TEMPLATE, "utf-16-be", ".ass", str(tmp_path), bom=b"\xfe\xff")
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_gbk_ass(self, tmp_path):
        """GBK encoding (7% of YYeTS files)."""
        path = _write_file(ASS_TEMPLATE, "gbk", ".ass", str(tmp_path))
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_gb18030_ass(self, tmp_path):
        """GB18030 encoding (2% of YYeTS files)."""
        path = _write_file(ASS_TEMPLATE, "gb18030", ".ass", str(tmp_path))
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_latin1_ass(self, tmp_path):
        """Latin-1 encoding (2% of YYeTS files)."""
        # Latin-1 can't encode CJK, use ASCII-only content
        content = ASS_TEMPLATE.replace("他决不会放弃塔伯特", "Bonjour le monde")
        path = _write_file(content, "latin-1", ".ass", str(tmp_path))
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "Bonjour" in cap.supervisions[0].text

    def test_utf16_srt(self, tmp_path):
        """UTF-16 SRT file."""
        path = _write_file(SRT_TEMPLATE, "utf-16-le", ".srt", str(tmp_path), bom=b"\xff\xfe")
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_gbk_srt(self, tmp_path):
        """GBK SRT file."""
        path = _write_file(SRT_TEMPLATE, "gbk", ".srt", str(tmp_path))
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2
        assert "塔伯特" in cap.supervisions[0].text

    def test_utf8_still_works(self, tmp_path):
        """Plain UTF-8 (no BOM) must still work."""
        path = _write_file(ASS_TEMPLATE, "utf-8", ".ass", str(tmp_path))
        cap = Caption.read(str(path))
        assert len(cap.supervisions) == 2

    def test_encoding_metadata_preserved(self, tmp_path):
        """Reader should detect and store the original encoding in metadata."""
        path = _write_file(ASS_TEMPLATE, "utf-16-le", ".ass", str(tmp_path), bom=b"\xff\xfe")
        cap = Caption.read(str(path))
        # After P0-1, metadata should include detected encoding
        assert cap.metadata.get("encoding") is not None
