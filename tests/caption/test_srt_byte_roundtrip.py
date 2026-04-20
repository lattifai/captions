"""Byte-level SRT roundtrip fidelity tests.

Covers three known pysubs2 serialization losses for bilingual/subtitle-group SRTs
(reproduced from a real YYeTs Endeavour S09E01 bilingual file):

1. UTF-8 BOM is already handled for SRT — verify still preserved.
2. CRLF line terminators become LF on write (pysubs2 emits ``\\n``).
3. Inline ASS override tags ``{\\an1}{\\pos(18.9,254.933)}`` are stripped by the
   pysubs2 SRT parser. Subtitle groups use these tags to position sign/credit/
   title text on screen. Losing them collapses all positioned text to the
   default lower-center, overlaying the dialogue region.
4. Millisecond timestamps drift ``-1 ms`` (``,820`` → ``,819``) because the
   writer uses ``int(x * 1000)`` instead of ``round(x * 1000)``.
"""

import re
from pathlib import Path

from lattifai.caption import Caption


# Fixture: UTF-8 BOM + CRLF throughout + inline ASS override tags on sign rows +
# a timestamp with a digit that triggers float truncation (``,820`` → ``,819``).
YYETS_STYLE_SRT_UTF8_BOM_CRLF = (
    "\ufeff"  # BOM
    "1\r\n"
    "00:00:00,020 --> 00:00:06,660\r\n"
    "{\\an1}{\\pos(18.9,254.933)}《摩斯探长前传》前情提要\r\n"
    "\r\n"
    "2\r\n"
    "00:00:54,500 --> 00:00:59,500\r\n"
    "{\\an4}{\\pos(67,235)}翻|校|监        草草          后期      吉吉\r\n"
    "\r\n"
    "3\r\n"
    "00:17:05,820 --> 00:17:07,700\r\n"
    "{\\an8}泰晤士河谷城堡门警局\r\n"
    "\r\n"
    "4\r\n"
    "00:23:43,980 --> 00:23:48,980\r\n"
    "我们都很看好你\r\n"
    "We all think a lot of you, you know?\r\n"
    "\r\n"
)


def _write_and_read_back(tmp_path: Path) -> Path:
    inp = tmp_path / "in.srt"
    inp.write_bytes(YYETS_STYLE_SRT_UTF8_BOM_CRLF.encode("utf-8"))
    cap = Caption.read(str(inp))
    out = tmp_path / "out.srt"
    cap.write(str(out))
    return out


def test_utf8_bom_preserved_on_roundtrip(tmp_path: Path) -> None:
    """SRT UTF-8 BOM must survive read + write."""
    out = _write_and_read_back(tmp_path)
    data = out.read_bytes()
    assert data[:3] == b"\xef\xbb\xbf", (
        f"UTF-8 BOM lost on SRT write; first 6 bytes: {data[:6]!r}"
    )


def test_crlf_line_terminators_preserved(tmp_path: Path) -> None:
    """CRLF input must round-trip as CRLF output (no bare LF)."""
    inp = tmp_path / "in.srt"
    inp.write_bytes(YYETS_STYLE_SRT_UTF8_BOM_CRLF.encode("utf-8"))
    cap = Caption.read(str(inp))
    assert cap.metadata.get("line_terminator") == "\r\n"

    out = tmp_path / "out.srt"
    cap.write(str(out))
    data = out.read_bytes()

    bare_lf_count = len(re.findall(rb"(?<!\r)\n", data))
    crlf_count = data.count(b"\r\n")
    assert bare_lf_count == 0, (
        f"Found {bare_lf_count} bare LF on SRT output (expected 0); "
        f"CRLF count: {crlf_count}"
    )


def test_inline_ass_override_tags_preserved(tmp_path: Path) -> None:
    """Inline ``{\\an1}{\\pos(...)}`` must survive SRT roundtrip.

    Subtitle groups rely on these tags to pin sign/credit/title lines to
    specific screen positions. pysubs2 strips them on parse by default.
    """
    out = _write_and_read_back(tmp_path)
    text = out.read_bytes().decode("utf-8-sig")

    assert "{\\an1}{\\pos(18.9,254.933)}《摩斯探长前传》前情提要" in text, (
        "cue 1 ``\\an1``+``\\pos`` override stripped on roundtrip"
    )
    assert "{\\an4}{\\pos(67,235)}翻|校|监" in text, (
        "cue 2 ``\\an4``+``\\pos`` override stripped on roundtrip"
    )
    assert "{\\an8}泰晤士河谷城堡门警局" in text, (
        "cue 3 ``\\an8`` override stripped on roundtrip"
    )


def test_millisecond_timestamps_no_drift(tmp_path: Path) -> None:
    """``00:17:05,820`` must stay ``,820`` (not ``,819`` from int truncation)."""
    out = _write_and_read_back(tmp_path)
    text = out.read_bytes().decode("utf-8-sig")
    assert "00:17:05,820 --> 00:17:07,700" in text, (
        "cue 3 timestamp drifted (int truncation instead of round). "
        f"Lines containing 00:17:05: {[ln for ln in text.splitlines() if '17:05' in ln]}"
    )
