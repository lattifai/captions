"""Tests for the language-detection helper backed by lingua-py.

Used by the bilingual-cue splitter to decide whether two sides of a ``\\n``
boundary belong to distinct languages.
"""

from __future__ import annotations

import pytest

from lattifai.caption.parsers.language_detector import (
    detect_language,
    detect_script,
    is_distinct_language_pair,
)


# ============================================================================
# detect_language — basic script coverage
# ============================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        # Chinese
        ("我们都很看好你", "zh"),
        ("摩斯", "zh"),
        ("贱人", "zh"),
        ("泰晤士河谷城堡门警局", "zh"),
        # English — strings long enough for lingua to disambiguate.
        # Short Latin words (2-3 chars) are inherently ambiguous in a
        # multi-Romance-language candidate set; use detect_script for those.
        ("We all think a lot of you, you know?", "en"),
        ("Morse, please come in.", "en"),
        # Japanese
        ("ありがとうございます", "ja"),
        ("日本語の字幕", "ja"),
        # Korean
        ("안녕하세요", "ko"),
    ],
)
def test_detect_language_basic(text: str, expected: str) -> None:
    assert detect_language(text) == expected


# ============================================================================
# detect_script — coarse Unicode-block dominant-script detection
# ============================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("OK", "latin"),
        ("Sir", "latin"),
        ("The", "latin"),
        ("Morse.", "latin"),
        ("Hello", "latin"),
        ("café", "latin"),            # accented Latin counts
        ("你好", "east_asian"),
        ("ありがとう", "east_asian"),
        ("안녕하세요", "east_asian"),
        ("猫", "east_asian"),          # 1-char still works
        ("Привет", "cyrillic"),
        ("سلام", "arabic"),
        ("नमस्ते", "devanagari"),
        ("สวัสดี", "thai"),
        ("שלום", "hebrew"),
        ("Καλημέρα", "greek"),
    ],
)
def test_detect_script_basic(text: str, expected: str) -> None:
    assert detect_script(text) == expected


def test_detect_script_empty_or_digits() -> None:
    assert detect_script("") is None
    assert detect_script("   ") is None
    assert detect_script("1972") is None
    assert detect_script("...") is None


def test_detect_language_empty_returns_none() -> None:
    """Empty / whitespace-only strings return ``None`` (no signal)."""
    assert detect_language("") is None
    assert detect_language("   ") is None


def test_detect_language_digits_only_returns_none() -> None:
    """Pure numeric / punctuation has no language signal."""
    assert detect_language("1972") is None
    assert detect_language("...") is None


# ============================================================================
# is_distinct_language_pair — the bilingual-split gate
# ============================================================================


def test_pair_bilingual_zh_en() -> None:
    """Real bilingual cue: Chinese + English."""
    assert is_distinct_language_pair(
        "我们都很看好你",
        "We all think a lot of you, you know?",
    ) is True


def test_pair_bilingual_short_forms() -> None:
    """Bilingual works on 2-char primary / single-word secondary."""
    assert is_distinct_language_pair("摩斯", "Morse.") is True
    assert is_distinct_language_pair("你好", "OK") is True


def test_pair_bilingual_ja_en() -> None:
    """Japanese + English counts as bilingual."""
    assert is_distinct_language_pair("日本語の字幕", "Japanese subtitles") is True


def test_pair_bilingual_ko_en() -> None:
    """Korean + English counts as bilingual."""
    assert is_distinct_language_pair("안녕하세요", "Hello") is True


def test_pair_same_language_english_wrap() -> None:
    """English long sentence soft-wrapped into 2 lines — NOT bilingual."""
    assert is_distinct_language_pair(
        "Long English sentence wrapped",
        "to fit the screen width.",
    ) is False


def test_pair_same_language_chinese_wrap() -> None:
    """Chinese title + sub-line (show-name + episode) — NOT bilingual."""
    assert is_distinct_language_pair(
        "摩斯探长前传",
        "第九季  第一集",
    ) is False


def test_pair_same_language_chinese_multiline_dialogue() -> None:
    """Multi-line Chinese dialogue (both sides Chinese) — NOT bilingual."""
    assert is_distinct_language_pair(
        "这是上句",
        "接着下句",
    ) is False


def test_pair_zh_ja_disambiguation() -> None:
    """Chinese vs Japanese (shared Han script) — distinct via kana presence."""
    assert is_distinct_language_pair("日本語の字幕", "这是中文字幕") is True


def test_pair_number_only_side_returns_false() -> None:
    """One side has no language signal (digits/punct only) → not bilingual."""
    assert is_distinct_language_pair("1972年", "1972") is False


def test_pair_empty_side_returns_false() -> None:
    """Empty side (user authored \\N at end of line) → not bilingual."""
    assert is_distinct_language_pair("我们都很看好你", "") is False
    assert is_distinct_language_pair("", "Hello") is False


def test_pair_punctuation_heavy_mixed() -> None:
    """A translator-note-style mixed line — each side has its own script.

    Example: ``"see pink elephants"是烂醉如泥的意思`` is semantically one mixed
    line (Chinese-dominant). When split as a pair with two Latin chars and
    all Chinese on the other, the pair IS distinct. We only take this path
    when an upstream splitter already decided to try splitting, so this
    behaviour is correct: respect the attempted split.
    """
    # Upstream would probably never pass this as a pair, but verify behaviour
    # is consistent: if the two sides are distinct languages, say yes.
    assert is_distinct_language_pair(
        '"see pink elephants"',
        "是烂醉如泥的意思",
    ) is True
