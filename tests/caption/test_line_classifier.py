"""Line-type classification for forced alignment skip decisions.

Covers the 7 categories a subtitle-group file mixes with dialogue:
  - staff_credit, branding, banner, title, sign, translator_note, karaoke

Fixtures come from Endeavour S09E01 bilingual ASS (real字幕组 data).
"""

from __future__ import annotations

import pytest

from lattifai.caption.parsers.text_parser import classify_line_type


# ============================================================================
# staff_credit — translator/editor role + name
# ============================================================================


def test_classify_role_plus_name() -> None:
    """Classic `翻译 Name` form (early in file)."""
    assert classify_line_type("翻译 草草", start=55.0) == "staff_credit"


def test_classify_combined_role_marker() -> None:
    """Real Endeavour form: `翻|校|监` combined role (first 120s)."""
    assert (
        classify_line_type(
            "翻|校|监        草草          后期      吉吉",
            start=54.5,
            ass_raw_text=(
                "{\\an4\\fad(500,500)\\fn方正大标宋_GBK\\bord0\\shad2"
                "\\fs18\\fsp-0.6\\pos(67,235)}翻|校|监   ..."
            ),
        )
        == "staff_credit"
    )


def test_classify_time_line_staff() -> None:
    """Time-axis crew: `时间轴` + multiple names."""
    assert (
        classify_line_type(
            "时间轴     chance-coco     树     无关风粤",
            start=59.5,
        )
        == "staff_credit"
    )


# ============================================================================
# branding — domain / group name / disclaimer
# ============================================================================


def test_classify_branding_domain() -> None:
    assert classify_line_type("www.yyets.com", start=5.0) == "branding"


# ============================================================================
# banner — recap / prologue screen text (\an1 + low y, recap keyword)
# ============================================================================


def test_classify_banner_recap() -> None:
    """Recap banner: `\\an1` + near-top `\\pos` + 『前情提要』keyword."""
    assert (
        classify_line_type(
            "《摩斯探长前传》前情提要",
            start=0.02,
            ass_raw_text=(
                "{\\fad(860,0)\\an1\\fs20\\fscx80\\bord0\\shad0"
                "\\fn方正华隶_GBK\\pos(18.9,254.933)}《摩斯探长前传》前情提要"
            ),
        )
        == "banner"
    )


# ============================================================================
# title — large animated font for show/episode title
# ============================================================================


def test_classify_title_show_name() -> None:
    """Title: `\\an8` + `\\fs35+` + `\\t(...\\fs)` animation."""
    assert (
        classify_line_type(
            "摩斯探长前传\n第九季  第一集",
            start=153.94,
            ass_raw_text=(
                "{\\an8\\fs35\\fscx80\\bord0\\shad0\\pos(191.3,160)"
                "\\t(0,4000,\\fs39)\\fn方正华隶_GBK}摩斯探长前传"
                "\\N{\\fs20\\t(0,5200,\\fs24)}第九季  第一集"
            ),
        )
        == "title"
    )


def test_classify_title_act_marker() -> None:
    """Sub-title: `序曲` / 『第X幕』 (大字号动画)."""
    assert (
        classify_line_type(
            "序曲",
            start=219.86,
            ass_raw_text=(
                "{\\an8\\fs35\\fscx80\\bord0\\shad0\\pos(191.3,160)"
                "\\t(0,4200,\\fsp1\\fs39)\\fn方正华隶_GBK}序曲"
            ),
        )
        == "title"
    )


# ============================================================================
# sign — on-screen text (\an8/9 + \bord1 + \b1 + small fs)
# ============================================================================


def test_classify_sign_location_label() -> None:
    """Sign: `\\an8\\bord1\\fs18\\b1` + short text."""
    assert (
        classify_line_type(
            "泰晤士河谷城堡门警局",
            start=596.82,
            ass_raw_text="{\\an8\\bord1\\fs18\\b1}泰晤士河谷城堡门警局",
        )
        == "sign"
    )


def test_classify_sign_single_word() -> None:
    """Sign: short single-word sign (e.g. `贱人` scrawled on mirror)."""
    assert (
        classify_line_type(
            "贱人",
            start=290.98,
            ass_raw_text="{\\an8\\bord1\\fs18\\b1}贱人",
        )
        == "sign"
    )


def test_classify_sign_with_fad() -> None:
    """Sign with fade animation: `\\fad(...)` + standard sign tags."""
    assert (
        classify_line_type(
            "萝丝·嘉兰",
            start=1423.98,
            ass_raw_text="{\\fad(860,180)\\an8\\bord1\\fs18\\b1}萝丝·嘉兰",
        )
        == "sign"
    )


# ============================================================================
# translator_note — commentary overlay (\an3/9 + small fs + \b0)
# ============================================================================


def test_classify_translator_note() -> None:
    """Translator note: `\\an3` + right-bottom `\\pos` + `\\fs16` + `\\b0`."""
    assert (
        classify_line_type(
            "摩斯所指的作家哈代\n和斯特兰奇所指的谐星哈迪英文名字相同",
            start=406.40,
            ass_raw_text=(
                "{\\an3\\fs16\\bord0\\fn微软雅黑\\b0\\pos(374,247)}"
                "摩斯所指的作家哈代\\N和斯特兰奇所指的谐星哈迪英文名字相同"
            ),
        )
        == "translator_note"
    )


# ============================================================================
# karaoke — \k / \kf / \ko tags
# ============================================================================


def test_classify_karaoke() -> None:
    assert (
        classify_line_type(
            "hello world",
            start=100.0,
            ass_raw_text="{\\k20}he{\\k15}llo {\\k30}world",
        )
        == "karaoke"
    )


def test_classify_karaoke_fill() -> None:
    assert (
        classify_line_type(
            "hello",
            start=100.0,
            ass_raw_text="{\\kf50}hello",
        )
        == "karaoke"
    )


# ============================================================================
# dialogue — must NOT be classified (return None)
# ============================================================================


def test_classify_dialogue_bilingual() -> None:
    """Standard bilingual dialogue cue — no classification."""
    assert (
        classify_line_type(
            "我们都很看好你\nWe all think a lot of you, you know?",
            start=3.65,
            ass_raw_text=(
                "我们都很看好你\\N{\\fn微软雅黑\\fs14}"
                "We all think a lot of you, you know?"
            ),
        )
        is None
    )


def test_classify_dialogue_plain() -> None:
    """Plain dialogue, no override tag."""
    assert classify_line_type("Hello, how are you?", start=60.0) is None


def test_classify_dialogue_with_speaker_fn_override() -> None:
    """Bilingual cue where a nested \\fs14 tag must NOT look like a sign."""
    assert (
        classify_line_type(
            "摩斯\nMorse.",
            start=334.61,
            ass_raw_text="摩斯\\N{\\fn微软雅黑\\fs14}Morse.",
        )
        is None
    )


# ============================================================================
# backwards compatibility — old two-arg signature still works
# ============================================================================


def test_classify_backward_compat_two_args() -> None:
    """Existing callers passing only (text, start) still get correct result."""
    assert classify_line_type("翻译 草草", start=55.0) == "staff_credit"
    assert classify_line_type("Normal dialogue", start=60.0) is None
