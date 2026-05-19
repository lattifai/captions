"""Tests for VTT cue-head '- ' speaker-change marker handling in SentenceSplitter.

Background
----------
VTT broadcast captions encode new-speaker turns with a leading dash:

    00:31.800 --> 00:35.730
    - FFmpeg is probably one of the biggest CPU users in the world.

Before this fix, ``SentenceSplitter._segment_supervisions`` joined cue
supervisions with ``" "`` and fed the long string to wtpsplit. wtpsplit
treats ``" - "`` as a connective (em-dash style insertion) and draws
the sentence boundary AFTER the dash, leaving it stranded as a trailing
token on the previous sentence ("...right? -"). Validated on a Lex
Fridman 3402-sup episode: 623 supervisions wrongly ended with ``-``.

Fix
---
``_extract_cue_head_speaker_change`` runs BEFORE Phase 1 of split_sentences:
strips leading ``- `` from supervision text and records
``custom['vtt_speaker_change'] = True``. Input list is never mutated
(operates on fastcopy'd objects).

These tests verify the static extraction logic + end-to-end behaviour
without invoking the heavy wtpsplit model (those are covered separately).
"""

from typing import List, Optional

import pytest

from lattifai.caption import SentenceSplitter, Supervision


def _sup(
    idx: int,
    text: str,
    speaker: Optional[str] = None,
    custom: Optional[dict] = None,
) -> Supervision:
    return Supervision(
        id=f"sup-{idx}",
        recording_id="rec",
        start=float(idx),
        duration=1.0,
        channel=0,
        text=text,
        speaker=speaker,
        custom=custom,
    )


# ---------------------------------------------------------------------------
# Static extraction: input mutation isolation + correctness
# ---------------------------------------------------------------------------


def test_extracts_leading_dash_and_records_marker():
    """Canonical case: '- FFmpeg ...' → 'FFmpeg ...' + custom marker."""
    sups = [_sup(0, "- FFmpeg is probably the biggest.")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)

    assert result[0].text == "FFmpeg is probably the biggest."
    assert result[0].custom == {"vtt_speaker_change": True}


def test_does_not_mutate_input_supervisions():
    """Input list and Supervision objects must be untouched (no side effects)."""
    sups = [_sup(0, "- FFmpeg is great.")]
    input_text = sups[0].text
    input_custom = sups[0].custom

    SentenceSplitter._extract_cue_head_speaker_change(sups)

    assert sups[0].text == input_text
    assert sups[0].custom is input_custom


def test_skips_supervisions_without_leading_dash():
    """Most sups (no dash) → pass through unchanged (same object identity OK)."""
    sups = [
        _sup(0, "Hello world."),
        _sup(1, "Another sentence."),
    ]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)

    # text + custom unchanged
    assert result[0].text == "Hello world."
    assert result[1].text == "Another sentence."
    assert result[0].custom is None
    assert result[1].custom is None


def test_does_not_strip_inline_or_word_internal_dash():
    """Only LEADING '- ' (dash + space) matches. Inline / word-internal stay."""
    cases = [
        "well-known feature",      # word-internal
        "Mm-hmm.",                  # word-internal
        "text - more text",         # inline em-dash style
        "compiler-",                # word-final cut-off
        "-FFmpeg without space",    # no space after dash
        "  - leading whitespace",   # leading WHITESPACE before dash, not just '- '
    ]
    sups = [_sup(i, t) for i, t in enumerate(cases)]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)

    for orig, out in zip(cases, result):
        assert out.text == orig
        assert out.custom is None


def test_preserves_existing_custom_fields():
    """Merging into existing custom dict must not drop other keys."""
    sups = [
        _sup(0, "- FFmpeg is great.", custom={"vtt_align": "left", "color": "red"}),
    ]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)

    assert result[0].text == "FFmpeg is great."
    assert result[0].custom == {
        "vtt_align": "left",
        "color": "red",
        "vtt_speaker_change": True,
    }


def test_preserves_speaker_label_when_present():
    """A cue with both '- ' and an explicit speaker label keeps both."""
    sups = [_sup(0, "- FFmpeg is great.", speaker="Jean-Baptiste")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)

    assert result[0].text == "FFmpeg is great."
    assert result[0].speaker == "Jean-Baptiste"
    assert result[0].custom == {"vtt_speaker_change": True}


def test_idempotent_on_already_extracted_supervisions():
    """Running twice produces the same output (no double-strip, no double-mark)."""
    sups = [_sup(0, "- FFmpeg is great.")]
    pass1 = SentenceSplitter._extract_cue_head_speaker_change(sups)
    pass2 = SentenceSplitter._extract_cue_head_speaker_change(pass1)

    assert pass2[0].text == "FFmpeg is great."
    assert pass2[0].custom == {"vtt_speaker_change": True}


def test_empty_text_passes_through():
    """Defensive: text='' or text=None → no error, no change."""
    sups = [_sup(0, ""), _sup(1, "normal")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    assert result[0].text == ""
    assert result[1].text == "normal"


def test_handles_empty_supervisions_list():
    """Defensive: [] → []."""
    assert SentenceSplitter._extract_cue_head_speaker_change([]) == []


# ---------------------------------------------------------------------------
# End-to-end: split_sentences applies the extraction before phase 1
# ---------------------------------------------------------------------------


def _has_isolated_trailing_dash(text: str) -> bool:
    """Match the same rule used by ai-podcast-pipeline's fix-dash script."""
    s = (text or "").rstrip()
    if not s.endswith("-"):
        return False
    if len(s) == 1:
        return True
    return s[-2].isspace() or s[-2] in '.,!?;:"\'”。，？！、'


def test_split_sentences_eliminates_trailing_dash_pollution():
    """Reproduces the Lex Fridman pattern at miniature scale.

    Before fix: wtpsplit would emit "...right? -" + "FFmpeg ..." after
    joining "...right?" and "- FFmpeg ..." with a space.
    After fix: the dash is removed pre-split, recorded as custom marker.
    No output supervision has an isolated trailing dash.
    """
    splitter = SentenceSplitter()
    supervisions = [
        _sup(0, "But that's okay, right?"),
        _sup(1, "- FFmpeg is probably the biggest CPU users in the world."),
        _sup(2, "Mm-hmm."),
        _sup(3, "- And just this one has two hundred and forty thousand."),
    ]

    result = splitter.split_sentences(supervisions)

    # No output sup should end with an isolated trailing dash.
    offenders = [(i, s.text) for i, s in enumerate(result) if _has_isolated_trailing_dash(s.text)]
    assert offenders == [], f"isolated trailing dash leaked through: {offenders}"

    # The original dash-bearing sups' marker should be preserved on some
    # output sup that overlaps with them in time. We don't assert exact
    # output count (wtpsplit may split sentences further) — just that the
    # marker survives somewhere.
    custom_marked = [s for s in result if (s.custom or {}).get("vtt_speaker_change")]
    assert len(custom_marked) >= 1, "vtt_speaker_change marker was lost through splitting"


def test_split_sentences_does_not_alter_dashless_input():
    """No-op on inputs with no '- ' markers — output content + count parity."""
    splitter = SentenceSplitter()
    supervisions = [
        _sup(0, "Hello world."),
        _sup(1, "Another sentence."),
    ]

    result = splitter.split_sentences(supervisions)

    full_text_in = " ".join(s.text for s in supervisions)
    full_text_out = " ".join(s.text for s in result)
    assert full_text_in == full_text_out
    for s in result:
        # No spurious vtt_speaker_change markers added.
        assert not (s.custom or {}).get("vtt_speaker_change")


# ---------------------------------------------------------------------------
# Marker preservation through Phase 4 (_distribute_time_info)
# ---------------------------------------------------------------------------


def test_marker_reaches_output_supervision_at_correct_position():
    """The vtt_speaker_change marker must end up on the OUTPUT supervision
    that contains text from the originally-marked input.

    With "...right? - FFmpeg ..." → cleaned to "...right? FFmpeg ...",
    wtpsplit boundary lands between '?' and 'FFmpeg', so the 'FFmpeg' sup
    is the one that should carry the marker.
    """
    splitter = SentenceSplitter()
    supervisions = [
        _sup(0, "That's okay, right?"),
        _sup(1, "- FFmpeg is great."),
    ]

    result = splitter.split_sentences(supervisions)

    # The output sup whose text comes from sup 1 should have the marker.
    # (We accept that wtpsplit may split sentences differently — find the
    # sup whose text starts with "FFmpeg".)
    ffmpeg_sups = [s for s in result if s.text.lstrip().startswith("FFmpeg")]
    assert ffmpeg_sups, f"expected an output sup starting with 'FFmpeg', got {[s.text for s in result]}"
    assert (ffmpeg_sups[0].custom or {}).get("vtt_speaker_change") is True, (
        f"marker missing on FFmpeg sup: custom={ffmpeg_sups[0].custom}"
    )

    # The PREVIOUS sup ("That's okay, right?") must NOT have the marker.
    right_sups = [s for s in result if "right" in s.text.lower()]
    assert right_sups
    for s in right_sups:
        assert not (s.custom or {}).get("vtt_speaker_change"), (
            f"marker leaked to wrong sup: {s.text!r} custom={s.custom}"
        )


def test_timing_preserved_after_dash_extraction():
    """Output supervisions' time spans must still cover the input range.

    The dash-strip shortens sup.text by 2 chars; _distribute_time_info
    uses char-position proportions for timing. We verify the final sup
    timings still align with input boundaries (start of first input,
    end of last input)."""
    splitter = SentenceSplitter()
    supervisions = [
        Supervision(id="s0", recording_id="r", start=10.0, duration=2.0,
                    channel=0, text="Hello world."),
        Supervision(id="s1", recording_id="r", start=12.5, duration=3.0,
                    channel=0, text="- FFmpeg is great."),
    ]

    result = splitter.split_sentences(supervisions)

    # Earliest output start should equal earliest input start.
    assert result[0].start == pytest.approx(supervisions[0].start, abs=0.01)
    # Latest output end should equal latest input end.
    last_end_in = supervisions[-1].start + supervisions[-1].duration
    last_end_out = result[-1].start + result[-1].duration
    assert last_end_out == pytest.approx(last_end_in, abs=0.5)


def test_speaker_label_preserved_through_dash_extraction():
    """A sup with explicit speaker label keeps the label after Phase 4."""
    splitter = SentenceSplitter()
    supervisions = [
        _sup(0, "Question?"),
        _sup(1, "- Yes, exactly.", speaker="Alice"),
    ]

    result = splitter.split_sentences(supervisions)

    yes_sups = [s for s in result if "Yes" in s.text or "exactly" in s.text]
    assert yes_sups
    assert yes_sups[0].speaker == "Alice"


# ---------------------------------------------------------------------------
# Real-world fixture replay: Lex Fridman pattern
# ---------------------------------------------------------------------------


# These are real cue texts from the Lex Fridman / Jean-Baptiste Kempf episode
# (data/LexFridman/2026-05-06_nepKKz-MzFM in ai-podcast-pipeline) where pre-fix
# behaviour produced isolated trailing dashes at 623 supervisions. Inputs are
# the raw VTT-parsed cue contents.
LEX_FRIDMAN_PATTERN_CUES = [
    "- The important is, is your code good? We care about",
    "excellent code. We don't care who you are. Like maybe you're a dog. I don't",
    "care, right? I don't care where you come from. I need to look at your code.",
    "Oh, yeah, but I'm an engineer at this very large company in",
    "Italy, in Germany, in the US. We don't care. We care about the",
    "quality of your code because this is what defines our community",
    "and which means that we have a lot of people who contribute who are some very different",
    "backgrounds and very introverted. Sure. But that's okay, right?",
    "- FFmpeg is probably one of the biggest CPU users in the world. Everything we've just",
    "said in the past couple of minutes, every sentence is someone's",
    "lifetime's work. There are books about every sentence. So the level of complexity",
    "in many cases is inordinate.",
    "- FFmpeg has one hundred thousand lines of assembly for all the codecs.",
    "- For all codecs. Mm-hmm.",
    "- And just this one has two hundred and forty thousand. Every cycle",
    "matters. We are talking about probably three billion",
]


def test_lex_fridman_pattern_no_isolated_trailing_dash():
    """Regression: ai-podcast-pipeline Lex Fridman 3402-sup episode.

    Before fix: 623 of 3332 output sups had isolated trailing dash.
    After fix: 0.

    This test uses a 16-cue slice replicating the same pattern. Asserts
    NO output sup ends with isolated trailing dash, and the originally-
    marked cues' markers survive to corresponding output sups.
    """
    splitter = SentenceSplitter()
    supervisions = [
        Supervision(id=f"s{i}", recording_id="r",
                    start=float(i) * 4.0, duration=4.0,
                    channel=0, text=text)
        for i, text in enumerate(LEX_FRIDMAN_PATTERN_CUES)
    ]
    cue_head_count_in = sum(1 for s in supervisions if s.text.startswith("- "))
    assert cue_head_count_in >= 4, "fixture should include several '- ' cues"

    result = splitter.split_sentences(supervisions)

    # No isolated trailing dash anywhere in output.
    offenders = [(i, s.text) for i, s in enumerate(result)
                 if _has_isolated_trailing_dash(s.text)]
    assert offenders == [], f"trailing dashes leaked: {offenders}"

    # The marker should appear on output sups (one per originally-marked
    # input cue — may be slightly more if wtpsplit splits further inside).
    marked = [s for s in result if (s.custom or {}).get("vtt_speaker_change")]
    assert len(marked) >= cue_head_count_in, (
        f"expected ≥{cue_head_count_in} marker-bearing output sups, got {len(marked)}"
    )

    # Sanity: full content (modulo dashes/whitespace) is preserved.
    full_in = " ".join(s.text.lstrip("- ").strip() for s in supervisions)
    full_out = " ".join(s.text for s in result)
    # Allow some whitespace normalisation differences.
    assert len(full_out.replace(" ", "")) >= 0.95 * len(full_in.replace(" ", ""))


# ---------------------------------------------------------------------------
# Multi-language: CJK supervisions with leading dash
# ---------------------------------------------------------------------------


def test_chinese_supervision_with_leading_dash():
    """CJK content after the '- ' marker should be cleaned the same way."""
    sups = [_sup(0, "- 你好世界。")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)

    assert result[0].text == "你好世界。"
    assert result[0].custom == {"vtt_speaker_change": True}


def test_japanese_supervision_with_leading_dash():
    sups = [_sup(0, "- こんにちは。")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    assert result[0].text == "こんにちは。"


def test_marker_does_not_strip_dash_in_emoji_or_unicode_prefix():
    """Defensive: only literal '- ' (ASCII hyphen + ASCII space) matches.

    em-dash, en-dash, fullwidth dash etc. are real prose, not VTT markers."""
    cases = [
        ("— FFmpeg is great", "— FFmpeg is great"),  # em dash
        ("– FFmpeg is great", "– FFmpeg is great"),  # en dash
        ("－ FFmpeg is great", "－ FFmpeg is great"),  # fullwidth dash
    ]
    sups = [_sup(i, t) for i, (t, _) in enumerate(cases)]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    for (orig, expected), out in zip(cases, result):
        assert out.text == expected
        assert out.custom is None


# ---------------------------------------------------------------------------
# All Supervision fields preserved through fastcopy
# ---------------------------------------------------------------------------


def test_all_supervision_fields_preserved_through_extraction():
    """fastcopy must preserve every field on the input supervision."""
    sup = Supervision(
        id="abc-123",
        recording_id="rec-xyz",
        start=42.5,
        duration=3.14,
        channel=2,
        text="- FFmpeg is great.",
        speaker="Alice",
        language="en",
        gender="female",
    )

    result = SentenceSplitter._extract_cue_head_speaker_change([sup])
    out = result[0]

    assert out.id == "abc-123"
    assert out.recording_id == "rec-xyz"
    assert out.start == 42.5
    assert out.duration == 3.14
    assert out.channel == 2
    assert out.text == "FFmpeg is great."  # stripped
    assert out.speaker == "Alice"
    assert out.language == "en"
    assert out.gender == "female"
    assert out.custom == {"vtt_speaker_change": True}


# ---------------------------------------------------------------------------
# Boundary positions: first / last sup
# ---------------------------------------------------------------------------


def test_first_sup_has_marker():
    """First sup of episode is allowed to carry a '- ' marker (intro)."""
    sups = [_sup(0, "- Welcome to the show."), _sup(1, "Today we discuss...")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    assert result[0].text == "Welcome to the show."
    assert result[0].custom == {"vtt_speaker_change": True}
    assert result[1].custom is None


def test_last_sup_has_marker():
    """Last sup of episode can carry a '- ' marker (rare, e.g. closing exchange)."""
    sups = [_sup(0, "Goodbye now."), _sup(1, "- See you next week!")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    assert result[1].text == "See you next week!"
    assert result[1].custom == {"vtt_speaker_change": True}


# ---------------------------------------------------------------------------
# Multiple consecutive '-' sups (every cue is a new speaker — rapid exchange)
# ---------------------------------------------------------------------------


def test_consecutive_dash_sups_all_get_marked():
    """Rapid back-and-forth: every cue starts with '- '. Each must be cleaned."""
    sups = [
        _sup(0, "- Hello."),
        _sup(1, "- Hi."),
        _sup(2, "- How are you?"),
        _sup(3, "- Good thanks."),
    ]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    for s in result:
        assert not s.text.startswith("- ")
        assert (s.custom or {}).get("vtt_speaker_change") is True


# ---------------------------------------------------------------------------
# Edge: text that is ONLY '- ' (whole-content junk)
# ---------------------------------------------------------------------------


def test_dash_only_text_becomes_empty_with_marker():
    """Edge: text='- ' (dash + space, no content) → text='' + marker."""
    sups = [_sup(0, "- ")]
    result = SentenceSplitter._extract_cue_head_speaker_change(sups)
    assert result[0].text == ""
    assert result[0].custom == {"vtt_speaker_change": True}
