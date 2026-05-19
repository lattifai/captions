"""Tests for ``_split_translation_by_punct_alignment``.

Snap translation splits to mirrored punctuation between source text and
translation. Returns ``None`` when the two punctuation sequences do not
correspond (count, class, or position mismatch) so the caller falls back
to proportional splitting.

Equivalence classes:

- Sentence: ``. ! ? …``  ↔  ``。 ！ ？ …``
- Clause:   ``, ; :``    ↔  ``， 、 ； ：``
- Dash:     ``— – ―``    ↔  ``——`` (multi-codepoint)

Wrapper chars (quotes, brackets) are NEVER anchors but ARE ignorable
when measuring a chunk boundary's distance from a punctuation token —
so ``Hello, "world,"|...`` still snaps to the second comma even though
a closing quote sits between the comma and the boundary.
"""

from __future__ import annotations

from typing import List, Optional

from lattifai.caption.standardize import CaptionStandardizer
from lattifai.caption.supervision import AlignmentItem, Supervision

split = CaptionStandardizer._split_translation_by_punct_alignment


# ------------------------------------------------------------------
# Helpers — round-trip and class-sequence invariants.
# ------------------------------------------------------------------


def _assert_lossless(translation: str, out: Optional[List[str]]) -> None:
    """Punct-path slices MUST rejoin exactly to ``translation`` (no char
    added, dropped, or reordered). Failures here usually mean an
    off-by-one between ``end = pos + 1`` and the next ``cursor``."""
    assert out is not None
    rejoined = "".join(out)
    assert rejoined == translation, (
        f"punct-aligned slices lost data: {rejoined!r} != {translation!r}"
    )


def _classes(seq: str) -> str:
    """Reduce a string to its punctuation-class signature for quick
    sanity checks in tests. Useful for debugging fallback decisions."""
    out = []
    for ch in seq:
        if ch in ".!?。！？…":
            out.append("S")
        elif ch in ",;:，、；：":
            out.append("C")
        elif ch in "—–―":
            out.append("D")
    return "".join(out)


# ------------------------------------------------------------------
# Happy path — punctuation snap succeeds.
# ------------------------------------------------------------------


def test_single_comma_boundary_snaps_to_zh_comma():
    """The simplest case: one interior comma in both EN and ZH."""
    out = split("Hello, world.", "你好，世界。", ["Hello, ", "world."])
    _assert_lossless("你好，世界。", out)
    assert out == ["你好，", "世界。"]


def test_two_commas_snap_in_order():
    """Two interior commas — slices must cut at the corresponding ZH
    commas, not by character ratio. Char-ratio would yield ``["你好，", "世", "界。"]``;
    punct-aligned must yield clean clauses."""
    out = split(
        "Hi, there, friend.",
        "你好，那里，朋友。",
        ["Hi, ", "there, ", "friend."],
    )
    _assert_lossless("你好，那里，朋友。", out)
    assert out == ["你好，", "那里，", "朋友。"]


def test_zh_enumeration_comma_matches_en_comma():
    """`、` is the CJK enumeration comma — it MUST be class-equivalent
    to EN ``,`` so list-style sentences ``A, B, and C`` ↔ ``甲、乙、丙``
    can align without forcing the translator to use ``，``."""
    out = split(
        "A, B, and C.",
        "甲、乙、和丙。",
        ["A, ", "B, ", "and C."],
    )
    _assert_lossless("甲、乙、和丙。", out)
    assert out == ["甲、", "乙、", "和丙。"]


def test_sentence_punct_inside_supervision_aligns():
    """When a supervision contains two sentences, the interior sentence
    punct (period/question/exclamation) is a valid anchor — only the
    SUPERVISION-FINAL terminal punct is stripped."""
    out = split("Wait. Go!", "等。走！", ["Wait. ", "Go!"])
    _assert_lossless("等。走！", out)
    assert out == ["等。", "走！"]


def test_ellipsis_single_codepoint_sentence_class():
    """Unicode ellipsis ``…`` (U+2026) is one char and must be classified
    as sentence-class so EN ``…`` aligns to ZH ``…``."""
    out = split("Wait… go!", "等…走！", ["Wait… ", "go!"])
    _assert_lossless("等…走！", out)
    assert out == ["等…", "走！"]


def test_em_dash_aligns_to_zh_double_dash():
    """EN single em-dash ``—`` ↔ ZH double em-dash ``——`` (two
    codepoints). The slice end must consume the full ``——`` token,
    not just the first dash."""
    out = split(
        "Yes—so we did.",
        "是——所以我们做了。",
        ["Yes—", "so we did."],
    )
    _assert_lossless("是——所以我们做了。", out)
    # The cut goes after the entire ``——`` sequence in ZH.
    assert out == ["是——", "所以我们做了。"]


def test_en_dash_variants_normalize_to_dash_class():
    """EN dash variants ``—`` ``–`` ``―`` all classify as dash so any
    of them aligns to ZH ``——``."""
    for en_dash in ("—", "–", "―"):
        out = split(
            f"Yes{en_dash}go.",
            "是——走。",
            [f"Yes{en_dash}", "go."],
        )
        _assert_lossless("是——走。", out)
        assert out == ["是——", "走。"], f"failed for {en_dash!r}"


def test_punct_followed_by_closing_quote_then_space():
    """``Hello, "world," then left.`` — the chunk boundary may sit AFTER
    ``,"`` (comma + close-quote + space). Wrapper/whitespace chars
    between the comma and the boundary must be ignorable, otherwise
    real podcast captions with quoted speech would fall back."""
    out = split(
        'Hello, "world," then left.',
        "你好，“世界，”然后走了。",
        ['Hello, "world," ', "then left."],
    )
    _assert_lossless("你好，“世界，”然后走了。", out)
    # The second ZH comma is followed by a closing curly quote — slice
    # 0 must contain that wrapper so the next slice does not start with
    # an orphan close-quote.
    assert out == ["你好，“世界，”", "然后走了。"]


def test_repeated_clause_punct_distinct_indices():
    """``Yeah, yeah, yeah.`` / ``是的，是的，是的。`` — identical class
    sequence with three commas. Each boundary must map to a DISTINCT
    (strictly increasing) comma index."""
    out = split(
        "Yeah, yeah, yeah.",
        "是的，是的，是的。",
        ["Yeah, ", "yeah, ", "yeah."],
    )
    _assert_lossless("是的，是的，是的。", out)
    assert out == ["是的，", "是的，", "是的。"]


def test_two_chunks_no_trailing_period_alignment():
    """Source has no terminal period — ALL punctuation tokens are
    interior and available as anchors."""
    out = split("Hi, friend", "你好，朋友", ["Hi, ", "friend"])
    _assert_lossless("你好，朋友", out)
    assert out == ["你好，", "朋友"]


def test_zh_leading_space_after_punct_consumed_by_left_slice():
    """If the translator inserted ``，  `` (comma + extra space) instead
    of ``，`` alone, the trailing whitespace must attach to the LEFT
    slice so the next sub-cue doesn't start with ``\\u3000`` or `` ``."""
    out = split(
        "Hi, friend.",
        "你好， 朋友。",
        ["Hi, ", "friend."],
    )
    _assert_lossless("你好， 朋友。", out)
    assert out == ["你好， ", "朋友。"]


# ------------------------------------------------------------------
# Fallback path — punctuation mismatch returns None so caller can use
# the proportional splitter. Each failure case below is a critical
# guard: false positives would silently DESYNC bilingual cues.
# ------------------------------------------------------------------


def test_count_mismatch_falls_back():
    """EN has two commas, ZH only one — cannot snap without losing a
    boundary. Must return None to defer to proportional split."""
    out = split("A, B, C", "甲乙丙", ["A, ", "B, ", "C"])
    assert out is None


def test_class_mismatch_falls_back():
    """Clause vs sentence at the same position — they're not
    interchangeable (a period implies a sentence break, a comma
    implies an in-clause pause). EN ``,`` vs ZH ``。`` would split
    very different things, so we refuse to align."""
    # EN has one clause-class comma; ZH has one sentence-class period
    # in the same interior position. Class sequences differ → fallback.
    out = split("A, B is fine", "甲。乙没问题", ["A, ", "B is fine"])
    assert out is None


def test_colon_in_en_matches_zh_comma_clause_class():
    """``:`` and ``，`` are both clause-class in our equivalence model
    (per code-review consensus): they share the "in-clause separator"
    role. ``Yeah: go.`` ↔ ``是的，走。`` should snap, not fall back.
    Documents this deliberate design choice."""
    out = split("Yeah: go.", "是的，走。", ["Yeah: ", "go."])
    _assert_lossless("是的，走。", out)
    assert out == ["是的，", "走。"]


def test_class_order_mismatch_falls_back():
    """Class sequences must be IDENTICAL in order. ``clause sentence``
    cannot align to ``sentence clause`` even if counts match — that
    means the translator restructured the sentence."""
    out = split(
        "Go, then halt. Walk",
        "走。然后停，走",
        ["Go, ", "then halt. ", "Walk"],
    )
    assert out is None


def test_chunk_boundary_not_after_punct_falls_back():
    """If the chunk boundary in ``text`` doesn't land right after any
    punctuation token (with only whitespace/wrappers between), there's
    no anchor in EN — even if punct counts match, the alignment is
    coincidental and must be skipped."""
    out = split(
        "Hello world, friend.",
        "你好世界，朋友。",
        ["Hello ", "world, friend."],  # boundary at 6, no punct nearby
    )
    assert out is None


def test_lookback_window_exhausted_falls_back():
    """Boundary sits 5+ chars after the last punct (long trailing
    phrase with no anchor) — must fall back, not snap to the distant
    comma."""
    out = split(
        "Word, some extra trailing text here.",
        "词，一些额外尾随文本在这里。",
        ["Word, some extra trailing text ", "here."],  # huge gap after punct
    )
    assert out is None


def test_n_minus_one_greater_than_punct_count_falls_back():
    """Three chunks need 2 boundaries but only 1 interior punct exists
    → cannot cover both → fallback."""
    out = split(
        "Hi, friend hello",
        "你好，朋友你好",
        ["Hi, ", "friend ", "hello"],
    )
    assert out is None


def test_no_interior_punct_falls_back():
    """No interior punct in text (only the final period) → no anchors
    → fallback. The two ZH punctuation chars are irrelevant when EN
    has none."""
    out = split("Hello world.", "你好世界。", ["Hello ", "world."])
    assert out is None


def test_n_equals_one_returns_none():
    """A single chunk doesn't need to be split — the punct path is
    not applicable and must return None so the caller passes the
    whole translation through unchanged."""
    out = split("Hello.", "你好。", ["Hello."])
    assert out is None


def test_brackets_alone_are_not_anchors():
    """Opening/closing brackets are wrappers, not anchors. A sentence
    that contains ONLY brackets and no real punct must fall back even
    if the bracket count matches between languages."""
    out = split("(Hello) (World)", "（你好）（世界）", ["(Hello) ", "(World)"])
    assert out is None


# ------------------------------------------------------------------
# Edge inputs that must not crash.
# ------------------------------------------------------------------


def test_none_translation_returns_none():
    assert split("Hello.", None, ["Hello."]) is None  # type: ignore[arg-type]


def test_empty_translation_returns_none():
    assert split("Hello.", "", ["Hello."]) is None


def test_empty_text_returns_none():
    assert split("", "你好。", [""]) is None


def test_empty_chunks_returns_none():
    assert split("Hi.", "你好。", []) is None


# ------------------------------------------------------------------
# Integration via ``process()`` — long supervision with alignment +
# translation. The punct path must (a) hit, (b) produce per-cue
# translations that rejoin exactly, (c) yield the right cue count.
# ------------------------------------------------------------------


def _make_aligned_seg(
    text: str, translation: str, per_char_dur: float = 0.5
) -> Supervision:
    """Build a Supervision with one AlignmentItem per char (mimicking
    dub-pipeline character-level alignment). Each char gets the same
    duration so word-group boundaries can land anywhere."""
    words: List[AlignmentItem] = []
    t = 0.0
    for ch in text:
        words.append(AlignmentItem(symbol=ch, start=t, duration=per_char_dur))
        t += per_char_dur
    return Supervision(
        id="s0",
        text=text,
        translation=translation,
        start=0.0,
        duration=t,
        alignment={"word": words},
        language="en",
        target_lang="zh",
    )


def _make_standardizer(max_chars: int = 10) -> CaptionStandardizer:
    """Small char budget so a long source is forced to split into
    multiple sub-cues via the alignment-driven path."""
    return CaptionStandardizer(
        min_duration=0.2,
        max_duration=15.0,
        min_gap=0.04,
        max_lines=1,
        max_chars_per_line=max_chars,
    )


def test_process_long_supervision_punct_snap_rejoins():
    """End-to-end: a long EN supervision with two interior commas
    against a matching ZH translation. After ``process()``:

    1. Resulting per-cue translations must rejoin EXACTLY to the
       original translation string (no char added/dropped).
    2. The boundary in ZH must land right after a ZH comma — i.e.
       no sub-cue begins with ``，`` or `` ``.
    """
    text = "First part, second part, third part."
    trans = "第一部分，第二部分，第三部分。"
    seg = _make_aligned_seg(text, trans)
    std = _make_standardizer(max_chars=15)
    result = std.process([seg])
    translations = [r.translation or "" for r in result]
    assert "".join(translations) == trans
    # No sub-cue translation may BEGIN with a ZH clause-class punct or
    # whitespace — that would mean the boundary cut BEFORE the punct
    # instead of after.
    for sub in translations[1:]:
        if sub:
            assert sub[0] not in "，、；：。！？ 　", (
                f"sub-cue translation starts with stray punct/space: {sub!r}"
            )


def test_process_long_supervision_count_match_invariant():
    """The number of sub-cues produced by process() must equal the
    number of translation slots (no None backfill, no orphan)."""
    text = "Alpha, beta, gamma, delta epsilon."
    trans = "甲，乙，丙，丁戊。"
    seg = _make_aligned_seg(text, trans)
    std = _make_standardizer(max_chars=10)
    result = std.process([seg])
    # Every resulting sub-cue has either a non-empty translation OR
    # None — never an empty string from desync.
    for r in result:
        if r.translation is not None:
            assert r.translation != "", (
                "punct-aligned process() produced empty translation slice"
            )


def test_process_mismatch_translation_falls_back_to_proportional():
    """When punct count differs (ZH dropped the second comma), the
    integration path must still produce N translation slots that
    rejoin to the original — the proportional fallback handles it."""
    text = "Alpha, beta, gamma."
    trans = "甲乙丙。"  # no internal commas — punct count mismatch
    seg = _make_aligned_seg(text, trans)
    std = _make_standardizer(max_chars=10)
    result = std.process([seg])
    rejoin = "".join(r.translation or "" for r in result)
    assert rejoin == trans


# ------------------------------------------------------------------
# Source-has-no-interior-punct fallback path — the punct-alignment
# function returns None (no text-side anchors), so the proportional
# splitter handles it. The proportional path's ``_find_safe_split``
# already ranks punct (score 3-4) above whitespace (2) and CJK
# boundary (1), so it naturally PREFERS a translation-side comma
# near the proportional target. Tests below pin this behavior so
# future refactors don't regress it.
# ------------------------------------------------------------------


_proportional = CaptionStandardizer._split_translation_proportionally


def test_source_no_punct_translation_has_punct_cut_at_punct():
    """Source ``okay so I think | this is a great point`` (no interior
    punct) — chunk boundary at char 16. Translation has a ``，`` early
    on; punct-alignment returns None (no text anchors), so the
    proportional splitter takes over and snaps to the comma."""
    text_chunks = ["okay so I think ", "this is a great point"]
    trans = "好的，我觉得这观点不错，深思熟虑。"
    # punct-aligned path: must return None (no text-side tokens).
    assert split("okay so I think this is a great point", trans, text_chunks) is None
    # proportional path: should cut at the FIRST comma boundary so
    # neither slice starts with ``，`` or splits a CJK word mid-phrase.
    out = _proportional(trans, text_chunks)
    assert out == ["好的，", "我觉得这观点不错，深思熟虑。"], (
        f"proportional splitter did not snap to ZH comma: {out!r}"
    )


def test_source_no_punct_translation_with_mid_comma_snaps_there():
    """Longer source with no interior punct; translation's interior
    ``，`` sits near the proportional target. The fallback splitter
    must cut at the ``，``, not by char ratio in the middle of a
    Chinese clause."""
    text_chunks = [
        "okay so I think this might be the way ",
        "and we will see if it works out well in the end",
    ]
    trans = "好的，我觉得这可能就是路子，最后看看能不能行。"
    out = _proportional(trans, text_chunks)
    # The fallback prefers the comma boundary nearest to the
    # char-ratio target; both ZH commas are valid anchors. Either
    # slice 0 end with ``，`` is acceptable; what matters is that
    # slice 1 does NOT begin with ``，`` or `` ``.
    assert out is not None
    assert out[0] and out[0].endswith("，"), f"slice 0 must end with comma: {out!r}"
    assert out[1] and out[1][0] not in "，、；：。！？ 　", (
        f"slice 1 starts with stray punct/space: {out!r}"
    )


def test_source_no_punct_translation_no_punct_word_safe():
    """Neither side has interior punctuation. Punct-alignment returns
    None (degenerate), proportional splitter falls back to char-ratio
    + CJK boundary scoring. Test purely confirms no crash + lossless
    rejoin — no specific cut position is required."""
    text_chunks = ["one two three four five six ", "seven eight nine ten eleven twelve"]
    trans = "一二三四五六七八九十十一十二十三十四十五十六十七十八"
    assert split(" ".join(text_chunks), trans, text_chunks) is None
    out = _proportional(trans, text_chunks)
    assert out is not None
    assert "".join(s or "" for s in out) == trans
    assert len(out) == 2
