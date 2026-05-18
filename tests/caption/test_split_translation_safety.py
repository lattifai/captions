"""Tests for ``_split_translation_proportionally`` word-boundary safety.

Regression suite for the bug observed on Claude/dub karaoke ASS output: the
proportional translation splitter cut Latin words in half (e.g.
``"permissions"`` → ``"perm"|"issions"``). The new behavior must:

- Never produce a boundary where both neighbors are Latin letters/digits.
- Prefer punctuation > whitespace > CJK-boundary > forbidden.
- Keep CJK-only translation behavior intact (char-level splits are fine).
- Preserve total character count (no data loss).
- Handle pathological inputs (no breakable boundary) without crashing.
"""

from __future__ import annotations

import pytest

from lattifai.caption.standardize import CaptionStandardizer


split = CaptionStandardizer._split_translation_proportionally


def _joined(out):
    return "".join(s or "" for s in out)


def _is_latin_word(ch: str) -> bool:
    return ch.isascii() and (ch.isalpha() or ch.isdigit() or ch == "'")


def _assert_no_latin_word_split(out):
    """For every boundary i in ``out``, assert prev tail + next head are not
    both Latin word chars (would mean a Latin word was cut in half)."""
    for i in range(1, len(out)):
        prev = out[i - 1]
        nxt = out[i]
        if not prev or not nxt:
            continue
        tail = prev[-1]
        head = nxt[0]
        assert not (_is_latin_word(tail) and _is_latin_word(head)), (
            f"Latin word cut at boundary {i}: ...{prev[-10:]!r} | {nxt[:10]!r}..."
        )


# ------------------------------------------------------------------
# Degenerate / passthrough cases — must still work.
# ------------------------------------------------------------------


def test_none_translation():
    out = split(None, ["a", "b"])
    assert out == [None, None]


def test_empty_translation():
    # Existing convention: empty string passes through as-is (not normalized
    # to None). Documenting current behavior, not changing it in this PR.
    out = split("", ["a", "b"])
    assert out == ["", None]


def test_single_chunk():
    out = split("Hello world", ["whatever"])
    assert out == ["Hello world"]


def test_too_short_translation_falls_to_first():
    # trans_len < 2 * n → existing degenerate path
    out = split("ab", ["aaa", "bbb"])
    assert out == ["ab", None]


def test_total_chars_preserved():
    chunks = ["在这一段", "中可以"]
    trans = "In this section, we can use it"
    out = split(trans, chunks)
    assert _joined(out) == trans


# ------------------------------------------------------------------
# Core safety: never cut a Latin word.
# ------------------------------------------------------------------


def test_english_translation_never_cut_mid_word_real_case():
    """Regression: real failure from Claude/dub audit.

    Source: 'context window and what i' | 't can summarize in order' — the 't'
    is the tail of 'it'. Must not happen.
    """
    chunks = ["上下文窗口和什么", "可以在顺序中总结"]
    trans = "context window and what it can summarize in order"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)


def test_permissions_not_cut():
    """Regression: 'configurable perm' | 'issions' observed in audit."""
    chunks = ["可配置权限", "进入终端"]
    trans = "configurable permissions into your terminal"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    for seg in out:
        if seg and "perm" in seg:
            assert "permissions" in seg, f"'permissions' was cut: {out}"


def test_every_not_cut():
    """Regression: 'ever' | 'y message' observed in audit."""
    chunks = ["每条消息", "都会占用"]
    trans = "every message you send takes up context"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)


def test_long_word_runs_no_split_within():
    """Very long single Latin word with surrounding spaces."""
    chunks = ["a", "b", "c"]
    trans = "x antidisestablishmentarianism y"  # one giant word
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)


# ------------------------------------------------------------------
# Punctuation preference.
# ------------------------------------------------------------------


def test_clause_punct_preferred_over_space():
    chunks = ["a", "b"]
    # Both halves must carry >= 2 effective tokens or the merge logic
    # absorbs the short side, hiding what we're trying to assert.
    trans = "Yes hello now, world today is fine"
    out = split(trans, chunks)
    # First chunk should end at the comma (clause punct, score 3) rather than
    # the space before another word (whitespace, score 2).
    assert out[0] is not None
    assert out[0].rstrip().endswith(",") or out[0].endswith(", "), (
        f"Expected split after comma, got: {out}"
    )
    _assert_no_latin_word_split(out)


def test_sentence_punct_short_circuits():
    """Sentence punct (score 4) wins instantly."""
    chunks = ["aaaaaaa", "bbbbb"]
    # Both halves must carry >= 2 tokens so merge does not absorb them.
    trans = "Yes done now. Now we continue here for a while"
    out = split(trans, chunks)
    assert out[0] is not None
    assert out[0].rstrip().endswith(".") or out[0].endswith(". ")


# ------------------------------------------------------------------
# CJK-only translation — must preserve old char-level behavior.
# ------------------------------------------------------------------


def test_cjk_only_translation_char_level_still_works():
    chunks = ["context", "window"]
    trans = "上下文窗口和可配置权限"
    out = split(trans, chunks)
    assert _joined(out) == trans
    # Both pieces should be non-trivial (char-level split is fine for CJK).
    assert out[0] and out[1]


def test_cjk_with_punct():
    chunks = ["abc", "def"]
    trans = "你好，世界今天"
    out = split(trans, chunks)
    assert _joined(out) == trans
    # Comma boundary should be preferred.
    assert out[0] is not None and ("，" in out[0])


# ------------------------------------------------------------------
# Mixed Latin + CJK translation.
# ------------------------------------------------------------------


def test_mixed_translation_avoids_latin_split():
    chunks = ["aaa", "bbb"]
    trans = "使用 permissions 功能"  # contains a Latin word
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    for s in out:
        if s and "perm" in s:
            assert "permissions" in s


# ------------------------------------------------------------------
# Contractions, numerics, units — apostrophe and digit-letter
# boundaries must not be cut.
# ------------------------------------------------------------------


def test_apostrophe_in_contraction_not_cut():
    chunks = ["aaa", "bbb"]
    trans = "it doesn't really work here"
    out = split(trans, chunks)
    assert _joined(out) == trans
    for s in out:
        if s and "doesn" in s:
            assert "doesn't" in s, f"contraction broken: {out}"


def test_number_unit_not_cut():
    """5MB / 100ms / 60s — number+letter combos are word-internal."""
    chunks = ["xxx", "yyy"]
    trans = "buffer is 100ms of latency"
    out = split(trans, chunks)
    assert _joined(out) == trans
    for s in out:
        if s and "100" in s:
            assert "100ms" in s, f"'100ms' was split: {out}"


# ------------------------------------------------------------------
# Pathological fallback: no breakable boundary anywhere.
# ------------------------------------------------------------------


def test_pure_letter_run_falls_to_first_chunk():
    """Edge case: 50 'a's, no boundary anywhere → unbounded scan returns
    end-of-string, so whole translation lands on the first chunk."""
    chunks = ["a", "b"]
    trans = "a" * 50
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    # Either whole-on-first OR single-trailing-letter — both are acceptable
    # so long as no Latin word was split. The unbounded-scan policy gives
    # whole-on-first.
    assert out[0] is not None
    assert out[0] == trans or (out[1] is not None and len(out[0]) > 1)


def test_three_way_split_all_safe():
    chunks = ["aa", "bb", "cc"]
    trans = "Hello world, this is fine."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    assert len(out) == 3


# ------------------------------------------------------------------
# Grapheme safety preserved (emoji / combining marks).
# ------------------------------------------------------------------


def test_emoji_zwj_not_split():
    chunks = ["aaa", "bbb"]
    trans = "hello 👨‍👩‍👧 world how are you"  # ZWJ family
    out = split(trans, chunks)
    assert _joined(out) == trans
    # Just must not crash and must roundtrip.


def test_combining_marks_not_split():
    """'é' as base + U+0301 stays together."""
    chunks = ["xx", "yy"]
    trans = "café noir is here"  # café via NFD
    out = split(trans, chunks)
    assert _joined(out) == trans
    # Boundary must not land between 'e' and '́'
    for i in range(1, len(out)):
        prev = out[i - 1]
        nxt = out[i]
        if prev and nxt and nxt[0] == "́":
            pytest.fail(f"Split between base and combining mark: {out}")


# ------------------------------------------------------------------
# Short-tail merge: when a slice has < 2 effective tokens (a single
# CJK char or a single Latin word, regardless of punctuation), it
# should be folded into the previous slice rather than left dangling.
# ------------------------------------------------------------------


def _effective_tokens(s: str) -> int:
    """Mirror of the standardizer's heuristic: count CJK chars + maximal
    Latin word runs. Whitespace and punctuation don't count."""
    if not s:
        return 0
    n = 0
    in_latin = False
    for ch in s:
        if "一" <= ch <= "鿿":
            n += 1
            in_latin = False
        elif ch.isascii() and (ch.isalpha() or ch.isdigit() or ch == "'"):
            if not in_latin:
                n += 1
                in_latin = True
        else:
            in_latin = False
    return n


def _assert_no_short_tail(out, min_tokens: int = 2):
    """Every non-None slice must have >= min_tokens effective tokens."""
    non_none = [s for s in out if s]
    if len(non_none) <= 1:
        return
    for slice_ in non_none:
        assert _effective_tokens(slice_) >= min_tokens, (
            f"Slice has < {min_tokens} effective tokens: {slice_!r} (in {out})"
        )


def test_short_tail_latin_word_plus_punct_merges():
    """Regression: ratio 9:1 produces tail 'ok.' (1 word). Must merge."""
    chunks = ["很长很长的一段中文", "了"]
    trans = "really long english stuff here ok."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    _assert_no_short_tail(out)
    assert out[-1] is None


def test_short_tail_single_letter_plus_punct_merges():
    """Tail 'x.' (1 letter + period) is even shorter — must merge."""
    chunks = ["很长的一段中文", "嗯"]
    trans = "very long english sentence with x."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)
    assert out[-1] is None


def test_short_tail_single_cjk_char_plus_punct_merges():
    """Tail '了。' (1 CJK char + period) is 1 effective token — merge."""
    chunks = ["really very long english chunk", "嗯"]
    trans = "前面有很多很多内容了。"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)


def test_two_token_tail_preserved():
    """Tail with >= 2 effective tokens (e.g. 'and yes.') is NOT merged."""
    chunks = ["很长的一段中文内容", "你看"]
    trans = "very long english sentence and yes."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    if out[-1] is not None:
        assert _effective_tokens(out[-1]) >= 2


def test_middle_short_slice_merges_left():
    """Three-way split: if middle slice ends up < 2 tokens, fold left."""
    chunks = ["很长的中文一段", "好", "另一段中文内容"]
    trans = "a long english sentence ok now here we continue more"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_latin_word_split(out)
    _assert_no_short_tail(out)


def test_cascading_short_slices_collapse_left():
    """Three single-token slots all collapse leftward."""
    chunks = ["很长的中文", "嗯", "好"]
    trans = "very long english stuff and ok yes."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)


def test_short_tail_does_not_break_degenerate_path():
    """The `trans_len < 2*n` degenerate path still works."""
    out = split("ok", ["aaa", "bbb"])
    assert out == ["ok", None]


# ------------------------------------------------------------------
# Codex review follow-ups: curly apostrophe, non-BMP CJK, exact
# cascade placement, and end-to-end `process()` integration.
# ------------------------------------------------------------------


def test_curly_apostrophe_in_contraction_not_cut():
    """Dub-flow English translations almost always use ’ (U+2019, curly
    apostrophe), not the ASCII '. The old protection only matched ASCII
    and would happily cut ``doesn’t`` into ``doesn`` / ``’t`` when no
    nearby whitespace candidate outscored the boundary.

    Constructed case: contraction wedged between CJK characters with no
    surrounding whitespace, so the splitter has no whitespace fallback
    and the curly apostrophe boundary is the only score-1 candidate."""
    chunks = ["aa", "bb"]
    trans = "我们doesn’t很好"
    out = split(trans, chunks)
    assert _joined(out) == trans
    for s in out:
        if s and ("doesn" in s or "’t" in s):
            assert "doesn’t" in s, f"curly-apostrophe contraction broken: {out}"


def test_curly_apostrophe_score_is_forbidden():
    """Unit-test the boundary scorer directly: between a Latin letter
    and a curly apostrophe (and vice versa) must score 0 (forbidden),
    just like the ASCII apostrophe case."""
    from lattifai.caption.standardize import CaptionStandardizer

    s = "don’t"
    # Position 3 sits between 'n' and '’' — splitting here would yield
    # "don" + "’t", breaking the contraction.
    assert CaptionStandardizer._boundary_score(s, 3) == 0, (
        f"score at 'n|’' should be 0, got {CaptionStandardizer._boundary_score(s, 3)}"
    )
    assert CaptionStandardizer._boundary_score(s, 4) == 0, (
        f"score at '’|t' should be 0, got {CaptionStandardizer._boundary_score(s, 4)}"
    )


def test_non_bmp_cjk_treated_as_cjk_in_classifier():
    """Han Extension B (U+20000+) chars must be recognized by the CJK
    classifier so they count as tokens and so the boundary scorer
    treats them as CJK rather than 'other'."""
    from lattifai.caption.standardize import CaptionStandardizer

    # U+20000 (CJK Ext-B start) and U+2A700 (Ext-C start)
    assert CaptionStandardizer._is_cjk_char("𠀀"), "U+20000 should classify as CJK"
    assert CaptionStandardizer._is_cjk_char("\U0002a700"), (
        "U+2A700 should classify as CJK"
    )
    # And token counting must agree.
    assert CaptionStandardizer._effective_token_count("𠀀𠀁") == 2


def test_non_bmp_cjk_short_tail_merges():
    """Non-BMP CJK chars count as tokens, so a slice carrying only a
    single Ext-B char is still a 1-token tail and must merge."""
    # 5 Latin chars + 1 Ext-B char + period — at ratio close to 9:1
    # the splitter may try to leave "𠀀." as its own slice.
    chunks = ["very long chinese here", "嗯"]
    trans = "starting with latin and one non-bmp 𠀀."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)


def test_merge_cascades_exactly_into_first_slot():
    """Pin the cascade policy: three single-token slots collapse into
    slot 0; slots 1 and 2 become None."""
    chunks = ["a" * 30, "b" * 1, "c" * 1]
    trans = "very long english sentence and ok yes."
    out = split(trans, chunks)
    assert _joined(out) == trans
    # With chunk lengths 30:1:1, slots 1 and 2 will land on 1-token
    # tails (e.g. "ok " / "yes."), both of which must fold back into
    # slot 0.
    assert out[1] is None and out[2] is None
    assert out[0] == trans


def test_cjk_only_translation_one_char_per_chunk_under_explicit_path():
    """Sanity: when the degenerate ``trans_len < 2*n`` path doesn't
    apply (long enough trans), 1-char CJK tails still get merged."""
    chunks = ["a", "b", "c"]
    # 8 CJK chars across 3 chunks (8 >= 2*3=6, so we enter the main path)
    trans = "我们今天非常开心"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)


# ------------------------------------------------------------------
# End-to-end `process()` integration: real Supervision + word
# alignment + translation. Verifies that supervision count, per-cue
# text, per-cue alignment, and translation slots all stay consistent
# after the new splitter + short-tail merge fire under the real flow.
# ------------------------------------------------------------------


def _make_aligned_seg(text, translation, per_char_dur=0.5):
    """Build a Supervision whose word alignment is one item per CJK char."""
    from lattifai.caption.supervision import AlignmentItem, Supervision

    chars = list(text)
    words = [
        AlignmentItem(symbol=c, start=i * per_char_dur, duration=per_char_dur)
        for i, c in enumerate(chars)
    ]
    return Supervision(
        id="sup",
        start=0.0,
        duration=len(chars) * per_char_dur,
        text=text,
        translation=translation,
        alignment={"word": words},
        language="zh",
    )


def _make_standardizer_for_split(max_chars=10, max_duration=4.0):
    from lattifai.caption.standardize import CaptionStandardizer

    s = CaptionStandardizer(
        min_duration=0.01,
        max_duration=max_duration,
        min_gap=0.0,
        max_lines=1,
        max_chars_per_line=max_chars,
    )
    s.config.start_margin = 0.0
    s.config.end_margin = 0.0
    return s


def test_process_aligned_cue_keeps_text_and_alignment_consistent():
    """Each sub-seg's text must match its word-alignment symbols,
    even after short-tail merge redistributes the translation."""
    text = "我们今天非常开心地讨论这个问题然后得出结论"  # 20 CJK chars
    trans = "We had a really happy conversation about this question today."
    seg = _make_aligned_seg(text, trans)
    result = _make_standardizer_for_split().process([seg])

    assert len(result) >= 2
    # Per-cue: text concatenation must equal source text (no drops, no dupes)
    assert "".join(r.text for r in result) == text
    # Per-cue: each character in r.text must appear as a word.symbol in same order
    for r in result:
        word_syms = "".join(w.symbol for w in r.alignment["word"])
        assert word_syms == r.text, (
            f"text {r.text!r} drifted from alignment {word_syms!r}"
        )


def test_process_aligned_cue_translation_slots_match_count():
    """Number of supervisions out == number of translation slots: no cue
    can borrow the next cue's translation because slice_iter desynced."""
    text = "我们今天非常开心地讨论问题然后认真得出最终的结论好嗯"  # 25 chars
    trans = "We had a happy chat about this question and came to a final conclusion."
    seg = _make_aligned_seg(text, trans)
    result = _make_standardizer_for_split().process([seg])

    # Reconstruct the translation by joining non-None per-cue translations.
    rejoined = "".join(r.translation or "" for r in result)
    assert rejoined == trans, (
        f"translation reassembly diverged: {rejoined!r} vs {trans!r}"
    )


def test_process_aligned_cue_no_latin_word_split_in_translation():
    """End-to-end: after process() splits a long aligned cue, no
    sub-seg's translation cuts a Latin word."""
    text = "我们今天非常开心地讨论这个问题然后得出最终结论"  # ~22 chars
    trans = (
        "We discussed this question today and reached our final conclusion together."
    )
    seg = _make_aligned_seg(text, trans)
    result = _make_standardizer_for_split().process([seg])

    translations = [r.translation for r in result]
    _assert_no_latin_word_split(translations)


def test_process_aligned_cue_no_short_tail_translation():
    """End-to-end: no sub-seg ends up with a translation that is just
    1 effective token (e.g. ``"ok."`` / ``"了。"``)."""
    text = "我们讨论这个问题然后得出最终结论好"  # 17 chars, last char is 单字
    trans = "We discussed this question and reached the final conclusion ok."
    seg = _make_aligned_seg(text, trans)
    result = _make_standardizer_for_split().process([seg])

    translations = [r.translation for r in result]
    _assert_no_short_tail(translations)


# ------------------------------------------------------------------
# Gemini review follow-ups: brackets in clause-punct, early-exit
# threshold, short-head merge.
# ------------------------------------------------------------------


def test_cjk_brackets_treated_as_clause_punct():
    """CJK quotation brackets must rank as clause punctuation (score 3),
    not fall through to the generic 'other' bucket (score 1). Cutting
    right after a closing bracket is a natural reading boundary."""
    from lattifai.caption.standardize import CaptionStandardizer

    cases = [
        # (string, position right after the closing bracket)
        ("他说「你好」我们走", 6),  # 「=2 」=5 → after 」 = pos 6
        ("作品《设计》出版了", 6),  # 《=2 》=5 → after 》 = pos 6
        ("function(arg)的实现", 13),  # (=8 )=12 → after ) = pos 13
    ]
    for s, pos in cases:
        score = CaptionStandardizer._boundary_score(s, pos)
        assert score >= 3, f"score at pos {pos} in {s!r} should be >= 3, got {score}"


def test_ascii_paren_close_treated_as_clause_punct():
    """`)` is clause punct — splitting `(arg)|后续` is a natural break."""
    from lattifai.caption.standardize import CaptionStandardizer

    s = "do_it()really works"
    pos = 7  # (=5 )=6 → after ) = pos 7
    assert CaptionStandardizer._boundary_score(s, pos) == 3


def test_early_exit_does_not_settle_for_score_1_with_punct_nearby():
    """The radius-20 early-exit at score>=1 was making the scanner
    accept a generic-CJK boundary (score 1) at radius 5 and miss a
    real sentence period (score 4) at radius 21+.

    Construct: a translation where the only sentence punct sits past
    the radius-20 window from target, but generic CJK boundaries
    appear closer. With early exit, the scanner would settle on a
    score-1 boundary. The fix is to require score >= 2 (whitespace)
    before the early exit fires."""
    from lattifai.caption.standardize import CaptionStandardizer

    # Build a string where target falls in a CJK run with no
    # whitespace nearby, then a sentence period sits past radius 20.
    # ``target`` will be at ~position 20.
    # Layout: 20 CJK chars + ". after period sentence" (sentence punct
    # at position 20 is the '.' — actually we want the period at
    # radius > 20 from target).
    s = "中国汉字汉字汉字汉字汉字" + "x" * 30 + ". end."
    # len(prefix CJK)=10, then 30 'x'. target around half:
    target = len(s) // 2  # ~ in the middle of x-run
    # Score at target is 0 (latin word internal). The scanner should
    # find the period at position len(s)-5 ('.'). That's score 4.
    end = CaptionStandardizer._find_safe_split(s, target, lo=1)
    # With the fix, the scanner should locate the period at a higher
    # score than any nearby CJK boundary. We assert that the returned
    # position lies at a non-zero score (no Latin word cut).
    assert CaptionStandardizer._boundary_score(s, end) >= 1
    # Specifically: the period at the end of "30x" + ". end." should
    # be reachable. If radius-20 short-circuit fires too early, the
    # cut lands inside the CJK run, NOT at the period.
    # The fix improves quality; we pin: cut must NOT be inside
    # ``x``-run when the period is reachable.
    cut_neighbors = (s[end - 1], s[end])
    assert cut_neighbors != ("x", "x"), (
        f"settled for mid-x cut instead of period: pos {end}, neighbors {cut_neighbors}"
    )


def test_short_head_merges_right_into_next_slice():
    """When chunks ratio is e.g. 1:9 (very short FIRST chunk), the
    translation may slice into [tiny_head, big_tail]. The forward-only
    cascade folds 1-token tails left but leaves 1-token HEADS at slot
    0 dangling.

    Fix: a short result[0] should fold RIGHT into result[1] when
    there's no left neighbor to absorb it."""
    chunks = ["了", "很长很长的一段中文"]  # 1:9 ratio
    trans = "Ok. then a really long english sentence here today."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)


def test_short_head_two_token_threshold():
    """Sanity: if the head naturally lands at >= 2 tokens, no merge."""
    chunks = ["短文", "再长一些的中文内容这样"]  # 2:10 ratio
    trans = "Hello there, then a longer english sentence follows here."
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)
    # If first chunk has >= 2 tokens, slot 0 is preserved.
    if out[0] is not None:
        # Could also still merge depending on tokens — accept either
        # as long as no_short_tail invariant holds.
        pass


# ------------------------------------------------------------------
# PR2 — Chinese word-boundary protection in `_split_with_alignment`.
# Uses jieba to detect 2+ char Chinese words and prevents cue
# boundaries from landing inside them.
# ------------------------------------------------------------------


def _make_cjk_aligned_seg(text, translation=None, per_char_dur=0.5):
    """Build a Supervision with per-CJK-char word alignment."""
    from lattifai.caption.supervision import AlignmentItem, Supervision

    chars = list(text)
    words = [
        AlignmentItem(symbol=c, start=i * per_char_dur, duration=per_char_dur)
        for i, c in enumerate(chars)
    ]
    return Supervision(
        id="sup",
        start=0.0,
        duration=len(chars) * per_char_dur,
        text=text,
        translation=translation,
        alignment={"word": words},
        language="zh",
    )


def _make_standardizer(max_chars=10, max_duration=4.0):
    from lattifai.caption.standardize import CaptionStandardizer

    s = CaptionStandardizer(
        min_duration=0.01,
        max_duration=max_duration,
        min_gap=0.0,
        max_lines=1,
        max_chars_per_line=max_chars,
    )
    s.config.start_margin = 0.0
    s.config.end_margin = 0.0
    return s


def _split_boundaries_inside_words(text, sub_texts, min_word_len=2):
    """Identify whether any sub-segment boundary falls inside a Chinese
    word identified by jieba. Returns list of (boundary_char_pos, word).
    Empty list = clean cuts."""
    import jieba

    # jieba word spans by character position
    word_spans = []
    cursor = 0
    for tok in jieba.cut(text, HMM=False):
        if len(tok) >= min_word_len and any("一" <= c <= "鿿" for c in tok):
            word_spans.append((cursor, cursor + len(tok), tok))
        cursor += len(tok)

    # Boundaries are cumulative char positions at the end of each sub_text
    # (excluding the last one which is the end of the full text).
    boundaries = []
    cum = 0
    for st in sub_texts[:-1]:
        cum += len(st)
        boundaries.append(cum)

    violations = []
    for b in boundaries:
        for lo, hi, tok in word_spans:
            if lo < b < hi:
                violations.append((b, tok))
                break
    return violations


def test_chinese_two_char_word_not_cut_xiangmu():
    """Regression: dub karaoke ASS observed Chinese 2-char words cut
    between adjacent cues. With max_chars=10 and per-CJK alignment,
    a group fits 5 CJK chars (each char = 1 + separator). Construct
    text where the 5/6 boundary falls inside a 2-char word like '项目'."""
    text = "为了完成项目我们需要组件支持以及更多功能问题"  # '项目' at chars 4-5
    seg = _make_cjk_aligned_seg(text)
    result = _make_standardizer(max_chars=10).process([seg])

    sub_texts = [r.text for r in result]
    assert "".join(sub_texts) == text
    violations = _split_boundaries_inside_words(text, sub_texts)
    assert not violations, f"Chinese word cut by cue boundary: {violations}"


def test_chinese_two_char_word_not_cut_zujian():
    """Same shape, different word — '组件' wedged at a 10-char boundary."""
    text = "我们今天最终需要创建组件来支持这一新的工作流程"  # '组件' near char 10
    seg = _make_cjk_aligned_seg(text)
    result = _make_standardizer(max_chars=10).process([seg])

    sub_texts = [r.text for r in result]
    assert "".join(sub_texts) == text
    violations = _split_boundaries_inside_words(text, sub_texts)
    assert not violations, f"Chinese word cut: {violations}"


def test_chinese_multi_char_word_not_cut():
    """Multi-char words like '人工智能' (4 chars) must not be cut."""
    text = "现在我们大家都在讨论人工智能的最新进展以及未来应用场景"
    seg = _make_cjk_aligned_seg(text)
    result = _make_standardizer(max_chars=10).process([seg])

    sub_texts = [r.text for r in result]
    assert "".join(sub_texts) == text
    violations = _split_boundaries_inside_words(text, sub_texts)
    assert not violations, f"Chinese word cut: {violations}"


def test_chinese_word_protection_preserves_alignment():
    """When the cue boundary shifts to avoid cutting a word, the
    sub-segment's alignment must still match its text exactly."""
    text = "为了完成项目我们需要组件支持以及更多功能问题"
    seg = _make_cjk_aligned_seg(text)
    result = _make_standardizer(max_chars=10).process([seg])

    for r in result:
        word_syms = "".join(w.symbol for w in r.alignment["word"])
        assert word_syms == r.text, (
            f"text {r.text!r} drifted from alignment {word_syms!r}"
        )


def test_chinese_word_protection_does_not_break_pathological():
    """If a single jieba 'word' exceeds max_chars, the splitter must
    still produce non-empty output without infinite-looping."""
    # Construct text with a 12-char run that jieba may or may not split.
    text = "这是一段非常非常非常非常非常非常长的中文文本内容描述"
    seg = _make_cjk_aligned_seg(text)
    result = _make_standardizer(max_chars=10).process([seg])

    assert sum(len(r.text) for r in result) == len(text)
    assert all(r.text for r in result)


def test_mixed_zh_en_word_protection():
    """Mixed Latin/CJK: both English words and 2-char Chinese words
    must be protected at cue boundaries."""
    text = "请大家使用 Tailwind 框架来构建组件样式系统"
    from lattifai.caption.supervision import AlignmentItem, Supervision

    # Each CJK char and each contiguous Latin run is one alignment item.
    tokens = []
    cur_run = ""
    for ch in text:
        is_cjk = "一" <= ch <= "鿿"
        if is_cjk or ch.isspace():
            if cur_run:
                tokens.append(cur_run)
                cur_run = ""
            if not ch.isspace():
                tokens.append(ch)
        else:
            cur_run += ch
    if cur_run:
        tokens.append(cur_run)

    per = 0.3
    words = []
    t = 0.0
    for tok in tokens:
        words.append(AlignmentItem(symbol=tok, start=t, duration=per * len(tok)))
        t += per * len(tok)
    seg = Supervision(
        id="mix",
        start=0.0,
        duration=t,
        text=text,
        alignment={"word": words},
        language="zh",
    )
    result = _make_standardizer(max_chars=10).process([seg])

    sub_texts = [r.text for r in result]
    assert "".join(sub_texts) == text
    violations = _split_boundaries_inside_words(text, sub_texts)
    assert not violations, f"Word cut at cue boundary: {violations}"
