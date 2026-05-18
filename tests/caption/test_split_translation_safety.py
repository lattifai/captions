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
    trans = "Hello, world today"
    out = split(trans, chunks)
    # First chunk should end at the comma (clause punct, score 3) rather than
    # the space before 'today' (whitespace, score 2) when both are reachable.
    assert out[0] is not None
    assert out[0].rstrip().endswith(",") or out[0].endswith(", "), (
        f"Expected split after comma, got: {out}"
    )
    _assert_no_latin_word_split(out)


def test_sentence_punct_short_circuits():
    """Sentence punct (score 4) wins instantly."""
    chunks = ["aaaaaaa", "bbbbb"]
    trans = "Done. Now we continue here for a while"
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


def test_pure_cjk_one_char_per_chunk_collapses():
    """4-chunk split of a 4-CJK-char translation: each 1-token slice
    collapses leftward until no slice has < 2 tokens."""
    chunks = ["a", "b", "c", "d"]
    trans = "我们今天"
    out = split(trans, chunks)
    assert _joined(out) == trans
    _assert_no_short_tail(out)
