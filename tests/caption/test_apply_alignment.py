"""Tests for ``Caption.apply_alignment()``.

Contract:
    apply_alignment(aligned: List[Supervision]) -> None

    Mutates ``self.supervisions`` in place:

    1. For each ``sup`` in ``aligned``, read ``sup.custom["align_index"]``
       (set by ``extract_for_alignment``) and write its
       ``start`` / ``duration`` / ``alignment["word"]`` onto
       ``self.supervisions[align_index]``. Rows without a valid index are
       silently skipped; rows in ``self`` whose index is never written
       are left untouched.

    2. After the index-matched writes, dialogue rows in ``self`` whose
       duration is ≤ 0.01s (timestamps dropped by the subtitle-group
       source) are re-timed by linear interpolation between the nearest
       aligned neighbours.
"""

import lattifai.caption.parsers.text_parser as text_parser

from lattifai.caption import Caption
from lattifai.caption.supervision import AlignmentItem, Supervision


def _cap(supervisions, source_format="srt"):
    sups = []
    for i, s in enumerate(supervisions):
        sups.append(
            Supervision(
                text=s["text"],
                start=s["start"],
                duration=s["duration"],
                custom=s.get("custom"),
                alignment=s.get("alignment"),
            )
        )
    return Caption(supervisions=sups, source_format=source_format)


def _aligned(idx: int, start: float, duration: float, words=None) -> Supervision:
    """Build a fake alignment result. ``idx`` is the 0-based position of the
    target row in ``self.supervisions`` (as ``extract_for_alignment`` would
    have stamped into ``sup.custom['align_index']``)."""
    sup = Supervision(
        text="", start=start, duration=duration,
        alignment={"word": words} if words is not None else None,
    )
    sup.align_index = idx
    return sup


# ---------------------------------------------------------------------------
# id-based write-back
# ---------------------------------------------------------------------------


def test_apply_writes_start_duration_for_matched_index() -> None:
    """One aligned row → one row in self updated in place."""
    caption = _cap([
        {"text": "Hello", "start": 1.0, "duration": 3.0},
    ])
    caption.apply_alignment([_aligned(0, start=1.234, duration=2.876)])
    assert caption.supervisions[0].start == 1.234
    assert caption.supervisions[0].duration == 2.876


def test_apply_writes_word_alignment_onto_matched_row() -> None:
    """Word-level alignment from aligned sup is copied over."""
    words = [
        AlignmentItem(symbol="Hello", start=1.0, duration=0.5, score=0.92),
        AlignmentItem(symbol="world", start=1.6, duration=0.4, score=0.88),
    ]
    caption = _cap([
        {"text": "Hello world", "start": 0.0, "duration": 3.0},
    ])
    caption.apply_alignment([_aligned(0, 1.0, 1.0, words=words)])
    assert caption.supervisions[0].alignment is not None
    assert caption.supervisions[0].alignment["word"] == words


def test_apply_leaves_unaddressed_self_rows_untouched() -> None:
    """Rows in self whose index isn't targeted keep their original timing."""
    caption = _cap([
        {"text": "dialogue", "start": 1.0, "duration": 3.0},
        {"text": "staff", "start": 100.0, "duration": 5.0},
    ])
    caption.apply_alignment([_aligned(0, 1.234, 2.876)])
    # row 0 updated, row 1 untouched.
    assert caption.supervisions[0].start == 1.234
    assert caption.supervisions[1].start == 100.0
    assert caption.supervisions[1].duration == 5.0


def test_apply_silently_ignores_out_of_range_indices() -> None:
    """An aligned sup with out-of-range index must NOT raise — just be skipped."""
    caption = _cap([
        {"text": "Hello", "start": 1.0, "duration": 3.0},
    ])
    caption.apply_alignment([
        _aligned(0, 1.234, 2.876),
        _aligned(99, 10.0, 5.0),  # out of range
    ])
    assert caption.supervisions[0].start == 1.234


def test_apply_mutates_in_place_and_returns_none() -> None:
    """apply_alignment is an imperative method; returns None."""
    caption = _cap([
        {"text": "Hi", "start": 0.0, "duration": 2.0},
    ])
    result = caption.apply_alignment([_aligned(0, 1.0, 1.0)])
    assert result is None
    assert caption.supervisions[0].start == 1.0


def test_apply_fast_path_skips_dialogue_classification(monkeypatch) -> None:
    """Fully covered, already-timed captions should take the simple write-back path."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fast path should not call classify_line_type")

    monkeypatch.setattr(text_parser, "classify_line_type", fail_if_called)

    caption = _cap([
        {"text": "Hello", "start": 1.0, "duration": 3.0},
        {"text": "World", "start": 5.0, "duration": 2.0},
    ])
    caption.apply_alignment([
        _aligned(0, 1.234, 2.876),
        _aligned(1, 5.678, 1.111),
    ])

    assert caption.supervisions[0].start == 1.234
    assert caption.supervisions[0].duration == 2.876
    assert caption.supervisions[1].start == 5.678
    assert caption.supervisions[1].duration == 1.111


# ---------------------------------------------------------------------------
# Dual-row (F2): each language row has its own id and is updated independently
# ---------------------------------------------------------------------------


def test_apply_dual_row_updates_aligned_language_row_only() -> None:
    """If alignment only has the English track, only that row gets updated.

    The Chinese row (different index) stays at its original timestamps.
    """
    caption = _cap([
        {"text": "我们都很看好你", "start": 1.0, "duration": 3.0},          # idx 0
        {"text": "We all think a lot of you", "start": 1.0, "duration": 3.0},  # idx 1
    ])
    caption.apply_alignment([_aligned(1, start=1.200, duration=2.600)])
    # en row (idx 1) updated
    assert caption.supervisions[1].start == 1.200
    assert caption.supervisions[1].duration == 2.600
    # zh row (idx 0) untouched
    assert caption.supervisions[0].start == 1.0
    assert caption.supervisions[0].duration == 3.0


# ---------------------------------------------------------------------------
# No-timing interpolation
# ---------------------------------------------------------------------------


def test_apply_interpolates_zero_duration_dialogue_between_neighbours() -> None:
    """Rows with duration ≤ 0.01 sit between aligned neighbours — interpolate.

    Layout: aligned (1-3s) → no-timing dialogue → aligned (7-10s).
    The middle row should land in the gap (somewhere in [3, 7)).
    """
    caption = _cap([
        {"text": "First line", "start": 1.0, "duration": 2.0},    # idx 0
        # dialogue but timestamps dropped by source
        {"text": "Middle line", "start": 0.0, "duration": 0.0},   # idx 1
        {"text": "Last line", "start": 7.0, "duration": 3.0},     # idx 2
    ])
    caption.apply_alignment([
        _aligned(0, 1.0, 2.0),   # unchanged
        _aligned(2, 7.0, 3.0),   # unchanged
    ])
    middle = caption.supervisions[1]
    # Must land somewhere in the gap [3.0, 7.0] with positive duration.
    assert 3.0 <= middle.start < 7.0
    assert middle.duration > 0.0
    assert middle.start + middle.duration <= 7.0


def test_apply_does_not_touch_zero_duration_non_dialogue_rows() -> None:
    """A zero-duration staff_credit row isn't dialogue → no interpolation."""
    caption = _cap([
        {"text": "First line", "start": 1.0, "duration": 2.0},     # idx 0
        {"text": "翻译 张三", "start": 0.0, "duration": 0.0},        # idx 1
        {"text": "Last line", "start": 7.0, "duration": 3.0},      # idx 2
    ])
    caption.apply_alignment([
        _aligned(0, 1.0, 2.0),
        _aligned(2, 7.0, 3.0),
    ])
    credit = caption.supervisions[1]
    assert credit.start == 0.0
    assert credit.duration == 0.0


# ---------------------------------------------------------------------------
# alignment_break_before — the bit stamped by extract_alignment_supervisions
# so that apply_alignment's no-timing interpolation respects segment
# boundaries instead of borrowing timestamps across a large gap.
# ---------------------------------------------------------------------------


def test_apply_does_not_use_right_anchor_across_alignment_break_before() -> None:
    """Regression: when the aligner returns a row tagged with
    ``alignment_break_before=True``, the no-timing interpolation on the
    preceding zero-duration dialogue run must NOT use that row as a
    right anchor — the boundary is semantic (post-gap, new scene), not
    a timing neighbour.
    """
    caption = _cap([
        {"text": "A", "start": 0.0, "duration": 1.0},      # idx 0 timed, end=1.0
        {"text": "B", "start": 1.0, "duration": 0.0},      # idx 1 zero-dur dialogue
        {"text": "C", "start": 50.0, "duration": 1.0},     # idx 2 timed, post-gap
    ])

    aligned_c = _aligned(2, start=50.0, duration=1.0)
    aligned_c.alignment_break_before = True  # boundary between idx 1 and idx 2

    caption.apply_alignment([
        _aligned(0, 0.0, 1.0),
        aligned_c,
    ])

    # Without the boundary bit, idx 1 would borrow idx 2's start (50.0)
    # as its right anchor and stretch its duration to ~49s. With the
    # bit, the right anchor is blocked; idx 1 falls back to one-sided
    # extend-from-left, which leaves its duration at 0.0.
    assert caption.supervisions[1].start == 1.0
    assert caption.supervisions[1].duration == 0.0
    # idx 2's own timing must still be written (the bit only blocks
    # cross-boundary interpolation, not direct write-back).
    assert caption.supervisions[2].start == 50.0
    assert caption.supervisions[2].duration == 1.0


def test_apply_splits_zero_duration_dialogue_run_at_alignment_break_before() -> None:
    """A zero-duration dialogue run that straddles a break boundary must
    be processed as two independent runs — pre-break uses the left-side
    anchor only, post-break uses the right-side anchor only.
    """
    caption = _cap([
        {"text": "A", "start": 0.0, "duration": 1.0},       # idx 0 timed, end=1.0
        {"text": "B1", "start": 1.0, "duration": 0.0},      # idx 1 zero-dur (pre-break)
        {"text": "B2", "start": 1.5, "duration": 0.0},      # idx 2 zero-dur (pre-break)
        {"text": "C", "start": 50.0, "duration": 1.0},      # idx 3 timed, post-gap
        {"text": "D1", "start": 51.0, "duration": 0.0},     # idx 4 zero-dur (post-break)
        {"text": "D2", "start": 51.5, "duration": 0.0},     # idx 5 zero-dur (post-break)
        {"text": "E", "start": 60.0, "duration": 1.0},      # idx 6 timed
    ])

    aligned_c = _aligned(3, start=50.0, duration=1.0)
    aligned_c.alignment_break_before = True

    caption.apply_alignment([
        _aligned(0, 0.0, 1.0),
        aligned_c,
        _aligned(6, 60.0, 1.0),
    ])

    # Pre-break run (idx 1, 2): must fall back to one-sided left anchor,
    # so both get duration 0.0 (not stretched across the 49-second gap).
    assert caption.supervisions[1].duration == 0.0
    assert caption.supervisions[2].duration == 0.0
    # Post-break run (idx 4, 5): has left=idx 3 (end=51.0) and right=idx 6
    # (start=60.0), no break on idx 6, so both get interpolated across
    # the 9-second span (duration ≈ 4.5s each).
    assert caption.supervisions[4].duration > 0.0
    assert caption.supervisions[5].duration > 0.0
    assert abs(caption.supervisions[4].duration - 4.5) < 1e-3
    assert abs(caption.supervisions[5].duration - 4.5) < 1e-3
