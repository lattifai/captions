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
