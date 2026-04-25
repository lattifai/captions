"""Bilingual detection fixtures — pin detect_bilingual_mode behaviour
across canonical caption shapes.

These fixtures synthesise the patterns we care about without shipping
real subtitle-group data (copyright + encoding noise). Each fixture is
the smallest configuration that triggers (or deliberately fails to
trigger) one of the three detect branches.
"""

from pathlib import Path

import pytest

from lattifai.caption import Caption
from lattifai.caption.bilingual import BilingualMode, detect_bilingual_mode

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "captions" / "bilingual"

FIXTURE_EXPECTATIONS = [
    # Positive cases — all three detect branches should fire.
    ("bilingual_line_by_line.srt", BilingualMode.LINE_BY_LINE),
    ("bilingual_alternating.ass", BilingualMode.SAME_TIMING_PAIRS),
    ("bilingual_ass_styles.ass", BilingualMode.STYLE_GROUPED),
    # Negative cases — mono captions that share a "shape" with
    # bilingual but fail one of the coverage / count guards.
    ("mono_with_sparse_newlines.srt", BilingualMode.NONE),
    ("mono_with_sync_pairs.srt", BilingualMode.NONE),
    ("mono_with_ass_styles.ass", BilingualMode.NONE),
]


@pytest.mark.parametrize("filename,expected_mode", FIXTURE_EXPECTATIONS)
def test_detect_bilingual_mode_on_fixture(filename, expected_mode):
    cap = Caption.read(FIXTURE_DIR / filename)
    assert detect_bilingual_mode(cap.supervisions, cap.source_format) == expected_mode


@pytest.mark.parametrize("filename,expected_mode", FIXTURE_EXPECTATIONS)
def test_has_bilingual_layout_on_fixture(filename, expected_mode):
    cap = Caption.read(FIXTURE_DIR / filename)
    detected = detect_bilingual_mode(cap.supervisions, cap.source_format)
    assert (detected != BilingualMode.NONE) == (expected_mode != BilingualMode.NONE)


def test_detect_bilingual_mode_returns_enum():
    cap = Caption.read(FIXTURE_DIR / "bilingual_line_by_line.srt")
    result = detect_bilingual_mode(cap.supervisions, cap.source_format)
    assert isinstance(result, BilingualMode)
    # Enum values double as strings for serialisation friendliness.
    assert result == "line_by_line"


def test_bilingual_mode_distinguishes_sync_pairs_from_style_grouped():
    """The two ASS bilingual fixtures differ structurally.

    ``bilingual_alternating.ass`` has cues at the same (start, duration)
    physically adjacent in the file → SAME_TIMING_PAIRS.
    ``bilingual_ass_styles.ass`` separates languages by style and offsets
    the timestamps by 100ms so adjacent-pair detection misses → falls
    back to the style branch → STYLE_GROUPED.
    """
    sync = Caption.read(FIXTURE_DIR / "bilingual_alternating.ass")
    grouped = Caption.read(FIXTURE_DIR / "bilingual_ass_styles.ass")
    sync_mode = detect_bilingual_mode(sync.supervisions, sync.source_format)
    grouped_mode = detect_bilingual_mode(grouped.supervisions, grouped.source_format)
    assert sync_mode == BilingualMode.SAME_TIMING_PAIRS
    assert grouped_mode == BilingualMode.STYLE_GROUPED
    # bilingual layout collapses both to True.
    assert sync_mode != BilingualMode.NONE
    assert grouped_mode != BilingualMode.NONE


def test_detect_bilingual_mode_not_cached():
    """Detector must reflect mutations to ``supervisions`` immediately.

    ``Caption`` is a mutable dataclass — caching the result would go
    stale the moment a caller edits ``cap.supervisions`` in place.
    """
    cap = Caption.read(FIXTURE_DIR / "bilingual_line_by_line.srt")
    assert detect_bilingual_mode(cap.supervisions, cap.source_format) == BilingualMode.LINE_BY_LINE
    # Strip the bilingual second half from every cue → should flip to NONE.
    for sup in cap.supervisions:
        sup.text = (sup.text or "").split("\n", 1)[0]
    assert detect_bilingual_mode(cap.supervisions, cap.source_format) == BilingualMode.NONE
