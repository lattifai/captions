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
from lattifai.caption.caption import BilingualMode

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "captions" / "bilingual"

FIXTURE_EXPECTATIONS = [
    # Positive cases — all three detect branches should fire.
    ("bilingual_line_by_line.srt", BilingualMode.LINE_BY_LINE),
    ("bilingual_alternating.ass", BilingualMode.ALTERNATING),
    ("bilingual_ass_styles.ass", BilingualMode.ALTERNATING),
    # Negative cases — mono captions that share a "shape" with
    # bilingual but fail one of the coverage / count guards.
    ("mono_with_sparse_newlines.srt", BilingualMode.NONE),
    ("mono_with_sync_pairs.srt", BilingualMode.NONE),
    ("mono_with_ass_styles.ass", BilingualMode.NONE),
]


@pytest.mark.parametrize("filename,expected_mode", FIXTURE_EXPECTATIONS)
def test_detect_bilingual_mode_on_fixture(filename, expected_mode):
    cap = Caption.read(FIXTURE_DIR / filename)
    assert cap.detect_bilingual_mode() == expected_mode


@pytest.mark.parametrize("filename,expected_mode", FIXTURE_EXPECTATIONS)
def test_has_bilingual_layout_on_fixture(filename, expected_mode):
    cap = Caption.read(FIXTURE_DIR / filename)
    assert cap.has_bilingual_layout == (expected_mode != BilingualMode.NONE)


def test_detect_bilingual_mode_returns_enum():
    cap = Caption.read(FIXTURE_DIR / "bilingual_line_by_line.srt")
    result = cap.detect_bilingual_mode()
    assert isinstance(result, BilingualMode)
    # Enum values double as strings for serialisation friendliness.
    assert result == "line_by_line"


def test_detect_bilingual_mode_not_cached():
    """Detector must reflect mutations to ``supervisions`` immediately.

    ``Caption`` is a mutable dataclass — caching the result would go
    stale the moment a caller edits ``cap.supervisions`` in place.
    """
    cap = Caption.read(FIXTURE_DIR / "bilingual_line_by_line.srt")
    assert cap.detect_bilingual_mode() == BilingualMode.LINE_BY_LINE
    # Strip the bilingual second half from every cue → should flip to NONE.
    for sup in cap.supervisions:
        sup.text = (sup.text or "").split("\n", 1)[0]
    assert cap.detect_bilingual_mode() == BilingualMode.NONE
