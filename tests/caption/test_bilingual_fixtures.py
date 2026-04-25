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

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "captions" / "bilingual"


@pytest.mark.parametrize(
    "filename,expected_mode",
    [
        # Positive cases — all three detect branches should fire.
        ("bilingual_line_by_line.srt", "line_by_line"),
        ("bilingual_alternating.ass", "alternating"),
        ("bilingual_ass_styles.ass", "alternating"),
        # Negative cases — mono captions that share a "shape" with
        # bilingual but fail one of the coverage / count guards.
        ("mono_with_sparse_newlines.srt", "none"),
        ("mono_with_sync_pairs.srt", "none"),
        ("mono_with_ass_styles.ass", "none"),
    ],
)
def test_detect_bilingual_mode_on_fixture(filename, expected_mode):
    cap = Caption.read(FIXTURE_DIR / filename)
    assert cap._detect_bilingual_mode() == expected_mode
