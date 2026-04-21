"""Unit tests for ``_extract_audit.audit_extract_products``.

These tests lock the audit rule set down so later edits to the
corpus smoke-test script cannot silently lose coverage. Each test
starts from a known-clean baseline, then poisons the extract product
with a single, surgical mutation and asserts the expected issue
signature is raised (and that unrelated issues do not bleed in).
"""

from typing import List, Tuple

import pytest

from lattifai.caption import Caption
from lattifai.caption.supervision import Supervision

from _extract_audit import (  # noqa: E402 — pytest prepends this dir to sys.path
    audit_extract_alignment,
    audit_extract_products,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sup(
    idx: int,
    text: str,
    start: float,
    duration: float,
    language: str | None,
    break_before: bool,
) -> Supervision:
    """Build a Supervision that already looks like an extract product."""
    s = Supervision(
        text=text,
        start=start,
        duration=duration,
        language=language,
        custom={"align_index": idx, "alignment_break_before": break_before},
    )
    return s


def _clean_mono() -> Tuple[Caption, List[Supervision], List[Supervision]]:
    """Return a clean mono-zh caption and its (primary, secondary) products.

    Evenly-spaced 2 s gaps, all within the adaptive-gap threshold, so
    every ``alignment_break_before`` is False.
    """
    src = [
        Supervision(text="第一句", start=0.0, duration=1.5, language="zh"),
        Supervision(text="第二句", start=2.0, duration=1.5, language="zh"),
        Supervision(text="第三句", start=4.0, duration=1.5, language="zh"),
    ]
    cap = Caption(supervisions=src)
    primary = [
        _sup(0, "第一句", 0.0, 1.5, "zh", False),
        _sup(1, "第二句", 2.0, 1.5, "zh", False),
        _sup(2, "第三句", 4.0, 1.5, "zh", False),
    ]
    return cap, primary, []


def _clean_f2() -> Tuple[Caption, List[Supervision], List[Supervision]]:
    """Return a clean F2 (alternating) bilingual caption + products."""
    src = [
        Supervision(text="第一句", start=0.0, duration=1.5, language="zh"),
        Supervision(text="First line", start=2.0, duration=1.5, language="en"),
        Supervision(text="第二句", start=4.0, duration=1.5, language="zh"),
        Supervision(text="Second line", start=6.0, duration=1.5, language="en"),
        Supervision(text="第三句", start=8.0, duration=1.5, language="zh"),
        Supervision(text="Third line", start=10.0, duration=1.5, language="en"),
    ]
    cap = Caption(supervisions=src)
    primary = [
        _sup(0, "第一句", 0.0, 1.5, "zh", False),
        _sup(2, "第二句", 4.0, 1.5, "zh", False),
        _sup(4, "第三句", 8.0, 1.5, "zh", False),
    ]
    secondary = [
        _sup(1, "First line", 2.0, 1.5, "en", False),
        _sup(3, "Second line", 6.0, 1.5, "en", False),
        _sup(5, "Third line", 10.0, 1.5, "en", False),
    ]
    return cap, primary, secondary


def _clean_f1() -> Tuple[Caption, List[Supervision], List[Supervision]]:
    """Return a clean F1 (inline) bilingual caption + products.

    Each source row embeds both CJK and Latin split on newline; the
    extractor emits the CJK half to primary and the Latin half to
    secondary, both stamped with the same ``align_index``.
    """
    src = [
        Supervision(text="第一句\nFirst line", start=0.0, duration=1.5, language=None),
        Supervision(text="第二句\nSecond line", start=2.0, duration=1.5, language=None),
        Supervision(text="第三句\nThird line", start=4.0, duration=1.5, language=None),
    ]
    cap = Caption(supervisions=src)
    primary = [
        _sup(0, "第一句", 0.0, 1.5, "zh", False),
        _sup(1, "第二句", 2.0, 1.5, "zh", False),
        _sup(2, "第三句", 4.0, 1.5, "zh", False),
    ]
    secondary = [
        _sup(0, "First line", 0.0, 1.5, "en", False),
        _sup(1, "Second line", 2.0, 1.5, "en", False),
        _sup(2, "Third line", 4.0, 1.5, "en", False),
    ]
    return cap, primary, secondary


def _issue_set(issues: List[str]) -> set[str]:
    return set(issues)


# ---------------------------------------------------------------------------
# Baselines: clean inputs must produce zero issues
# ---------------------------------------------------------------------------


def test_clean_mono_is_audit_clean() -> None:
    cap, primary, secondary = _clean_mono()
    assert audit_extract_products(cap, primary, secondary) == []


def test_clean_f2_is_audit_clean() -> None:
    cap, primary, secondary = _clean_f2()
    assert audit_extract_products(cap, primary, secondary) == []


def test_clean_f1_is_audit_clean() -> None:
    cap, primary, secondary = _clean_f1()
    assert audit_extract_products(cap, primary, secondary) == []


# ---------------------------------------------------------------------------
# align_index topology
# ---------------------------------------------------------------------------


def test_detects_align_index_out_of_range() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].align_index = 99
    issues = audit_extract_products(cap, primary, [])
    assert "primary: align_index out of range" in issues


def test_detects_align_index_missing() -> None:
    cap, primary, _ = _clean_mono()
    del primary[0].custom["align_index"]
    issues = audit_extract_products(cap, primary, [])
    assert "primary: align_index missing" in issues


def test_detects_duplicate_align_index_within_side() -> None:
    cap, primary, _ = _clean_mono()
    primary[1].align_index = primary[0].align_index
    issues = audit_extract_products(cap, primary, [])
    assert "primary: duplicate align_index within side" in issues


def test_detects_rows_not_in_source_order() -> None:
    cap, primary, _ = _clean_mono()
    primary[0], primary[-1] = primary[-1], primary[0]
    issues = audit_extract_products(cap, primary, [])
    assert "primary: rows not in source order" in issues


# ---------------------------------------------------------------------------
# Timing drift
# ---------------------------------------------------------------------------


def test_detects_start_drifted() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].start += 0.5
    issues = audit_extract_products(cap, primary, [])
    assert "primary: start drifted from source row" in issues


def test_detects_duration_drifted() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].duration += 0.5
    issues = audit_extract_products(cap, primary, [])
    assert "primary: duration drifted from source row" in issues


def test_detects_zero_duration_row() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].duration = 0.0
    issues = audit_extract_products(cap, primary, [])
    assert "primary: zero-duration row leaked through" in issues


# ---------------------------------------------------------------------------
# alignment_break_before contract
# ---------------------------------------------------------------------------


def test_detects_alignment_break_before_missing() -> None:
    cap, primary, _ = _clean_mono()
    del primary[0].custom["alignment_break_before"]
    issues = audit_extract_products(cap, primary, [])
    assert "primary: alignment_break_before missing-or-not-bool" in issues


def test_detects_alignment_break_before_wrong_type() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].custom["alignment_break_before"] = 1  # int, not bool
    issues = audit_extract_products(cap, primary, [])
    assert "primary: alignment_break_before missing-or-not-bool" in issues


def test_detects_alignment_break_before_disagrees_with_source() -> None:
    cap, primary, _ = _clean_mono()
    # All gaps in the baseline are small — no row should have a break.
    # Flip idx 1's flag on to disagree with the caption's own map.
    primary[1].custom["alignment_break_before"] = True
    issues = audit_extract_products(cap, primary, [])
    assert "primary: alignment_break_before disagrees with source gap" in issues


# ---------------------------------------------------------------------------
# Text sanity
# ---------------------------------------------------------------------------


def test_detects_empty_text() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].text = "   "
    issues = audit_extract_products(cap, primary, [])
    assert "primary: empty text" in issues


def test_detects_ass_override_tag() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].text = r"{\an8}第一句"
    issues = audit_extract_products(cap, primary, [])
    assert "primary: ASS override tag not stripped" in issues


def test_detects_newline_in_bilingual_text() -> None:
    cap, primary, secondary = _clean_f2()
    primary[0].text = "第一句\n多出一行"
    issues = audit_extract_products(cap, primary, secondary)
    assert "primary: newline leaked in bilingual text" in issues


def test_allows_newline_in_mono_text() -> None:
    """Newline is a bilingual-only red flag; mono captions may keep it."""
    cap, primary, _ = _clean_mono()
    primary[0].text = "第一句\n续行"
    issues = audit_extract_products(cap, primary, [])
    assert "primary: newline leaked in bilingual text" not in issues


def test_detects_drawing_line_type() -> None:
    cap, primary, _ = _clean_mono()
    primary[0].custom["line_type"] = "drawing"
    issues = audit_extract_products(cap, primary, [])
    assert "primary: drawing row leaked through" in issues


def test_detects_non_dialogue_row() -> None:
    """A staff-credit style text still gets classified by classify_line_type."""
    cap, primary, _ = _clean_mono()
    # "翻译 张三" is picked up as staff_credit by the text classifier.
    primary[0].text = "翻译 张三"
    issues = audit_extract_products(cap, primary, [])
    assert "primary: non-dialogue row leaked through" in issues


# ---------------------------------------------------------------------------
# Language labels
# ---------------------------------------------------------------------------


def test_detects_mixed_language_labels_within_side() -> None:
    cap, primary, _ = _clean_mono()
    primary[1].language = "en"
    issues = audit_extract_products(cap, primary, [])
    assert "primary: mixed language labels within side" in issues


def test_detects_bilingual_same_language_label() -> None:
    cap, primary, secondary = _clean_f2()
    for row in secondary:
        row.language = "zh"
        row.text = row.text.replace("line", "行")
    issues = audit_extract_products(cap, primary, secondary)
    assert "bilingual: primary/secondary share the same language label" in issues


def test_detects_bilingual_same_script_pair() -> None:
    """Two Latin-script langs should have been merged by same-script rollback."""
    cap, primary, secondary = _clean_f2()
    # Swap primary to French (Latin) so both sides live in the same bucket.
    for row in primary:
        row.language = "fr"
        row.text = row.text.replace("第", "L")
    issues = audit_extract_products(cap, primary, secondary)
    assert "bilingual: same-script pair survived rollback" in issues


# ---------------------------------------------------------------------------
# Bilingual topology
# ---------------------------------------------------------------------------


def test_detects_mixed_topology() -> None:
    """Partial overlap — neither F1-subset nor F2-disjoint."""
    cap, primary, secondary = _clean_f2()
    # Point one secondary row at a primary-owned source index so the
    # topology is neither a subset nor disjoint.
    secondary[0].align_index = primary[0].align_index  # overlap
    # keep the other secondaries disjoint
    issues = audit_extract_products(cap, primary, secondary)
    assert "bilingual: align_index topology is neither F1-subset nor F2-disjoint" in issues


def test_detects_f1_shared_index_break_before_mismatch() -> None:
    cap, primary, secondary = _clean_f1()
    # Flip the break_before on the secondary side of a shared index.
    secondary[0].custom["alignment_break_before"] = True
    issues = audit_extract_products(cap, primary, secondary)
    assert "bilingual: F1 shared-index break_before mismatch" in issues


# ---------------------------------------------------------------------------
# Row-count skew
# ---------------------------------------------------------------------------


def test_detects_row_count_skew_2x() -> None:
    """primary has 4 rows, secondary has 1 → >2x but ≤10x."""
    src = [
        Supervision(text="第一句", start=0.0, duration=1.5, language="zh"),
        Supervision(text="第二句", start=2.0, duration=1.5, language="zh"),
        Supervision(text="第三句", start=4.0, duration=1.5, language="zh"),
        Supervision(text="第四句", start=6.0, duration=1.5, language="zh"),
        Supervision(text="First line", start=8.0, duration=1.5, language="en"),
    ]
    cap = Caption(supervisions=src)
    primary = [
        _sup(0, "第一句", 0.0, 1.5, "zh", False),
        _sup(1, "第二句", 2.0, 1.5, "zh", False),
        _sup(2, "第三句", 4.0, 1.5, "zh", False),
        _sup(3, "第四句", 6.0, 1.5, "zh", False),
    ]
    secondary = [_sup(4, "First line", 8.0, 1.5, "en", False)]
    issues = audit_extract_products(cap, primary, secondary)
    assert "bilingual: row-count skew (>2x)" in issues
    assert "bilingual: extreme row-count skew (>10x)" not in issues


def test_detects_row_count_skew_10x() -> None:
    """primary has 11 rows, secondary has 1 → >10x."""
    src = [
        Supervision(text=f"第{i}句", start=float(i * 2), duration=1.5, language="zh")
        for i in range(11)
    ] + [Supervision(text="solo", start=30.0, duration=1.5, language="en")]
    cap = Caption(supervisions=src)
    primary = [
        _sup(i, f"第{i}句", float(i * 2), 1.5, "zh", False) for i in range(11)
    ]
    secondary = [_sup(11, "solo", 30.0, 1.5, "en", False)]
    issues = audit_extract_products(cap, primary, secondary)
    assert "bilingual: extreme row-count skew (>10x)" in issues


# ---------------------------------------------------------------------------
# Exception handling at the outer layer
# ---------------------------------------------------------------------------


def test_audit_extract_alignment_captures_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """If extract_alignment_supervisions raises, the outer audit returns a
    single ``RAISED …`` signature instead of propagating."""
    cap, _, _ = _clean_mono()

    def boom(self):
        raise RuntimeError("poisoned")

    monkeypatch.setattr(Caption, "extract_alignment_supervisions", boom)
    issues = audit_extract_alignment(cap)
    assert len(issues) == 1
    assert issues[0].startswith("RAISED RuntimeError: poisoned")


def test_audit_extract_alignment_is_clean_on_real_extract() -> None:
    """End-to-end sanity: a freshly constructed F2 caption survives the
    full ``extract_alignment_supervisions`` + audit pipeline clean."""
    # Use the same layout as _clean_f2 but let extract do the work.
    src = [
        Supervision(text="第一句", start=0.0, duration=1.5, language="zh"),
        Supervision(text="First line", start=2.0, duration=1.5, language="en"),
        Supervision(text="第二句", start=4.0, duration=1.5, language="zh"),
        Supervision(text="Second line", start=6.0, duration=1.5, language="en"),
        Supervision(text="第三句", start=8.0, duration=1.5, language="zh"),
        Supervision(text="Third line", start=10.0, duration=1.5, language="en"),
    ]
    cap = Caption(supervisions=src)
    assert audit_extract_alignment(cap) == []
