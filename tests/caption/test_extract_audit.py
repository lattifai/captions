"""Unit tests for ``_extract_audit.audit_extract_products``.

These tests lock the audit rule set down so later edits to the
corpus smoke-test script cannot silently lose coverage. Each test
starts from a known-clean baseline, then poisons the extract product
with a single, surgical mutation and asserts the expected issue
signature is raised (and that unrelated issues do not bleed in).
"""

from dataclasses import replace
from typing import List, Tuple

import pytest

from lattifai.caption import Caption
from lattifai.caption.bilingual import AlignmentPlan
from lattifai.caption.supervision import Supervision

from _extract_audit import (  # noqa: E402 — pytest prepends this dir to sys.path
    audit_extract_alignment,
    audit_extract_products,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sup(
    text: str,
    start: float,
    duration: float,
    language: str | None,
) -> Supervision:
    """Build a Supervision that already looks like an extract product."""
    return Supervision(
        text=text,
        start=start,
        duration=duration,
        language=language,
    )


Quad = Tuple[Caption, List[Supervision], List[Supervision], AlignmentPlan]


def _clean_mono() -> Quad:
    """Return a clean mono-zh caption and its (primary, secondary, plan)."""
    src = [
        Supervision(text="第一句", start=0.0, duration=1.5, language="zh"),
        Supervision(text="第二句", start=2.0, duration=1.5, language="zh"),
        Supervision(text="第三句", start=4.0, duration=1.5, language="zh"),
    ]
    cap = Caption(supervisions=src)
    primary = [
        _sup("第一句", 0.0, 1.5, "zh"),
        _sup("第二句", 2.0, 1.5, "zh"),
        _sup("第三句", 4.0, 1.5, "zh"),
    ]
    plan = AlignmentPlan(
        source_indices_primary=(0, 1, 2),
        source_indices_secondary=(),
    )
    return cap, primary, [], plan


def _clean_f2() -> Quad:
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
        _sup("第一句", 0.0, 1.5, "zh"),
        _sup("第二句", 4.0, 1.5, "zh"),
        _sup("第三句", 8.0, 1.5, "zh"),
    ]
    secondary = [
        _sup("First line", 2.0, 1.5, "en"),
        _sup("Second line", 6.0, 1.5, "en"),
        _sup("Third line", 10.0, 1.5, "en"),
    ]
    plan = AlignmentPlan(
        source_indices_primary=(0, 2, 4),
        source_indices_secondary=(1, 3, 5),
    )
    return cap, primary, secondary, plan


def _clean_f1() -> Quad:
    """Return a clean F1 (inline) bilingual caption + products.

    Each source row embeds both CJK and Latin split on newline; the
    extractor emits the CJK half to primary and the Latin half to
    secondary, both pointing at the same source row.
    """
    src = [
        Supervision(text="第一句\nFirst line", start=0.0, duration=1.5, language=None),
        Supervision(text="第二句\nSecond line", start=2.0, duration=1.5, language=None),
        Supervision(text="第三句\nThird line", start=4.0, duration=1.5, language=None),
    ]
    cap = Caption(supervisions=src)
    primary = [
        _sup("第一句", 0.0, 1.5, "zh"),
        _sup("第二句", 2.0, 1.5, "zh"),
        _sup("第三句", 4.0, 1.5, "zh"),
    ]
    secondary = [
        _sup("First line", 0.0, 1.5, "en"),
        _sup("Second line", 2.0, 1.5, "en"),
        _sup("Third line", 4.0, 1.5, "en"),
    ]
    plan = AlignmentPlan(
        source_indices_primary=(0, 1, 2),
        source_indices_secondary=(0, 1, 2),
    )
    return cap, primary, secondary, plan


# ---------------------------------------------------------------------------
# Baselines: clean inputs must produce zero issues
# ---------------------------------------------------------------------------


def test_clean_mono_is_audit_clean() -> None:
    cap, primary, secondary, plan = _clean_mono()
    assert audit_extract_products(cap, primary, secondary, plan) == []


def test_clean_f2_is_audit_clean() -> None:
    cap, primary, secondary, plan = _clean_f2()
    assert audit_extract_products(cap, primary, secondary, plan) == []


def test_clean_f1_is_audit_clean() -> None:
    cap, primary, secondary, plan = _clean_f1()
    assert audit_extract_products(cap, primary, secondary, plan) == []


# ---------------------------------------------------------------------------
# source_indices topology
# ---------------------------------------------------------------------------


def test_detects_source_index_out_of_range() -> None:
    cap, primary, secondary, plan = _clean_mono()
    plan = replace(plan, source_indices_primary=(99, 1, 2))
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: source index out of range" in issues


def test_detects_source_indices_length_mismatch() -> None:
    cap, primary, secondary, plan = _clean_mono()
    plan = replace(plan, source_indices_primary=(0, 1))  # 2 indices vs 3 rows
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: source_indices length mismatch" in issues


def test_detects_duplicate_source_index_within_side() -> None:
    cap, primary, secondary, plan = _clean_mono()
    plan = replace(plan, source_indices_primary=(0, 0, 2))
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: duplicate source index within side" in issues


def test_detects_rows_not_in_source_order() -> None:
    cap, primary, secondary, plan = _clean_mono()
    plan = replace(plan, source_indices_primary=(2, 1, 0))
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: rows not in source order" in issues


# ---------------------------------------------------------------------------
# Timing drift
# ---------------------------------------------------------------------------


def test_detects_start_drifted() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[0].start += 0.5
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: start drifted from source row" in issues


def test_detects_duration_drifted() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[0].duration += 0.5
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: duration drifted from source row" in issues


def test_detects_zero_duration_row() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[0].duration = 0.0
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: zero-duration row leaked through" in issues


# ---------------------------------------------------------------------------
# Text sanity
# ---------------------------------------------------------------------------


def test_detects_empty_text() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[0].text = "   "
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: empty text" in issues


def test_detects_ass_override_tag() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[0].text = r"{\an8}第一句"
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: ASS override tag not stripped" in issues


def test_detects_newline_in_bilingual_text() -> None:
    cap, primary, secondary, plan = _clean_f2()
    primary[0].text = "第一句\n多出一行"
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: newline leaked in bilingual text" in issues


def test_allows_newline_in_mono_text() -> None:
    """Newline is a bilingual-only red flag; mono captions may keep it."""
    cap, primary, secondary, plan = _clean_mono()
    primary[0].text = "第一句\n续行"
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: newline leaked in bilingual text" not in issues


def test_detects_drawing_line_type() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[0].custom = {"line_type": "drawing"}
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: drawing row leaked through" in issues


def test_detects_non_dialogue_row() -> None:
    """A staff-credit style text still gets classified by classify_line_type."""
    cap, primary, secondary, plan = _clean_mono()
    # "翻译 张三" is picked up as staff_credit by the text classifier.
    primary[0].text = "翻译 张三"
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: non-dialogue row leaked through" in issues


# ---------------------------------------------------------------------------
# Language labels
# ---------------------------------------------------------------------------


def test_detects_mixed_language_labels_within_side() -> None:
    cap, primary, secondary, plan = _clean_mono()
    primary[1].language = "en"
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "primary: mixed language labels within side" in issues


def test_detects_bilingual_same_language_label() -> None:
    cap, primary, secondary, plan = _clean_f2()
    for row in secondary:
        row.language = "zh"
        row.text = row.text.replace("line", "行")
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "bilingual: primary/secondary share the same language label" in issues


def test_detects_bilingual_same_script_pair() -> None:
    """Two Latin-script langs should have been merged by same-script rollback."""
    cap, primary, secondary, plan = _clean_f2()
    # Swap primary to French (Latin) so both sides live in the same bucket.
    for row in primary:
        row.language = "fr"
        row.text = row.text.replace("第", "L")
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "bilingual: same-script pair survived rollback" in issues


# ---------------------------------------------------------------------------
# Bilingual topology
# ---------------------------------------------------------------------------


def test_detects_mixed_topology() -> None:
    """Partial overlap — neither F1-subset nor F2-disjoint."""
    cap, primary, secondary, plan = _clean_f2()
    # Point one secondary index at a primary-owned source so the
    # topology is neither a subset nor disjoint.
    plan = replace(plan, source_indices_secondary=(0, 3, 5))
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "bilingual: source-index topology is neither F1-subset nor F2-disjoint" in issues


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
        _sup("第一句", 0.0, 1.5, "zh"),
        _sup("第二句", 2.0, 1.5, "zh"),
        _sup("第三句", 4.0, 1.5, "zh"),
        _sup("第四句", 6.0, 1.5, "zh"),
    ]
    secondary = [_sup("First line", 8.0, 1.5, "en")]
    plan = AlignmentPlan(
        source_indices_primary=(0, 1, 2, 3),
        source_indices_secondary=(4,),
    )
    issues = audit_extract_products(cap, primary, secondary, plan)
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
        _sup(f"第{i}句", float(i * 2), 1.5, "zh") for i in range(11)
    ]
    secondary = [_sup("solo", 30.0, 1.5, "en")]
    plan = AlignmentPlan(
        source_indices_primary=tuple(range(11)),
        source_indices_secondary=(11,),
    )
    issues = audit_extract_products(cap, primary, secondary, plan)
    assert "bilingual: extreme row-count skew (>10x)" in issues


# ---------------------------------------------------------------------------
# Exception handling at the outer layer
# ---------------------------------------------------------------------------


def test_audit_extract_alignment_captures_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """If extract_alignment_supervisions raises, the outer audit returns a
    single ``RAISED …`` signature instead of propagating."""
    import _extract_audit as audit_module

    cap, _, _, _ = _clean_mono()

    def boom(*args, **kwargs):
        raise RuntimeError("poisoned")

    monkeypatch.setattr(audit_module, "extract_alignment_supervisions", boom)
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
