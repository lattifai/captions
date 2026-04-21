"""Rule-based audit for ``Caption.extract_alignment_supervisions``.

This module is a test/diagnostic helper — it is imported by the
corpus smoke-test script and by ``test_extract_audit.py`` to keep the
two in lock-step.

The audit enforces invariants that must hold regardless of ground
truth, so it can be pointed at any corpus or hand-crafted caption and
return a short, groupable list of issue signatures.

The audit is factored into two layers so that unit tests can feed
*poisoned* ``(primary, secondary)`` triples directly, without going
through ``extract_alignment_supervisions`` itself:

- ``audit_extract_products(cap, primary, secondary)`` runs the rules.
- ``audit_extract_alignment(cap)`` calls ``extract`` first, catches
  any exception as a ``RAISED …`` signature, then delegates.
"""

from typing import Iterable, List

from lattifai.caption import Caption
from lattifai.caption.caption import (
    _ALIGNMENT_CJK_DISTINCT,
    _ALIGNMENT_OVERRIDE_RE,
    _ALIGNMENT_SCRIPT_BUCKETS,
)
from lattifai.caption.parsers.text_parser import classify_line_type
from lattifai.caption.supervision import Supervision


def classify_error(exc: BaseException) -> str:
    """One-line signature for grouping similar failures across a corpus."""
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


def _same_alignment_script(a: str | None, b: str | None) -> bool:
    """Mirror of ``Caption._same_alignment_script``.

    True when the two language labels fall into the same coarse script
    bucket *and* are not two distinct CJK siblings (zh/ja/ko are
    allowed to coexist since lingua disambiguates them reliably on
    full-caption aggregates).
    """
    if not a or not b:
        return False
    if a in _ALIGNMENT_CJK_DISTINCT and b in _ALIGNMENT_CJK_DISTINCT and a != b:
        return False
    return _ALIGNMENT_SCRIPT_BUCKETS.get(a, "latin") == _ALIGNMENT_SCRIPT_BUCKETS.get(b, "latin")


def _collect_align_indexes(rows: Iterable[Supervision]) -> set[int]:
    """Pull align_index off each row (attribute or ``custom`` fallback)."""
    out: set[int] = set()
    for r in rows:
        idx = getattr(r, "align_index", None)
        if idx is None and r.custom:
            idx = r.custom.get("align_index")
        if isinstance(idx, int):
            out.add(idx)
    return out


def _collect_break_flags(rows: Iterable[Supervision]) -> dict[int, bool]:
    """Return ``{align_index: alignment_break_before}`` for rows that carry both."""
    out: dict[int, bool] = {}
    for r in rows:
        idx = getattr(r, "align_index", None)
        if idx is None and r.custom:
            idx = r.custom.get("align_index")
        abb = (r.custom or {}).get("alignment_break_before")
        if isinstance(idx, int) and isinstance(abb, bool):
            out[idx] = abb
    return out


def _expected_break_before(cap: Caption) -> list[bool]:
    """Recompute the adaptive-gap break map the way caption.py does.

    Mirrors ``_compute_break_before`` in
    ``src/lattifai/caption/caption.py`` — adaptive gap threshold
    ``max(2.0, min(5.0, 3 × median_positive_gap))``, same-timing
    neighbours (F2 atomic unit) are never boundaries.
    """
    sups = cap.supervisions
    n = len(sups)
    if n < 2:
        return [False] * n

    gaps: list[float] = []
    for j in range(1, n):
        prev_end = (sups[j - 1].start or 0.0) + (sups[j - 1].duration or 0.0)
        g = (sups[j].start or 0.0) - prev_end
        if g > 0:
            gaps.append(g)

    if not gaps:
        return [False] * n

    med = sorted(gaps)[len(gaps) // 2]
    threshold = max(2.0, min(5.0, 3.0 * med))
    out = [False] * n
    for j in range(1, n):
        if abs((sups[j].start or 0.0) - (sups[j - 1].start or 0.0)) < 0.01:
            continue
        prev_end = (sups[j - 1].start or 0.0) + (sups[j - 1].duration or 0.0)
        g = (sups[j].start or 0.0) - prev_end
        if g >= threshold:
            out[j] = True
    return out


def audit_extract_products(
    cap: Caption,
    primary: List[Supervision],
    secondary: List[Supervision],
) -> List[str]:
    """Run the audit rules against an already-extracted ``(primary, secondary)`` pair.

    This is the layer unit tests exercise directly with poisoned
    triples — no ``extract_alignment_supervisions`` call involved.

    Returns a sorted, de-duplicated list of short issue signatures so
    the caller can aggregate them across a corpus.

    Invariants enforced (see module docstring for the narrative):
      - per-side ``align_index`` in range and unique within the side
      - per-side rows emitted in source order (non-decreasing index)
      - ``duration > 0.01`` and non-empty text
      - ``start`` AND ``duration`` match the source row within 10 ms
      - no residual ``{\\…}`` ASS override tags
      - no ``line_type == "drawing"``, ``classify_line_type is None``
      - each side carries a uniform ``language`` label
      - ``alignment_break_before`` is a ``bool`` and matches the
        caption's own adaptive-gap map per source row
      - bilingual output: distinct language labels and alignment-
        script buckets, no newline leakage, roughly balanced row counts
      - F1-subset vs F2-disjoint topology; mixed topologies flagged
      - F1 shared-index break_before agrees on both sides
    """
    issues: list[str] = []

    n_src = len(cap.supervisions)
    is_bilingual = bool(secondary)
    expected_break = _expected_break_before(cap)

    def _side_issues(rows: List[Supervision], side_name: str) -> None:
        seen_idx: set[int] = set()
        langs: set[str | None] = set()
        prev_idx = -1
        for row in rows:
            align_index = getattr(row, "align_index", None)
            if align_index is None and row.custom:
                align_index = row.custom.get("align_index")

            if align_index is None:
                issues.append(f"{side_name}: align_index missing")
            elif not isinstance(align_index, int) or align_index < 0 or align_index >= n_src:
                issues.append(f"{side_name}: align_index out of range")
            else:
                if align_index in seen_idx:
                    issues.append(f"{side_name}: duplicate align_index within side")
                seen_idx.add(align_index)

                if align_index < prev_idx:
                    issues.append(f"{side_name}: rows not in source order")
                prev_idx = align_index

                src = cap.supervisions[align_index]
                if abs(row.start - src.start) > 0.01:
                    issues.append(f"{side_name}: start drifted from source row")
                if abs((row.duration or 0.0) - (src.duration or 0.0)) > 0.01:
                    issues.append(f"{side_name}: duration drifted from source row")

                abb = (row.custom or {}).get("alignment_break_before")
                if not isinstance(abb, bool):
                    issues.append(f"{side_name}: alignment_break_before missing-or-not-bool")
                elif abb != expected_break[align_index]:
                    issues.append(f"{side_name}: alignment_break_before disagrees with source gap")

            if row.duration is None or row.duration <= 0.01:
                issues.append(f"{side_name}: zero-duration row leaked through")

            text = row.text or ""
            if not text.strip():
                issues.append(f"{side_name}: empty text")
            if _ALIGNMENT_OVERRIDE_RE.search(text):
                issues.append(f"{side_name}: ASS override tag not stripped")
            if "\n" in text and is_bilingual:
                issues.append(f"{side_name}: newline leaked in bilingual text")

            custom = row.custom or {}
            if custom.get("line_type") == "drawing":
                issues.append(f"{side_name}: drawing row leaked through")
            if classify_line_type(
                text,
                start=row.start,
                ass_raw_text=custom.get("ass_raw_text"),
                duration=row.duration,
            ) is not None:
                issues.append(f"{side_name}: non-dialogue row leaked through")

            langs.add(row.language)

        if len(langs) > 1:
            issues.append(f"{side_name}: mixed language labels within side")

    _side_issues(primary, "primary")
    if secondary:
        _side_issues(secondary, "secondary")

    if is_bilingual:
        p_lang = primary[0].language if primary else None
        s_lang = secondary[0].language
        if p_lang and s_lang and p_lang == s_lang:
            issues.append("bilingual: primary/secondary share the same language label")
        if _same_alignment_script(p_lang, s_lang):
            issues.append("bilingual: same-script pair survived rollback")

        p_idx = _collect_align_indexes(primary)
        s_idx = _collect_align_indexes(secondary)
        if p_idx and s_idx:
            is_f1 = s_idx.issubset(p_idx)
            is_f2 = not (s_idx & p_idx)
            if not (is_f1 or is_f2):
                issues.append("bilingual: align_index topology is neither F1-subset nor F2-disjoint")
            elif is_f1:
                p_breaks = _collect_break_flags(primary)
                s_breaks = _collect_break_flags(secondary)
                for idx in s_idx:
                    if idx in p_breaks and idx in s_breaks and p_breaks[idx] != s_breaks[idx]:
                        issues.append("bilingual: F1 shared-index break_before mismatch")
                        break

        if primary and secondary:
            ratio = max(len(primary), len(secondary)) / max(1, min(len(primary), len(secondary)))
            if ratio > 10.0:
                issues.append("bilingual: extreme row-count skew (>10x)")
            elif ratio > 2.0:
                issues.append("bilingual: row-count skew (>2x)")

    return sorted(set(issues))


def audit_extract_alignment(cap: Caption) -> List[str]:
    """Run ``extract_alignment_supervisions`` and audit its output.

    Returns a list of issue signatures (empty if clean). An exception
    during extraction is surfaced as a single ``RAISED …`` signature.
    """
    try:
        primary, secondary = cap.extract_alignment_supervisions()
    except Exception as exc:  # noqa: BLE001
        return [f"RAISED {classify_error(exc)}"]
    return audit_extract_products(cap, primary, secondary)
