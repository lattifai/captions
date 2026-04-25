"""Bilingual layout detection, extraction, and merging.

This module owns the entire "is this caption bilingual, and if so how"
question. ``Caption`` delegates to it for ``detect_bilingual_mode`` and
``extract_alignment_supervisions``; ``Caption.merge_bilingual`` calls
``merge_line_by_line`` / ``merge_alternating`` directly.

Functions here take primitive inputs (a list of ``Supervision`` and a
``source_format`` string) so they can be tested without instantiating a
``Caption``. The class methods are thin wrappers that pass
``self.supervisions`` / ``self.source_format`` through.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .supervision import Supervision, fastcopy


class BilingualMode(str, Enum):
    """Pre-merge bilingual layout detected on raw caption rows.

    ``NONE`` is the overwhelming common case — most captions are mono.
    The other three modes describe distinct **structural** patterns
    subtitle groups use to ship two languages within a single file:

    - ``LINE_BY_LINE``: one cue, two lines split by ``\\n``
      (CJK on top, Latin below or vice versa).
    - ``SAME_TIMING_PAIRS``: two adjacent cues at the same
      ``(start, duration)`` carrying the two languages.
    - ``STYLE_GROUPED``: ASS file with one style per language;
      cues for each language are grouped and **not necessarily
      adjacent** in the file. Pairing requires sorting by timestamp.

    The split between ``SAME_TIMING_PAIRS`` and ``STYLE_GROUPED`` was
    introduced after we caught real ASS subtitle-group files where the
    two languages share the same timestamps but are emitted in two
    separate runs (e.g. all 580 English rows first, then all 582
    Chinese rows). The old umbrella ``ALTERNATING`` mode treated this
    as a sync-pair layout, found no adjacent pairs, and routed the
    file to a Layer 2/3 rollback that silently degraded it to mono.
    """

    NONE = "none"
    LINE_BY_LINE = "line_by_line"
    SAME_TIMING_PAIRS = "same_timing_pairs"
    STYLE_GROUPED = "style_grouped"


# ---------------------------------------------------------------------------
# Plan structure consumed by ``Caption.apply_alignment``.
#
# ``extract_alignment_supervisions`` returns extracted ``primary`` and
# ``secondary`` supervisions plus a frozen ``AlignmentPlan`` whose
# ``source_indices_primary[i]`` / ``source_indices_secondary[i]`` tell
# ``apply_alignment`` which row to write each aligned result back to.
# F1 inline captions share the same source index across the two sides;
# F2 dual-row uses each row's original position.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentPlan:
    """Side-channel structure produced by ``extract_alignment_supervisions``.

    ``source_indices_primary[i]`` / ``source_indices_secondary[i]`` give the
    0-based source-row index for ``primary[i]`` / ``secondary[i]``. F1 inline
    captions share the same index across the two sides; F2 dual-row uses
    each row's original position.
    """

    source_indices_primary: Tuple[int, ...] = ()
    source_indices_secondary: Tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Module-level constants shared by every step (detect / extract / merge).
# ---------------------------------------------------------------------------

_ALIGNMENT_OVERRIDE_RE = re.compile(r"\{\\[^}]*\}")
_ALIGNMENT_CJK_DISTINCT = {"zh", "ja", "ko"}
_ALIGNMENT_SCRIPT_BUCKETS = {
    "zh": "east_asian",
    "ja": "east_asian",
    "ko": "east_asian",
    "east_asian": "east_asian",
    "ru": "cyrillic",
    "uk": "cyrillic",
    "bg": "cyrillic",
    "be": "cyrillic",
    "mk": "cyrillic",
    "sr": "cyrillic",
    "kk": "cyrillic",
    "mn": "cyrillic",
    "cyrillic": "cyrillic",
    "ar": "arabic",
    "fa": "arabic",
    "ur": "arabic",
    "arabic": "arabic",
    "he": "hebrew",
    "hebrew": "hebrew",
    "el": "greek",
    "greek": "greek",
    "hi": "devanagari",
    "mr": "devanagari",
    "devanagari": "devanagari",
    "th": "thai",
    "thai": "thai",
    "hy": "armenian",
    "armenian": "armenian",
    "ka": "georgian",
    "georgian": "georgian",
    "bn": "bengali",
    "bengali": "bengali",
    "ta": "tamil",
    "tamil": "tamil",
    "te": "telugu",
    "telugu": "telugu",
    "gu": "gujarati",
    "gujarati": "gujarati",
    "pa": "gurmukhi",
    "gurmukhi": "gurmukhi",
}


# ---------------------------------------------------------------------------
# Internal helpers (module-level so detect / extract can share them).
# ---------------------------------------------------------------------------

def _strip_alignment_text(text: str) -> str:
    """Strip ASS override tags and collapse embedded whitespace.

    Alignment operates on the token stream, not the rendered layout, so
    soft-wrap newlines (``医生\\n先生`` or ``What a lovely day\\nit is
    today.``) become single-space-separated runs before the lattice
    tokenizer ever sees them.
    """
    cleaned = _ALIGNMENT_OVERRIDE_RE.sub("", text or "")
    return " ".join(cleaned.split())


def _is_alignable(sup: Supervision, plain: str) -> bool:
    from .parsers.text_parser import classify_line_type

    if not plain or sup.duration is None or sup.duration <= 0.01:
        return False
    custom = sup.custom or {}
    if custom.get("line_type") == "drawing":
        return False
    ass_raw = custom.get("ass_raw_text")
    return classify_line_type(
        plain, start=sup.start,
        ass_raw_text=ass_raw, duration=sup.duration,
    ) is None


def _vote_lang(texts: List[str]) -> Optional[str]:
    """Aggregate-voting language detection for a group of texts.

    Joining and feeding lingua a single long string scores ~98.5 % on
    FLORES vs. ~95.4 % per-row, so this is markedly more robust than
    running detection on each short dialogue line.
    """
    from .parsers.language_detector import detect_language, detect_script

    if not texts:
        return None
    joined = " ".join(texts)
    return detect_language(joined) or detect_script(joined)


def _same_alignment_script(a: Optional[str], b: Optional[str]) -> bool:
    """Are two lang labels confusable for alignment purposes?

    True when they map to the same coarse script bucket AND they aren't
    distinct CJK siblings (zh/ja/ko disambiguate cleanly on full-caption
    aggregates).
    """
    if not a or not b:
        return False
    if a in _ALIGNMENT_CJK_DISTINCT and b in _ALIGNMENT_CJK_DISTINCT and a != b:
        return False
    return _ALIGNMENT_SCRIPT_BUCKETS.get(a, "latin") == _ALIGNMENT_SCRIPT_BUCKETS.get(b, "latin")


def _is_simple_mono(supervisions: List[Supervision], source_format: Optional[str]) -> bool:
    """Performance gate: can this mono caption skip the structured loop?

    **Not** a semantic gate. The bilingual question is settled upstream
    by :func:`detect_bilingual_mode`; this predicate only asks whether
    the (already-mono) caption is shaped simply enough to skip the
    per-row strip/classify pipeline and go straight to a 1:1 supervision
    copy.

    Returns ``False`` whenever a row needs *something* — ASS override
    stripping, soft-newline collapse, branding/staff removal,
    same-timing pair handling — even though the file as a whole is
    mono. Those still produce a correct mono extraction; they just go
    through the structured loop.
    """
    from .parsers.text_parser import _BRANDING_KEYWORDS, _STAFF_ROLES

    if source_format in {"ass", "ssa"}:
        return False

    prev_start = None
    prev_duration = None
    for sup in supervisions:
        text = sup.text or ""
        if not text or "\n" in text or sup.duration is None or sup.duration <= 0.01:
            return False

        custom = sup.custom or {}
        if custom and (
            custom.get("line_type") is not None
            or custom.get("ass_raw_text")
            or custom.get("ass_style")
        ):
            return False

        same_timing_pair = (
            prev_start is not None
            and abs(sup.start - prev_start) < 0.01
            and abs(sup.duration - prev_duration) < 0.01
        )
        if same_timing_pair:
            return False

        stripped = text.strip()
        if not stripped:
            return False
        if sup.start <= 120.0 and len(stripped) <= 50:
            if _STAFF_ROLES.match(stripped):
                return False
            lower = stripped.lower()
            if any(keyword in lower for keyword in _BRANDING_KEYWORDS):
                return False

        prev_start = sup.start
        prev_duration = sup.duration

    return True


# ---------------------------------------------------------------------------
# Public API: detect / extract / merge.
# ---------------------------------------------------------------------------

def detect_bilingual_mode(
    supervisions: List[Supervision],
    source_format: Optional[str] = None,
) -> BilingualMode:
    """Detect the bilingual arrangement of the raw caption (pre-merge).

    Priority (most specific → most ambiguous):

    1. Text contains ``\\n`` with CJK/Latin split → ``LINE_BY_LINE``.
       Strongest signal: an explicit within-cue break with different
       scripts on each side. Checked first so ASS files whose dialogue
       uses inline ``\\N`` (F1) aren't misclassified as dual-row just
       because a handful of sign/title rows share a style name with
       dialogue.
    2. Adjacent same-timing pairs with different CJK ratios →
       ``SAME_TIMING_PAIRS``.
    3. ASS style names correlate with different languages →
       ``STYLE_GROUPED``.
    4. Otherwise → ``NONE`` (monolingual).

    Each branch enforces a ≥ 20 % coverage floor so a few stray
    bilingual-shaped rows in an otherwise-mono caption don't flip the
    whole file to bilingual.

    Not cached — see ``Caption.has_bilingual_layout``.

    ``source_format`` is currently unused by the detector but accepted
    to keep the signature symmetric with :func:`extract_alignment_supervisions`.
    """
    from .parsers.text_parser import cjk_ratio

    sups = supervisions
    if not sups:
        return BilingualMode.NONE

    # 1. line_by_line via explicit \n split.
    #    Count a cue as "bilingual" only when the two halves are CLEARLY
    #    different scripts — one side CJK-dominant, the other Latin-
    #    dominant (or vice versa). Merely "different ratios" isn't
    #    enough: a pure Chinese cue like ``1972年3月21日\n1972年3月26日``
    #    has CJK ratio < 1.0 because the digits drop it, and a
    #    permissive ``abs(r1 - r2) > 0.3`` would wrongly flag it.
    #
    #    We also require the bilingual rows to cover a non-trivial
    #    share of the whole caption (≥ 20 %). Without this, a live-
    #    broadcast dub with 2400 mono Chinese rows + 240 bilingual
    #    quotes (seen on Oscars/sports captions) would flip to
    #    line_by_line and leave 2400 mono rows stranded as
    #    "untranslated primary".
    newline_bilingual = 0
    newline_total = 0
    for sup in sups:
        text = sup.text or ""
        if "\n" in text:
            newline_total += 1
            lines = text.split("\n", 1)
            r1 = cjk_ratio(lines[0])
            r2 = cjk_ratio(lines[1])
            if (r1 >= 0.6 and r2 <= 0.2) or (r2 >= 0.6 and r1 <= 0.2):
                newline_bilingual += 1
    bilingual_coverage = newline_bilingual / len(sups) if sups else 0.0
    if (
        newline_total >= 2
        and newline_bilingual / newline_total > 0.5
        and bilingual_coverage >= 0.2
    ):
        return BilingualMode.LINE_BY_LINE

    # 2. Same-timing pairs (alternating pattern).
    #    We count both how many pairs exist AND what share of the
    #    total rows they cover. Two stray CJK-vs-Latin pairs in a
    #    1000-row mono caption (e.g. a CJK-heavy title card stacked
    #    over a Latin effect line) used to trip this branch at
    #    ``pair_count >= 2``, producing absurd splits like 1228/1.
    #    Require the pairs to cover ≥ 20 % of the rows to claim a
    #    genuine alternating arrangement.
    pair_count = 0
    cjk_diff_count = 0
    i = 0
    while i + 1 < len(sups):
        s1, s2 = sups[i], sups[i + 1]
        if abs(s1.start - s2.start) < 0.01 and abs(s1.duration - s2.duration) < 0.01:
            pair_count += 1
            r1 = cjk_ratio(s1.text or "")
            r2 = cjk_ratio(s2.text or "")
            if abs(r1 - r2) > 0.4:
                cjk_diff_count += 1
            i += 2
        else:
            i += 1
    pair_coverage = (2 * pair_count) / len(sups) if sups else 0.0
    if (
        pair_count >= 2
        and cjk_diff_count / pair_count > 0.5
        and pair_coverage >= 0.2
    ):
        return BilingualMode.SAME_TIMING_PAIRS

    # 3. ASS style-based split (e.g., "中文 1080" vs "英文 1080").
    #
    #    Distinct from SAME_TIMING_PAIRS even though both express
    #    "two language tracks at the same timestamps". Style-grouped
    #    ASS files can list each language in a contiguous run
    #    (English rows 0-579, then Chinese rows 580-1161 at the
    #    same start/duration as 0-579). Adjacent-pair detection
    #    misses this — pairing has to go through (start, duration)
    #    after grouping by style.
    #
    #    Both the high-CJK and low-CJK style must individually cover
    #    ≥ 20 % of all supervisions. Without this floor, a mono CJK
    #    caption with a handful of Latin "Sign" / "Title" rows trips
    #    bilingual because the Sign style averages cjk_ratio ≈ 0
    #    while the Default body averages ≈ 1 — the max-min spread
    #    looks like a real bilingual split even though Sign covers
    #    only 2/12 = 17 % of the file. The 20 % floor matches the
    #    other two branches and turns the script into a single
    #    coherent definition of "bilingual coverage".
    style_cjk: dict[str, list[float]] = {}
    for sup in sups:
        custom = getattr(sup, "custom", None) or {}
        style = custom.get("ass_style", "")
        if style and sup.text:
            style_cjk.setdefault(style, []).append(cjk_ratio(sup.text))
    if len(style_cjk) >= 2:
        avg_ratios = {s: sum(r) / len(r) for s, r in style_cjk.items()}
        counts = {s: len(r) for s, r in style_cjk.items()}
        high_style = max(avg_ratios, key=avg_ratios.get)
        low_style = min(avg_ratios, key=avg_ratios.get)
        spread = avg_ratios[high_style] - avg_ratios[low_style]
        min_coverage = min(counts[high_style], counts[low_style]) / len(sups)
        if spread > 0.4 and min_coverage >= 0.2:
            return BilingualMode.STYLE_GROUPED

    return BilingualMode.NONE


def extract_alignment_supervisions(
    supervisions: List[Supervision],
    source_format: Optional[str] = None,
) -> Tuple[List[Supervision], List[Supervision], AlignmentPlan]:
    """Extract per-language alignable supervisions from a (possibly
    bilingual) caption.

    Returns
    -------
    (primary_sups, secondary_sups, plan)
        ``primary_sups`` holds the detected primary-language rows, in
        source order. ``secondary_sups`` holds the other language for
        bilingual captions; ``[]`` for mono.

        ``plan`` is an :class:`AlignmentPlan` whose ``source_indices_primary[i]``
        / ``source_indices_secondary[i]`` give the 0-based source-row index
        back into ``supervisions`` for ``primary[i]`` / ``secondary[i]``.
        F1 inline shares one index across both sides; F2 dual-row uses
        each row's original position.

    Each returned ``Supervision`` is a ``fastcopy`` carrying:
      - ``text``: plaintext in that language (ASS override tags stripped)
      - ``language``: ISO-639-1 decided by *group-level aggregated voting*.
      - ``start`` / ``duration``: original timestamps (unchanged).

    Excluded from both lists:
      - non-dialogue rows (``classify_line_type`` ≠ ``None``)
      - zero/near-zero-duration rows (``duration ≤ 0.01s``).

    Safety layers (prevent misclassification from slipping through):
      1. Aggregated voting (this function).
      2. Same-script rollback: if the "secondary" group is detected to
         share the alignment script with the primary, merge secondary
         back into primary and treat as mono. CJK siblings (zh/ja/ko)
         are exempt.
      3. Skew rollback: tiny secondary side relative to primary →
         mode misdetection, roll back to mono.

    ``supervisions`` is not mutated.
    """
    from .parsers.text_parser import cjk_ratio

    def _make_plan(
        primary_idx: List[int],
        secondary_idx: List[int],
    ) -> AlignmentPlan:
        return AlignmentPlan(
            source_indices_primary=tuple(primary_idx),
            source_indices_secondary=tuple(secondary_idx),
        )

    # Single decision point: ask once whether the file shows a bilingual
    # layout. Most captions are mono and skip every bilingual-shaped
    # branch below; only files with a real bilingual structure pay the
    # dual-side extraction cost.
    mode = detect_bilingual_mode(supervisions, source_format)

    primary: List[Supervision] = []
    secondary: List[Supervision] = []
    primary_texts: List[str] = []
    secondary_texts: List[str] = []
    primary_idx: List[int] = []
    secondary_idx: List[int] = []

    if mode == BilingualMode.NONE and _is_simple_mono(supervisions, source_format):
        for i, sup in enumerate(supervisions):
            text = _strip_alignment_text(sup.text)
            side = fastcopy(sup, text=text, translation=None, custom=dict(sup.custom or {}))
            primary.append(side)
            primary_texts.append(text)
            primary_idx.append(i)
        p_lang = _vote_lang(primary_texts)
        for sup in primary:
            sup.language = p_lang
        return primary, [], _make_plan(primary_idx, [])

    if mode == BilingualMode.STYLE_GROUPED:
        # ASS files where the two languages are partitioned by
        # ``ass_style`` rather than by adjacent timestamp. The
        # high-CJK style supplies primary rows and the low-CJK
        # style supplies the secondary side; minor "decoration"
        # styles (op/bgm/ed/LOGO/lyric…) are assigned to the side
        # whose average CJK ratio is closest to the row's own —
        # that handles the common case of OP/ED themes that ride
        # in the "wrong" language relative to the body dialogue
        # (Japanese theme song over a Chinese-translated film,
        # for instance).
        #
        # Per-side language voting uses **only** the dominant
        # style on each side. Mixing in minor styles' text would
        # let an OP block of Japanese lyrics flip the primary's
        # vote from ``zh`` to ``ja`` even though 1377 of 1462
        # primary rows are clearly Chinese — and that misvote
        # would then trip the same-script rollback in Layer 2 and
        # silently degrade a real bilingual file to mono.
        #
        # Each row's source position is recorded in the parallel
        # ``primary_idx`` / ``secondary_idx`` lists (F2 dual-row):
        # primary and secondary are independent runs over the
        # source list. The Layer 1-3 fallbacks below are skipped:
        # STYLE_GROUPED's grouping is structural (already grounded
        # in ass_style), not a heuristic that needs rolling back.
        # The one degenerate case worth catching — high_style and
        # low_style turn out to share an alignment script — is
        # handled inline before assigning rows.
        style_cjk: dict[str, list[float]] = {}
        for sup in supervisions:
            style = (sup.custom or {}).get("ass_style", "")
            if style and sup.text:
                style_cjk.setdefault(style, []).append(cjk_ratio(sup.text))
        avg_cjk = {s: sum(r) / len(r) for s, r in style_cjk.items() if r}
        high_style = max(avg_cjk, key=avg_cjk.get) if avg_cjk else None
        low_style = min(avg_cjk, key=avg_cjk.get) if avg_cjk else None

        high_texts = [s.text for s in supervisions
                      if (s.custom or {}).get("ass_style") == high_style and s.text]
        low_texts = [s.text for s in supervisions
                     if (s.custom or {}).get("ass_style") == low_style and s.text]
        p_lang = _vote_lang(high_texts)
        s_lang = _vote_lang(low_texts)

        if not s_lang or _same_alignment_script(p_lang, s_lang):
            # Style split exists but the two sides resolve to the
            # same alignment script — treat as mono and fall through
            # to the structured-mono main loop below.
            pass
        else:
            high_avg = avg_cjk.get(high_style, 1.0)
            low_avg = avg_cjk.get(low_style, 0.0)
            for i, sup in enumerate(supervisions):
                text = _strip_alignment_text(sup.text)
                if not text or not _is_alignable(sup, text):
                    continue
                style = (sup.custom or {}).get("ass_style", "")
                side = fastcopy(sup, text=text, translation=None, custom=dict(sup.custom or {}))
                if style == low_style:
                    side.language = s_lang
                    secondary.append(side)
                    secondary_idx.append(i)
                elif style == high_style:
                    side.language = p_lang
                    primary.append(side)
                    primary_idx.append(i)
                else:
                    # Minor style: route by per-row CJK distance.
                    row_cjk = cjk_ratio(text)
                    if abs(row_cjk - high_avg) <= abs(row_cjk - low_avg):
                        side.language = p_lang
                        primary.append(side)
                        primary_idx.append(i)
                    else:
                        side.language = s_lang
                        secondary.append(side)
                        secondary_idx.append(i)
            return primary, secondary, _make_plan(primary_idx, secondary_idx)

        # Same-script fallback path: rebuild primary as a flat mono
        # extraction over all rows (matches the structured-mono main
        # loop's behaviour for source_format == 'ass').
        primary, secondary = [], []
        primary_texts, secondary_texts = [], []
        primary_idx, secondary_idx = [], []
        mode = BilingualMode.NONE  # downstream loop becomes mono.

    if mode != BilingualMode.STYLE_GROUPED:
        i = 0

        while i < len(supervisions):
            sup = supervisions[i]
            step = 1
            raw_primary = sup.text or ""
            raw_secondary = ""
            secondary_index = i

            if mode == BilingualMode.LINE_BY_LINE:
                lines = raw_primary.split("\n")
                if len(lines) >= 2:
                    raw_primary = lines[0]
                    raw_secondary = lines[1]
            elif (
                mode == BilingualMode.SAME_TIMING_PAIRS
                and i + 1 < len(supervisions)
            ):
                next_sup = supervisions[i + 1]
                if abs(sup.start - next_sup.start) < 0.01 and abs(sup.duration - next_sup.duration) < 0.01:
                    raw_secondary = next_sup.text or ""
                    secondary_index = i + 1
                    step = 2

            t1 = _strip_alignment_text(raw_primary)
            t2 = _strip_alignment_text(raw_secondary)
            # Primary must be present and alignable — a cue whose top
            # line boils down to pure override tags (``{\a6}``) has no
            # content to align and its "secondary" half has no 1:1
            # counterpart either, so the whole cue is dropped. Secondary
            # is optional (mono rows legitimately have no t2), but when
            # present it must also pass the dialogue classifier. Pre-fix
            # we only probed ``probe = t1 or t2``, which let disclaimer
            # pairs like ``视频资料来自网络 版权归BBC所有\n仅供学习交流使用…``
            # slip through (t1 escapes classify_line_type; t2 is caught
            # as branding).
            if not t1 or not _is_alignable(sup, t1):
                i += step
                continue
            if t2 and not _is_alignable(sup, t2):
                i += step
                continue
            base_custom = dict(sup.custom or {})
            side = fastcopy(sup, text=t1, translation=None, custom=dict(base_custom))
            primary.append(side)
            primary_texts.append(t1)
            primary_idx.append(i)
            if t2:
                side = fastcopy(sup, text=t2, translation=None, custom=dict(base_custom))
                secondary.append(side)
                secondary_texts.append(t2)
                secondary_idx.append(secondary_index)
            i += step

    # ---- Layer 1: aggregate voting ----
    p_lang = _vote_lang(primary_texts)
    s_lang = _vote_lang(secondary_texts) if secondary_texts else None

    # ---- Layer 2: same-script rollback (mono mis-split as bilingual) ----
    if s_lang and _same_alignment_script(p_lang, s_lang):
        # Merge by source index (not naive extend) so F2-style splits
        # that happen to share the same script (e.g. pure-JP captions
        # with stray same-timing pairs) recover in source order.
        merged = sorted(
            zip(primary + secondary, primary_texts + secondary_texts, primary_idx + secondary_idx),
            key=lambda triple: triple[2],
        )
        primary = [p for p, _, _ in merged]
        primary_texts = [t for _, t, _ in merged]
        primary_idx = [k for _, _, k in merged]
        secondary, secondary_texts, secondary_idx = [], [], []
        p_lang = _vote_lang(primary_texts)
        s_lang = None

    # ---- Layer 3: skew rollback (tiny secondary → mode misdetection) ----
    # A genuine bilingual caption produces two sides of comparable size.
    # Which threshold triggers rollback depends on topology:
    #   - F1 inline (s_idx ⊆ p_idx): a perfect bilingual has s_idx ==
    #     p_idx with ratio ≈ 0.5; a degenerate F1 has ratio < 5 % and
    #     should drop the stray secondary entries. Use the ratio test
    #     only — absolute count is meaningless here because 3+3 F1 is
    #     still real bilingual content.
    #   - F2 alternating (s_idx disjoint p_idx): use both absolute
    #     count AND ratio. This covers Gemini's 4-row + 1-noise
    #     counter-example (25 % ratio but N < 10, statistically
    #     insignificant) as well as the 1000-row ASS files with a
    #     dozen stray sync pairs.
    if secondary:
        total = len(primary) + len(secondary)
        primary_indexes = set(primary_idx)
        is_f1 = all(k in primary_indexes for k in secondary_idx)
        ratio = len(secondary) / total

        if is_f1:
            should_rollback = ratio < 0.05
        else:
            # F2 real bilingual has ratio ≈ 0.5 (each pair contributes
            # to both sides). A small-N split where ratio drops below
            # ~1/3 is almost always a couple of stray same-timing
            # rows in an otherwise-mono caption, not a real F2.
            should_rollback = ratio < 0.05 or (len(secondary) < 10 and ratio < 0.33)

        if should_rollback:
            if not is_f1:
                # F2 uses disjoint indexes: primary holds even-ish
                # source rows and secondary holds their odd-ish
                # partners. A naive extend() would append all
                # secondary rows after primary, producing a sequence
                # like [0, 2, 4, …, 1, 3, 5, …] that violates the
                # "rows emitted in source order" contract. Merge by
                # source index instead so downstream consumers
                # (apply_alignment, audits) see a clean non-
                # decreasing index sequence.
                merged = sorted(
                    zip(primary + secondary, primary_texts + secondary_texts, primary_idx + secondary_idx),
                    key=lambda triple: triple[2],
                )
                primary = [p for p, _, _ in merged]
                primary_texts = [t for _, t, _ in merged]
                primary_idx = [k for _, _, k in merged]
            secondary, secondary_texts, secondary_idx = [], [], []
            p_lang = _vote_lang(primary_texts)
            s_lang = None

    for sup in primary:
        sup.language = p_lang
    for sup in secondary:
        sup.language = s_lang

    return primary, secondary, _make_plan(primary_idx, secondary_idx)


def merge_line_by_line(
    supervisions: List[Supervision],
    primary_language: Optional[str],
    secondary_language: Optional[str],
) -> List[Supervision]:
    """Split each supervision's text by newline into text + translation.

    F1 inline: each merged supervision keeps the source row's original
    timing; alignment write-back is later driven by the
    :class:`AlignmentPlan` produced by ``extract_alignment_supervisions``,
    not by per-row stamping.
    """
    new_sups = []
    for sup in supervisions:
        text = sup.text or ""
        lines = text.split("\n")
        if len(lines) >= 2:
            new_sup = fastcopy(
                sup,
                text=lines[0].strip(),
                translation=lines[1].strip(),
                language=primary_language or sup.language,
                target_lang=secondary_language,
            )
        else:
            new_sup = fastcopy(sup, language=primary_language or sup.language)
        new_sups.append(new_sup)
    return new_sups


def merge_alternating(
    supervisions: List[Supervision],
    primary_language: Optional[str],
    secondary_language: Optional[str],
) -> List[Supervision]:
    """Merge consecutive same-timing supervisions into text + translation.

    F2 dual-row collapses two adjacent same-timing supervisions into one
    bilingual cue. Source-row indices are not stamped onto the merged
    supervisions: alignment write-back is driven by the
    :class:`AlignmentPlan` produced by ``extract_alignment_supervisions``,
    which already knows that primary and secondary live at distinct
    source rows.
    """
    new_sups = []
    i = 0
    while i < len(supervisions):
        sup = supervisions[i]
        if i + 1 < len(supervisions):
            next_sup = supervisions[i + 1]
            # Same timing -> merge
            if abs(sup.start - next_sup.start) < 0.01 and abs(sup.duration - next_sup.duration) < 0.01:
                new_sup = fastcopy(
                    sup,
                    translation=next_sup.text,
                    language=primary_language or sup.language,
                    target_lang=secondary_language,
                )
                new_sups.append(new_sup)
                i += 2
                continue
        new_sup = fastcopy(sup, language=primary_language or sup.language)
        new_sups.append(new_sup)
        i += 1
    return new_sups
