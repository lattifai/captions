"""Caption data structure for storing subtitle information with metadata."""

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


class BilingualMode(str, Enum):
    """Pre-merge bilingual layout detected on raw caption rows.

    ``NONE`` is the overwhelming common case — most captions are mono.
    ``LINE_BY_LINE`` and ``ALTERNATING`` reflect the two patterns subtitle
    groups use to ship two languages within a single file.
    """

    NONE = "none"
    LINE_BY_LINE = "line_by_line"
    ALTERNATING = "alternating"

if TYPE_CHECKING:
    from .config import ASSConfig, LRCConfig, SRTConfig, RenderConfig, StandardizationConfig
    from .formats.nle.audition import AuditionCSVConfig, EdiMarkerConfig
    from .formats.nle.avid import AvidDSConfig
    from .formats.nle.fcpxml import FCPXMLConfig
    from .formats.nle.premiere import PremiereXMLConfig
    from .formats.ttml import TTMLConfig

    FormatConfig = Union[
        ASSConfig, LRCConfig, SRTConfig, TTMLConfig,
        FCPXMLConfig, PremiereXMLConfig,
        AvidDSConfig, AuditionCSVConfig, EdiMarkerConfig,
    ]

from .config import InputCaptionFormat, OutputCaptionFormat  # noqa: F401
from .exceptions import CaptionParseError, FormatDetectionError, FormatNotSupportedError
from .formats import detect_format, detect_format_from_content, get_reader, get_writer
from .supervision import AlignmentItem, Pathlike, Supervision, fastcopy

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


@dataclass
class Caption:
    """Container for caption/subtitle data with metadata.

    Encapsulates a list of supervisions (subtitle segments) along with
    metadata such as language, kind, format information, and source file details.
    """

    supervisions: List[Supervision] = field(default_factory=list)
    """List of supervision segments containing text and timing information."""

    language: Optional[str] = None
    """Language code (e.g., 'en', 'zh', 'es')."""

    target_lang: Optional[str] = None
    """Target language code for translation."""

    kind: Optional[str] = None
    """Caption kind/type (e.g., 'captions', 'subtitles', 'descriptions')."""

    source_format: Optional[str] = None
    """Original format of the caption file (e.g., 'vtt', 'srt', 'json')."""

    source_path: Optional[Pathlike] = None
    """Path to the source caption file."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional custom metadata as key-value pairs."""

    def __len__(self) -> int:
        """Return the number of supervision segments."""
        return len(self.supervisions)

    def __iter__(self):
        """Iterate over supervision segments."""
        return iter(self.supervisions)

    def __getitem__(self, index):
        """Get supervision segment by index."""
        return self.supervisions[index]

    def __bool__(self) -> bool:
        """Return True if caption has supervisions."""
        return len(self) > 0

    @property
    def is_empty(self) -> bool:
        """Check if caption has no supervisions."""
        return len(self.supervisions) == 0

    @property
    def duration(self) -> Optional[float]:
        """
        Get total duration of the caption in seconds.

        Returns:
            Total duration from first to last supervision, or None if empty
        """
        if not self.supervisions:
            return None
        return self.supervisions[-1].end - self.supervisions[0].start

    @property
    def start_time(self) -> Optional[float]:
        """Get start time of first supervision."""
        if not self.supervisions:
            return None
        return self.supervisions[0].start

    @property
    def end_time(self) -> Optional[float]:
        """Get end time of last supervision."""
        if not self.supervisions:
            return None
        return self.supervisions[-1].end

    def append(self, supervision: Supervision) -> None:
        """Add a supervision segment to the caption."""
        self.supervisions.append(supervision)

    def extend(self, supervisions: List[Supervision]) -> None:
        """Add multiple supervision segments to the caption."""
        self.supervisions.extend(supervisions)

    def filter_by_speaker(self, speaker: str) -> "Caption":
        """
        Create a new Caption with only supervisions from a specific speaker.

        Args:
            speaker: Speaker identifier to filter by

        Returns:
            New Caption instance with filtered supervisions
        """
        filtered_sups = [sup for sup in self.supervisions if sup.speaker == speaker]
        return Caption(
            supervisions=filtered_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def get_speakers(self) -> List[str]:
        """
        Get list of unique speakers in the caption.

        Returns:
            Sorted list of unique speaker identifiers
        """
        speakers = {sup.speaker for sup in self.supervisions if sup.speaker}
        return sorted(speakers)

    @property
    def has_translation(self) -> bool:
        """Check if any supervision has translation data."""
        return any(sup.has_translation for sup in self.supervisions)

    def set_translations(self, translations: List[str], target_lang: Optional[str] = None) -> "Caption":
        """Set translations for supervisions.

        Args:
            translations: List of translated strings, one per supervision
            target_lang: Language code of the translations (e.g., 'zh', 'ja')

        Returns:
            Self for chaining
        """
        if len(translations) != len(self.supervisions):
            raise ValueError(
                f"Number of translations ({len(translations)}) must match "
                f"number of supervisions ({len(self.supervisions)})"
            )
        for sup, trans in zip(self.supervisions, translations):
            sup.translation = trans
            if target_lang:
                sup.target_lang = target_lang
        if target_lang:
            self.target_lang = target_lang
        return self

    def strip_translations(self) -> "Caption":
        """Remove all translation data from supervisions.

        Returns:
            Self for chaining
        """
        for sup in self.supervisions:
            sup.translation = None
            sup.target_lang = None
        self.target_lang = None
        return self

    def merge_bilingual(
        self,
        mode: str = "line_by_line",
        primary_language: Optional[str] = None,
        secondary_language: Optional[str] = None,
    ) -> "Caption":
        """Parse existing bilingual text into translation fields.

        Args:
            mode: "line_by_line" splits each supervision's text by newline
                  (first line -> text, second line -> translation);
                  "alternating" merges consecutive supervisions with same timing
                  (first -> text, second -> translation);
                  "auto" detects the pattern automatically:
                    1. ASS style names suggest language split -> alternating by style
                    2. Same-timing pairs with CJK vs Latin -> alternating
                    3. Text contains \\n with CJK/Latin split -> line_by_line
                    4. Otherwise -> no merge (monolingual)
            primary_language: Language code for the primary text
            secondary_language: Language code for the translation

        Returns:
            New Caption with translation fields populated
        """
        if mode == "auto":
            mode = self._detect_bilingual_mode()

        if mode == "line_by_line":
            new_sups = self._merge_line_by_line(primary_language, secondary_language)
        elif mode == "alternating":
            new_sups = self._merge_alternating(primary_language, secondary_language)
        elif mode == "none":
            # Monolingual: no merge needed, but stamp align_index so
            # extract_alignment_supervisions can still drive write-back.
            new_sups = []
            for i, sup in enumerate(self.supervisions):
                copy = fastcopy(sup, language=primary_language or sup.language)
                copy.align_index = i
                new_sups.append(copy)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'line_by_line', 'alternating', or 'none'.")

        return Caption(
            supervisions=new_sups,
            language=primary_language or self.language,
            target_lang=secondary_language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    @property
    def has_bilingual_layout(self) -> bool:
        """``True`` if the raw caption shows a bilingual structure.

        Derived from :meth:`detect_bilingual_mode` — non-NONE modes flip
        this to ``True``. Distinct from :attr:`has_translation`, which
        reports the post-merge state (``Supervision.translation`` set).

        The method (not a cached property) — ``Caption`` is mutable, so
        any cache would have to be invalidated on every supervision edit
        and that's strictly more dangerous than re-running the cheap
        scan. Callers that need to ask repeatedly should bind the result
        to a local.
        """
        return self.detect_bilingual_mode() != BilingualMode.NONE

    def detect_bilingual_mode(self) -> BilingualMode:
        """Detect the bilingual arrangement of the raw caption (pre-merge).

        Priority (most specific → most ambiguous):

        1. Text contains ``\\n`` with CJK/Latin split → ``LINE_BY_LINE``.
           Strongest signal: an explicit within-cue break with different
           scripts on each side. Checked first so ASS files whose dialogue
           uses inline ``\\N`` (F1) aren't misclassified as dual-row just
           because a handful of sign/title rows share a style name with
           dialogue.
        2. Same-timing pairs with different CJK ratios → ``ALTERNATING``.
        3. ASS style names correlate with different languages → ``ALTERNATING``.
        4. Otherwise → ``NONE`` (monolingual).

        Each branch enforces a ≥ 20 % coverage floor so a few stray
        bilingual-shaped rows in an otherwise-mono caption don't flip
        the whole file to bilingual.

        Not cached — see :attr:`has_bilingual_layout`.
        """
        from .parsers.text_parser import cjk_ratio

        sups = self.supervisions
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
            return BilingualMode.ALTERNATING

        # 3. ASS style-based split (e.g., "中文 1080" vs "英文 1080")
        #
        #    Both the high-CJK and low-CJK style must individually cover
        #    ≥ 20 % of all supervisions. Without this floor, a mono CJK
        #    caption with a handful of Latin "Sign" / "Title" rows trips
        #    ALTERNATING because the Sign style averages cjk_ratio ≈ 0
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
                return BilingualMode.ALTERNATING

        return BilingualMode.NONE

    def _detect_bilingual_mode(self) -> str:
        """Deprecated string-returning alias of :meth:`detect_bilingual_mode`.

        Removed in Step 7 of the bilingual refactor — Step 4-6 callers
        still expect ``str``.
        """
        return self.detect_bilingual_mode().value

    def extract_alignment_supervisions(
        self,
    ) -> "tuple[list[Supervision], list[Supervision]]":
        """Extract per-language alignable supervisions from a (possibly
        bilingual) caption.

        Returns
        -------
        (primary_sups, secondary_sups)
            ``primary_sups`` holds the detected primary-language rows, in
            source order. ``secondary_sups`` holds the other language for
            bilingual captions; ``[]`` for mono.

        Each returned ``Supervision`` is a ``fastcopy`` carrying:
          - ``text``: plaintext in that language (ASS override tags stripped)
          - ``language``: ISO-639-1 decided by *group-level aggregated
            voting*: all plaintext strings in the group are joined and fed
            to lingua once. Long aggregated text scores ~98.5 % accuracy on
            FLORES (vs. 95.4 % for per-row lookups), so this is markedly
            more robust than running lingua on each short dialogue line.
          - ``align_index`` (on ``sup.custom``): 0-based index back into
            ``self.supervisions`` for ``apply_alignment`` to write results.
            F1 inline shares one index across both sides; F2 dual-row
            independently uses each row's original position.
          - ``start`` / ``duration``: original timestamps (unchanged).

        Excluded from both lists:
          - non-dialogue rows (``classify_line_type`` ≠ ``None``: staff_credit,
            karaoke, sign, title, banner, translator_note, branding, drawing)
          - zero/near-zero-duration rows (``duration ≤ 0.01s``) — these get
            re-timed by neighbour interpolation inside ``apply_alignment``.

        Safety layers (prevent misclassification from slipping through):
          1. Aggregated voting (this function).
          2. Same-script rollback: if the "secondary" group is detected to
             share the alignment script with the primary (e.g. short-text
             Latin→de/fr/nl outliers, or a mono CJK file erroneously split
             by an F1 heuristic on number-heavy cues), merge secondary
             back into primary and treat as mono. CJK siblings (zh/ja/ko)
             are exempt — lingua disambiguates them reliably on full-caption
             aggregates, so anime-style zh+ja/zh+ko/ja+ko captions stay
             bilingual.

        ``self`` is not mutated.
        """
        from .parsers.language_detector import detect_language, detect_script
        from .parsers.text_parser import _BRANDING_KEYWORDS, _STAFF_ROLES, classify_line_type

        def _strip(text: str) -> str:
            # Strip ASS override tags, then collapse embedded whitespace
            # (including soft-wrap newlines) to single spaces. Alignment
            # operates on the token stream, not the rendered layout, so
            # ``医生\n先生`` and ``What a lovely day\nit is today.`` are
            # both just single-line inputs with a stray break that
            # should not reach the lattice tokenizer.
            cleaned = _ALIGNMENT_OVERRIDE_RE.sub("", text or "")
            return " ".join(cleaned.split())

        def _is_alignable(sup: Supervision, plain: str) -> bool:
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

        def _vote_lang(texts: List[str]) -> "Optional[str]":
            """Aggregate-voting language detection for a group of texts."""
            if not texts:
                return None
            joined = " ".join(texts)
            return detect_language(joined) or detect_script(joined)

        def _same_alignment_script(a: "Optional[str]", b: "Optional[str]") -> bool:
            """Are two lang labels confusable for alignment purposes?

            True when they map to the same coarse script bucket AND they
            aren't distinct CJK siblings (zh/ja/ko disambiguate cleanly on
            full-caption aggregates).
            """
            if not a or not b:
                return False
            if a in _ALIGNMENT_CJK_DISTINCT and b in _ALIGNMENT_CJK_DISTINCT and a != b:
                return False
            return _ALIGNMENT_SCRIPT_BUCKETS.get(a, "latin") == _ALIGNMENT_SCRIPT_BUCKETS.get(b, "latin")

        def _can_fast_extract_mono() -> bool:
            if self.source_format in {"ass", "ssa"}:
                return False

            prev_start = None
            prev_duration = None
            for sup in self.supervisions:
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

        def _compute_break_before() -> List[bool]:
            """Mark source rows whose preceding gap exceeds the adaptive threshold.

            The adaptive threshold is ``max(2.0, min(5.0, 3 × median_positive_gap))``,
            clamped to the 2-5 s window that accommodates both tight dialogue
            and typical scene/commercial cuts. Same-timing neighbours (F2
            candidates) are never marked as boundaries — their gap is ~0 by
            definition and F2 must stay contiguous. The output is per-source-row
            and later stamped onto each extracted side via
            ``Supervision.custom["alignment_break_before"]`` so that
            ``apply_alignment`` can refuse to interpolate across the boundary.
            """
            sups = self.supervisions
            n = len(sups)
            if n < 2:
                return [False] * n
            gaps: List[float] = []
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
                # Same-timing pairs are the F2 atomic unit; never split them.
                if abs((sups[j].start or 0.0) - (sups[j - 1].start or 0.0)) < 0.01:
                    continue
                prev_end = (sups[j - 1].start or 0.0) + (sups[j - 1].duration or 0.0)
                g = (sups[j].start or 0.0) - prev_end
                if g >= threshold:
                    out[j] = True
            return out

        break_before = _compute_break_before()

        primary, secondary = [], []
        primary_texts, secondary_texts = [], []

        if _can_fast_extract_mono():
            for i, sup in enumerate(self.supervisions):
                text = _strip(sup.text)
                side = fastcopy(sup, text=text, translation=None, custom=dict(sup.custom or {}))
                side.align_index = i
                side.alignment_break_before = break_before[i]
                primary.append(side)
                primary_texts.append(text)
            p_lang = _vote_lang(primary_texts)
            for sup in primary:
                sup.language = p_lang
            return primary, []

        mode = self._detect_bilingual_mode()
        i = 0

        while i < len(self.supervisions):
            sup = self.supervisions[i]
            step = 1
            raw_primary = sup.text or ""
            raw_secondary = ""
            secondary_index = i

            if mode == "line_by_line":
                lines = raw_primary.split("\n")
                if len(lines) >= 2:
                    raw_primary = lines[0]
                    raw_secondary = lines[1]
            elif mode == "alternating" and i + 1 < len(self.supervisions):
                next_sup = self.supervisions[i + 1]
                if abs(sup.start - next_sup.start) < 0.01 and abs(sup.duration - next_sup.duration) < 0.01:
                    raw_secondary = next_sup.text or ""
                    secondary_index = i + 1
                    step = 2

            t1 = _strip(raw_primary)
            t2 = _strip(raw_secondary)
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
            side.align_index = i
            side.alignment_break_before = break_before[i]
            primary.append(side)
            primary_texts.append(t1)
            if t2:
                side = fastcopy(sup, text=t2, translation=None, custom=dict(base_custom))
                side.align_index = secondary_index
                side.alignment_break_before = break_before[secondary_index]
                secondary.append(side)
                secondary_texts.append(t2)
            i += step

        # ---- Layer 1: aggregate voting ----
        p_lang = _vote_lang(primary_texts)
        s_lang = _vote_lang(secondary_texts) if secondary_texts else None

        # ---- Layer 2: same-script rollback (mono mis-split as bilingual) ----
        if s_lang and _same_alignment_script(p_lang, s_lang):
            # Merge by align_index (not naive extend) so F2-style splits
            # that happen to share the same script (e.g. pure-JP captions
            # with stray same-timing pairs) recover in source order.
            merged = sorted(
                zip(primary + secondary, primary_texts + secondary_texts),
                key=lambda pt: pt[0].align_index,
            )
            primary = [p for p, _ in merged]
            primary_texts = [t for _, t in merged]
            secondary, secondary_texts = [], []
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
            primary_indexes = {s.align_index for s in primary}
            is_f1 = all(s.align_index in primary_indexes for s in secondary)
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
                    # align_index instead so downstream consumers
                    # (apply_alignment, audits) see a clean non-
                    # decreasing index sequence.
                    merged = sorted(
                        zip(primary + secondary, primary_texts + secondary_texts),
                        key=lambda pt: pt[0].align_index,
                    )
                    primary = [p for p, _ in merged]
                    primary_texts = [t for _, t in merged]
                secondary, secondary_texts = [], []
                p_lang = _vote_lang(primary_texts)
                s_lang = None

        for sup in primary:
            sup.language = p_lang
        for sup in secondary:
            sup.language = s_lang

        return primary, secondary

    def apply_alignment(self, aligned: List[Supervision]) -> None:
        """Write aligned timestamps back into ``self.supervisions`` in place.

        Parameters
        ----------
        aligned :
            Supervisions produced by the force aligner. Each row carries
            the ``align_index`` (in ``sup.custom``) set by
            ``extract_for_alignment`` — the 0-based index into the original
            ``self.supervisions``. Updated timing is on ``start`` /
            ``duration`` and, optionally, word-level alignment under
            ``alignment["word"]``.

        Behaviour
        ---------
        1. **index-matched write-back** — for each row in ``aligned``, read
           its ``align_index`` and update the row at that index in
           ``self.supervisions``. Rows without a valid index are silently
           skipped (defensive); rows in ``self`` whose index is never
           written are left untouched.

        2. **No-timing interpolation** — afterwards, any dialogue row in
           ``self`` with ``duration ≤ 0.01`` is re-timed by linearly
           interpolating between the closest aligned neighbours on either
           side. A row is "dialogue" iff ``classify_line_type`` returns
           ``None`` for it (i.e. it isn't a staff_credit / sign / title /
           karaoke / banner / translator_note / branding / drawing row).
           Rows in the middle of a run of zero-duration dialogue are
           distributed uniformly across the available gap.
        """
        sups = self.supervisions
        n = len(sups)
        if n == 0:
            return

        def _write_back(target: Supervision, aligned_sup: Supervision) -> None:
            target.start = aligned_sup.start
            target.duration = aligned_sup.duration
            if aligned_sup.alignment is not None:
                target.alignment = {"word": aligned_sup.alignment.get("word")}

        # Fast path: every row already has usable timing and alignment covers
        # the whole caption one-to-one, so simple index-matched write-back is enough.
        fast_path = len(aligned) == n and all(sup.duration is not None and sup.duration > 0.01 for sup in sups)
        if fast_path:
            seen = [False] * n
            for aligned_sup in aligned:
                idx = (aligned_sup.custom or {}).get("align_index")
                if idx is None or idx < 0 or idx >= n or seen[idx]:
                    fast_path = False
                    break
                seen[idx] = True
            if fast_path:
                for aligned_sup in aligned:
                    _write_back(sups[aligned_sup.align_index], aligned_sup)
                return

        from .parsers.text_parser import classify_line_type

        # ---- Step 1: index-matched timestamp copy ----
        written = [False] * n
        break_before = [False] * n
        for aligned_sup in aligned:
            idx = (aligned_sup.custom or {}).get("align_index")
            if idx is None or idx < 0 or idx >= n:
                continue
            _write_back(sups[idx], aligned_sup)
            written[idx] = True
            # Segment boundary signal stamped by extract_alignment_supervisions.
            # Blocks cross-boundary interpolation in Step 2 below.
            if (aligned_sup.custom or {}).get("alignment_break_before"):
                break_before[idx] = True

        # ---- Step 2: interpolate no-timing dialogue rows ----
        timed = [False] * n
        dialogue = [False] * n
        prev_timed = [-1] * n
        next_timed = [-1] * n

        last_timed = -1
        for i, sup in enumerate(sups):
            # Break boundary: rows before row ``i`` cannot borrow a right
            # anchor past this point, so reset the running left anchor.
            if break_before[i]:
                last_timed = -1
            prev_timed[i] = last_timed
            has_timing = sup.duration is not None and sup.duration > 0.01
            timed[i] = has_timing
            if has_timing:
                last_timed = i
                continue
            if written[i]:
                continue
            custom = sup.custom or {}
            if custom.get("line_type") == "drawing":
                continue
            dialogue[i] = classify_line_type(
                sup.text or "",
                start=sup.start,
                ass_raw_text=custom.get("ass_raw_text"),
                duration=sup.duration,
            ) is None

        last_timed = -1
        for i in range(n - 1, -1, -1):
            next_timed[i] = last_timed
            if timed[i]:
                last_timed = i
            # Same signal from the right-scan side: post-boundary timed
            # rows must not serve as a left anchor for the pre-boundary
            # zero-duration run.
            if break_before[i]:
                last_timed = -1

        i = 0
        while i < n:
            if not dialogue[i]:
                i += 1
                continue

            j = i + 1
            # Dialogue runs cannot extend across a break boundary either.
            while j < n and dialogue[j] and not break_before[j]:
                j += 1

            left_idx = prev_timed[i]
            right_idx = next_timed[j - 1]
            run_len = j - i

            if left_idx != -1 and right_idx != -1:
                gap_start = round(sups[left_idx].start + sups[left_idx].duration, 4)
                span = round(max(sups[right_idx].start - gap_start, 0.0), 4)
                slot = round(span / run_len, 4)
                for offset, idx in enumerate(range(i, j)):
                    sups[idx].start = round(gap_start + offset * slot, 4)
                    sups[idx].duration = slot
            elif left_idx != -1:
                start = round(sups[left_idx].start + sups[left_idx].duration, 4)
                for idx in range(i, j):
                    sups[idx].start = start
                    sups[idx].duration = 0.0
            elif right_idx != -1:
                start = round(sups[right_idx].start, 4)
                for idx in range(i, j):
                    sups[idx].start = start
                    sups[idx].duration = 0.0

            i = j

    def _merge_line_by_line(
        self, primary_language: Optional[str], secondary_language: Optional[str]
    ) -> List[Supervision]:
        """Split each supervision's text by newline into text + translation.

        Stamps ``sup.custom["align_index"]`` with the original row index so
        ``extract_alignment_supervisions`` / ``apply_alignment`` can write
        results back by position (F1 inline — both sides share the same
        source row, so one index serves both).
        """
        new_sups = []
        for orig_idx, sup in enumerate(self.supervisions):
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
            new_sup.align_index = orig_idx
            new_sups.append(new_sup)
        return new_sups

    def _merge_alternating(
        self, primary_language: Optional[str], secondary_language: Optional[str]
    ) -> List[Supervision]:
        """Merge consecutive same-timing supervisions into text + translation.

        Stamps ``sup.custom["align_index"]`` with the row index of the merged
        primary side; when two rows are fused (F2 dual-row), the secondary
        side's original row index is stamped on
        ``sup.custom["translation_align_index"]`` so write-back can target
        each language's original row independently.
        """
        new_sups = []
        i = 0
        while i < len(self.supervisions):
            sup = self.supervisions[i]
            if i + 1 < len(self.supervisions):
                next_sup = self.supervisions[i + 1]
                # Same timing -> merge
                if abs(sup.start - next_sup.start) < 0.01 and abs(sup.duration - next_sup.duration) < 0.01:
                    new_sup = fastcopy(
                        sup,
                        translation=next_sup.text,
                        language=primary_language or sup.language,
                        target_lang=secondary_language,
                    )
                    new_sup.align_index = i
                    new_sup.translation_align_index = i + 1
                    new_sups.append(new_sup)
                    i += 2
                    continue
            new_sup = fastcopy(sup, language=primary_language or sup.language)
            new_sup.align_index = i
            new_sups.append(new_sup)
            i += 1
        return new_sups

    def shift_time(self, seconds: float) -> "Caption":
        """
        Create a new Caption with all timestamps shifted by given seconds.

        Args:
            seconds: Number of seconds to shift (positive delays, negative advances)

        Returns:
            New Caption instance with shifted timestamps
        """
        shifted_sups = []
        for sup in self.supervisions:
            # Calculate physical time range
            raw_start = sup.start + seconds
            raw_end = sup.end + seconds

            # Skip segments that end before 0
            if raw_end <= 0:
                continue

            # Clip start to 0 if negative
            if raw_start < 0:
                final_start = 0.0
                final_duration = raw_end
            else:
                final_start = raw_start
                final_duration = sup.duration

            # Handle alignment (word-level timestamps)
            final_alignment = None
            original_alignment = getattr(sup, "alignment", None)
            if original_alignment and "word" in original_alignment:
                new_words = []
                for word in original_alignment["word"]:
                    w_start = word.start + seconds
                    w_end = w_start + word.duration

                    # Skip words that end before 0
                    if w_end <= 0:
                        continue

                    # Clip start to 0 if negative
                    if w_start < 0:
                        w_final_start = 0.0
                        w_final_duration = w_end
                    else:
                        w_final_start = w_start
                        w_final_duration = word.duration

                    new_words.append(
                        AlignmentItem(
                            symbol=word.symbol,
                            start=w_final_start,
                            duration=w_final_duration,
                            score=word.score,
                        )
                    )

                # Copy original alignment dict structure and update words
                final_alignment = original_alignment.copy()
                final_alignment["word"] = new_words

            shifted_sups.append(
                Supervision(
                    text=sup.text,
                    start=final_start,
                    duration=final_duration,
                    speaker=sup.speaker,
                    id=sup.id,
                    recording_id=sup.recording_id if hasattr(sup, "recording_id") else "",
                    channel=getattr(sup, "channel", 0),
                    language=sup.language,
                    alignment=final_alignment,
                    custom=sup.custom,
                )
            )

        return Caption(
            supervisions=shifted_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def with_margins(
        self,
        start_margin: float = 0.10,
        end_margin: float = 0.10,
        min_gap: float = 0.08,
        collision_mode: str = "trim",
    ) -> "Caption":
        """
        Create a new Caption with segment boundaries adjusted based on word-level alignment.

        Uses supervision.alignment['word'] to recalculate segment start/end times
        with the specified margins applied around the actual speech boundaries.

        Args:
            start_margin: Seconds to extend before the first word (default: 0.10)
            end_margin: Seconds to extend after the last word (default: 0.10)
            min_gap: Minimum gap between segments for collision handling (default: 0.08)
            collision_mode: How to handle segment overlap - 'trim' or 'gap' (default: 'trim')

        Returns:
            New Caption instance with adjusted timestamps

        Note:
            Segments without alignment data will keep their original timestamps.

        Example:
            >>> caption = Caption.read("aligned.srt")
            >>> adjusted = caption.with_margins(start_margin=0.05, end_margin=0.15)
            >>> adjusted.write("output.srt")
        """
        from .standardize import apply_margins_to_captions

        adjusted_sups = apply_margins_to_captions(
            self.supervisions,
            start_margin=start_margin,
            end_margin=end_margin,
            min_gap=min_gap,
            collision_mode=collision_mode,
        )

        return Caption(
            supervisions=adjusted_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def to_string(
        self,
        format: str = "srt",
        render: Optional["RenderConfig"] = None,
        format_config: Optional["FormatConfig"] = None,
    ) -> str:
        """
        Return caption content in specified format.

        Args:
            format: Output format (e.g., 'srt', 'vtt', 'ass')
            render: RenderConfig controlling rendering and output behavior
            format_config: Format-specific configuration (ASSConfig, TTMLConfig, etc.)

        Returns:
            String containing formatted captions
        """
        return self.to_bytes(output_format=format, render=render, format_config=format_config).decode("utf-8")

    def to_dict(self) -> Dict:
        """
        Convert Caption to dictionary representation.

        Returns:
            Dictionary with caption data and metadata
        """
        return {
            "supervisions": [sup.to_dict() for sup in self.supervisions],
            "language": self.language,
            "target_lang": self.target_lang,
            "kind": self.kind,
            "source_format": self.source_format,
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata,
            "duration": self.duration,
            "num_segments": len(self.supervisions),
            "speakers": self.get_speakers(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Caption":
        """Create Caption from a dictionary (inverse of to_dict).

        Accepts the same structure as to_dict() output and CaptionData schema.
        Ignores computed fields (duration, num_segments, speakers).

        Args:
            data: Dictionary with caption fields.

        Returns:
            New Caption instance.
        """
        sups = data.get("supervisions", [])
        supervisions = [
            Supervision.from_dict(s) if isinstance(s, dict) else s
            for s in sups
        ]
        return cls(
            supervisions=supervisions,
            language=data.get("language"),
            target_lang=data.get("target_lang"),
            kind=data.get("kind"),
            source_format=data.get("source_format"),
            source_path=data.get("source_path"),
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def from_supervisions(
        cls,
        supervisions: List[Supervision],
        language: Optional[str] = None,
        target_lang: Optional[str] = None,
        kind: Optional[str] = None,
        source_format: Optional[str] = None,
        source_path: Optional[Pathlike] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Caption":
        """
        Create Caption from a list of supervisions.

        Args:
            supervisions: List of supervision segments
            language: Language code
            target_lang: Target language code for bilingual captions
            kind: Caption kind/type
            source_format: Original format
            source_path: Source file path
            metadata: Additional metadata

        Returns:
            New Caption instance
        """
        return cls(
            supervisions=supervisions,
            language=language,
            target_lang=target_lang,
            kind=kind,
            source_format=source_format,
            source_path=source_path,
            metadata=metadata or {},
        )

    @classmethod
    def from_string(
        cls,
        content: str,
        format: Optional[str] = None,
        normalize_text: bool = True,
    ) -> "Caption":
        """
        Create Caption from string content.

        Args:
            content: Caption content as string
            format: Caption format (e.g., 'srt', 'vtt', 'ass').
                Auto-detected from content when omitted.
            normalize_text: Whether to normalize text during reading

        Returns:
            New Caption instance

        Raises:
            FormatDetectionError: If format cannot be auto-detected from content.
            FormatNotSupportedError: If the specified format has no registered reader.
            CaptionParseError: If the reader fails to parse the content.

        Example:
            >>> caption = Caption.from_string(srt_content)          # auto-detect
            >>> caption = Caption.from_string(srt_content, "srt")   # explicit
        """
        if not format or format == "auto":
            format = detect_format_from_content(content)
            if not format:
                raise FormatDetectionError(
                    "Unable to detect caption format from content. "
                    "Please specify the 'format' parameter explicitly."
                )

        reader_cls = get_reader(format)
        if not reader_cls:
            from .formats.pysubs2 import Pysubs2Format

            reader_cls = Pysubs2Format

        # Sentinel newline: readers use "\n" presence to distinguish
        # content from file paths. Without this, single-line content
        # (e.g., short CJK text) would be misidentified.
        if "\n" not in content:
            content += "\n"

        try:
            result = reader_cls.parse(content, normalize_text=normalize_text)
        except (FormatDetectionError, FormatNotSupportedError):
            raise
        except Exception as exc:
            raise CaptionParseError(f"Failed to parse {format} content: {exc}") from exc

        return cls(
            supervisions=result.supervisions,
            language=result.language,
            target_lang=result.target_lang,
            kind=result.kind,
            source_format=format,
            metadata=result.format_metadata,
        )

    def to_bytes(
        self,
        output_format: Optional[str] = None,
        render: Optional["RenderConfig"] = None,
        format_config: Optional["FormatConfig"] = None,
    ) -> bytes:
        """
        Convert caption to bytes.

        Args:
            output_format: Output format (e.g., 'srt', 'vtt', 'ass'). Defaults to source_format or 'srt'
            render: RenderConfig controlling rendering and output behavior
            format_config: Format-specific configuration (ASSConfig, TTMLConfig, etc.)

        Returns:
            Caption content as bytes

        Example:
            >>> caption = Caption.read("input.srt")
            >>> data = caption.to_bytes()
            >>> vtt_data = caption.to_bytes(output_format="vtt")
        """
        return self.write(
            None,
            format_config=format_config,
            render=render,
            _output_format=output_format,
        )

    @classmethod
    def read(
        cls,
        path: Union[Pathlike, io.BytesIO, io.StringIO],
        format: Optional[str] = None,
        normalize_text: bool = True,
        encoding: str = "utf-8",
    ) -> "Caption":
        """
        Read caption file or in-memory data and return Caption object.

        Args:
            path: Path to caption file, or BytesIO/StringIO object with caption content.
            format: Caption format. Auto-detected from file extension or content
                when omitted.
            normalize_text: Whether to normalize text during reading.
            encoding: Character encoding for BytesIO / file reading (default utf-8).

        Returns:
            Caption object containing supervisions and metadata.

        Raises:
            ValueError: If format cannot be determined.
            FileNotFoundError: If file path does not exist.
        """
        source_path: Optional[str] = None
        detected_encoding: Optional[str] = None

        # --- Load content into memory string ---
        if isinstance(path, (io.BytesIO, io.StringIO)):
            content = path.read().decode(encoding, errors="replace") if isinstance(path, io.BytesIO) else path.read()
        else:
            file_path = Path(str(path))
            if not file_path.is_file():
                raise FileNotFoundError(f"Caption file not found: {file_path}")
            source_path = str(file_path)
            if not format or format == "auto":
                format = detect_format(source_path) or file_path.suffix.lstrip(".").lower()
            # Use encoding detection for robust handling of UTF-16/GBK/GB18030 files.
            # Pure-utf-8 files round-trip through the BOM branch unchanged.
            from .formats.pysubs2 import detect_file_encoding

            content, detected_encoding = detect_file_encoding(file_path)

        # --- Resolve format: explicit > file extension > content sniffing ---
        if not format or format == "auto":
            format = detect_format_from_content(content)

        # --- Parse ---
        caption = cls.from_string(content, format=format, normalize_text=normalize_text)
        caption.source_path = source_path
        # Preserve the real on-disk encoding for downstream consumers (e.g.
        # roundtripping back to the original file encoding). parse()'s
        # from_string branch can't see it because the string is already decoded.
        if detected_encoding and "encoding" not in caption.metadata:
            caption.metadata["encoding"] = detected_encoding

        # Detect dominant line terminator so writers can preserve Windows-style
        # (CRLF) files (common in ASS output from Arctime/Aegisub on Windows).
        # Without this, pysubs2's ``\n`` output silently flips CRLF to LF and
        # turns a byte-faithful roundtrip into a full-file diff.
        if content and "line_terminator" not in caption.metadata:
            crlf = content.count("\r\n")
            total_lf = content.count("\n")
            bare_lf = total_lf - crlf
            if crlf > bare_lf and crlf > 0:
                caption.metadata["line_terminator"] = "\r\n"
            elif bare_lf > 0:
                caption.metadata["line_terminator"] = "\n"

        # P2-3: detect language from filename when not already set by the reader.
        # Patterns like ".简体中文&英文.ass" / ".CN&EN.srt" / ".双语.ass" are
        # free metadata available at zero parse cost.
        if not caption.language and source_path:
            from .parsers.text_parser import detect_language_from_filename

            lang, tgt = detect_language_from_filename(source_path)
            if lang:
                caption.language = lang
            if tgt and not caption.target_lang:
                caption.target_lang = tgt

        return caption

    def write(
        self,
        path: Union[Pathlike, io.BytesIO, None] = None,
        format_config: Optional["FormatConfig"] = None,
        render: Optional["RenderConfig"] = None,
        standardization: Optional["StandardizationConfig"] = None,
        _output_format: Optional[str] = None,
    ) -> Union[Pathlike, bytes]:
        """
        Write caption to file or return as bytes.

        Args:
            path: Path to output caption file, BytesIO object, or None to return bytes
            format_config: Format-specific configuration (ASSConfig, TTMLConfig, etc.)
            render: RenderConfig controlling include_speaker, word_level, translation_first
            standardization: Broadcast standardization (min/max duration, CPS, margins)

        Returns:
            Path to the written file if path is a file path, or bytes if path is BytesIO/None
        """
        from .config import ASSConfig, RenderConfig, apply_color_scheme

        effective_render = render or RenderConfig()

        # Apply karaoke color scheme from ASSConfig
        if isinstance(format_config, ASSConfig) and format_config.karaoke_color_scheme:
            format_config = apply_color_scheme(format_config.karaoke_color_scheme, format_config)

        supervisions = self.supervisions

        # Apply broadcast standardization if configured
        if standardization:
            from .standardize import CaptionStandardizer

            standardizer = CaptionStandardizer(
                min_duration=standardization.min_duration,
                max_duration=standardization.max_duration,
                min_gap=standardization.min_gap,
                max_lines=standardization.max_lines,
                max_chars_per_line=standardization.max_chars_per_line,
            )
            supervisions = standardizer.process(supervisions)
            if standardization.start_margin is not None:
                supervisions = standardizer.apply_margins(
                    supervisions,
                    start_margin=standardization.start_margin,
                    end_margin=standardization.end_margin or 0.10,
                )

        # Roundtrip metadata: merge format_metadata with Caption-level attrs
        # so writers (e.g., VTT) can access kind/language from the metadata dict.
        effective_metadata = dict(self.metadata) if self.metadata else {}
        if self.kind:
            effective_metadata.setdefault("kind", self.kind)
        if self.language:
            effective_metadata.setdefault("language", self.language)

        # For JSON format: build full Caption-level metadata dict
        caption_level_metadata = {
            "language": self.language,
            "target_lang": self.target_lang,
            "kind": self.kind,
            "source_format": self.source_format,
            "metadata": effective_metadata,
        }
        caption_level_metadata = {k: v for k, v in caption_level_metadata.items() if v is not None}

        # Determine output format: explicit > path extension > source format > "srt"
        if _output_format:
            fmt = _output_format.lower()
        elif isinstance(path, (io.BytesIO, type(None))):
            fmt = self.source_format or "srt"
        else:
            fmt = detect_format(str(path)) or Path(str(path)).suffix.lstrip(".").lower() or "srt"

        # Special casing for professional formats
        ext = fmt
        if isinstance(path, (str, Path)):
            path_str = str(path)
            if path_str.endswith("_avid.txt"):
                ext = "avid_ds"
            elif "audition" in path_str.lower() and path_str.endswith(".csv"):
                ext = "audition_csv"
            elif "edimarker" in path_str.lower() and path_str.endswith(".csv"):
                ext = "edimarker_csv"
            elif "imsc" in path_str.lower() and path_str.endswith(".ttml"):
                ext = "imsc1"
            elif "ebu" in path_str.lower() and path_str.endswith(".ttml"):
                ext = "ebu_tt_d"

        writer_cls = get_writer(ext)
        if not writer_cls:
            from .formats.pysubs2 import Pysubs2Format

            writer_cls = Pysubs2Format

        writer_metadata = caption_level_metadata if ext == "json" else effective_metadata

        if isinstance(path, (str, Path)):
            return writer_cls.write(
                supervisions,
                path,
                metadata=writer_metadata,
                render=effective_render,
                config=format_config,
            )

        content = writer_cls.to_bytes(
            supervisions,
            metadata=writer_metadata,
            render=effective_render,
            config=format_config,
        )
        if isinstance(path, io.BytesIO):
            path.write(content)
            path.seek(0)
        return content

    def __repr__(self) -> str:
        """String representation of Caption."""
        lang = f"lang={self.language}" if self.language else "lang=unknown"
        kind_str = f"kind={self.kind}" if self.kind else ""
        parts = [f"Caption({len(self.supervisions)} segments", lang]
        if kind_str:
            parts.append(kind_str)
        if self.duration:
            parts.append(f"duration={self.duration:.2f}s")
        return ", ".join(parts) + ")"
