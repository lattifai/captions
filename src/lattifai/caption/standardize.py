"""
Caption Standardization Module

Implements broadcast-grade caption standardization following Netflix/BBC guidelines:
- Timeline cleanup (min/max duration, gap checking)
- Smart text line breaking
- Quality validation

Reference Standards:
- Netflix Timed Text Style Guide
- BBC Subtitle Guidelines
- EBU-TT-D Standard
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from .config import StandardizationConfig
from .supervision import Supervision, fastcopy

# Zero-width joiner used to glue emoji sequences (👨‍👩‍👧) into one grapheme.
_ZWJ = "\u200d"

__all__ = [
    "CaptionStandardizer",
    "CaptionValidator",
    "StandardizationConfig",
    "ValidationResult",
    "standardize_captions",
    "apply_margins_to_captions",
]


@dataclass
class ValidationResult:
    """Validation result."""

    valid: bool = True
    """Whether all validations passed"""

    warnings: List[str] = field(default_factory=list)
    """List of warning messages"""

    # Statistics
    avg_cps: float = 0.0
    """Average reading speed (chars/sec)"""

    max_cpl: int = 0
    """Maximum characters per line"""

    segments_too_short: int = 0
    """Number of segments too short"""

    segments_too_long: int = 0
    """Number of segments too long"""

    gaps_too_small: int = 0
    """Number of gaps too small"""


class CaptionStandardizer:
    """
    Caption standardization processor.

    Processing flow:
    1. Timeline cleanup - Adjust duration and gaps
    2. Text formatting - Smart line breaking
    3. Validation - Generate quality metrics

    Example:
        >>> standardizer = CaptionStandardizer(min_duration=0.8, max_chars_per_line=42)
        >>> processed = standardizer.process(supervisions)
    """

    # Chinese/Japanese punctuation (for line break priority)
    # Note: '' (U+2018/2019 curly quotes) excluded — they appear in English
    # contractions (they're, can't, it's) and must NOT trigger word splitting.
    CJK_PUNCTUATION = (
        r"[，。、？！：；·…—～" "（）【】〔〕〖〗《》〈〉「」『』〘〙〚〛]"
    )

    # English/Western punctuation
    EN_PUNCTUATION = r"[,.!?;:\-–—«»‹›]"

    # All splittable punctuation (for line break search)
    ALL_PUNCTUATION = r"[，。、？！：；·…—～,.!?;:\-–—\s]"

    def __init__(
        self,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        min_gap: float = 0.08,
        max_lines: int = 2,
        max_chars_per_line: int = 42,
    ):
        """
        Initialize standardizer.

        Args:
            min_duration: Minimum duration (seconds)
            max_duration: Maximum duration (seconds)
            min_gap: Minimum gap (seconds)
            max_lines: Maximum number of lines
            max_chars_per_line: Maximum characters per line
        """
        self.config = StandardizationConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_gap=min_gap,
            max_lines=max_lines,
            max_chars_per_line=max_chars_per_line,
        )

    def process(self, segments: List[Supervision]) -> List[Supervision]:
        """
        Main processing entry point.

        Args:
            segments: List of original caption segments

        Returns:
            List of processed caption segments
        """
        if not segments:
            return []

        # 1. Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start)

        # 2. Timeline cleanup
        processed = self._sanitize_timeline(sorted_segments)

        # 3. Split oversized segments (text exceeds max_lines × max_chars_per_line)
        processed = self._split_long_segments(processed)

        # 4. Text formatting (line breaks within each segment)
        processed = self._format_texts(processed)

        return processed

    def _sanitize_timeline(self, segments: List[Supervision]) -> List[Supervision]:
        """
        Timeline cleanup.

        Processing logic:
        A. Gap check - Ensure sufficient gap between subtitles
        B. Min duration check - Extend too-short subtitles
        C. Max duration check - Truncate too-long subtitles

        Priority: Gap > Min duration (insufficient gap causes display issues)
        """
        result: List[Supervision] = []

        for i, seg in enumerate(segments):
            # Create new instance
            new_seg = self._copy_segment(seg)

            # A. Check gap with previous subtitle
            if result:
                prev_seg = result[-1]
                prev_end = prev_seg.start + prev_seg.duration
                gap = new_seg.start - prev_end

                if gap < self.config.min_gap:
                    # Gap too small or overlap
                    # Target: prev_end_new + min_gap = new_seg.start
                    # => prev_duration_new = new_seg.start - min_gap - prev_seg.start
                    target_prev_duration = (
                        new_seg.start - self.config.min_gap - prev_seg.start
                    )

                    if target_prev_duration >= self.config.min_duration:
                        # Safe to shorten previous subtitle (still meets min duration)
                        result[-1] = self._copy_segment(
                            prev_seg, duration=target_prev_duration
                        )
                    else:
                        # Shortening previous would go below min duration, delay current start
                        new_start = prev_end + self.config.min_gap
                        duration_diff = new_start - seg.start
                        new_duration = max(
                            0.1,  # Ensure at least some duration
                            new_seg.duration - duration_diff,
                        )
                        new_seg = self._copy_segment(
                            new_seg, start=new_start, duration=new_duration
                        )

            # B. Min duration check
            if new_seg.duration < self.config.min_duration:
                # Check if extending would overlap with next subtitle
                next_start = (
                    segments[i + 1].start if i + 1 < len(segments) else float("inf")
                )
                max_extend = next_start - new_seg.start - self.config.min_gap
                new_duration = min(
                    self.config.min_duration, max(max_extend, new_seg.duration)
                )
                new_seg = self._copy_segment(new_seg, duration=new_duration)

            # C. Max duration check — NO hard truncation here.
            # A cue that exceeds max_duration must be split into sub-segments
            # by _split_long_segments (by word timestamps if alignment exists,
            # otherwise by proportional char ratio). Truncating would silently
            # drop captions while the dubbed/original audio is still playing.

            result.append(new_seg)

        return result

    def _split_long_segments(self, segments: List[Supervision]) -> List[Supervision]:
        """Split segments that exceed either the text OR the duration budget.

        Text budget: ``max_lines × max_chars_per_line`` characters.
        Duration budget: ``max_duration`` seconds.

        Uses word alignment data when available for:
        - Precise timing from word timestamps (not character-ratio guessing)
        - Preferring split points at larger inter-word gaps (natural pauses)
        - Proper margin handling (start_margin / end_margin per sub-segment)

        Falls back to proportional (char-ratio) timing when no alignment is
        present. A cue is NEVER hard-truncated — if the duration budget can't
        be met by the split itself (e.g., extremely long text with no word
        alignment), the sub-segments still cover the original time range.
        """
        max_text_len = self.config.max_lines * self.config.max_chars_per_line
        max_dur = self.config.max_duration
        result: List[Supervision] = []

        for seg in segments:
            text = self._normalize_text(seg.text or "")
            over_text = len(text) > max_text_len
            over_dur = seg.duration > max_dur
            if not (over_text or over_dur):
                result.append(seg)
                continue

            words = self._get_word_alignment(seg)
            if words and len(words) >= 2:
                sub_segs = self._split_with_alignment(seg, words, max_text_len, max_dur)
            else:
                sub_segs = self._split_without_alignment(
                    seg, text, max_text_len, max_dur
                )

            result.extend(sub_segs)

        return result

    def _split_with_alignment(
        self, seg: Supervision, words: List, max_text_len: int, max_duration: float
    ) -> List[Supervision]:
        """Split a segment using word alignment data for precise timing.

        Strategy:
        1. Accumulate words until EITHER the character budget OR the duration
           budget (``max_duration``) is reached.
        2. At a budget boundary, look back for a natural pause to split at —
           but ONLY if current chars >= 75% of the char budget (avoid
           under-filled segments).
        3. Each sub-segment gets timing from its first/last word + margins.
        4. Prevent orphans: if last group < 25% of budget, merge into previous.
        """
        sm = self.config.start_margin or 0.0
        em = self.config.end_margin or 0.0

        def group_span(group: List) -> float:
            if not group:
                return 0.0
            return (group[-1].start + group[-1].duration) - group[0].start

        # Build word groups that fit within max_text_len AND max_duration
        groups: List[List] = []
        current_group: List = []
        current_chars = 0

        for word in words:
            word_len = len(word.symbol)
            separator = 1 if current_chars > 0 else 0
            new_len = current_chars + separator + word_len

            # Prospective group span if we appended this word
            if current_group:
                prospective_span = (word.start + word.duration) - current_group[0].start
            else:
                prospective_span = word.duration
            over_text = new_len > max_text_len
            over_dur = prospective_span > max_duration

            if (over_text or over_dur) and current_group:
                # Budget exceeded — find best split point
                best_split = None
                if over_text:
                    best_split = self._find_best_gap_split(
                        current_group, current_chars, max_text_len
                    )
                if best_split is not None:
                    overflow = current_group[best_split:]
                    current_group = current_group[:best_split]
                    groups.append(current_group)
                    current_group = overflow + [word]
                else:
                    groups.append(current_group)
                    current_group = [word]
                current_chars = sum(len(w.symbol) for w in current_group) + max(
                    0, len(current_group) - 1
                )
            else:
                current_group.append(word)
                current_chars = new_len

        if current_group:
            groups.append(current_group)

        # Prevent orphans: merge last group into previous if too short — but
        # only when the merged group still fits the duration budget (otherwise
        # we'd re-create the over-long cue we just worked to split).
        min_orphan_len = int(max_text_len * 0.25)
        if len(groups) >= 2:
            last_chars = sum(len(w.symbol) for w in groups[-1]) + max(
                0, len(groups[-1]) - 1
            )
            merged_span = (
                groups[-1][-1].start + groups[-1][-1].duration - groups[-2][0].start
            )
            if last_chars < min_orphan_len and merged_span <= max_duration:
                groups[-2].extend(groups[-1])
                groups.pop()

        # Slice the ORIGINAL text at the char positions of each group's first
        # word — never reconstruct by joining w.symbol values. See CLAUDE.md
        # "Multilingual Text Convention": joining symbols inserts ASCII spaces
        # between every CJK character and destroys user-authored whitespace
        # and punctuation in mixed-language text.
        non_empty_groups = [g for g in groups if g]
        group_texts = self._slice_text_by_word_groups(seg.text or "", non_empty_groups)

        trans_slices = self._split_translation_proportionally(
            getattr(seg, "translation", None), group_texts
        )

        # Create sub-segments from word groups
        result: List[Supervision] = []
        slice_iter = iter(trans_slices)
        text_iter = iter(group_texts)
        for group in groups:
            if not group:
                continue

            text = next(text_iter)
            first_start = group[0].start
            last_end = group[-1].start + group[-1].duration

            seg_start = max(0, first_start - sm)
            seg_end = last_end + em

            if result:
                prev_end = result[-1].start + result[-1].duration
                if seg_start < prev_end + self.config.min_gap:
                    seg_start = prev_end + self.config.min_gap

            # min_duration is a soft floor — it must NOT push end past
            # the parent supervision's boundary or into the next cue.
            # Without this clamp, very short trailing groups (e.g. 0.46 s
            # "Um") get inflated to min_duration (0.6-0.8 s by default),
            # the new end overruns the source cue's end_time, and at
            # \an5/\an2 alignment libass stacks the inflated cue on top
            # of the next speaker's first sub-cue. Clamp first to the
            # parent supervision's end, then to the original word group
            # span — whichever room exists.
            seg_floor = max(self.config.min_duration, seg_end - seg_start)
            parent_end = seg.start + seg.duration
            available = max(0.0, parent_end - seg_start)
            duration = min(seg_floor, available) if available > 0 else seg_floor

            result.append(
                self._copy_segment(
                    seg,
                    text=text,
                    start=seg_start,
                    duration=duration,
                    alignment={"word": list(group)},
                    translation=next(slice_iter, None),
                )
            )

        return result if result else [seg]

    @staticmethod
    def _slice_text_by_word_groups(text: str, groups: List[List]) -> List[str]:
        """Slice the original ``text`` into per-group substrings, using the
        char position of each group's first word as the cut boundary.

        This preserves CJK spacing (no ASCII space between Chinese chars),
        Latin word spacing (``"Terry Tao"`` stays together), and any
        user-authored whitespace/punctuation between words.

        Walks ``text`` and ``groups`` in order; each word's symbol is located
        via ``text.find(symbol, cursor)``. If a word's symbol can't be found
        (text/alignment mismatch), falls back to the last known cursor.
        Runs in O(len(text) + total_word_chars) since each ``find`` advances
        the cursor monotonically.
        """
        if not groups:
            return []
        if not text:
            return [""] * len(groups)

        # For each group, record the char position where its first word
        # appears in ``text``.
        group_starts: List[int] = []
        cursor = 0
        for group in groups:
            if not group:
                group_starts.append(cursor)
                continue
            first_sym = group[0].symbol
            if not first_sym:
                group_starts.append(cursor)
                continue
            found = text.find(first_sym, cursor)
            if found < 0:
                # Fallback: alignment symbol missing from text (normalization
                # drift, token rewriting, etc.). Use the running cursor so
                # we never emit garbage slices.
                group_starts.append(cursor)
            else:
                group_starts.append(found)
                cursor = found + len(first_sym)
            # Advance cursor through the rest of the group's words so the
            # next group's first-word search starts after this group.
            for w in group[1:]:
                if not w.symbol:
                    continue
                nxt = text.find(w.symbol, cursor)
                if nxt >= 0:
                    cursor = nxt + len(w.symbol)

        # Each group's slice runs from its start to the next group's start
        # (last group runs to end-of-text). Inter-word whitespace lives on
        # the LEFT side of the boundary (i.e. trailing on group i), which
        # keeps concatenation lossless: "".join(slices) == text. No .strip()
        # here — a trailing invisible space is harmless when rendered, but
        # stripping it silently drops user-authored characters.
        group_ends = group_starts[1:] + [len(text)]
        slices: List[str] = []
        for start, end in zip(group_starts, group_ends):
            slices.append(text[start:end])
        return slices

    def _find_best_gap_split(
        self, current_group: List, current_chars: int, max_text_len: int
    ) -> Optional[int]:
        """Find the best split point based on inter-word gaps.

        Only considers gap-based early splitting when the group is already
        >= 75% of the budget. This prevents under-filled segments.

        Within the valid range, prefers the largest inter-word gap (>300ms)
        as a natural speech pause point.

        Returns:
            Index within current_group to split at, or None to split at the end.
        """
        if len(current_group) < 3:
            return None

        # Find the 75% budget threshold in word count
        min_chars_for_gap = int(max_text_len * 0.75)
        min_word_idx = None
        running_chars = 0
        for idx, w in enumerate(current_group):
            running_chars += len(w.symbol) + (1 if idx > 0 else 0)
            if running_chars >= min_chars_for_gap and min_word_idx is None:
                min_word_idx = idx + 1  # Split AFTER this word

        if min_word_idx is None or min_word_idx >= len(current_group):
            return None  # Group too short for gap-based splitting

        # Search from 75% threshold to end for the largest gap
        best_idx = None
        best_gap = -1.0

        for i in range(max(1, min_word_idx), len(current_group)):
            prev_end = current_group[i - 1].start + current_group[i - 1].duration
            gap = current_group[i].start - prev_end
            if gap > best_gap:
                best_gap = gap
                best_idx = i

        # Only use gap-based split if gap is a clear speech pause (>300ms)
        if best_gap > 0.3 and best_idx is not None:
            return best_idx

        return None

    def _split_without_alignment(
        self, seg: Supervision, text: str, max_text_len: int, max_duration: float
    ) -> List[Supervision]:
        """Split a segment without alignment data (char-ratio fallback).

        Honors both the text budget (``max_text_len``) and the duration budget
        (``max_duration``). Text is split into ``N = max(ceil(len/max_text_len),
        ceil(duration/max_duration))`` chunks; duration is then distributed by
        character ratio so the sub-segments together cover *exactly* the
        original time range — never more (would push into the next sup) and
        never less (would silently drop captions under dubbed audio).
        """
        n_by_text = (
            max(1, math.ceil(len(text) / max_text_len)) if max_text_len > 0 else 1
        )
        n_by_dur = (
            max(1, math.ceil(seg.duration / max_duration)) if max_duration > 0 else 1
        )
        n_chunks = max(n_by_text, n_by_dur)

        if n_chunks <= 1:
            return [seg]

        # Pick a chunk-length target that yields at least n_chunks pieces.
        # We cap at max_text_len so a duration-driven split still honors the
        # text budget.
        target_len = max(1, min(max_text_len, math.ceil(len(text) / n_chunks)))
        chunks = self._split_text_into_chunks(text, target_len)

        # If the chunker produced fewer pieces than we need for the duration
        # budget (e.g., very short text with no word-break opportunities),
        # fall back to hard character slices so we can still hit n_chunks.
        if len(chunks) < n_chunks and len(text) >= n_chunks:
            step = max(1, math.ceil(len(text) / n_chunks))
            chunks = [text[i : i + step] for i in range(0, len(text), step)]

        if len(chunks) <= 1:
            return [seg]

        total_chars = sum(len(c) for c in chunks)
        if total_chars <= 0:
            return [seg]

        trans_slices = self._split_translation_proportionally(
            getattr(seg, "translation", None), chunks
        )

        # Distribute seg.duration proportionally. Reserve ``min_gap`` between
        # each pair of sub-segments so the timeline still has a readable
        # visual break, and stay within the ORIGINAL time range so we never
        # push into the next sup or silently drop captions under dubbed audio.
        n = len(chunks)
        gap = self.config.min_gap or 0.0
        total_gap = max(0.0, gap * (n - 1))
        available = max(0.0, seg.duration - total_gap)

        result: List[Supervision] = []
        seg_end = round(seg.start + seg.duration, 4)
        current_start = round(seg.start, 4)
        for i, (chunk, trans_slice) in enumerate(zip(chunks, trans_slices)):
            if i == n - 1:
                # Last chunk absorbs any rounding error so the final end
                # lands exactly on the original seg end.
                chunk_duration = max(0.0, round(seg_end - current_start, 4))
            else:
                ratio = len(chunk) / total_chars
                chunk_duration = round(available * ratio, 4)

            result.append(
                self._copy_segment(
                    seg,
                    text=chunk,
                    start=current_start,
                    duration=chunk_duration,
                    translation=trans_slice,
                )
            )
            current_start = round(current_start + chunk_duration + gap, 4)

        return result

    def _split_text_into_chunks(self, text: str, max_chunk_len: int) -> List[str]:
        """Split text into chunks at word boundaries.

        Each chunk fits within max_chunk_len characters (with ~10% tolerance
        to avoid mid-word splits).
        """
        text = self._normalize_text(text)
        if len(text) <= max_chunk_len:
            return [text]

        chunks: List[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= max_chunk_len:
                chunks.append(remaining.strip())
                break

            split_pos = self._find_split_point(remaining, max_chunk_len)
            chunks.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        return [c for c in chunks if c]

    def _format_texts(self, segments: List[Supervision]) -> List[Supervision]:
        """Apply text formatting to all subtitles."""
        return [
            self._copy_segment(seg, text=self._smart_split_text(seg.text or ""))
            for seg in segments
        ]

    def _smart_split_text(self, text: str) -> str:
        """
        Smart text line breaking.

        Priority:
        1. CJK punctuation (，。！？ etc.)
        2. English punctuation (,.!? etc.)
        3. Whitespace
        4. Hard truncation

        Args:
            text: Original text

        Returns:
            Text with line breaks
        """
        # Fast-path: text already fits and has no embedded newlines / double
        # spaces. Return it verbatim so we don't accidentally strip boundary
        # whitespace that ``_split_with_alignment`` deliberately preserved
        # between adjacent sub-segments (needed to keep "Terry Tao 的" from
        # collapsing to "Terry Tao的" in mixed-language captions).
        if (
            len(text) <= self.config.max_chars_per_line
            and "\n" not in text
            and "  " not in text
        ):
            return text

        # Clean text
        text = self._normalize_text(text)

        # Check if line break is needed
        if len(text) <= self.config.max_chars_per_line:
            return text

        lines: List[str] = []
        remaining = text

        for _ in range(self.config.max_lines):
            if len(remaining) <= self.config.max_chars_per_line:
                lines.append(remaining)
                remaining = ""
                break

            # Find best split point
            split_pos = self._find_split_point(
                remaining, self.config.max_chars_per_line
            )

            lines.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        # If remaining text exists and max lines reached, append to last line
        if remaining and lines:
            # Choose to append (may exceed char limit) rather than truncate
            lines[-1] = lines[-1] + " " + remaining if lines[-1] else remaining

        return "\n".join(lines)

    def _find_split_point(self, text: str, max_len: int) -> int:
        """
        Find best split point at a word boundary.

        Only splits at positions that are word boundaries:
        - After whitespace (split between words)
        - After punctuation followed by whitespace (end of clause/sentence)

        Never splits mid-word (e.g., "they're" stays intact).

        Strategy: Search near max_len (40%-110% range), prefer:
        1. After CJK punctuation + whitespace (sentence boundary)
        2. After English punctuation + whitespace (clause boundary)
        3. At whitespace (word boundary)

        Args:
            text: Text to split
            max_len: Maximum length

        Returns:
            Split position index
        """
        search_start = int(max_len * 0.4)
        search_end = min(len(text), int(max_len * 1.1))

        best_pos = max_len
        best_priority = 999  # Lower is better

        for i in range(min(search_end, len(text)) - 1, search_start - 1, -1):
            char = text[i]

            if char.isspace():
                # Whitespace: always a valid word boundary
                priority = 3
                # Check if preceded by punctuation (upgrade priority)
                if i > 0:
                    prev_priority = self._get_split_priority(text[i - 1])
                    if prev_priority < 3:
                        priority = prev_priority
            elif i + 1 < len(text) and text[i + 1].isspace():
                # Punctuation followed by whitespace: valid word boundary
                priority = self._get_split_priority(char)
                if priority >= 999:
                    continue  # Not actual punctuation
            else:
                # Mid-word character: never split here
                continue

            if priority < best_priority:
                best_priority = priority
                best_pos = i + 1

                if priority == 1:
                    break

        return best_pos

    def _get_split_priority(self, char: str) -> int:
        """
        Get character split priority.

        Returns:
            1 = CJK punctuation (highest priority)
            2 = English punctuation
            3 = Whitespace
            999 = Other characters (not suitable for splitting)
        """
        if re.match(self.CJK_PUNCTUATION, char):
            return 1
        elif re.match(self.EN_PUNCTUATION, char):
            return 2
        elif char.isspace():
            return 3
        return 999

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text.

        - Remove excess whitespace
        - Remove existing newlines (will be reformatted)
        - Unify spaces
        """
        # Remove existing newlines
        text = text.replace("\n", " ")
        # Merge excess whitespace
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def _copy_segment(
        self,
        seg: Supervision,
        **overrides,
    ) -> Supervision:
        """
        Create a copy of Supervision preserving ALL dataclass fields by default.

        Uses fastcopy so bilingual fields (translation, target_lang), scoring,
        and any future dataclass fields are carried through automatically —
        callers only need to override what they want to change.

        Args:
            seg: Original segment
            **overrides: Fields to override

        Returns:
            New Supervision instance
        """
        return fastcopy(seg, **overrides)

    @staticmethod
    def _is_combining(ch: str) -> bool:
        """True for Unicode combining marks (category Mn / Mc / Me).

        These characters decorate the preceding base character (e.g. the NFD
        form of ``é`` is ``e`` + U+0301). Splitting between a base and its
        combining marks yields orphaned diacritics and mojibake.
        """
        return unicodedata.category(ch) in ("Mn", "Mc", "Me")

    @staticmethod
    def _advance_to_grapheme_boundary(s: str, idx: int) -> int:
        """Move ``idx`` forward until it sits on a safe grapheme-cluster boundary.

        Handles the three edge cases that real bilingual text throws at a
        naive code-point splitter:

        - Combining marks (Mn/Mc/Me): advance past them so the base + marks
          stay together.
        - Zero-width joiner: if the char at ``idx-1`` or ``idx`` is ZWJ, keep
          advancing until the glued emoji sequence ends.
        - Regional-indicator pairs (🇨🇳 etc.): never split a two-RI pair.

        Falls back to ``len(s)`` if the cluster reaches the end.
        """
        n = len(s)
        if idx <= 0 or idx >= n:
            return max(0, min(idx, n))

        # Skip forward across combining marks that belong to the previous base.
        while idx < n and CaptionStandardizer._is_combining(s[idx]):
            idx += 1

        # If we're sitting inside a ZWJ-joined emoji sequence, skip to the end
        # of the cluster. ZWJ chains can span multiple code points including
        # emoji modifiers and variation selectors.
        while idx < n and (s[idx - 1] == _ZWJ or s[idx] == _ZWJ):
            idx += 1
            while idx < n and CaptionStandardizer._is_combining(s[idx]):
                idx += 1

        # Regional-indicator flags are always pairs — if we landed between the
        # two halves, advance one more.
        if 0 < idx < n:
            prev_ri = 0x1F1E6 <= ord(s[idx - 1]) <= 0x1F1FF
            curr_ri = 0x1F1E6 <= ord(s[idx]) <= 0x1F1FF
            if prev_ri and curr_ri:
                idx += 1

        return min(idx, n)

    # Word-boundary safety helpers for translation splitting.
    # Latin word chars (letters, digits, apostrophe) form indivisible runs:
    # cutting between two such chars splits a real word (``permissions`` →
    # ``perm|issions``) or a contraction (``don't`` → ``don|'t``) or a
    # number+unit (``100ms`` → ``100|ms``). These must be forbidden.
    _CJK_LO = "一"
    _CJK_HI = "鿿"

    @staticmethod
    def _is_latin_word_char(ch: str) -> bool:
        if not ch:
            return False
        return ch.isascii() and (ch.isalpha() or ch.isdigit() or ch == "'")

    @staticmethod
    def _is_cjk_char(ch: str) -> bool:
        if not ch:
            return False
        return CaptionStandardizer._CJK_LO <= ch <= CaptionStandardizer._CJK_HI

    @staticmethod
    def _boundary_score(s: str, i: int) -> int:
        """Score a candidate split position ``i`` (Python slice index).

        ``s[:i]`` ends with ``s[i-1]``, ``s[i:]`` starts with ``s[i]``.
        Higher = better. Score 0 means forbidden (would split a Latin word).

        Ordering:
            4 — right after sentence-ending punctuation (. ! ? 。！？)
            3 — right after clause punctuation (, ; : 、，；：—–)
            2 — at whitespace (either side)
            1 — at a CJK boundary (either neighbor is CJK)
            0 — forbidden (Latin-word internal)

        Note: callers must not pass ``i = 0`` or ``i = len(s)``; those are
        not splits but edges. The scanner in :py:meth:`_find_safe_split`
        already filters them out, so they cannot accidentally short-circuit
        a real boundary in the middle of the string.
        """
        n = len(s)
        # Defensive: edges should never be evaluated, but if they are,
        # treat them as the lowest acceptable score so they cannot beat
        # any real punctuation/whitespace boundary inside the string.
        if i <= 0 or i >= n:
            return 1
        prev, nxt = s[i - 1], s[i]
        if CaptionStandardizer._is_latin_word_char(
            prev
        ) and CaptionStandardizer._is_latin_word_char(nxt):
            return 0
        if prev in ".!?。！？":
            return 4
        if prev in ",;:、，；：—–":
            return 3
        if prev.isspace() or nxt.isspace():
            return 2
        if CaptionStandardizer._is_cjk_char(prev) or CaptionStandardizer._is_cjk_char(
            nxt
        ):
            return 1
        # Mixed punctuation / other non-word chars — accept at low priority.
        return 1

    @staticmethod
    def _find_safe_split(s: str, target: int, lo: int) -> int:
        """Find a split position in ``[lo, len(s)]`` closest to ``target``
        that does NOT cut inside a Latin word.

        Strategy: symmetric outward scan from ``target``. Track the best
        candidate by ``_boundary_score``; expand the radius until either a
        top-tier boundary (sentence punct) is hit or both ends are exhausted.

        If the whole searchable range is Latin-word-internal (e.g.
        ``"aaaaaaaa"`` with no break opportunity), returns ``len(s)`` so the
        caller leaves the entire remaining translation on the current chunk
        rather than emitting a forbidden cut.
        """
        n = len(s)
        if target >= n:
            return n
        if target <= lo:
            target = lo

        best_pos: Optional[int] = None
        best_score = -1
        radius = 0
        # Inclusive bounds for scanning. ``lo`` is the minimum valid cut
        # (cursor + 1) — anything below would emit an empty slice.
        max_left_reach = target - lo
        max_right_reach = n - target

        while True:
            candidates = (
                (target,) if radius == 0 else (target - radius, target + radius)
            )
            for pos in candidates:
                # ``lo <= pos <= n-1`` — exclude ``pos == n`` (end of string)
                # so it never wins over a real boundary in the middle. The
                # "give up and keep whole" fallback is handled explicitly
                # below when no acceptable score was found.
                if pos < lo or pos >= n:
                    continue
                score = CaptionStandardizer._boundary_score(s, pos)
                if score > best_score:
                    best_pos, best_score = pos, score
                    if score >= 4:
                        return best_pos
            # Exit when:
            # 1) Both sides exhausted (no further candidates possible).
            # 2) We already have a non-forbidden boundary and have scanned
            #    a reasonable window (>= 20 chars on each side).
            if radius >= max_left_reach and radius >= max_right_reach:
                # If even an unbounded scan found nothing acceptable
                # (score still 0), refuse to cut: return n so the whole
                # remaining translation stays on the current chunk.
                if best_score <= 0:
                    return n
                return best_pos if best_pos is not None else n
            if best_score >= 1 and radius >= 20:
                return best_pos  # type: ignore[return-value]
            radius += 1

    @staticmethod
    def _effective_token_count(s: str) -> int:
        """Count meaning-carrying tokens in a translation slice.

        Each CJK character counts as one token; each maximal run of Latin
        word chars (letters/digits/apostrophe) counts as one token.
        Whitespace and punctuation are skipped.

        Used by :py:meth:`_merge_short_tail_slices` to decide whether a
        slice is too small to stand on its own (e.g. ``"ok."`` and
        ``"了。"`` each carry only 1 token — visually they read as a
        dangling fragment).
        """
        if not s:
            return 0
        count = 0
        in_latin = False
        for ch in s:
            if CaptionStandardizer._is_cjk_char(ch):
                count += 1
                in_latin = False
            elif CaptionStandardizer._is_latin_word_char(ch):
                if not in_latin:
                    count += 1
                    in_latin = True
            else:
                in_latin = False
        return count

    @staticmethod
    def _merge_short_tail_slices(
        slices: List[Optional[str]], min_tokens: int = 2
    ) -> List[Optional[str]]:
        """Fold slices that carry fewer than ``min_tokens`` tokens into
        the most recent non-None slice on the left.

        Why: severe chunk-ratio imbalance (e.g. ``[long_zh, "了"]`` ⇒
        ratio ~9:1) often pushes the translation cut to land near the
        end, leaving a 1-token tail like ``"ok."`` or ``"了。"`` on its
        own. Such a tail reads as a dangling fragment when rendered —
        users complained explicitly that the dangling char + punct
        should have stayed with the previous chunk.

        Walks left-to-right and accumulates short slices into the
        previous slot, replacing the original position with ``None``.
        Cascading short slices collapse leftward, so three single-token
        slices end up merged into the first slot.
        """
        if len(slices) <= 1:
            return list(slices)
        result: List[Optional[str]] = list(slices)
        for i in range(1, len(result)):
            cur = result[i]
            if cur is None:
                continue
            if CaptionStandardizer._effective_token_count(cur) >= min_tokens:
                continue
            j = i - 1
            while j >= 0 and result[j] is None:
                j -= 1
            if j < 0:
                continue
            result[j] = (result[j] or "") + cur
            result[i] = None
        return result

    @staticmethod
    def _split_translation_proportionally(
        translation: Optional[str], text_chunks: List[str]
    ) -> List[Optional[str]]:
        """Distribute a translation string across N text chunks by char ratio.

        Preserves total character count so bilingual data is never silently
        dropped when the source text is split into sub-segments. Split points
        are snapped to:

        1. **Word-safe boundaries** — never cut between two Latin
           letters/digits/apostrophe (no ``permissions`` → ``perm|issions``).
           Punctuation is preferred over whitespace, whitespace over
           CJK-boundary, CJK-boundary over other.
        2. **Unicode grapheme clusters** — combining marks and emoji ZWJ
           sequences stay together.

        Degenerate cases:

        - ``translation`` is None/empty or there is only one chunk → the full
          value goes on the first slot and the rest are None.
        - Translation is meaningfully shorter than the number of chunks
          (``len(translation) < 2 * n``) → place the whole translation on the
          first chunk rather than dribbling out 1-char fragments.
        - The remaining translation has no safe boundary (e.g. one giant
          Latin token) → keep it whole on the current chunk and emit ``None``
          for the rest. Better a long single line than a broken word.

        Post-process: slices carrying fewer than 2 effective tokens (a
        single CJK char or a single Latin word, regardless of punctuation)
        are folded into the previous slice. Prevents dangling tails like
        ``"ok."`` or ``"了。"`` from landing on their own chunk under
        severe ratio imbalance — see :py:meth:`_merge_short_tail_slices`.
        """
        n = len(text_chunks)
        if n == 0:
            return []
        if not translation or n == 1:
            return [translation] + [None] * (n - 1)

        total = sum(len(c) for c in text_chunks)
        if total <= 0:
            return [translation] + [None] * (n - 1)

        trans_len = len(translation)

        # Extreme-ratio fallback: if every chunk would get < 2 chars on
        # average, proportional splitting just produces unreadable flecks of
        # translation. Keep the whole translation on the first chunk instead.
        if trans_len < 2 * n:
            return [translation] + [None] * (n - 1)

        out: List[Optional[str]] = []
        cursor = 0
        for i, chunk in enumerate(text_chunks):
            if i == n - 1:
                out.append(translation[cursor:] or None)
                continue
            ratio = len(chunk) / total
            target = cursor + max(1, int(round(trans_len * ratio)))
            target = min(target, trans_len)
            end = CaptionStandardizer._find_safe_split(
                translation, target, lo=cursor + 1
            )
            end = CaptionStandardizer._advance_to_grapheme_boundary(translation, end)
            slice_ = translation[cursor:end]
            out.append(slice_ or None)
            cursor = end
        return CaptionStandardizer._merge_short_tail_slices(out)

    def apply_margins(
        self,
        segments: List[Supervision],
        start_margin: Optional[float] = None,
        end_margin: Optional[float] = None,
    ) -> List[Supervision]:
        """
        Recalculate segment boundaries based on word-level alignment.

        Uses precise word-level timestamps from supervision.alignment['word']
        to recalculate segment start/end times.

        Args:
            segments: List of subtitles with alignment data
            start_margin: Start margin (overrides config default)
            end_margin: End margin (overrides config default)

        Returns:
            List of subtitles with new margins applied

        Note:
            - Segments without alignment data keep original timestamps
            - Automatically handles boundary collisions

        Example:
            >>> standardizer = CaptionStandardizer()
            >>> adjusted = standardizer.apply_margins(
            ...     supervisions, start_margin=0.05, end_margin=0.15
            ... )
        """
        if not segments:
            return []

        # Resolve margins: parameter > config > 0.0 (no adjustment)
        sm = (
            start_margin
            if start_margin is not None
            else (self.config.start_margin or 0.0)
        )
        em = end_margin if end_margin is not None else (self.config.end_margin or 0.0)

        # Sort by start time
        sorted_segs = sorted(segments, key=lambda s: s.start)
        result: List[Supervision] = []

        for seg in sorted_segs:
            # Get word alignment
            words = self._get_word_alignment(seg)

            if not words:
                # No alignment data, keep original
                result.append(self._copy_segment(seg))
                continue

            # Calculate precise boundaries
            first_word_start = words[0].start
            last_word_end = words[-1].start + words[-1].duration

            # Apply margin (0.0 means no adjustment, just use word boundaries)
            new_start = max(0, first_word_start - sm)
            new_end = last_word_end + em

            # Collision detection (with previous segment)
            if result:
                prev_end = result[-1].start + result[-1].duration
                if new_start < prev_end + self.config.min_gap:
                    new_start = self._resolve_collision(
                        prev_end, new_start, first_word_start, sm
                    )

            new_duration = new_end - new_start
            result.append(
                self._copy_segment(seg, start=new_start, duration=new_duration)
            )

        return result

    def _get_word_alignment(self, seg: Supervision) -> List:
        """
        Safely get word alignment data.

        Args:
            seg: Subtitle segment

        Returns:
            Word alignment list, or empty list if not present
        """
        alignment = getattr(seg, "alignment", None)
        if alignment and "word" in alignment:
            return alignment["word"]
        return []

    def _resolve_collision(
        self,
        prev_end: float,
        new_start: float,
        first_word_start: float,
        start_margin: float,
    ) -> float:
        """
        Resolve collision with previous segment.

        Args:
            prev_end: End time of previous segment
            new_start: Currently calculated start time
            first_word_start: Start time of first word in current segment
            start_margin: Requested start_margin

        Returns:
            Adjusted start time
        """
        if self.config.margin_collision_mode == "gap":
            # Force maintain min_gap
            return prev_end + self.config.min_gap
        else:
            # Trim mode: preserve margin as much as possible, but not beyond speech start
            available_margin = first_word_start - (prev_end + self.config.min_gap)
            actual_margin = max(0, min(start_margin, available_margin))
            return first_word_start - actual_margin


class CaptionValidator:
    """
    Caption quality validator.

    Validates subtitles against broadcast standards and generates quality metrics report.

    Example:
        >>> validator = CaptionValidator()
        >>> result = validator.validate(supervisions)
        >>> if not result.valid:
        ...     print(result.warnings)
    """

    def __init__(
        self,
        config: Optional[StandardizationConfig] = None,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        min_gap: float = 0.08,
        max_chars_per_line: int = 42,
    ):
        """
        Initialize validator.

        Args:
            config: Standardization config (if provided, ignores other params)
            min_duration: Minimum duration
            max_duration: Maximum duration
            min_gap: Minimum gap
            max_chars_per_line: Maximum characters per line
        """
        if config:
            self.config = config
        else:
            self.config = StandardizationConfig(
                min_duration=min_duration,
                max_duration=max_duration,
                min_gap=min_gap,
                max_chars_per_line=max_chars_per_line,
            )

    def validate(self, segments: List[Supervision]) -> ValidationResult:
        """
        Validate subtitles and return quality metrics.

        Args:
            segments: List of subtitle segments

        Returns:
            ValidationResult containing validation results and metrics
        """
        result = ValidationResult()

        if not segments:
            return result

        total_cps = 0.0
        prev_end = 0.0

        for i, seg in enumerate(segments):
            text = seg.text or ""
            duration = seg.duration

            # CPS calculation (excluding newlines)
            text_length = len(text.replace("\n", ""))
            cps = text_length / duration if duration > 0 else 0
            total_cps += cps

            # CPL calculation
            lines = text.split("\n")
            max_line_len = max((len(line) for line in lines), default=0)
            result.max_cpl = max(result.max_cpl, max_line_len)

            # Duration check
            if duration < self.config.min_duration:
                result.segments_too_short += 1
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): duration {duration:.2f}s < min {self.config.min_duration}s"
                )

            if duration > self.config.max_duration:
                result.segments_too_long += 1
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): duration {duration:.2f}s > max {self.config.max_duration}s"
                )

            # Gap check — float tolerance (1e-6) absorbs sub-ms drift from
            # caption-time arithmetic, which otherwise flags perfectly valid
            # 80ms gaps as 79.999ms.
            if i > 0:
                gap = seg.start - prev_end
                if gap >= 0 and gap < self.config.min_gap - 1e-6:
                    result.gaps_too_small += 1
                    result.warnings.append(
                        f"Segment {i} (id={seg.id}): gap {gap:.3f}s < min {self.config.min_gap}s"
                    )

            # CPL check
            if max_line_len > self.config.max_chars_per_line:
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): line length {max_line_len} > max {self.config.max_chars_per_line}"
                )

            # CPS check (reading speed too fast)
            if cps > self.config.optimal_cps * 1.5:  # Exceeds optimal by 50%
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): CPS {cps:.1f} exceeds recommended {self.config.optimal_cps}"
                )

            prev_end = seg.start + seg.duration

        # Calculate average CPS
        result.avg_cps = total_cps / len(segments)

        # Determine if validation passed
        result.valid = (
            result.segments_too_short == 0
            and result.segments_too_long == 0
            and result.gaps_too_small == 0
        )

        return result


def standardize_captions(
    segments: List[Supervision],
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    min_gap: float = 0.08,
    max_lines: int = 2,
    max_chars_per_line: int = 42,
) -> List[Supervision]:
    """
    Convenience function: Standardize caption list.

    Args:
        segments: List of original caption segments
        min_duration: Minimum duration (seconds)
        max_duration: Maximum duration (seconds)
        min_gap: Minimum gap (seconds)
        max_lines: Maximum number of lines
        max_chars_per_line: Maximum characters per line

    Returns:
        List of processed caption segments

    Example:
        >>> from lattifai.caption import standardize_captions
        >>> processed = standardize_captions(supervisions, max_chars_per_line=22)
    """
    standardizer = CaptionStandardizer(
        min_duration=min_duration,
        max_duration=max_duration,
        min_gap=min_gap,
        max_lines=max_lines,
        max_chars_per_line=max_chars_per_line,
    )
    return standardizer.process(segments)


def apply_margins_to_captions(
    segments: List[Supervision],
    start_margin: float = 0.10,
    end_margin: float = 0.10,
    min_gap: float = 0.08,
    collision_mode: str = "trim",
) -> List[Supervision]:
    """
    Convenience function: Recalculate caption boundaries based on word-level alignment.

    Uses precise word-level timestamps from supervision.alignment['word']
    to recalculate segment start/end times.

    Args:
        segments: List of caption segments with alignment data
        start_margin: Start margin (seconds) - extends before first word
        end_margin: End margin (seconds) - extends after last word
        min_gap: Minimum gap (seconds) - for collision handling
        collision_mode: Collision mode 'trim' or 'gap'

    Returns:
        List of caption segments with new margins applied

    Example:
        >>> from lattifai.caption import apply_margins_to_captions
        >>> adjusted = apply_margins_to_captions(
        ...     supervisions, start_margin=0.05, end_margin=0.15
        ... )
    """
    standardizer = CaptionStandardizer(min_gap=min_gap)
    standardizer.config.start_margin = start_margin
    standardizer.config.end_margin = end_margin
    standardizer.config.margin_collision_mode = collision_mode
    return standardizer.apply_margins(
        segments, start_margin=start_margin, end_margin=end_margin
    )
