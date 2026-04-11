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

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import StandardizationConfig
from .supervision import Supervision, fastcopy

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
    CJK_PUNCTUATION = r"[，。、？！：；·…—～""（）【】〔〕〖〗《》〈〉「」『』〘〙〚〛]"

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
                    target_prev_duration = new_seg.start - self.config.min_gap - prev_seg.start

                    if target_prev_duration >= self.config.min_duration:
                        # Safe to shorten previous subtitle (still meets min duration)
                        result[-1] = self._copy_segment(prev_seg, duration=target_prev_duration)
                    else:
                        # Shortening previous would go below min duration, delay current start
                        new_start = prev_end + self.config.min_gap
                        duration_diff = new_start - seg.start
                        new_duration = max(
                            0.1,  # Ensure at least some duration
                            new_seg.duration - duration_diff,
                        )
                        new_seg = self._copy_segment(new_seg, start=new_start, duration=new_duration)

            # B. Min duration check
            if new_seg.duration < self.config.min_duration:
                # Check if extending would overlap with next subtitle
                next_start = segments[i + 1].start if i + 1 < len(segments) else float("inf")
                max_extend = next_start - new_seg.start - self.config.min_gap
                new_duration = min(self.config.min_duration, max(max_extend, new_seg.duration))
                new_seg = self._copy_segment(new_seg, duration=new_duration)

            # C. Max duration check
            if new_seg.duration > self.config.max_duration:
                new_seg = self._copy_segment(new_seg, duration=self.config.max_duration)

            result.append(new_seg)

        return result

    def _split_long_segments(self, segments: List[Supervision]) -> List[Supervision]:
        """Split segments whose text exceeds max_lines × max_chars_per_line.

        Uses word alignment data when available for:
        - Precise timing from word timestamps (not character-ratio guessing)
        - Preferring split points at larger inter-word gaps (natural pauses)
        - Proper margin handling (start_margin / end_margin per sub-segment)

        Falls back to proportional timing when no alignment data is present.
        """
        max_text_len = self.config.max_lines * self.config.max_chars_per_line
        result: List[Supervision] = []

        for seg in segments:
            text = self._normalize_text(seg.text or "")
            if len(text) <= max_text_len:
                result.append(seg)
                continue

            words = self._get_word_alignment(seg)
            if words and len(words) >= 2:
                sub_segs = self._split_with_alignment(seg, words, max_text_len)
            else:
                sub_segs = self._split_without_alignment(seg, text, max_text_len)

            result.extend(sub_segs)

        return result

    def _split_with_alignment(
        self, seg: Supervision, words: List, max_text_len: int
    ) -> List[Supervision]:
        """Split a segment using word alignment data for precise timing.

        Strategy:
        1. Accumulate words until character budget is reached
        2. At budget boundary, look back for a natural gap to split at —
           but ONLY if current chars >= 75% of budget (avoid under-filled segments)
        3. Each sub-segment gets timing from its first/last word + margins
        4. Prevent orphans: if last group < 25% of budget, merge into previous
        """
        sm = self.config.start_margin or 0.0
        em = self.config.end_margin or 0.0

        # Build word groups that fit within max_text_len
        groups: List[List] = []
        current_group: List = []
        current_chars = 0

        for i, word in enumerate(words):
            word_len = len(word.symbol)
            separator = 1 if current_chars > 0 else 0
            new_len = current_chars + separator + word_len

            if new_len > max_text_len and current_group:
                # Budget exceeded — find best split point
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
                current_chars = sum(len(w.symbol) for w in current_group) + max(0, len(current_group) - 1)
            else:
                current_group.append(word)
                current_chars = new_len

        if current_group:
            groups.append(current_group)

        # Prevent orphans: merge last group into previous if too short
        min_orphan_len = int(max_text_len * 0.25)
        if len(groups) >= 2:
            last_chars = sum(len(w.symbol) for w in groups[-1]) + max(0, len(groups[-1]) - 1)
            if last_chars < min_orphan_len:
                groups[-2].extend(groups[-1])
                groups.pop()

        # Pre-compute per-group text so we can split bilingual translation
        # proportionally across sub-segments.
        group_texts = [" ".join(w.symbol for w in g) for g in groups if g]
        trans_slices = self._split_translation_proportionally(
            getattr(seg, "translation", None), group_texts
        )

        # Create sub-segments from word groups
        result: List[Supervision] = []
        slice_iter = iter(trans_slices)
        for group in groups:
            if not group:
                continue

            text = " ".join(w.symbol for w in group)
            first_start = group[0].start
            last_end = group[-1].start + group[-1].duration

            seg_start = max(0, first_start - sm)
            seg_end = last_end + em

            if result:
                prev_end = result[-1].start + result[-1].duration
                if seg_start < prev_end + self.config.min_gap:
                    seg_start = prev_end + self.config.min_gap

            duration = max(self.config.min_duration, seg_end - seg_start)

            result.append(self._copy_segment(
                seg,
                text=text,
                start=seg_start,
                duration=duration,
                alignment={"word": list(group)},
                translation=next(slice_iter, None),
            ))

        return result if result else [seg]

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
        self, seg: Supervision, text: str, max_text_len: int
    ) -> List[Supervision]:
        """Split a segment without alignment data (proportional timing fallback)."""
        chunks = self._split_text_into_chunks(text, max_text_len)
        if len(chunks) <= 1:
            return [seg]

        total_chars = sum(len(c) for c in chunks)
        current_start = seg.start
        result: List[Supervision] = []

        trans_slices = self._split_translation_proportionally(
            getattr(seg, "translation", None), chunks
        )

        for chunk, trans_slice in zip(chunks, trans_slices):
            ratio = len(chunk) / total_chars if total_chars > 0 else 1.0 / len(chunks)
            chunk_duration = max(self.config.min_duration, seg.duration * ratio)

            result.append(self._copy_segment(
                seg,
                text=chunk,
                start=current_start,
                duration=chunk_duration,
                translation=trans_slice,
            ))
            current_start += chunk_duration + self.config.min_gap

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
        return [self._copy_segment(seg, text=self._smart_split_text(seg.text or "")) for seg in segments]

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
            split_pos = self._find_split_point(remaining, self.config.max_chars_per_line)

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
    def _split_translation_proportionally(
        translation: Optional[str], text_chunks: List[str]
    ) -> List[Optional[str]]:
        """Distribute a translation string across N text chunks by char ratio.

        Preserves total character count so bilingual data is never silently
        dropped when the source text is split into sub-segments. The split is
        naive — no CJK word-boundary awareness — but guarantees integrity.

        Returns a list of ``len(text_chunks)`` entries; when ``translation`` is
        None/empty or there is only one chunk, the full value goes on the first
        slot and the rest are None.
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
        out: List[Optional[str]] = []
        cursor = 0
        for i, chunk in enumerate(text_chunks):
            if i == n - 1:
                out.append(translation[cursor:] or None)
            else:
                ratio = len(chunk) / total
                end = min(cursor + max(1, int(round(trans_len * ratio))), trans_len)
                slice_ = translation[cursor:end]
                out.append(slice_ or None)
                cursor = end
        return out

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
        sm = start_margin if start_margin is not None else (self.config.start_margin or 0.0)
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
                    new_start = self._resolve_collision(prev_end, new_start, first_word_start, sm)

            new_duration = new_end - new_start
            result.append(self._copy_segment(seg, start=new_start, duration=new_duration))

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

            # Gap check
            if i > 0:
                gap = seg.start - prev_end
                if gap < self.config.min_gap and gap >= 0:
                    result.gaps_too_small += 1
                    result.warnings.append(f"Segment {i} (id={seg.id}): gap {gap:.3f}s < min {self.config.min_gap}s")

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
        result.valid = result.segments_too_short == 0 and result.segments_too_long == 0 and result.gaps_too_small == 0

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
    return standardizer.apply_margins(segments, start_margin=start_margin, end_margin=end_margin)
