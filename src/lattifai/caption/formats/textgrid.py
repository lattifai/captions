"""Praat TextGrid format handler.

TextGrid is Praat's native annotation format, commonly used in phonetics research.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..supervision import Pathlike, Supervision
from . import register_format
from .base import FormatHandler


def _is_event(sup: Supervision) -> bool:
    """Detect if a supervision is an event type.

    Event detection via:
    1. custom["segment_type"] == "event"
    2. Text format [xxx] (e.g., [Applause], [Music])
    """
    if sup.custom and sup.custom.get("segment_type") == "event":
        return True
    text = (sup.text or "").strip()
    return text.startswith("[") and text.endswith("]") and len(text) > 2


def _assign_event_tiers(events: List[Supervision]) -> Dict[str, List]:
    """Assign events to non-overlapping tiers using greedy algorithm.

    Returns dict mapping tier names to lists of (start, end, text) tuples.
    Tier names: "Event", "Event2", "Event3", ...
    """
    tiers: Dict[str, List] = {}

    for event in sorted(events, key=lambda x: x.start):
        assigned = False
        tier_num = 1

        while not assigned:
            tier_name = "Event" if tier_num == 1 else f"Event{tier_num}"

            if tier_name not in tiers:
                tiers[tier_name] = []

            # Check overlap with last interval in this tier
            if not tiers[tier_name] or tiers[tier_name][-1][1] <= event.start:
                tiers[tier_name].append((event.start, event.end, event.text or ""))
                assigned = True
            else:
                tier_num += 1

    return tiers


@register_format("textgrid")
class TextGridFormat(FormatHandler):
    """Praat TextGrid format for phonetic analysis."""

    extensions = [".textgrid"]
    description = "Praat TextGrid - phonetics research format"

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read TextGrid format using tgt library.

        Preserves tier information in Supervision.custom:
        - textgrid_tier: Original tier name
        - textgrid_tier_index: Original tier index (for ordering)
        """
        from tgt import read_textgrid

        if cls.is_content(source):
            # Write to temp file for tgt library
            with tempfile.NamedTemporaryFile(suffix=".textgrid", delete=False, mode="w") as f:
                f.write(source)
                temp_path = f.name
            try:
                tgt = read_textgrid(temp_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            tgt = read_textgrid(str(source))

        supervisions = []
        for tier_idx, tier in enumerate(tgt.tiers):
            for interval in tier.intervals:
                supervisions.append(
                    Supervision(
                        text=interval.text,
                        start=interval.start_time,
                        duration=interval.end_time - interval.start_time,
                        speaker=tier.name,
                        custom={
                            "textgrid_tier": tier.name,
                            "textgrid_tier_index": tier_idx,
                        },
                    )
                )

        return sorted(supervisions, key=lambda x: x.start)

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Path:
        """Write TextGrid format using tgt library.

        Args:
            supervisions: List of supervisions to write
            output_path: Output file path
            metadata: Optional metadata (for API consistency)

        Note:
            Events (text like [Applause] or segment_type="event") are placed
            in separate tiers (Event, Event2, ...) to handle overlaps.
        """
        from tgt import Interval, IntervalTier, TextGrid, write_to_file

        output_path = Path(output_path)
        tg = TextGrid()

        # Group utterances by speaker
        speaker_utterances: Dict[str, List] = {}
        events = []
        words = []
        scores = {"utterances": [], "words": []}

        for sup in sorted(supervisions, key=lambda x: x.start):
            # Separate events from utterances
            if _is_event(sup):
                events.append(sup)
                continue

            text = sup.text or ""
            # Respect original_speaker flag: if False, treat as no speaker
            use_speaker = sup.speaker or ""
            if use_speaker and hasattr(sup, "custom") and sup.custom and not sup.custom.get("original_speaker", True):
                use_speaker = ""
            speaker_key = use_speaker

            speaker_utterances.setdefault(speaker_key, []).append(Interval(sup.start, sup.end, text))

            # Extract word-level alignment if present
            alignment = getattr(sup, "alignment", None)
            if alignment and "word" in alignment:
                for item in alignment["word"]:
                    words.append(Interval(item.start, item.end, item.symbol))
                    if item.score is not None:
                        scores["words"].append(Interval(item.start, item.end, f"{item.score:.2f}"))

            if hasattr(sup, "custom") and sup.custom and "score" in sup.custom:
                scores["utterances"].append(Interval(sup.start, sup.end, f"{sup.custom['score']:.2f}"))

        # Add utterance tiers - split by speaker if speakers exist
        has_speakers = any(k for k in speaker_utterances if k)
        if has_speakers:
            for speaker_name in sorted(speaker_utterances):
                tier_name = speaker_name if speaker_name else "utterances"
                tg.add_tier(IntervalTier(name=tier_name, objects=speaker_utterances[speaker_name]))
        else:
            all_utterances = speaker_utterances.get("", [])
            tg.add_tier(IntervalTier(name="utterances", objects=all_utterances))

        # Add event tiers (Event, Event2, ...) for overlapping events
        if events:
            event_tiers = _assign_event_tiers(events)
            # Sort tier names: Event, Event2, Event3, ...
            for tier_name in sorted(event_tiers.keys(), key=lambda x: (len(x), x)):
                intervals = [Interval(s, e, t) for s, e, t in event_tiers[tier_name]]
                tg.add_tier(IntervalTier(name=tier_name, objects=intervals))

        if words:
            tg.add_tier(IntervalTier(name="words", objects=words))

        if scores["utterances"]:
            tg.add_tier(IntervalTier(name="utterance_scores", objects=scores["utterances"]))
        if scores["words"]:
            tg.add_tier(IntervalTier(name="word_scores", objects=scores["words"]))

        write_to_file(tg, str(output_path), format="long")
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bytes:
        """Convert to TextGrid format bytes.

        Args:
            supervisions: List of supervisions to convert
            metadata: Optional metadata (currently unused, for API consistency)
        """
        # TextGrid requires file I/O due to tgt library implementation
        with tempfile.NamedTemporaryFile(suffix=".textgrid", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cls.write(supervisions, tmp_path, metadata=metadata, **kwargs)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def _extract_textgrid_metadata(cls, content: str) -> Dict[str, Any]:
        """Extract TextGrid metadata from content string."""
        import re

        metadata: Dict[str, Any] = {}
        match = re.search(r"xmin\s*=\s*([\d.]+)", content)
        if match:
            metadata["textgrid_xmin"] = float(match.group(1))
        match = re.search(r"xmax\s*=\s*([\d.]+)", content)
        if match:
            metadata["textgrid_xmax"] = float(match.group(1))
        tier_names = re.findall(r'name\s*=\s*"([^"]+)"', content)
        if tier_names:
            metadata["textgrid_tiers"] = tier_names
        return metadata

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str], **kwargs) -> Dict[str, Any]:
        """Extract TextGrid metadata. Deprecated: use parse() instead."""
        if cls.is_content(source):
            return cls._extract_textgrid_metadata(source)
        try:
            with open(source, "r", encoding="utf-8") as f:
                return cls._extract_textgrid_metadata(f.read())
        except Exception:
            return {}

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> "ParseResult":
        """Parse TextGrid in a single pass."""
        from .base import ParseResult

        supervisions = cls.read(source, normalize_text=normalize_text, **kwargs)
        if cls.is_content(source):
            content = source
        else:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                return ParseResult(supervisions=supervisions)
        return ParseResult(
            supervisions=supervisions,
            format_metadata=cls._extract_textgrid_metadata(content),
        )
