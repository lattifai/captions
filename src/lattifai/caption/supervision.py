"""
Supervision data structures for caption processing.

Core data structures originally from lhotse (https://github.com/lhotse-speech/lhotse),
copied here to remove the dependency while maintaining API compatibility.

Original source:
- lhotse.utils: Pathlike, Seconds
- lhotse.supervision: AlignmentItem, SupervisionSegment

Lhotse is licensed under the Apache License 2.0.
Copyright (c) 2020-2024 The Lhotse Authors
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

# Type aliases from lhotse.utils
Pathlike = Union[Path, str]
Seconds = float


def _asdict_nonull(dclass) -> Dict[str, Any]:
    """
    Recursively convert a dataclass into a dict, removing all fields with None value.
    Copied from lhotse.utils.asdict_nonull.
    """

    def non_null_dict_factory(collection):
        d = dict(collection)
        remove_keys = [key for key, val in d.items() if val is None]
        for k in remove_keys:
            del d[k]
        return d

    return asdict(dclass, dict_factory=non_null_dict_factory)


def _fastcopy(dataclass_obj, **kwargs):
    """
    Returns a new object with the same member values.
    Selected members can be overwritten with kwargs.
    Copied from lhotse.utils.fastcopy.
    """
    return type(dataclass_obj)(**{**dataclass_obj.__dict__, **kwargs})


def _add_durations(*durs: Seconds, sampling_rate: int = 48000) -> Seconds:
    """
    Adds durations in a way that avoids floating point precision issues.
    Simplified from lhotse.utils.add_durations.
    """
    tot_num_samples = sum(round(d * sampling_rate) for d in durs)
    return tot_num_samples / sampling_rate


class AlignmentItem(NamedTuple):
    """
    This class contains an alignment item, for example a word, along with its
    start time (w.r.t. the start of recording) and duration. It can potentially
    be used to store other kinds of alignment items, such as subwords, pdfid's etc.

    Copied from lhotse.supervision.AlignmentItem.
    """

    symbol: str
    start: Seconds
    duration: Seconds

    # Score is an optional aligner-specific measure of confidence.
    # A simple measure can be an average probability of "symbol" across
    # frames covered by the AlignmentItem.
    score: Optional[float] = None

    @staticmethod
    def deserialize(data: Union[List, Dict]) -> "AlignmentItem":
        if isinstance(data, dict):
            # Support loading alignments stored in the format we had before Lhotse v1.8
            return AlignmentItem(*list(data.values()))
        return AlignmentItem(*data)

    def serialize(self) -> list:
        return list(self)

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_offset(self, offset: Seconds) -> "AlignmentItem":
        """Return an identical AlignmentItem, but with the offset added to the start field."""
        return AlignmentItem(
            start=_add_durations(self.start, offset),
            duration=self.duration,
            symbol=self.symbol,
            score=self.score,
        )

    def trim(self, end: Seconds, start: Seconds = 0) -> "AlignmentItem":
        """Trim the alignment item to fit within the given time range."""
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        return AlignmentItem(
            symbol=self.symbol,
            start=max(start, self.start),
            duration=_add_durations(self.duration, -end_exceeds_by, -start_exceeds_by),
        )

    def transform(self, transform_fn: Callable[[str], str]) -> "AlignmentItem":
        """Perform specified transformation on the alignment content."""
        return AlignmentItem(
            symbol=transform_fn(self.symbol),
            start=self.start,
            duration=self.duration,
            score=self.score,
        )


@dataclass
class SupervisionSegment:
    """
    SupervisionSegment represents a time interval (segment) annotated with some
    supervision labels and/or metadata, such as the transcription, the speaker identity,
    the language, etc.

    Copied from lhotse.supervision.SupervisionSegment with CustomFieldMixin inlined.
    """

    id: str
    recording_id: str
    start: Seconds
    duration: Seconds
    channel: Union[int, List[int]] = 0
    text: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    gender: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None
    alignment: Optional[Dict[str, List[AlignmentItem]]] = None

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_alignment(self, kind: str, alignment: List[AlignmentItem]) -> "SupervisionSegment":
        alis = self.alignment
        if alis is None:
            alis = {}
        alis[kind] = alignment
        return _fastcopy(self, alignment=alis)

    def with_offset(self, offset: Seconds) -> "SupervisionSegment":
        """Return an identical SupervisionSegment, but with the offset added to the start field."""
        return SupervisionSegment(
            id=self.id,
            recording_id=self.recording_id,
            start=round(self.start + offset, ndigits=8),
            duration=self.duration,
            channel=self.channel,
            text=self.text,
            language=self.language,
            speaker=self.speaker,
            gender=self.gender,
            custom=self.custom,
            alignment=self.alignment,
        )

    def trim(self, end: Seconds, start: Seconds = 0) -> "SupervisionSegment":
        """
        Return an identical SupervisionSegment, but ensure that self.start is not negative
        and self.end does not exceed the end parameter.
        """
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        return _fastcopy(
            self,
            start=max(start, self.start),
            duration=_add_durations(self.duration, -end_exceeds_by, -start_exceeds_by),
            alignment=(
                {type_: [item.trim(end=end, start=start) for item in ali] for type_, ali in self.alignment.items()}
                if self.alignment
                else None
            ),
        )

    def transform_text(self, transform_fn: Callable[[str], str]) -> "SupervisionSegment":
        """Return a copy of the current segment with transformed text field."""
        if self.text is None:
            return self
        return _fastcopy(self, text=transform_fn(self.text))

    def transform_alignment(
        self, transform_fn: Callable[[str], str], type: Optional[str] = "word"
    ) -> "SupervisionSegment":
        """Return a copy of the current segment with transformed alignment field."""
        if self.alignment is None:
            return self
        return _fastcopy(
            self,
            alignment={
                ali_type: [item.transform(transform_fn=transform_fn) if ali_type == type else item for item in ali]
                for ali_type, ali in self.alignment.items()
            },
        )

    def to_dict(self) -> dict:
        if self.alignment is None:
            return _asdict_nonull(self)
        else:
            alis = {kind: [item.serialize() for item in ali] for kind, ali in self.alignment.items()}
            data = _asdict_nonull(_fastcopy(self, alignment=None))
            data["alignment"] = alis
            return data

    @staticmethod
    def from_dict(data: dict) -> "SupervisionSegment":
        if "alignment" in data:
            data["alignment"] = {k: [AlignmentItem.deserialize(x) for x in v] for k, v in data["alignment"].items()}
        return SupervisionSegment(**data)


@dataclass
class Supervision(SupervisionSegment):
    """
    Extended SupervisionSegment with simplified initialization.

    Note: The `alignment` field is inherited from SupervisionSegment:
        alignment: Optional[Dict[str, List[AlignmentItem]]] = None

    Structure of alignment when return_details=True:
        {
            'word': [
                AlignmentItem(symbol='hello', start=0.0, duration=0.5, score=0.95),
                AlignmentItem(symbol='world', start=0.6, duration=0.4, score=0.92),
                ...
            ]
        }
    """

    text: Optional[str] = None
    speaker: Optional[str] = None
    id: str = ""
    recording_id: str = ""
    start: Seconds = 0.0
    duration: Seconds = 0.0


__all__ = ["Pathlike", "Seconds", "AlignmentItem", "SupervisionSegment", "Supervision"]
