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


def fastcopy(dataclass_obj, **kwargs):
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


def _to_native(value: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if value is None:
        return None
    # Check for numpy types without importing numpy
    val_type = type(value).__name__
    if val_type in ("float32", "float64", "float16"):
        return float(value)
    if val_type in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8"):
        return int(value)
    if val_type == "bool_":
        return bool(value)
    if val_type == "ndarray":
        return value.tolist()
    return value


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
class Supervision:
    """
    Supervision represents a time interval (segment) annotated with some
    supervision labels and/or metadata, such as the transcription, the speaker identity,
    the language, etc.

    Based on lhotse.supervision.SupervisionSegment with CustomFieldMixin inlined.

    Supports both attribute-style and dict-style access to custom fields:
        # Attribute access (via custom dict)
        segment.my_field = value     # sets custom["my_field"]
        value = segment.my_field     # gets custom["my_field"]
        del segment.my_field         # deletes custom["my_field"]

        # Dict-style access (to dataclass fields AND custom fields)
        segment["text"] = "hello"    # sets dataclass field
        segment["my_field"] = value  # sets custom["my_field"]
        value = segment["my_field"]  # gets from dataclass field or custom

    Structure of alignment field:
        {
            'word': [
                AlignmentItem(symbol='hello', start=0.0, duration=0.5, score=0.95),
                AlignmentItem(symbol='world', start=0.6, duration=0.4, score=0.92),
                ...
            ]
        }
    """

    # Primary fields (with defaults for simplified initialization)
    text: Optional[str] = None
    start: Seconds = 0.0
    duration: Seconds = 0.0

    # Metadata fields
    id: str = ""
    recording_id: str = ""
    channel: Union[int, List[int]] = 0
    language: Optional[str] = None
    speaker: Optional[str] = None
    gender: Optional[str] = None

    # Translation fields
    translation: Optional[str] = None
    target_lang: Optional[str] = None

    # Extension fields
    custom: Optional[Dict[str, Any]] = None
    alignment: Optional[Dict[str, List[AlignmentItem]]] = None

    # =========================================================================
    # CustomFieldMixin-style attribute access (from lhotse.custom)
    # =========================================================================

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Store custom attributes in self.custom dict for serialization.
        Setting None removes the attribute from custom.
        Numpy types are auto-converted to Python native types.
        """
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            custom = self.custom if self.custom is not None else {}
            if value is None:
                custom.pop(key, None)
            else:
                custom[key] = _to_native(value)
            if custom:
                object.__setattr__(self, "custom", custom)

    def __getattr__(self, name: str) -> Any:
        """
        Access custom fields via attribute syntax.
        Raises AttributeError if not found.
        """
        custom = object.__getattribute__(self, "custom")
        if custom is not None and name in custom:
            return custom[name]
        raise AttributeError(f"No such attribute: {name}")

    def __delattr__(self, key: str) -> None:
        """Support del segment.custom_attr syntax."""
        if key in self.__dataclass_fields__:
            super().__delattr__(key)
            return
        if self.custom is None or key not in self.custom:
            raise AttributeError(f"No such member: '{key}'")
        del self.custom[key]

    # =========================================================================
    # Dict-style access methods
    # =========================================================================

    def __getitem__(self, key: str) -> Any:
        """Get field value by key (dataclass fields or custom)."""
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        if self.custom is not None and key in self.custom:
            return self.custom[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set field value by key (dataclass fields or custom).
        Numpy types are auto-converted to Python native types.
        """
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            if self.custom is None:
                object.__setattr__(self, "custom", {})
            self.custom[key] = _to_native(value)

    def __delitem__(self, key: str) -> None:
        """Delete custom field by key."""
        if key in self.__dataclass_fields__:
            raise KeyError(f"Cannot delete dataclass field: {key}")
        if self.custom is None or key not in self.custom:
            raise KeyError(key)
        del self.custom[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists (in dataclass fields or custom)."""
        if key in self.__dataclass_fields__:
            return True
        return self.custom is not None and key in self.custom

    def keys(self):
        """Return all keys (dataclass fields + custom keys)."""
        result = list(self.__dataclass_fields__.keys())
        if self.custom:
            result.extend(self.custom.keys())
        return result

    def values(self):
        """Return all values (dataclass fields + custom values)."""
        result = [getattr(self, k) for k in self.__dataclass_fields__]
        if self.custom:
            result.extend(self.custom.values())
        return result

    def items(self):
        """Return all (key, value) pairs."""
        result = [(k, getattr(self, k)) for k in self.__dataclass_fields__]
        if self.custom:
            result.extend(self.custom.items())
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key with default."""
        try:
            return self[key]
        except KeyError:
            return default

    def has_custom(self, name: str) -> bool:
        """Check if custom attribute exists."""
        return self.custom is not None and name in self.custom

    def with_custom(self, name: str, value: Any) -> "Supervision":
        """Return a copy with an extra custom field."""
        cpy = fastcopy(self, custom=self.custom.copy() if self.custom is not None else {})
        cpy.custom[name] = value
        return cpy

    def drop_custom(self, name: str) -> Optional["Supervision"]:
        """Remove custom attribute and return self."""
        if self.custom is None or name not in self.custom:
            return None
        del self.custom[name]
        return self

    # =========================================================================
    # Core methods
    # =========================================================================

    @property
    def is_bilingual(self) -> bool:
        return self.translation is not None and len(self.translation) > 0

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_alignment(self, kind: str, alignment: List[AlignmentItem]) -> "Supervision":
        alis = self.alignment
        if alis is None:
            alis = {}
        alis[kind] = alignment
        return fastcopy(self, alignment=alis)

    def with_offset(self, offset: Seconds) -> "Supervision":
        """Return an identical Supervision, but with the offset added to the start field."""
        return Supervision(
            id=self.id,
            recording_id=self.recording_id,
            start=round(self.start + offset, ndigits=8),
            duration=self.duration,
            channel=self.channel,
            text=self.text,
            language=self.language,
            speaker=self.speaker,
            gender=self.gender,
            translation=self.translation,
            target_lang=self.target_lang,
            custom=self.custom,
            alignment=self.alignment,
        )

    def trim(self, end: Seconds, start: Seconds = 0) -> "Supervision":
        """
        Return an identical Supervision, but ensure that self.start is not negative
        and self.end does not exceed the end parameter.
        """
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        return fastcopy(
            self,
            start=max(start, self.start),
            duration=_add_durations(self.duration, -end_exceeds_by, -start_exceeds_by),
            alignment=(
                {type_: [item.trim(end=end, start=start) for item in ali] for type_, ali in self.alignment.items()}
                if self.alignment
                else None
            ),
        )

    def transform_text(self, transform_fn: Callable[[str], str]) -> "Supervision":
        """Return a copy of the current segment with transformed text field."""
        if self.text is None:
            return self
        return fastcopy(self, text=transform_fn(self.text))

    def transform_alignment(self, transform_fn: Callable[[str], str], type: Optional[str] = "word") -> "Supervision":
        """Return a copy of the current segment with transformed alignment field."""
        if self.alignment is None:
            return self
        return fastcopy(
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
            data = _asdict_nonull(fastcopy(self, alignment=None))
            data["alignment"] = alis
            return data

    @staticmethod
    def from_dict(data: dict) -> "Supervision":
        if "alignment" in data:
            data["alignment"] = {k: [AlignmentItem.deserialize(x) for x in v] for k, v in data["alignment"].items()}
        return Supervision(**data)


__all__ = ["Pathlike", "Seconds", "AlignmentItem", "Supervision"]
