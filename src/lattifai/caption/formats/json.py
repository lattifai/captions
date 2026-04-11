"""JSON format handler for structured caption data.

JSON is the most flexible format for storing caption data, supporting:
- Segment-level timing (start, end)
- Word-level alignment (words array with per-word timestamps)
- Speaker labels
- Document-level metadata mirroring the Caption class
- Custom metadata

Output structure:
```json
{
    "language": "en",
    "target_lang": "zh",
    "metadata": {"videoId": "abc123"},
    "duration": 2.5,
    "num_segments": 1,
    "speakers": ["Speaker 1"],
    "supervisions": [
        {
            "text": "Hello world",
            "start": 0.0,
            "end": 2.5,
            "speaker": "Speaker 1",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.6, "end": 2.5}
            ]
        }
    ]
}
```

Also reads legacy v1 flat arrays for backward compatibility:
```json
[{"text": "Hello world", "start": 0.0, "end": 2.5}]
```
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler, ParseResult


@register_format("json")
class JSONFormat(FormatHandler):
    """JSON format for structured caption data.

    Features:
    - Document-level output mirroring the Caption class 1:1
    - Preserves full segment structure with timing
    - Supports word-level alignment in 'words' field
    - Round-trip compatible (read/write preserves all data)
    - Human-readable with indentation

    Read: auto-detects v1 (flat array) and v2 (document with 'supervisions' key).
    Write: always outputs v2 document structure.
    """

    extensions = [".json"]
    description = "JSON - structured caption data with word-level support"

    # ── Private helpers ──────────────────────────────────────────────

    @classmethod
    def _load_json(cls, source):
        """Load JSON from a file path or raw string content."""
        if cls.is_content(source):
            return json.loads(source)
        # Detect JSON content that is_content() misses (short strings starting with [ or {)
        if isinstance(source, str) and source.lstrip()[:1] in ("[", "{"):
            try:
                return json.loads(source)
            except (json.JSONDecodeError, ValueError):
                pass
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _parse_supervisions(cls, items: list, normalize_text: bool = True) -> list[Supervision]:
        """Parse a JSON array into Supervision objects."""
        from ..supervision import AlignmentItem

        supervisions = []
        for item in items:
            text = item.get("text", "")
            if normalize_text:
                text = normalize_text_fn(text)

            # Parse word-level alignment if present
            alignment = None
            if "words" in item and item["words"]:
                word_alignments = []
                for word_item in item["words"]:
                    word_text = word_item.get("word", "")
                    word_start = word_item.get("start", 0)
                    if "duration" in word_item:
                        word_duration = word_item["duration"]
                    elif "end" in word_item:
                        word_duration = word_item["end"] - word_start
                    else:
                        word_duration = 0
                    word_alignments.append(
                        AlignmentItem(
                            symbol=word_text, start=word_start, duration=word_duration, score=word_item.get("score")
                        )
                    )
                if word_alignments:
                    alignment = {"word": word_alignments}

            # Support both 'duration' and 'end' fields for segment timing
            start = item.get("start", 0)
            if "duration" in item:
                duration = item["duration"]
            elif "end" in item:
                duration = item["end"] - start
            else:
                duration = 0

            supervisions.append(
                Supervision(
                    id=item.get("id", ""),
                    recording_id=item.get("recording_id", ""),
                    channel=item.get("channel", 0),
                    text=text,
                    start=start,
                    duration=duration,
                    language=item.get("language"),
                    speaker=item.get("speaker"),
                    gender=item.get("gender"),
                    score=item.get("score"),
                    translation=item.get("translation"),
                    target_lang=item.get("target_lang"),
                    custom=item.get("custom"),
                    alignment=alignment,
                )
            )

        return supervisions

    @classmethod
    def _serialize_supervision(
        cls, sup: Supervision, *, include_words: bool = True
    ) -> dict[str, Any]:
        """Serialize a single Supervision to the canonical JSON segment dict.

        Args:
            sup: Supervision to serialize.
            include_words: When False, omit the ``words`` array even if word
                alignment is present (used by ``word_level=False``).
        """
        item: dict[str, Any] = {}
        if sup.id:
            item["id"] = sup.id
        if sup.recording_id:
            item["recording_id"] = sup.recording_id
        # channel default is 0; emit only if explicitly different
        if sup.channel != 0:
            item["channel"] = sup.channel
        item["text"] = sup.text
        item["start"] = round(sup.start, 4)
        item["end"] = round(sup.end, 4)
        if sup.language:
            item["language"] = sup.language
        if sup.speaker:
            item["speaker"] = sup.speaker
        if sup.gender:
            item["gender"] = sup.gender
        if sup.score is not None:
            item["score"] = round(sup.score, 4)

        # Word-level alignment is preserved by default (JSON is lossless),
        # but the caller can suppress it via include_words=False. Empty lists
        # are treated as "no data" to avoid emitting "words": [].
        from .base import has_word_alignment as _has_word_align

        if include_words and _has_word_align(sup):
            item["words"] = []
            for w in sup.alignment["word"]:
                word_dict: dict[str, Any] = {
                    "word": w.symbol,
                    "start": round(w.start, 4),
                    "end": round(w.start + w.duration, 4),
                }
                if w.score is not None:
                    word_dict["score"] = round(w.score, 4)
                item["words"].append(word_dict)

        if sup.translation:
            item["translation"] = sup.translation
        if sup.target_lang:
            item["target_lang"] = sup.target_lang

        if hasattr(sup, "custom") and sup.custom:
            custom = dict(sup.custom)
            custom.pop("speaker_diarize2", None)
            if custom:
                item["custom"] = custom

        return item

    @classmethod
    def _build_document(
        cls,
        supervisions: list[Supervision],
        metadata: dict[str, Any] | None = None,
        *,
        include_words: bool = True,
    ) -> dict[str, Any]:
        """Build document dict mirroring Caption-level fields.

        Args:
            supervisions: List of Supervision objects
            metadata: Optional dict of Caption-level fields (language, target_lang,
                kind, source_format, metadata, etc.)
            include_words: Forwarded to ``_serialize_supervision``.
        """
        meta = dict(metadata or {})

        # Auto-compute derived fields
        speakers = sorted({sup.speaker for sup in supervisions if sup.speaker})
        duration = round(max((sup.end for sup in supervisions), default=0), 4)

        # Remove fields that are computed or should not be serialized
        for key in ("supervisions", "duration", "num_segments", "speakers", "source_path"):
            meta.pop(key, None)

        # Build document with Caption field order
        doc: dict[str, Any] = {}
        for key in ("language", "target_lang", "kind", "source_format"):
            if key in meta:
                doc[key] = meta.pop(key)
        if "metadata" in meta:
            doc["metadata"] = meta.pop("metadata")
        # Remaining user-supplied fields
        doc.update(meta)
        # Computed fields
        doc["duration"] = duration
        doc["num_segments"] = len(supervisions)
        doc["speakers"] = speakers
        # Segments
        doc["supervisions"] = [
            cls._serialize_supervision(sup, include_words=include_words) for sup in supervisions
        ]

        return doc

    # ── Public API ───────────────────────────────────────────────────

    @classmethod
    def read(cls, source, normalize_text: bool = True, **kwargs) -> list[Supervision]:
        """Read JSON format (v1 flat array or v2 document, auto-detected).

        Args:
            source: File path or JSON string content
            normalize_text: Whether to normalize text content

        Returns:
            List of Supervision objects with alignment data if present
        """
        supervisions, _ = cls.read_document(source, normalize_text=normalize_text, **kwargs)
        return supervisions

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> ParseResult:
        """Parse JSON including document-level metadata (v2 only).

        For v2 documents, lifts top-level fields (language, target_lang, kind,
        metadata) into ParseResult so Caption.read() can preserve them.
        v1 flat arrays have no document metadata — only supervisions are returned.

        Args:
            source: File path or JSON string content
            normalize_text: Whether to normalize text content

        Returns:
            ParseResult with supervisions and document-level metadata.
        """
        supervisions, doc_metadata = cls.read_document(source, normalize_text=normalize_text, **kwargs)

        # v2 document: lift top-level fields; user metadata lives under "metadata" key.
        # v1 flat array: doc_metadata is empty dict, all getters return None.
        user_metadata = doc_metadata.get("metadata") or {}

        return ParseResult(
            supervisions=supervisions,
            language=doc_metadata.get("language"),
            target_lang=doc_metadata.get("target_lang"),
            kind=doc_metadata.get("kind"),
            format_metadata=user_metadata,
        )

    @classmethod
    def read_document(
        cls, source, normalize_text: bool = True, **kwargs
    ) -> tuple[list[Supervision], dict[str, Any]]:
        """Read JSON and return both supervisions and document metadata.

        Auto-detects format:
        - v1 (list): returns (supervisions, {})
        - v2 (dict with 'supervisions'): returns (supervisions, metadata)

        Args:
            source: File path or JSON string content
            normalize_text: Whether to normalize text content

        Returns:
            Tuple of (supervisions, document_metadata)
        """
        data = cls._load_json(source)

        if isinstance(data, list):
            return cls._parse_supervisions(data, normalize_text=normalize_text), {}

        if isinstance(data, dict) and "supervisions" in data:
            supervisions = cls._parse_supervisions(data["supervisions"], normalize_text=normalize_text)
            doc_metadata = {k: v for k, v in data.items() if k != "supervisions"}
            return supervisions, doc_metadata

        raise ValueError("Invalid JSON caption format: expected a list or a document object with 'supervisions'")

    @classmethod
    def write(
        cls,
        supervisions: list[Supervision],
        output_path,
        **kwargs,
    ) -> Path:
        """Write JSON document format.

        Args:
            supervisions: List of Supervision objects
            output_path: Output file path
            **kwargs: Pass metadata=dict for Caption-level fields, style for output behavior

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: list[Supervision], **kwargs) -> bytes:
        """Convert to JSON document format bytes.

        Always outputs v2 document structure with Caption-level metadata.

        Args:
            supervisions: List of Supervision objects
            **kwargs: Pass metadata=dict for Caption-level fields
                (language, target_lang, kind, source_format, metadata, etc.)

        Returns:
            JSON content as UTF-8 encoded bytes
        """
        import logging

        from .base import count_supervisions_with_words

        metadata = kwargs.pop("metadata", None)

        # JSON is a lossless serializer: word data is preserved by default,
        # but RenderConfig.word_level=False explicitly opts out of word arrays.
        # word_level=True is informational here (data is already preserved),
        # but if the user explicitly asks for word-level and there's nothing
        # to write, surface a warning so they can debug their producer.
        _, _, word_level = cls._unpack_render(**kwargs)
        include_words = word_level is not False
        if word_level is True:
            n_total = len(supervisions)
            n_with = count_supervisions_with_words(supervisions)
            if n_with == 0 and n_total > 0:
                logging.getLogger("lattifai.caption").warning(
                    "json: word_level=True but no supervisions carry word alignment; "
                    "output will not contain a 'words' array"
                )

        data = cls._build_document(supervisions, metadata=metadata, include_words=include_words)
        return json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8")
