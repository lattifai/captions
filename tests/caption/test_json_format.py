#!/usr/bin/env python3
"""Tests for JSON format document-level metadata handling.

JSON is the only caption format that stores Caption-level metadata
(language, target_lang, kind, metadata) as top-level document fields.
These tests ensure Caption.read() preserves those fields.
"""

import json

import pytest

from lattifai.caption import Caption, Supervision


V2_DOC_WITH_LANGUAGE = {
    "language": "zh",
    "metadata": {
        "videoId": "la0CaZ2R8EY",
        "sourceType": "dub",
        "sampleRate": 24000,
    },
    "duration": 12.5,
    "num_segments": 2,
    "speakers": ["Alice", "Bob"],
    "supervisions": [
        {
            "text": "你好世界",
            "start": 0.0,
            "end": 5.0,
            "speaker": "Alice",
            "translation": "Hello world",
        },
        {
            "text": "再见",
            "start": 7.5,
            "end": 12.5,
            "speaker": "Bob",
            "translation": "Goodbye",
        },
    ],
}


class TestJSONDocumentMetadata:
    """Verify Caption.read() preserves document-level JSON fields."""

    def test_read_preserves_language(self, tmp_path):
        """Top-level 'language' field must be loaded into Caption.language."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(V2_DOC_WITH_LANGUAGE, ensure_ascii=False), encoding="utf-8")

        caption = Caption.read(json_file)

        assert caption.language == "zh", (
            f"Expected language='zh' from document field, got {caption.language!r}"
        )

    def test_read_preserves_metadata(self, tmp_path):
        """Top-level 'metadata' dict must be loaded into Caption.metadata."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(V2_DOC_WITH_LANGUAGE, ensure_ascii=False), encoding="utf-8")

        caption = Caption.read(json_file)

        assert caption.metadata is not None
        assert caption.metadata.get("videoId") == "la0CaZ2R8EY"
        assert caption.metadata.get("sourceType") == "dub"
        assert caption.metadata.get("sampleRate") == 24000

    def test_read_preserves_target_lang(self, tmp_path):
        """Top-level 'target_lang' must be loaded into Caption.target_lang."""
        doc = dict(V2_DOC_WITH_LANGUAGE)
        doc["target_lang"] = "en"

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

        caption = Caption.read(json_file)

        assert caption.target_lang == "en"

    def test_read_supervisions_still_work(self, tmp_path):
        """Regression: supervision parsing must still work alongside metadata."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(V2_DOC_WITH_LANGUAGE, ensure_ascii=False), encoding="utf-8")

        caption = Caption.read(json_file)

        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "你好世界"
        assert caption.supervisions[0].start == 0.0
        assert caption.supervisions[0].end == 5.0
        assert caption.supervisions[0].speaker == "Alice"
        assert caption.supervisions[0].translation == "Hello world"

    def test_v1_flat_array_still_works(self, tmp_path):
        """Regression: legacy v1 flat array format must still be readable."""
        v1_data = [
            {"text": "Hello", "start": 0.0, "end": 2.0},
            {"text": "World", "start": 2.5, "end": 4.5},
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(v1_data))

        caption = Caption.read(json_file)

        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "Hello"
        # v1 has no document metadata — language should be None
        assert caption.language is None

    def test_round_trip_preserves_document_metadata(self, tmp_path):
        """Write + read back must preserve language, target_lang, metadata."""
        original = Caption(
            supervisions=[
                Supervision(text="Hello", start=0.0, duration=2.0, speaker="Alice"),
            ],
            language="en",
            target_lang="zh",
            metadata={"videoId": "abc123", "custom_key": "custom_value"},
        )

        json_file = tmp_path / "roundtrip.json"
        original.write(json_file)

        readback = Caption.read(json_file)

        assert readback.language == "en"
        assert readback.target_lang == "zh"
        assert readback.metadata.get("videoId") == "abc123"
        assert readback.metadata.get("custom_key") == "custom_value"
        assert len(readback.supervisions) == 1
        assert readback.supervisions[0].text == "Hello"
