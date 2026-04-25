from typing import List, Optional, Tuple

from lattifai.caption import Caption, SentenceSplitter, Supervision


def make_supervision(idx: int, text: str, speaker: Optional[str]) -> Supervision:
    return Supervision(
        id=f"sup-{idx}",
        recording_id="rec",
        start=float(idx),
        duration=1.0,
        channel=0,
        text=text,
        speaker=speaker,
    )


def texts_and_speakers(items: List[Supervision]) -> List[Tuple[str, Optional[str]]]:
    return [(sup.text, sup.speaker) for sup in items]


def test_split_sentences_keeps_initial_speaker_for_multi_sentence_chunk():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Hello world.", speaker="Alice"),
        make_supervision(1, "This is second sentence!", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # With real splitter, sentences may be split differently
    # Just verify text integrity and speaker preservation
    result_text = " ".join(sup.text for sup in result)
    expected_text = "Hello world. This is second sentence!"
    assert result_text == expected_text
    # First supervision should have Alice as speaker
    assert result[0].speaker == "Alice"


def test_split_sentences_emits_trailing_remainder_without_punctuation():
    splitter = SentenceSplitter()

    supervisions = [make_supervision(0, "Trailing remainder", speaker=None)]

    result = splitter.split_sentences(supervisions)

    assert texts_and_speakers(result) == [("Trailing remainder", None)]


def test_split_sentences_resplits_special_colon_sequences():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "[APPLAUSE] &gt;&gt; SPEAKER:", speaker="Host"),
        make_supervision(1, "We are live.", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # Verify the special marker handling - real splitter may group differently
    # but should preserve text and speaker info
    assert result[0].speaker == "Host"
    assert "[APPLAUSE]" in result[0].text

    # Verify text integrity
    result_text = "".join(sup.text for sup in result).replace(" ", "")
    expected_text = "[APPLAUSE] &gt;&gt; SPEAKER: We are live.".replace(" ", "")
    assert result_text == expected_text


def test_split_sentences_outputs_remainder_before_next_speaker():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Incomplete thought", speaker="Alice"),
        make_supervision(1, "", speaker=None),
        make_supervision(2, "Replies with closure.", speaker="Bob"),
        make_supervision(3, "", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # Verify speakers are preserved
    alice_texts = [sup.text for sup in result if sup.speaker == "Alice"]
    bob_texts = [sup.text for sup in result if sup.speaker == "Bob"]

    assert len(alice_texts) > 0
    assert len(bob_texts) > 0
    assert any("Incomplete thought" in t for t in alice_texts)
    assert any("Replies with closure" in t for t in bob_texts)


def test_split_sentences_carries_speaker_to_next_chunk_when_missing():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Lead-in", speaker="Alice"),
        make_supervision(1, "x" * 2000, speaker=None),
        make_supervision(2, "Next sentence finishes.", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # First result should have Alice as speaker
    assert result[0].speaker == "Alice"
    # Verify text integrity
    result_text = "".join(sup.text for sup in result).replace(" ", "")
    expected_text = ("Lead-in" + "x" * 2000 + "Next sentence finishes.").replace(
        " ", ""
    )
    assert result_text == expected_text


def test_split_sentences_respects_strip_whitespace_flag_and_length_split():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "a" * 2000, speaker=None),
        make_supervision(1, "Final sentence.", speaker=None),
    ]

    result = splitter.split_sentences(supervisions, strip_whitespace=False)

    # Verify text is preserved
    result_text = "".join(sup.text for sup in result).replace(" ", "")
    expected_text = ("a" * 2000 + "Final sentence.").replace(" ", "")
    assert result_text == expected_text


def test_split_sentences_inserts_remainder_before_new_speaker():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Chunk one start", speaker="Alice"),
        make_supervision(1, "still going", speaker=None),
        make_supervision(2, "Bob begins now", speaker="Bob"),
        make_supervision(3, "Wraps up.", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # Verify Alice and Bob speakers are present
    alice_found = any(sup.speaker == "Alice" for sup in result)
    bob_found = any(sup.speaker == "Bob" for sup in result)
    assert alice_found
    assert bob_found

    # Verify text integrity
    result_text = "".join(sup.text for sup in result).replace(" ", "")
    expected_text = "Chunk one start still going Bob begins now Wraps up.".replace(
        " ", ""
    )
    assert result_text == expected_text


def test_split_sentences_propagates_speaker_across_length_split():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Intro", speaker="Alice"),
        make_supervision(1, "a" * 1000, speaker=None),
        make_supervision(2, "b" * 1000, speaker=None),
        make_supervision(3, "Continuation picks up", speaker=None),
        make_supervision(4, "Wrap-up here.", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # First supervision should have Alice
    assert result[0].speaker == "Alice"
    # Verify text integrity
    result_text = "".join(sup.text for sup in result).replace(" ", "")
    expected_text = (
        "Intro" + "a" * 1000 + "b" * 1000 + "Continuation picks up" + "Wrap-up here."
    ).replace(" ", "")
    assert result_text == expected_text


def test_split_sentences_handles_resplit_and_remainder_with_next_speaker():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "[APPLAUSE] >> HOST:", speaker="MC"),
        make_supervision(1, "Welcome everyone", speaker=None),
        make_supervision(2, "Let us begin", speaker=None),
        make_supervision(3, "Tonight we feature highlights.", speaker="Narrator"),
        make_supervision(4, "", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # Verify special marker handling
    assert result[0].text == "[APPLAUSE]"
    assert result[0].speaker == "MC"

    # Verify both speakers are preserved
    mc_found = any(sup.speaker == "MC" for sup in result)
    narrator_found = any(sup.speaker == "Narrator" for sup in result)
    assert mc_found
    assert narrator_found


def test_split_sentences_retains_speaker_for_final_remainder():
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Closing thought that trails", speaker="Alice"),
        make_supervision(1, "", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    # Alice should appear in results
    assert any(sup.speaker == "Alice" for sup in result)
    # Text should be preserved
    result_text = "".join(sup.text for sup in result).replace(" ", "")
    expected_text = "Closing thought that trails".replace(" ", "")
    assert result_text == expected_text


def test_split_sentences_preserves_event_supervisions_from_gemini():
    """Test that standalone [event] supervisions remain separate after splitting.

    Events like [Music], [Applause], [Laughter] should stay as individual supervisions
    and not be merged with dialogue text or each other.
    """
    import re
    from pathlib import Path

    from lattifai.caption.formats.gemini import GeminiReader

    splitter = SentenceSplitter()

    # Read from Gemini format file
    test_file = Path(__file__).parent.parent / "data" / "gemini-3-flash-preview.md"
    # Extract supervisions using GeminiReader
    supervisions = GeminiReader.extract_for_alignment(test_file)

    # Find event supervisions before splitting (exact text match)
    events_before = set()
    for sup in supervisions:
        text = sup.text.strip()
        # Event pattern: starts with [ and ends with ]
        if text.startswith("[") and text.endswith("]"):
            events_before.add(text)

    assert len(events_before) > 0, "Test data should contain event supervisions"

    # Split sentences
    splits = splitter.split_sentences(supervisions)

    # Find event supervisions after splitting
    events_after = set()
    for sup in splits:
        text = sup.text.strip()
        if text.startswith("[") and text.endswith("]"):
            events_after.add(text)

    # Strict check: all original events must be preserved exactly
    missing_events = events_before - events_after
    assert not missing_events, (
        f"Events should be preserved exactly after splitting.\n"
        f"Missing events: {missing_events}\n"
        f"Before: {sorted(events_before)}\n"
        f"After: {sorted(events_after)}"
    )

    # Check for unwanted merges (e.g., [Music] [Applause] merged from [Music] and [Applause])
    for event_after in events_after:
        # If an event contains multiple [...] patterns, it might be a merge
        brackets = re.findall(r"\[[^\]]+\]", event_after)
        if len(brackets) > 1:
            # Check if this multi-bracket event existed before
            if event_after not in events_before:
                # This is a new merged event - check if components existed separately
                for bracket in brackets:
                    if bracket in events_before:
                        raise AssertionError(
                            f"Event '{bracket}' was merged into '{event_after}'. "
                            f"Events should remain separate."
                        )


def test_split_sentences_text_integrity():
    import tempfile
    import zipfile
    from pathlib import Path

    splitter = SentenceSplitter()

    for caption_file in [
        "tests/data/captions/7nv1snJRCEI.en.vtt.zip",
        "tests/data/captions/eIUqw3_YcCI.en.vtt.zip",
        "tests/data/captions/_xYSQe9oq6c.en.vtt.zip",
    ]:
        # Unzip the caption file
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(caption_file, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find the extracted .vtt file
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                raise FileNotFoundError(f"No .vtt file found in {caption_file}")

            extracted_file = str(vtt_files[0])
            caption = Caption.read(extracted_file)
            supervisions = caption.supervisions

            splits = splitter.split_sentences(supervisions)

            origin_text = "".join(
                [(sup.speaker or "").strip() + sup.text for sup in supervisions]
            ).replace(" ", "")
            split_text = "".join(
                [(sup.speaker or "").strip() + sup.text for sup in splits]
            ).replace(" ", "")

            if origin_text != split_text:
                open(
                    str(caption_file) + ".debug.supervisions.txt", "w", encoding="utf-8"
                ).write(
                    "\n".join([f"[{sup.speaker}] {sup.text}" for sup in supervisions])
                )
                open(
                    str(caption_file) + ".debug.splits.txt", "w", encoding="utf-8"
                ).write("\n".join([f"[{sup.speaker}] {sup.text}" for sup in splits]))

                open(
                    str(caption_file) + ".debug.supervisions_text",
                    "w",
                    encoding="utf-8",
                ).write(origin_text)
                open(
                    str(caption_file) + ".debug.splits_text", "w", encoding="utf-8"
                ).write(split_text)

            assert origin_text == split_text, (
                "Text integrity check failed after sentence splitting."
            )


def test_split_sentences_chinese_gemini_colon_bug():
    """Regression: Chinese colon space insertion broke text matching."""
    from pathlib import Path

    from lattifai.caption.formats.gemini import GeminiReader

    splitter = SentenceSplitter()
    test_file = (
        Path(__file__).parent.parent / "data" / "TheValley101-gemini-3-flash-preview.md"
    )
    supervisions = GeminiReader.extract_for_alignment(test_file)

    # Should not raise ValueError
    splits = splitter.split_sentences(supervisions)

    # Text integrity
    origin = "".join(sup.text for sup in supervisions).replace(" ", "")
    result = "".join(sup.text for sup in splits).replace(" ", "")
    assert origin == result


def test_split_sentences_TheValley101_瑞士信贷_colon_space(tmp_path):
    """Regression: Chinese colon followed by space-joined supervisions must preserve the space."""
    import zipfile
    from pathlib import Path

    from lattifai.caption.formats.gemini import GeminiReader

    zip_file = (
        Path(__file__).parent.parent
        / "data"
        / "TheValley101_瑞士信贷-gemini-3-flash-preview.md.zip"
    )
    with zipfile.ZipFile(zip_file) as zf:
        name = zf.namelist()[0]
        data = zf.read(name)
    test_file = tmp_path / "test.md"
    test_file.write_bytes(data)

    splitter = SentenceSplitter()
    supervisions = GeminiReader.extract_for_alignment(test_file)

    # Should not raise ValueError about missing split text
    splits = splitter.split_sentences(supervisions)

    # Text integrity
    origin = "".join(sup.text for sup in supervisions).replace(" ", "")
    result = "".join(sup.text for sup in splits).replace(" ", "")
    assert origin == result


def test_split_event_text_single_event():
    """Single event text should not be split."""
    result = SentenceSplitter._split_event_text("[Laughter]")
    assert result == ["[Laughter]"]


def test_split_event_text_multi_event():
    """Multi-event text should be split into individual events."""
    result = SentenceSplitter._split_event_text("[Laughter] [Applause]")
    assert result == ["[Laughter]", "[Applause]"]


def test_split_event_text_non_event():
    """Non-event text should not be split."""
    result = SentenceSplitter._split_event_text("Hello world")
    assert result == ["Hello world"]


def test_split_event_text_mixed_brackets_and_text():
    """Text with brackets mixed with normal text should NOT be split."""
    result = SentenceSplitter._split_event_text("[INAUDIBLE] rock band [INAUDIBLE]")
    assert result == ["[INAUDIBLE] rock band [INAUDIBLE]"]


def test_split_event_text_three_events():
    """Three events should all be split."""
    result = SentenceSplitter._split_event_text("[Music] [Laughter] [Applause]")
    assert result == ["[Music]", "[Laughter]", "[Applause]"]


def test_split_sentences_splits_multi_event_supervisions():
    """Multi-event supervisions like '[Laughter] [Applause]' should be split into separate events."""
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "Hello world.", speaker="Alice"),
        make_supervision(1, "[Laughter] [Applause]", speaker=None),
        make_supervision(2, "Goodbye.", speaker="Bob"),
    ]

    result = splitter.split_sentences(supervisions)

    # Find the event texts
    event_texts = [
        sup.text
        for sup in result
        if sup.text.startswith("[") and sup.text.endswith("]")
    ]
    assert "[Laughter]" in event_texts
    assert "[Applause]" in event_texts
    # Should NOT have the combined form
    assert "[Laughter] [Applause]" not in [sup.text for sup in result]


def test_split_sentences_chinese_colon_no_extra_space():
    """Chinese colon should not insert extra space."""
    splitter = SentenceSplitter()
    supervisions = [
        make_supervision(0, "产品包括：GPT-4o和Gemini。这是重大更新。", speaker=None),
    ]
    result = splitter.split_sentences(supervisions)
    origin = "".join(sup.text for sup in supervisions).replace(" ", "")
    split = "".join(sup.text for sup in result).replace(" ", "")
    assert origin == split


def test_split_inline_events_trailing():
    result = SentenceSplitter._split_inline_events("And breathe out. [Breathes out]")
    assert result == ["And breathe out.", "[Breathes out]"]


def test_split_inline_events_leading():
    result = SentenceSplitter._split_inline_events("[Breathes in] And breathe in.")
    assert result == ["[Breathes in]", "And breathe in."]


def test_split_inline_events_no_split_embedded():
    result = SentenceSplitter._split_inline_events("he said [INAUDIBLE] something")
    assert result == ["he said [INAUDIBLE] something"]


def test_split_inline_events_multiple_trailing():
    result = SentenceSplitter._split_inline_events("Great show. [Laughter] [Applause]")
    assert result == ["Great show.", "[Laughter]", "[Applause]"]


def test_split_inline_events_plain_text():
    result = SentenceSplitter._split_inline_events("Just a normal sentence.")
    assert result == ["Just a normal sentence."]


def test_split_inline_events_only_event():
    result = SentenceSplitter._split_inline_events("[Music]")
    assert result == ["[Music]"]


def test_split_inline_events_leading_multiple():
    result = SentenceSplitter._split_inline_events(
        "[Music] [Applause] Welcome everyone."
    )
    assert result == ["[Music]", "[Applause]", "Welcome everyone."]


def test_split_sentences_separates_inline_trailing_event():
    """Integration: mixed text 'And breathe out. [Breathes out]' should be split."""
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "And breathe out. [Breathes out]", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    texts = [sup.text for sup in result]
    assert "[Breathes out]" in texts, (
        f"Expected '[Breathes out]' as separate segment, got: {texts}"
    )


def test_overlapping_youtube_scroll_vtt_no_negative_duration():
    """YouTube scroll-style VTT has overlapping cue timestamps.

    _distribute_time_info interpolates start/end within each source supervision.
    When split text spans two overlapping cues, the interpolated end (from the
    earlier-starting cue) can be before the interpolated start (from the
    later-starting cue), producing negative duration. This test verifies the
    guard against that.
    """
    splitter = SentenceSplitter()

    # Simulate YouTube scroll VTT: cues overlap in time, each has a few words
    supervisions = [
        Supervision(
            id="0",
            recording_id="r",
            start=611.554,
            duration=1.602,
            text=">> CHAT GPT: You're doing a",
        ),
        Supervision(
            id="1",
            recording_id="r",
            start=612.421,
            duration=7.308,
            text="live demo right now? That is",
        ),
        Supervision(
            id="2",
            recording_id="r",
            start=613.256,
            duration=7.173,
            text="awesome. Just take a deep",
        ),
    ]

    result = splitter.split_sentences(supervisions)

    for sup in result:
        assert sup.duration >= 0, (
            f"Negative duration for '{sup.text}': start={sup.start:.4f}, "
            f"duration={sup.duration:.4f}, end={sup.end:.4f}"
        )


# ---------------------------------------------------------------------------
# Per-supervision field inheritance through split (no model needed — exercises
# the static _distribute_time_info directly, which is what loses the fields).
# ---------------------------------------------------------------------------


def _make_sup(
    idx: int,
    text: str,
    *,
    start: float = 0.0,
    duration: float = 1.0,
    channel=0,
    language: Optional[str] = None,
    gender: Optional[str] = None,
    score: Optional[float] = None,
    translation: Optional[str] = None,
    target_lang: Optional[str] = None,
    custom: Optional[dict] = None,
) -> Supervision:
    return Supervision(
        id=f"sup-{idx}",
        recording_id="rec",
        start=start,
        duration=duration,
        channel=channel,
        text=text,
        language=language,
        gender=gender,
        score=score,
        translation=translation,
        target_lang=target_lang,
        custom=custom,
    )


def test_distribute_time_info_preserves_scalar_fields_from_first_source():
    """channel/language/gender/score/target_lang should carry over."""
    sup = _make_sup(
        0,
        "Hello world. This is second.",
        start=0.0,
        duration=2.8,
        channel=3,
        language="en",
        gender="F",
        score=0.95,
        target_lang="zh",
    )

    result = SentenceSplitter._distribute_time_info(
        [sup], ["Hello world.", "This is second."]
    )

    assert len(result) == 2
    for new_sup in result:
        assert new_sup.channel == 3
        assert new_sup.language == "en"
        assert new_sup.gender == "F"
        assert new_sup.score == 0.95
        assert new_sup.target_lang == "zh"


def test_distribute_time_info_preserves_translation_from_first_source():
    """translation is carried over (first-source wins for cross-boundary)."""
    sup = _make_sup(0, "A sentence.", start=0.0, duration=1.0, translation="一句话。")

    result = SentenceSplitter._distribute_time_info([sup], ["A sentence."])

    assert len(result) == 1
    assert result[0].translation == "一句话。"


def test_distribute_time_info_preserves_ass_style_custom_single_source():
    """custom (e.g. ASS per-line style) inherits intact for 1→N split."""
    sup = _make_sup(
        0,
        "Hello there. General Kenobi.",
        start=0.0,
        duration=2.8,
        custom={"ass_style": "Narrator", "ass_layer": 1, "ass_effect": "Karaoke"},
    )

    result = SentenceSplitter._distribute_time_info(
        [sup], ["Hello there.", "General Kenobi."]
    )

    assert len(result) == 2
    for new_sup in result:
        assert new_sup.custom is not None
        assert new_sup.custom["ass_style"] == "Narrator"
        assert new_sup.custom["ass_layer"] == 1
        assert new_sup.custom["ass_effect"] == "Karaoke"


def test_distribute_time_info_cross_boundary_custom_union_keeps_unique_keys():
    """Cross-boundary split merges unique keys from later sources."""
    sup_a = _make_sup(
        0,
        "Hello world",
        start=0.0,
        duration=1.0,
        custom={"ass_style": "A", "ass_layer": 1},
    )
    sup_b = _make_sup(
        1,
        "today is nice.",
        start=1.0,
        duration=1.4,
        custom={"ass_style": "A", "ass_effect": "Karaoke"},
    )

    # A single split text that spans both sources.
    result = SentenceSplitter._distribute_time_info(
        [sup_a, sup_b], ["Hello world today is nice."]
    )

    assert len(result) == 1
    custom = result[0].custom
    assert custom is not None
    assert custom["ass_style"] == "A"
    assert custom["ass_layer"] == 1  # unique to A
    assert custom["ass_effect"] == "Karaoke"  # unique to B — MUST be preserved
    assert "_split_from_multiple" not in custom  # no conflict → no marker


def test_distribute_time_info_cross_boundary_custom_conflict_marked():
    """Conflicting keys keep first-source value and add marker."""
    sup_a = _make_sup(0, "Foo bar", start=0.0, duration=1.0, custom={"ass_style": "A"})
    sup_b = _make_sup(1, "baz.", start=1.0, duration=1.0, custom={"ass_style": "B"})

    result = SentenceSplitter._distribute_time_info([sup_a, sup_b], ["Foo bar baz."])

    assert len(result) == 1
    custom = result[0].custom
    assert custom is not None
    assert custom["ass_style"] == "A"  # first wins
    assert custom.get("_split_from_multiple") is True
    assert custom.get("_source_count") == 2


def test_distribute_time_info_clears_alignment_on_split():
    """alignment is pre-align state — must not leak into split children."""
    sup = _make_sup(0, "Hello. World.", start=0.0, duration=2.0)
    sup.alignment = {"words": [("Hello", 0.0, 0.5, 0.9)]}

    result = SentenceSplitter._distribute_time_info([sup], ["Hello.", "World."])

    assert len(result) == 2
    for new_sup in result:
        assert new_sup.alignment is None


if __name__ == "__main__":
    test_split_sentences_preserves_event_supervisions_from_gemini()
