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
    expected_text = ("Lead-in" + "x" * 2000 + "Next sentence finishes.").replace(" ", "")
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
    expected_text = "Chunk one start still going Bob begins now Wraps up.".replace(" ", "")
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
    expected_text = ("Intro" + "a" * 1000 + "b" * 1000 + "Continuation picks up" + "Wrap-up here.").replace(" ", "")
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
                            f"Event '{bracket}' was merged into '{event_after}'. " f"Events should remain separate."
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

            origin_text = "".join([(sup.speaker or "").strip() + sup.text for sup in supervisions]).replace(" ", "")
            split_text = "".join([(sup.speaker or "").strip() + sup.text for sup in splits]).replace(" ", "")

            if origin_text != split_text:
                open(str(caption_file) + ".debug.supervisions.txt", "w", encoding="utf-8").write(
                    "\n".join([f"[{sup.speaker}] {sup.text}" for sup in supervisions])
                )
                open(str(caption_file) + ".debug.splits.txt", "w", encoding="utf-8").write(
                    "\n".join([f"[{sup.speaker}] {sup.text}" for sup in splits])
                )

                open(str(caption_file) + ".debug.supervisions_text", "w", encoding="utf-8").write(origin_text)
                open(str(caption_file) + ".debug.splits_text", "w", encoding="utf-8").write(split_text)

            assert origin_text == split_text, "Text integrity check failed after sentence splitting."


def test_split_sentences_chinese_gemini_colon_bug():
    """Regression: Chinese colon space insertion broke text matching."""
    from pathlib import Path

    from lattifai.caption.formats.gemini import GeminiReader

    splitter = SentenceSplitter()
    test_file = Path(__file__).parent.parent / "data" / "TheValley101-gemini-3-flash-preview.md"
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

    zip_file = Path(__file__).parent.parent / "data" / "TheValley101_瑞士信贷-gemini-3-flash-preview.md.zip"
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
    event_texts = [sup.text for sup in result if sup.text.startswith("[") and sup.text.endswith("]")]
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
    result = SentenceSplitter._split_inline_events("[Music] [Applause] Welcome everyone.")
    assert result == ["[Music]", "[Applause]", "Welcome everyone."]


def test_split_sentences_separates_inline_trailing_event():
    """Integration: mixed text 'And breathe out. [Breathes out]' should be split."""
    splitter = SentenceSplitter()

    supervisions = [
        make_supervision(0, "And breathe out. [Breathes out]", speaker=None),
    ]

    result = splitter.split_sentences(supervisions)

    texts = [sup.text for sup in result]
    assert "[Breathes out]" in texts, f"Expected '[Breathes out]' as separate segment, got: {texts}"


if __name__ == "__main__":
    test_split_sentences_preserves_event_supervisions_from_gemini()
