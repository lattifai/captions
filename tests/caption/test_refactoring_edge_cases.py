#!/usr/bin/env python3
"""
Test suite for refactored caption format edge cases, specifically original_speaker flag.
"""

from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision


class TestRefactoringEdgeCases:
    """Test specific logic introduced or refactored in the format handlers."""

    @pytest.mark.parametrize("format_ext", ["srt", "vtt", "sbv", "txt", "tsv", "aud", "TextGrid"])
    def test_original_speaker_flag_respect(self, tmp_path, format_ext):
        """
        Verify that original_speaker=False in custom metadata suppresses speaker inclusion
        even when include_speaker=True.

        Note: ASS and CSV formats have separate speaker columns/fields that always contain
        the speaker value regardless of original_speaker flag. These formats are excluded
        from this parametrized test and have their own specific tests.
        """
        supervisions = [
            Supervision(
                text="Hello world", start=1.0, duration=2.0, speaker="ALICE", custom={"original_speaker": False}
            ),
            Supervision(text="How are you", start=4.0, duration=2.0, speaker="BOB", custom={"original_speaker": True}),
            Supervision(
                text="I'm fine",
                start=7.0,
                duration=2.0,
                speaker="CHARLIE",
                # Defaults to True
            ),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / f"test.{format_ext}"

        # Write with include_speaker_in_text=True
        caption.write(output_file)

        # Read back or check content
        content = output_file.read_text(encoding="utf-8")

        # ALICE should NOT be in the text content (her original_speaker=False)
        assert "ALICE" not in content, f"Speaker ALICE should be suppressed in {format_ext}"

        # BOB and CHARLIE SHOULD be in the text content (either as labels or in specific columns)
        assert "BOB" in content, f"Speaker BOB should be present in {format_ext}"
        assert "CHARLIE" in content, f"Speaker CHARLIE should be present in {format_ext}"

    def test_original_speaker_flag_ass_format(self, tmp_path):
        """Test original_speaker flag behavior in ASS format.

        ASS format has a separate 'name' field for speaker which is always populated.
        The original_speaker flag controls whether speaker appears in the text field.
        """
        supervisions = [
            Supervision(
                text="Hello world", start=1.0, duration=2.0, speaker="ALICE", custom={"original_speaker": False}
            ),
            Supervision(text="How are you", start=4.0, duration=2.0, speaker="BOB", custom={"original_speaker": True}),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "test.ass"
        caption.write(output_file)

        content = output_file.read_text(encoding="utf-8")

        # ALICE has original_speaker=False → text should NOT have speaker prefix
        # BOB has original_speaker=True → text should have speaker prefix with separator
        assert ",ALICE," in content  # Name field
        assert ",BOB," in content  # Name field
        assert ",,Hello world" in content  # ALICE: no prefix in text
        assert ",,BOB: How are you" in content  # BOB: prefix with separator

    def test_original_speaker_flag_csv_format(self, tmp_path):
        """Test original_speaker flag behavior in CSV format.

        CSV format has a separate 'speaker' column which is always populated.
        The original_speaker flag controls whether speaker appears in the text column.
        """
        supervisions = [
            Supervision(
                text="Hello world", start=1.0, duration=2.0, speaker="ALICE", custom={"original_speaker": False}
            ),
            Supervision(text="How are you", start=4.0, duration=2.0, speaker="BOB", custom={"original_speaker": True}),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "test.csv"
        caption.write(output_file)

        content = output_file.read_text(encoding="utf-8")

        # Both speakers appear in speaker column
        # BOB should appear merged in text column: "BOB How are you"
        assert "BOB How are you" in content
        # ALICE should appear in speaker column but not merged in text: just "Hello world"
        assert "Hello world" in content

    def test_srt_no_double_colon(self, tmp_path):
        """Verify that SRT output doesn't have redundant colons when speaker has one."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="[SPEAKER_01]:"),
        ]
        caption = Caption.from_supervisions(supervisions)
        srt_file = tmp_path / "test.srt"
        caption.write(srt_file)

        content = srt_file.read_text(encoding="utf-8")
        # Should be "[SPEAKER_01]: Hello", not "[SPEAKER_01]:: Hello"
        assert "[SPEAKER_01]: Hello" in content
        assert "[SPEAKER_01]::" not in content

    def test_sbv_speaker_format(self, tmp_path):
        """Verify SBV speaker formatting (Speaker: Text)."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="ALICE"),
        ]
        caption = Caption.from_supervisions(supervisions)
        sbv_file = tmp_path / "test.sbv"
        caption.write(sbv_file)

        content = sbv_file.read_text(encoding="utf-8")
        assert "ALICE: Hello" in content

    def test_aud_speaker_format(self, tmp_path):
        """Verify AUD speaker formatting (speaker text)."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="ALICE"),
        ]
        caption = Caption.from_supervisions(supervisions)
        aud_file = tmp_path / "test.aud"
        caption.write(aud_file)

        content = aud_file.read_text(encoding="utf-8")
        # Speaker prepended with ': ' separator
        assert "ALICE: Hello" in content

    def test_txt_speaker_format(self, tmp_path):
        """Verify TXT speaker formatting (speaker: text)."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="ALICE"),
        ]
        caption = Caption.from_supervisions(supervisions)
        txt_file = tmp_path / "test.txt"
        caption.write(txt_file)

        content = txt_file.read_text(encoding="utf-8")
        # Speaker prepended with ': ' separator
        assert "ALICE: Hello" in content
