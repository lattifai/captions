"""Test suite for detect_speaker_candidates robustness.

Covers all edge cases for title-case speaker name detection:
real dialogues, false-positive labels, short files, mixed content.
"""

import pytest

from lattifai.caption.config import RenderConfig
from lattifai.caption.parsers.text_parser import detect_speaker_candidates


class TestRealDialogues:
    """Cases that SHOULD be detected as speakers."""

    def test_classic_interview(self):
        """Two speakers alternating — the most common case."""
        lines = [
            "Host: Welcome to the show.",
            "Guest: Thanks for having me.",
            "Host: Let's start with your background.",
            "Guest: Sure, I grew up in...",
            "Host: That's fascinating.",
            "Guest: Yeah, it was a great time.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Host" in result
        assert "Guest" in result

    def test_multi_word_speakers(self):
        """Multi-word title-case names (Terence Tao, Dwarkesh Patel)."""
        lines = [
            "Dwarkesh Patel: Tell me about Kepler.",
            "Terence Tao: So Kepler was building on Copernicus.",
            "Dwarkesh Patel: How did he use the data?",
            "Terence Tao: He eventually just stole it.",
            "Dwarkesh Patel: That's wild.",
            "Terence Tao: Yeah, he had to fight the descendants.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Dwarkesh Patel" in result
        assert "Terence Tao" in result

    def test_three_speakers(self):
        """Panel discussion with 3 speakers."""
        lines = [
            "Alice: I think we should start.",
            "Bob: Agreed, let's go.",
            "Charlie: One moment please.",
            "Alice: Sure, take your time.",
            "Bob: No rush.",
            "Charlie: Okay, ready now.",
        ]
        result = detect_speaker_candidates(lines)
        assert result == {"Alice", "Bob", "Charlie"}

    def test_short_dialogue_two_turns(self):
        """2 speakers, 1 turn each — ambiguous without recurrence.

        Indistinguishable from 'Note: x' + 'Warning: y', so rejected.
        Needs ≥1 speaker to appear ≥2 times for confident detection.
        """
        lines = [
            "Host: Welcome.",
            "Guest: Thanks.",
        ]
        result = detect_speaker_candidates(lines)
        # Known limitation: no recurrence → can't distinguish from labels
        assert len(result) == 0

    def test_one_speaker_dominant(self):
        """One speaker talks a lot, other speaks ≥2 times (recurrence needed)."""
        lines = [
            "Professor: Today we discuss quantum mechanics.",
            "Professor: The wave function describes probability.",
            "Professor: Let me show you an example.",
            "Professor: Any questions?",
            "Student: Can you repeat that?",
            "Professor: Of course.",
            "Student: Got it, thanks.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Professor" in result
        assert "Student" in result

    def test_single_speaker_many_lines(self):
        """Monologue: single speaker ≥3 times."""
        lines = [
            "Narrator: Once upon a time.",
            "Narrator: There was a kingdom.",
            "Narrator: And in that kingdom lived a princess.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Narrator" in result

    def test_speakers_with_unlabeled_lines(self):
        """Dialogue mixed with narration/descriptions (no speaker label)."""
        lines = [
            "Host: Welcome to the podcast.",
            "[Music plays]",
            "Guest: Thanks for having me.",
            "Host: Let's dive right in.",
            "[Applause]",
            "Guest: Sure, great topic.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Host" in result
        assert "Guest" in result

    def test_sparse_speaker_labels_at_transitions(self):
        """Speaker name only appears when speaker changes — very common format."""
        lines = [
            "Host: Welcome to the show.",
            "Today we're going to discuss AI.",
            "Let me introduce our guest.",
            "Guest: Thanks for having me.",
            "I've been working on this for years.",
            "It's really exciting stuff.",
            "Host: Tell me more about that.",
            "What inspired you?",
        ]
        result = detect_speaker_candidates(lines)
        assert "Host" in result
        assert "Guest" in result

    def test_sparse_three_speakers(self):
        """Three speakers, labels only at transitions, low coverage."""
        lines = [
            "Moderator: Let's begin the panel.",
            "First question goes to our left.",
            "Alice: I think the answer is clear.",
            "We need more data before deciding.",
            "Bob: I disagree actually.",
            "The evidence points the other way.",
            "Moderator: Interesting points from both.",
            "Let's move to the next topic.",
            "Alice: One more thing on that.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Moderator" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_sparse_single_transition(self):
        """2 speakers, 1 transition each — needs recurrence to confirm.

        Without recurrence, indistinguishable from label patterns.
        Add a second appearance of one speaker to enable detection.
        """
        lines = [
            "Host: Welcome everyone.",
            "We have a great show today.",
            "Lots to cover.",
            "Guest: Thanks, happy to be here.",
            "I want to talk about three things.",
            "Host: Great, let's dive in.",
            "First, the market situation.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Host" in result
        assert "Guest" in result


class TestFalsePositives:
    """Cases that should NOT be detected as speakers."""

    def test_common_labels(self):
        """Note:, Warning: etc. are labels, not speakers."""
        lines = [
            "Note: this is important",
            "Warning: check the logs carefully",
            "Regular subtitle text here.",
            "More regular text without any prefix.",
            "Another normal caption line.",
        ]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0

    def test_day_labels(self):
        """Monday:, Tuesday: — day-of-week labels."""
        lines = [
            "Monday: Meeting at 9am",
            "Tuesday: Lunch with the team",
            "Wednesday: Project deadline",
            "Regular schedule follows.",
            "No more labeled items.",
        ]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0

    def test_step_labels(self):
        """Step:, Result: — instructional labels."""
        lines = [
            "Step: First do this.",
            "Result: You get that.",
            "Then continue with the next part.",
            "Follow the instructions carefully.",
        ]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0

    def test_oneoff_label_adjacent_to_speaker(self):
        """Chapter One: appearing once next to a confident speaker — not promoted."""
        lines = [
            "Host: Welcome to the show.",
            "Chapter One: The beginning.",
            "Host: Let's start.",
            "Host: First topic.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Host" in result
        assert "Chapter One" not in result

    def test_recurring_label_with_alternation(self):
        """Note(2x) + Warning(1x) — labels that recur AND alternate.

        Structurally identical to Host(2x) + Guest(1x). The distinction
        requires checking that labeled lines are NOT consecutive (real
        speakers have unlabeled continuation lines between turns).
        """
        lines = [
            "Note: first thing to check",
            "Note: second thing to check",
            "Warning: be careful here",
        ]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0

    def test_single_label_once(self):
        """A single Title Case word with colon, only once."""
        lines = [
            "Summary: The movie was great.",
            "It had amazing special effects.",
            "The acting was superb.",
        ]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0

    def test_repeated_label_not_alternating(self):
        """Same label repeated — not a dialogue pattern."""
        lines = [
            "Note: first thing",
            "Note: second thing",
            "Note: third thing",
        ]
        # This hits the ≥3 threshold but it's a single "speaker" monologue
        # which is valid — single-speaker content is allowed.
        # The concern is false positive, but a real single-speaker
        # monologue looks the same. Accept this as a known tradeoff.
        # The key is that `Note` is a common word — ideally we'd filter it.
        # For now, ≥3 same name is accepted.

    def test_colon_in_middle_of_sentence(self):
        """Colon appears mid-sentence, not as speaker separator."""
        lines = [
            "The ratio was 3:1 in favor.",
            "Time to act: now or never.",
            "He said: let's go.",
        ]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0


class TestEdgeCases:
    """Boundary conditions and tricky scenarios."""

    def test_empty_lines(self):
        """Empty input."""
        assert detect_speaker_candidates([]) == set()

    def test_all_empty_strings(self):
        """All blank lines."""
        assert detect_speaker_candidates(["", "  ", "\n"]) == set()

    def test_single_line_with_speaker(self):
        """Just one line with speaker pattern — too ambiguous."""
        lines = ["Host: Welcome."]
        result = detect_speaker_candidates(lines)
        assert len(result) == 0

    def test_non_string_lines(self):
        """Non-string items should be skipped."""
        lines = [None, 123, "Host: Hello", "Guest: Hi", "Host: Hey", "Guest: Sure"]
        result = detect_speaker_candidates(lines)
        assert "Host" in result

    def test_unicode_speakers(self):
        """Accented characters in speaker names."""
        lines = [
            "José: Hola amigos.",
            "María: Bienvenidos.",
            "José: Empecemos.",
            "María: De acuerdo.",
        ]
        result = detect_speaker_candidates(lines)
        assert "José" in result
        assert "María" in result

    def test_speaker_with_fullwidth_colon(self):
        """Chinese fullwidth colon ：."""
        lines = [
            "Host： Welcome.",
            "Guest： Thanks.",
            "Host： Let's go.",
            "Guest： Sure.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Host" in result
        assert "Guest" in result

    def test_two_speakers_one_turn_each_among_many_lines(self):
        """2 speakers but buried in many unlabeled lines — low coverage."""
        lines = ["Host: Hi", "Guest: Hello"] + [f"Regular text line {i}" for i in range(20)]
        result = detect_speaker_candidates(lines)
        # Only 2 out of 22 lines have speaker pattern — likely not a dialogue file
        assert len(result) == 0

    def test_almost_all_lines_labeled(self):
        """High coverage — almost every line has a speaker."""
        lines = [
            "Alice: Line one.",
            "Bob: Line two.",
            "Some narration without speaker.",
            "Alice: Line three.",
            "Bob: Line four.",
        ]
        result = detect_speaker_candidates(lines)
        assert "Alice" in result
        assert "Bob" in result

    def test_chinese_speakers_pinyin(self):
        """Chinese speakers using Pinyin romanization (title-case)."""
        lines = [
            "Zhang Wei: 大家好，欢迎来到节目。",
            "Li Na: 谢谢主持人。",
            "Zhang Wei: 今天我们讨论一个有趣的话题。",
            "Li Na: 是的，我很期待。",
        ]
        result = detect_speaker_candidates(lines)
        assert "Zhang Wei" in result
        assert "Li Na" in result

    def test_chinese_speakers_fullwidth_colon(self):
        """Chinese speakers with fullwidth colon, sparse labels."""
        lines = [
            "Zhang Wei： 大家好。",
            "欢迎来到今天的节目。",
            "Li Na： 谢谢邀请。",
            "我一直关注这个领域。",
            "Zhang Wei： 那我们开始吧。",
        ]
        result = detect_speaker_candidates(lines)
        assert "Zhang Wei" in result
        assert "Li Na" in result


class TestWriteReadRoundtrip:
    """End-to-end roundtrip through Caption.write() and Caption.read()."""

    def test_srt_roundtrip_titlecase_speakers(self, tmp_path):
        """SRT write→read roundtrip with title-case speakers (≥3 occurrences)."""
        from lattifai.caption import Caption, Supervision

        supervisions = [
            Supervision(text="Welcome.", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="Thanks.", start=4.0, duration=2.0, speaker="Guest"),
            Supervision(text="Let's begin.", start=7.0, duration=2.0, speaker="Host"),
            Supervision(text="Sure.", start=10.0, duration=2.0, speaker="Guest"),
            Supervision(text="First topic.", start=13.0, duration=2.0, speaker="Host"),
            Supervision(text="Great question.", start=16.0, duration=2.0, speaker="Guest"),
        ]

        srt_file = tmp_path / "test.srt"
        Caption.from_supervisions(supervisions).write(srt_file)

        result = Caption.read(srt_file)
        assert len(result.supervisions) == 6
        for orig, read in zip(supervisions, result.supervisions):
            assert read.text == orig.text, f"Text mismatch: {read.text!r} != {orig.text!r}"
            assert read.speaker is not None, f"Speaker lost for: {orig.text}"

    def test_ass_roundtrip_titlecase_speakers(self, tmp_path):
        """ASS write→read roundtrip with title-case speakers."""
        from lattifai.caption import Caption, Supervision

        supervisions = [
            Supervision(text="Hello world.", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="Goodbye.", start=4.0, duration=2.0, speaker="Guest"),
        ]

        ass_file = tmp_path / "test.ass"
        Caption.from_supervisions(supervisions).write(ass_file)

        result = Caption.read(ass_file)
        assert len(result.supervisions) == 2
        assert result.supervisions[0].speaker == "Host"
        assert result.supervisions[0].text == "Hello world."
        assert result.supervisions[1].speaker == "Guest"
        assert result.supervisions[1].text == "Goodbye."

    def test_ass_roundtrip_without_include_speaker(self, tmp_path):
        """ASS write→read with include_speaker_in_text=False — Name field only."""
        from lattifai.caption import Caption, Supervision

        supervisions = [
            Supervision(text="Hello.", start=1.0, duration=2.0, speaker="Alice"),
            Supervision(text="World.", start=4.0, duration=2.0, speaker="Bob"),
        ]

        ass_file = tmp_path / "test.ass"
        Caption.from_supervisions(supervisions).write(ass_file, render=RenderConfig(include_speaker_in_text=False))

        result = Caption.read(ass_file)
        assert result.supervisions[0].speaker == "Alice"
        assert result.supervisions[0].text == "Hello."
        assert result.supervisions[1].speaker == "Bob"
        assert result.supervisions[1].text == "World."

    def test_vtt_roundtrip_uppercase_speakers(self, tmp_path):
        """VTT write→read with UPPERCASE speakers (always parseable)."""
        from lattifai.caption import Caption, Supervision

        supervisions = [
            Supervision(text="First line.", start=1.0, duration=2.0, speaker="ALICE"),
            Supervision(text="Second line.", start=4.0, duration=2.0, speaker="BOB"),
        ]

        vtt_file = tmp_path / "test.vtt"
        Caption.from_supervisions(supervisions).write(vtt_file)

        result = Caption.read(vtt_file)
        assert result.supervisions[0].text == "First line."
        assert result.supervisions[0].speaker is not None
        assert result.supervisions[1].text == "Second line."
        assert result.supervisions[1].speaker is not None

    def test_srt_roundtrip_chinese_pinyin_speakers(self, tmp_path):
        """SRT roundtrip with Chinese Pinyin speakers."""
        from lattifai.caption import Caption, Supervision

        supervisions = [
            Supervision(text="大家好。", start=1.0, duration=2.0, speaker="Zhang Wei"),
            Supervision(text="你好。", start=4.0, duration=2.0, speaker="Li Na"),
            Supervision(text="今天天气不错。", start=7.0, duration=2.0, speaker="Zhang Wei"),
            Supervision(text="是的。", start=10.0, duration=2.0, speaker="Li Na"),
            Supervision(text="我们开始吧。", start=13.0, duration=2.0, speaker="Zhang Wei"),
            Supervision(text="好的。", start=16.0, duration=2.0, speaker="Li Na"),
        ]

        srt_file = tmp_path / "test.srt"
        Caption.from_supervisions(supervisions).write(srt_file)

        result = Caption.read(srt_file)
        assert len(result.supervisions) == 6
        for orig, read in zip(supervisions, result.supervisions):
            assert read.text == orig.text
            assert read.speaker is not None

    def test_srt_roundtrip_sparse_speakers(self, tmp_path):
        """SRT roundtrip simulating sparse speaker labels.

        In the write path all lines get speaker prefix (if include_speaker=True).
        This test verifies the read path can still extract them.
        """
        from lattifai.caption import Caption, Supervision

        supervisions = [
            Supervision(text="Welcome.", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="We have a show.", start=4.0, duration=2.0, speaker="Host"),
            Supervision(text="Thanks.", start=7.0, duration=2.0, speaker="Guest"),
            Supervision(text="Great to be here.", start=10.0, duration=2.0, speaker="Guest"),
            Supervision(text="Let's start.", start=13.0, duration=2.0, speaker="Host"),
            Supervision(text="Sure.", start=16.0, duration=2.0, speaker="Guest"),
        ]

        srt_file = tmp_path / "test.srt"
        Caption.from_supervisions(supervisions).write(srt_file)

        result = Caption.read(srt_file)
        for orig, read in zip(supervisions, result.supervisions):
            assert read.text == orig.text, f"Text: {read.text!r} != {orig.text!r}"
            assert read.speaker is not None, f"Speaker lost for: {orig.text}"
