import re
from typing import List, Optional, Tuple

from .punctuation import END_PUNCTUATION
from .supervision import Supervision, fastcopy
from .utils import _resolve_model_path

# Pre-compiled regex patterns for performance
_SPECIAL_PATTERN_1 = re.compile(r"^(\[[^\]]+\])\s+(&gt;&gt;|>>)\s+(.+)$")
_SPECIAL_PATTERN_2 = re.compile(r"^(\[[^\]]+\])\s+([^:]+:)(.*)$")
_MULTI_EVENT_RE = re.compile(r"\[[^\]]+\]")
_TRAILING_EVENTS_RE = re.compile(r"^(.+?)\s+(\[[^\]]+\](?:\s+\[[^\]]+\])*)$")
_LEADING_EVENTS_RE = re.compile(r"^(\[[^\]]+\](?:\s+\[[^\]]+\])*)\s+(.+)$")

# Leading run of ASS override-tag blocks: ``{\an8}{\fad(0,500)}…``.
# After a split we keep only this prefix on ``ass_raw_text`` so positioning /
# fade tags propagate to each fragment via the ASS writer's tag-prefix
# fallback, while the (now-stale) text body is dropped.
_ASS_TAG_PREFIX_RE = re.compile(r"^(\{[^}]*\})+")


class SentenceSplitter:
    """Lazy-initialized sentence splitter using wtpsplit."""

    def __init__(
        self,
        device: str = "cpu",
        model_hub: Optional[str] = "modelscope",
        lazy_init: bool = True,
    ):
        """Initialize sentence splitter with lazy loading.

        Args:
            device: Device to run the model on (cpu, cuda, mps)
            model_hub: Model hub to use (None for huggingface, "modelscope" for modelscope)
        """
        self.device = device
        self.model_hub = model_hub
        self._splitter = None
        if not lazy_init:
            self._init_splitter()

    def _init_splitter(self) -> None:
        """Initialize the sentence splitter model on first use."""
        if self._splitter is not None:
            return

        import onnxruntime as ort
        from wtpsplit import SaT

        providers = []
        device = self.device
        if (
            device.startswith("cuda")
            and ort.get_all_providers().count("CUDAExecutionProvider") > 0
        ):
            providers.append("CUDAExecutionProvider")
        elif (
            device.startswith("mps")
            and ort.get_all_providers().count("MPSExecutionProvider") > 0
        ):
            providers.append("MPSExecutionProvider")

        if self.model_hub == "modelscope":
            downloaded_path = _resolve_model_path(
                "LattifAI/OmniTokenizer", model_hub="modelscope"
            )
            sat = SaT(
                f"{downloaded_path}/sat-3l-sm",
                tokenizer_name_or_path=f"{downloaded_path}/xlm-roberta-base",
                ort_providers=providers + ["CPUExecutionProvider"],
            )
        else:
            sat_path = _resolve_model_path(
                "segment-any-text/sat-3l-sm", model_hub="huggingface"
            )
            sat = SaT(
                sat_path,
                tokenizer_name_or_path="facebookAI/xlm-roberta-base",
                hub_prefix="segment-any-text",
                ort_providers=providers + ["CPUExecutionProvider"],
            )
        # Fix for transformers 5.x: minimal tokenizer_config.json may lose
        # special token mappings. Patch them back for xlm-roberta-base and
        # refresh the cached special_tokens list that SaT.__init__ snapshots.
        tok = sat.tokenizer
        if tok.cls_token_id is None or tok.sep_token_id is None:
            tok.add_special_tokens(
                {
                    "cls_token": "<s>",
                    "sep_token": "</s>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                }
            )
            sat.special_tokens = [tok.cls_token, tok.sep_token, tok.pad_token]

        self._splitter = sat

    @staticmethod
    def _distribute_time_info(
        input_supervisions: List[Supervision],
        split_texts: List[str],
    ) -> List[Supervision]:
        """Distribute time information from input supervisions to split sentences.

        Args:
            input_supervisions: Original supervisions with time information
            split_texts: List of split sentence texts

        Returns:
            List of Supervision objects with distributed time information.
            Custom attributes are inherited from first_sup with conflict markers.
        """
        if not input_supervisions:
            return [
                Supervision(text=text, id="", recording_id="", start=0, duration=0)
                for text in split_texts
            ]

        input_text = " ".join(sup.text for sup in input_supervisions)

        # Pre-compute supervision position mapping: (start_pos, end_pos, supervision)
        sup_ranges = []
        char_pos = 0
        for sup in input_supervisions:
            sup_start = char_pos
            sup_end = char_pos + len(sup.text)
            sup_ranges.append((sup_start, sup_end, sup))
            char_pos = sup_end + 1  # +1 for space separator

        result = []
        search_start = 0
        sup_idx = 0

        for split_text in split_texts:
            text_start = input_text.find(split_text, search_start)
            if text_start == -1:
                raise ValueError(
                    f"Could not find split text '{split_text}' in input supervisions."
                )

            text_end = text_start + len(split_text)
            search_start = text_end

            first_sup = None
            last_sup = None
            first_char_idx = None
            last_char_idx = None
            overlapping_customs = []
            sliced_word_items: List = []

            for i in range(sup_idx, len(sup_ranges)):
                sup_start, sup_end, sup = sup_ranges[i]

                if sup_end <= text_start:
                    sup_idx = i + 1
                    continue
                if sup_start >= text_end:
                    break

                if first_sup is None:
                    first_sup = sup
                    first_char_idx = max(0, text_start - sup_start)

                last_sup = sup
                last_char_idx = min(len(sup.text) - 1, text_end - 1 - sup_start)

                if getattr(sup, "custom", None):
                    overlapping_customs.append(sup.custom)

                # Slice this source's word-level alignment by the global text
                # range, then collect the AlignmentItems whose symbol falls
                # inside [local_start, local_end). Walk sup.text + words in
                # order with a cursor (per the multilingual-text rule in
                # CLAUDE.md) so CJK char-level symbols and Latin word symbols
                # both map back to the right substring.
                sup_alignment = getattr(sup, "alignment", None)
                if sup_alignment and sup_alignment.get("word"):
                    local_start = max(0, text_start - sup_start)
                    local_end = min(len(sup.text or ""), text_end - sup_start)
                    src_text = sup.text or ""
                    cursor = 0
                    for w in sup_alignment["word"]:
                        if not w.symbol:
                            continue
                        pos = src_text.find(w.symbol, cursor)
                        if pos < 0:
                            continue
                        word_end = pos + len(w.symbol)
                        if word_end > local_start and pos < local_end:
                            sliced_word_items.append(w)
                        cursor = word_end

            if first_sup is None or last_sup is None:
                raise ValueError(
                    f"Could not find supervisions for split text: {split_text}"
                )

            if len(first_sup.text) == 0 or len(last_sup.text) == 0:
                start_time = first_sup.start
                end_time = last_sup.start + last_sup.duration
            else:
                start_time = (
                    first_sup.start
                    + (first_char_idx / len(first_sup.text)) * first_sup.duration
                )
                end_time = (
                    last_sup.start
                    + ((last_char_idx + 1) / len(last_sup.text)) * last_sup.duration
                )

            # Guard against negative duration from overlapping source supervisions
            # (e.g., YouTube scroll-style VTT where cues overlap in time).
            # When the interpolated end precedes start, fall back to the
            # last supervision's end boundary.
            if end_time < start_time:
                end_time = max(last_sup.start + last_sup.duration, start_time + 0.1)

            merged_custom = SentenceSplitter._merge_customs(overlapping_customs)
            merged_custom = SentenceSplitter._sanitize_custom_for_split(merged_custom)

            sliced_alignment = (
                {"word": sliced_word_items} if sliced_word_items else None
            )

            # Inherit all per-supervision fields (channel / language /
            # gender / score / translation / target_lang / recording_id /
            # …) from the first overlapping source so per-line caption
            # metadata (e.g. ASS Style, VTT voice) survives the split.
            # ``text`` / ``start`` / ``duration`` / ``id`` / ``custom`` /
            # ``alignment`` are freshly assigned; ``speaker`` is deferred
            # to the Phase 5 assignment step, which owns the "only first
            # sentence carries the speaker" semantics.
            result.append(
                fastcopy(
                    first_sup,
                    id="",
                    text=split_text,
                    start=start_time,
                    duration=end_time - start_time,
                    custom=merged_custom,
                    alignment=sliced_alignment,
                    speaker=None,
                )
            )

        return result

    @staticmethod
    def _merge_customs(customs: List[Optional[dict]]) -> Optional[dict]:
        """Union-merge ``custom`` dicts from overlapping source supervisions.

        - Zero non-empty entries → ``None``.
        - One non-empty entry → shallow copy.
        - Multiple entries → union by key. First source wins on conflicts;
          unique keys from later sources are preserved so no per-line
          metadata is silently dropped. When any conflict exists the result
          is tagged with ``_split_from_multiple`` / ``_source_count`` so
          callers can detect mixed-source segments.
        """
        non_empty = [c for c in customs if c]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return dict(non_empty[0])
        merged = dict(non_empty[0])
        has_conflict = False
        for other in non_empty[1:]:
            for key, value in other.items():
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    has_conflict = True
        if has_conflict:
            merged["_split_from_multiple"] = True
            merged["_source_count"] = len(non_empty)
        return merged

    @staticmethod
    def _sanitize_custom_for_split(
        custom: Optional[dict],
    ) -> Optional[dict]:
        """Strip ASS roundtrip artifacts that can't survive a split intact.

        - ``ass_raw_event_body`` / ``ass_raw_event_type``: the raw Dialogue
          line refers to the *un-split* text. Letting the splicer reuse it
          would force every fragment to re-emit the original full
          Text/Margins/Effect/etc., undoing the split. Drop both so
          ``_splice_raw_event_bodies`` skips these events and pysubs2's
          per-fragment output becomes authoritative.
        - ``ass_raw_text``: trim down to the leading override-tag block
          (``{\\an8}{\\fad(0,500)}…``) so positioning / animation tags
          propagate to each fragment via the writer's tag-prefix fallback.
          The text body wouldn't survive a split byte-faithfully and
          would otherwise trigger spurious raw-text restoration when a
          fragment happens to ``_norm``-equal the full original.
        """
        if not custom:
            return custom
        new_custom = dict(custom)
        new_custom.pop("ass_raw_event_body", None)
        new_custom.pop("ass_raw_event_type", None)
        raw_text = new_custom.get("ass_raw_text")
        if raw_text is not None:
            m = _ASS_TAG_PREFIX_RE.match(raw_text)
            prefix = m.group(0) if m else ""
            if prefix:
                new_custom["ass_raw_text"] = prefix
            else:
                new_custom.pop("ass_raw_text", None)
        return new_custom

    @staticmethod
    def _is_event_text(text: str) -> bool:
        """Check if text is an event marker like [Music], [Applause], etc."""
        text = text.strip()
        return text.startswith("[") and text.endswith("]")

    @staticmethod
    def _split_event_text(text: str) -> List[str]:
        """Split multi-event text '[Laughter] [Applause]' into individual events.

        Only splits if the text is entirely composed of bracketed items separated
        by whitespace (e.g., '[A] [B]' but NOT '[A] some text [B]').
        """
        stripped = text.strip()
        events = _MULTI_EVENT_RE.findall(stripped)
        if len(events) > 1 and " ".join(events) == stripped:
            return events
        return [text]

    @staticmethod
    def _split_inline_events(text: str) -> List[str]:
        """Split trailing/leading event markers from normal text.

        Non-speech events (``[Laughter]``, ``[Applause]``, ``[Music]``, …) are
        not pronounceable text and must live in their own supervision so TTS
        skips them and translation handles them as discrete units. They are
        split out whenever they sit at the start or end of a segment,
        independent of whether the spoken side ends with sentence punctuation
        — speakers often trail off into laughter without a period.

        Inline events embedded mid-sentence (e.g. ``"he said [INAUDIBLE]
        something"``) are preserved as-is: the surrounding speech is one
        continuous utterance, and splitting at the event would fragment a
        single semantic clause.

        Examples:
            "And breathe out. [Breathes out]" -> ["And breathe out.", "[Breathes out]"]
            "It's nice for [laughter]"        -> ["It's nice for", "[laughter]"]
            "[Breathes in] And breathe in."   -> ["[Breathes in]", "And breathe in."]
            "he said [INAUDIBLE] something"   -> ["he said [INAUDIBLE] something"]
        """
        text = text.strip()
        if not text:
            return [text]

        # Check trailing events: "text [Event1] [Event2]"
        # Always split — the event tokens themselves are not speech and
        # belong in their own supervision regardless of preceding punctuation.
        match = _TRAILING_EVENTS_RE.match(text)
        if match:
            prefix = match.group(1).strip()
            events_str = match.group(2)
            if prefix:
                events = _MULTI_EVENT_RE.findall(events_str)
                return [prefix] + events

        # Check leading events: "[Event1] [Event2] text"
        match = _LEADING_EVENTS_RE.match(text)
        if match:
            events_str = match.group(1)
            suffix = match.group(2).strip()
            if suffix:
                events = _MULTI_EVENT_RE.findall(events_str)
                return events + [suffix]

        return [text]

    @staticmethod
    def _resplit_special_sentence_types(sentence: str) -> List[str]:
        """Re-split [EVENT] >> SPEAKER: patterns into separate parts."""
        sentence = sentence.strip()

        if match := _SPECIAL_PATTERN_1.match(sentence):
            return [match.group(1), f"{match.group(2)} {match.group(3)}"]

        if match := _SPECIAL_PATTERN_2.match(sentence):
            remaining = match.group(3).strip()
            speaker_part = (
                f"{match.group(2)} {remaining}" if remaining else match.group(2)
            )
            return [match.group(1), speaker_part]

        return [sentence]

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        return any(text.endswith(p) for p in END_PUNCTUATION)

    @staticmethod
    def _ends_with_colon(text: str) -> bool:
        """Check if text ends with a colon (speaker introduction)."""
        return text.endswith(":") or text.endswith("：")

    @staticmethod
    def _is_cjk_char(ch: str) -> bool:
        """Check if a character is CJK (Chinese, Japanese, Korean)."""
        cp = ord(ch)
        return (
            0x4E00 <= cp <= 0x9FFF
            or 0x3040 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7AF
            or 0x3400 <= cp <= 0x4DBF
        )

    @staticmethod
    def _needs_space_after(text: str) -> bool:
        """Check if space is needed after colon-ending text (False for CJK context)."""
        stripped = text.rstrip(":：")
        if stripped and SentenceSplitter._is_cjk_char(stripped[-1]):
            return False
        return True

    def _segment_supervisions(
        self, supervisions: List[Supervision]
    ) -> Tuple[List[str], List[Optional[str]], set]:
        """Segment supervisions into text chunks with speakers.

        Returns:
            texts: List of concatenated text segments
            speakers: List of speakers for each segment (parallel to texts)
            event_indices: Set of indices that are event segments
        """
        texts: List[str] = []
        speakers: List[Optional[str]] = []
        event_indices: set = set()

        segment_start = 0
        accumulated_len = 0

        def flush_segment(end_idx: int, speaker: Optional[str]) -> None:
            nonlocal segment_start, accumulated_len
            if segment_start <= end_idx:
                if len(speakers) < len(texts) + 1:
                    speakers.append(speaker)
                text = " ".join(
                    sup.text for sup in supervisions[segment_start : end_idx + 1]
                )
                texts.append(text)
                segment_start = end_idx + 1
                accumulated_len = 0

        for idx, sup in enumerate(supervisions):
            is_last = idx == len(supervisions) - 1
            is_event = self._is_event_text(sup.text)

            if is_event:
                if segment_start < idx:
                    flush_segment(idx - 1, None)
                # Split multi-event text like "[Laughter] [Applause]" into individual events
                sub_events = self._split_event_text(sup.text)
                if len(sub_events) > 1:
                    # Don't flush as a single segment; emit each sub-event individually
                    segment_start = idx + 1
                    accumulated_len = 0
                    for sub_event in sub_events:
                        speakers.append(sup.speaker)
                        texts.append(sub_event)
                        event_indices.add(len(texts) - 1)
                else:
                    flush_segment(idx, sup.speaker)
                    event_indices.add(len(texts) - 1)
                    accumulated_len = 0
                continue

            accumulated_len += len(sup.text)

            if sup.speaker:
                if segment_start < idx:
                    flush_segment(idx - 1, None)
                    accumulated_len = len(sup.text)

                next_has_speaker = not is_last and supervisions[idx + 1].speaker
                if is_last or next_has_speaker:
                    flush_segment(idx, sup.speaker)
                else:
                    speakers.append(sup.speaker)

            elif accumulated_len >= 2000 or is_last:
                flush_segment(idx, None)

        return texts, speakers, event_indices

    def _apply_sentence_splitting(
        self,
        texts: List[str],
        event_indices: set,
        strip_whitespace: bool,
        threshold: float,
    ) -> List[List[str]]:
        """Apply wtpsplit to non-event texts.

        Returns list of split sentence lists, one per input text.
        """
        texts_to_split = [t for i, t in enumerate(texts) if i not in event_indices]

        if texts_to_split:
            split_results = list(
                self._splitter.split(
                    texts_to_split,
                    threshold=threshold,
                    strip_whitespace=strip_whitespace,
                    batch_size=8,
                )
            )
        else:
            split_results = []

        # Reconstruct with events preserved as single-item lists
        sentences = []
        split_idx = 0
        for i in range(len(texts)):
            if i in event_indices:
                sentences.append([texts[i]])
            else:
                # Post-process: split inline events from each sentence
                raw = split_results[split_idx]
                expanded = []
                for sent in raw:
                    expanded.extend(self._split_inline_events(sent))
                sentences.append(expanded)
                split_idx += 1

        return sentences

    def _process_special_patterns(
        self, sentences: List[str], segment_text: str = None
    ) -> Tuple[List[str], str]:
        """Process sentences to handle special patterns like [EVENT] >> SPEAKER:.

        Args:
            sentences: List of split sentences to process
            segment_text: Original segment text (space-joined supervisions) used to
                determine the correct separator when merging colon-ending text with
                the next sentence.

        Returns:
            processed: List of processed sentences
            remainder: Any trailing content that needs to be carried forward
        """
        processed: List[str] = []
        remainder = ""

        for idx, sentence in enumerate(sentences):
            if remainder:
                sentence = remainder + sentence
                remainder = ""

            parts = self._resplit_special_sentence_types(sentence)

            if self._ends_with_colon(parts[-1]):
                # Determine separator from original text context if available
                if segment_text:
                    pos = segment_text.find(parts[-1])
                    if pos != -1:
                        after_pos = pos + len(parts[-1])
                        sep = (
                            " "
                            if after_pos < len(segment_text)
                            and segment_text[after_pos] == " "
                            else ""
                        )
                    else:
                        sep = " " if self._needs_space_after(parts[-1]) else ""
                else:
                    sep = " " if self._needs_space_after(parts[-1]) else ""
                if idx < len(sentences) - 1:
                    sentences[idx + 1] = parts[-1] + sep + sentences[idx + 1]
                else:
                    remainder = parts[-1] + sep
                processed.extend(parts[:-1])
            else:
                processed.extend(parts)

        return processed, remainder

    def _process_sentences_with_speakers(
        self,
        speakers: List[Optional[str]],
        sentences: List[List[str]],
        event_indices: set,
        texts: List[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """Process split sentences with speaker tracking and remainder handling.

        Returns list of (text, speaker) tuples.
        """
        results: List[Tuple[str, Optional[str]]] = []
        remainder = ""
        remainder_speaker: Optional[str] = None

        for seg_idx, (speaker, seg_sentences) in enumerate(zip(speakers, sentences)):
            is_event = seg_idx in event_indices

            # Handle remainder from previous segment
            if remainder:
                if is_event:
                    results.append((remainder.strip(), remainder_speaker))
                elif seg_sentences:
                    seg_sentences[0] = remainder + seg_sentences[0]
                    speaker = remainder_speaker or speaker
                remainder = ""
                remainder_speaker = None

            if not seg_sentences:
                continue

            # Event segments: emit as-is
            if is_event:
                results.append((seg_sentences[0], speaker))
                continue

            # Process special patterns
            segment_text = texts[seg_idx] if texts else None
            processed, pattern_remainder = self._process_special_patterns(
                seg_sentences, segment_text
            )
            if pattern_remainder:
                remainder = pattern_remainder

            if not processed:
                if remainder:
                    processed = [remainder.strip()]
                    remainder = ""
                else:
                    continue

            # Emit sentences based on whether last one is complete
            if self._ends_with_punctuation(processed[-1]):
                for i, text in enumerate(processed):
                    results.append((text, speaker if i == 0 else None))
                speaker = None
            else:
                for i, text in enumerate(processed[:-1]):
                    results.append((text, speaker if i == 0 else None))

                remainder = processed[-1] + " " + remainder

                next_has_speaker = (
                    seg_idx < len(speakers) - 1 and speakers[seg_idx + 1] is not None
                )
                if next_has_speaker:
                    emit_speaker = speaker if len(processed) == 1 else None
                    results.append((remainder.strip(), emit_speaker))
                    remainder = ""
                    remainder_speaker = None
                elif len(processed) == 1:
                    remainder_speaker = speaker
                    if seg_idx < len(speakers) - 1 and speakers[seg_idx + 1] is None:
                        speakers[seg_idx + 1] = speaker
                elif len(processed) > 1:
                    speaker = None
                    remainder_speaker = None

        if remainder.strip():
            results.append((remainder.strip(), remainder_speaker))

        return results

    def split_sentences(
        self,
        supervisions: List[Supervision],
        strip_whitespace: bool = True,
        threshold: float = 0.35,
    ) -> List[Supervision]:
        """Split supervisions into sentences using the sentence splitter.

        Preserves speaker information across segment boundaries and keeps
        event supervisions (e.g., [Music], [Applause]) as separate segments.

        Args:
            supervisions: List of Supervision objects to split
            strip_whitespace: Whether to strip whitespace from split sentences
            threshold: wtpsplit segmentation threshold. Higher values yield
                more conservative segmentation (fewer splits); protects short
                proper-noun phrases like "Terry Tao" from being torn apart on
                sparse-context fragments.

                Default 0.35 is the empirically validated sweet spot across
                ASR output, rolling-caption YouTube subtitles, podcast
                transcripts, interviews, and lectures: it preserves long-
                segment subdivision while avoiding the junk cuts that
                lower thresholds produce around fillers ("um", "you know")
                and self-repairs. Use 0.20 only for dense monologues where
                0.35 leaves overly long supervisions; values <= 0.15 tend
                to fragment fillers into sub-second standalone cues.

        Returns:
            List of Supervision objects with split sentences
        """
        self._init_splitter()

        # Phase 1: Segment supervisions by speaker/event boundaries
        texts, speakers, event_indices = self._segment_supervisions(supervisions)

        if len(speakers) != len(texts):
            raise ValueError(
                f"len(speakers)={len(speakers)} != len(texts)={len(texts)}"
            )

        # Phase 2: Apply sentence splitting to non-event segments
        sentences = self._apply_sentence_splitting(
            texts, event_indices, strip_whitespace, threshold
        )

        # Phase 3: Process sentences with speaker tracking
        texts_with_speakers = self._process_sentences_with_speakers(
            speakers, sentences, event_indices, texts
        )

        # Phase 4: Distribute time information
        split_texts = [text for text, _ in texts_with_speakers]
        result_supervisions = self._distribute_time_info(supervisions, split_texts)

        # Phase 5: Assign speaker information
        for sup, (_, speaker) in zip(result_supervisions, texts_with_speakers):
            if speaker:
                sup.speaker = speaker

        return result_supervisions
