import logging
import re
from typing import Optional, Tuple

# Timestamp pattern: [start-end] text
# Example: [1.23-4.56] Hello world
TIMESTAMP_PATTERN = re.compile(r"^\[([\d.]+)-([\d.]+)\]\s*(.*)$")

# Speaker change markers in captions:
# - ">> Name:" or "&gt;&gt; Name:" — speaker with name
# - ">>" or "&gt;&gt;" alone — anonymous speaker change (YouTube auto-captions)
SPEAKER_CHANGE_RE = re.compile(r"^(?:>>|&gt;&gt;)\s*$")
SPEAKER_PATTERN = re.compile(r"((?:>>|&gt;&gt;|>|&gt;).*?[:：])\s*(.*)")

# Transcriber Output Example:
# 26:19.919 --> 26:34.921
# [SPEAKER_01]: 越来越多的科技巨头入...
SPEAKER_LATTIFAI = re.compile(r"(^\[SPEAKER_.*?\][:：])\s*(.*)")

# NISHTHA BHATIA: Hey, everyone.
# DIETER: Oh, hey, Nishtha.
# GEMINI: That might
SPEAKER_PATTERN2 = re.compile(r"^([A-Z]{1,15}(?:\s+[A-Z]{1,15})?[:：])\s*(.*)$")

# Title-case speaker names (e.g. "Dwarkesh Patel:", "Terence Tao:")
# Only matches if the name candidate is in _speaker_candidates to avoid false positives
_speaker_candidates: set = set()


def normalize_text(text: str) -> str:
    """Normalize caption text by:
    - Decoding common HTML entities
    - Collapsing multiple whitespace into a single space
    - Converting curly apostrophes to straight ones in common contractions

    Note: HTML tags (<b>, <i>, <u>, <font>) are intentionally preserved
    to allow roundtrip fidelity for formats that support them (SRT, VTT).
    """
    if not text:
        return ""

    # # Remove HTML tags first (replace with space to avoid concatenation)
    # text = re.sub(r"<[^>]+>", " ", text)

    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        # \N intentionally NOT replaced here — it is ASS-specific line break
        # syntax and must be preserved for bilingual subtitle roundtrip.
        # The ASS reader converts \N via pysubs2 event.plaintext (see P0-2).
        "…": " ",  # replace ellipsis with space to avoid merging words
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # Convert curly apostrophes to straight apostrophes for common English contractions
    text = re.sub(r"([a-zA-Z])’([tsdm]|ll|re|ve)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([0-9])’([s])\b", r"\1'\2", text, flags=re.IGNORECASE)

    # Collapse whitespace (after replacements)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def set_speaker_candidates(candidates: set) -> None:
    """Pre-set known speaker names for title-case matching.

    When candidates are set, ``parse_speaker_text`` will also match lines
    starting with ``Name:`` where *Name* is in the candidate set. This avoids
    false positives on ordinary sentences while supporting mixed-case speaker
    labels like ``Dwarkesh Patel:`` or ``Terence Tao:``.
    """
    global _speaker_candidates
    _speaker_candidates = {c.rstrip(":").rstrip("：") for c in candidates}


def detect_speaker_candidates(lines) -> set:
    """Auto-detect recurring title-case speaker names from caption lines.

    Scans for ``Title Case Name:`` prefixes (1-4 capitalized words + colon).

    Detection signals (any one sufficient):
    1. **Frequency**: any single name ≥3 times → high confidence.
    2. **Dialogue pattern**: ≥2 distinct names + alternation (A→B) +
       recurrence (≥1 name ≥2x) + gaps (unlabeled lines between labeled
       lines). Real speakers have continuation lines between turns;
       consecutive labels (Note/Warning) do not.
    """
    name_colon = re.compile(
        r"^([A-Z][a-zA-Z\u00C0-\u024F'\-]+" r"(?:\s+[A-Z][a-zA-Z\u00C0-\u024F'\-]+){0,3})" r"[:：]\s+"
    )
    counts: dict = {}
    sequence: list = []  # (name, line_index) for gap detection
    line_idx = 0
    for line in lines:
        if isinstance(line, str) and line.strip():
            m = name_colon.match(line)
            if m:
                name = m.group(1)
                counts[name] = counts.get(name, 0) + 1
                sequence.append((name, line_idx))
            line_idx += 1

    if not counts:
        return set()

    # Signal 1: names appearing ≥3 times are high-confidence speakers.
    # Promote adjacent names only if they ALSO recur (≥2x). One-off labels
    # like "Chapter One:" adjacent to a real speaker are NOT promoted.
    confident = {name for name, c in counts.items() if c >= 3}
    if confident:
        names = [s[0] for s in sequence]
        promoted = set(confident)
        for a, b in zip(names, names[1:]):
            if a in confident and b not in confident and counts[b] >= 2:
                promoted.add(b)
            elif b in confident and a not in confident and counts[a] >= 2:
                promoted.add(a)
        return promoted

    # Signal 2: dialogue pattern — requires ≥2 distinct names + structural evidence.
    # Two sub-cases:
    #   a) Sparse labels (gaps between labeled lines): alternation + recurrence + gaps.
    #      Gaps prove speaker turns have continuation lines; labels don't.
    #   b) Fully labeled (every line has a name): every adjacent pair must differ.
    #      Writer output has speaker on every line; labels repeat (Note→Note).
    if len(counts) >= 2:
        names = [s[0] for s in sequence]
        indices = [s[1] for s in sequence]
        has_alternation = any(a != b for a, b in zip(names, names[1:]))
        has_recurrence = any(c >= 2 for c in counts.values())
        has_gaps = any(b - a > 1 for a, b in zip(indices, indices[1:]))
        all_labeled = len(sequence) == line_idx
        all_alternate = len(names) >= 2 and all(a != b for a, b in zip(names, names[1:]))

        if has_alternation and has_recurrence and has_gaps:
            return set(counts.keys())
        if all_labeled and all_alternate and len(names) >= 4:
            return set(counts.keys())

    return set()


def parse_speaker_text(line) -> Tuple[Optional[str], str]:
    """Parse a line of text to extract speaker and content.

    Returns:
        (speaker, text) where speaker is:
        - ">>" for anonymous speaker change markers
        - "Name:" for named speakers
        - None if no speaker detected
    """
    # Anonymous speaker change: bare >> or &gt;&gt; (no colon needed)
    stripped = line.strip()
    if SPEAKER_CHANGE_RE.match(stripped):
        return ">>", ""

    # Also handle >> prefixed to text without colon: ">> some text"
    for prefix in ("&gt;&gt;", ">>"):
        if stripped.startswith(prefix):
            rest = stripped[len(prefix):].strip()
            if rest and ":" not in rest and "：" not in rest:
                return ">>", rest

    if ":" not in line and "：" not in line:
        return None, line

    # Named speaker with >> prefix and colon: ">> Name: text"
    match = SPEAKER_PATTERN.match(line)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = SPEAKER_LATTIFAI.match(line)
    if match:
        if len(match.groups()) != 2:
            raise ValueError(f"Expected 2 groups in SPEAKER_LATTIFAI match, got {match.groups()}")
        if not match.group(1):
            logging.error(f"ParseSub LINE [{line}]")
        else:
            return match.group(1).strip(), match.group(2).strip()

    match = SPEAKER_PATTERN2.match(line)
    if match:
        if len(match.groups()) != 2:
            raise ValueError(f"Expected 2 groups in SPEAKER_PATTERN2 match, got {match.groups()}")
        return match.group(1).strip(), match.group(2).strip()

    # Title-case speaker matching via pre-set or auto-detected candidates
    if _speaker_candidates:
        for sep in (":", "："):
            idx = line.find(sep)
            if idx > 0:
                prefix = line[:idx].strip()
                if prefix in _speaker_candidates:
                    return prefix + sep, line[idx + 1 :].strip()

    return None, line


def parse_timestamp_text(line: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Parse a line of text to extract timestamp and content.

    Format: [start-end] text
    Example: [1.23-4.56] Hello world

    Args:
        line: Input line to parse

    Returns:
        Tuple of (start_time, end_time, text)
        - start_time: Start timestamp in seconds, or None if not found
        - end_time: End timestamp in seconds, or None if not found
        - text: The text content after the timestamp
    """
    match = TIMESTAMP_PATTERN.match(line)
    if match:
        try:
            start = float(match.group(1))
            end = float(match.group(2))
            text = match.group(3).strip()
            return start, end, text
        except ValueError:
            # If conversion fails, treat as plain text
            return None, None, line

    return None, None, line


def cjk_ratio(text: str) -> float:
    """Calculate the ratio of CJK characters in text.

    CJK Unified Ideographs (U+4E00..U+9FFF), CJK Extension A (U+3400..U+4DBF),
    and fullwidth forms are counted. Punctuation and whitespace are excluded
    from the denominator.

    Args:
        text: Input text string.

    Returns:
        Ratio of CJK characters (0.0 to 1.0). Returns 0.0 for empty text.
    """
    if not text:
        return 0.0
    # Count only alphanumeric + CJK chars (skip punctuation/whitespace)
    cjk_count = 0
    char_count = 0
    for ch in text:
        cp = ord(ch)
        is_cjk = (
            0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF     # CJK Extension A
            or 0xF900 <= cp <= 0xFAFF     # CJK Compatibility Ideographs
            or 0x20000 <= cp <= 0x2A6DF   # CJK Extension B
            or 0xFF00 <= cp <= 0xFFEF     # Fullwidth Forms
        )
        is_letter = ch.isalnum() or is_cjk
        if is_letter:
            char_count += 1
            if is_cjk:
                cjk_count += 1
    return cjk_count / char_count if char_count > 0 else 0.0


# =============================================================================
# Filename language detection (P2-3)
# =============================================================================

# Ordered from most specific (bilingual) to least specific (monolingual).
# Each entry: (pattern, language, target_language)
_LANG_PATTERNS = [
    # Bilingual: simplified Chinese + English
    (r"[.\s](?:简体中文&英文|简体&英文|CN&EN|chs&eng)[.\s]", "zh", "en"),
    # Bilingual: traditional Chinese + English
    (r"[.\s](?:繁体&英文|繁體&英文|cht&eng)[.\s]", "zh_tw", "en"),
    # Bilingual: generic
    (r"[.\s](?:双语|bilingual)[.\s]", "zh", "en"),
    # Monolingual: simplified Chinese
    (r"[.\s](?:简体中文|简体|chs|CHS)[.\s]", "zh", None),
    (r"[.\s]CN[.\s]", "zh", None),
    # Monolingual: traditional Chinese
    (r"[.\s](?:繁体|繁體|cht|CHT)[.\s]", "zh_tw", None),
    # Monolingual: English
    (r"[.\s](?:英文|eng)[.\s]", "en", None),
    (r"[.\s]EN[.\s]", "en", None),
]
_LANG_COMPILED = [(re.compile(p, re.IGNORECASE), lang, target) for p, lang, target in _LANG_PATTERNS]


def detect_language_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract language information from subtitle filename patterns.

    Recognizes common Chinese fansub naming conventions like:
    - Show.S01E01.简体中文&英文.ass -> ("zh", "en")
    - Show.CN.srt -> ("zh", None)

    Args:
        filename: Subtitle filename (basename or full path).

    Returns:
        Tuple of (language, target_language). Both None if no pattern matches.
    """
    # Ensure dots at boundaries for matching
    name = "." + filename + "."
    for pattern, lang, target in _LANG_COMPILED:
        if pattern.search(name):
            return lang, target
    return None, None


# Staff credit role keywords (Chinese fansub conventions)
_STAFF_ROLES = re.compile(
    r"^\s*(翻译|校对|时间轴|后期|总监|压制|监制|特效|听译|编辑|审核|"
    r"片源|录制|制作|调轴|打轴)\s+(.+?)\s*$"
)

# Branding / disclaimer keywords
_BRANDING_KEYWORDS = [
    "yyets", "zimuzu", "人人影视", "字幕组", "字幕社",
    "www.", ".com", ".tv", ".net", ".org",
    "原创翻译", "双语字幕", "仅供交流", "禁止商用",
    "仅供学习", "请勿用于商业",
]


def classify_line_type(text: str, start: float = 0.0) -> Optional[str]:
    """Classify a subtitle line as staff credit, branding, or normal dialogue.

    Staff credits and branding typically appear in the first 120 seconds
    and are short (< 50 chars).

    Args:
        text: Subtitle text content.
        start: Start time in seconds (staff credits are early in the file).

    Returns:
        "staff_credit", "branding", or None (normal dialogue).
    """
    if not text or start > 120.0 or len(text) > 50:
        return None

    stripped = text.strip()

    # Staff credit: role + name pattern
    if _STAFF_ROLES.match(stripped):
        return "staff_credit"

    # Branding / disclaimer
    lower = stripped.lower()
    for kw in _BRANDING_KEYWORDS:
        if kw in lower:
            return "branding"

    return None


if __name__ == "__main__":
    pattern = re.compile(r">>\s*(.*?)\s*[:：]\s*(.*)")
    pattern = re.compile(r"(>>.*?[:：])\s*(.*)")

    test_strings = [
        ">>Key: Value",
        ">>  Key with space : Value with space ",
        ">>  全角键 ： 全角值",
        ">>Key：Value xxx. >>Key：Value",
    ]

    for text in test_strings:
        match = pattern.match(text)
        if match:
            print(f"Input: '{text}'")
            print(f"Speaker:   '{match.group(1)}'")
            print(f"Content: '{match.group(2)}'")
            print("-------------")

    # pattern2
    test_strings2 = ["NISHTHA BHATIA: Hey, everyone.", "DIETER: Oh, hey, Nishtha.", "GEMINI: That might"]
    for text in test_strings2:
        match = SPEAKER_PATTERN2.match(text)
        if match:
            print(f"  Input: '{text}'")
            print(f"Speaker: '{match.group(1)}'")
            print(f"Content: '{match.group(2)}'")
            print("-------------")
        else:
            raise ValueError(f"No match for: '{text}'")
