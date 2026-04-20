"""Language detection helper.

Two layers of detection are exposed:

1. **Script (coarse)** — ``detect_script()`` uses Unicode block counting to
   map text to one of ``latin`` / ``east_asian`` / ``cyrillic`` / ``arabic``
   / ``devanagari`` / ``greek`` / ``thai`` / ``hebrew``. Deterministic, zero
   dependency, always works (even on 2-char strings).

2. **Language (precise)** — ``detect_language()`` delegates to ``lingua`` in
   low-accuracy mode. Returns an ISO 639-1 code on longer strings; may
   return ``None`` on short Latin strings where lingua cannot disambiguate
   e.g. English vs Italian.

The primary consumer — ``is_distinct_language_pair()`` — decides whether
two ``\\n``-separated halves of a caption cue represent two different
languages. It compares **scripts first** (handles CJK↔Latin robustly) and
falls back to ISO language only when both sides share a script.

Default candidate language set covers the 10 languages subtitle groups
most often translate between: English, Chinese, Japanese, Korean, Spanish,
French, German, Russian, Portuguese, Italian. Memory ≈ 17 MB.
"""

import re
from threading import Lock
from typing import Optional, Sequence

# Default candidate set: covers 99% of subtitle-group source languages.
# Each additional language costs roughly 2-3 MB of resident memory.
_DEFAULT_CANDIDATE_LANGS = (
    "en",  # English
    "zh",  # Chinese
    "ja",  # Japanese
    "ko",  # Korean
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "ru",  # Russian
    "pt",  # Portuguese
    "it",  # Italian
)

# Script-level unicode block ranges. Order matters for ``detect_script``:
# earlier entries take priority when a text has chars from multiple scripts
# (we return whichever has the largest char count).
_SCRIPT_RANGES = {
    # Latin (ASCII + Latin-1 Supplement + Extended-A/B, accented letters)
    "latin": r"A-Za-z\u00c0-\u024f",
    # CJK ideographs + kana + Hangul, collapsed under one bucket for the
    # purposes of bilingual-cue splitting. Use ``detect_language`` for
    # zh/ja/ko disambiguation.
    "east_asian": r"\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af",
    "cyrillic": r"\u0400-\u04ff",
    "greek": r"\u0370-\u03ff",
    "arabic": r"\u0600-\u06ff",
    "devanagari": r"\u0900-\u097f",
    "thai": r"\u0e00-\u0e7f",
    "hebrew": r"\u0590-\u05ff",
}

# Pre-compile a counter per script.
_SCRIPT_COUNTERS = {
    name: re.compile(f"[{rng}]") for name, rng in _SCRIPT_RANGES.items()
}

# Matches text that has no language-bearing characters at all (digits/punct
# /space only). lingua returns garbage on such inputs; short-circuit to None.
_HAS_LETTER_RE = re.compile(
    f"[{''.join(_SCRIPT_RANGES.values())}]"
)

_detector = None
_detector_langs: tuple = ()
_lock = Lock()


def _build_detector(candidate_langs: Sequence[str]):
    """Build a lingua LanguageDetector restricted to the given ISO codes.

    Runs in low-accuracy mode: fewer n-gram resources loaded, accuracy loss
    is negligible on caption-length strings and saves ~60% memory.
    """
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        return None

    # Map ISO 639-1 → lingua Language. Kept small; extend as needed.
    iso_to_lang = {
        "en": Language.ENGLISH,
        "zh": Language.CHINESE,
        "ja": Language.JAPANESE,
        "ko": Language.KOREAN,
        "es": Language.SPANISH,
        "fr": Language.FRENCH,
        "de": Language.GERMAN,
        "ru": Language.RUSSIAN,
        "pt": Language.PORTUGUESE,
        "it": Language.ITALIAN,
        "ar": Language.ARABIC,
        "hi": Language.HINDI,
        "th": Language.THAI,
        "vi": Language.VIETNAMESE,
        "tr": Language.TURKISH,
        "nl": Language.DUTCH,
        "pl": Language.POLISH,
        "sv": Language.SWEDISH,
    }
    langs = [iso_to_lang[c] for c in candidate_langs if c in iso_to_lang]
    if len(langs) < 2:  # lingua requires ≥2 languages
        return None
    return (
        LanguageDetectorBuilder.from_languages(*langs)
        .with_low_accuracy_mode()
        .build()
    )


def set_candidate_languages(langs: Sequence[str]) -> None:
    """Reconfigure the detector for a different candidate set.

    Rebuilds the lingua detector on next detection call. Thread-safe.
    """
    global _detector, _detector_langs
    with _lock:
        _detector_langs = tuple(langs)
        _detector = None  # trigger lazy rebuild


def _get_detector():
    global _detector, _detector_langs
    if _detector is not None:
        return _detector
    with _lock:
        if _detector is None:
            langs = _detector_langs or _DEFAULT_CANDIDATE_LANGS
            _detector_langs = tuple(langs)
            _detector = _build_detector(langs)
    return _detector


def detect_language(
    text: str, candidate_langs: Optional[Sequence[str]] = None
) -> Optional[str]:
    """Detect the ISO 639-1 language code of ``text``.

    Returns ``None`` when:
      * text is empty / whitespace
      * text contains no language-bearing characters (digits/punct only)
      * lingua is not installed or cannot decide

    ``candidate_langs`` overrides the module-level default for this call.
    To change the permanent default use ``set_candidate_languages``.
    """
    if not text or not text.strip():
        return None
    if not _HAS_LETTER_RE.search(text):
        return None

    if candidate_langs is not None:
        # Per-call override: build a one-shot detector.
        detector = _build_detector(list(candidate_langs))
    else:
        detector = _get_detector()
    if detector is None:
        return None

    result = detector.detect_language_of(text)
    if result is None:
        return None
    return result.iso_code_639_1.name.lower()


def detect_script(text: str) -> Optional[str]:
    """Return the dominant Unicode script of ``text``.

    One of: ``latin`` / ``east_asian`` / ``cyrillic`` / ``greek`` / ``arabic``
    / ``devanagari`` / ``thai`` / ``hebrew``, or ``None`` when no letter-class
    character is present.

    Deterministic and zero-dependency. Accurate on 1-char strings (returns
    whichever script the character belongs to). The primary gate used by
    ``is_distinct_language_pair`` — lingua is only consulted when both sides
    share a script.
    """
    if not text:
        return None
    counts = {
        name: len(counter.findall(text))
        for name, counter in _SCRIPT_COUNTERS.items()
    }
    best_name, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count == 0:
        return None
    return best_name


def is_distinct_language_pair(
    side_a: str, side_b: str, candidate_langs: Optional[Sequence[str]] = None
) -> bool:
    """Return True when the two sides represent distinct languages.

    Decision ladder (first condition to match wins):

    1. Either side is empty / has no letters → ``False`` (nothing to compare).
    2. Sides have **different dominant scripts** (e.g., EA vs Latin) →
       ``True``. This handles 99% of字幕组 bilingual cues with zero
       dependency on lingua and works on 1-char strings like ``"OK"``.
    3. Sides share a script; consult lingua for ISO-level language:
       * both ISO codes returned and differ → ``True``
       * otherwise → ``False`` (can't prove distinct → assume soft-wrap)
    """
    if not side_a or not side_a.strip():
        return False
    if not side_b or not side_b.strip():
        return False
    script_a = detect_script(side_a)
    script_b = detect_script(side_b)
    if script_a is None or script_b is None:
        return False
    if script_a != script_b:
        return True
    # Same script: consult lingua for finer-grained distinction.
    lang_a = detect_language(side_a, candidate_langs=candidate_langs)
    lang_b = detect_language(side_b, candidate_langs=candidate_langs)
    if lang_a is None or lang_b is None:
        return False
    return lang_a != lang_b
