"""Benchmark lingua-py language detection on FLORES-200 devtest.

Purpose
-------
Measure per-language accuracy of ``detect_language()`` (wrapper around
``lingua``) at subtitle-like sentence granularity, so we can:

* decide which languages to include in the default candidate set;
* document per-language recall/precision for users;
* surface confusion pairs that need extra heuristics (e.g. zh ↔ ja).

Dataset
-------
`FLORES-200 devtest` — 1012 parallel sentences per language, hand-curated
by Meta AI, sentences are of natural subtitle length (mean ~25 words).
Loaded via HuggingFace ``datasets`` (``Muennighoff/flores200`` mirror).

Usage
-----
::

    python scripts/benchmark_language_detector.py              # all 75 lingua langs
    python scripts/benchmark_language_detector.py --max 200    # 200 sentences/lang
    python scripts/benchmark_language_detector.py --langs en,zh,ja

Output
------
``scripts/benchmark_results/lingua_flores.md`` (Markdown report).
"""

import argparse
import os
import re
import resource
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Public FLORES-200 tarball (~25 MB, no auth required).
FLORES_TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
_DEFAULT_CACHE = Path(
    os.environ.get("LATTIFAI_CACHE", Path.home() / ".cache" / "lattifai")
)


# ---------------------------------------------------------------------------
# FLORES-200 → lingua ISO 639-1 mapping
# ---------------------------------------------------------------------------
# FLORES-200 uses ``<iso639-3>_<script>``; lingua exposes ISO 639-1. Only
# include a FLORES code when (a) lingua supports the language and (b) there
# is an unambiguous 1:1 ISO 639-1 target. Where FLORES has multiple scripts
# (Chinese, Azerbaijani) we pick the most common one.
#
# Coverage: 72 / 75 lingua languages. Omitted:
# - LATIN (la):     no FLORES devtest target
# - ESPERANTO (eo): no FLORES devtest target
# - BOKMAL (nb):    FLORES has ``nob_Latn`` but clashes with Nynorsk ``nno_Latn``
#                   under a single ISO 639-1 ``no``. Include only Bokmal.
FLORES_TO_ISO: Dict[str, str] = {
    "afr_Latn": "af",   # Afrikaans
    "als_Latn": "sq",   # Albanian (Tosk)
    "arb_Arab": "ar",   # Arabic (Standard)
    "hye_Armn": "hy",   # Armenian
    "azj_Latn": "az",   # Azerbaijani (North, Latin)
    "eus_Latn": "eu",   # Basque
    "bel_Cyrl": "be",   # Belarusian
    "ben_Beng": "bn",   # Bengali
    "nob_Latn": "nb",   # Norwegian Bokmål
    "bos_Latn": "bs",   # Bosnian
    "bul_Cyrl": "bg",   # Bulgarian
    "cat_Latn": "ca",   # Catalan
    "zho_Hans": "zh",   # Chinese (Simplified) — single Chinese class in lingua
    "hrv_Latn": "hr",   # Croatian
    "ces_Latn": "cs",   # Czech
    "dan_Latn": "da",   # Danish
    "nld_Latn": "nl",   # Dutch
    "eng_Latn": "en",   # English
    "est_Latn": "et",   # Estonian
    "fin_Latn": "fi",   # Finnish
    "fra_Latn": "fr",   # French
    "lug_Latn": "lg",   # Ganda
    "kat_Geor": "ka",   # Georgian
    "deu_Latn": "de",   # German
    "ell_Grek": "el",   # Greek
    "guj_Gujr": "gu",   # Gujarati
    "heb_Hebr": "he",   # Hebrew
    "hin_Deva": "hi",   # Hindi
    "hun_Latn": "hu",   # Hungarian
    "isl_Latn": "is",   # Icelandic
    "ind_Latn": "id",   # Indonesian
    "gle_Latn": "ga",   # Irish
    "ita_Latn": "it",   # Italian
    "jpn_Jpan": "ja",   # Japanese
    "kaz_Cyrl": "kk",   # Kazakh
    "kor_Hang": "ko",   # Korean
    "lvs_Latn": "lv",   # Latvian (Standard)
    "lit_Latn": "lt",   # Lithuanian
    "mkd_Cyrl": "mk",   # Macedonian
    "zsm_Latn": "ms",   # Malay (Standard)
    "mri_Latn": "mi",   # Maori
    "mar_Deva": "mr",   # Marathi
    "khk_Cyrl": "mn",   # Mongolian (Halh)
    "nno_Latn": "nn",   # Norwegian Nynorsk
    "pes_Arab": "fa",   # Persian (Iranian)
    "pol_Latn": "pl",   # Polish
    "por_Latn": "pt",   # Portuguese
    "pan_Guru": "pa",   # Punjabi
    "ron_Latn": "ro",   # Romanian
    "rus_Cyrl": "ru",   # Russian
    "srp_Cyrl": "sr",   # Serbian
    "sna_Latn": "sn",   # Shona
    "slk_Latn": "sk",   # Slovak
    "slv_Latn": "sl",   # Slovene
    "som_Latn": "so",   # Somali
    "sot_Latn": "st",   # Southern Sotho
    "spa_Latn": "es",   # Spanish
    "swh_Latn": "sw",   # Swahili
    "swe_Latn": "sv",   # Swedish
    "tgl_Latn": "tl",   # Tagalog
    "tam_Taml": "ta",   # Tamil
    "tel_Telu": "te",   # Telugu
    "tha_Thai": "th",   # Thai
    "tso_Latn": "ts",   # Tsonga
    "tsn_Latn": "tn",   # Tswana
    "tur_Latn": "tr",   # Turkish
    "ukr_Cyrl": "uk",   # Ukrainian
    "urd_Arab": "ur",   # Urdu
    "vie_Latn": "vi",   # Vietnamese
    "cym_Latn": "cy",   # Welsh
    "xho_Latn": "xh",   # Xhosa
    "yor_Latn": "yo",   # Yoruba
    "zul_Latn": "zu",   # Zulu
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

# CJK/kana/Hangul — single characters are standalone tokens, so we approximate
# word count by character count divided by 2 (a typical CJK "word" averages
# 1.5-2 chars).
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_SHORT_MAX = 10
_LONG_MIN = 30


def bucket_by_length(text: str) -> str:
    """Classify ``text`` into ``short`` / ``medium`` / ``long`` by word count.

    Whitespace languages use the naive whitespace word count. CJK (which
    has no word boundaries) uses ``char_count / 2`` as a rough equivalent.
    """
    cjk_count = len(_CJK_RE.findall(text))
    if cjk_count > 0 and cjk_count >= len(text.replace(" ", "")) / 2:
        word_count = cjk_count / 2
    else:
        word_count = len(text.split())

    if word_count < _SHORT_MAX:
        return "short"
    if word_count > _LONG_MIN:
        return "long"
    return "medium"


def compute_metrics(
    predictions: List[Optional[str]],
    labels: List[str],
) -> Dict:
    """Aggregate per-language precision / recall / confusion pairs.

    * ``None`` predictions count as misses but do not pollute any predicted
      language's precision denominator (they have no claimed label).
    * Confusion pairs are keyed ``(true_label, predicted_label)``.
    """
    assert len(predictions) == len(labels), "pred/label length mismatch"

    total = len(labels)
    correct = sum(1 for p, y in zip(predictions, labels) if p == y)

    per_lang: Dict[str, Dict] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "samples": 0, "correct": 0}
    )
    confusion: Counter = Counter()

    for pred, true in zip(predictions, labels):
        per_lang[true]["samples"] += 1
        if pred == true:
            per_lang[true]["tp"] += 1
            per_lang[true]["correct"] += 1
        else:
            per_lang[true]["fn"] += 1
            if pred is not None:
                confusion[(true, pred)] += 1
                per_lang[pred]["fp"] += 1

    # Finalise per-language precision / recall.
    for lang, counts in per_lang.items():
        tp = counts["tp"]
        precision_den = tp + counts["fp"]
        recall_den = counts["samples"]
        counts["precision"] = tp / precision_den if precision_den else 0.0
        counts["recall"] = tp / recall_den if recall_den else 0.0
        top = sorted(
            [(other, confusion[(lang, other)]) for other in per_lang
             if (lang, other) in confusion and confusion[(lang, other)] > 0],
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        counts["top_confusions"] = top

    return {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "per_lang": dict(per_lang),
        "confusion_pairs": dict(confusion),
    }


def format_report(metrics: Dict) -> str:
    """Render a benchmark report as GitHub-flavoured Markdown."""
    lines: List[str] = []
    lines.append("# Lingua-py Accuracy on FLORES-200 devtest")
    lines.append("")

    # Overall ---------------------------------------------------------------
    lines.append("## Overall")
    lines.append("")
    acc_pct = metrics["overall_accuracy"] * 100
    lines.append(f"- Total samples: **{metrics['total_samples']}**")
    lines.append(f"- Accuracy: **{acc_pct:.2f}%**")
    if "accuracy_mode" in metrics:
        lines.append(f"- Accuracy mode: **{metrics['accuracy_mode']}**")
    if "rss_detector_mb" in metrics:
        lines.append(f"- Detector RSS: {metrics['rss_detector_mb']:.0f} MB "
                     f"(peak {metrics['rss_peak_mb']:.0f} MB)")
    if "elapsed_seconds" in metrics:
        rate = metrics["total_samples"] / max(metrics["elapsed_seconds"], 1e-9)
        lines.append(f"- Throughput: {rate:.0f} sent/s "
                     f"({metrics['elapsed_seconds']:.1f}s total)")
    lines.append("")

    # Per-language ----------------------------------------------------------
    lines.append("## Per-language")
    lines.append("")
    lines.append("| ISO | Samples | Correct | Precision | Recall | Top-3 confusions |")
    lines.append("|-----|---------|---------|-----------|--------|------------------|")
    rows = sorted(
        metrics["per_lang"].items(),
        key=lambda kv: kv[1]["recall"],
        reverse=True,
    )
    for lang, c in rows:
        if c["samples"] == 0:
            continue
        confusions = ", ".join(f"{other}:{n}" for other, n in c.get("top_confusions", []))
        lines.append(
            f"| {lang} | {c['samples']} | {c['correct']} "
            f"| {c['precision']*100:.1f}% | {c['recall']*100:.1f}% | {confusions or '—'} |"
        )
    lines.append("")

    # By length bucket ------------------------------------------------------
    if "by_bucket" in metrics:
        lines.append("## By length bucket")
        lines.append("")
        lines.append("| Bucket | Samples | Accuracy |")
        lines.append("|--------|---------|----------|")
        for bucket in ("short", "medium", "long"):
            b = metrics["by_bucket"].get(bucket)
            if not b:
                continue
            lines.append(
                f"| {bucket} | {b['samples']} | {b['accuracy']*100:.2f}% |"
            )
        lines.append("")

    # Top confusion pairs ---------------------------------------------------
    lines.append("## Top confusion pairs")
    lines.append("")
    pairs = sorted(
        metrics["confusion_pairs"].items(), key=lambda kv: kv[1], reverse=True
    )[:30]
    if pairs:
        lines.append("| True → Predicted | Count |")
        lines.append("|------------------|-------|")
        for (true, pred), n in pairs:
            lines.append(f"| {true} ↔ {pred} | {n} |")
    else:
        lines.append("No confusions recorded.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark flow
# ---------------------------------------------------------------------------


def ensure_flores_dataset(cache_dir: Path = _DEFAULT_CACHE) -> Path:
    """Download + extract the FLORES-200 tarball on first run; return the
    root path of the extracted ``flores200_dataset/`` directory.

    Idempotent: downloads are skipped when the devtest directory already
    exists. Uses only the stdlib so the script has no optional deps.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / "flores200_dataset"
    devtest_dir = root / "devtest"
    if devtest_dir.is_dir() and any(devtest_dir.iterdir()):
        return root

    tarball = cache_dir / "flores200_dataset.tar.gz"
    if not tarball.exists():
        print(f"Downloading FLORES-200 tarball (~25 MB) to {tarball}…", flush=True)
        urllib.request.urlretrieve(FLORES_TARBALL_URL, tarball)

    print(f"Extracting to {root}…", flush=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(cache_dir)
    if not devtest_dir.is_dir():
        raise RuntimeError(
            f"Extracted FLORES-200 but devtest dir not found under {root}"
        )
    return root


def load_flores_devtest(
    flores_codes: List[str], max_per_lang: Optional[int] = None
) -> Dict[str, List[str]]:
    """Load FLORES-200 devtest sentences for the given FLORES language codes.

    Returns ``{iso_639_1: [sentences]}``.
    """
    root = ensure_flores_dataset()
    devtest_dir = root / "devtest"
    result: Dict[str, List[str]] = {}
    for code in flores_codes:
        iso = FLORES_TO_ISO[code]
        path = devtest_dir / f"{code}.devtest"
        if not path.is_file():
            print(f"  WARN {code} → {iso}: {path.name} not found — skipped")
            continue
        sentences = path.read_text(encoding="utf-8").splitlines()
        if max_per_lang:
            sentences = sentences[:max_per_lang]
        result[iso] = sentences
    return result


def _peak_rss_mb() -> float:
    """Process peak RSS in MB. macOS returns bytes, Linux returns KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / 1024 / 1024
    return raw / 1024


def run_benchmark(
    candidate_langs: Optional[List[str]] = None,
    max_per_lang: Optional[int] = None,
    low_accuracy: bool = True,
) -> Dict:
    """Load FLORES + run lingua detector + compute metrics.

    ``low_accuracy`` toggles lingua's low-accuracy mode. Setting False
    enables the full n-gram resource set (~3× RAM, ~3× slower, notably
    better on similar-language pairs like sr/hr/bs or id/ms).
    """
    from lattifai.caption.parsers.language_detector import (
        _HAS_LETTER_RE,
        _build_detector,
    )

    flores_codes = list(FLORES_TO_ISO.keys())
    if candidate_langs:
        wanted = set(candidate_langs)
        flores_codes = [c for c in flores_codes if FLORES_TO_ISO[c] in wanted]

    iso_codes = [FLORES_TO_ISO[c] for c in flores_codes]

    rss_before = _peak_rss_mb()
    build_start = time.time()
    detector = _build_detector(iso_codes, low_accuracy=low_accuracy)
    build_elapsed = time.time() - build_start
    rss_after_build = _peak_rss_mb()
    if detector is None:
        raise RuntimeError("Failed to build lingua detector (is lingua-py installed?)")
    mode = "low-accuracy" if low_accuracy else "high-accuracy"
    print(f"  built {mode} detector for {len(iso_codes)} langs in "
          f"{build_elapsed:.1f}s (RSS +{rss_after_build - rss_before:.0f} MB)")

    print(f"Loading FLORES-200 devtest for {len(flores_codes)} languages…")
    data = load_flores_devtest(flores_codes, max_per_lang=max_per_lang)

    print("Running detector…")
    predictions: List[Optional[str]] = []
    labels: List[str] = []
    bucket_hits: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "correct": 0}
    )
    start = time.time()
    for iso, sentences in data.items():
        for sent in sentences:
            # Mirror ``detect_language`` short-circuit for empty/letter-less text.
            if not sent or not sent.strip() or not _HAS_LETTER_RE.search(sent):
                pred = None
            else:
                result = detector.detect_language_of(sent)
                pred = result.iso_code_639_1.name.lower() if result else None
            predictions.append(pred)
            labels.append(iso)
            bucket = bucket_by_length(sent)
            bucket_hits[bucket]["samples"] += 1
            if pred == iso:
                bucket_hits[bucket]["correct"] += 1
    elapsed = time.time() - start
    rss_peak = _peak_rss_mb()
    print(f"  detected {len(predictions)} sentences in {elapsed:.1f}s "
          f"({len(predictions)/max(elapsed, 1e-9):.0f} sent/s, peak RSS {rss_peak:.0f} MB)")

    metrics = compute_metrics(predictions, labels)
    metrics["by_bucket"] = {
        b: {"samples": v["samples"],
            "accuracy": v["correct"] / v["samples"] if v["samples"] else 0.0}
        for b, v in bucket_hits.items()
    }
    metrics["elapsed_seconds"] = elapsed
    metrics["accuracy_mode"] = mode
    metrics["build_seconds"] = build_elapsed
    metrics["rss_detector_mb"] = rss_after_build - rss_before
    metrics["rss_peak_mb"] = rss_peak
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max sentences per language (default: use full 1012)."
    )
    parser.add_argument(
        "--langs", default=None,
        help="Comma-separated ISO 639-1 codes to benchmark (default: all 72).",
    )
    parser.add_argument(
        "--high-accuracy", action="store_true",
        help="Use lingua's full-fidelity mode (3× RAM, 3× slower, better on "
             "similar-language pairs like sr/hr/bs).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Markdown output path. Defaults to "
             "scripts/benchmark_results/lingua_flores_[low|high]_accuracy.md.",
    )
    args = parser.parse_args()

    if args.output is None:
        tag = "high_accuracy" if args.high_accuracy else "low_accuracy"
        args.output = (
            Path(__file__).resolve().parent
            / "benchmark_results" / f"lingua_flores_{tag}.md"
        )

    candidates = [c.strip() for c in args.langs.split(",")] if args.langs else None
    metrics = run_benchmark(
        candidate_langs=candidates,
        max_per_lang=args.max,
        low_accuracy=not args.high_accuracy,
    )
    report = format_report(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nReport written to {args.output}")
    print(f"Overall accuracy: {metrics['overall_accuracy']*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
