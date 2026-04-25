"""Measure how often Layer 2 / Layer 3 rollback fires under the new
single-gate flow. We do not need to instrument caption.py — the
invariant suffices: ``detect_bilingual_mode`` says *bilingual* but
``extract_alignment_supervisions`` returns ``secondary == []`` ⇒ a
rollback path must have run.

A near-zero rate means the public detector is already accurate enough
that the rollback layers are dead weight on the bilingual path; a high
rate means we still rely on them and dropping them would regress real
captions.

Usage:
    python scripts/probe_rollback_rate.py [--root DIR] [--formats srt,ass]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Reuse the same corpus loader as smoke_roundtrip_corpus.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from smoke_roundtrip_corpus import DEFAULT_ROOT, collect_files  # noqa: E402

from lattifai.caption import Caption  # noqa: E402
from lattifai.caption.caption import BilingualMode  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--formats", default="srt,ass")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(args.formats.split(","))
    entries = collect_files(args.root, formats)
    if args.limit:
        entries = entries[: args.limit]

    mode_counts: Counter[str] = Counter()
    rollback_counts: Counter[str] = Counter()
    failed: list[tuple[str, str]] = []

    for label, fmt, source in entries:
        try:
            if hasattr(source, "read"):
                cap = Caption.read(source, format=fmt)
            else:
                cap = Caption.read(source)
        except Exception as exc:
            failed.append((label, f"read: {exc}"))
            continue
        try:
            mode = cap.detect_bilingual_mode()
        except Exception as exc:
            failed.append((label, f"detect: {exc}"))
            continue
        mode_counts[mode.value] += 1
        if mode == BilingualMode.NONE:
            continue
        try:
            _, secondary, _ = cap.extract_alignment_supervisions()
        except Exception as exc:
            failed.append((label, f"extract: {exc}"))
            continue
        if not secondary:
            rollback_counts[mode.value] += 1

    total = sum(mode_counts.values())
    print(f"=== probe complete ({total} files, {len(failed)} skipped) ===")
    print()
    print("[detect_bilingual_mode counts]")
    for mode_value in ("none", "line_by_line", "same_timing_pairs", "style_grouped"):
        print(f"  {mode_value:14} {mode_counts.get(mode_value, 0)}")
    print()
    print("[rollback rate among detected-bilingual files]")
    for mode_value in ("line_by_line", "same_timing_pairs", "style_grouped"):
        detected = mode_counts.get(mode_value, 0)
        rolled_back = rollback_counts.get(mode_value, 0)
        rate = (rolled_back / detected * 100.0) if detected else 0.0
        print(f"  {mode_value:14} {rolled_back}/{detected}  ({rate:.1f}%)")
    if failed:
        print()
        print(f"[skipped: {len(failed)}]")
        for path, why in failed[:10]:
            print(f"  {why}: {path}")

    # Dump rollback-triggering paths (so we can drill into why detect
    # was over-eager on those particular files).
    rollback_paths = []
    for label, fmt, source in entries:
        try:
            if hasattr(source, "read"):
                cap = Caption.read(source, format=fmt)
            else:
                cap = Caption.read(source)
            mode = cap.detect_bilingual_mode()
            if mode == BilingualMode.NONE:
                continue
            _, secondary, _ = cap.extract_alignment_supervisions()
            if not secondary:
                rollback_paths.append((mode.value, label, len(cap.supervisions)))
        except Exception:
            continue
    if rollback_paths:
        print()
        print(f"[rollback-triggered files: {len(rollback_paths)}]")
        for mode_value, label, n in rollback_paths:
            print(f"  {mode_value:14} n={n:5}  {label}")


if __name__ == "__main__":
    main()
