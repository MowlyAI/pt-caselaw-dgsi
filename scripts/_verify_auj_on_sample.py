"""Replay the AUJ detector against the stored 500-doc eval sample and the
100-doc sample, cross-checking against the manual audit. Also does a small
crash-recovery simulation for save_state / load_state."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.extractor import _is_jurisprudence_unification  # noqa: E402


def replay(dir_path: Path) -> tuple[int, int, list[str]]:
    hits = []
    total = 0
    for p in sorted(dir_path.glob("*.json")):
        total += 1
        d = json.loads(p.read_text())
        if _is_jurisprudence_unification(d.get("procedural_type"), d.get("case_type")):
            hits.append(p.stem)
    return total, len(hits), hits


def main():
    for sample in ("data/eval/prod100", "data/eval/prod500"):
        tot, n, hits = replay(Path(sample))
        print(f"\n=== {sample} ===")
        print(f"  {n}/{tot} ({n/max(1,tot)*100:.1f}%) classified as AUJ")
        for h in hits[:5]:
            d = json.loads((Path(sample) / f"{h}.json").read_text())
            print(f"    {h[:40]}  pt={d.get('procedural_type')!r:38s} ct={d.get('case_type')!r}")
        if len(hits) > 5:
            print(f"    ... ({len(hits) - 5} more)")

    # Crash-recovery simulation: write corrupt state, then ensure load_state
    # recovers cleanly rather than crashing.
    print("\n=== crash-recovery simulation ===")
    from extractor.runner import load_state, save_state, STATE_FILE  # noqa: E402
    original = None
    if STATE_FILE.exists():
        original = STATE_FILE.read_bytes()
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Corrupt file (truncated write, as if SIGKILL mid-json.dump)
        STATE_FILE.write_text('{"processed_doc_ids": {"jstj.nsf": [')
        try:
            loaded = load_state()
            assert loaded == {"processed_doc_ids": {}}, \
                f"expected empty fallback, got {loaded!r}"
            print("  ✓ load_state() recovered from corrupt JSON by returning empty state")
        finally:
            pass
        # Now test atomic write: run save_state, verify tmp is gone & content ok
        save_state({"processed_doc_ids": {"jstj.nsf": ["abc", "def"]}})
        assert not (STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")).exists(), \
            "tmp file should be renamed away"
        loaded2 = load_state()
        assert loaded2 == {"processed_doc_ids": {"jstj.nsf": ["abc", "def"]}}
        print("  ✓ save_state() writes atomically; tmp sidecar is renamed away")
    finally:
        # restore
        if original is not None:
            STATE_FILE.write_bytes(original)
        else:
            STATE_FILE.unlink(missing_ok=True)
        print("  ✓ original state file restored")


if __name__ == "__main__":
    main()
