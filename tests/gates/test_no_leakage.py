#!/usr/bin/env python3
"""test_no_leakage.py — assert evals/dev ∩ evals/heldout = ∅ by content hash.

A seed that appears in both splits would let the Builder tune on held-out data,
invalidating every gate result that follows.

Pass conditions:
1. Both evals/seeds/dev/ and evals/seeds/heldout/ contain at least one *.json
   file.  An empty directory produces a VACUOUS PASS warning and the test is
   skipped — a vacuous pass is not a real pass.
2. No JSON file in dev/ has the same SHA-256 as any file in heldout/.

Run standalone : uv run python tests/gates/test_no_leakage.py
Run via pytest : uv run pytest tests/gates/test_no_leakage.py -s
"""

import hashlib
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEV_DIR = REPO_ROOT / "evals" / "seeds" / "dev"
HELDOUT_DIR = REPO_ROOT / "evals" / "seeds" / "heldout"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_hashes(directory: Path) -> dict[str, str]:
    """Return {sha256: filename} for every *.json in *directory*."""
    return {
        _sha256(f): f.name
        for f in sorted(directory.glob("*.json"))
    }


def run_leakage_check() -> dict:
    dev_hashes = _collect_hashes(DEV_DIR)
    heldout_hashes = _collect_hashes(HELDOUT_DIR)

    vacuous = False
    vacuous_reason = ""
    if not dev_hashes:
        vacuous = True
        vacuous_reason = f"DEV dir is empty: {DEV_DIR}"
    elif not heldout_hashes:
        vacuous = True
        vacuous_reason = f"HELDOUT dir is empty: {HELDOUT_DIR}"

    overlap = {
        sha: (dev_hashes[sha], heldout_hashes[sha])
        for sha in dev_hashes
        if sha in heldout_hashes
    }

    return {
        "dev_count": len(dev_hashes),
        "heldout_count": len(heldout_hashes),
        "overlap_count": len(overlap),
        "overlap": overlap,   # {sha: (dev_file, heldout_file)}
        "vacuous": vacuous,
        "vacuous_reason": vacuous_reason,
    }


def _print_report(r: dict) -> None:
    print()
    print(f"{'dev seeds':>12}: {r['dev_count']}")
    print(f"{'heldout seeds':>12}: {r['heldout_count']}")
    print(f"{'overlap':>12}: {r['overlap_count']}")
    print()
    if r["vacuous"]:
        print(f"VACUOUS PASS — {r['vacuous_reason']}")
        print("This is NOT a real pass. Fill the missing split before trusting gate results.")
    elif r["overlap_count"] == 0:
        print("PASS — dev ∩ heldout = ∅")
    else:
        print("FAIL — leaked seeds:")
        for sha, (dev_f, heldout_f) in r["overlap"].items():
            print(f"  sha256={sha[:16]}…  dev/{dev_f}  ==  heldout/{heldout_f}")
    print()


def test_no_leakage() -> None:
    """pytest entry-point: fail on overlap, skip with warning on vacuous."""
    result = run_leakage_check()
    _print_report(result)

    if result["vacuous"]:
        warnings.warn(
            f"VACUOUS PASS: {result['vacuous_reason']} — "
            "leakage check cannot run on an empty split.",
            stacklevel=2,
        )
        import pytest
        pytest.skip(f"VACUOUS PASS — {result['vacuous_reason']}")

    assert result["overlap_count"] == 0, (
        f"{result['overlap_count']} seed(s) appear in both dev and heldout:\n"
        + "\n".join(
            f"  sha256={sha[:16]}…  dev/{df}  ==  heldout/{hf}"
            for sha, (df, hf) in result["overlap"].items()
        )
    )


if __name__ == "__main__":
    result = run_leakage_check()
    _print_report(result)

    if result["vacuous"]:
        print("GATE: VACUOUS PASS (not a real pass)")
        sys.exit(2)

    print(f"GATE: {'PASS' if result['overlap_count'] == 0 else 'FAIL'}")
    sys.exit(0 if result["overlap_count"] == 0 else 1)
