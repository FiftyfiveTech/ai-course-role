#!/usr/bin/env python3
"""fetch_seeds.py — fetch 20 scenario seeds from the HF dataset viewer API.

No bulk download: each row is fetched individually by row index.
Idempotent: skips files that already exist.

Usage:
    uv run python scripts/fetch_seeds.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed manifest
# 20 seeds = 8 dev + 12 heldout
# Sources: allenai/soda (7), Anthropic/hh-rlhf (7),
#          bitext/Bitext-customer-support-llm-chatbot-training-dataset (6)
# ---------------------------------------------------------------------------

SEEDS = [
    # ── dev (8) ─────────────────────────────────────────────────────────────
    # allenai/soda  Apache-2.0
    dict(seed_id="soda-0100",  dataset="allenai/soda", split_hf="train", row_index=100,   licence="Apache-2.0", eval_split="dev"),
    dict(seed_id="soda-0500",  dataset="allenai/soda", split_hf="train", row_index=500,   licence="Apache-2.0", eval_split="dev"),
    dict(seed_id="soda-1000",  dataset="allenai/soda", split_hf="train", row_index=1000,  licence="Apache-2.0", eval_split="dev"),
    # Anthropic/hh-rlhf  MIT
    dict(seed_id="hh-80002",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80002, licence="MIT", eval_split="dev"),
    dict(seed_id="hh-80012",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80012, licence="MIT", eval_split="dev"),
    dict(seed_id="hh-80019",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80019, licence="MIT", eval_split="dev"),
    # bitext customer support  CC-BY-4.0
    dict(seed_id="bitext-1000", dataset="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split_hf="train", row_index=1000,  licence="CC-BY-4.0", eval_split="dev"),
    dict(seed_id="bitext-5000", dataset="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split_hf="train", row_index=5000,  licence="CC-BY-4.0", eval_split="dev"),

    # ── heldout (12) ────────────────────────────────────────────────────────
    # allenai/soda  Apache-2.0
    dict(seed_id="soda-0502",  dataset="allenai/soda", split_hf="train", row_index=502,   licence="Apache-2.0", eval_split="heldout"),
    dict(seed_id="soda-1500",  dataset="allenai/soda", split_hf="train", row_index=1500,  licence="Apache-2.0", eval_split="heldout"),
    dict(seed_id="soda-2000",  dataset="allenai/soda", split_hf="train", row_index=2000,  licence="Apache-2.0", eval_split="heldout"),
    dict(seed_id="soda-2500",  dataset="allenai/soda", split_hf="train", row_index=2500,  licence="Apache-2.0", eval_split="heldout"),
    # Anthropic/hh-rlhf  MIT
    dict(seed_id="hh-80007",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80007, licence="MIT", eval_split="heldout"),
    dict(seed_id="hh-80014",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80014, licence="MIT", eval_split="heldout"),
    dict(seed_id="hh-80015",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80015, licence="MIT", eval_split="heldout"),
    dict(seed_id="hh-80023",   dataset="Anthropic/hh-rlhf", split_hf="train", row_index=80023, licence="MIT", eval_split="heldout"),
    # bitext customer support  CC-BY-4.0
    dict(seed_id="bitext-3000",  dataset="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split_hf="train", row_index=3000,  licence="CC-BY-4.0", eval_split="heldout"),
    dict(seed_id="bitext-7000",  dataset="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split_hf="train", row_index=7000,  licence="CC-BY-4.0", eval_split="heldout"),
    dict(seed_id="bitext-10000", dataset="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split_hf="train", row_index=10000, licence="CC-BY-4.0", eval_split="heldout"),
    dict(seed_id="bitext-12000", dataset="bitext/Bitext-customer-support-llm-chatbot-training-dataset", split_hf="train", row_index=12000, licence="CC-BY-4.0", eval_split="heldout"),
]

PROVENANCE = {
    "allenai/soda": "https://huggingface.co/datasets/allenai/soda",
    "Anthropic/hh-rlhf": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset": (
        "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    ),
}

HF_VIEWER = "https://datasets-server.huggingface.co/rows"
REPO_ROOT = Path(__file__).parent.parent
SEEDS_DIR = REPO_ROOT / "evals" / "seeds"


def fetch_row(dataset: str, split_hf: str, row_index: int) -> dict:
    url = (
        f"{HF_VIEWER}"
        f"?dataset={urllib.parse.quote(dataset, safe='')}"
        f"&config=default"
        f"&split={split_hf}"
        f"&offset={row_index}"
        f"&length=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "role-fetch-seeds/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    rows = data.get("rows", [])
    if not rows:
        raise ValueError(f"no rows returned for {dataset} row {row_index}")
    return rows[0]["row"]


def main() -> None:
    dev_dir = SEEDS_DIR / "dev"
    heldout_dir = SEEDS_DIR / "heldout"
    dev_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    fetched = skipped = 0
    for s in SEEDS:
        out_dir = dev_dir if s["eval_split"] == "dev" else heldout_dir
        out_path = out_dir / f"{s['seed_id']}.json"

        if out_path.exists():
            print(f"  skip  {s['seed_id']} (exists)")
            skipped += 1
            continue

        print(f"  fetch {s['seed_id']}  {s['dataset']} row {s['row_index']} ...")
        raw = fetch_row(s["dataset"], s["split_hf"], s["row_index"])

        record = {
            "seed_id": s["seed_id"],
            "source_dataset": s["dataset"],
            "row_index": s["row_index"],
            "licence": s["licence"],
            "provenance_url": PROVENANCE[s["dataset"]],
            "eval_split": s["eval_split"],
            "fetch_url": (
                f"{HF_VIEWER}"
                f"?dataset={urllib.parse.quote(s['dataset'], safe='')}"
                f"&config=default&split={s['split_hf']}"
                f"&offset={s['row_index']}&length=1"
            ),
            "raw": raw,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"    -> evals/seeds/{s['eval_split']}/{s['seed_id']}.json")
        fetched += 1
        time.sleep(0.4)  # polite pacing

    n_dev = sum(1 for s in SEEDS if s["eval_split"] == "dev")
    n_heldout = sum(1 for s in SEEDS if s["eval_split"] == "heldout")
    print(
        f"\nDone. fetched={fetched} skipped={skipped} "
        f"total={len(SEEDS)} ({n_dev} dev / {n_heldout} heldout)"
    )


if __name__ == "__main__":
    main()
