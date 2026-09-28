"""Unit tests for ROLE-024: agreement scorer, checked against hand-computed fixtures.

Every expected exact/band/kappa number below is derived by hand in the
comments — the assertions check the code against that independent arithmetic,
not against whatever the code itself happens to produce.
"""

import json

import pytest

from role.agreement import (
    cohens_kappa,
    load_hand_scores,
    load_machine_scores,
    pair_and_score,
)
from role.scorecard import RUBRIC_ITEMS


def _all_items(score: int) -> dict[str, int]:
    return {name: score for name in RUBRIC_ITEMS}


# --- Fixture 1: general case, one item, n=10 -------------------------------
#
# hand/machine pairs for "clarity":
#   (0,0) (0,0) (0,1) (1,1) (1,1) (1,2) (2,0) (2,2) (2,2) (2,2)
#
# Confusion matrix matrix[hand][machine]:
#         m=0  m=1  m=2   row
#   h=0    2    1    0     3
#   h=1    0    2    1     3
#   h=2    1    0    3     4
#   col    3    3    4    10
#
# p_o = (2 + 2 + 3) / 10 = 7/10 = 0.7
# p_e = (3*3 + 3*3 + 4*4) / 100 = (9 + 9 + 16) / 100 = 34/100 = 0.34
# kappa = (0.7 - 0.34) / (1 - 0.34) = 0.36 / 0.66 = 6/11 ~= 0.545454...
# exact = p_o = 0.7
# band: only |hand-machine| == 2 cells are excluded: (h=0,m=2)=0, (h=2,m=0)=1
#       -> 1 excluded pair out of 10 -> band = 9/10 = 0.9

_FIXTURE_1_PAIRS = [
    (0, 0), (0, 0), (0, 1), (1, 1), (1, 1), (1, 2), (2, 0), (2, 2), (2, 2), (2, 2),
]


def test_cohens_kappa_general_case():
    assert cohens_kappa(_FIXTURE_1_PAIRS) == pytest.approx(6 / 11)


def test_pair_and_score_general_case():
    # Every session needs all RUBRIC_ITEMS (real hand/machine files always do);
    # only "clarity" carries the fixture's pairs, the rest are filler.
    hand = {f"s{i}": {**_all_items(1), "clarity": h} for i, (h, _m) in enumerate(_FIXTURE_1_PAIRS)}
    machine = {
        f"s{i}": {**_all_items(1), "clarity": m} for i, (_h, m) in enumerate(_FIXTURE_1_PAIRS)
    }
    result = pair_and_score(hand, machine)["clarity"]
    assert result.n == 10
    assert result.exact_agreement == pytest.approx(0.7)
    assert result.band_agreement == pytest.approx(0.9)
    assert result.kappa == pytest.approx(6 / 11)


# --- Fixture 2: perfect agreement, mixed distribution -----------------------
#
# pairs: (0,0) (1,1) (2,2) (1,1) (0,0)  -> counts: 0x2, 1x2, 2x1, n=5
# p_o = 5/5 = 1.0
# p_e = (2/5)^2 + (2/5)^2 + (1/5)^2 = 0.16 + 0.16 + 0.04 = 0.36
# kappa = (1.0 - 0.36) / (1 - 0.36) = 0.64 / 0.64 = 1.0 exactly

_FIXTURE_2_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 1), (0, 0)]


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa(_FIXTURE_2_PAIRS) == 1.0


def test_pair_and_score_perfect_agreement():
    hand = {f"s{i}": {**_all_items(1), "clarity": h} for i, (h, _m) in enumerate(_FIXTURE_2_PAIRS)}
    machine = {
        f"s{i}": {**_all_items(1), "clarity": m} for i, (_h, m) in enumerate(_FIXTURE_2_PAIRS)
    }
    result = pair_and_score(hand, machine)["clarity"]
    assert result.exact_agreement == 1.0
    assert result.band_agreement == 1.0
    assert result.kappa == 1.0


# --- Fixture 3: degenerate / undefined kappa --------------------------------
#
# Same constant category for both raters, every session: (2,2) x4
# p_o = 1.0, p_e = (4/4)^2 = 1.0 -> undefined (0/0), must return None.


def test_cohens_kappa_undefined_when_both_raters_constant_and_equal():
    assert cohens_kappa([(2, 2)] * 4) is None


def test_pair_and_score_undefined_kappa_scoped_to_kappa_only():
    hand = {f"s{i}": {**_all_items(1), "clarity": 2} for i in range(4)}
    machine = {f"s{i}": {**_all_items(1), "clarity": 2} for i in range(4)}
    result = pair_and_score(hand, machine)["clarity"]
    assert result.kappa is None
    assert result.exact_agreement == 1.0
    assert result.band_agreement == 1.0


# Constant-but-different categories: p_o=0, p_e=0 -> kappa=0.0 exactly, a
# real well-defined chance-level value, distinct from the undefined None case.


def test_cohens_kappa_zero_is_not_confused_with_undefined():
    assert cohens_kappa([(0, 1)] * 4) == 0.0


# --- Loud-error paths --------------------------------------------------------


def test_pair_and_score_raises_on_mismatched_seed_ids():
    hand = {"a": _all_items(1), "b": _all_items(1)}
    machine = {"a": _all_items(1), "c": _all_items(1)}
    with pytest.raises(ValueError) as excinfo:
        pair_and_score(hand, machine)
    message = str(excinfo.value)
    assert "b" in message
    assert "c" in message


def test_pair_and_score_raises_on_empty_input():
    with pytest.raises(ValueError, match="empty"):
        pair_and_score({}, {})


def test_cohens_kappa_raises_on_empty_pairs():
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([])


# --- Loaders (tmp_path-based) ------------------------------------------------


def _write_hand_file(tmp_path, seed_id: str, scores: "dict[str, int] | None" = None):
    body = {
        "seed_id": seed_id,
        "scorer": "test@example.com",
        "scored_at": "2026-01-01",
        "rubric_version": "evals/RUBRIC.md",
        "scored_speaker": "Agent (response field)",
        "items": {
            name: {"score": (scores or _all_items(1))[name], "evidence": "because I said so"}
            for name in RUBRIC_ITEMS
        },
        "total": sum((scores or _all_items(1)).values()),
        "max": 12,
    }
    (tmp_path / f"{seed_id}.score.json").write_text(json.dumps(body), encoding="utf-8")


def test_load_hand_scores_reads_score_field(tmp_path):
    _write_hand_file(tmp_path, "seed1", _all_items(2))
    result = load_hand_scores(tmp_path)
    assert result == {"seed1": _all_items(2)}
    # evidence/scorer/total must not leak into the parsed shape
    assert "evidence" not in result["seed1"]
    assert "scorer" not in result


def test_load_hand_scores_raises_on_filename_seed_id_mismatch(tmp_path):
    _write_hand_file(tmp_path, "seed1")
    (tmp_path / "wrong-name.score.json").write_text(
        (tmp_path / "seed1.score.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "seed1.score.json").unlink()
    with pytest.raises(ValueError, match="seed1"):
        load_hand_scores(tmp_path)


def test_load_hand_scores_raises_on_missing_rubric_item(tmp_path):
    _write_hand_file(tmp_path, "seed1")
    body = json.loads((tmp_path / "seed1.score.json").read_text(encoding="utf-8"))
    del body["items"]["clarity"]
    (tmp_path / "seed1.score.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="clarity"):
        load_hand_scores(tmp_path)


def test_load_machine_scores_reads_scorecard(tmp_path):
    body = {
        "session_id": "seed1",
        "rubric_version": "evals/RUBRIC.md",
        "items": {
            name: {"score": 2, "evidence": [{"turn_id": "seed1:1"}]} for name in RUBRIC_ITEMS
        },
    }
    (tmp_path / "seed1.json").write_text(json.dumps(body), encoding="utf-8")
    result = load_machine_scores(tmp_path)
    assert result == {"seed1": _all_items(2)}
