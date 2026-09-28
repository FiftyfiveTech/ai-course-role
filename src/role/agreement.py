"""Agreement scorer: evaluator-agent scores vs the hand scores, per rubric item.

evals/dev/<seed_id>.score.json (hand, ROLE-015) predates the Scorecard pydantic
schema (ROLE-022) — its evidence is a free-text string, not list[TurnRef], and
it's keyed by seed_id, not session_id. This module does not force it into
Scorecard; it parses both the hand files and the machine evals/scores/dev/
files (real Scorecard output, ROLE-023) into one common {seed_id: {item:
score}} shape and reuses RUBRIC_ITEMS as the source of truth for required
items.

Dev-only, read-only: never touches evals/heldout/ (sealed, Evaluator-only per
CLAUDE.md's blind-labelling rule) or evals/seeds/.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from role.scorecard import RUBRIC_ITEMS, Scorecard

REPO_ROOT = Path(__file__).parent.parent.parent
EVALS_DEV_DIR = REPO_ROOT / "evals" / "dev"
SCORES_DEV_DIR = REPO_ROOT / "evals" / "scores" / "dev"


@dataclass(frozen=True)
class ItemAgreement:
    """One rubric item's agreement between hand and machine scores."""

    item: str
    n: int
    exact_agreement: float
    band_agreement: float
    kappa: "float | None"  # None only when p_e == 1.0 (undefined, not 0.0)


def load_hand_scores(dev_dir: Path = EVALS_DEV_DIR) -> dict[str, dict[str, int]]:
    """Parse evals/dev/*.score.json into {seed_id: {item: score}}, dropping
    evidence/scorer/scored_at/total/max — this scorer only needs the ints."""
    result: dict[str, dict[str, int]] = {}
    for path in sorted(dev_dir.glob("*.score.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        expected_id = path.name.removesuffix(".score.json")
        seed_id = body.get("seed_id")
        if seed_id != expected_id:
            raise ValueError(
                f"{path.name}: seed_id {seed_id!r} does not match filename {expected_id!r}"
            )
        items = body.get("items", {})
        scores: dict[str, int] = {}
        for name in RUBRIC_ITEMS:
            if name not in items:
                raise ValueError(f"{path.name}: missing rubric item {name!r}")
            score = items[name].get("score")
            if not isinstance(score, int) or isinstance(score, bool) or score not in (0, 1, 2):
                raise ValueError(f"{path.name}: item {name!r} has invalid score {score!r}")
            scores[name] = score
        result[seed_id] = scores
    return result


def load_machine_scores(scores_dir: Path = SCORES_DEV_DIR) -> dict[str, dict[str, int]]:
    """Parse evals/scores/dev/*.json (real Scorecard output) into
    {seed_id: {item: score}}."""
    result: dict[str, dict[str, int]] = {}
    for path in sorted(scores_dir.glob("*.json")):
        scorecard = Scorecard.model_validate_json(path.read_text(encoding="utf-8"))
        expected_id = path.stem
        if scorecard.session_id != expected_id:
            raise ValueError(
                f"{path.name}: session_id {scorecard.session_id!r} does not match "
                f"filename {expected_id!r}"
            )
        result[scorecard.session_id] = {
            name: item.score for name, item in scorecard.items.items()
        }
    return result


def _confusion_matrix(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """matrix[hand_score][machine_score] = count, fixed over categories [0,1,2]
    so a category that never appears still contributes a correct zero row/col."""
    matrix = [[0, 0, 0] for _ in range(3)]
    for hand_score, machine_score in pairs:
        matrix[hand_score][machine_score] += 1
    return matrix


def cohens_kappa(pairs: list[tuple[int, int]]) -> "float | None":
    """Unweighted Cohen's kappa over categories {0,1,2}.

    Returns None iff p_e == 1.0. Proof that's the only singularity: p_e=1
    forces sum(row_k*col_k) == n*n with sum(row_k) == sum(col_k) == n, which
    (by Cauchy-Schwarz on the marginals) forces both marginals to be fully
    concentrated on the same category k* — meaning every pair is (k*, k*),
    which forces p_o == 1.0 too. So (1 - p_e) never vanishes without also
    making the numerator vanish identically; ZeroDivisionError is therefore
    structurally unreachable and never caught here.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("cohens_kappa: pairs is empty")

    matrix = _confusion_matrix(pairs)
    p_o = sum(matrix[k][k] for k in range(3)) / n
    row_marginals = [sum(matrix[i]) for i in range(3)]
    col_marginals = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    p_e = sum(row_marginals[k] * col_marginals[k] for k in range(3)) / (n * n)

    if p_e == 1.0:
        return None
    return (p_o - p_e) / (1 - p_e)


def _score_item(item: str, pairs: list[tuple[int, int]]) -> ItemAgreement:
    n = len(pairs)
    exact = sum(h == m for h, m in pairs) / n
    band = sum(abs(h - m) <= 1 for h, m in pairs) / n
    return ItemAgreement(
        item=item, n=n, exact_agreement=exact, band_agreement=band, kappa=cohens_kappa(pairs)
    )


def pair_and_score(
    hand: dict[str, dict[str, int]], machine: dict[str, dict[str, int]]
) -> dict[str, ItemAgreement]:
    """Pair hand and machine scores by seed_id and compute per-item agreement.

    Raises loudly on any seed_id present in one input but not the other, and
    on fully-empty input — no silent intersection-only fallback.
    """
    if not hand and not machine:
        raise ValueError("pair_and_score: no seed_ids to pair — both inputs are empty")

    missing_from_machine = sorted(hand.keys() - machine.keys())
    missing_from_hand = sorted(machine.keys() - hand.keys())
    if missing_from_machine or missing_from_hand:
        raise ValueError(
            f"pair_and_score: seed_id sets differ — "
            f"{len(missing_from_machine)} in hand but not machine: {missing_from_machine}; "
            f"{len(missing_from_hand)} in machine but not hand: {missing_from_hand}"
        )

    common_ids = sorted(hand.keys())
    result: dict[str, ItemAgreement] = {}
    for item in RUBRIC_ITEMS:
        pairs = [(hand[sid][item], machine[sid][item]) for sid in common_ids]
        result[item] = _score_item(item, pairs)
    return result
