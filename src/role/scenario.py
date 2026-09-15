"""Scenario loader: reads scenarios/*.yaml into typed Scenario objects.

A malformed or missing file raises ValueError immediately — no silent skips.

Schema (all fields required):
  id:         str   — slug used as filename stem
  title:      str   — display name
  difficulty: str   — easy | medium | hard
  goal:       str   — what the trainee should practise
  persona:
    name:     str   — character name shown in the session header
    policy:   str   — prompt filename relative to prompts/
    opening:  str   — instruction sent to the persona to produce its first line
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCENARIOS_DIR = Path(__file__).parent.parent.parent / "scenarios"
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_REQUIRED_TOP = {"id", "title", "difficulty", "goal", "persona"}
_REQUIRED_PERSONA = {"name", "policy", "opening"}
_VALID_DIFFICULTY = {"easy", "medium", "hard"}


@dataclass
class Scenario:
    id: str
    title: str
    difficulty: str
    goal: str
    persona_name: str
    policy_path: Path   # absolute path to the prompt file
    opening: str        # instruction that produces the persona's first utterance


def load(path: Path) -> Scenario:
    """Load and validate one scenario YAML; raise ValueError on any problem."""
    if not path.exists():
        raise ValueError(f"Scenario file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")

    missing = _REQUIRED_TOP - raw.keys()
    if missing:
        raise ValueError(f"{path}: missing required fields: {sorted(missing)}")

    persona = raw["persona"]
    if not isinstance(persona, dict):
        raise ValueError(f"{path}: 'persona' must be a mapping")
    missing_p = _REQUIRED_PERSONA - persona.keys()
    if missing_p:
        raise ValueError(f"{path}: persona missing required fields: {sorted(missing_p)}")

    difficulty = raw["difficulty"]
    if difficulty not in _VALID_DIFFICULTY:
        raise ValueError(
            f"{path}: difficulty must be one of {sorted(_VALID_DIFFICULTY)}, got {difficulty!r}"
        )

    policy_path = PROMPTS_DIR / persona["policy"]
    if not policy_path.exists():
        raise ValueError(f"{path}: persona policy file not found: {policy_path}")

    return Scenario(
        id=str(raw["id"]),
        title=str(raw["title"]),
        difficulty=difficulty,
        goal=str(raw["goal"]),
        persona_name=str(persona["name"]),
        policy_path=policy_path,
        opening=str(persona["opening"]),
    )


def load_all(directory: Path = SCENARIOS_DIR) -> list[Scenario]:
    """Load every *.yaml in *directory*; raise ValueError on the first bad file."""
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValueError(f"No scenario files found in {directory}")
    return [load(p) for p in paths]
