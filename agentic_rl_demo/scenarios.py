from __future__ import annotations

import json
from pathlib import Path

from .models import Scenario


DEFAULT_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def available_scenarios(directory: Path | None = None) -> list[str]:
    root = directory or DEFAULT_SCENARIO_DIR
    return sorted(path.stem for path in root.glob("*.json"))


def load_scenario(name: str, directory: Path | None = None) -> Scenario:
    root = directory or DEFAULT_SCENARIO_DIR
    path = root / f"{name}.json"
    if not path.exists():
        choices = ", ".join(available_scenarios(root))
        raise FileNotFoundError(f"unknown scenario '{name}'. Available: {choices}")
    with path.open(encoding="utf-8") as stream:
        return Scenario.from_dict(json.load(stream))
