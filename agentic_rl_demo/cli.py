from __future__ import annotations

import argparse
import json

from .runtime import run_scenario
from .scenarios import available_scenarios, load_scenario
from .verifier import score_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic safe-agent scenario and inspect its reward.")
    parser.add_argument("scenario", nargs="?", default="successful_refund", help="Scenario name")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        print("\n".join(available_scenarios()))
        return 0

    try:
        scenario = load_scenario(args.scenario)
    except FileNotFoundError as error:
        print(error)
        return 2

    trajectory = run_scenario(scenario)
    verification = score_trajectory(scenario, trajectory)

    if args.json:
        print(json.dumps({"scenario": scenario.title, "trajectory": trajectory.to_dict(), "verification": verification.to_dict()}, indent=2))
        return 0

    print(f"\n{scenario.title}")
    print("=" * len(scenario.title))
    print(scenario.description)
    print(f"\nCustomer: {scenario.customer_message}\n")
    for event in trajectory.events:
        marker = {"success": "✓", "error": "↻" if event.retryable else "✗", "blocked": "⊘"}[event.status]
        suffix = f" — {event.error}" if event.error else ""
        attempt = f" (attempt {event.attempt})" if event.attempt > 1 else ""
        print(f"  {marker} {event.tool}{attempt}{suffix}")
    print(f"\nAgent response: {trajectory.final_message}")
    print("\nVerifier")
    print("--------")
    for name, value in verification.subscores.items():
        print(f"  {name:<14} {value:.2f}")
    print(f"  {'raw reward':<14} {verification.raw_reward:.2f}")
    print(f"  {'final reward':<14} {verification.reward:.2f}")
    if verification.active_caps:
        print("\nActive reward caps:")
        for name in verification.active_caps:
            print(f"  - {name}: {verification.cap_reasons[name]}")
    else:
        print("\nNo reward caps triggered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
