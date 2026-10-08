"""A lightweight, synthetic demo of safe agent execution and verification."""

from .runtime import run_scenario
from .verifier import score_trajectory

__all__ = ["run_scenario", "score_trajectory"]
__version__ = "0.1.0"
