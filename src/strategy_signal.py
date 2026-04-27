"""Compute edge between model probability and market-implied probability.
Simple placeholder that computes edge and returns trade suggestion when above threshold.
"""
from typing import Dict


def compute_edge(p_model: float, q_market: float) -> Dict:
    edge = p_model - q_market
    return {"edge": edge, "trade": edge}
