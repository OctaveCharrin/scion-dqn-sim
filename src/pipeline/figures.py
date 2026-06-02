"""LNCS-style figure metadata for evaluation result plots."""

from __future__ import annotations

from typing import Mapping

COLUMN_WIDTH: float = 3.5
FULL_WIDTH: float = 7.0

METHOD_DISPLAY_NAMES: Mapping[str, str] = {
    "dqn": "DQN (Enhanced)",
    "simple_dqn": "DQN (Simple)",
    "scoring_simple_dqn": "DQN (Path-Scoring Simple)",
    "scoring_enhanced_dqn": "DQN (Path-Scoring Enhanced)",
    "conditional_dqn": "DQN (Conditional)",
    "scoring_dqn": "DQN (Path-Scoring Simple)",
    "shortest_path": "Shortest Path",
    "widest_path": "Widest Path",
    "lowest_latency": "Lowest Latency",
    "ecmp": "ECMP",
    "random": "Random",
    "scion_default": "SCION Default",
}

METHOD_COLORS: Mapping[str, str] = {
    "dqn": "#1f77b4",
    "simple_dqn": "#17becf",
    "scoring_simple_dqn": "#bcbd22",
    "scoring_enhanced_dqn": "#7f7f7f",
    "conditional_dqn": "#d62728",
    "scoring_dqn": "#bcbd22",
    "shortest_path": "#ff7f0e",
    "widest_path": "#2ca02c",
    "lowest_latency": "#d62728",
    "ecmp": "#9467bd",
    "random": "#8c564b",
    "scion_default": "#e377c2",
}


def apply_lncs_style() -> None:
    """Configure matplotlib's rcParams for LNCS-style figures."""
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"]
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 9
    rcParams["figure.titlesize"] = 12


def display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")
