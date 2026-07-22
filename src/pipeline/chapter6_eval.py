"""Chapter 6 evaluation logic: intent-conditioning ablation, intent alignment,
and probing / single-path-ceiling analysis.

This module holds the reusable logic behind the thin ``evaluation/`` CLIs
(``eval_ablation_intent.py``, ``eval_intent_alignment.py``,
``eval_probing_ceiling.py``, ``run_chapter6.py``). It is strictly additive: it
imports the existing evaluation env, baselines, and agent loaders and never
modifies the legacy pipeline.

Three studies, all over the held-out last 14 days (``EVAL_HOURS``):

1. ``run_ablation`` — Flat DQN vs Conditional-Concat vs Conditional-FiLM across
   the five named reward profiles. Produces the mean intent-weighted reward
   matrix plus a *behavioral divergence* metric (how much a method re-ranks
   paths as the intent vector changes). Concat/Flat ≈ 0; FiLM ≫ 0.
2. ``run_intent_alignment`` — the FiLM checkpoint only: the 5×5 reward matrix
   R(intent_told, intent_scored) (diagonal dominance) and the distribution of
   chosen-path metrics (latency / bandwidth / trust) per conditioning intent.
3. ``run_probing_ceiling`` — FiLM vs heuristics: selection quality vs probing
   cost, and application performance vs realized congestion level (the ceiling).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.shortest_path import ShortestPathSelector
from src.baselines.widest_path import WidestPathSelector
from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import load_conditional_scoring_agent
from src.rl.reward_profiles import RewardProfile, get_profile
from src.simulation.evaluation_env import FLAT_GLOBAL_DIM, RewardWeights
from src.simulation.run_context import compute_action_dim, load_run_context, make_env

# Held-out evaluation window: the last 14 days (training uses the first 14).
EVAL_HOURS: List[int] = list(range(14 * 24, 28 * 24))
MAX_EVAL_PAIRS = 32

# The intents used in Chapter 6. These MUST match the profiles the conditional
# models were trained on (``get_conditional_training_profiles()`` defaults to the
# DISTINCTIVE set) -- otherwise the agent is evaluated on weight vectors it never
# saw. We use the four *selection-relevant* distinctive intents and deliberately
# exclude ``probe_minimal``/``probe_averse``: those differ only in ``w_probe``,
# which is path-independent for a single-path selector and so cannot change the
# chosen path (they are degenerate for an intent-conditioning ablation and are
# instead exercised by the probing-overhead study, Section~probing).
INTENT_PROFILES: List[str] = [
    "bandwidth_max",
    "delay_averse",
    "loss_averse",
    "balanced_extreme",
]
INTENT_LABELS: Dict[str, str] = {
    "bandwidth_max": "Throughput",
    "delay_averse": "Low-Latency",
    "loss_averse": "Low-Loss",
    "balanced_extreme": "Balanced",
}

# Conditioning DQN methods and their checkpoint filenames. ``conditional_concat``
# is the naive value-only-concat baseline (intent in the value stream only) that
# the chapter contrasts with FiLM. The legacy 2-stream concat architecture
# (``dqn_conditional_concat_model.pth``) is intentionally not evaluated here: it is
# statistically tied with FiLM on quality but adds no principle, so the chapter
# reports the clean three-way Flat -> value-only concat -> FiLM. The legacy trainer
# and agent class remain available for anyone who wants that variant.
CONDITIONAL_CHECKPOINTS: Dict[str, str] = {
    "conditional_film": "dqn_conditional_scoring_model.pth",
    "conditional_concat": "dqn_conditional_value_concat_model.pth",
}
FLAT_CHECKPOINT = "dqn_model.pth"

# Canonical (neutral) trust weights for reporting intrinsic path trust, so trust
# is comparable across intents rather than re-scaled by each profile.
_TRUST_W3 = 0.5
_TRUST_W4 = 0.5

METHOD_LABELS: Dict[str, str] = {
    "flat_dqn": "Flat DQN",
    "conditional_concat": "Conditional-Concat",
    "conditional_film": "Conditional-FiLM",
    "shortest_path": "Shortest-Path",
    "widest_path": "Widest-Path",
    "lowest_latency": "Lowest-Latency",
    "random": "Random",
}


# --------------------------------------------------------------------------- io
def _load_checkpoint(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _intrinsic_trust(latency_ms: float, loss_rate: float) -> float:
    delay = min(100.0, float(latency_ms)) / 100.0
    return max(0.0, min(1.0, 1.0 - (_TRUST_W3 * float(loss_rate) + _TRUST_W4 * delay)))


def _profiles() -> List[RewardProfile]:
    return [get_profile(name) for name in INTENT_PROFILES]


def _eval_pairs(
    pair_pool: Sequence[Tuple[int, int]], max_pairs: int
) -> List[Tuple[int, int]]:
    return list(pair_pool[: min(len(pair_pool), max_pairs)])


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------- agent loading
def load_conditional_agents(run_path: Path) -> Dict[str, Any]:
    """Load whichever conditioning DQN checkpoints exist (FiLM and/or concat)."""
    agents: Dict[str, Any] = {}
    for method, fname in CONDITIONAL_CHECKPOINTS.items():
        ckpt = _load_checkpoint(run_path / fname)
        if ckpt:
            agent = load_conditional_scoring_agent(ckpt)
            agent.epsilon = 0.0
            # Deterministic greedy evaluation: disable dropout so path selection
            # is a pure function of state (otherwise divergence is dropout noise).
            agent.q_network.eval()
            agents[method] = agent
    return agents


def load_flat_agent(run_path: Path, action_dim: int) -> Optional[Any]:
    ckpt = _load_checkpoint(run_path / FLAT_CHECKPOINT)
    if not ckpt:
        return None
    cfg = ckpt.get("config") or EnhancedDQNConfig()
    agent = EnhancedDQNAgent(
        int(ckpt.get("state_dim", FLAT_GLOBAL_DIM)),
        int(ckpt.get("action_dim", action_dim)),
        cfg,
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.epsilon = 0.0
    agent.q_network.eval()  # deterministic greedy eval (no dropout noise)
    return agent


# ---------------------------------------------------------- selection primitives
def _select_conditional(agent: Any, env: Any) -> Optional[int]:
    """Greedy path index for a conditioning (FiLM/concat) agent, or None if no paths."""
    state = env.observe_scoring_conditional()
    if state["paths"].shape[0] == 0:
        return None
    action = int(agent.act(state, evaluate=True))
    n = len(env.available_paths)
    if action >= n:
        action = 0
    return action


def _select_flat(agent: Any, env: Any, action_dim: int) -> Optional[int]:
    n = len(env.available_paths)
    if n == 0:
        return None
    state = env.observe_flat()
    mask = env.action_mask(action_dim)
    action = int(agent.act(state, action_mask=mask))
    if action >= n:
        valid = np.where(mask)[0]
        action = int(valid[0]) if len(valid) else 0
    return action


def _select_baseline(
    method: str, selector: Any, env: Any, pair: Tuple[int, int]
) -> Tuple[Optional[int], float, int]:
    """Probe every path (heuristic behaviour), then select. Returns (action, probe_cost_ms, n_probes)."""
    n = len(env.available_paths)
    if n == 0:
        return None, 0.0, 0
    step_probe_cost = 0.0
    metrics_list: List[Dict[str, Any]] = []
    full_probe = method == "widest_path"
    for path_idx in range(n):
        m = (
            env.probe_path_full(path_idx)
            if full_probe
            else env.probe_path_latency(path_idx)
        )
        step_probe_cost += env.last_probe_cost_ms
        metrics_list.append(m)
    if method == "random":
        action = int(np.random.default_rng(hash((pair, n)) & 0xFFFFFFFF).integers(0, n))
    else:
        flow_stub = {"src": int(pair[0]), "dst": int(pair[1])}
        action = int(
            selector.select_path(
                env.available_paths,
                metrics_list,
                flow_stub,
                np.zeros(1, dtype=np.float32),
            )
        )
    if action >= n:
        action = 0
    return action, step_probe_cost, n


# --------------------------------------------------------------------- ablation
def run_ablation(
    run_path: Path,
    out_dir: Path,
    *,
    max_pairs: int = MAX_EVAL_PAIRS,
    progress: Optional[Callable[[str], None]] = None,
    run_context: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Table 6.1: mean intent-weighted reward + behavioral divergence per method."""
    log = progress or (lambda _m: None)
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _goodput_cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs)

    agents: Dict[str, Any] = {}
    if (run_path / FLAT_CHECKPOINT).is_file():
        action_dim = compute_action_dim_from_run(run_path, path_store)
        flat = load_flat_agent(run_path, action_dim)
    else:
        action_dim = 1
        flat = None
    if flat is not None:
        agents["flat_dqn"] = flat
    agents.update(load_conditional_agents(run_path))
    if not agents:
        raise FileNotFoundError(
            f"No ablation checkpoints found under {run_path} "
            f"(need any of {FLAT_CHECKPOINT}, {', '.join(CONDITIONAL_CHECKPOINTS.values())})."
        )

    method_order = [
        m
        for m in (
            "flat_dqn",
            "conditional_concat",
            "conditional_film",
        )
        if m in agents
    ]
    profiles = _profiles()
    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=RewardWeights(),
    )

    reward_rows: List[Dict[str, Any]] = []
    divergence_rows: List[Dict[str, Any]] = []

    for method in method_order:
        agent = agents[method]
        # Per (pair, hour) context: choice + per-profile reward.
        # Accumulators per profile.
        prof_reward: Dict[str, List[float]] = {p.name: [] for p in profiles}
        prof_goodput: Dict[str, List[float]] = {p.name: [] for p in profiles}
        prof_latency: Dict[str, List[float]] = {p.name: [] for p in profiles}
        prof_trust: Dict[str, List[float]] = {p.name: [] for p in profiles}
        divergences: List[float] = []
        entropies: List[float] = []

        for hour_idx in EVAL_HOURS:
            for pair in eval_pairs:
                choices: List[int] = []
                context_ok = True
                for profile in profiles:
                    env.reward_weights = profile.weights
                    env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
                    n = len(env.available_paths)
                    if n == 0:
                        context_ok = False
                        break
                    if method == "flat_dqn":
                        action = _select_flat(agent, env, action_dim)
                    else:
                        action = _select_conditional(agent, env)
                    if action is None:
                        context_ok = False
                        break
                    reward, _done, info = env.apply_action(action, probe="full")
                    pm = info["path_metrics"]
                    prof_reward[profile.name].append(float(reward))
                    bw = float(pm.get("bandwidth_mbps") or 0.0)
                    prof_goodput[profile.name].append(bw)
                    prof_latency[profile.name].append(float(pm.get("latency_ms", 50.0)))
                    prof_trust[profile.name].append(
                        _intrinsic_trust(
                            pm.get("latency_ms", 50.0), pm.get("loss_rate", 0.0)
                        )
                    )
                    choices.append(int(action))
                if not context_ok or not choices:
                    continue
                n_paths_ctx = max(2, len(env.available_paths))
                distinct = len(set(choices))
                denom = min(len(profiles), n_paths_ctx) - 1
                divergences.append((distinct - 1) / denom if denom > 0 else 0.0)
                entropies.append(_choice_entropy(choices))

        for profile in profiles:
            rewards = prof_reward[profile.name]
            n = len(rewards)
            reward_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "profile": profile.name,
                    "profile_label": INTENT_LABELS.get(profile.name, profile.name),
                    "n_selections": n,
                    "reward_mean": float(np.mean(rewards)) if n else float("nan"),
                    "reward_std": float(np.std(rewards)) if n else float("nan"),
                    "goodput_mean_mbps": (
                        float(np.mean(prof_goodput[profile.name]))
                        if n
                        else float("nan")
                    ),
                    "latency_mean_ms": (
                        float(np.mean(prof_latency[profile.name]))
                        if n
                        else float("nan")
                    ),
                    "trust_mean": (
                        float(np.mean(prof_trust[profile.name])) if n else float("nan")
                    ),
                }
            )
        divergence_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "behavioral_divergence": (
                    float(np.mean(divergences)) if divergences else 0.0
                ),
                "choice_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
                "n_contexts": len(divergences),
            }
        )
        log(f"  ablation: {method} done ({len(divergences)} contexts)")

    reward_csv = out_dir / "ablation_reward_matrix.csv"
    _write_csv(
        reward_csv,
        [
            "method",
            "method_label",
            "profile",
            "profile_label",
            "n_selections",
            "reward_mean",
            "reward_std",
            "goodput_mean_mbps",
            "latency_mean_ms",
            "trust_mean",
        ],
        reward_rows,
    )
    div_csv = out_dir / "ablation_behavioral_divergence.csv"
    _write_csv(
        div_csv,
        [
            "method",
            "method_label",
            "behavioral_divergence",
            "choice_entropy_mean",
            "n_contexts",
        ],
        divergence_rows,
    )
    table_path = out_dir / "table_6_1.tex"
    _write_ablation_table(
        table_path, method_order, profiles, reward_rows, divergence_rows
    )

    return {
        "reward_csv": str(reward_csv),
        "divergence_csv": str(div_csv),
        "table_tex": str(table_path),
        "methods": method_order,
        "reward_rows": reward_rows,
        "divergence_rows": divergence_rows,
    }


def _choice_entropy(choices: Sequence[int]) -> float:
    if not choices:
        return 0.0
    _, counts = np.unique(np.asarray(choices), return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def compute_action_dim_from_run(run_path: Path, path_store: Any) -> int:
    import json

    with open(run_path / "selected_pair.json") as f:
        selected_pair = json.load(f)
    return compute_action_dim(path_store, selected_pair)


def _write_ablation_table(
    path: Path,
    methods: Sequence[str],
    profiles: Sequence[RewardProfile],
    reward_rows: Sequence[Dict[str, Any]],
    divergence_rows: Sequence[Dict[str, Any]],
) -> None:
    reward_lookup = {(r["method"], r["profile"]): r["reward_mean"] for r in reward_rows}
    div_lookup = {r["method"]: r["behavioral_divergence"] for r in divergence_rows}
    col_spec = "l" + "c" * len(profiles) + "c"
    header_cells = " & ".join(INTENT_LABELS.get(p.name, p.name) for p in profiles)
    lines = [
        "% Auto-generated by src/pipeline/chapter6_eval.py -- do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        f"Method & {header_cells} & Adaptivity \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = []
        for p in profiles:
            val = reward_lookup.get((method, p.name), float("nan"))
            cells.append("--" if val != val else f"{val:.3f}")
        div = div_lookup.get(method, float("nan"))
        div_str = "--" if div != div else f"{div:.3f}"
        label = METHOD_LABELS.get(method, method)
        lines.append(f"{label} & " + " & ".join(cells) + f" & {div_str} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ------------------------------------------------------------- intent alignment
def run_intent_alignment(
    run_path: Path,
    out_dir: Path,
    *,
    max_pairs: int = MAX_EVAL_PAIRS,
    progress: Optional[Callable[[str], None]] = None,
    run_context: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Figure 6.1: R(intent_told, intent_scored) + chosen-path metric distributions (FiLM)."""
    log = progress or (lambda _m: None)
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _goodput_cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs)

    agents = load_conditional_agents(run_path)
    agent = agents.get("conditional_film")
    if agent is None:
        raise FileNotFoundError(
            f"FiLM checkpoint {CONDITIONAL_CHECKPOINTS['conditional_film']} not found under {run_path}."
        )

    profiles = _profiles()
    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=RewardWeights(),
    )

    # R[i][j] accumulators: told intent i, scored under intent j.
    matrix_acc: Dict[str, Dict[str, List[float]]] = {
        pi.name: {pj.name: [] for pj in profiles} for pi in profiles
    }
    # Selection metrics per conditioning intent.
    metric_rows: List[Dict[str, Any]] = []

    for told in profiles:
        for hour_idx in EVAL_HOURS:
            for pair in eval_pairs:
                env.reward_weights = told.weights
                env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
                if len(env.available_paths) == 0:
                    continue
                action = _select_conditional(agent, env)
                if action is None:
                    continue
                _reward, _done, info = env.apply_action(action, probe="full")
                pm = info["path_metrics"]
                max_bw = info.get("max_available_path_bw")
                probe_cost = float(info.get("step_probe_cost_ms", 0.0))
                # Score the same chosen path under every intent j.
                for scored in profiles:
                    r_j = env.compute_reward(
                        pm,
                        max_possible_bw=max_bw,
                        probe_cost_ms=probe_cost,
                        num_probes_in_step=1,
                        weights=scored.weights.to_dict(),
                    )
                    matrix_acc[told.name][scored.name].append(float(r_j))
                metric_rows.append(
                    {
                        "intent": told.name,
                        "intent_label": INTENT_LABELS.get(told.name, told.name),
                        "latency_ms": float(pm.get("latency_ms", 50.0)),
                        "bandwidth_mbps": float(pm.get("bandwidth_mbps") or 0.0),
                        "trust": _intrinsic_trust(
                            pm.get("latency_ms", 50.0), pm.get("loss_rate", 0.0)
                        ),
                    }
                )
        log(f"  intent alignment: told={told.name} done")

    matrix_rows: List[Dict[str, Any]] = []
    for told in profiles:
        for scored in profiles:
            vals = matrix_acc[told.name][scored.name]
            matrix_rows.append(
                {
                    "intent_told": told.name,
                    "intent_told_label": INTENT_LABELS.get(told.name, told.name),
                    "intent_scored": scored.name,
                    "intent_scored_label": INTENT_LABELS.get(scored.name, scored.name),
                    "reward_mean": float(np.mean(vals)) if vals else float("nan"),
                    "n": len(vals),
                }
            )

    matrix_csv = out_dir / "intent_reward_matrix.csv"
    _write_csv(
        matrix_csv,
        [
            "intent_told",
            "intent_told_label",
            "intent_scored",
            "intent_scored_label",
            "reward_mean",
            "n",
        ],
        matrix_rows,
    )
    metrics_csv = out_dir / "intent_selection_metrics.csv"
    _write_csv(
        metrics_csv,
        ["intent", "intent_label", "latency_ms", "bandwidth_mbps", "trust"],
        metric_rows,
    )

    return {
        "matrix_csv": str(matrix_csv),
        "metrics_csv": str(metrics_csv),
        "matrix_rows": matrix_rows,
    }


# --------------------------------------------------------- probing / ceiling
def _baseline_selectors() -> Dict[str, Any]:
    return {
        "shortest_path": ShortestPathSelector(),
        "widest_path": WidestPathSelector(),
        "lowest_latency": LowestLatencySelector(),
        "random": None,  # handled inline
    }


def run_probing_ceiling(
    run_path: Path,
    out_dir: Path,
    *,
    max_pairs: int = MAX_EVAL_PAIRS,
    n_congestion_bins: int = 6,
    progress: Optional[Callable[[str], None]] = None,
    run_context: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Figures 6.2 & 6.3: quality vs probe cost, and QoE vs congestion (single-path ceiling)."""
    log = progress or (lambda _m: None)
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _goodput_cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs)

    film = load_conditional_agents(run_path).get("conditional_film")
    if film is None:
        raise FileNotFoundError(
            f"FiLM checkpoint {CONDITIONAL_CHECKPOINTS['conditional_film']} not found under {run_path}."
        )
    baselines = _baseline_selectors()
    method_order = [
        "conditional_film",
        "shortest_path",
        "widest_path",
        "lowest_latency",
        "random",
    ]

    # Evaluate under a Balanced intent so goodput/reward are comparable to
    # heuristics. Use ``balanced_extreme`` (in the conditional training set), not
    # the untrained ``balanced`` profile, so the FiLM agent is fed a weight vector
    # it actually learned to condition on.
    balanced = get_profile("balanced_extreme")
    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=balanced.weights,
    )

    # Per-selection records for congestion binning.
    per_sel: Dict[str, List[Dict[str, float]]] = {m: [] for m in method_order}
    # Per-method probe accounting.
    probe_acc: Dict[str, Dict[str, float]] = {
        m: {"probe_cost_ms": 0.0, "n_probes": 0.0, "n_sel": 0.0} for m in method_order
    }

    for method in method_order:
        for hour_idx in EVAL_HOURS:
            for pair in eval_pairs:
                env.reward_weights = balanced.weights
                env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
                n = len(env.available_paths)
                if n == 0:
                    continue
                # Mean utilization across the flow's candidate paths at this hour.
                congestion = float(env.observe_flat()[2])
                if method == "conditional_film":
                    action = _select_conditional(film, env)
                    if action is None:
                        continue
                    reward, _done, info = env.apply_action(action, probe="full")
                    step_probe_cost = float(info.get("step_probe_cost_ms", 0.0))
                    n_probes = 1
                else:
                    action, step_probe_cost, n_probes = _select_baseline(
                        method, baselines[method], env, pair
                    )
                    if action is None:
                        continue
                    _obs, reward, _done, info = env.step(
                        action,
                        step_probe_cost_ms=step_probe_cost,
                        num_probes_in_step=max(1, n_probes),
                    )
                pm = info["path_metrics"]
                goodput = float(pm.get("bandwidth_mbps") or 0.0)
                per_sel[method].append(
                    {
                        "congestion": congestion,
                        "goodput": goodput,
                        "reward": float(reward),
                    }
                )
                probe_acc[method]["probe_cost_ms"] += step_probe_cost
                probe_acc[method]["n_probes"] += n_probes
                probe_acc[method]["n_sel"] += 1
        log(f"  probing/ceiling: {method} done")

    # --- Figure 6.2 data: quality vs probe cost (method-level) ---
    quality_rows: List[Dict[str, Any]] = []
    for method in method_order:
        acc = probe_acc[method]
        n_sel = acc["n_sel"] or 1.0
        recs = per_sel[method]
        goodputs = [r["goodput"] for r in recs]
        rewards = [r["reward"] for r in recs]
        quality_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "probe_cost_per_selection_ms": acc["probe_cost_ms"] / n_sel,
                "probes_per_selection": acc["n_probes"] / n_sel,
                "goodput_mean_mbps": (
                    float(np.mean(goodputs)) if goodputs else float("nan")
                ),
                "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
                "n_selections": int(acc["n_sel"]),
            }
        )
    quality_csv = out_dir / "probing_quality.csv"
    _write_csv(
        quality_csv,
        [
            "method",
            "method_label",
            "probe_cost_per_selection_ms",
            "probes_per_selection",
            "goodput_mean_mbps",
            "reward_mean",
            "n_selections",
        ],
        quality_rows,
    )

    # --- Figure 6.3 data: QoE vs congestion bin ---
    all_congestion = np.array(
        [r["congestion"] for recs in per_sel.values() for r in recs], dtype=np.float64
    )
    ceiling_rows: List[Dict[str, Any]] = []
    if all_congestion.size:
        edges = np.quantile(all_congestion, np.linspace(0, 1, n_congestion_bins + 1))
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([all_congestion.min(), all_congestion.max() + 1e-9])
        for method in method_order:
            recs = per_sel[method]
            if not recs:
                continue
            cong = np.array([r["congestion"] for r in recs])
            good = np.array([r["goodput"] for r in recs])
            rew = np.array([r["reward"] for r in recs])
            bin_idx = np.clip(np.digitize(cong, edges[1:-1]), 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                sel = bin_idx == b
                if not sel.any():
                    continue
                lo, hi = edges[b], edges[b + 1]
                ceiling_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "congestion_bin": b,
                        "congestion_lo": float(lo),
                        "congestion_hi": float(hi),
                        "congestion_mid": float((lo + hi) / 2.0),
                        "goodput_mean_mbps": float(good[sel].mean()),
                        "reward_mean": float(rew[sel].mean()),
                        "n": int(sel.sum()),
                    }
                )
    ceiling_csv = out_dir / "ceiling_by_congestion.csv"
    _write_csv(
        ceiling_csv,
        [
            "method",
            "method_label",
            "congestion_bin",
            "congestion_lo",
            "congestion_hi",
            "congestion_mid",
            "goodput_mean_mbps",
            "reward_mean",
            "n",
        ],
        ceiling_rows,
    )

    return {
        "quality_csv": str(quality_csv),
        "ceiling_csv": str(ceiling_csv),
        "quality_rows": quality_rows,
        "ceiling_rows": ceiling_rows,
    }
