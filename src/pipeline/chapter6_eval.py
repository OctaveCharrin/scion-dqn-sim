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

from src.baselines.ecmp import ECMPSelector
from src.baselines.lowest_latency import LowestLatencySelector
from src.baselines.scion_default import SCIONDefaultSelector
from src.baselines.shortest_path import ShortestPathSelector
from src.baselines.widest_path import WidestPathSelector
from src.rl.dqn_agent_enhanced import EnhancedDQNAgent, EnhancedDQNConfig
from src.rl.dqn_agent_scoring_conditional import load_conditional_scoring_agent
from src.rl.reward_profiles import RewardProfile, get_profile
from src.simulation.evaluation_env import (
    FLAT_GLOBAL_DIM,
    RewardWeights,
    encode_reward_weights_for_conditional,
)
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

# Conditioning DQN methods and their checkpoint filenames. The three differ only
# in *where* the intent vector enters the dueling head:
#   conditional_concat         -- value stream only. Cannot re-rank by the
#                                 cancellation argument (a path-independent term
#                                 drops out of the argmax); the structural claim
#                                 is unit-tested in tests/test_chapter6_eval.py.
#   conditional_concat_2stream -- both streams, so the advantage stream sees the
#                                 intent alongside each path's features. This is
#                                 the honest control: it *can* re-rank. It was
#                                 previously excluded here, which left the
#                                 ablation comparing FiLM only against a variant
#                                 that was incapable of competing.
#   conditional_film           -- FiLM modulation of the per-path features.
CONDITIONAL_CHECKPOINTS: Dict[str, str] = {
    "conditional_film": "dqn_conditional_scoring_model.pth",
    "conditional_concat": "dqn_conditional_value_concat_model.pth",
    "conditional_concat_2stream": "dqn_conditional_concat_model.pth",
}
# Which conditional variant the single-agent studies (intent alignment, probing,
# ceiling) profile. Kept at the historical default so existing callers are
# unaffected; pass agent_key/--agent to profile the variant the thesis ships.
DEFAULT_CONDITIONAL_AGENT = "conditional_film"
FLAT_CHECKPOINT = "dqn_model.pth"
# Unconditioned path-scoring agent: the architectural control that separates
# "conditioning helps" from "per-path scoring helps".
SCORING_CHECKPOINT = "dqn_scoring_enhanced_model.pth"
# Per-context argmax of the true reward under the intent being scored. Not a
# deployable policy -- it is the per-intent upper bound that turns the ablation
# table into a normalized optimality gap.
ORACLE_METHOD = "oracle"

# Canonical (neutral) trust weights for reporting intrinsic path trust, so trust
# is comparable across intents rather than re-scaled by each profile.
_TRUST_W3 = 0.5
_TRUST_W4 = 0.5

METHOD_LABELS: Dict[str, str] = {
    "flat_dqn": "Flat DQN",
    "scoring_enhanced": "Scoring DQN (uncond.)",
    "conditional_concat": "Value-Concat",
    "conditional_concat_2stream": "Two-Stream-Concat",
    "conditional_film": "Conditional-FiLM",
    "oracle": "Oracle",
    "shortest_path": "Shortest-Path",
    "widest_path": "Widest-Path",
    "lowest_latency": "Lowest-Latency",
    "ecmp": "ECMP",
    "scion_default": "SCION-Default",
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


def _profiles(names: Optional[Sequence[str]] = None) -> List[RewardProfile]:
    return [get_profile(name) for name in (names or INTENT_PROFILES)]


def _eval_pairs(
    pair_pool: Sequence[Tuple[int, int]],
    max_pairs: int,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Evaluation pairs: an explicit list if given (e.g. a held-out set), else the
    first ``max_pairs`` of the pool."""
    if pairs:
        chosen = [(int(a), int(b)) for a, b in pairs]
        return chosen[: min(len(chosen), max_pairs)] if max_pairs else chosen
    return list(pair_pool[: min(len(pair_pool), max_pairs)])


def _eval_hours(hour_stride: int = 1) -> List[int]:
    stride = max(1, int(hour_stride))
    return EVAL_HOURS[::stride]


def _full_probe_cost_ms(env: Any, hop_count: int) -> float:
    """Cost the environment would charge for a full bandwidth probe of one path."""
    return float(env.bandwidth_probe_cost_ms) + float(
        env.per_hop_full_probe_cost_ms
    ) * int(hop_count)


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


def _require_conditional_agent(run_path: Path, agent_key: str) -> Any:
    """Load one conditional agent by key, failing loudly if it is unknown or absent."""
    if agent_key not in CONDITIONAL_CHECKPOINTS:
        known = ", ".join(sorted(CONDITIONAL_CHECKPOINTS))
        raise KeyError(f"Unknown conditional agent {agent_key!r}. Known agents: {known}.")
    agent = load_conditional_agents(run_path).get(agent_key)
    if agent is None:
        raise FileNotFoundError(
            f"Checkpoint {CONDITIONAL_CHECKPOINTS[agent_key]} for {agent_key!r} "
            f"not found under {run_path}."
        )
    return agent


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


def load_scoring_agent(run_path: Path) -> Optional[Any]:
    """Unconditioned enhanced path-scoring agent (the architectural control)."""
    ckpt = _load_checkpoint(run_path / SCORING_CHECKPOINT)
    if not ckpt:
        return None
    from src.rl.dqn_agent_scoring_enhanced import EnhancedPathScoringDQNAgent
    from src.simulation.evaluation_env import PATH_FEATURE_DIM, SCORING_GLOBAL_DIM

    agent = EnhancedPathScoringDQNAgent(
        global_dim=int(ckpt.get("global_dim", SCORING_GLOBAL_DIM)),
        path_dim=int(ckpt.get("path_dim", PATH_FEATURE_DIM)),
        config=ckpt.get("config") or EnhancedDQNConfig(),
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.epsilon = 0.0
    agent.q_network.eval()
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
        flow_stub = {
            "src": int(pair[0]),
            "dst": int(pair[1]),
            "source_as": int(pair[0]),
            "destination_as": int(pair[1]),
            "flow_id": 0,
        }
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
    profiles: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    hour_stride: int = 1,
    include_oracle: bool = True,
) -> Dict[str, Any]:
    """Table 6.1: mean intent-weighted reward + behavioral divergence per method.

    Every method and every intent is scored on the *same* decision contexts: the
    context is established once per (pair, hour) by a single ``reset``, and each
    policy is then scored with ``evaluate_action``, which does not advance the
    clock. That makes the per-context reward arrays paired by construction (they
    are persisted for the significance test) and is what makes evaluating six
    methods affordable.
    """
    log = progress or (lambda _m: None)
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _goodput_cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs, pairs)
    eval_hours = _eval_hours(hour_stride)

    agents: Dict[str, Any] = {}
    if (run_path / FLAT_CHECKPOINT).is_file():
        action_dim = compute_action_dim_from_run(run_path, path_store)
        flat = load_flat_agent(run_path, action_dim)
    else:
        action_dim = 1
        flat = None
    if flat is not None:
        agents["flat_dqn"] = flat
    scoring = load_scoring_agent(run_path)
    if scoring is not None:
        agents["scoring_enhanced"] = scoring
    agents.update(load_conditional_agents(run_path))
    if not agents:
        raise FileNotFoundError(
            f"No ablation checkpoints found under {run_path} "
            f"(need any of {FLAT_CHECKPOINT}, {SCORING_CHECKPOINT}, "
            f"{', '.join(CONDITIONAL_CHECKPOINTS.values())})."
        )

    method_order = [
        m
        for m in (
            "flat_dqn",
            "scoring_enhanced",
            "conditional_concat",
            "conditional_concat_2stream",
            "conditional_film",
        )
        if m in agents
    ]
    if include_oracle:
        method_order.append(ORACLE_METHOD)
    profile_list = _profiles(profiles)
    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=RewardWeights(),
    )

    # method -> profile -> list of per-context values, in a single shared order.
    def _acc() -> Dict[str, Dict[str, List[float]]]:
        return {m: {p.name: [] for p in profile_list} for m in method_order}

    acc_reward = _acc()
    acc_goodput = _acc()
    acc_latency = _acc()
    acc_trust = _acc()
    acc_hops = _acc()
    acc_choice: Dict[str, Dict[str, List[int]]] = {
        m: {p.name: [] for p in profile_list} for m in method_order
    }
    n_paths_seen: List[int] = []
    n_contexts = 0

    for hour_idx in eval_hours:
        for pair in eval_pairs:
            env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
            n = len(env.available_paths)
            if n == 0:
                continue
            # Metrics and the reward normalizer are intent-independent, so they
            # are computed once and shared by every method and every intent.
            pms = env.path_metrics_snapshot()
            max_bw = max((float(m.get("bandwidth_mbps") or 0.0) for m in pms), default=0.0)
            probe_costs = [
                _full_probe_cost_ms(env, m.get("hop_count", 1)) for m in pms
            ]
            n_paths_seen.append(n)
            n_contexts += 1

            for profile in profile_list:
                env.reward_weights = profile.weights
                # The per-path feature block depends on the intent (its trust
                # column uses w3/w4), so it is rebuilt per intent -- but shared
                # across the scoring/conditional agents.
                obs = env.observe_scoring()
                cond_obs = None
                flat_state = None

                for method in method_order:
                    if method == ORACLE_METHOD:
                        best_i, best_r = 0, -np.inf
                        for i, pm in enumerate(pms):
                            r = env.compute_reward(
                                pm,
                                max_possible_bw=max_bw,
                                probe_cost_ms=probe_costs[i],
                                num_probes_in_step=1,
                            )
                            if r > best_r:
                                best_i, best_r = i, r
                        action, reward = best_i, float(best_r)
                    else:
                        if method == "flat_dqn":
                            if flat_state is None:
                                flat_state = (
                                    env.observe_flat(),
                                    env.action_mask(action_dim),
                                )
                            action = int(
                                agents[method].act(
                                    flat_state[0], action_mask=flat_state[1]
                                )
                            )
                            if action >= n:
                                valid = np.where(flat_state[1])[0]
                                action = int(valid[0]) if len(valid) else 0
                        elif method == "scoring_enhanced":
                            action = int(agents[method].act(obs, evaluate=True))
                        else:
                            if cond_obs is None:
                                cond_obs = {
                                    "global": np.concatenate(
                                        [
                                            obs["global"],
                                            encode_reward_weights_for_conditional(
                                                env.reward_weights
                                            ),
                                        ]
                                    ).astype(np.float32),
                                    "paths": obs["paths"],
                                }
                            action = int(agents[method].act(cond_obs, evaluate=True))
                        if action >= n:
                            action = 0
                        reward = env.compute_reward(
                            pms[action],
                            max_possible_bw=max_bw,
                            probe_cost_ms=probe_costs[action],
                            num_probes_in_step=1,
                        )

                    pm = pms[action]
                    name = profile.name
                    acc_reward[method][name].append(float(reward))
                    acc_goodput[method][name].append(float(pm.get("bandwidth_mbps") or 0.0))
                    acc_latency[method][name].append(float(pm.get("latency_ms", 50.0)))
                    acc_trust[method][name].append(
                        _intrinsic_trust(
                            pm.get("latency_ms", 50.0), pm.get("loss_rate", 0.0)
                        )
                    )
                    acc_hops[method][name].append(float(pm.get("hop_count", 1)))
                    acc_choice[method][name].append(int(action))

        if n_contexts and n_contexts % (24 * max(1, len(eval_pairs))) == 0:
            log(f"  ablation: hour {hour_idx} ({n_contexts} contexts so far)")

    reward_rows: List[Dict[str, Any]] = []
    divergence_rows: List[Dict[str, Any]] = []
    oracle_mean = {
        p.name: (
            float(np.mean(acc_reward[ORACLE_METHOD][p.name]))
            if include_oracle and acc_reward[ORACLE_METHOD][p.name]
            else float("nan")
        )
        for p in profile_list
    }
    mean_n_paths = float(np.mean(n_paths_seen)) if n_paths_seen else float("nan")

    for method in method_order:
        for profile in profile_list:
            name = profile.name
            rewards = acc_reward[method][name]
            n = len(rewards)
            mean_r = float(np.mean(rewards)) if n else float("nan")
            ref = oracle_mean.get(name, float("nan"))
            reward_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "profile": name,
                    "profile_label": INTENT_LABELS.get(name, name),
                    "n_selections": n,
                    "reward_mean": mean_r,
                    "reward_std": float(np.std(rewards)) if n else float("nan"),
                    "goodput_mean_mbps": (
                        float(np.mean(acc_goodput[method][name])) if n else float("nan")
                    ),
                    "latency_mean_ms": (
                        float(np.mean(acc_latency[method][name])) if n else float("nan")
                    ),
                    "trust_mean": (
                        float(np.mean(acc_trust[method][name])) if n else float("nan")
                    ),
                    "hops_mean": (
                        float(np.mean(acc_hops[method][name])) if n else float("nan")
                    ),
                    "n_paths_mean": mean_n_paths,
                    "optimality_gap": (
                        float(ref - mean_r) if ref == ref and mean_r == mean_r else float("nan")
                    ),
                }
            )
        # Behavioral divergence: how much the chosen path moves with the intent,
        # over contexts shared by every method.
        divergences: List[float] = []
        entropies: List[float] = []
        for ctx_i in range(n_contexts):
            choices = [acc_choice[method][p.name][ctx_i] for p in profile_list]
            n_paths_ctx = max(2, n_paths_seen[ctx_i])
            denom = min(len(profile_list), n_paths_ctx) - 1
            divergences.append((len(set(choices)) - 1) / denom if denom > 0 else 0.0)
            entropies.append(_choice_entropy(choices))
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
        log(f"  ablation: {method} done ({n_contexts} contexts)")

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
            "hops_mean",
            "n_paths_mean",
            "optimality_gap",
        ],
        reward_rows,
    )
    # Paired per-context rewards for the significance test (task 6.4). Contexts
    # are in identical order for every (method, profile) key.
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "ablation_per_context_rewards.npz",
        **{
            f"{m}|{p.name}": np.asarray(acc_reward[m][p.name], dtype=np.float32)
            for m in method_order
            for p in profile_list
        },
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
        table_path, method_order, profile_list, reward_rows, divergence_rows
    )

    return {
        "reward_csv": str(reward_csv),
        "divergence_csv": str(div_csv),
        "table_tex": str(table_path),
        "per_context_npz": str(out_dir / "ablation_per_context_rewards.npz"),
        "methods": method_order,
        "profiles": [p.name for p in profile_list],
        "n_contexts": n_contexts,
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
    agent_key: str = DEFAULT_CONDITIONAL_AGENT,
) -> Dict[str, Any]:
    """Figure 6.1: R(intent_told, intent_scored) + chosen-path metric distributions.

    ``agent_key`` selects which conditional agent is profiled; it must be a key of
    ``CONDITIONAL_CHECKPOINTS``. The chapter reports whichever variant it ships.
    """
    log = progress or (lambda _m: None)
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _goodput_cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs)

    agent = _require_conditional_agent(run_path, agent_key)

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
                lat_ms = float(pm.get("latency_ms", 50.0))
                loss = float(pm.get("loss_rate", 0.0))
                w = told.weights
                metric_rows.append(
                    {
                        "intent": told.name,
                        "intent_label": INTENT_LABELS.get(told.name, told.name),
                        "latency_ms": lat_ms,
                        "bandwidth_mbps": float(pm.get("bandwidth_mbps") or 0.0),
                        "trust": _intrinsic_trust(lat_ms, loss),
                        # Trust scored under this intent's own w3/w4 rather than
                        # the canonical 0.5/0.5, which is the comparison the
                        # "Low-Loss intent -> higher trust" panel is really after.
                        "trust_own_w": max(
                            0.0,
                            min(
                                1.0,
                                1.0
                                - (w.w3 * loss + w.w4 * min(100.0, lat_ms) / 100.0),
                            ),
                        ),
                        # What the Low-Loss intent actually optimizes.
                        "loss_rate": loss,
                        "hop_count": int(pm.get("hop_count", 1)),
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
        [
            "intent",
            "intent_label",
            "latency_ms",
            "bandwidth_mbps",
            "trust",
            "trust_own_w",
            "loss_rate",
            "hop_count",
        ],
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
        "ecmp": ECMPSelector(),
        "scion_default": SCIONDefaultSelector(),
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
    profiles: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    agent_key: str = DEFAULT_CONDITIONAL_AGENT,
) -> Dict[str, Any]:
    """Figures 6.2 & 6.3: quality vs probe cost, and QoE vs congestion (single-path ceiling).

    ``profiles`` selects which intents to run the comparison under. The first one
    drives ``probing_quality.csv`` / ``ceiling_by_congestion.csv`` (the schema the
    chapter figures read); with more than one, an additional
    ``probing_quality_by_intent.csv`` carries every (method, intent) pair, which
    is what supports the stronger claim that one conditioned policy tracks the
    *per-intent* strongest heuristic.

    ``agent_key`` selects which conditional agent stands in for the learned
    selector; it must be a key of ``CONDITIONAL_CHECKPOINTS``.
    """
    log = progress or (lambda _m: None)
    ctx = run_context or load_run_context(run_path)
    topology_data, path_store, link_states, pair_pool, _goodput_cap = ctx
    eval_pairs = _eval_pairs(pair_pool, max_pairs, pairs)

    learned = _require_conditional_agent(run_path, agent_key)
    baselines = _baseline_selectors()
    method_order = [
        agent_key,
        "shortest_path",
        "widest_path",
        "lowest_latency",
        "ecmp",
        "scion_default",
        "random",
    ]

    # Default: a Balanced intent so goodput/reward are comparable to heuristics.
    # ``balanced_extreme`` is in the conditional training set, unlike the
    # ``balanced`` profile, so the agent is fed a weight vector it learned on.
    profile_list = _profiles(profiles or ["balanced_extreme"])
    primary = profile_list[0]
    env = make_env(
        topology_data,
        path_store,
        link_states,
        eval_pairs,
        episode_length=1,
        rng_seed=7,
        reward_weights=primary.weights,
    )

    # (profile, method) -> per-selection records for congestion binning.
    per_sel: Dict[Tuple[str, str], List[Dict[str, float]]] = {
        (p.name, m): [] for p in profile_list for m in method_order
    }
    probe_acc: Dict[Tuple[str, str], Dict[str, float]] = {
        (p.name, m): {"probe_cost_ms": 0.0, "n_probes": 0.0, "n_sel": 0.0}
        for p in profile_list
        for m in method_order
    }

    for profile in profile_list:
        for method in method_order:
            key = (profile.name, method)
            for hour_idx in EVAL_HOURS:
                for pair in eval_pairs:
                    env.reward_weights = profile.weights
                    env.reset(source_as=pair[0], dest_as=pair[1], hour_idx=hour_idx)
                    n = len(env.available_paths)
                    if n == 0:
                        continue
                    # Mean utilization across the flow's candidate paths this hour.
                    congestion = float(env.observe_flat()[2])
                    if method == agent_key:
                        action = _select_conditional(learned, env)
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
                    per_sel[key].append(
                        {
                            "congestion": congestion,
                            "goodput": float(pm.get("bandwidth_mbps") or 0.0),
                            "reward": float(reward),
                        }
                    )
                    probe_acc[key]["probe_cost_ms"] += step_probe_cost
                    probe_acc[key]["n_probes"] += n_probes
                    probe_acc[key]["n_sel"] += 1
            log(f"  probing/ceiling: {profile.name}/{method} done")

    # --- Figure 6.2 data: quality vs probe cost (method-level) ---
    def _quality_rows_for(profile_name: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for method in method_order:
            acc = probe_acc[(profile_name, method)]
            n_sel = acc["n_sel"] or 1.0
            recs = per_sel[(profile_name, method)]
            goodputs = [r["goodput"] for r in recs]
            rewards = [r["reward"] for r in recs]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "probe_cost_per_selection_ms": acc["probe_cost_ms"] / n_sel,
                    "probes_per_selection": acc["n_probes"] / n_sel,
                    "goodput_mean_mbps": (
                        float(np.mean(goodputs)) if goodputs else float("nan")
                    ),
                    "reward_mean": (
                        float(np.mean(rewards)) if rewards else float("nan")
                    ),
                    "n_selections": int(acc["n_sel"]),
                }
            )
        return rows

    quality_fields = [
        "method",
        "method_label",
        "probe_cost_per_selection_ms",
        "probes_per_selection",
        "goodput_mean_mbps",
        "reward_mean",
        "n_selections",
    ]
    quality_rows = _quality_rows_for(primary.name)
    quality_csv = out_dir / "probing_quality.csv"
    _write_csv(quality_csv, quality_fields, quality_rows)

    by_intent_csv = None
    if len(profile_list) > 1:
        by_intent_rows: List[Dict[str, Any]] = []
        for profile in profile_list:
            for row in _quality_rows_for(profile.name):
                by_intent_rows.append(
                    {
                        "profile": profile.name,
                        "profile_label": INTENT_LABELS.get(profile.name, profile.name),
                        **row,
                    }
                )
        by_intent_csv = out_dir / "probing_quality_by_intent.csv"
        _write_csv(
            by_intent_csv, ["profile", "profile_label"] + quality_fields, by_intent_rows
        )

    # --- Figure 6.3 data: QoE vs congestion bin ---
    all_congestion = np.array(
        [r["congestion"] for m in method_order for r in per_sel[(primary.name, m)]],
        dtype=np.float64,
    )
    ceiling_rows: List[Dict[str, Any]] = []
    if all_congestion.size:
        edges = np.quantile(all_congestion, np.linspace(0, 1, n_congestion_bins + 1))
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([all_congestion.min(), all_congestion.max() + 1e-9])
        for method in method_order:
            recs = per_sel[(primary.name, method)]
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
        "quality_by_intent_csv": str(by_intent_csv) if by_intent_csv else None,
        "profiles": [p.name for p in profile_list],
        "quality_rows": quality_rows,
        "ceiling_rows": ceiling_rows,
    }
