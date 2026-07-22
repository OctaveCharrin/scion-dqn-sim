"""Integration test for the Chapter 6 pipeline (intent conditioning + ceiling).

Fabricates a tiny in-memory run context (no BRITE / no traffic sim), trains a
Conditional-FiLM and a Conditional-Concat model for a handful of episodes, then
runs the three chapter-6 evaluations and renders the figures. Verifies the
concat trainer emits the ``dueling_concat`` architecture and that every CSV /
LaTeX / PNG artifact is produced and well-formed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.pipeline import chapter6_eval as ce
from src.pipeline import chapter6_figures as cf
from src.rl.dqn_agent_scoring_conditional import (
    CONDITIONAL_ARCH_LEGACY,
    CONDITIONAL_ARCH_VALUE_CONCAT,
    CONDITIONAL_ARCH_WEIGHT_FILM,
    ConditionalPathScoringDQNAgent,
    LegacyConditionalPathScoringDQNAgent,
    ValueConcatConditionalPathScoringDQNAgent,
    load_conditional_scoring_agent,
)
from src.rl.path_selection_train import (
    ScoringHyperparams,
    train_conditional_scoring_dqn,
)
from src.simulation.link_state_store import LinkTrafficState
from src.simulation.path_store import InMemoryPathStore

_EVAL_HOURS = [0, 1, 2]
_PAIRS = [(1, 4), (1, 5), (2, 6)]


def _make_paths(specs):
    """specs: list of (hop_count, total_latency, min_bw)."""
    paths = []
    for hop_count, lat, bw in specs:
        hops = [{"as": h} for h in range(1, hop_count + 2)]
        paths.append(
            {
                "hops": hops,
                "static_metrics": {
                    "hop_count": hop_count,
                    "total_latency": float(lat),
                    "min_bandwidth": float(bw),
                },
            }
        )
    return paths


def _build_context():
    store = InMemoryPathStore()
    # Three distinct paths per pair: a low-latency one, a high-bandwidth one,
    # and a lossy middle one, so different intents can prefer different paths.
    path_specs = [(2, 20.0, 200.0), (4, 60.0, 900.0), (3, 40.0, 400.0)]
    for src, dst in _PAIRS:
        store.set_paths(src, dst, _make_paths(path_specs))

    # Legacy hourly link states: per pair, per path dynamic metrics that vary by
    # hour (rising utilization) so congestion binning has signal.
    legacy = {}
    for hour in _EVAL_HOURS:
        u = 0.2 + 0.25 * hour  # 0.2, 0.45, 0.7
        by_pair = {}
        for src, dst in _PAIRS:
            by_pair[f"pair_{src}_{dst}"] = {
                "path_0": {
                    "latency_ms": 20.0 + 5 * hour,
                    "available_bandwidth_mbps": 180.0 * (1 - u),
                    "loss_rate": 0.01 * hour,
                    "utilization": u,
                },
                "path_1": {
                    "latency_ms": 60.0 + 5 * hour,
                    "available_bandwidth_mbps": 850.0 * (1 - u),
                    "loss_rate": 0.02 * hour,
                    "utilization": u,
                },
                "path_2": {
                    "latency_ms": 40.0 + 5 * hour,
                    "available_bandwidth_mbps": 380.0 * (1 - u),
                    "loss_rate": 0.05 * hour,
                    "utilization": u,
                },
            }
        legacy[hour] = {"by_pair": by_pair}

    link_states = LinkTrafficState(link_keys=[], hours={}, legacy_by_hour=legacy)
    pair_pool = list(_PAIRS)
    topology_data = {"nodes": [{"as_id": a} for a in range(1, 8)]}
    goodput_cap = 500.0
    return (topology_data, store, link_states, pair_pool, goodput_cap)


def _tiny_hp():
    return ScoringHyperparams(
        learning_rate=1e-3,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        buffer_size=256,
        min_buffer_size=8,
        batch_size=8,
        target_update_every=10,
        hidden_dim=32,
        n_hidden_layers=2,
        tau=0.1,
    )


def _read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    run_path = tmp_path_factory.mktemp("ch6_run")
    ctx = _build_context()
    hp = _tiny_hp()
    # Train FiLM (default), value-only concat (naive ablation), and the legacy
    # 2-stream concat for a few episodes each.
    film_stats = train_conditional_scoring_dqn(
        run_path, hp, num_episodes=4, episode_length=2, run_context=ctx, quiet=True
    )
    value_concat_stats = train_conditional_scoring_dqn(
        run_path,
        hp,
        num_episodes=4,
        episode_length=2,
        run_context=ctx,
        quiet=True,
        architecture=CONDITIONAL_ARCH_VALUE_CONCAT,
    )
    concat_stats = train_conditional_scoring_dqn(
        run_path,
        hp,
        num_episodes=4,
        episode_length=2,
        run_context=ctx,
        quiet=True,
        architecture=CONDITIONAL_ARCH_LEGACY,
    )
    return run_path, ctx, film_stats, value_concat_stats, concat_stats


def test_concat_trainer_emits_legacy_architecture(trained_run):
    run_path, _ctx, film_stats, value_concat_stats, concat_stats = trained_run
    assert film_stats["architecture"] == CONDITIONAL_ARCH_WEIGHT_FILM
    assert value_concat_stats["architecture"] == CONDITIONAL_ARCH_VALUE_CONCAT
    assert concat_stats["architecture"] == CONDITIONAL_ARCH_LEGACY
    assert (run_path / "dqn_conditional_scoring_model.pth").is_file()
    assert (run_path / "dqn_conditional_value_concat_model.pth").is_file()
    assert (run_path / "dqn_conditional_concat_model.pth").is_file()
    assert (run_path / "dqn_conditional_concat_training_stats.json").is_file()


def test_loaded_agents_match_architecture(trained_run):
    run_path, _ctx, _f, _v, _c = trained_run
    import torch

    film_ckpt = torch.load(
        run_path / "dqn_conditional_scoring_model.pth", weights_only=False
    )
    value_concat_ckpt = torch.load(
        run_path / "dqn_conditional_value_concat_model.pth", weights_only=False
    )
    concat_ckpt = torch.load(
        run_path / "dqn_conditional_concat_model.pth", weights_only=False
    )
    assert isinstance(
        load_conditional_scoring_agent(film_ckpt), ConditionalPathScoringDQNAgent
    )
    assert isinstance(
        load_conditional_scoring_agent(value_concat_ckpt),
        ValueConcatConditionalPathScoringDQNAgent,
    )
    assert isinstance(
        load_conditional_scoring_agent(concat_ckpt),
        LegacyConditionalPathScoringDQNAgent,
    )


def test_value_concat_argmax_is_intent_invariant():
    """Structural guarantee: with intent-independent path features, the value-only
    concat cannot re-rank paths -- changing the reward-weight vector shifts V(s)
    uniformly and leaves argmax over path advantages unchanged. (In the full env a
    tiny residual can arise because one path *feature*, trust, is itself a function
    of the weights; that leak is orthogonal to the conditioning mechanism.)"""
    import torch

    from src.rl.dqn_agent_enhanced import EnhancedDQNConfig
    from src.rl.dqn_agent_scoring_conditional import (
        ValueOnlyConcatDuelingPathScoringDQN,
    )

    cfg = EnhancedDQNConfig(hidden_dim=32, n_hidden_layers=2, dropout_rate=0.0)
    sg, wd, pd = 7, 5, 6
    net = ValueOnlyConcatDuelingPathScoringDQN(sg, wd, pd, cfg).eval()
    torch.manual_seed(0)
    scoring = torch.randn(4, sg)
    paths = torch.randn(4, 6, pd)
    mask = torch.ones(4, 6, dtype=torch.bool)
    w1 = torch.rand(4, wd)
    w2 = torch.rand(4, wd)
    with torch.no_grad():
        q1 = net(torch.cat([scoring, w1], 1), paths, mask)
        q2 = net(torch.cat([scoring, w2], 1), paths, mask)
    # Same path wins under any intent...
    assert bool((q1.argmax(1) == q2.argmax(1)).all())
    # ...and the per-path Q difference is a constant (a pure V(s) shift).
    diff = q1 - q2
    assert bool((diff - diff[:, :1]).abs().max() < 1e-5)


def test_full_chapter6_pipeline(trained_run, tmp_path, monkeypatch):
    run_path, ctx, *_ = trained_run
    monkeypatch.setattr(ce, "EVAL_HOURS", _EVAL_HOURS)
    out_dir = tmp_path / "chapter6_test"
    out_dir.mkdir()

    abl = ce.run_ablation(run_path, out_dir, max_pairs=3, run_context=ctx)
    # Two conditional methods evaluated (no flat checkpoint): the naive value-only
    # concat baseline and FiLM. The legacy 2-stream concat checkpoint is trained by
    # the fixture (verified elsewhere) but intentionally excluded from the chapter
    # ablation, so it must not appear here.
    assert set(abl["methods"]) == {
        "conditional_concat",
        "conditional_film",
    }
    assert Path(abl["reward_csv"]).is_file()
    assert Path(abl["divergence_csv"]).is_file()
    assert Path(abl["table_tex"]).is_file()
    div_rows = _read_rows(Path(abl["divergence_csv"]))
    assert len(div_rows) == 2
    for r in div_rows:
        assert 0.0 <= float(r["behavioral_divergence"]) <= 1.0
    # Reward matrix: 2 methods x len(INTENT_PROFILES) profiles.
    n_intents = len(ce.INTENT_PROFILES)
    assert len(_read_rows(Path(abl["reward_csv"]))) == 2 * n_intents
    tex = Path(abl["table_tex"]).read_text()
    assert "\\begin{tabular}" in tex and "Conditional-FiLM" in tex

    align = ce.run_intent_alignment(run_path, out_dir, max_pairs=3, run_context=ctx)
    matrix_rows = _read_rows(Path(align["matrix_csv"]))
    assert len(matrix_rows) == n_intents * n_intents
    assert Path(align["metrics_csv"]).is_file()

    pc = ce.run_probing_ceiling(
        run_path, out_dir, max_pairs=3, n_congestion_bins=3, run_context=ctx
    )
    q_rows = _read_rows(Path(pc["quality_csv"]))
    methods = {r["method"] for r in q_rows}
    assert "conditional_film" in methods and "widest_path" in methods
    assert Path(pc["ceiling_csv"]).is_file()

    figs = cf.generate_all_figures(out_dir, metric="goodput")
    assert set(figs) == {
        "fig_6_1_heatmap",
        "fig_6_1_boxplots",
        "fig_6_2_quality_vs_probe",
        "fig_6_3_ceiling",
    }
    for path in figs.values():
        assert Path(path).is_file() and Path(path).stat().st_size > 0
