#!/usr/bin/env python3
"""Build a genuinely held-out source--destination pair set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

from src.pipeline.run_dirs import resolve_run_dir


def trained_pairs(
    pair_pool: Sequence[Tuple[int, int]], episode_counts: Sequence[int], seed: int
) -> set:
    """Pairs visited by training runs of the given episode counts."""
    seen = set()
    for n_episodes in episode_counts:
        rng = random.Random(seed)
        for _ in range(n_episodes):
            seen.add(tuple(rng.choice(pair_pool)))
    return seen


def stratified_sample(
    candidates: Sequence[Tuple[int, int]], n_pairs: int, seed: int
) -> List[Tuple[int, int]]:
    """Round-robin over source ASes so the sample is not one AS's neighbourhood."""
    by_src = defaultdict(list)
    for pair in candidates:
        by_src[pair[0]].append(pair)
    rng = random.Random(seed)
    sources = sorted(by_src)
    rng.shuffle(sources)
    for src in sources:
        rng.shuffle(by_src[src])
    out: List[Tuple[int, int]] = []
    while len(out) < n_pairs and any(by_src[s] for s in sources):
        for src in sources:
            if by_src[src]:
                out.append(by_src[src].pop())
                if len(out) >= n_pairs:
                    break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--n-pairs", type=int, default=32)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[533, 666],
        help="Episode counts of the trained agents (flat / conditional).",
    )
    parser.add_argument("--pair-rng-seed", type=int, default=123)
    parser.add_argument("--sample-seed", type=int, default=20260725)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_dir or resolve_run_dir())
    with open(run_path / "selected_pair.json") as f:
        pair_pool = [(int(a), int(b)) for a, b in json.load(f)["pair_pool"]]

    seen = trained_pairs(pair_pool, args.episodes, args.pair_rng_seed)
    unseen = [p for p in pair_pool if p not in seen]
    chosen = stratified_sample(unseen, args.n_pairs, args.sample_seed)

    out_path = args.out or (run_path / "heldout_pairs.json")
    payload = {
        "run_dir": str(run_path),
        "pair_pool_size": len(pair_pool),
        "trained_pairs_visited": len(seen),
        "never_trained_pairs": len(unseen),
        "episode_counts": args.episodes,
        "pair_rng_seed": args.pair_rng_seed,
        "sample_seed": args.sample_seed,
        "n_source_ases": len({p[0] for p in chosen}),
        "pairs": [[int(a), int(b)] for a, b in chosen],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"pair pool                 : {len(pair_pool)}")
    print(f"pairs visited in training : {len(seen)}")
    print(f"never-trained pairs       : {len(unseen)}")
    print(
        f"sampled                   : {len(chosen)} pairs "
        f"across {payload['n_source_ases']} source ASes"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
