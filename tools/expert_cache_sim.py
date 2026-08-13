"""Expert-cache simulation for DeepSeek-V4-Flash on TT-QuietBox 2.

The model's expert weights overflow device DRAM by ~24 GB, so ~25% of them live
in host memory and stream over PCIe at ~23x lower bandwidth than GDDR6. Whether
that is a 2x tax or a 5x tax depends entirely on the *cache hit rate*, which in
turn depends on how skewed and how temporally stable MoE routing actually is.

A capacity-proportional hit rate (75%) is the pessimistic floor -- it assumes
routing is uniform and memoryless. This sweeps skew and temporal correlation to
bound the real answer, and accepts a recorded routing trace so the same
machinery can be re-run against the real model once hardware is available.

Run:  PYTHONPATH=pylibs python3 tools/expert_cache_sim.py
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

import budget as B

GB = 1024**3


# --------------------------------------------------------------------------
# Trace generation
# --------------------------------------------------------------------------


def zipf_weights(n: int, s: float) -> np.ndarray:
    """Normalized Zipf(s) probabilities over n experts. s=0 is uniform."""
    w = 1.0 / np.power(np.arange(1, n + 1), s)
    return w / w.sum()


def synth_trace(steps: int, layers: int, experts: int, top_k: int, batch: int,
                skew: float, reuse: float, rng: np.random.Generator) -> list[list[np.ndarray]]:
    """Routing trace: trace[step][layer] -> array of distinct expert ids.

    `skew` is the Zipf exponent. `reuse` is the probability that a step reuses
    the previous step's expert set for a layer, modelling the fact that
    consecutive tokens in one conversation tend to hit similar experts.
    """
    p = zipf_weights(experts, skew)
    # Each layer gets its own random permutation of the popularity ranking, so
    # "expert 0" is not globally hot across all layers.
    perms = [rng.permutation(experts) for _ in range(layers)]
    trace: list[list[np.ndarray]] = []
    prev: list[np.ndarray | None] = [None] * layers

    for _ in range(steps):
        step: list[np.ndarray] = []
        for l in range(layers):
            if prev[l] is not None and rng.random() < reuse:
                sel = prev[l]
            else:
                draws = rng.choice(experts, size=batch * top_k, p=p)
                sel = np.unique(perms[l][draws])
            step.append(sel)
            prev[l] = sel
        trace.append(step)
    return trace


# --------------------------------------------------------------------------
# Cache policies. One independent cache per layer, since an expert tensor
# belongs to exactly one layer.
# --------------------------------------------------------------------------


def simulate(fit, ev, layers: int, experts: int, capacity: int, policy: str,
             pin_frac: float = 0.5) -> float:
    """Fraction of expert *fetches* in `ev` that are served from device DRAM.

    Anything with a fitted component (a pinned hot set) is fitted on `fit` and scored
    on the disjoint `ev`. Scoring a fitted set against its own trace inflates the
    result badly enough to invert the conclusion: at ~1800 draws per layer, picking
    "the top 193 of 256 by observed count" captures 86.6% of a perfectly *uniform*
    stream whose true ceiling is 75.4%, purely from counting noise. That decays to
    76.6% at 100x the trace length, which is the tell. Keep the split.
    """
    if capacity >= experts:
        return 1.0

    hits = total = 0

    if policy == "capacity":
        # Pessimistic baseline: residency is uncorrelated with demand.
        for step in ev:
            for sel in step:
                total += len(sel)
                hits += len(sel) * capacity / experts
        return hits / total

    def hot_sets(source, n: int) -> list[set[int]]:
        counts = [np.zeros(experts, dtype=np.int64) for _ in range(layers)]
        for step in source:
            for l, sel in enumerate(step):
                counts[l][sel] += 1
        return [set(np.argsort(-c)[:n].tolist()) for c in counts]

    if policy == "oracle-static":
        # "Oracle" only in the sense of a prior profiling pass; it does not get to
        # see the traffic it is scored on.
        pinned = hot_sets(fit, capacity)
        for step in ev:
            for l, sel in enumerate(step):
                total += len(sel)
                hits += sum(1 for e in sel if e in pinned[l])
        return hits / total

    # LRU, optionally with a pinned hot set occupying `pin_frac` of capacity.
    n_pin = int(capacity * pin_frac) if policy == "pinned+lru" else 0
    n_lru = capacity - n_pin
    pinned = hot_sets(fit, n_pin) if n_pin else [set() for _ in range(layers)]

    caches: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(layers)]

    def run(source, measure: bool) -> None:
        nonlocal hits, total
        for step in source:
            for l, sel in enumerate(step):
                cache = caches[l]
                for e in sel:
                    if measure:
                        total += 1
                    if e in pinned[l]:
                        hits += measure
                        continue
                    if e in cache:
                        hits += measure
                        cache.move_to_end(e)
                    else:
                        cache[e] = None
                        if len(cache) > n_lru:
                            cache.popitem(last=False)

    run(fit, False)  # warm the LRU so cold-start misses are not charged to eval
    run(ev, True)
    return hits / total


# --------------------------------------------------------------------------


def decode_tok_s(c, tensors, scheme, batch: int, hit_rate: float, hw=B.QB2) -> float:
    # Assumes the streamed pool sits behind the x16 card (see PLAN.md section 2). The
    # even-sharding alternative caps at ~16 GB/s and cannot reach the floor at any hit
    # rate, so scoring cache policies against it would only measure the bad layout.
    d = B.decode_step(c, tensors, scheme, batch, hit_rate, hw,
                      h2d_gbps=hw.host_to_device_gbps_fast_card)
    return d["tok_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "refs/hf_dsv4/config.json"))
    ap.add_argument("--trace", help="JSON routing trace recorded from a real run")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    c = B.Config.load(Path(args.config))
    tensors = B.enumerate_tensors(c)
    scheme = B.SCHEMES["bfp4_b experts / fp8_e4m3 dense"]

    reserve_gb = 12.0
    weight_budget = (B.QB2.dram_total_gb - reserve_gb) * GB
    room = weight_budget - B.dense_bytes(tensors, scheme)
    resident_frac = min(1.0, room / B.expert_bytes(tensors, scheme))
    capacity = int(round(resident_frac * c.n_routed_experts))

    L, E, k = len(c.layer_kinds), c.n_routed_experts, c.num_experts_per_tok
    print("=" * 76)
    print("Expert cache simulation -- DeepSeek-V4-Flash on QuietBox 2")
    print("=" * 76)
    print(f"{L} MoE layers x {E} experts, top-{k}, batch {args.batch}")
    print(f"device holds {resident_frac * 100:.1f}% of experts -> capacity {capacity}/{E} per layer\n")

    rng = np.random.default_rng(args.seed)

    if args.trace:
        raw = json.loads(Path(args.trace).read_text())
        trace = [[np.asarray(sel) for sel in step] for step in raw]
        cut = len(trace) // 2
        fit, ev = trace[:cut], trace[cut:]
        print(f"using recorded trace: {len(trace)} steps ({cut} fit / {len(ev)} eval)\n")
        for policy in ("capacity", "lru", "pinned+lru", "oracle-static"):
            hr = simulate(fit, ev, L, E, capacity, policy)
            print(f"  {policy:14s} hit {hr * 100:5.1f}%   {decode_tok_s(c, tensors, scheme, args.batch, hr):6.1f} tok/s")
        return

    print("Synthetic sweep. 'skew' is the Zipf exponent over experts (0 = perfectly")
    print("balanced, which is what the aux-loss-free objective targets); 'reuse' is")
    print("the chance a decode step repeats the previous step's experts.\n")
    print(f"{'skew':>5s} {'reuse':>6s} | " + " ".join(f"{p:>13s}" for p in
          ("capacity", "lru", "pinned+lru", "oracle-static")))
    print("-" * 76)

    for skew in (0.0, 0.3, 0.6, 1.0):
        for reuse in (0.0, 0.3, 0.6):
            trace = synth_trace(2 * args.steps, L, E, k, args.batch, skew, reuse, rng)
            fit, ev = trace[: args.steps], trace[args.steps :]
            cells = []
            for policy in ("capacity", "lru", "pinned+lru", "oracle-static"):
                hr = simulate(fit, ev, L, E, capacity, policy)
                ts = decode_tok_s(c, tensors, scheme, args.batch, hr)
                cells.append(f"{hr * 100:4.1f}% {ts:5.0f}t/s")
            print(f"{skew:>5.1f} {reuse:>6.1f} | " + " ".join(f"{x:>13s}" for x in cells))

    full = decode_tok_s(c, tensors, scheme, args.batch, 1.0)
    print("-" * 76)
    print(f"fully resident (unreachable, for scale): {full:.0f} tok/s")
    print("\nRead the 'capacity' column as the floor and 'oracle-static' as the ceiling.")
    print("The gap between them is what a good residency policy is worth.")


if __name__ == "__main__":
    main()
