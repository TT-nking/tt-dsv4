"""Measure MoE expert-routing locality: the number that sets our decode throughput.

`docs/PLAN.md` §4 shows that DeepSeek-V4-Flash's expert weights overflow QuietBox 2's
GDDR6 by ~24 GB, so a quarter of them stream over PCIe. Whether that costs us nothing or
half our throughput depends on the device-side cache hit rate, and the simulation says
hit rate is governed almost entirely by *temporal reuse* — how often consecutive decode
steps route to experts the previous step already used. Holding popularity flat, moving
reuse from 0 to 0.6 moves batch-1 decode from 98 to 182 tok/s.

Nothing in the published config tells us where the real model sits on that axis, and it
needs only router weights and a forward pass — no TT hardware and no working TT model.
This measures it.

Two things are measured, both as *excess over a matched control* rather than raw rates:

  skew   how unevenly the trained router spreads load across experts, as a fitted Zipf
         exponent. V4-Flash uses aux-loss-free balancing (`score_correction_bias`),
         whose explicit purpose is to flatten this, so we expect it near 0 and should
         treat anything higher as a bonus.
  reuse  consecutive-step expert-set overlap, above the overlap between *distant* steps
         of the same run. Using distant steps as the control matters: raw overlap is
         inflated by popularity alone, and this subtracts that out so the number
         reflects genuine temporal locality.

Both map onto the axes `expert_cache_sim.py` sweeps, and the emitted trace can be fed
straight back into it with `--trace` for a hit rate on real routing.

Model choice: DeepSeek-V4-Flash's own routers would be ideal but the checkpoint does not
fit in host RAM at bf16. The default proxy is Moonlight-16B-A3B, which is DeepSeek-V3
architecture trained with the same aux-loss-free balancing, so its routing dynamics are
the closest available match. Any HF MoE works via --model.

Intended to run on the QuietBox 2 (256 GB host DDR5), not on a laptop.

Run:
  python3 tools/measure_reuse.py --steps 400
  python3 tools/measure_reuse.py --model allenai/OLMoE-1B-7B-0924 --steps 400
  python3 tools/expert_cache_sim.py --trace traces/<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Long-form single-topic text, because the workload we care about is long-context
# interactive generation (PLAN.md §9) and topic drift would understate reuse.
PROMPT = """The design of memory hierarchies in modern accelerators reflects a persistent
tension between capacity and bandwidth. Static random access memory placed close to the
compute units offers extraordinary bandwidth and negligible latency, but its area cost
per bit is high enough that practical designs provide only a few hundred megabytes.
Dynamic memory attached over a wide bus provides tens of gigabytes at perhaps a tenth of
the bandwidth, and host memory reached across a peripheral interconnect provides hundreds
of gigabytes at a further order of magnitude reduction. Every level of this hierarchy
exists because the level above it is too small, and every level imposes a cost that the
programmer must either hide or pay. Mixture-of-experts models interact with this
hierarchy in an unusual way. A dense model of a given parameter count reads all of its
weights for every token, so its arithmetic intensity is fixed and its performance is
straightforwardly predicted by dividing parameter bytes by available bandwidth. A sparse
model activates only a small fraction of its parameters per token, which reduces the
bytes read per token dramatically, but it does so by making the set of bytes read depend
on the input. This input dependence is the entire difficulty. If the active set were
known in advance, the weights could be staged into fast memory ahead of time and the
sparsity would translate directly into speed. Because the active set is chosen by a
learned router evaluated at run time, the memory system must either hold everything in
fast memory or accept that some fraction of reads will miss and stall. The question of
how often those misses occur is not a property of the hardware at all. It is a property
of the trained router, and specifically of how much the router's choices correlate across
consecutive positions in a sequence. A router whose choices are independent from token to
token forces the memory system into the worst case, where the probability of a hit is
simply the fraction of experts that happen to be resident. A router whose choices persist
across neighbouring tokens allows a cache to work, because the experts needed for the
next token are disproportionately the ones already present from the last. Training
objectives complicate this picture further, since load-balancing losses and bias
corrections are introduced precisely to prevent the router from favouring particular
experts, and in flattening the popularity distribution they remove the easiest source of
predictability. What remains is temporal locality, and whether it survives balanced
routing is an empirical question rather than a theoretical one."""


def find_routers(model, n_experts: int):
    """Locate the per-layer router projections.

    Routers are the `nn.Linear` (or plain weight-holding module) whose output width is
    the expert count. Matching on shape rather than name keeps this working across
    OLMoE (`mlp.gate`), Qwen (`mlp.gate`), and DeepSeek (`mlp.gate` with a separate
    `e_score_correction_bias`), which do not agree on naming.
    """
    found = []
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or w.ndim != 2:
            continue
        if w.shape[0] == n_experts and "gate" in name.split(".")[-1] or (
            w.shape[0] == n_experts and "router" in name.lower()
        ):
            found.append((name, mod))
    # Deduplicate while preserving depth order.
    seen, out = set(), []
    for name, mod in found:
        if id(mod) not in seen:
            seen.add(id(mod))
            out.append((name, mod))
    return out


class RouterRecorder:
    """Captures top-k expert selections per router per forward pass."""

    def __init__(self, routers, top_k: int):
        self.routers = routers
        self.top_k = top_k
        self.handles = []
        self.current: dict[int, np.ndarray] = {}
        self.index = {id(m): i for i, (_, m) in enumerate(routers)}

    def _hook(self, mod, args, output):
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.reshape(-1, logits.shape[-1]).float()
        # Plain top-k on the router logits. Models with group-limited routing
        # (DeepSeek V2/V3) restrict candidates to a subset of expert groups first, so
        # this can differ slightly from the true selection; it does not bias the
        # *reuse* statistic, which only cares about how selections move together.
        idx = torch.topk(logits, k=min(self.top_k, logits.shape[-1]), dim=-1).indices
        self.current[self.index[id(mod)]] = idx.cpu().numpy()

    def attach(self):
        for _, mod in self.routers:
            self.handles.append(mod.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of `a`'s experts that also appear in `b`."""
    return float(np.isin(a, b).mean()) if a.size else 0.0


def fit_zipf(counts: np.ndarray) -> float:
    """Least-squares Zipf exponent from a popularity histogram.

    Slope of log(frequency) against log(rank); 0 means perfectly balanced.
    """
    c = np.sort(counts[counts > 0])[::-1].astype(float)
    if c.size < 8:
        return 0.0
    rank = np.arange(1, c.size + 1, dtype=float)
    slope = np.polyfit(np.log(rank), np.log(c / c.sum()), 1)[0]
    return float(-slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="moonshotai/Moonlight-16B-A3B",
                    help="any HF MoE checkpoint; default is DeepSeek-V3 architecture "
                         "with aux-loss-free balancing")
    ap.add_argument("--steps", type=int, default=400, help="decode steps to record")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="trace JSON path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    print(f"loading {args.model} ({args.dtype} on {args.device}) ...", flush=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    n_experts = next(getattr(cfg, a) for a in
                     ("n_routed_experts", "num_experts", "num_local_experts",
                      "moe_num_experts") if getattr(cfg, a, None))
    top_k = next(getattr(cfg, a) for a in
                 ("num_experts_per_tok", "moe_top_k", "num_selected_experts")
                 if getattr(cfg, a, None))

    routers = find_routers(model, n_experts)
    if not routers:
        raise SystemExit(f"no routers with output width {n_experts} found")
    print(f"{len(routers)} MoE layers x {n_experts} experts, top-{top_k}")

    rec = RouterRecorder(routers, top_k)
    rec.attach()

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(args.device)
    print(f"prefill {ids.shape[1]} tokens, then {args.steps} greedy decode steps",
          flush=True)

    trace: list[list[np.ndarray]] = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)

        for step in range(args.steps):
            rec.current.clear()
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            # One token per step, so each router fired on a single row.
            trace.append([np.unique(rec.current[i][0])
                          for i in range(len(routers)) if i in rec.current])
            if (step + 1) % 50 == 0:
                print(f"  {step + 1}/{args.steps}", flush=True)
    rec.detach()

    n_layers = len(trace[0])
    rng = np.random.default_rng(args.seed)

    # Adjacent-step overlap, and the same statistic between distant steps of the same
    # run. The distant pairs are the control: they share the run's popularity
    # distribution but none of its temporal structure, so the difference isolates
    # locality from skew.
    adj, far = [], []
    for t in range(1, len(trace)):
        for l in range(n_layers):
            adj.append(overlap(trace[t][l], trace[t - 1][l]))
    for _ in range(len(trace)):
        t = rng.integers(0, len(trace))
        u = rng.integers(0, len(trace))
        if abs(t - u) < 20:
            continue
        for l in range(n_layers):
            far.append(overlap(trace[t][l], trace[u][l]))

    adj_m, far_m = float(np.mean(adj)), float(np.mean(far))
    # synth_trace() reuses the previous set with probability p and otherwise redraws,
    # giving expected overlap p + (1-p)*baseline. Invert for the equivalent p.
    reuse_param = (adj_m - far_m) / (1.0 - far_m) if far_m < 1.0 else 0.0

    counts = np.zeros((n_layers, n_experts), dtype=np.int64)
    for step in trace:
        for l, sel in enumerate(step):
            counts[l][sel] += 1
    skews = [fit_zipf(counts[l]) for l in range(n_layers)]

    print("\n" + "=" * 70)
    print(f"routing locality: {args.model}")
    print("=" * 70)
    print(f"steps recorded            {len(trace)}")
    print(f"layers x experts          {n_layers} x {n_experts}, top-{top_k}")
    print(f"random-collision floor    {top_k / n_experts * 100:.1f}%")
    print(f"adjacent-step overlap     {adj_m * 100:.1f}%")
    print(f"distant-step overlap      {far_m * 100:.1f}%   (popularity-only control)")
    print(f"excess temporal locality  {(adj_m - far_m) * 100:+.1f} points")
    print()
    print(f"--> reuse  {reuse_param:.2f}   (equivalent expert_cache_sim parameter)")
    print(f"--> skew   {np.mean(skews):.2f}   (mean fitted Zipf exponent, 0 = balanced)")
    print()
    if reuse_param >= 0.45:
        print("reuse >= 0.45: enough to hide PCIe streaming behind compute; the")
        print("stretch tier in PLAN.md SS9 (~120 tok/s) is reachable.")
    elif reuse_param >= 0.2:
        print("moderate reuse: expect to land between the target and stretch tiers.")
    else:
        print("low reuse: streaming will not hide, and decode is capped near the")
        print("capacity floor (~100 tok/s roofline). Prefetch/speculation matters more.")

    out_path = Path(args.out or f"traces/{args.model.split('/')[-1]}-{len(trace)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([[s.tolist() for s in step] for step in trace]))
    print(f"\ntrace -> {out_path}")
    print(f"replay: python3 tools/expert_cache_sim.py --trace {out_path}")


if __name__ == "__main__":
    main()
