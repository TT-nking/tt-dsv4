# Bringing up DeepSeek-V4-Flash on TT-QuietBox 2

Status: planning complete, implementation starting. No hardware access yet.

All figures below are reproducible with `python3 tools/budget.py`, which derives them
from the published `config.json` rather than from prose.

---

## 1. Verdict up front

DeepSeek-V4-Flash **does not fit** in QuietBox 2's 128 GB of GDDR6 in any format a
tensor can currently be stored in, and it is not close: the smallest available packing
is **151.9 GB, about 24 GB over budget**. Weight streaming from host DRAM is therefore
not an optimization we might add later — it is a load-bearing part of the design and
has to be in the architecture from day one.

Because streaming is load-bearing, the host link is on the critical path, and here the box
has an awkward property: its motherboard has only one x16-capable slot, so the two p300
cards get **~63 GB/s and ~8 GB/s** rather than matching links. Sharding experts uniformly —
the obvious layout, and the one the existing demos imply — puts the slow card in the
critical path and caps decode at ~18 tok/s. Confining the streamed working set to the fast
card lifts the same ceiling to ~70 tok/s. **A ~4x swing that costs nothing but placement,
and the difference between missing and clearing our performance floor.** Details and the
one-command way to verify it are in §2.

The good news is that the model's own design works strongly in our favour everywhere
else. The KV cache at a full 1M-token context is only **3.9 GB**, which is the whole
point of the CSA/HCA hybrid attention, and it means context length is essentially free
on this box. The problem is entirely the 277 B of expert weights.

Two pieces of context make the size of the ask concrete:

- Tenstorrent markets QuietBox 2 for models "up to 120B parameters". At 284 B total we
  are at roughly 2.4x the advertised envelope. That it is feasible at all is due to the
  model being only 13 B activated.
- Tenstorrent's own MoE config for this exact model,
  `models/common/modules/moe/configs/deepseek_v4_flash.yaml`, targets
  **`mesh_shape: [16, 8]` — 128 chips**, at 32 batch per device. We are attempting the
  same model on **4**. That is not a reason to stop, because the 128-chip config is
  built for throughput serving at batch 32/device while we are targeting an interactive
  batch-1 workstation, but it does mean no existing configuration can be reused as-is
  and we should expect to be the first to run this model at this scale.

The strategy that falls out of this is to compete on **long-context interactive use**
rather than throughput — the 1M-token KV cache is only 3.9 GB, and long single-sequence
generation is also the regime where the expert cache hides the streaming penalty best.
§9 sets the performance targets, §10 the path to shipping it. The one measurement that
most affects the outcome does not need hardware and has not been taken yet.

## 2. Target hardware

| | |
|---|---|
| Chips | 4x Blackhole (2x p300c cards) |
| Tensix cores | 480 (120/chip) @ 1.35 GHz |
| SRAM | 720 MB (180 MB/chip) |
| DRAM | 128 GB GDDR6 (32 GB/chip) |
| DRAM bandwidth | 2048 GB/s aggregate (512 GB/s per chip) |
| Compute | 2654 TFLOP/s BlockFP8 (all 4 chips) |
| Chip interconnect | Warp400, 2x per card + inter-card ARP6 cable |
| Host | Ryzen 7 9700X, 256 GB DDR5-5600 (~90 GB/s) |
| Board | ASRock B850M-C, mATX |
| Host link | card A Gen5 x16 (~63 GB/s), card B Gen4 x4 (~8 GB/s) — see below |
| OS | Ubuntu 24.04.3 LTS |

The two numbers that drive every decision: **2048 GB/s** on-device versus **~90 GB/s**
from host memory. Anything we are forced to stream costs us a ~23x bandwidth penalty.

### The two cards do not have equal host bandwidth

This is the single most consequential hardware detail on the box, and it is easy to miss.
The B850M-C has exactly one x16-capable slot. From the board manual: `PCIE1` is Gen5 x16
with Ryzen 9000, `PCIE4` is an x16-*length* slot wired **x4** Gen4, and `PCIE2`/`PCIE3`
run at x1. AM5 Granite Ridge does not have the lanes to do better. So whichever slots the
two p300 cards occupy, one gets ~63 GB/s to the host and the other gets at most ~8 GB/s.

That asymmetry decides whether streaming is viable, because streaming bandwidth is set by
the *slowest* path carrying traffic, not the sum of the links:

| Expert placement | Effective host->device | Decode ceiling @ batch 1 |
|---|---|---|
| Sharded evenly over all 4 chips | 16 GB/s (2x the x4 link) | ~18 tok/s |
| Streamed pool behind the x16 card only | 63 GB/s | ~70 tok/s |

A ~4x throughput swing from placement alone, and the difference between missing the
30 tok/s floor of §9 and clearing the 60 tok/s target. The uniform sharding that every
tutorial reaches for is the one layout that cannot hit the floor.

**Design consequence.** Expert residency has to be deliberately asymmetric: the x4 card
holds only *resident* experts and never streams, while the streaming pool lives entirely
in the x16 card's 64 GB. This costs us half the device memory as streaming backing store,
so the resident fraction on the fast card is lower than the 75% whole-box figure, and it
couples expert placement to the parallelism strategy in §5 — experts cannot be assigned to
chips by index alone. It also means the mHC/router all-to-all is asymmetric, since the two
cards play different roles.

**Verify before designing around it.** The above is inferred from the board model, not
measured. On the box:

```bash
lspci -d 1e52: -vv | grep -E 'LnkSta|LnkCap'   # Tenstorrent vendor ID
```

Confirm the negotiated width per card. If Tenstorrent ships a variant board, a riser, or
a PCIe switch, these numbers change and the placement conclusion may relax — which would
be good news. Either way it is a five-second check that gates a large design decision.

## 3. What is actually new in V4

Derived from the paper (arXiv 2606.19348) and the HF reference implementation, both
in `refs/`. Verified parameter count 284.33 B against the published 284 B, and 12.74 B
activated against the published 13 B, so the module inventory in `tools/budget.py` is
trustworthy.

**Layer schedule** (43 layers, from `compress_ratios`): layers 0–1 are sliding-window
only, then layers 2–42 alternate CSA (m=4, even indices, 21 layers) and HCA (m'=128,
odd indices, 20 layers). The first 3 MLP layers use hash routing, the rest top-k.

Five things are genuinely new relative to DeepSeek-V3, and each is a distinct
engineering risk:

1. **Compressed Sparse Attention (CSA)** — pools every 4 tokens into one KV entry with
   a softmax-gated weighted sum plus a learned position bias, using *overlapping*
   windows (each entry draws on 2m slots, so consecutive windows share state). A
   Lightning Indexer then scores queries against the pooled keys and keeps the top 512.
2. **Heavily Compressed Attention (HCA)** — same pooling at m'=128, non-overlapping,
   no indexer, dense attention over all pooled entries.
3. **Manifold-Constrained Hyper-Connections (mHC)** — the residual stream is 4 parallel
   streams (`hc_mult=4`), mixed by a matrix projected onto the doubly-stochastic
   manifold by **20 Sinkhorn-Knopp iterations, per layer, per token**.
4. **Shared-KV MQA with head_dim=512** — one KV head read as both key *and* value.
   Because V carries RoPE, the attention output gets the conjugate rotation applied at
   position `-i` to strip the absolute component back out.
5. **Grouped low-rank output projection** — 64x512=32768 wide attention output, split
   into 8 groups, each projected to 1024, then mixed to 4096.

Plus two things the `gpt_oss` demo appears to give us for free — per-head learnable
attention sinks and clamped SwiGLU. Both turn out to be only partial fits; see §6.

## 4. The memory wall, quantified

A tensor on Blackhole can be stored as `Bfp8_b`, `Bfp4_b`, `Fp8_e4m3`, `Float16(_b)` or
`Float32` (`tt::tt_metal::DataType`). The hardware's `tt::DataFormat` list is wider than
that, and the gap between the two is what kills the only packing that fits (below).
Blackhole does **not** support MXFP4 in
either sense; that is Quasar-only
(`tt_metal/common/tt_backend_api_types.cpp`). This matters because DeepSeek ships the
experts as FP4 with `ue8m0` block scales, which is essentially MXFP4; we cannot store
it natively and must transcode to `Bfp4_b` at 4.5 bits/element instead of 4.25.

| packing | experts | dense | total | vs 128 GB |
|---|---|---|---|---|
| as-shipped fp4/fp8 *(not BH-legal)* | 131.0 GB | 7.8 GB | 138.8 GB | +10.8 |
| mxfp4/fp8_e4m3 *(not BH-legal)* | 137.1 GB | 7.8 GB | 144.8 GB | +16.8 |
| **bfp4_b experts / fp8_e4m3 dense** | 145.1 GB | 6.8 GB | **151.9 GB** | **+23.9** |
| bfp4_b experts + bfp4_b attn | 145.1 GB | 5.0 GB | 150.1 GB | +22.1 |
| bfp2_b experts / bfp8_b dense *(not storable)* | 80.6 GB | 8.1 GB | 88.7 GB | *fits* |

Experts are **97.4%** of all parameters, so nothing we do to the dense weights matters
much — squeezing attention from `Bfp8_b` to `Bfp4_b` saves only 1.8 GB against a 24 GB
deficit. Only the experts move the needle.

### 2-bit experts are not available, which removes the only packing that fits

The last row is the only one under 128 GB, and it is not reachable today. `Bfp2_b`
exists as a `tt::DataFormat` and `is_supported_wormhole_blackhole` accepts it, but
weight storage is bounded by `tt::tt_metal::DataType`, which has **no `BFLOAT2_B`
entry** — and `datatype_to_dataformat_converter`
(`tt_metal/impl/tensor/tensor_types.cpp:91`) has no case that emits `Bfp2_b`. The
hardware can unpack 2-bit blocks; there is no way to construct a tensor holding them.

This matters because it is easy to look at the format list and conclude the model fits.
It does not. Making 2-bit work would mean adding a new tensor dtype to ttnn end to end
(tensor, packer config, matmul program factories), which is upstream work in tt-metal
rather than something we can do inside a model demo. Even then, these weights came out
of FP4 quantization-aware training, and closing the gap by demoting only the coldest
experts would require demoting **56%** of them — not a tail, most of the model. I would
not propose it as the default even if the dtype existed.

**Conclusion: streaming is not one option among several. It is the only option.**

### KV cache is a non-issue

| context | total KV |
|---|---|
| 8 K | 0.03 GB |
| 64 K | 0.24 GB |
| 256 K | 0.97 GB |
| 1 M | **3.86 GB** |

Full 1M context costs under 4 GB. We should treat long context as a headline feature of
this port rather than something to ration.

### What streaming costs

With ~75% of experts resident and the remainder streamed over PCIe, against the same
model with everything resident. Both streaming placements from §2 are shown, since the
gap between them dwarfs every other effect in this table:

| batch | streamed evenly (16 GB/s) | streamed via x16 card (63 GB/s) | fully resident |
|---|---|---|---|
| 1 | 18 tok/s | 70 tok/s | 188 tok/s |
| 8 | 19 tok/s | 76 tok/s | 482 tok/s |
| 64 | 34 tok/s | 135 tok/s | 1020 tok/s |

These are pure bandwidth ceilings with no compute or launch overhead, so treat them as
upper bounds on the *ratio*, not as forecasts. Two things matter in the shape. First,
placement is worth ~4x and is entirely ours to choose. Second, even in the good column
streaming costs ~2.7x at batch 1 and **7.6x at batch 64**, because larger batches touch
nearly every expert and destroy any residency advantage.

That shape sets the product decision. A $9,999 desk-side workstation is an interactive,
low-batch machine. We should optimize hard for batch 1–8 and treat high-throughput
serving as out of scope for this platform.

### How good does the expert cache have to be?

The 75% figure assumes routing is uniform and memoryless, so residency equals capacity.
`tools/expert_cache_sim.py` sweeps how much better a real policy can do, varying the
routing skew (Zipf exponent) and the chance that consecutive decode steps reuse the same
experts. Every fitted policy is fitted on one half of the trace and scored on the other.
At batch 1:

| policy | balanced, no reuse | balanced + reuse 0.6 | mild skew + reuse | strong skew |
|---|---|---|---|---|
| capacity floor | 75.4% → 71 tok/s | 75.4% → 71 tok/s | 75.4% → 71 tok/s | 75.4% → 71 tok/s |
| plain LRU | 74.8% → 69 tok/s | **90.1% → 177 tok/s** | 84.7% → 114 tok/s | 92.8% → 183 tok/s |
| pinned hot set + LRU | 74.9% → 69 tok/s | 90.1% → 175 tok/s | 85.1% → 117 tok/s | 93.1% → 184 tok/s |
| static popularity | 75.3% → 71 tok/s | 75.9% → 72 tok/s | 79.5% → 85 tok/s | 93.5% → 184 tok/s |

These assume the streamed pool sits behind the x16 card (§2). Under even sharding every
cell in the table falls below the 30 tok/s floor regardless of hit rate, which is why
placement is settled first and cache policy second.

Four conclusions worth acting on:

1. **The target is ~86% hit rate, not 100%.** That is where PCIe transfer time drops
   below on-chip compute time and hides behind it completely; past it, decode saturates
   at the on-device bound and further cache work buys nothing. A much easier goal than
   "make it all fit".
2. **Temporal reuse, not popularity skew, is the variable that decides this.** Holding
   routing perfectly balanced and moving reuse from 0 to 0.6 moves decode from 69 to 177
   tok/s — nearly the entire available range. This matters because V4-Flash uses
   `score_correction_bias` (aux-loss-free balancing), whose explicit purpose is to
   flatten expert popularity. We should design assuming **skew ≈ 0** and treat any skew
   we find as a bonus.
3. **Static popularity pinning is worthless in exactly the regime we expect.** At skew 0
   it scores 75.3%, i.e. indistinguishable from placing experts arbitrarily. It only
   earns its keep under strong skew, which the training objective suppresses.
4. **Plain LRU is the right default.** The pinned hot set adds at most ~0.4 points over
   a warmed LRU anywhere in the sweep. Prefer the simpler policy; revisit only if a real
   trace shows meaningful skew.

> An earlier version of this section reported static pinning as the *ceiling* (86.6% hit
> under balanced routing) and concluded that plain LRU was a trap. That was a
> measurement error: the hot set was fitted and scored on the same trace, and at ~1800
> draws per expert per layer, counting noise alone lets "the top 193 of 256" capture
> 86.6% of a perfectly uniform stream. Fitting on held-out data collapses it to 75.3%.
> The conclusions above are the corrected ones and they point the opposite way.

The one thing this cannot tell us is where the real model sits on the reuse axis. That
is now **the single highest-value measurement in the whole project**, since it alone
spans a 2.6x range in decode throughput. It does not need QuietBox 2 and it does not
need a working TT model — only the router weights and a forward pass, which means it can
be done on CPU today (see §9). `expert_cache_sim.py --trace` consumes the result
directly.

## 5. Parallelism strategy

**TP=4 for attention, EP=4 for MoE**, mirroring the existing `deepseek_v3` demo's
approach (though that targets Galaxy, i.e. 32–64 Wormhole chips, not 4 Blackhole).

- *Attention*: 64 heads / 4 = 16 heads per chip. The grouped output projection has 8
  groups, giving a clean 2 groups per chip, with an all-reduce after `o_b_proj`.
- *MoE*: 64 of 256 experts per chip, all-to-all dispatch/combine. Per token per layer
  this is ~24 KB dispatch (fp8) + ~48 KB combine (bf16); at 43 layers that is ~3.1 MB
  per token, which is small but not free at batch 1 where it is pure latency.
- *KV*: MQA has a single KV head, so it cannot be split head-wise. Replicate the 3.9 GB
  across chips (simple, costs 3.9 GB/chip) or shard along sequence. Start replicated.

Per-chip budget check: 32 GB minus ~1.7 GB sharded dense minus 3.9 GB KV leaves
~26.4 GB for experts, against 36.3 GB needed. Confirms the ~73–75% residency figure
independently.

### EP=4 by expert index is the layout §2 rules out

The clean "64 consecutive experts per chip" split spreads the streamed remainder evenly
over both cards, which is exactly the 16 GB/s case. The fix is to keep EP=4 for *compute*
but make **residency** asymmetric: the two chips on the x4 card hold a fully resident
slice and never stream, while the streamed working set is confined to the x16 card's two
chips.

Sizing it: of the 36.3 GB of experts per-chip-equivalent, ~26.4 GB fits per chip. Give the
x4 card's two chips their full resident slice and let the x16 card absorb the deficit,
which lands roughly 2/3 of the streamed traffic on ~1/2 the memory. That trade is
favourable only because the fast link is ~8x the slow one.

Two costs to design around. Dispatch/combine becomes asymmetric, so the all-to-all is no
longer a uniform collective and load balancing has to account for the fact that the x16
card's experts have variable latency. And expert-to-chip assignment becomes a placement
*decision* rather than `expert_id % 4` — which means the routing statistics from §4 feed
into placement, not just cache policy. Worth confirming the `lspci` widths (§2) before
building any of this, since a switched or bifurcated topology would let us keep the simple
layout.

## 6. Op inventory: reuse vs build

tt-metal is much further along on V4 than expected. It already vendors the V4 HF
reference and a `DeepSeekV4FlashConfig`, and several ops are not merely reusable but
were *written for this model* and are tested at its exact geometry.

### Already exists, V4-specific

| Op | Evidence it targets V4-Flash |
|---|---|
| `ttnn.experimental.indexer_score_dsa` | The Lightning Indexer scorer, `Σ_h ReLU(q·k)·w`. Test sweeps `heads=64, dim=128` — exactly V4-Flash's indexer geometry. Blackhole-only. |
| `ttnn.experimental.deepseek_prefill.moe_hash_gate` | Nanobind doc says "DeepSeek-V4 hash-routing MoE gate"; test config `(256, 6, 1.5, 2048)` is labelled *DeepSeek-V4 Flash*, with `sqrtsoftplus`. |
| `moe_grouped_topk` (`n_groups=1`) | Parametrised `dsv4flash-1g256e` — the top-6-of-256 learned gate for layers 3+. |
| `unified_routed_expert_ffn`, `dispatch`, `combine`, `post_combine_reduce` | Full prefill MoE stack; `test_single_routed_expert.py` parametrised on `DeepSeekV4FlashConfig`. |
| `ttnn.transformer.sparse_sdpa` | Gathered top-k attention over a compressed cache — the shape of the CSA read. |
| `scaled_dot_product_attention` (+ decode) | Carries `attention_sink`. Paged KV via `paged_fill_cache` / `paged_update_cache`. |

`models/demos/deepseek_v3_d_p` (disaggregated prefill) is the primary reuse surface and is
Blackhole-first. What does *not* exist is a `DeepSeekV4FlashAdapter` or a V4
`TtPrefillTransformer` — the config and the ops are there, the assembled model is not.

**QuietBox 2 is below the smallest mesh this stack is tested on.** Its CI-gated coverage
is `(4, 2)` on a Blackhole LoudBox and `(8, 4)` on a Blackhole Galaxy
(`tests/conftest.py:53`) — 8 and 32 chips. QuietBox 2 is `ClusterType::P300_X2`, 2 P300
cards and **4** chips, so `(2, 2)` or `(1, 4)`. Compounding it,
`tests/cache/test_mla_cache.py:41` notes "Blackhole forms whole-box meshes only", so we
cannot carve a tested 8-chip shape out of a 4-chip box — the mesh is the whole box.

That same comment describes a "4-device QuietBox" as a *Wormhole* shape, i.e. the
original QuietBox, not QuietBox 2. An earlier version of this section cited it as
evidence that these tests already run on QuietBox 2 hardware. They do not, and the two
boxes are different architectures: original QuietBox is Wormhole (`T3K`-class), QuietBox 2
is Blackhole (`P300_X2`). Nothing in the disaggregated-prefill test matrix targets 4
Blackhole chips.

Practical consequence: we will be the first to run this stack at this mesh size, so
expect to hit untested paths in dispatch/combine and fabric setup rather than just
missing kernels. Worth budgeting time for, and worth running the existing
`deepseek_v3_d_p` tests at `(2, 2)` on the box early to find out what breaks before any
V4 work depends on it.

### Must be built (in rough order of risk)

1. **CSA / HCA compressors.** The single biggest gap. Softmax-gated pooling of the KV
   time axis, at m=4 with *overlapping* windows and m'=128 without. Nothing analogous
   exists — every sparse-attention path in tt-metal today consumes a cache someone else
   compressed. The overlap state machine that carries window `w-1`'s Ca slice across a
   forward boundary is the fiddly part.
2. **Attention sinks in the sparse path.** Sinks exist in dense SDPA, decode SDPA and
   ring-joint SDPA — but **not in `sparse_sdpa`**, which is the kernel CSA needs. This
   was previously recorded here as "sink support already"; that is true only of the
   dense path. Either `sparse_sdpa` gains a sink term or CSA needs a fused variant.
3. **Top-512 selection.** `ttnn.topk`'s multicore path caps at **k ≤ 64**; V4 needs
   `index_topk=512` over up to 262144 pooled entries at 1M context. Needs a multi-stage
   top-k, a custom kernel, or block-pooled scores like the MSA indexer variant.
4. **mHC / Sinkhorn.** 20 row/column normalisation iterations on a 4x4 per token, per
   layer, twice per layer. Trivial FLOPs, terrible op-launch profile — needs one fused
   kernel, not 40 dispatches.
5. **SwiGLU clamped at 10.0.** The fused expert kernel offers `Silu` (no clamp) or
   `SwiGluOai` with α and limit **hardcoded to gpt-oss's 1.702 / 7.0**. V4 needs
   clamp-then-SiLU at 10.0: either a parameterised limit or composition with
   `ttnn.clamp` outside the fused path.
6. **YaRN RoPE for the compress branches** (θ=160000, factor 16, interleaved). Not
   wired in the V4 TT stack.
7. **Inverse-RoPE on the attention output** — the conjugate rotation at `-i`. Cheap,
   just new.
8. **Grouped output projection** — block-diagonal bmm; likely maps onto batched matmul.

### Open hardware questions

- `sparse_sdpa` is tested at `K_DIM=576` (the MLA latent width), not V4's **512**. The
  op looks dimension-agnostic but is unvalidated at our geometry. Note the HF reference
  disables FlashAttention precisely because `head_dim=512` exceeds its 256 cap, so this
  is the recurring sharp edge of this architecture.
- Warp400 link bandwidth is unmeasured, and decides whether EP all-to-all is free or a
  bottleneck. The string "Warp400" does not appear anywhere in the open repos.

### Reference cross-validation, and a bug worth upstreaming

Our NumPy spec was verified against HuggingFace's `transformers.models.deepseek_v4`, but
Tenstorrent's op tests validate against a *separate* vendored copy under
`models/demos/deepseek_v3_d_p/reference/deepseek_v4/`. If those disagreed, every golden
in `goldens/` would encode the wrong semantics and we would only find out when kernels
started failing tests we did not write. `tools/xref_ttmetal.py` runs identical weights
through both.

**Result: bit-exact, 0.000e+00 on all ten comparisons** across prefill and eight
incremental decode steps. The goldens transfer, and the NumPy reference is a valid spec
against Tenstorrent's own tests.

Getting there surfaced three defects in the vendored copy, which together mean **it
cannot do autoregressive decode at all**:

1. Both cache classes rename `_layer_type` to `layer_type`. Registration into
   `DYNAMIC_LAYER_TYPE_MAPPING` happens in `CacheLayerMixin.__init_subclass__` keyed on
   `_layer_type` (`transformers/cache_utils.py:38`), so neither class ever registers and
   `DynamicCache(config=...)` raises `KeyError: 'compressed_sparse_attention'`.
2. `DeepseekV4HCACache.__init__` calls `super().__init__(config)` positionally, but
   `DynamicSlidingWindowLayer.__init__` is `(self, sliding_window: int, **kwargs)`, so
   `sliding_window` becomes a config object. Both classes also dropped the `**kwargs`
   that absorbs the `sliding_window` kwarg `DynamicCache` passes in.
3. `DeepseekV4Model.forward` binds `return_cache = past_key_values if use_cache else
   None` *before* creating the cache, so the first forward builds a `DynamicCache`, uses
   it, and then returns `None`.

The third is the one that bites silently: with `use_cache=True` the model returns no
cache, so a decode loop keeps re-running prefill and quietly produces wrong output
rather than raising. That is exactly the failure mode that would have made our decode
goldens look mysteriously wrong.

`patches/tt-metal-v4-cache-bootstrap.patch` fixes all three. With it applied the model
bootstraps a correct cache (`[sliding, sliding, CSA, HCA, CSA, HCA]` for the tiny
config), incremental decode advances the sequence properly, and `generate()` works.
Worth upstreaming early — it is small, self-contained, and a useful first contribution
that establishes contact with the team who own the V4 work.

## 7. Phased plan

**Phase 0 — Foundations.** *Done.* Architecture extracted and validated against
published parameter counts, hardware modeled, stack ingested, op inventory taken.

**Phase 1 — Reference and goldens.** *No hardware required; in progress.*
Build a tiny config that exercises all three attention types, both MLP routing modes,
and mHC, then dump per-submodule golden tensors from the HF reference. Every TT kernel
gets tested against these before anything touches a real checkpoint. Also: extract
routing traces from the reference to measure the expert-locality question from §4,
since that determines whether the streaming design is comfortable or painful.

**Phase 2 — Op-level TT-NN implementation.** *Needs hardware or `ttsim`.*
Work through the build list in §6 bottom-up, each against its golden. tt-metal ships a
Blackhole functional simulator (`libttsim_bh.so`, via `TT_METAL_SIMULATOR`) which should
let us validate correctness before the box is available, though it is slow and the
binaries appear to come from an internal CI artifact rather than a public release.

**Phase 3 — Layer and model assembly.** Decoder block, then the 43-layer stack with
TP=4/EP=4, still on random weights at reduced scale.

**Phase 4 — Weight pipeline.** Transcode the real checkpoint: FP8 `ue8m0`-block dense
to `Fp8_e4m3`/`Bfp8_b`, FP4 experts to `Bfp4_b`. Needs a careful numerical study, since
this is a lossy format change on QAT'd weights. Reuse
`deepseek_v3/scripts/dequantize_hf_checkpoint.py` as the starting point.

**Phase 5 — Expert residency and streaming.** Warmed LRU per layer, double-buffered
prefetch over PCIe overlapped with compute, driven by the routing statistics from
Phase 1. Hold the pinned hot set in reserve for the case where a real trace shows skew.

**Phase 6 — Serving and optimization.** Land in `tt-inference-server` (§10), then
profile-guided optimization against `tt-npe`.

## 8. Practical constraints

The development machine is macOS on Apple Silicon, which **cannot build or run
tt-metal** — it is Linux-only, and there is no TT hardware attached. Phase 1 is fully
achievable here (it is pure PyTorch). Phase 2 onward needs either a Linux box with the
simulator or the colleague's QuietBox 2. The repo is therefore being laid out so the
whole thing clones and runs on the target machine with no local state.

Realistically, none of the Phase 4 work can be validated without the full 284 B
checkpoint, which is ~140 GB of downloads and needs the 4 TB NVMe on the box itself.

## 9. What "competitive" should mean here

We should not compete on raw throughput. The memory wall costs roughly 2x at batch 1 and
about 5x at batch 64, so a benchmark that rewards large-batch serving is one we lose by
construction — and it is the benchmark Tenstorrent's own 128-chip `[16, 8]` config is
built to win. Picking that fight on 4 chips is choosing to look bad at something the
hardware was never asked to do.

The fight worth picking is **long-context interactive use**, where three things line up:

1. **KV cache at 1M tokens is 3.9 GB.** That is the entire point of the CSA/HCA hybrid,
   and it is a structural advantage rather than a tuning result. A dense-attention model
   of comparable size needs one to two orders of magnitude more KV at that context, which
   is why 1M-token context is normally a datacenter capability.
2. **Long single-sequence generation is the high-reuse regime**, which is exactly where
   §4 shows the expert cache performing well. The workload that showcases the context
   window is also the workload that hides the streaming penalty. These reinforce.
3. **Batch 1–8 is what a desk-side box is for.** Optimizing for it is honest rather than
   a concession.

So the headline claim to aim at is *a 284 B model with a 1M-token context, answering
interactively, on a desk-side workstation* — not tokens per second in isolation.

### Targets

Decode figures below are batch 1. Roofline is the pure-bandwidth ceiling from
`tools/budget.py`; real kernels typically realize 50–70% of that once launch overhead,
collectives and imperfect overlap are counted, and that discount is the main reason these
targets sit well below the roofline.

All three tiers assume the §2 placement fix. Without it the roofline itself is 18 tok/s
and the floor is unreachable, so placement is a precondition rather than an optimization.

| tier | batch-1 decode | rationale |
|---|---|---|
| floor | 30 tok/s | comfortably faster than reading speed; the model is *usable*. ~42% realization against the 71 tok/s capacity-floor roofline, so reachable with no cache cleverness at all |
| target | 60 tok/s | needs either ~85% realization of the capacity floor, or any modest amount of temporal reuse to lift the roofline |
| stretch | 120 tok/s | needs hit rate ≥ 86%, i.e. streaming fully hidden behind compute |

The upper tiers are gated mostly on the reuse measurement, not on kernel quality. That is
worth restating: **much of the difference between the target and stretch tiers is a
property of the model we have not measured yet, not work we have not done.**

Two supporting targets: time-to-first-token should stay interactive at long context
(prefill is compute-bound and parallelizes well, so this should be the easy one), and
accuracy must land within noise of the GPU reference on the eval set, since a fast wrong
answer is not a showcase.

### The measurement that unblocks this

Expert reuse across consecutive decode steps needs only router weights and a forward
pass — no TT hardware, no working TT model, no 284 B checkpoint. It can be run on CPU
now, either against V4-Flash's routers alone or against a smaller aux-loss-free MoE as a
proxy. Given it spans a 1.9x range in decode throughput and gates the stretch tier, it
should come before any kernel work.

## 10. The showcase path: land in tt-inference-server

For this to be an easy path to success for users rather than a demo script, it has to
arrive where Tenstorrent users already look. `tt-inference-server` is that surface, and
it is more complete than expected:

- `vllm-tt-metal/src/run_vllm_api_server.py` — OpenAI-compatible endpoint
- `tt-vllm-plugin` / `tt-sglang-plugin` — the serving backends
- `docker-entrypoint.sh` and `charts/` — container and Helm packaging
- `workflows/model_specs/{dev,prod}` — the model registry
- `docs/add_support_for_new_model.md` — a documented four-step onboarding process

That last file is effectively the definition of done, and step 4 is *Add Performance
Targets* while step 3 is *Add Accuracy Evals* including **GPU reference scores**. The
project's own onboarding process requires benchmarking against GPUs, which is where the
competitive numbers in §9 should ultimately be sourced rather than asserted here.

The user-facing result should be: pull a container, run one command, get an
OpenAI-compatible endpoint on `localhost`, and point any existing client at it —
including `tt-studio` for a chat UI. No bespoke harness.

### Staging it so there is always something to show

The full 284 B model is a long pole and gating the first demo on it is what turns this
into a project with nothing to show for six months. Two milestones instead:

- **Milestone A — architecture correct, reduced scale.** A small V4-Flash-shaped model
  (real architecture, fewer experts and layers) served through the normal path. This
  proves every novel op end to end, needs no streaming and no 140 GB download, and is a
  genuine result: *the V4 architecture runs on Blackhole*. It is also the natural vehicle
  for upstreaming the CSA/HCA/mHC kernels, which are useful to Tenstorrent independently
  of whether the 284 B model ever fits.
- **Milestone B — full 284 B with streaming.** The headline, built on a path already
  proven correct by A.

Milestone A is reachable without solving the memory wall at all, which is what makes it
the right thing to build first.
