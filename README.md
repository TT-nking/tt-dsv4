# DeepSeek-V4-Flash on TT-QuietBox 2

Bring-up of `deepseek-ai/DeepSeek-V4-Flash` (284 B total / 13 B activated, 1 M context)
on a 4-chip Blackhole TT-QuietBox 2 using Tenstorrent's open-source stack.

Start with **[docs/PLAN.md](docs/PLAN.md)** — feasibility analysis, memory strategy, and
the phased implementation plan.

## Current state

Nothing has run on real silicon: there is no QuietBox 2 attached, and tt-metal is
Linux-only while this machine is macOS/arm64. Everything here is therefore analysis or
pure-PyTorch/NumPy reference work that runs anywhere.

Two results shape the whole port:

- **The model does not fit.** In the narrowest format Blackhole supports natively it
  needs ~152 GB against 128 GB of device DRAM, so streaming expert weights from the
  256 GB of host DDR5 is a load-bearing part of the design rather than a later
  optimization. Experts are 97.4% of all parameters.
- **Long context is nearly free.** The CSA/HCA hybrid attention puts the KV cache at
  3.9 GB even at the full 1 M tokens, so context length should be treated as a headline
  feature of this port rather than something to ration.

The operator reference is complete and verified: every novel V4 op is reimplemented
from scratch and checked against tensors recorded from the HuggingFace model, 70/70
checks passing. That is the contract each TT-NN kernel will be written against.

## Layout

```
docs/PLAN.md               analysis, strategy, phased plan
tools/budget.py            parameter / memory / roofline model
tools/expert_cache_sim.py  expert residency policy simulation
tools/tiny_config.py       small config exercising every structural V4 feature
tools/smoke_ref.py         checks HF prefill and incremental decode agree
tools/gen_goldens.py       records golden tensors from the HF reference
tools/dsv4_numpy.py        framework-free NumPy reference for every novel op
tools/verify_numpy.py      checks that reference against the goldens
refs/                      upstream sources, not checked in (see below)
goldens/                   generated test fixtures, not checked in
```

## Setup

Host-side tooling only; no Tenstorrent runtime is involved. Needs Python 3.13 —
3.14 has no torch wheels yet.

```bash
python3.13 -m pip install --target=.pydeps -r requirements.txt
export PYTHONPATH=.pydeps
```

A virtualenv works equally well; `--target` is used here only because the sandbox on
this machine blocks `venv` creation.

Fetch the reference material (~600 MB, not vendored):

```bash
./scripts/fetch_refs.sh
```

## Running the analysis

```bash
python3.13 tools/budget.py               # parameter counts, footprint, KV, roofline
python3.13 tools/expert_cache_sim.py     # residency policy sweep
python3.13 tools/expert_cache_sim.py --batch 8
```

`budget.py` derives everything from the published `config.json` rather than from
hardcoded numbers. It reproduces the published 284 B parameter count to within 0.1% and
the 13 B activated count to within 2%, which is the check that the module inventory is
right.

## Reference and goldens

```bash
python3.13 tools/smoke_ref.py       # HF reference sanity check
python3.13 tools/gen_goldens.py     # record goldens -> goldens/
python3.13 tools/verify_numpy.py    # check the NumPy reference against them
```

`tiny_config.py` defines a ~1.8 M parameter DeepSeek-V4 that shrinks the widths and
depths but preserves every structural feature: all three attention types, both MoE
routing modes, the mHC residual streams with the real 20 Sinkhorn iterations, and the
grouped output projection. `gen_goldens.py` runs it and records the inputs, weights and
outputs of all 230 submodule invocations, in both the prefill and single-token decode
regimes — the compressors take genuinely different code paths in the two, so both are
needed.

`dsv4_numpy.py` then reimplements each operator as a plain function over arrays,
decomposed into the primitives TT-NN actually exposes. It exists because the PyTorch
model is not a usable brief for a kernel author: its behaviour is spread across five
other model files and hidden in cache objects. `verify_numpy.py` closes the loop,
confirming the reimplementation matches the recorded goldens to ~3e-7 relative across
all three attention types, both compressors, the Lightning Indexer's top-k selection,
the Sinkhorn projection, both routers, and the experts.

## What needs hardware

Everything from Phase 2 onward. In rough priority order, the first things to run on a
real QuietBox 2:

1. **Record a routing trace** from the real checkpoint and feed it to
   `expert_cache_sim.py --trace`. This resolves the single largest open question — the
   achievable expert cache hit rate — and needs only a forward pass, not a TT model.
2. Confirm the SDPA kernels accept `head_dim=512`. Most attention kernels assume ≤128 or
   ≤256; the HF reference disables FlashAttention for exactly this reason.
3. Measure actual Warp400 link bandwidth, which decides whether expert-parallel
   all-to-all is free or a bottleneck.
