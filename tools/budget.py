"""Parameter, memory and roofline budget for DeepSeek-V4-Flash on TT-QuietBox 2.

Enumerates every weight tensor of the model from the published config, assigns a
storage dtype per tensor class, and reports whether the result fits in the
QuietBox 2 GDDR6 pool. Also models KV cache growth and a decode-step roofline.

Run:  python3 tools/budget.py [--config refs/hf_dsv4/config.json]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Hardware: TT-QuietBox 2 (Blackhole), 4x ASIC across 2x p300c cards.
# Sources: docs.tenstorrent.com QB2 guide + p300c card spec.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    name: str = "TT-QuietBox 2 (Blackhole)"
    chips: int = 4
    tensix_per_chip: int = 120
    ai_clock_ghz: float = 1.35
    sram_mb_per_chip: int = 180
    dram_gb_per_chip: int = 32
    # p300c card = 2 chips, 1024 GB/s per card -> 512 GB/s per chip.
    dram_bw_gbps_per_chip: float = 512.0
    # Marketing figure 2654 TFLOP/s BlockFP8 across all 4 chips.
    blockfp8_tflops_total: float = 2654.0
    host_dram_gb: int = 256
    # DDR5-5600 dual channel = 2 * 5600 MT/s * 8 B = 89.6 GB/s theoretical.
    host_dram_bw_gbps: float = 89.6
    # PCIe Gen5 x16 ~= 63 GB/s usable per card, 2 cards.
    pcie_gbps_per_card: float = 63.0
    cards: int = 2

    @property
    def dram_total_gb(self) -> float:
        return self.chips * self.dram_gb_per_chip

    @property
    def dram_bw_total_gbps(self) -> float:
        return self.chips * self.dram_bw_gbps_per_chip

    @property
    def sram_total_mb(self) -> float:
        return self.chips * self.sram_mb_per_chip

    @property
    def host_to_device_gbps(self) -> float:
        """Weight streaming is capped by whichever of host DRAM / PCIe is slower."""
        return min(self.host_dram_bw_gbps, self.pcie_gbps_per_card * self.cards)


QB2 = Hardware()

# --------------------------------------------------------------------------
# Storage formats. Tenstorrent block-float formats carry one shared 8-bit
# exponent per 16-element block, so the effective width is `mantissa + 0.5`.
# --------------------------------------------------------------------------

BITS_PER_ELEM = {
    "fp32": 32.0,
    "bf16": 16.0,
    "fp16": 16.0,
    # DeepSeek's shipped formats (reference only -- not all are Blackhole-legal).
    "fp8": 8.0 + 8.0 / (128 * 128),  # e4m3 + ue8m0 scale per 128x128 block
    "fp4": 4.0 + 8.0 / 128,  # e2m1 + ue8m0 scale per 128 elements
    "mxfp4": 4.25,  # e2m1 + e8m0 per 32 -- QUASAR ONLY, not Blackhole
    # Tenstorrent block float: 8-bit shared exponent per 16-element block.
    "bfp8_b": 8.5,
    "bfp4_b": 4.5,
    "bfp2_b": 2.5,
    "fp8_e4m3": 8.0,  # native on Blackhole (not on Wormhole)
}

# Formats a *tensor* can actually be stored in, which is a narrower set than the
# formats the hardware can unpack. `tt::DataFormat` (LLK level) includes Bfp2_b and
# `is_supported_wormhole_blackhole` accepts it, but `tt::tt_metal::DataType` has no
# BFLOAT2_B entry and `datatype_to_dataformat_converter`
# (tt_metal/impl/tensor/tensor_types.cpp:91) has no case producing Bfp2_b — so there
# is no way to construct a 2-bit tensor. Weight storage is bounded by DataType, not
# DataFormat, which is why bfp2_b is excluded here.
BLACKHOLE_LEGAL = {"fp32", "bf16", "fp16", "bfp8_b", "bfp4_b", "fp8_e4m3"}

# Reachable only if someone adds BFLOAT2_B plumbing to ttnn. Kept in the model
# because it is the only packing that fits, so it quantifies what that work buys.
NEEDS_NEW_DTYPE_SUPPORT = {"bfp2_b"}


def bytes_of(numel: int, dtype: str) -> float:
    return numel * BITS_PER_ELEM[dtype] / 8.0


# --------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------

RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}


@dataclass
class Tensor:
    name: str
    numel: int
    cls: str  # attn | expert | shared_expert | router | embed | compressor | indexer | mhc | norm
    layer: int | None = None


@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    sliding_window: int
    hc_mult: int
    qk_rope_head_dim: int
    layer_kinds: list[str] = field(default_factory=list)
    compress_rates: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text())
        n = raw["num_hidden_layers"]
        kinds = [RATIO_TO_TYPE[r] for r in raw["compress_ratios"][:n]]
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            moe_intermediate_size=raw["moe_intermediate_size"],
            num_hidden_layers=n,
            num_attention_heads=raw["num_attention_heads"],
            head_dim=raw["head_dim"],
            q_lora_rank=raw["q_lora_rank"],
            o_groups=raw["o_groups"],
            o_lora_rank=raw["o_lora_rank"],
            n_routed_experts=raw["n_routed_experts"],
            n_shared_experts=raw["n_shared_experts"],
            num_experts_per_tok=raw["num_experts_per_tok"],
            index_n_heads=raw["index_n_heads"],
            index_head_dim=raw["index_head_dim"],
            index_topk=raw["index_topk"],
            sliding_window=raw["sliding_window"],
            hc_mult=raw["hc_mult"],
            qk_rope_head_dim=raw["qk_rope_head_dim"],
            layer_kinds=kinds,
            compress_rates={"csa": 4, "hca": 128},
        )


def enumerate_tensors(c: Config) -> list[Tensor]:
    """One entry per weight tensor, mirroring modeling_deepseek_v4.py."""
    t: list[Tensor] = []
    H, D = c.hidden_size, c.head_dim
    heads_dim = c.num_attention_heads * D  # 32768

    t.append(Tensor("embed_tokens", c.vocab_size * H, "embed"))
    t.append(Tensor("lm_head", c.vocab_size * H, "embed"))
    t.append(Tensor("norm", H, "norm"))
    t.append(Tensor("hyper_head.fn", c.hc_mult * c.hc_mult * H, "mhc"))

    for i, kind in enumerate(c.layer_kinds):
        # --- attention backbone (all layer kinds) ---
        t += [
            Tensor("q_a_proj", H * c.q_lora_rank, "attn", i),
            Tensor("q_b_proj", c.q_lora_rank * heads_dim, "attn", i),
            Tensor("kv_proj", H * D, "attn", i),
            Tensor("o_a_proj", c.o_groups * c.o_lora_rank * (heads_dim // c.o_groups), "attn", i),
            Tensor("o_b_proj", c.o_groups * c.o_lora_rank * H, "attn", i),
            Tensor("q_a_norm", c.q_lora_rank, "norm", i),
            Tensor("kv_norm", D, "norm", i),
            Tensor("sinks", c.num_attention_heads, "norm", i),
        ]

        # --- long-range compressor ---
        if kind == "hca":
            m = c.compress_rates["hca"]
            t += [
                Tensor("hca.kv_proj", H * D, "compressor", i),
                Tensor("hca.gate_proj", H * D, "compressor", i),
                Tensor("hca.position_bias", m * D, "compressor", i),
                Tensor("hca.kv_norm", D, "norm", i),
            ]
        elif kind == "csa":
            m = c.compress_rates["csa"]
            ih = c.index_head_dim
            t += [
                Tensor("csa.kv_proj", H * 2 * D, "compressor", i),
                Tensor("csa.gate_proj", H * 2 * D, "compressor", i),
                Tensor("csa.position_bias", m * 2 * D, "compressor", i),
                Tensor("csa.kv_norm", D, "norm", i),
                Tensor("idx.kv_proj", H * 2 * ih, "indexer", i),
                Tensor("idx.gate_proj", H * 2 * ih, "indexer", i),
                Tensor("idx.position_bias", m * 2 * ih, "indexer", i),
                Tensor("idx.q_b_proj", c.q_lora_rank * c.index_n_heads * ih, "indexer", i),
                Tensor("idx.weights_proj", H * c.index_n_heads, "indexer", i),
                Tensor("idx.kv_norm", ih, "norm", i),
            ]

        # --- MoE ---
        I = c.moe_intermediate_size
        t += [
            Tensor("experts.gate_up_proj", c.n_routed_experts * 2 * I * H, "expert", i),
            Tensor("experts.down_proj", c.n_routed_experts * H * I, "expert", i),
            Tensor("shared.gate_up_proj", c.n_shared_experts * 2 * I * H, "shared_expert", i),
            Tensor("shared.down_proj", c.n_shared_experts * H * I, "shared_expert", i),
            Tensor("router.weight", c.n_routed_experts * H, "router", i),
        ]

        # --- mHC + norms ---
        mix = (2 + c.hc_mult) * c.hc_mult
        t += [
            Tensor("attn_hc.fn", mix * c.hc_mult * H, "mhc", i),
            Tensor("ffn_hc.fn", mix * c.hc_mult * H, "mhc", i),
            Tensor("input_layernorm", H, "norm", i),
            Tensor("post_attention_layernorm", H, "norm", i),
        ]
    return t


# --------------------------------------------------------------------------
# Quantization schemes: class -> dtype
# --------------------------------------------------------------------------

SCHEMES: dict[str, dict[str, str]] = {
    # Reference point: what DeepSeek ships (fp4 experts / fp8 dense). NOT Blackhole-legal.
    "[ref] as-shipped fp4 experts / fp8 dense": {
        "expert": "fp4", "shared_expert": "fp8", "attn": "fp8", "compressor": "fp8",
        "indexer": "fp4", "router": "bf16", "embed": "bf16", "mhc": "bf16", "norm": "bf16",
    },
    # Reference point: if Blackhole had MXFP4 (it does not; Quasar does).
    "[ref] mxfp4 experts / fp8_e4m3 dense": {
        "expert": "mxfp4", "shared_expert": "fp8_e4m3", "attn": "fp8_e4m3", "compressor": "fp8_e4m3",
        "indexer": "mxfp4", "router": "bf16", "embed": "bf16", "mhc": "bf16", "norm": "bf16",
    },
    # --- Blackhole-legal ---
    "bfp4_b experts / bfp8_b dense": {
        "expert": "bfp4_b", "shared_expert": "bfp8_b", "attn": "bfp8_b", "compressor": "bfp8_b",
        "indexer": "bfp4_b", "router": "bf16", "embed": "bf16", "mhc": "bf16", "norm": "bf16",
    },
    "bfp4_b experts / fp8_e4m3 dense": {
        "expert": "bfp4_b", "shared_expert": "fp8_e4m3", "attn": "fp8_e4m3", "compressor": "fp8_e4m3",
        "indexer": "bfp4_b", "router": "bf16", "embed": "fp8_e4m3", "mhc": "bf16", "norm": "bf16",
    },
    "bfp4_b experts + bfp4_b attn": {
        "expert": "bfp4_b", "shared_expert": "bfp8_b", "attn": "bfp4_b", "compressor": "bfp8_b",
        "indexer": "bfp4_b", "router": "bf16", "embed": "fp8_e4m3", "mhc": "bf16", "norm": "bf16",
    },
    # Only packing that fits, but needs BFLOAT2_B tensor support that does not exist.
    "[blocked] bfp2_b experts / bfp8_b dense": {
        "expert": "bfp2_b", "shared_expert": "bfp8_b", "attn": "bfp8_b", "compressor": "bfp8_b",
        "indexer": "bfp4_b", "router": "bf16", "embed": "bf16", "mhc": "bf16", "norm": "bf16",
    },
}

GB = 1024**3


def summarize(c: Config, tensors: list[Tensor], scheme: dict[str, str]) -> dict:
    by_cls: dict[str, list[float]] = {}
    for t in tensors:
        n, b = by_cls.setdefault(t.cls, [0.0, 0.0]), None
        n[0] += t.numel
        n[1] += bytes_of(t.numel, scheme[t.cls])
    return by_cls


# --------------------------------------------------------------------------
# KV cache model
# --------------------------------------------------------------------------


def kv_cache_bytes(c: Config, seq_len: int, kv_dtype_bits: float = 8.0) -> dict[str, float]:
    """KV bytes for one sequence at `seq_len` tokens.

    RoPE dims are stored bf16 and the rest fp8 (paper 2.3.4), so the effective
    width is a blend over head_dim.
    """
    D, rd = c.head_dim, c.qk_rope_head_dim
    blended_bits = ((D - rd) * kv_dtype_bits + rd * 16.0) / D
    out = {"sliding": 0.0, "csa_pool": 0.0, "csa_indexer": 0.0, "hca_pool": 0.0}

    for kind in c.layer_kinds:
        # Every layer keeps the raw sliding window (K==V, one head).
        out["sliding"] += c.sliding_window * D * blended_bits / 8.0
        if kind == "csa":
            entries = seq_len // c.compress_rates["csa"]
            out["csa_pool"] += entries * D * blended_bits / 8.0
            ih = c.index_head_dim
            idx_bits = ((ih - rd) * 4.0 + rd * 16.0) / ih  # indexer QK path is fp4
            out["csa_indexer"] += entries * ih * idx_bits / 8.0
        elif kind == "hca":
            entries = seq_len // c.compress_rates["hca"]
            out["hca_pool"] += entries * D * blended_bits / 8.0
    return out


# --------------------------------------------------------------------------
# Decode roofline
# --------------------------------------------------------------------------


def dense_bytes(tensors: list[Tensor], scheme: dict[str, str]) -> float:
    return sum(bytes_of(t.numel, scheme[t.cls]) for t in tensors if t.cls != "expert")


def expert_bytes(tensors: list[Tensor], scheme: dict[str, str]) -> float:
    return sum(bytes_of(t.numel, scheme[t.cls]) for t in tensors if t.cls == "expert")


def demote_fraction_to_fit(c: Config, tensors: list[Tensor], scheme: dict[str, str],
                           lo_dtype: str, budget_bytes: float) -> float:
    """Fraction of experts that must drop from the scheme dtype to `lo_dtype` to fit.

    Returns 0.0 if it already fits, or >1.0 if even full demotion is not enough.
    """
    hi_b = expert_bytes(tensors, scheme)
    lo_b = hi_b * BITS_PER_ELEM[lo_dtype] / BITS_PER_ELEM[scheme["expert"]]
    room = budget_bytes - dense_bytes(tensors, scheme)
    if hi_b <= room:
        return 0.0
    if hi_b == lo_b:
        return float("inf")
    return (hi_b - room) / (hi_b - lo_b)


def decode_step(c: Config, tensors: list[Tensor], scheme: dict[str, str], batch: int,
                resident_expert_frac: float, hw: Hardware = QB2) -> dict[str, float]:
    """Bytes crossing each bus for one decode step, and the resulting time.

    Dense weights are always device-resident and read once per step regardless of
    batch. Expert weights are read once per *distinct* expert touched by the
    batch; with balanced routing the distinct count is the coupon-collector
    expectation. A `1 - resident_expert_frac` share of those reads misses device
    DRAM and must be streamed over PCIe from host memory.
    """
    dense = dense_bytes(tensors, scheme)

    E, k = c.n_routed_experts, c.num_experts_per_tok
    expected_distinct = E * (1.0 - (1.0 - 1.0 / E) ** (batch * k))

    per_expert_numel = 2 * c.moe_intermediate_size * c.hidden_size + c.hidden_size * c.moe_intermediate_size
    per_expert_b = bytes_of(per_expert_numel, scheme["expert"])
    touched = expected_distinct * per_expert_b * len(c.layer_kinds)

    # Every touched expert is ultimately read from device memory; a streamed one
    # additionally has to be written there first, so it costs device bandwidth
    # twice over plus the PCIe transfer. This keeps full residency as the true
    # upper bound.
    streamed = touched * (1.0 - resident_expert_frac)
    on_chip = dense + touched + streamed

    t_chip_ms = on_chip / (hw.dram_bw_total_gbps * 1e9) * 1e3
    t_stream_ms = streamed / (hw.host_to_device_gbps * 1e9) * 1e3
    step_ms = max(t_chip_ms, t_stream_ms)  # assume perfect prefetch overlap
    return {"distinct": expected_distinct, "touched": touched, "on_chip": on_chip,
            "streamed": streamed, "t_chip_ms": t_chip_ms, "t_stream_ms": t_stream_ms,
            "step_ms": step_ms, "tok_s": batch / (step_ms / 1e3)}


def fmt_gb(b: float) -> str:
    return f"{b / GB:8.2f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "refs/hf_dsv4/config.json"))
    args = ap.parse_args()

    c = Config.load(Path(args.config))
    tensors = enumerate_tensors(c)
    total_params = sum(t.numel for t in tensors)

    n_csa = c.layer_kinds.count("csa")
    n_hca = c.layer_kinds.count("hca")
    n_sw = c.layer_kinds.count("sliding")

    print("=" * 78)
    print("DeepSeek-V4-Flash  x  TT-QuietBox 2 (Blackhole)  --  budget model")
    print("=" * 78)
    print(f"\nLayers: {c.num_hidden_layers}  ({n_sw} sliding-only, {n_csa} CSA m=4, {n_hca} HCA m'=128)")
    print(f"Total parameters: {total_params / 1e9:.2f} B   (published: 284 B)")

    print("\n--- parameters by class ---")
    by_cls_n: dict[str, int] = {}
    for t in tensors:
        by_cls_n[t.cls] = by_cls_n.get(t.cls, 0) + t.numel
    for cls, n in sorted(by_cls_n.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:16s} {n / 1e9:9.3f} B  ({100 * n / total_params:5.2f}%)")

    # Activated params per token.
    per_layer_expert = 2 * c.moe_intermediate_size * c.hidden_size + c.hidden_size * c.moe_intermediate_size
    act = (by_cls_n["attn"] + by_cls_n["compressor"] + by_cls_n["indexer"]
           + by_cls_n["shared_expert"] + by_cls_n["router"] + by_cls_n["mhc"]
           + c.num_experts_per_tok * per_layer_expert * c.num_hidden_layers)
    print(f"\nActivated params / token: {act / 1e9:.2f} B   (published: 13 B)")

    print("\n--- weight footprint by scheme ---")
    print(f"{'scheme':42s} {'experts':>11s} {'dense':>11s} {'total':>11s}  fits 128GB?")
    for name, scheme in SCHEMES.items():
        used = set(scheme.values())
        blocked = used & NEEDS_NEW_DTYPE_SUPPORT
        illegal = used - BLACKHOLE_LEGAL - NEEDS_NEW_DTYPE_SUPPORT
        exp_b = expert_bytes(tensors, scheme)
        tot_b = exp_b + dense_bytes(tensors, scheme)
        usable = QB2.dram_total_gb * GB
        verdict = "YES" if tot_b < usable else f"NO  (over by {(tot_b - usable) / GB:.1f} GB)"
        if illegal:
            verdict += "   [not BH-legal: " + ",".join(sorted(illegal)) + "]"
        if blocked:
            verdict += "   [no ttnn DataType: " + ",".join(sorted(blocked)) + "]"
        print(f"{name:42s} {fmt_gb(exp_b)} {fmt_gb(tot_b - exp_b)} {fmt_gb(tot_b)}  {verdict}")

    print(f"\nQB2 device DRAM: {QB2.dram_total_gb:.0f} GB   host DRAM: {QB2.host_dram_gb} GB"
          f"   host->device: {QB2.host_to_device_gbps:.1f} GB/s")

    print("\n--- KV cache (single sequence) ---")
    print(f"{'context':>10s} {'sliding':>10s} {'csa pool':>10s} {'csa idx':>10s} {'hca pool':>10s} {'total':>11s}")
    for L in (8192, 65536, 262144, 1048576):
        kv = kv_cache_bytes(c, L)
        tot = sum(kv.values())
        print(f"{L:>10,d} {kv['sliding'] / 1e6:9.1f}M {kv['csa_pool'] / 1e6:9.1f}M "
              f"{kv['csa_indexer'] / 1e6:9.1f}M {kv['hca_pool'] / 1e6:9.1f}M {tot / GB:10.3f} GB")

    # Runtime reserve: 1M-token KV for a few sequences, activations, buffers.
    reserve_gb = 12.0
    weight_budget = (QB2.dram_total_gb - reserve_gb) * GB
    print(f"\n--- residency (device budget {QB2.dram_total_gb - reserve_gb:.0f} GB "
          f"after {reserve_gb:.0f} GB runtime reserve) ---")
    print(f"{'scheme':42s} {'resident E':>11s}  {'demote to bfp2_b to fit':>24s}")
    for name, scheme in SCHEMES.items():
        room = weight_budget - dense_bytes(tensors, scheme)
        eb = expert_bytes(tensors, scheme)
        frac = min(1.0, room / eb)
        dem = demote_fraction_to_fit(c, tensors, scheme, "bfp2_b", weight_budget)
        dem_s = "not needed" if dem == 0 else (f"{dem * 100:.0f}% of experts" if dem <= 1 else "impossible")
        print(f"{name:42s} {frac * 100:>10.1f}%  {dem_s:>24s}")

    print("\n--- decode roofline: bfp4_b experts / fp8_e4m3 dense (Blackhole-legal) ---")
    scheme = SCHEMES["bfp4_b experts / fp8_e4m3 dense"]
    room = weight_budget - dense_bytes(tensors, scheme)
    rf = min(1.0, room / expert_bytes(tensors, scheme))
    print(f"expert residency {rf * 100:.1f}%  ->  {(1 - rf) * 100:.1f}% of expert reads stream over PCIe")
    print(f"{'batch':>6s} {'distinct E':>11s} {'read/step':>11s} {'on-chip ms':>11s} "
          f"{'stream ms':>11s} {'step ms':>9s} {'tok/s':>9s}")
    for B in (1, 2, 4, 8, 16, 32, 64):
        d = decode_step(c, tensors, scheme, B, rf)
        print(f"{B:>6d} {d['distinct']:>11.1f} {d['touched'] / GB:>8.2f} GB "
              f"{d['t_chip_ms']:>10.2f} {d['t_stream_ms']:>10.2f} {d['step_ms']:>8.2f} {d['tok_s']:>9.1f}")

    print("\n--- same, if everything is made to fit on device (no streaming) ---")
    print(f"{'batch':>6s} {'read/step':>11s} {'step ms':>9s} {'tok/s':>9s}")
    for B in (1, 2, 4, 8, 16, 32, 64):
        d = decode_step(c, tensors, scheme, B, 1.0)
        print(f"{B:>6d} {d['touched'] / GB:>8.2f} GB {d['step_ms']:>8.2f} {d['tok_s']:>9.1f}")

    print("\n(tok/s is a pure-bandwidth ceiling: no compute, kernel launch, or "
          "collective overheads.)")


if __name__ == "__main__":
    main()
