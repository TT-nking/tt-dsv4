"""A small DeepSeek-V4 config that keeps every structural feature of V4-Flash.

Bring-up runs against this instead of the 284B checkpoint: it exercises all three
attention types, both MoE router types, the mHC residual streams and the grouped
output projection, while staying small enough to run on a laptop CPU in seconds.

Shrunk:   vocab, hidden, layer count, expert count, head count/dim.
Preserved: layer schedule shape (sliding -> CSA/HCA interleave), hash-MoE bootstrap,
           hc_mult / sinkhorn iters, compress rates, the nope|rope split ratio,
           and every architectural constant that changes the *math* rather than
           the size.
"""

from __future__ import annotations

TINY = dict(
    vocab_size=512,
    hidden_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    q_lora_rank=32,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    o_groups=2,
    o_lora_rank=16,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=4,
    sliding_window=8,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rms_norm_eps=1e-6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    norm_topk_prob=True,
    tie_word_embeddings=False,
    torch_dtype="float32",
    # Legacy-style kwargs, folded in by DeepseekV4Config.__post_init__. Layer 0/1 are
    # sliding-only, then CSA (m=4) / HCA (m'=128) alternate, matching V4-Flash's
    # published `compress_ratios`.
    compress_ratios=[0, 0, 4, 128, 4, 128],
    num_hash_layers=2,
    qk_rope_head_dim=8,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 256,
        "type": "yarn",
    },
)

# HCA's compress rate (128) exceeds this config's sequence lengths, which would leave
# every HCA layer with zero compressed entries and silently skip that code path.
TINY_SMALL_HCA = {**TINY, "compress_rate_hca": 16}


def build(overrides: dict | None = None):
    from transformers.models.deepseek_v4 import DeepseekV4Config

    cfg = dict(TINY_SMALL_HCA)
    if overrides:
        cfg.update(overrides)
    return DeepseekV4Config(**cfg)
