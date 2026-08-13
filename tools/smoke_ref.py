"""Smoke-test the HuggingFace DeepSeek-V4 reference at tiny scale.

Confirms we can build the model, run prefill and incremental decode, and that the
two agree — the invariant every Tenstorrent kernel will later be checked against.

Run:  PYTHONPATH=.pydeps python3.13 tools/smoke_ref.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_config import build  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = build()
    print(f"transformers {__import__('transformers').__version__}  torch {torch.__version__}")
    print(f"layer_types      : {cfg.layer_types}")
    print(f"mlp_layer_types  : {cfg.mlp_layer_types}")
    print(f"compress_rates   : {cfg.compress_rates}")
    print(f"qk_rope_head_dim : {cfg.qk_rope_head_dim}  (head_dim {cfg.head_dim})")

    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM(cfg).to(torch.float32).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params           : {n_params / 1e6:.2f} M")

    # The hash router's token->expert table is a zero-filled buffer at init, which
    # would send every token to expert 0; randomize so the hash path is exercised.
    g = torch.Generator().manual_seed(1)
    for name, buf in model.named_buffers():
        if name.endswith("tid2eid"):
            buf.copy_(torch.randint(0, cfg.n_routed_experts, buf.shape, generator=g))

    seq = 40
    ids = torch.randint(0, cfg.vocab_size, (1, seq), generator=g)

    with torch.no_grad():
        full = model(ids).logits
    print(f"\nprefill logits   : {tuple(full.shape)}  finite={torch.isfinite(full).all().item()}")

    # Prefill a prefix, then decode the rest one token at a time. The compressor
    # carries rolling window state across calls, so this is the real test of whether
    # cached decode reproduces the single-shot result.
    prefix = 32
    with torch.no_grad():
        out = model(ids[:, :prefix], use_cache=True)
        step_logits = [out.logits[:, -1]]
        past = out.past_key_values
        for t in range(prefix, seq):
            out = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits.append(out.logits[:, -1])

    decoded = torch.stack(step_logits, dim=1)
    reference = full[:, prefix - 1 :]
    err = (decoded - reference).abs().max().item()
    rel = err / reference.abs().max().item()
    print(f"decode vs prefill: max_abs_err={err:.3e}  rel={rel:.3e}")
    print("RESULT:", "PASS" if rel < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
