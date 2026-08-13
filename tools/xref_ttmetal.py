"""Cross-check tt-metal's vendored V4 reference against the HuggingFace one.

`tools/dsv4_numpy.py` is the spec our TT kernels will be written to, and it was verified
against HuggingFace's `transformers.models.deepseek_v4`. But Tenstorrent's own op tests
validate against a *separate* vendored copy at
`models/demos/deepseek_v3_d_p/reference/deepseek_v4/`. If those two implementations
disagree numerically, then every golden in `goldens/` encodes the wrong semantics and we
would not find out until kernels started failing tests we did not write.

Reading the diff says the copies differ only in vendoring mechanics (import rewrites,
`nn.Buffer` vs `register_buffer`, a cache-plumbing refactor). This checks that claim by
running the same weights through both.

Run:
  PYTHONPATH=.pydeps:tools:refs/tt-metal/models/demos/deepseek_v3_d_p/reference \
    python3 tools/xref_ttmetal.py
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

import tiny_config

TOL = 1e-5


def build_pair():
    """Same tiny config and same weights, one model from each implementation."""
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM as HFModel

    from deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as TTConfig
    from deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as TTModel

    torch.manual_seed(0)
    hf_cfg = tiny_config.build()
    hf = HFModel(hf_cfg).to(torch.float32).eval()

    tt_cfg = TTConfig(**tiny_config.TINY_SMALL_HCA)
    tt = TTModel(tt_cfg).to(torch.float32).eval()

    # The hash router's tid2eid is a frozen lookup that ships as zeros; leaving it at
    # zeros would route every token to expert 0 and hide any disagreement in that path.
    g = torch.Generator().manual_seed(1234)
    sd = hf.state_dict()
    for name, buf in sd.items():
        if name.endswith("tid2eid"):
            sd[name] = torch.randint(0, hf_cfg.n_routed_experts, buf.shape, generator=g)
    hf.load_state_dict(sd)

    missing, unexpected = tt.load_state_dict(sd, strict=False)
    return hf, tt, hf_cfg, missing, unexpected


def compare(tag: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    err = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), 1e-12)
    rel = err / scale
    ok = rel < TOL
    print(f"  {tag:28s} max_abs={err:.3e}  rel={rel:.3e}  {'ok' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    hf, tt, cfg, missing, unexpected = build_pair()

    print("=" * 72)
    print("tt-metal vendored V4 reference  vs  HuggingFace V4 reference")
    print("=" * 72)
    print(f"tiny config: {cfg.num_hidden_layers} layers, {cfg.n_routed_experts} experts, "
          f"top-{cfg.num_experts_per_tok}, hidden {cfg.hidden_size}")
    if missing or unexpected:
        print(f"state_dict missing={len(missing)} unexpected={len(unexpected)}")
        for n in list(missing)[:8]:
            print(f"    missing:    {n}")
        for n in list(unexpected)[:8]:
            print(f"    unexpected: {n}")
    else:
        print("state_dict: identical key sets, weights copied exactly")
    print()

    g = torch.Generator().manual_seed(7)
    seq = 40
    ids = torch.randint(0, cfg.vocab_size, (1, seq), generator=g)

    oks = []
    with torch.no_grad():
        print("prefill (full sequence):")
        oks.append(compare("logits", hf(ids).logits, tt(ids).logits))

        # The vendored copy cannot bootstrap its own cache (see the note printed
        # below), so hand both models a cache explicitly rather than letting the
        # first forward create one. That isolates the bug from the question we are
        # actually asking, which is whether the *math* agrees.
        print("\ncache bootstrap check (use_cache=True, no cache passed in):")
        boot = tt(ids[:, :4], use_cache=True).past_key_values
        print(f"  tt-metal returns past_key_values={boot!r}"
              f"  {'<- BUG' if boot is None else ''}")
        print(f"  HuggingFace returns "
              f"{type(hf(ids[:, :4], use_cache=True).past_key_values).__name__}")

        # Incremental decode exercises the CSA overlap state and HCA pooling across a
        # cache boundary, which is where a cache-plumbing refactor would show up.
        print("\nincremental decode (prefix 32, then 8 single steps), cache pre-seeded:")
        prefix = 32
        hf_past = DynamicCache(config=hf.config)
        tt_past = DynamicCache(config=tt.config)
        hf_out = hf(ids[:, :prefix], past_key_values=hf_past, use_cache=True)
        tt_out = tt(ids[:, :prefix], past_key_values=tt_past, use_cache=True)
        oks.append(compare("prefill logits", hf_out.logits, tt_out.logits))
        hf_past, tt_past = hf_out.past_key_values, tt_out.past_key_values

        for i in range(prefix, seq):
            step = ids[:, i : i + 1]
            h = hf(step, past_key_values=hf_past, use_cache=True)
            t = tt(step, past_key_values=tt_past, use_cache=True)
            hf_past, tt_past = h.past_key_values, t.past_key_values
            oks.append(compare(f"step {i - prefix}", h.logits, t.logits))

    print("\n" + "-" * 72)
    print(f"RESULT: {sum(oks)}/{len(oks)} comparisons agree "
          f"(rel tolerance {TOL:g})  ->  {'PASS' if all(oks) else 'FAIL'}")
    if all(oks):
        print("The vendored copy is numerically the HuggingFace model, so goldens\n"
              "generated from HF are valid against Tenstorrent's own op tests.")


if __name__ == "__main__":
    main()
