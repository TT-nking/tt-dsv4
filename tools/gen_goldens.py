"""Capture golden input/output tensors for every DeepSeek-V4 submodule.

The Tenstorrent implementation is built op by op, and each op needs something to be
correct *against*. This runs the HuggingFace reference at tiny scale with seeded
weights and records, for every module of interest, its inputs, its parameters and
its outputs. A TTNN kernel is then correct when, fed the recorded inputs and
parameters, it reproduces the recorded outputs.

Both prefill and single-token decode are captured, because the compressors take
genuinely different code paths in the two regimes (decode hits the partial-window
buffering and the cross-call overlap state that prefill never exercises).

Run:  PYTHONPATH=.pydeps python3.13 tools/gen_goldens.py
Out:  goldens/<phase>/<module_path>.npz  +  goldens/manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_config import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "goldens"

# Module class names we want goldens for. Everything novel in V4 is here; the
# plain nn.Linear / embedding layers are omitted because a matmul needs no golden.
CAPTURE = {
    "DeepseekV4RMSNorm",
    "DeepseekV4UnweightedRMSNorm",
    "DeepseekV4GroupedLinear",
    "DeepseekV4HCACompressor",
    "DeepseekV4CSACompressor",
    "DeepseekV4Indexer",
    "DeepseekV4IndexerScorer",
    "DeepseekV4Attention",
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4MLP",
    "DeepseekV4Experts",
    "DeepseekV4TopKRouter",
    "DeepseekV4HashRouter",
    "DeepseekV4SparseMoeBlock",
    "DeepseekV4DecoderLayer",
    "DeepseekV4RotaryEmbedding",
}

# Composite modules own child projections whose weights a standalone reimplementation
# needs, so record their parameters recursively rather than just the direct ones.
CAPTURE_RECURSIVE = {
    "DeepseekV4HCACompressor",
    "DeepseekV4CSACompressor",
    "DeepseekV4Indexer",
    "DeepseekV4IndexerScorer",
    "DeepseekV4Attention",
    "DeepseekV4MLP",
    "DeepseekV4Experts",
    "DeepseekV4TopKRouter",
    "DeepseekV4HashRouter",
}


def to_np(x):
    """Flatten an arbitrary module input/output into recordable arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    if isinstance(x, (int, float, bool)):
        return np.array(x)
    return None


def flatten(prefix, obj, into):
    if isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            flatten(f"{prefix}{i}_", v, into)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}{k}_", v, into)
        return
    arr = to_np(obj)
    if arr is not None:
        into[prefix.rstrip("_")] = arr


class Recorder:
    def __init__(self, model):
        self.model = model
        self.phase = None
        self.records: dict[str, dict[str, np.ndarray]] = {}
        self.handles = []

    def _hook(self, path, cls):
        def fn(module, args, kwargs, output):
            if self.phase is None:
                return
            key = f"{self.phase}/{path}"
            # A module can fire more than once (e.g. the shared rotary embedding);
            # keep the first call so goldens stay deterministic.
            if key in self.records:
                return
            rec: dict[str, np.ndarray] = {}
            flatten("in_", args, rec)
            flatten("kw_", kwargs, rec)
            flatten("out_", output, rec)
            recurse = cls in CAPTURE_RECURSIVE
            for pname, p in module.named_parameters(recurse=recurse):
                rec[f"param_{pname}"] = p.detach().to(torch.float32).cpu().numpy()
            for bname, b in module.named_buffers(recurse=recurse):
                arr = to_np(b)
                if arr is not None:
                    rec[f"buffer_{bname}"] = arr
            rec["_class"] = np.array(cls)
            self.records[key] = rec

        return fn

    def attach(self):
        for path, module in self.model.named_modules():
            cls = type(module).__name__
            if cls in CAPTURE:
                self.handles.append(module.register_forward_hook(self._hook(path, cls), with_kwargs=True))

    def detach(self):
        for h in self.handles:
            h.remove()


def main() -> None:
    torch.manual_seed(0)
    cfg = build()
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM(cfg).to(torch.float32).eval()

    g = torch.Generator().manual_seed(1)
    for name, buf in model.named_buffers():
        if name.endswith("tid2eid"):
            buf.copy_(torch.randint(0, cfg.n_routed_experts, buf.shape, generator=g))
        if name.endswith("e_score_correction_bias"):
            buf.copy_(torch.randn(buf.shape, generator=g) * 0.1)

    rec = Recorder(model)
    rec.attach()

    seq, prefix = 40, 32
    ids = torch.randint(0, cfg.vocab_size, (1, seq), generator=g)

    with torch.no_grad():
        # Phase 1: prefill. Full-sequence path; windows close in bulk.
        rec.phase = "prefill"
        out = model(ids[:, :prefix], use_cache=True)
        past = out.past_key_values
        prefill_logits = out.logits

        # Phase 2: decode. Single token against warm cache; exercises the partial-window
        # buffer and the CSA cross-call overlap state.
        rec.phase = "decode"
        out2 = model(ids[:, prefix : prefix + 1], past_key_values=past, use_cache=True)
        decode_logits = out2.logits
        rec.phase = None

    rec.detach()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": cfg.to_dict(),
        "input_ids": ids.tolist(),
        "prefill_len": prefix,
        "modules": {},
    }

    for key, arrays in sorted(rec.records.items()):
        path = OUT / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        cls = str(arrays.pop("_class"))
        np.savez_compressed(path, **arrays)
        manifest["modules"][key] = {
            "class": cls,
            "file": str(path.relative_to(OUT)),
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        }

    np.savez_compressed(
        OUT / "model_io.npz",
        input_ids=ids.numpy(),
        prefill_logits=prefill_logits.detach().numpy(),
        decode_logits=decode_logits.detach().numpy(),
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    by_class: dict[str, int] = {}
    for meta in manifest["modules"].values():
        by_class[meta["class"]] = by_class.get(meta["class"], 0) + 1

    print(f"wrote {len(manifest['modules'])} module goldens to {OUT}")
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {cls}")


if __name__ == "__main__":
    main()
