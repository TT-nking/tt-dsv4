"""Check the NumPy reference in `dsv4_numpy.py` against the recorded goldens.

Every op the Tenstorrent port has to implement is exercised here against the
HuggingFace model's actual outputs. If this passes, the NumPy file is a trustworthy
spec; if a TTNN kernel disagrees with it later, the kernel is wrong.

Run:  PYTHONPATH=.pydeps python3.13 tools/verify_numpy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsv4_numpy as ref  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "goldens"
TOL = 2e-5

results: list[tuple[str, str, float, bool]] = []


def check(op: str, name: str, got: np.ndarray, want: np.ndarray) -> None:
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        results.append((op, f"{name} SHAPE {got.shape}!={want.shape}", float("inf"), False))
        return
    finite = np.isfinite(want)
    scale = max(np.abs(want[finite]).max(), 1e-9) if finite.any() else 1.0
    # -inf entries are structural mask values and must land in exactly the same places.
    if (~finite).any() and not np.array_equal(np.isfinite(got), finite):
        results.append((op, f"{name} MASK-MISMATCH", float("inf"), False))
        return
    err = np.abs(got[finite] - want[finite]).max() / scale if finite.any() else 0.0
    results.append((op, name, float(err), bool(err < TOL)))


def load(phase: str, path: str) -> dict[str, np.ndarray]:
    return dict(np.load(GOLD / phase / f"{path}.npz"))


def main() -> None:
    manifest = json.loads((GOLD / "manifest.json").read_text())
    cfg = manifest["config"]
    eps = cfg["rms_norm_eps"]
    hc_mult, hc_iters, hc_eps = cfg["hc_mult"], cfg["hc_sinkhorn_iters"], cfg["hc_eps"]
    head_dim, n_heads = cfg["head_dim"], cfg["num_attention_heads"]
    o_groups = cfg["o_groups"]
    idx_dim, idx_heads = cfg["index_head_dim"], cfg["index_n_heads"]
    topk, n_exp = cfg["num_experts_per_tok"], cfg["n_routed_experts"]
    scaling_f, limit = cfg["routed_scaling_factor"], cfg["swiglu_limit"]
    rates = cfg["compress_rates"]

    mods = manifest["modules"]

    def paths(cls: str, phase: str = "prefill"):
        for key, meta in sorted(mods.items()):
            ph, _, p = key.partition("/")
            if ph == phase and meta["class"] == cls:
                yield p

    # ---- norms -----------------------------------------------------------
    for p in list(paths("DeepseekV4RMSNorm"))[:6]:
        d = load("prefill", p)
        check("rms_norm", p, ref.rms_norm(d["in_0"], d["param_weight"], eps), d["out"])
    for p in list(paths("DeepseekV4UnweightedRMSNorm"))[:6]:
        d = load("prefill", p)
        check("unweighted_rms_norm", p, ref.unweighted_rms_norm(d["in_0"], eps), d["out"])

    # ---- grouped output projection ---------------------------------------
    for p in list(paths("DeepseekV4GroupedLinear"))[:3]:
        d = load("prefill", p)
        check("grouped_linear", p, ref.grouped_linear(d["in_0"], d["param_weight"], o_groups), d["out"])

    # ---- mHC --------------------------------------------------------------
    for p in list(paths("DeepseekV4HyperConnection"))[:4]:
        d = load("prefill", p)
        post, comb, coll = ref.hyper_connection(
            d["in_0"], d["param_fn"], d["param_base"], d["param_scale"], hc_mult, hc_iters, hc_eps, eps
        )
        check("hyper_connection.post", p, post, d["out_0"])
        check("hyper_connection.comb", p, comb, d["out_1"])
        check("hyper_connection.collapsed", p, coll, d["out_2"])
        # The Sinkhorn output must actually be doubly stochastic, else the
        # non-expansive-propagation guarantee the architecture relies on is lost.
        rows = np.abs(comb.sum(-1) - 1.0).max()
        cols = np.abs(comb.sum(-2) - 1.0).max()
        results.append(("sinkhorn.doubly_stochastic", p, float(max(rows, cols)), max(rows, cols) < 1e-3))

    for p in paths("DeepseekV4HyperHead"):
        d = load("prefill", p)
        got = ref.hyper_head(d["in_0"], d["param_hc_fn"], d["param_hc_base"], d["param_hc_scale"], hc_eps, eps)
        check("hyper_head", p, got, d["out"])

    # ---- lightning indexer scorer ----------------------------------------
    for p in paths("DeepseekV4IndexerScorer"):
        d = load("prefill", p)
        got = ref.indexer_score(d["in_0"], d["in_1"], d["in_2"], d["param_weights_proj.weight"], idx_dim, idx_heads)
        check("indexer_score", p, got, d["out"])

    # ---- routers ----------------------------------------------------------
    for p in paths("DeepseekV4TopKRouter"):
        d = load("prefill", p)
        logits, w, idx = ref.topk_router(
            d["in_0"], d["param_weight"], d["buffer_e_score_correction_bias"], topk, scaling_f
        )
        check("topk_router.logits", p, logits, d["out_0"])
        # Selection is a set, not an ordering: torch.topk(sorted=False) makes the
        # order unspecified, so compare as sets and compare weights set-wise too.
        same = np.array_equal(np.sort(idx, -1), np.sort(d["out_2"], -1))
        results.append(("topk_router.indices", p, 0.0 if same else 1.0, same))
        check("topk_router.weights", p, np.sort(w, -1), np.sort(d["out_1"], -1))

    for p in paths("DeepseekV4HashRouter"):
        d = load("prefill", p)
        logits, w, idx = ref.hash_router(
            d["in_0"], d["in_1"], d["param_weight"], d["buffer_tid2eid"], scaling_f
        )
        check("hash_router.logits", p, logits, d["out_0"])
        check("hash_router.weights", p, w, d["out_1"])
        same = np.array_equal(idx, d["out_2"])
        results.append(("hash_router.indices", p, 0.0 if same else 1.0, same))

    # ---- experts ----------------------------------------------------------
    for p in list(paths("DeepseekV4Experts"))[:3]:
        d = load("prefill", p)
        got = ref.experts(d["in_0"], d["in_1"], d["in_2"], d["param_gate_up_proj"], d["param_down_proj"], limit)
        check("experts", p, got, d["out"])

    for p in list(paths("DeepseekV4MLP"))[:3]:
        d = load("prefill", p)
        got = ref.dense_mlp(
            d["in_0"], d["param_gate_proj.weight"], d["param_up_proj.weight"], d["param_down_proj.weight"], limit
        )
        check("dense_mlp", p, got, d["out"])

    # ---- compressors ------------------------------------------------------
    # Prefill only: the cache is empty on entry, so first_window_position is 0 and
    # there is no carried overlap state to reconstruct.
    for p in paths("DeepseekV4HCACompressor"):
        d = load("prefill", p)
        m = rates["heavily_compressed_attention"]
        got = ref.hca_compress(
            d["in_0"], d["param_kv_proj.weight"], d["param_gate_proj.weight"], d["param_position_bias"],
            d["param_kv_norm.weight"], d["buffer_rotary_emb.compress_inv_freq"], m, eps,
        )
        check("hca_compress", p, got[:, None], d["out_0"])
        bias = ref.hca_block_bias(d["in_2"], got.shape[1], m)
        check("hca_block_bias", p, bias, d["out_1"])

    for p in paths("DeepseekV4CSACompressor"):
        d = load("prefill", p)
        m = rates["compressed_sparse_attention"]
        got = ref.csa_compress(
            d["in_0"], d["param_kv_proj.weight"], d["param_gate_proj.weight"], d["param_position_bias"],
            d["param_kv_norm.weight"], d["buffer_rotary_emb.compress_inv_freq"], m, head_dim, eps,
        )
        check("csa_compress", p, got[:, None], d["out_0"])

        # Full indexer path: compress at index_head_dim, score, top-k, block bias.
        idx_c = ref.csa_compress(
            d["in_0"], d["param_indexer.kv_proj.weight"], d["param_indexer.gate_proj.weight"],
            d["param_indexer.position_bias"], d["param_indexer.kv_norm.weight"],
            d["buffer_indexer.rotary_emb.compress_inv_freq"], m, idx_dim, eps,
        )
        pos = d["in_2"]
        cos_q, sin_q = ref.rope_cos_sin(pos, d["buffer_indexer.rotary_emb.compress_inv_freq"])
        B, S = pos.shape
        q = ref.linear(d["in_1"], d["param_indexer.q_b_proj.weight"]).reshape(B, S, -1, idx_dim).transpose(0, 2, 1, 3)
        q = ref.apply_rope(q, cos_q, sin_q).transpose(0, 2, 1, 3)
        scores = ref.indexer_score(q, idx_c, d["in_0"], d["param_indexer.scorer.weights_proj.weight"], idx_dim, idx_heads)
        sel = ref.indexer_topk(scores, pos, m, cfg["index_topk"])
        bias = ref.csa_block_bias(sel, got.shape[1], S)
        check("csa_indexer_block_bias", p, bias, d["out_1"])

    # ---- full attention block --------------------------------------------
    for p in list(paths("DeepseekV4Attention"))[:4]:
        d = load("prefill", p)
        layer = int(p.split(".")[2])
        kind = cfg["layer_types"][layer]
        rope = "main" if kind == "sliding_attention" else "compress"
        cos, sin = d[f"kw_position_embeddings_{rope}_0"], d[f"kw_position_embeddings_{rope}_1"]
        h, pos = d["in_0"], d["kw_position_ids"]
        B, S, _ = h.shape

        q_res = ref.rms_norm(ref.linear(h, d["param_q_a_proj.weight"]), d["param_q_a_norm.weight"], eps)
        q = ref.linear(q_res, d["param_q_b_proj.weight"]).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        q = ref.apply_rope(ref.unweighted_rms_norm(q, eps), cos, sin)

        kv = ref.rms_norm(ref.linear(h, d["param_kv_proj.weight"]), d["param_kv_norm.weight"], eps)
        kv = ref.apply_rope(kv.reshape(B, S, 1, head_dim).transpose(0, 2, 1, 3), cos, sin)

        mask = d["kw_attention_mask"]
        if kind != "sliding_attention":
            comp = load("prefill", f"{p}.compressor")
            kv = np.concatenate([kv, comp["out_0"]], axis=2)
            mask = np.concatenate([mask, comp["out_1"]], axis=-1)

        out = ref.sink_attention(q, kv, kv, mask, head_dim**-0.5, d["param_sinks"])
        out = ref.apply_rope(out.transpose(0, 2, 1, 3), cos, -sin).transpose(0, 2, 1, 3)
        grouped = ref.grouped_linear(out.reshape(B, S, o_groups, -1), d["param_o_a_proj.weight"], o_groups)
        got = ref.linear(grouped.reshape(B, S, -1), d["param_o_b_proj.weight"])
        check(f"attention[{kind}]", p, got, d["out_0"])

    # ---- report -----------------------------------------------------------
    width = max(len(op) for op, *_ in results)
    by_op: dict[str, list[tuple[float, bool]]] = {}
    for op, _, err, ok in results:
        by_op.setdefault(op, []).append((err, ok))

    print(f"{'op':{width}s}  {'n':>3s}  {'max rel err':>12s}   status")
    print("-" * (width + 32))
    n_fail = 0
    for op, rows in by_op.items():
        worst = max(r[0] for r in rows)
        ok = all(r[1] for r in rows)
        n_fail += sum(1 for r in rows if not r[1])
        print(f"{op:{width}s}  {len(rows):3d}  {worst:12.3e}   {'ok' if ok else 'FAIL'}")

    print("-" * (width + 32))
    total = len(results)
    print(f"{total - n_fail}/{total} checks passed (tolerance {TOL:g})")
    if n_fail:
        print("\nfailures:")
        for op, name, err, ok in results:
            if not ok:
                print(f"  {op:{width}s}  {name}  err={err:.3e}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
