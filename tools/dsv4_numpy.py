"""Framework-free NumPy reference for the DeepSeek-V4 operators.

This is the spec the Tenstorrent kernels are written against. The HuggingFace model
is the ground truth, but it is expressed in PyTorch modules with cache objects and
inherited behaviour spread across five other model files, which is not a usable
brief for someone writing a TTNN kernel. Everything here is a plain function over
plain arrays, decomposed into the primitives TTNN actually exposes (matmul, reduce,
softmax, gather, elementwise), and every function is checked against a recorded
golden by `verify_numpy.py`.

Shape conventions:
    B  batch          S  query positions      T  compressed/key positions
    H  heads          D  head_dim             E  hidden_size
"""

from __future__ import annotations

import numpy as np

FP = np.float32
NEG_INF = -np.inf


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    # An all -inf slice (a window whose slots are entirely masked) would give
    # inf - inf = nan; pin those rows to 0 instead, matching torch's behaviour of
    # producing a uniform-zero row only after the exp.
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(FP)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(FP)


def softplus(x: np.ndarray) -> np.ndarray:
    # log1p(exp(x)) with the standard large-x guard.
    x = x.astype(np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(FP)


def sqrt_softplus(x: np.ndarray) -> np.ndarray:
    """Router scoring function (`scoring_func="sqrtsoftplus"`)."""
    return np.sqrt(softplus(x))


def silu(x: np.ndarray) -> np.ndarray:
    return (x * sigmoid(x)).astype(FP)


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`F.linear`: w is [out, in], contracting over the trailing axis of x."""
    return (x.astype(FP) @ w.astype(FP).T).astype(FP)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    v = x.astype(np.float64)
    return (weight.astype(FP) * (v * (1.0 / np.sqrt(np.mean(v * v, -1, keepdims=True) + eps))).astype(FP)).astype(FP)


def unweighted_rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    v = x.astype(np.float64)
    return (v * (1.0 / np.sqrt(np.mean(v * v, -1, keepdims=True) + eps))).astype(FP)


# ---------------------------------------------------------------------------
# rotary
# ---------------------------------------------------------------------------


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Interleaved convention (GLM's): pairs channels (0,1), (2,3), ..."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack((-x2, x1), axis=-1).reshape(*x.shape[:-1], -1)


def rope_cos_sin(positions: np.ndarray, inv_freq: np.ndarray, attention_scaling: float = 1.0):
    """cos/sin at half width — one entry per interleaved pair.

    `positions` is [B, S]; result is [B, S, rope_dim // 2]. V4 forces
    `attention_factor=1.0` on the YaRN branch, so the scaling is 1 in both configs.
    """
    freqs = positions.astype(np.float64)[..., None] * inv_freq.astype(np.float64)[None, None, :]
    return (np.cos(freqs) * attention_scaling).astype(FP), (np.sin(freqs) * attention_scaling).astype(FP)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, unsqueeze_dim: int = 1) -> np.ndarray:
    """Rotate only the trailing `rope_dim` channels; leave the leading nope part alone.

    V4 lays each head out as [nope | rope], so the rotation applies to `x[..., -rd:]`.
    """
    cos = np.expand_dims(np.repeat(cos, 2, axis=-1), unsqueeze_dim)
    sin = np.expand_dims(np.repeat(sin, 2, axis=-1), unsqueeze_dim)
    rd = cos.shape[-1]
    nope, rope = x[..., :-rd], x[..., -rd:]
    rotated = (rope.astype(FP) * cos + rotate_half(rope).astype(FP) * sin).astype(FP)
    return np.concatenate([nope, rotated], axis=-1)


# ---------------------------------------------------------------------------
# manifold-constrained hyper-connections (paper 2.2)
# ---------------------------------------------------------------------------


def sinkhorn(comb: np.ndarray, iters: int, eps: float) -> np.ndarray:
    """Project a positive matrix onto the doubly-stochastic manifold.

    Note the asymmetry: the first normalisation is over axis -2 (columns), then each
    subsequent iteration does rows-then-columns. The result is doubly stochastic but
    NOT symmetric, so the consuming matmul's transpose direction matters.
    """
    comb = comb / (np.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (np.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (np.sum(comb, axis=-2, keepdims=True) + eps)
    return comb.astype(FP)


def hyper_connection(hidden_streams, fn, base, scale, hc_mult, sinkhorn_iters, eps, norm_eps):
    """One mHC mixing site. Returns (post, comb, collapsed).

    `hidden_streams` is [B, S, hc, E]; `collapsed` is [B, S, E] and feeds the sublayer.
    """
    hc = hc_mult
    flat = unweighted_rms_norm(hidden_streams.reshape(*hidden_streams.shape[:2], -1), norm_eps)
    mixed = linear(flat, fn)
    pre_w, post_w, comb_w = mixed[..., :hc], mixed[..., hc : 2 * hc], mixed[..., 2 * hc :]
    pre_b, post_b, comb_b = base[:hc], base[hc : 2 * hc], base[2 * hc :]
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = sigmoid(pre_w * pre_scale + pre_b) + eps
    post = 2.0 * sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.reshape(hc, hc)
    comb = softmax(comb_logits, axis=-1) + eps
    comb = sinkhorn(comb, sinkhorn_iters, eps)

    collapsed = np.sum(pre[..., None] * hidden_streams, axis=2).astype(FP)
    return post.astype(FP), comb, collapsed


def hyper_head(x, hc_fn, hc_base, hc_scale, eps, norm_eps):
    """Final collapse of the hc streams down to one sequence."""
    flat = unweighted_rms_norm(x.reshape(*x.shape[:2], -1), norm_eps)
    pre = sigmoid(linear(flat, hc_fn) * hc_scale + hc_base) + eps
    return np.sum(pre[..., None] * x, axis=2).astype(FP)


# ---------------------------------------------------------------------------
# grouped output projection (paper 2.3.1)
# ---------------------------------------------------------------------------


def grouped_linear(x: np.ndarray, weight: np.ndarray, n_groups: int) -> np.ndarray:
    """Block-diagonal projection: each head-group is projected independently.

    x is [..., g, in_per_group]; weight is [g * d_g, in_per_group]; out is [..., g, d_g].
    """
    lead = x.shape[:-2]
    in_per_group = x.shape[-1]
    w = weight.reshape(n_groups, -1, in_per_group).transpose(0, 2, 1)  # [g, in, d_g]
    xg = x.reshape(-1, n_groups, in_per_group).transpose(1, 0, 2)  # [g, N, in]
    y = np.matmul(xg.astype(FP), w.astype(FP)).transpose(1, 0, 2)  # [N, g, d_g]
    return y.reshape(*lead, n_groups, -1).astype(FP)


# ---------------------------------------------------------------------------
# compressors (paper 2.3.1 / 2.3.2)
# ---------------------------------------------------------------------------


def _pool_windows(chunk_kv, chunk_gate, kv_norm_w, norm_eps):
    """Softmax-gated pooling of each window down to one entry."""
    w = softmax(chunk_gate, axis=2)
    return rms_norm(np.sum(chunk_kv * w, axis=2), kv_norm_w, norm_eps)


def hca_compress(hidden_states, kv_w, gate_w, position_bias, kv_norm_w, inv_freq, compress_rate, norm_eps,
                 first_window_position=0):
    """Heavily Compressed Attention compressor: non-overlapping windows of m' tokens.

    Returns compressed entries [B, T, D] where T = floor(S / m').
    """
    B, S, _ = hidden_states.shape
    kv = linear(hidden_states, kv_w)
    gate = linear(hidden_states, gate_w)
    usable = (S // compress_rate) * compress_rate
    if usable == 0:
        return np.zeros((B, 0, kv_w.shape[0]), FP)

    n_win = usable // compress_rate
    chunk_kv = kv[:, :usable].reshape(B, n_win, compress_rate, -1)
    chunk_gate = gate[:, :usable].reshape(B, n_win, compress_rate, -1) + position_bias

    compressed = _pool_windows(chunk_kv, chunk_gate, kv_norm_w, norm_eps)
    positions = (np.arange(n_win) * compress_rate + first_window_position)[None, :].repeat(B, 0)
    cos, sin = rope_cos_sin(positions, inv_freq)
    return apply_rope(compressed[:, None], cos, sin)[:, 0]


def _csa_overlap_layout(chunk_kv, chunk_gate, head_dim, prior_kv=None, prior_gate=None):
    """Build the width-2m / stride-m overlapped window tensor used by CSA.

    Each source token emits two series in one tensor: Ca = [..., :D] is its
    contribution to the NEXT window, Cb = [..., D:] to the CURRENT one. Window w
    therefore pools window w-1's Ca together with window w's Cb. Window 0's first
    half comes from the previous forward call (`prior_*`), or stays masked off.
    """
    B, n_win, m, _ = chunk_kv.shape
    new_kv = np.zeros((B, n_win, 2 * m, head_dim), FP)
    new_gate = np.full((B, n_win, 2 * m, head_dim), NEG_INF, FP)
    new_kv[:, :, m:] = chunk_kv[..., head_dim:]
    new_gate[:, :, m:] = chunk_gate[..., head_dim:]
    if n_win > 1:
        new_kv[:, 1:, :m] = chunk_kv[:, :-1, :, :head_dim]
        new_gate[:, 1:, :m] = chunk_gate[:, :-1, :, :head_dim]
    if prior_kv is not None:
        new_kv[:, 0, :m] = prior_kv
        new_gate[:, 0, :m] = prior_gate
    return new_kv, new_gate


def csa_compress(hidden_states, kv_w, gate_w, position_bias, kv_norm_w, inv_freq, compress_rate, head_dim,
                 norm_eps, first_window_position=0, prior_kv=None, prior_gate=None):
    """Compressed Sparse Attention compressor: overlapping windows, width 2m, stride m."""
    B, S, _ = hidden_states.shape
    kv = linear(hidden_states, kv_w)
    gate = linear(hidden_states, gate_w)
    usable = (S // compress_rate) * compress_rate
    if usable == 0:
        return np.zeros((B, 0, head_dim), FP)

    n_win = usable // compress_rate
    chunk_kv = kv[:, :usable].reshape(B, n_win, compress_rate, -1)
    chunk_gate = gate[:, :usable].reshape(B, n_win, compress_rate, -1) + position_bias

    new_kv, new_gate = _csa_overlap_layout(chunk_kv, chunk_gate, head_dim, prior_kv, prior_gate)
    compressed = _pool_windows(new_kv, new_gate, kv_norm_w, norm_eps)
    positions = (np.arange(n_win) * compress_rate + first_window_position)[None, :].repeat(B, 0)
    cos, sin = rope_cos_sin(positions, inv_freq)
    return apply_rope(compressed[:, None], cos, sin)[:, 0]


# ---------------------------------------------------------------------------
# lightning indexer (paper 2.3.1, eqs. 13-17)
# ---------------------------------------------------------------------------


def indexer_score(q, compressed_kv, hidden_states, weights_proj_w, index_head_dim, index_n_heads):
    """sum_h w_{t,h} * ReLU(q_{t,h} . k_s), reduced over heads.

    q is [B, S, H, dI]; compressed_kv is [B, T, dI]; result is [B, S, T].
    The ReLU before the head-weighted sum is what makes this cheap: scores are
    non-negative so the reduction cannot cancel, and it needs no softmax.
    """
    scores = np.matmul(q.astype(np.float64), compressed_kv.astype(np.float64).transpose(0, 2, 1)[:, None])
    scores = np.maximum(scores, 0.0) * (index_head_dim**-0.5)
    weights = linear(hidden_states, weights_proj_w).astype(np.float64) * (index_n_heads**-0.5)
    return np.sum(scores * weights[..., None], axis=2).astype(FP)


def indexer_topk(index_scores, position_ids, compress_rate, index_topk):
    """Top-k compressed entries per query, with non-causal picks marked -1.

    A query at position t may only see compressed entry w if w < (t + 1) // m, since
    entry w summarises source tokens [w*m, (w+1)*m). Early queries have fewer ready
    entries than k, so the surplus picks are returned as a -1 sentinel rather than
    silently pointing into the future.
    """
    B, S, T = index_scores.shape
    if T == 0:
        return np.zeros((B, S, 0), np.int64)
    k = min(index_topk, T)
    threshold = (position_ids + 1) // compress_rate  # [B, S]
    entries = np.arange(T).reshape(1, 1, -1)
    masked = np.where(entries >= threshold[..., None], NEG_INF, index_scores)
    # argpartition then order, mirroring torch.topk(sorted=False) selection semantics.
    idx = np.argsort(-masked, axis=-1, kind="stable")[..., :k]
    return np.where(idx >= threshold[..., None], -1, idx).astype(np.int64)


def csa_block_bias(top_k_indices, compressed_len, seq_len):
    """Per-query additive bias over compressed entries: 0 where selected, -inf elsewhere."""
    B, S, _ = top_k_indices.shape
    valid = top_k_indices >= 0
    safe = np.where(valid, top_k_indices, compressed_len)
    bias = np.full((B, 1, S, compressed_len + 1), NEG_INF, FP)
    np.put_along_axis(bias, safe[:, None], 0.0, axis=-1)
    return bias[..., :compressed_len]


def hca_block_bias(position_ids, compressed_len, compress_rate):
    """HCA attends every ready compressed entry — causality only, no selection."""
    B, S = position_ids.shape
    entries = np.arange(compressed_len).reshape(1, 1, 1, -1)
    threshold = ((position_ids + 1) // compress_rate)[:, None, :, None]
    return np.where(entries >= threshold, NEG_INF, 0.0).astype(FP)


# ---------------------------------------------------------------------------
# attention with learned sinks
# ---------------------------------------------------------------------------


def sink_attention(q, k, v, attention_mask, scaling, sinks):
    """MQA attention with a per-head learned sink logit.

    The sink is an extra column in the softmax that is dropped afterwards, letting a
    head place mass on "nothing" — so rows do not have to sum to 1 over real keys.
    q is [B, H, S, D]; k/v are [B, 1, T, D] and broadcast across heads.
    """
    B, H, S, D = q.shape
    k = np.broadcast_to(k, (B, H, k.shape[2], D))
    v = np.broadcast_to(v, (B, H, v.shape[2], D))
    logits = np.matmul(q.astype(FP), k.astype(FP).transpose(0, 1, 3, 2)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask
    sink_col = np.broadcast_to(sinks.reshape(1, H, 1, 1), (B, H, S, 1))
    combined = np.concatenate([logits, sink_col], axis=-1)
    combined = combined - np.max(combined, axis=-1, keepdims=True)
    probs = softmax(combined, axis=-1)[..., :-1]
    return np.matmul(probs, v.astype(FP)).transpose(0, 2, 1, 3).astype(FP)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def topk_router(hidden_states, weight, e_score_correction_bias, top_k, routed_scaling_factor):
    """Learned router. Selection uses the bias-corrected score; weights do not."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = linear(flat, weight)
    scores = sqrt_softplus(logits)
    idx = np.argsort(-(scores + e_score_correction_bias), axis=-1, kind="stable")[:, :top_k]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / (np.sum(w, axis=-1, keepdims=True) + 1e-20)
    return logits, (w * routed_scaling_factor).astype(FP), idx.astype(np.int64)


def hash_router(hidden_states, input_ids, weight, tid2eid, routed_scaling_factor):
    """Hash-MoE bootstrap: expert choice is a frozen token-id lookup, not learned."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = linear(flat, weight)
    scores = sqrt_softplus(logits)
    idx = tid2eid[input_ids.reshape(-1)].astype(np.int64)
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / (np.sum(w, axis=-1, keepdims=True) + 1e-20)
    return logits, (w * routed_scaling_factor).astype(FP), idx


def clamped_swiglu(gate_up: np.ndarray, limit: float) -> np.ndarray:
    """SwiGLU with asymmetric clamping: gate clamped above, up clamped both ways."""
    gate, up = np.split(gate_up, 2, axis=-1)
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    return (silu(gate) * up).astype(FP)


def experts(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, limit):
    """Routed experts, gathered per expert.

    Weights follow `F.linear`'s [out, in] convention: `gate_up_proj` is
    [E, 2*I, hidden] and `down_proj` is [E, hidden, I]. Each expert is visited once
    over the tokens routed to it — the access pattern that makes expert weights the
    dominant memory traffic at decode time.
    """
    n_tokens, _ = hidden_states.shape
    out = np.zeros((n_tokens, down_proj.shape[-2]), FP)
    for e in np.unique(top_k_index):
        tok, slot = np.where(top_k_index == e)
        if tok.size == 0:
            continue
        cur = clamped_swiglu(linear(hidden_states[tok], gate_up_proj[e]), limit)
        cur = linear(cur, down_proj[e]) * top_k_weights[tok, slot][:, None]
        np.add.at(out, tok, cur.astype(FP))
    return out


def dense_mlp(x, gate_w, up_w, down_w, limit):
    """Shared expert: same clamping as the routed experts."""
    gate = np.minimum(linear(x, gate_w), limit)
    up = np.clip(linear(x, up_w), -limit, limit)
    return linear(silu(gate) * up, down_w)
