"""
attention_scratch.py -- scaled dot-product attention from scratch in numpy.

The whole file builds one idea:

    attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

i.e. "for each query, take a weighted average of the value vectors,
where the weights come from how similar the query is to each key."

We implement:
  1. scaled dot-product attention (single head)
  2. causal masking (a token may only attend to itself + the past)
  3. multi-head attention (project into h subspaces, attend in each, concat)
then we:
  4. verify all shapes,
  5. verify attention weights sum to 1 (each row of the softmax is a distribution),
  6. show WHY we divide by sqrt(d_k): at d_k=64, unscaled logits have variance
     ~64, softmax saturates into a near one-hot -> entropy collapses,
  7. check our numpy multi-head attention against
     torch.nn.functional.scaled_dot_product_attention numerically.

Run: uv run python ml/04_deep_learning/attention_scratch.py
"""

import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Softmax (numerically stable) -- the only nonlinearity attention needs
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    """Stable softmax: subtracting the max doesn't change the result
    (softmax is shift-invariant) but keeps exp() from overflowing."""
    x = x - x.max(axis=axis, keepdims=True)  # shape preserved
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Scaled dot-product attention (single head)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, causal=False, scale=None):
    """
    Q: (..., T_q, d_k)   queries: "what am I looking for?"
    K: (..., T_k, d_k)   keys:    "what do I contain?" (matched against Q)
    V: (..., T_k, d_v)   values:  "what do I hand over if selected?"

    Returns:
      out: (..., T_q, d_v)   each row = weighted average of rows of V
      att: (..., T_q, T_k)   the attention weights (each row sums to 1)

    '...' is any batch shape, e.g. (B,) or (B, n_heads).
    """
    d_k = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)

    # scores[i, j] = <q_i, k_j> : similarity of query i to key j
    # (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
    scores = (Q @ np.swapaxes(K, -1, -2)) * scale

    if causal:
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        # mask[i, j] = True where j > i  (the future). -inf logits -> weight 0.
        mask = np.triu(np.ones((T_q, T_k), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    att = softmax(scores, axis=-1)      # (..., T_q, T_k), rows sum to 1
    out = att @ V                       # (..., T_q, d_v)
    return out, att


# ---------------------------------------------------------------------------
# 3. Multi-head attention
# ---------------------------------------------------------------------------
# One head can only measure similarity in ONE learned subspace. h heads
# project x into h different d_head-dimensional subspaces, attend
# independently in each ("subspace views"), then concat + mix with W_o.
class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        s = 1.0 / np.sqrt(d_model)  # small init, scale doesn't matter here
        # In real implementations these are one fused (d_model, 3*d_model) matmul.
        self.Wq = rng.standard_normal((d_model, d_model)) * s  # (d, d)
        self.Wk = rng.standard_normal((d_model, d_model)) * s  # (d, d)
        self.Wv = rng.standard_normal((d_model, d_model)) * s  # (d, d)
        self.Wo = rng.standard_normal((d_model, d_model)) * s  # (d, d)

    def split_heads(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        Q = self.split_heads(x @ self.Wq)  # (B, h, T, d_head)
        K = self.split_heads(x @ self.Wk)  # (B, h, T, d_head)
        V = self.split_heads(x @ self.Wv)  # (B, h, T, d_head)

        out, att = scaled_dot_product_attention(Q, K, V, causal=causal)
        # out: (B, h, T, d_head) -> concat heads -> (B, T, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        return out @ self.Wo, att          # (B, T, d_model), (B, h, T, T)


# ---------------------------------------------------------------------------
# Demos / checks
# ---------------------------------------------------------------------------
def entropy(p, axis=-1):
    """Shannon entropy of each softmax row, in nats. Max = ln(T) (uniform),
    min = 0 (one-hot)."""
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("1) Single-head attention: shapes and row sums")
    print("=" * 70)
    B, T, d_k, d_v = 2, 5, 8, 8
    Q = rng.standard_normal((B, T, d_k))
    K = rng.standard_normal((B, T, d_k))
    V = rng.standard_normal((B, T, d_v))
    out, att = scaled_dot_product_attention(Q, K, V)
    print(f"Q {Q.shape}  K {K.shape}  V {V.shape}")
    print(f"out {out.shape}   (expected ({B}, {T}, {d_v}))")
    print(f"att {att.shape}   (expected ({B}, {T}, {T}))")
    assert out.shape == (B, T, d_v) and att.shape == (B, T, T)
    row_sums = att.sum(axis=-1)
    print(f"attention rows sum to 1?  max |row_sum - 1| = "
          f"{np.abs(row_sums - 1).max():.2e}")
    assert np.allclose(row_sums, 1.0)

    print()
    print("=" * 70)
    print("2) Causal masking: no weight on the future")
    print("=" * 70)
    _, att_c = scaled_dot_product_attention(Q, K, V, causal=True)
    print("attention matrix for batch 0 (rows=queries, cols=keys):")
    with np.printoptions(precision=3, suppress=True):
        print(att_c[0])
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    print(f"max weight above the diagonal: {att_c[:, upper].max():.2e} "
          f"(should be exactly 0)")
    assert att_c[:, upper].max() == 0.0
    # Row 0 can only attend to itself -> weight 1.0 on position 0.
    assert np.allclose(att_c[:, 0, 0], 1.0)

    print()
    print("=" * 70)
    print("3) Why divide by sqrt(d_k)?  Softmax entropy at d_k = 64")
    print("=" * 70)
    # If q, k have i.i.d. entries with mean 0, var 1, then Var(<q,k>) = d_k.
    # At d_k=64 the logits have std 8: softmax saturates -> near one-hot rows.
    d = 64
    Tn = 16
    q = rng.standard_normal((Tn, d))
    k = rng.standard_normal((Tn, d))
    logits = q @ k.T                          # (T, T)
    print(f"empirical Var(<q,k>) = {logits.var():7.2f}   (theory: d_k = {d})")
    print(f"after /sqrt(d_k):      {(logits / np.sqrt(d)).var():7.2f}   (theory: 1)")
    H_unscaled = entropy(softmax(logits)).mean()
    H_scaled = entropy(softmax(logits / np.sqrt(d))).mean()
    print(f"mean softmax entropy WITHOUT scaling: {H_unscaled:.3f} nats")
    print(f"mean softmax entropy WITH    scaling: {H_scaled:.3f} nats")
    print(f"max possible entropy ln({Tn}) =        {np.log(Tn):.3f} nats")
    print("-> unscaled rows are near one-hot (entropy ~0): gradients through\n"
          "   softmax vanish; scaling keeps the distribution soft and trainable.")

    print()
    print("=" * 70)
    print("4) Multi-head attention: shapes + numerical check vs PyTorch")
    print("=" * 70)
    B, T, d_model, n_heads = 2, 7, 16, 4
    x = rng.standard_normal((B, T, d_model))
    mha = MultiHeadAttention(d_model, n_heads, rng)
    y, att = mha(x, causal=True)
    print(f"x {x.shape} -> y {y.shape}   att {att.shape} "
          f"(B, n_heads, T, T)")
    assert y.shape == (B, T, d_model)
    assert att.shape == (B, n_heads, T, T)

    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    # Same projections, same input, run the core attention through torch.
    Qh = mha.split_heads(x @ mha.Wq)  # (B, h, T, d_head)
    Kh = mha.split_heads(x @ mha.Wk)
    Vh = mha.split_heads(x @ mha.Wv)
    t_out = F.scaled_dot_product_attention(
        torch.from_numpy(Qh), torch.from_numpy(Kh), torch.from_numpy(Vh),
        is_causal=True,
    ).numpy()                                             # (B, h, T, d_head)
    y_torch = t_out.transpose(0, 2, 1, 3).reshape(B, T, d_model) @ mha.Wo
    max_err = np.abs(y - y_torch).max()
    print(f"max |numpy - torch| = {max_err:.2e}")
    assert max_err < 1e-10, "numpy and torch attention disagree!"
    print("numpy multi-head attention matches torch.nn.functional."
          "scaled_dot_product_attention. All checks passed.")


if __name__ == "__main__":
    main()
