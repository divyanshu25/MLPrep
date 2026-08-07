"""
cnn_scratch.py -- convolution forward AND backward in pure numpy, no magic.

Plan:
  1. conv2d forward, written with explicit loops so every index is visible.
  2. conv2d backward (grads w.r.t. input, weights, bias), also with loops.
     Key fact we make concrete: the backward pass of a convolution is
     itself a convolution -- dL/dx is a "full" convolution of dL/dy with
     the 180-degree-rotated kernel; dL/dW is a convolution of x with dL/dy.
     We don't rely on that identity though: we derive each gradient entry
     directly from the forward loop and then VERIFY against finite
     differences, which is the ground truth.
  3. max-pool 2x2 forward/backward (backward routes the gradient to the
     argmax position, zero elsewhere).
  4. End-to-end sanity check: train a 1-conv-layer classifier on synthetic
     8x8 images (horizontal bar vs vertical bar) and watch the loss drop.

Run: uv run python ml/04_deep_learning/cnn_scratch.py
"""

import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. conv2d forward (valid padding, stride 1) -- loops for clarity
# ---------------------------------------------------------------------------
def conv2d_forward(x, W, b):
    """
    x: (N, C_in, H, W_in)          input images
    W: (C_out, C_in, KH, KW)       kernels
    b: (C_out,)                    bias per output channel

    out: (N, C_out, H_out, W_out)  with H_out = H - KH + 1 (no padding, stride 1)

    out[n, co, i, j] = b[co] + sum_{ci, u, v} x[n, ci, i+u, j+v] * W[co, ci, u, v]
    i.e. each output pixel is a dot product between a kernel and the patch
    of the input directly "under" it.
    """
    N, C_in, H, Wi = x.shape
    C_out, C_in2, KH, KW = W.shape
    assert C_in == C_in2
    H_out, W_out = H - KH + 1, Wi - KW + 1
    out = np.zeros((N, C_out, H_out, W_out))
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i:i + KH, j:j + KW]   # (C_in, KH, KW)
                    out[n, co, i, j] = np.sum(patch * W[co]) + b[co]
    return out


# ---------------------------------------------------------------------------
# 2. conv2d backward -- read each gradient off the forward equation
# ---------------------------------------------------------------------------
def conv2d_backward(x, W, dout):
    """
    dout: (N, C_out, H_out, W_out) = dL/dout

    Forward:  out[n,co,i,j] = b[co] + sum_{ci,u,v} x[n,ci,i+u,j+v] * W[co,ci,u,v]

    Chain rule, term by term:
      dL/dW[co,ci,u,v] = sum_{n,i,j} dout[n,co,i,j] * x[n,ci,i+u,j+v]
        (each weight touched every output position -> sum over all of them;
         this is x cross-correlated with dout)
      dL/db[co]        = sum_{n,i,j} dout[n,co,i,j]
      dL/dx[n,ci,p,q]  = sum over every (co,i,j,u,v) with p=i+u, q=j+v of
                         dout[n,co,i,j] * W[co,ci,u,v]
        (an input pixel contributed to every output pixel whose receptive
         field covered it; this is the "full conv with flipped kernel")
    We implement all three by scattering from the forward loop, which is
    literally the chain rule applied to each multiply-accumulate.
    """
    N, C_in, H, Wi = x.shape
    C_out, _, KH, KW = W.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)
    dW = np.zeros_like(W)
    db = dout.sum(axis=(0, 2, 3))                          # (C_out,)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    g = dout[n, co, i, j]                  # scalar
                    # forward used x[n, :, i:i+KH, j:j+KW] * W[co]
                    dW[co] += g * x[n, :, i:i + KH, j:j + KW]
                    dx[n, :, i:i + KH, j:j + KW] += g * W[co]
    return dx, dW, db


# ---------------------------------------------------------------------------
# 3. Max pooling 2x2 stride 2
# ---------------------------------------------------------------------------
def maxpool_forward(x, size=2):
    """
    x: (N, C, H, W) -> out: (N, C, H//size, W//size)
    Also returns the argmax mask so backward knows where the max came from.
    """
    N, C, H, W = x.shape
    H_out, W_out = H // size, W // size
    out = np.zeros((N, C, H_out, W_out))
    argmax = np.zeros((N, C, H_out, W_out, 2), dtype=int)  # (row, col) of max
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    window = x[n, c, i*size:(i+1)*size, j*size:(j+1)*size]
                    out[n, c, i, j] = window.max()
                    r, cidx = np.unravel_index(window.argmax(), window.shape)
                    argmax[n, c, i, j] = (i*size + r, j*size + cidx)
    return out, argmax


def maxpool_backward(dout, argmax, x_shape, size=2):
    """Max is locally an identity for the winning element and ignores the
    rest -> gradient flows only to the argmax position, 0 elsewhere."""
    dx = np.zeros(x_shape)
    N, C, H_out, W_out = dout.shape
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    r, cidx = argmax[n, c, i, j]
                    dx[n, c, r, cidx] += dout[n, c, i, j]
    return dx


# ---------------------------------------------------------------------------
# 4. Finite-difference gradient checking -- the ground truth
# ---------------------------------------------------------------------------
def numeric_grad(f, x, eps=1e-5):
    """Central differences: df/dx_i ~= (f(x + eps e_i) - f(x - eps e_i)) / 2eps.
    O(eps^2) accurate. Slow (one function eval per parameter) but exact-ish;
    this is how you check ANY backward pass you ever write."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(a).max() + np.abs(b).max() + 1e-12)


def check_conv_gradients():
    print("=" * 70)
    print("Gradient check: conv2d backward vs finite differences")
    print("=" * 70)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 5, 5))    # (N=2, C_in=3, 5, 5)
    W = rng.standard_normal((4, 3, 3, 3))    # (C_out=4, C_in=3, 3, 3)
    b = rng.standard_normal(4)
    # Random projection makes the loss a scalar sensitive to every output.
    P = rng.standard_normal((2, 4, 3, 3))    # same shape as out

    def loss():
        return np.sum(conv2d_forward(x, W, b) * P)

    dout = P                                  # dL/dout = P since L = sum(out*P)
    dx, dW, db = conv2d_backward(x, W, dout)

    for name, analytic, var in [("dx", dx, x), ("dW", dW, W), ("db", db, b)]:
        numeric = numeric_grad(loss, var)
        e = rel_err(analytic, numeric)
        print(f"  {name}: max relative error = {e:.2e}")
        assert e < 1e-6, f"{name} gradient check FAILED"
    print("  conv2d backward is correct.\n")


def check_pool_gradients():
    print("=" * 70)
    print("Gradient check: maxpool backward vs finite differences")
    print("=" * 70)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 2, 4, 4))
    P = rng.standard_normal((2, 2, 2, 2))

    def loss():
        out, _ = maxpool_forward(x)
        return np.sum(out * P)

    out, argmax = maxpool_forward(x)
    dx = maxpool_backward(P, argmax, x.shape)
    numeric = numeric_grad(loss, x)
    e = rel_err(dx, numeric)
    print(f"  dx: max relative error = {e:.2e}")
    assert e < 1e-6
    print("  maxpool backward is correct.\n")


# ---------------------------------------------------------------------------
# 5. End-to-end: tiny conv classifier on synthetic bars
# ---------------------------------------------------------------------------
def make_bars(n, rng):
    """8x8 images: class 0 = one horizontal bar, class 1 = one vertical bar,
    at a random position, plus noise. A conv net should nail this: a single
    oriented-edge kernel separates the classes."""
    X = rng.standard_normal((n, 1, 8, 8)) * 0.1
    y = rng.integers(0, 2, size=n)
    for i in range(n):
        pos = rng.integers(1, 7)
        if y[i] == 0:
            X[i, 0, pos, :] += 1.0     # horizontal bar (a bright row)
        else:
            X[i, 0, :, pos] += 1.0     # vertical bar (a bright column)
    return X, y


def train_bars_classifier():
    print("=" * 70)
    print("End-to-end: 1-conv-layer classifier on 8x8 bar images")
    print("=" * 70)
    rng = np.random.default_rng(42)
    Xtr, ytr = make_bars(256, rng)

    # Architecture:
    #   conv 3x3 (1 -> 4 channels)   (N,1,8,8) -> (N,4,6,6)
    #   ReLU
    #   maxpool 2x2                  (N,4,6,6) -> (N,4,3,3)
    #   flatten                      (N, 36)
    #   linear -> 2 logits           (N, 2)
    #   softmax cross-entropy
    C_out, K = 4, 3
    Wc = rng.standard_normal((C_out, 1, K, K)) * 0.3
    bc = np.zeros(C_out)
    Wl = rng.standard_normal((C_out * 3 * 3, 2)) * 0.1
    bl = np.zeros(2)
    lr, batch, steps = 0.1, 32, 60

    losses = []
    for step in range(steps):
        idx = rng.integers(0, len(Xtr), size=batch)
        x, y = Xtr[idx], ytr[idx]
        N = len(x)

        # ---- forward ----
        z1 = conv2d_forward(x, Wc, bc)                # (N,4,6,6)
        a1 = np.maximum(z1, 0)                        # ReLU
        p1, argmax = maxpool_forward(a1)              # (N,4,3,3)
        flat = p1.reshape(N, -1)                      # (N,36)
        logits = flat @ Wl + bl                       # (N,2)
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        acc = (probs.argmax(1) == y).mean()
        losses.append(loss)

        # ---- backward (chain rule, layer by layer, newest first) ----
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0               # softmax+CE gradient
        dlogits /= N                                  # (N,2)
        dWl = flat.T @ dlogits                        # (36,2)
        dbl = dlogits.sum(0)                          # (2,)
        dflat = dlogits @ Wl.T                        # (N,36)
        dp1 = dflat.reshape(p1.shape)                 # (N,4,3,3)
        da1 = maxpool_backward(dp1, argmax, a1.shape) # (N,4,6,6)
        dz1 = da1 * (z1 > 0)                          # ReLU gate
        _, dWc, dbc = conv2d_backward(x, Wc, dz1)

        # ---- SGD ----
        Wc -= lr * dWc; bc -= lr * dbc
        Wl -= lr * dWl; bl -= lr * dbl

        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {loss:.4f}  batch acc {acc:.2f}")

    assert losses[-1] < 0.5 * losses[0], "loss did not decrease enough"
    print(f"  loss went {losses[0]:.3f} -> {losses[-1]:.3f}: learning works.\n")


if __name__ == "__main__":
    # Tiny hand-checkable example first: 5x5 input, 3x3 kernel of ones.
    print("=" * 70)
    print("Warm-up: 3x3 all-ones kernel over a 5x5 ramp (sums each patch)")
    print("=" * 70)
    x = np.arange(25, dtype=float).reshape(1, 1, 5, 5)   # 0..24
    W = np.ones((1, 1, 3, 3))
    out = conv2d_forward(x, W, np.zeros(1))
    print("input:\n", x[0, 0])
    print("output (each entry = sum of the 3x3 patch under it):\n", out[0, 0])
    # top-left patch is [[0,1,2],[5,6,7],[10,11,12]] -> sums to 54
    assert out[0, 0, 0, 0] == 54.0
    print()

    check_conv_gradients()
    check_pool_gradients()
    train_bars_classifier()
    print("All checks passed.")
