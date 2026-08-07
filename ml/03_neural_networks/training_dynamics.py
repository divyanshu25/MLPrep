"""training_dynamics.py — numeric experiments behind the folklore. No plots;
every claim in notes.md that "X goes wrong" is reproduced here as printed
numbers you can watch go wrong.

  EXPT 1: activation variance across depth — bad init vs Xavier vs Kaiming
  EXPT 2: per-layer gradient magnitude — sigmoid vs ReLU (vanishing gradients)
  EXPT 3: learning rate too low / right / too high (slow / converge / diverge)

Run:  uv run python ml/03_neural_networks/training_dynamics.py
"""

import numpy as np

np.random.seed(42)


# ============================================================================
# EXPT 1: activation variance layer by layer
# ============================================================================
# Theory (notes.md section e): for z = W x with iid mean-0 weights,
#     Var(z) = n_in * Var(w) * Var(x)
# so each layer MULTIPLIES the variance by g = n_in * Var(w). After d layers
# the scale factor is g^d — geometric in depth. Only g ≈ 1 survives.
# Kaiming adds the factor 2 because ReLU zeroes half of a symmetric input,
# halving the second moment.
def expt1_activation_variance(depth=30, width=256, n_samples=512):
    print("=" * 76)
    print("EXPT 1: std of activations across depth (width=256, 30 tanh/relu layers)")
    print("=" * 76)

    X = np.random.randn(n_samples, width)  # unit-variance input

    schemes = [
        # (label, weight std, nonlinearity)
        ("tanh, std=0.01 (too small)", lambda n: 0.01, np.tanh),
        ("tanh, std=1.0  (too big)  ", lambda n: 1.0, np.tanh),
        ("tanh, Xavier 1/sqrt(n)    ", lambda n: np.sqrt(1.0 / n), np.tanh),
        ("relu, Xavier 1/sqrt(n)    ", lambda n: np.sqrt(1.0 / n), lambda z: np.maximum(0, z)),
        ("relu, Kaiming sqrt(2/n)   ", lambda n: np.sqrt(2.0 / n), lambda z: np.maximum(0, z)),
    ]

    report_layers = [1, 2, 5, 10, 20, 30]
    print(f"{'init scheme':<28} | " + " | ".join(f"L{d:<2} std" for d in report_layers))
    print("-" * 76)
    for label, std_fn, act in schemes:
        np.random.seed(42)  # same weights draw pattern for comparability
        h = X
        stds = {}
        for layer in range(1, depth + 1):
            W = np.random.randn(width, width) * std_fn(width)
            h = act(h @ W)
            if layer in report_layers:
                stds[layer] = h.std()
        print(f"{label:<28} | " + " | ".join(f"{stds[d]:>7.4f}" for d in report_layers))

    print("""
Reading it:
  * std=0.01: activations collapse to ~0 within a few layers (g << 1). The
    deep layers compute on numerical dust; gradients (which carry activation
    factors) die too.
  * std=1.0 with tanh: pre-activations explode, so tanh SATURATES — std pins
    near 1.0 because outputs sit at +/-1. Saturated tanh has derivative ~0:
    exploding forward becomes vanishing backward.
  * Xavier keeps tanh healthy through all 30 layers (g ~ 1).
  * Xavier with ReLU decays ~(1/sqrt(2))^d — ReLU halves the 2nd moment each
    layer and Xavier doesn't compensate. 30 layers -> ~180x shrink.
  * Kaiming's factor 2 fixes exactly that: ReLU std holds steady.
""")


# ============================================================================
# EXPT 2: vanishing gradients — sigmoid vs ReLU, measured, not asserted
# ============================================================================
# We build a deep net, run one forward/backward with the SAME upstream
# gradient, and print ||dL/dh_k|| per layer. Backprop multiplies one Jacobian
# diag(act'(z_k)) @ W_k per layer; sigmoid contributes a factor <= 0.25
# per layer (max of s(1-s)), ReLU contributes exactly 1 on active units.
def expt2_vanishing_gradients(depth=20, width=128, n_samples=256):
    print("=" * 76)
    print("EXPT 2: gradient norm per layer, 20-layer net, sigmoid vs relu")
    print("=" * 76)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    X = np.random.randn(n_samples, width)

    results = {}
    for name in ["sigmoid", "relu"]:
        np.random.seed(42)  # identical weights for both runs
        # Xavier init for both, so the ONLY difference is the activation.
        Ws = [np.random.randn(width, width) * np.sqrt(1.0 / width) for _ in range(depth)]

        # forward, caching pre-activations
        h, caches = X, []
        for W in Ws:
            z = h @ W
            if name == "sigmoid":
                a = sigmoid(z)
                local = a * (1 - a)          # sigma' = s(1-s), max 0.25
            else:
                a = np.maximum(0, z)
                local = (z > 0).astype(float)  # relu' = 1[z>0]
            caches.append((W, local))
            h = a

        # backward from a fixed unit upstream gradient at the top
        g = np.ones((n_samples, width)) / np.sqrt(width)
        norms = []
        for W, local in reversed(caches):
            g = (g * local) @ W.T            # chain rule: diag(act') then W^T
            norms.append(np.abs(g).mean())
        results[name] = norms[::-1]          # index 0 = earliest layer

    print(f"{'layer':>6} | {'sigmoid  mean|grad|':>20} | {'relu  mean|grad|':>18} | ratio relu/sigmoid")
    print("-" * 76)
    for k in [0, 2, 4, 8, 12, 16, 19]:
        s, r = results["sigmoid"][k], results["relu"][k]
        print(f"{k:>6} | {s:>20.3e} | {r:>18.3e} | {r / s:>12.1e}")

    print("""
Reading it (layer 0 = closest to the input, i.e. gradient traveled 20 layers):
  * sigmoid: gradient decays geometrically — each layer multiplies by
    E[s(1-s)] ~ 0.2 or less, so by layer 0 the signal is ~1e-13 of the top.
    The early layers of this network would effectively never train.
  * relu (same weights!): active units pass gradient with factor exactly 1,
    so the decay is orders of magnitude milder. This difference — not any
    exotic trick — is the main reason deep nets became trainable.
""")


# ============================================================================
# EXPT 3: learning rate too low / right / too high
# ============================================================================
# Same tiny 2-layer softmax classifier on 2 gaussian blobs, same init, only
# the learning rate changes: watch slow-crawl, healthy convergence, and
# divergence to nan as three printed loss columns.
def expt3_learning_rates(n_steps=200):
    print("=" * 76)
    print("EXPT 3: effect of learning rate (same model/data/init, 3 different lr)")
    print("=" * 76)

    # Data: 4 gaussian blobs in an XOR arrangement (2 classes). Deliberately
    # NOT linearly separable: on separable data a wildly diverging model can
    # accidentally land on a perfect separator with huge margin and report
    # loss ~ 0, muddying the demo. XOR blobs force real learning.
    np.random.seed(42)
    n = 50
    centers = [(2, 2), (-2, -2), (2, -2), (-2, 2)]
    X = np.vstack([np.random.randn(n, 2) + c for c in centers])
    y = np.array([0] * n + [0] * n + [1] * n + [1] * n)

    def stable_ce(Z, y):
        m = Z.max(axis=1, keepdims=True)
        logp = Z - (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True)))
        N = len(y)
        loss = -logp[np.arange(N), y].mean()
        dZ = np.exp(logp)
        dZ[np.arange(N), y] -= 1
        return loss, dZ / N

    def run(lr):
        np.random.seed(42)
        W1 = np.random.randn(2, 32) * np.sqrt(2.0 / 2)
        b1 = np.zeros(32)
        W2 = np.random.randn(32, 2) * np.sqrt(1.0 / 32)
        b2 = np.zeros(2)
        hist = []
        for t in range(1, n_steps + 1):
            Z1 = X @ W1 + b1
            A1 = np.maximum(0, Z1)
            Z2 = A1 @ W2 + b2
            loss, dZ2 = stable_ce(Z2, y)
            dW2, db2 = A1.T @ dZ2, dZ2.sum(0)
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * (Z1 > 0)
            dW1, db1 = X.T @ dZ1, dZ1.sum(0)
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
            if t in (1, 5, 10, 25, 50, 100, 150, 200):
                hist.append((t, loss))
        acc = ((np.maximum(0, X @ W1 + b1) @ W2 + b2).argmax(1) == y).mean()
        return hist, acc

    lrs = [("too low (1e-4)", 1e-4), ("healthy (0.1)", 0.1), ("too high (75)", 75.0)]
    with np.errstate(all="ignore"):  # divergence to inf/nan is the point
        runs = [(label, *run(lr)) for label, lr in lrs]

    def fmt(x):
        if np.isnan(x):
            return f"{'nan':>16}"
        if x > 1e4:
            return f"{x:>16.2e}"
        return f"{x:>16.4f}"

    print(f"{'step':>6} | " + " | ".join(f"{label:>16}" for label, _, _ in runs))
    print("-" * 66)
    for i in range(len(runs[0][1])):
        t = runs[0][1][i][0]
        print(f"{t:>6} | " + " | ".join(fmt(h[i][1]) for _, h, _ in runs))
    print()
    for label, _, acc in runs:
        print(f"  final accuracy [{label:>16}]: {acc * 100:5.1f}%")
    assert np.isnan(runs[2][1][-1][1]), "expected the high-lr run to diverge to nan"

    print("""
Reading it:
  * too low:  loss creeps down; after 200 steps it has barely moved and
    accuracy is mediocre. Nothing is broken — you're just wasting compute
    (and might wrongly conclude the model 'doesn't work').
  * healthy:  smooth, fast convergence to low loss and ~96% accuracy.
  * too high: each step overshoots the minimum by more than it approached it;
    the loss grows geometrically and finally overflows to nan, and accuracy
    is chance (50%). This is the classic 'loss went to nan' failure — the
    first fix to try is always lr / 10.
""")


if __name__ == "__main__":
    expt1_activation_variance()
    expt2_vanishing_gradients()
    expt3_learning_rates()
