"""mlp_numpy.py - a vectorized 2-layer MLP in pure numpy, everything by hand.

Architecture (classic 2-layer / 1-hidden-layer MLP):

    X (N,D) --W1,b1--> Z1 (N,H) --ReLU--> A1 (N,H) --W2,b2--> Z2 (N,C) --softmax+CE--> loss

Contents:
  1. Numerically-stable softmax cross-entropy (log-sum-exp trick).
  2. Manual backward pass with every matrix gradient DERIVED in comments.
  3. Gradient check against centered finite differences.
  4. Training on a synthetic 3-class spiral (linearly inseparable on purpose),
     comparing SGD vs SGD+momentum vs Adam on the identical problem/init.

Run:  uv run python ml/03_neural_networks/mlp_numpy.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# Data: 3-class spiral (Karpathy's CS231n classic). A linear model gets ~50%
# at best here; the hidden layer is what untangles the arms.
# ----------------------------------------------------------------------------
def make_spiral(points_per_class=100, n_classes=3, noise=0.2):
    N, C = points_per_class, n_classes
    X = np.zeros((N * C, 2))
    y = np.zeros(N * C, dtype=int)
    for c in range(C):
        ix = range(N * c, N * (c + 1))
        r = np.linspace(0.0, 1.0, N)                                # radius
        t = np.linspace(c * 4, (c + 1) * 4, N) + np.random.randn(N) * noise  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


# ----------------------------------------------------------------------------
# Numerically stable log-softmax / cross-entropy
# ----------------------------------------------------------------------------
def stable_logsumexp(Z):
    """log(sum_j exp(z_j)) computed without overflow.

    exp(z) overflows float64 at z ~ 709 (and float32 at ~88); logits get there
    easily mid-training. Softmax is shift-invariant - subtracting any constant
    m from all logits leaves p unchanged:
        exp(z_i - m) / sum_j exp(z_j - m) = p_i        (e^{-m} cancels)
    Choosing m = max(z) makes every exponent <= 0 (no overflow) and at least
    one term equal to 1 (denominator >= 1, no log(0)).
    """
    m = Z.max(axis=1, keepdims=True)
    return m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True))


def softmax_ce(Z, y):
    """Return (mean loss, dLoss/dZ) for logits Z (N,C) and int labels y (N,).

    Loss per example:  L_i = -log p_{i, y_i} = -z_{i,y_i} + logsumexp(z_i)
    Computed as log-probs directly (never log(softmax): log of an underflowed
    0.0 is -inf).

    Gradient derivation (per example, from notes.md section c):
        dL/dz_k = -1[k == y] + exp(z_k)/sum_j exp(z_j) = p_k - y_onehot_k
    We average the loss over N examples, so each example's gradient also gets
    the 1/N factor.
    """
    logp = Z - stable_logsumexp(Z)                # (N,C) log-probabilities
    N = Z.shape[0]
    loss = -logp[np.arange(N), y].mean()
    P = np.exp(logp)                              # softmax probabilities
    dZ = P.copy()
    dZ[np.arange(N), y] -= 1.0                    # p - y  (y as one-hot)
    dZ /= N                                       # d(mean)/d(each term) = 1/N
    return loss, dZ


# ----------------------------------------------------------------------------
# The model: forward + manual backward
# ----------------------------------------------------------------------------
def init_params(D, H, C):
    """Kaiming init for the ReLU hidden layer (std = sqrt(2/fan_in)), Xavier-ish
    for the linear output. See notes.md section (e) for the derivation."""
    return {
        "W1": np.random.randn(D, H) * np.sqrt(2.0 / D),
        "b1": np.zeros(H),
        "W2": np.random.randn(H, C) * np.sqrt(1.0 / H),
        "b2": np.zeros(C),
    }


def forward(params, X):
    """Forward pass; returns logits and a cache of intermediates.

    The cache is not an optimization detail - backward's local gradients are
    functions of forward values (dZ1 needs the ReLU mask, dW2 needs A1, dW1
    needs X), so they must be kept alive. This is why training memory scales
    with activations."""
    Z1 = X @ params["W1"] + params["b1"]     # (N,H)  pre-activations
    A1 = np.maximum(0.0, Z1)                 # (N,H)  ReLU
    Z2 = A1 @ params["W2"] + params["b2"]    # (N,C)  logits
    return Z2, (X, Z1, A1)


def backward(params, cache, dZ2):
    """Manual backprop. All shapes: X (N,D), W1 (D,H), A1 (N,H), W2 (H,C), dZ2 (N,C).

    The master rule for a linear layer Z = A @ W + b with upstream G = dL/dZ:

      dL/dW = A^T @ G
        Derivation (element-wise): Z_nc = sum_h A_nh W_hc + b_c, so
        dZ_nc/dW_hc = A_nh and W_hc touches Z_nc for every n. Chain rule sums
        over everything downstream that W_hc influences:
            dL/dW_hc = sum_n G_nc * A_nh = (A^T G)_hc.
        (The sum over n is the "gradients add over fan-out" rule: one weight is
        REUSED by every example in the batch, so contributions accumulate.)

      dL/dA = G @ W^T
        Derivation: A_nh touches Z_nc for every class c, with dZ_nc/dA_nh = W_hc:
            dL/dA_nh = sum_c G_nc W_hc = (G W^T)_nh.

      dL/db = G.sum(axis=0)
        Derivation: dZ_nc/db_c = 1 and b_c touches every example n, so the
        gradient is the column sum of G.

      Sanity check that generalizes: the shapes only fit one way
        (D,H) = (D,N)@(N,H), (N,H) = (N,C)@(C,H) - memorize the rule, recover
        the transposes from shapes under pressure.
    """
    X, Z1, A1 = cache
    grads = {}

    # ---- output linear layer: Z2 = A1 @ W2 + b2 ----
    grads["W2"] = A1.T @ dZ2                 # (H,C)
    grads["b2"] = dZ2.sum(axis=0)            # (C,)
    dA1 = dZ2 @ params["W2"].T               # (N,H)

    # ---- ReLU: A1 = max(0, Z1) ----
    # Elementwise gate: derivative is 1 where Z1 > 0, else 0. Gradient flows
    # only through units that were active on the forward pass.
    dZ1 = dA1 * (Z1 > 0)                     # (N,H)

    # ---- hidden linear layer: Z1 = X @ W1 + b1 (same master rule) ----
    grads["W1"] = X.T @ dZ1                  # (D,H)
    grads["b1"] = dZ1.sum(axis=0)            # (H,)
    # dX = dZ1 @ W1.T would be next if there were layers below.

    return grads


def loss_and_grads(params, X, y):
    Z2, cache = forward(params, X)
    loss, dZ2 = softmax_ce(Z2, y)
    return loss, backward(params, cache, dZ2)


# ----------------------------------------------------------------------------
# Gradient check: analytic backward vs centered finite differences
# ----------------------------------------------------------------------------
def gradient_check():
    print("=" * 72)
    print("PART 1: gradient check (analytic backward vs finite differences)")
    print("=" * 72)
    # Tiny everything so the check is fast; float64 makes 1e-5 steps accurate.
    X = np.random.randn(7, 2)
    y = np.random.randint(0, 3, size=7)
    params = init_params(D=2, H=5, C=3)

    _, grads = loss_and_grads(params, X, y)

    h = 1e-5
    for name, W in params.items():
        # Check a handful of random entries per tensor (checking all is fine
        # here, but sampling is the realistic habit for big models).
        num = np.zeros_like(W)
        it = np.ndindex(W.shape)
        for idx in it:
            old = W[idx]
            W[idx] = old + h
            lp, _ = loss_and_grads(params, X, y)
            W[idx] = old - h
            lm, _ = loss_and_grads(params, X, y)
            W[idx] = old
            # centered difference: O(h^2) error (odd Taylor terms cancel)
            num[idx] = (lp - lm) / (2 * h)
        rel = np.abs(grads[name] - num) / (np.abs(grads[name]) + np.abs(num) + 1e-12)
        print(f"  {name}: max relative error = {rel.max():.3e}")
        assert rel.max() < 1e-6, f"gradient check FAILED for {name}"
    print("  -> PASS: the hand-derived matrix gradients are correct\n")


# ----------------------------------------------------------------------------
# Optimizers - same update interface so the comparison is apples-to-apples
# ----------------------------------------------------------------------------
def make_sgd(lr):
    def update(params, grads, t):
        for k in params:
            params[k] -= lr * grads[k]
    return update


def make_momentum(lr, beta=0.9):
    v = {}
    def update(params, grads, t):
        for k in params:
            # v accumulates an exponentially-decayed sum of past gradients;
            # for a persistent gradient g it converges to g/(1-beta) - up to
            # 10x SGD's step along consistent directions, while oscillating
            # components cancel. (Derivation in notes.md section f.)
            v[k] = beta * v.get(k, 0.0) + grads[k]
            params[k] -= lr * v[k]
    return update


def make_adam(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m, v = {}, {}
    def update(params, grads, t):  # t is 1-based
        for k in params:
            m[k] = beta1 * m.get(k, 0.0) + (1 - beta1) * grads[k]
            v[k] = beta2 * v.get(k, 0.0) + (1 - beta2) * grads[k] ** 2
            # Bias correction: EMAs start at 0, so E[m_t] = g * (1 - beta1^t)
            # (geometric series - derived in notes.md). Dividing by 1-beta^t
            # unbiases them; without it, step 1's update is ~sqrt(1/(1-beta2))
            # ≈ 30x too large. The correction fades as t grows.
            mhat = m[k] / (1 - beta1 ** t)
            vhat = v[k] / (1 - beta2 ** t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
    return update


# ----------------------------------------------------------------------------
# Training loop + optimizer comparison
# ----------------------------------------------------------------------------
def train(X, y, optimizer_factory, n_steps=300, H=64, log_every=30):
    np.random.seed(42)  # identical init for every optimizer: fair comparison
    params = init_params(D=X.shape[1], H=H, C=y.max() + 1)
    update = optimizer_factory()
    history = []
    for t in range(1, n_steps + 1):
        loss, grads = loss_and_grads(params, X, y)  # full batch: tiny dataset
        update(params, grads, t)
        if t % log_every == 0 or t == 1:
            history.append((t, loss))
    Z2, _ = forward(params, X)
    acc = (Z2.argmax(axis=1) == y).mean()
    return history, acc


def compare_optimizers():
    print("=" * 72)
    print("PART 2: train on 3-class spiral - SGD vs momentum vs Adam")
    print("=" * 72)
    X, y = make_spiral(points_per_class=100, n_classes=3)
    print(f"dataset: {X.shape[0]} points, 2 features, 3 classes (spiral)")
    print("(a linear model tops out near ~50% here - the hidden layer does the work)\n")

    runs = {
        "SGD(lr=1.0)":            lambda: make_sgd(lr=1.0),
        "Momentum(lr=0.5,b=0.9)": lambda: make_momentum(lr=0.5, beta=0.9),
        "Adam(lr=0.01)":          lambda: make_adam(lr=0.01),
    }

    results = {}
    for name, factory in runs.items():
        history, acc = train(X, y, factory, n_steps=300)
        results[name] = (history, acc)

    # Loss-curve table: one column per optimizer, aligned by step
    steps = [t for t, _ in results["SGD(lr=1.0)"][0]]
    header = f"{'step':>6} | " + " | ".join(f"{n:>22}" for n in runs)
    print(header)
    print("-" * len(header))
    for i, t in enumerate(steps):
        row = f"{t:>6} | " + " | ".join(f"{results[n][0][i][1]:>22.4f}" for n in runs)
        print(row)
    print()
    for name in runs:
        print(f"  final train accuracy [{name:>22}]: {results[name][1] * 100:5.1f}%")

    assert results["Adam(lr=0.01)"][1] > 0.90, "Adam failed to fit the spiral"
    print("\nReading the table: momentum beats plain SGD at the same step count by")
    print("accelerating along consistent gradient directions; Adam additionally")
    print("rescales per-parameter, making one global lr work for all tensors.")


if __name__ == "__main__":
    gradient_check()
    compare_optimizers()
