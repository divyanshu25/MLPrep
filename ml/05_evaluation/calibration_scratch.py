"""Calibration from scratch (numpy only): reliability table, ECE, temperature scaling.

Synthetic setup: nature assigns each example a true probability p_i of being
positive; labels are drawn y_i ~ Bernoulli(p_i). A well-calibrated model would
report p_i. Our "overconfident network" instead reports sigmoid(k * logit(p_i))
with k = 3: same ranking (monotone transform -> identical AUC), but probabilities
pushed toward 0/1. Temperature scaling divides the logits by T; note that fitting
T should recover roughly T ~ k = 3, undoing the distortion.

Run: uv run python ml/05_evaluation/calibration_scratch.py
"""

import numpy as np

np.random.seed(42)

N = 20000
OVERCONFIDENCE = 3.0  # logit multiplier of the miscalibrated model
N_BINS = 10
EPS = 1e-12


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_data(rng):
    p_true = rng.beta(2.0, 2.0, N)              # true P(y=1|x), spread over (0,1)
    y = (rng.uniform(size=N) < p_true).astype(int)
    z_model = OVERCONFIDENCE * logit(p_true)    # overconfident logits
    p_model = sigmoid(z_model)
    return y, z_model, p_model


def reliability_table(y, p, n_bins=N_BINS, label=""):
    """Bin by predicted probability; compare mean confidence vs empirical accuracy.

    Uses P(y=1) directly (binary), i.e. bins of predicted positive-probability
    vs observed positive rate. Returns ECE.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    print(f"Reliability table {label} (bins of predicted P(y=1)):")
    header = f"{'bin':>12} {'count':>7} {'mean conf':>10} {'frac pos':>9} {'|gap|':>8}"
    print(header)
    print("-" * len(header))
    ece = 0.0
    for b in range(n_bins):
        mask = which == b
        n_b = int(mask.sum())
        if n_b == 0:
            print(f"[{edges[b]:.1f}, {edges[b+1]:.1f}) {'-':>7}")
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        gap = abs(acc - conf)
        ece += (n_b / len(y)) * gap
        print(f"[{edges[b]:.1f}, {edges[b+1]:.1f}) {n_b:>7} {conf:>10.4f} {acc:>9.4f} {gap:>8.4f}")
    print("-" * len(header))
    print(f"ECE {label} = {ece:.4f}")
    return ece


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def fit_temperature(y, z, lo=0.05, hi=20.0, tol=1e-6):
    """Golden-section search for T minimizing validation NLL of sigmoid(z / T).

    NLL as a function of T is unimodal here; golden-section needs no gradients
    and no scipy.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0

    def obj(t):
        return nll(y, sigmoid(z / t))

    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = obj(c), obj(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = obj(d)
    return 0.5 * (a + b)


def auc_rank(y, s):
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)  # no ties expected (continuous scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y, z, p = make_data(rng)

    # Split: fit T on a validation half, report on a test half (post-hoc
    # calibration is always fit on held-out data, never on training data).
    idx = rng.permutation(N)
    val, test = idx[: N // 2], idx[N // 2:]

    print("=" * 62)
    print(f"Overconfident model: logits multiplied by k={OVERCONFIDENCE}")
    print("=" * 62)
    ece_before = reliability_table(y[test], p[test], label="BEFORE")
    print(f"NLL before   = {nll(y[test], p[test]):.4f}")
    print(f"Brier before = {brier(y[test], p[test]):.4f}")
    print()

    T = fit_temperature(y[val], z[val])
    p_cal = sigmoid(z / T)
    print(f"Fitted temperature on validation half: T = {T:.4f} "
          f"(true distortion factor was {OVERCONFIDENCE})")
    print()

    ece_after = reliability_table(y[test], p_cal[test], label="AFTER")
    print(f"NLL after   = {nll(y[test], p_cal[test]):.4f}")
    print(f"Brier after = {brier(y[test], p_cal[test]):.4f}")
    print()

    auc_b = auc_rank(y[test], p[test])
    auc_a = auc_rank(y[test], p_cal[test])
    print(f"AUC before = {auc_b:.6f}, AUC after = {auc_a:.6f} "
          f"(identical: z/T is a monotone transform)")
    print()
    print(f"Summary: ECE {ece_before:.4f} -> {ece_after:.4f} after temperature scaling.")
    assert ece_after < ece_before
    assert abs(auc_b - auc_a) < 1e-12
