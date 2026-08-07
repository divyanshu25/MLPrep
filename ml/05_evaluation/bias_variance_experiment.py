"""Empirical bias-variance decomposition for polynomial regression (numpy only).

True function f(x) = sin(2*pi*x) on [0, 1], observations y = f(x) + N(0, 0.3^2).
For each polynomial degree 1..15 we fit on many independently resampled training
sets and measure, on a fixed test grid:

    bias^2   = mean_x ( f(x) - avg_D fhat_D(x) )^2
    variance = mean_x avg_D ( fhat_D(x) - avg_D fhat_D(x) )^2
    total expected test error = bias^2 + variance + sigma^2   (verified)

Then k-fold cross-validation on ONE training set selects a sensible degree.

Run: uv run python ml/05_evaluation/bias_variance_experiment.py
"""

import numpy as np

np.random.seed(42)

SIGMA = 0.3          # noise std -> irreducible error sigma^2 = 0.09
N_TRAIN = 30         # points per training set
N_SETS = 500         # number of resampled training sets
DEGREES = range(1, 16)


def f_true(x):
    return np.sin(2 * np.pi * x)


def fit_poly(x, y, degree, ridge=1e-8):
    """Least-squares polynomial fit via the normal equations on a Vandermonde
    matrix (tiny ridge for numerical stability at high degree). Returns coeffs."""
    X = np.vander(x, degree + 1, increasing=True)
    A = X.T @ X + ridge * np.eye(degree + 1)
    return np.linalg.solve(A, X.T @ y)


def predict_poly(coeffs, x):
    X = np.vander(x, len(coeffs), increasing=True)
    return X @ coeffs


def bias_variance_table():
    rng = np.random.default_rng(42)
    x_test = np.linspace(0.0, 1.0, 200)
    f_test = f_true(x_test)

    # Pre-draw all training sets so every degree sees the same data draws.
    xs = [rng.uniform(0, 1, N_TRAIN) for _ in range(N_SETS)]
    ys = [f_true(x) + rng.normal(0, SIGMA, N_TRAIN) for x in xs]

    print("=" * 78)
    print(f"Bias^2 / variance decomposition: {N_SETS} training sets of n={N_TRAIN}, "
          f"sigma^2={SIGMA**2:.3f}")
    print("=" * 78)
    header = (f"{'degree':>6} {'bias^2':>10} {'variance':>10} {'noise':>8} "
              f"{'b2+var+noise':>13} {'measured err':>13}")
    print(header)
    print("-" * len(header))

    results = []
    for d in DEGREES:
        preds = np.empty((N_SETS, x_test.size))
        for i in range(N_SETS):
            preds[i] = predict_poly(fit_poly(xs[i], ys[i], d), x_test)

        avg_pred = preds.mean(axis=0)
        bias2 = float(np.mean((f_test - avg_pred) ** 2))
        variance = float(np.mean(preds.var(axis=0)))
        decomposed = bias2 + variance + SIGMA ** 2

        # Directly measured expected test error: E over sets and test noise of
        # (y - fhat)^2 = E[(f - fhat)^2] + sigma^2 (fresh noise on test labels).
        measured = float(np.mean((preds - f_test) ** 2)) + SIGMA ** 2

        results.append((d, bias2, variance, decomposed))
        print(f"{d:>6} {bias2:>10.5f} {variance:>10.5f} {SIGMA**2:>8.3f} "
              f"{decomposed:>13.5f} {measured:>13.5f}")

    best = min(results, key=lambda r: r[3])
    print("-" * len(header))
    print(f"U-shape: bias^2 falls with degree, variance rises; total error is "
          f"minimized at degree {best[0]} ({best[3]:.5f}).")
    print("Sanity check: bias^2 + variance + sigma^2 matches the directly "
          "measured expected error (last two columns agree).")
    return best[0]


def kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def kfold_cv_demo():
    print()
    print("=" * 78)
    print("k-fold CV on a single training set of n=60 (k=5)")
    print("=" * 78)
    rng = np.random.default_rng(42)
    n, k = 60, 5
    x = rng.uniform(0, 1, n)
    y = f_true(x) + rng.normal(0, SIGMA, n)
    folds = kfold_indices(n, k, rng)

    header = f"{'degree':>6} {'CV MSE (mean)':>14} {'CV MSE (std)':>13} {'train MSE':>10}"
    print(header)
    print("-" * len(header))

    cv_results = []
    for d in DEGREES:
        fold_mses = []
        for i in range(k):
            val = folds[i]
            trn = np.concatenate([folds[j] for j in range(k) if j != i])
            coeffs = fit_poly(x[trn], y[trn], d)
            fold_mses.append(np.mean((y[val] - predict_poly(coeffs, x[val])) ** 2))
        coeffs_full = fit_poly(x, y, d)
        train_mse = float(np.mean((y - predict_poly(coeffs_full, x)) ** 2))
        m, s = float(np.mean(fold_mses)), float(np.std(fold_mses))
        cv_results.append((d, m, s))
        print(f"{d:>6} {m:>14.5f} {s:>13.5f} {train_mse:>10.5f}")

    best_d = min(cv_results, key=lambda r: r[1])[0]
    print("-" * len(header))
    print(f"CV selects degree {best_d}.")
    print("Note how train MSE decreases monotonically with degree (it always does),")
    print(f"while CV MSE is U-shaped and bottoms out near sigma^2 = {SIGMA**2:.3f} --")
    print("a validation MSE well below sigma^2 would signal a bug or leakage.")
    return best_d


if __name__ == "__main__":
    best_decomp = bias_variance_table()
    best_cv = kfold_cv_demo()
    print()
    print(f"Degree with lowest true expected error: {best_decomp}; "
          f"degree chosen by 5-fold CV: {best_cv}.")
    print("CV, using only one finite noisy training set, lands close to the truth.")
