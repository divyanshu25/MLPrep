"""k-means and PCA from scratch (numpy only).

What this script does:
  1. k-means via Lloyd's algorithm with an inertia trace that demonstrates
     the objective decreases monotonically every iteration (this is the
     alternating-minimization proof made visible).
  2. PCA via eigendecomposition of the covariance matrix, verified against
     sklearn (components match up to sign), plus reconstruction error as a
     function of the number of retained components.

Run:  uv run python ml/02_classical_ml/kmeans_pca_scratch.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# k-means
# ----------------------------------------------------------------------------
def kmeans(X, k, n_iters=100, seed=0):
    """Lloyd's algorithm. Returns (centers, labels, inertia_trace).

    inertia = sum_i ||x_i - c_{z_i}||^2, recorded after every full
    (assign, update) step so we can check it never increases.
    """
    rng = np.random.default_rng(seed)
    # init: pick k distinct data points as starting centers
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    trace = []
    labels = None
    for _ in range(n_iters):
        # Assignment step: nearest center for every point.
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n, k)
        new_labels = d2.argmin(axis=1)
        # Update step: each center = mean of its assigned points.
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
        inertia = float(((X - centers[new_labels]) ** 2).sum())
        trace.append(inertia)
        if labels is not None and np.array_equal(labels, new_labels):
            break  # assignments stable -> converged
        labels = new_labels
    return centers, labels, trace


def run_kmeans():
    print("=" * 70)
    print("K-MEANS (Lloyd's algorithm)")
    print("=" * 70)

    # Three Gaussian blobs in 2D with some overlap (so Lloyd's takes a few
    # iterations to settle instead of converging instantly).
    n_per = 300
    blobs = [
        np.random.randn(n_per, 2) * 1.1 + np.array([0.0, 0.0]),
        np.random.randn(n_per, 2) * 1.1 + np.array([4.0, 1.0]),
        np.random.randn(n_per, 2) * 1.1 + np.array([2.0, 3.5]),
    ]
    X = np.vstack(blobs)
    true_labels = np.repeat(np.arange(3), n_per)

    centers, labels, trace = kmeans(X, k=3, seed=7)

    print(f"\nconverged in {len(trace)} iterations; inertia trace:")
    for i, v in enumerate(trace):
        print(f"  iter {i:2d}: inertia = {v:12.4f}")

    diffs = np.diff(trace)
    assert np.all(diffs <= 1e-9), "inertia increased -- Lloyd's proof violated!"
    print("PASS: inertia is monotonically non-increasing (as proved by "
          "alternating minimization).")

    # Cluster quality: purity against the generating labels.
    purity = 0
    for j in range(3):
        members = true_labels[labels == j]
        if len(members):
            purity += np.bincount(members).max()
    purity /= len(X)
    print(f"cluster purity vs. ground truth = {purity:.4f}")
    print(f"final centers:\n{np.round(centers, 3)}")

    # sklearn check: same-ish inertia (allow small slack; init differs).
    from sklearn.cluster import KMeans
    sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    print(f"\nsklearn KMeans inertia = {sk.inertia_:.4f} (ours: {trace[-1]:.4f})")
    assert trace[-1] <= sk.inertia_ * 1.05, "our inertia much worse than sklearn"
    print("PASS: within 5% of sklearn's inertia.")


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------
def pca_fit(X):
    """PCA via eigendecomposition of the covariance matrix.

    Returns (mean, components, explained_variance) where components rows are
    unit eigenvectors sorted by decreasing eigenvalue.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (len(X) - 1)           # (d, d) sample covariance
    eigvals, eigvecs = np.linalg.eigh(cov)      # ascending order for symmetric
    order = np.argsort(eigvals)[::-1]
    return mean, eigvecs[:, order].T, eigvals[order]


def pca_transform(X, mean, components, k):
    return (X - mean) @ components[:k].T


def pca_reconstruct(Z, mean, components, k):
    return Z @ components[:k] + mean


def run_pca():
    print("\n" + "=" * 70)
    print("PCA (eigendecomposition of covariance)")
    print("=" * 70)

    # Data with a clear low-dimensional structure: 3 latent factors -> 10 dims.
    n, d, latent = 500, 10, 3
    Z_true = np.random.randn(n, latent)
    A = np.random.randn(latent, d) * 2.0
    X = Z_true @ A + 0.3 * np.random.randn(n, d)

    mean, comps, evals = pca_fit(X)
    total_var = evals.sum()
    print("\nexplained variance ratio by component:")
    for i, v in enumerate(evals):
        print(f"  PC{i + 1:2d}: {v / total_var:.4f}")

    # sklearn check: eigenvalues and components (up to sign) must match.
    from sklearn.decomposition import PCA
    sk = PCA().fit(X)
    ev_diff = float(np.max(np.abs(evals - sk.explained_variance_)))
    print(f"\nmax |our eigenvalues - sklearn explained_variance_| = {ev_diff:.2e}")
    assert ev_diff < 1e-8, "eigenvalues disagree with sklearn"

    # Components can differ by sign; compare |cosine| of each pair.
    cos = np.abs(np.sum(comps * sk.components_, axis=1))
    print(f"min |cos(our PC, sklearn PC)| over components = {cos.min():.6f}")
    assert np.all(cos > 1 - 1e-8), "component directions disagree with sklearn"
    print("PASS: matches sklearn PCA (up to sign).")

    # Reconstruction error vs number of components.
    print("\nreconstruction error (mean squared, per entry) vs k:")
    prev = np.inf
    for k in range(1, d + 1):
        Zk = pca_transform(X, mean, comps, k)
        Xr = pca_reconstruct(Zk, mean, comps, k)
        err = float(np.mean((X - Xr) ** 2))
        # identity check: error = mean of discarded eigenvalues * (d-? scaling)
        marker = "  <- latent dim of the generator" if k == latent else ""
        print(f"  k = {k:2d}: MSE = {err:10.6f}{marker}")
        assert err <= prev + 1e-12, "reconstruction error should be non-increasing"
        prev = err
    print("Note the cliff after k=3: the data was generated from 3 latent "
          "factors,\nso 3 components capture nearly everything; the rest is "
          "the 0.3-sigma noise floor.")


if __name__ == "__main__":
    run_kmeans()
    run_pca()
    print("\nAll checks passed.")
