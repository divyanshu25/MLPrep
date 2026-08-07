"""Gaussian mixture model fit by EM, from scratch (numpy only).

What this script does:
  1. EM for a full-covariance GMM with log-domain responsibilities
     (logsumexp), on synthetic 2D data drawn from 3 known Gaussians.
     Prints the log-likelihood trace and asserts it is monotonically
     non-decreasing (the ELBO sandwich proof of notes.md SS11.2, made
     visible).
  2. Compares recovered means/weights against the true generating
     parameters, and against sklearn's GaussianMixture.
  3. Demonstrates the k-means connection (notes.md SS11.4): EM with
     covariances frozen to a tiny shared spherical eps*I produces exactly
     the same hard assignments as Lloyd's algorithm from the same init.

Run:  uv run python ml/02_classical_ml/gmm_em_scratch.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# Numerically stable building blocks
# ----------------------------------------------------------------------------
def logsumexp(a, axis=None):
    """log(sum(exp(a))) without overflow: subtract the max first."""
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def log_gaussian(X, mu, Sigma):
    """log N(x; mu, Sigma) for every row of X. Uses a Cholesky solve
    (never inverts Sigma explicitly)."""
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    diff = X - mu                                  # (n, d)
    # Solve L z = diff^T  =>  z = L^{-1} diff^T; then mahalanobis = ||z||^2.
    z = np.linalg.solve(L, diff.T)                 # (d, n)
    maha = np.sum(z ** 2, axis=0)                  # (n,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


# ----------------------------------------------------------------------------
# EM for a GMM
# ----------------------------------------------------------------------------
def gmm_em(X, k, n_iters=200, tol=1e-8, seed=0, init_means=None,
           freeze_cov=None, reg_covar=1e-6):
    """EM for a full-covariance GMM. Returns (pi, means, covs, ll_trace, gamma).

    freeze_cov: if given (a single (d,d) matrix), every component's covariance
    is held fixed at it and weights are held uniform -- the constrained model
    whose eps->0 limit is k-means (notes SS11.4).
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    if init_means is None:
        means = X[rng.choice(n, size=k, replace=False)].copy()
    else:
        means = init_means.copy()
    covs = np.array([np.cov(X.T) + reg_covar * np.eye(d) for _ in range(k)])
    if freeze_cov is not None:
        covs = np.array([freeze_cov.copy() for _ in range(k)])
    pi = np.full(k, 1.0 / k)

    ll_trace = []
    gamma = None
    for _ in range(n_iters):
        # ---- E-step, entirely in log space ----
        # log_joint[i, j] = log pi_j + log N(x_i; mu_j, Sigma_j)
        log_joint = np.stack(
            [np.log(pi[j]) + log_gaussian(X, means[j], covs[j]) for j in range(k)],
            axis=1)                                          # (n, k)
        log_norm = logsumexp(log_joint, axis=1)              # (n,) = log p(x_i)
        log_gamma = log_joint - log_norm[:, None]            # log responsibilities
        gamma = np.exp(log_gamma)                            # (n, k)

        ll = float(np.sum(log_norm))
        ll_trace.append(ll)
        if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol:
            break

        # ---- M-step (closed forms derived in notes SS11.3) ----
        N = gamma.sum(axis=0)                                # effective counts
        means = (gamma.T @ X) / N[:, None]                   # weighted means
        if freeze_cov is None:
            pi = N / n
            for j in range(k):
                diff = X - means[j]
                covs[j] = (gamma[:, j, None] * diff).T @ diff / N[j]
                covs[j] += reg_covar * np.eye(d)
        # frozen: covariances and (uniform) weights stay put; only means move.
    return pi, means, covs, ll_trace, gamma


def lloyds_kmeans(X, init_centers, n_iters=200):
    """Plain Lloyd's algorithm from given initial centers."""
    centers = init_centers.copy()
    labels = None
    for _ in range(n_iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        for j in range(len(centers)):
            mask = new_labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
        if labels is not None and np.array_equal(labels, new_labels):
            break
        labels = new_labels
    return centers, labels


def match_components(est_means, true_means):
    """Greedy matching of estimated components to true ones (k is small)."""
    k = len(true_means)
    remaining = list(range(k))
    order = np.zeros(k, dtype=int)
    for t in range(k):
        dists = [np.linalg.norm(est_means[j] - true_means[t]) for j in remaining]
        pick = remaining[int(np.argmin(dists))]
        order[t] = pick
        remaining.remove(pick)
    return order


# ----------------------------------------------------------------------------
# Part 1: fit a 3-component GMM and check the EM guarantees
# ----------------------------------------------------------------------------
def run_gmm():
    print("=" * 70)
    print("GMM VIA EM (log-domain responsibilities, full covariances)")
    print("=" * 70)

    # Three anisotropic Gaussians -- exactly the case where k-means'
    # spherical assumption is wrong and a full-covariance GMM shines.
    true_means = np.array([[0.0, 0.0], [5.0, 1.0], [2.0, 5.0]])
    true_covs = np.array([
        [[1.0, 0.6], [0.6, 1.0]],       # tilted ellipse
        [[1.2, -0.4], [-0.4, 0.5]],     # tilted the other way
        [[0.4, 0.0], [0.0, 1.5]],       # tall and thin
    ])
    true_pi = np.array([0.5, 0.3, 0.2])
    n = 1500
    counts = np.random.multinomial(n, true_pi)
    X = np.vstack([
        np.random.multivariate_normal(true_means[j], true_covs[j], counts[j])
        for j in range(3)
    ])

    pi, means, covs, ll_trace, gamma = gmm_em(X, k=3, seed=3)

    print(f"\nconverged in {len(ll_trace)} EM iterations; log-likelihood trace "
          "(every 5th):")
    for i in range(0, len(ll_trace), 5):
        print(f"  iter {i:3d}: LL = {ll_trace[i]:14.4f}")
    print(f"  iter {len(ll_trace)-1:3d}: LL = {ll_trace[-1]:14.4f}  (final)")

    diffs = np.diff(ll_trace)
    assert np.all(diffs >= -1e-7), "log-likelihood decreased -- EM proof violated!"
    print("PASS: log-likelihood is monotonically non-decreasing "
          "(ELBO sandwich, notes SS11.2).")

    order = match_components(means, true_means)
    print("\nrecovered vs. true parameters (components matched greedily):")
    for t in range(3):
        j = order[t]
        print(f"  comp {t}: true mean {np.round(true_means[t], 3)}  "
              f"est mean {np.round(means[j], 3)}  "
              f"| true pi {true_pi[t]:.3f}  est pi {pi[j]:.3f}")
    mean_err = max(np.linalg.norm(means[order[t]] - true_means[t]) for t in range(3))
    print(f"max mean error: {mean_err:.4f}")
    assert mean_err < 0.25, "recovered means too far from truth"
    print("PASS: recovered means within 0.25 of the generating means.")

    # ---- compare against sklearn ----
    from sklearn.mixture import GaussianMixture
    sk = GaussianMixture(n_components=3, covariance_type="full",
                         random_state=0, n_init=5).fit(X)
    sk_order = match_components(sk.means_, true_means)
    print("\nsklearn GaussianMixture comparison:")
    sk_ll = sk.score(X) * len(X)      # sklearn reports mean per-sample LL
    print(f"  ours   final LL = {ll_trace[-1]:14.4f}")
    print(f"  sklearn final LL = {sk_ll:14.4f}")
    for t in range(3):
        print(f"  comp {t}: ours {np.round(means[order[t]], 3)}  "
              f"sklearn {np.round(sk.means_[sk_order[t]], 3)}")
    max_gap = max(np.linalg.norm(means[order[t]] - sk.means_[sk_order[t]])
                  for t in range(3))
    print(f"  max |ours - sklearn| over means: {max_gap:.4f}")
    assert max_gap < 0.1, "our EM disagrees with sklearn's fit"
    assert ll_trace[-1] >= sk_ll - 1.0, "our LL notably below sklearn's"
    print("PASS: matches sklearn (same optimum up to tolerance).")
    return X


# ----------------------------------------------------------------------------
# Part 2: the k-means connection (notes SS11.4)
# ----------------------------------------------------------------------------
def run_kmeans_connection(X):
    print()
    print("=" * 70)
    print("K-MEANS = EM WITH TINY FROZEN SPHERICAL COVARIANCES")
    print("=" * 70)

    d = X.shape[1]
    eps = 1e-3
    rng = np.random.default_rng(11)
    init = X[rng.choice(len(X), size=3, replace=False)].copy()

    # EM with Sigma_j frozen at eps*I and uniform weights: responsibilities
    # are a softmax over -||x - mu_j||^2 / (2 eps) -> one-hot as eps -> 0.
    _, em_means, _, ll_trace, gamma = gmm_em(
        X, k=3, init_means=init, freeze_cov=eps * np.eye(d), reg_covar=0.0)
    em_labels = gamma.argmax(axis=1)
    hardness = gamma.max(axis=1).min()
    print(f"\nfrozen-covariance EM: eps = {eps}; min over points of "
          f"max responsibility = {hardness:.6f}")
    print("(responsibilities are effectively one-hot -- soft E-step has "
          "collapsed to the k-means assignment step)")

    km_centers, km_labels = lloyds_kmeans(X, init)

    agree = float(np.mean(em_labels == km_labels))
    print(f"\nassignment agreement with Lloyd's from the same init: "
          f"{agree:.4%}")
    assert agree == 1.0, "frozen-covariance EM did not reproduce k-means"
    print("PASS: hard-EM assignments are identical to k-means assignments.")
    print("center gap (EM weighted means vs. k-means means):",
          np.round(np.linalg.norm(em_means - km_centers, axis=1), 8))


if __name__ == "__main__":
    X = run_gmm()
    run_kmeans_connection(X)
    print("\nAll checks passed.")
