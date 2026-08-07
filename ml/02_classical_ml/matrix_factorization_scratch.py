"""Matrix factorization for collaborative filtering, from scratch (numpy only).

What this script does:
  1. Generates a synthetic low-rank ratings matrix (true rank 5, plus user/
     item biases and noise) and hides most entries -- the observed-entries-
     only setting where plain SVD fails (notes.md SS13.3).
  2. Trains MF with biases two ways:
       (a) SGD over observed ratings,
       (b) ALS, where each block update is exactly ridge regression
           (notes SS13.4: p_u = (Q_u^T Q_u + lam I)^{-1} Q_u^T r_u).
     Prints train/test RMSE for both against global-mean and item-mean
     baselines.
  3. Rank sweep r in {1, 2, 5, 10, 20}: test RMSE bottoms out at the true
     rank -- below it the model underfits, above it the extra factors fit
     noise.

Run:  uv run python ml/02_classical_ml/matrix_factorization_scratch.py
"""

import numpy as np

np.random.seed(42)

# ----------------------------------------------------------------------------
# Synthetic data: R = mu + b_u + b_i + P Q^T + noise, mostly missing
# ----------------------------------------------------------------------------
M_USERS, N_ITEMS, TRUE_RANK = 200, 120, 5
NOISE_STD = 0.4
OBS_FRAC = 0.25          # ~25% of entries observed (real data is far sparser)

mu_true = 3.5
bu_true = 0.5 * np.random.randn(M_USERS)
bi_true = 0.5 * np.random.randn(N_ITEMS)
P_true = np.random.randn(M_USERS, TRUE_RANK) / np.sqrt(TRUE_RANK)
Q_true = np.random.randn(N_ITEMS, TRUE_RANK)
R_full = (mu_true + bu_true[:, None] + bi_true[None, :] + P_true @ Q_true.T
          + NOISE_STD * np.random.randn(M_USERS, N_ITEMS))

# Observe a random subset; split observed entries 80/20 into train/test.
obs_mask = np.random.rand(M_USERS, N_ITEMS) < OBS_FRAC
obs_u, obs_i = np.where(obs_mask)
perm = np.random.permutation(len(obs_u))
n_train = int(0.8 * len(perm))
tr, te = perm[:n_train], perm[n_train:]
train = (obs_u[tr], obs_i[tr], R_full[obs_u[tr], obs_i[tr]])
test = (obs_u[te], obs_i[te], R_full[obs_u[te], obs_i[te]])
print(f"matrix {M_USERS}x{N_ITEMS} = {M_USERS*N_ITEMS} entries; "
      f"observed {len(obs_u)} ({len(obs_u)/(M_USERS*N_ITEMS):.1%}); "
      f"train {len(tr)}, test {len(te)}")


def rmse(pred, actual):
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
u_tr, i_tr, r_tr = train
u_te, i_te, r_te = test
g_mean = r_tr.mean()

# Item mean, regularized toward the global mean (an item with few ratings
# shouldn't get an extreme mean -- same shrinkage logic as ridge).
BETA = 5.0
item_sum = np.bincount(i_tr, weights=r_tr, minlength=N_ITEMS)
item_cnt = np.bincount(i_tr, minlength=N_ITEMS)
item_mean = (item_sum + BETA * g_mean) / (item_cnt + BETA)

baseline_rows = [
    ("global mean", rmse(np.full_like(r_tr, g_mean), r_tr),
     rmse(np.full_like(r_te, g_mean), r_te)),
    ("item mean (shrunk)", rmse(item_mean[i_tr], r_tr),
     rmse(item_mean[i_te], r_te)),
]


# ----------------------------------------------------------------------------
# (a) MF with biases, trained by SGD
# ----------------------------------------------------------------------------
def mf_sgd(train, rank, n_epochs=40, lr=0.02, lam=0.05, seed=0):
    u_tr, i_tr, r_tr = train
    rng = np.random.default_rng(seed)
    mu = r_tr.mean()
    bu = np.zeros(M_USERS)
    bi = np.zeros(N_ITEMS)
    P = 0.1 * rng.standard_normal((M_USERS, rank))
    Q = 0.1 * rng.standard_normal((N_ITEMS, rank))
    idx = np.arange(len(r_tr))
    for _ in range(n_epochs):
        rng.shuffle(idx)
        for t in idx:
            u, i, r = u_tr[t], i_tr[t], r_tr[t]
            e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (e - lam * bu[u])
            bi[i] += lr * (e - lam * bi[i])
            pu = P[u].copy()
            P[u] += lr * (e * Q[i] - lam * P[u])
            Q[i] += lr * (e * pu - lam * Q[i])
    return mu, bu, bi, P, Q


# ----------------------------------------------------------------------------
# (b) MF with biases, trained by ALS
# ----------------------------------------------------------------------------
def mf_als(train, rank, n_sweeps=15, lam=1.0, lam_b=5.0, seed=0):
    """Every update below is an exact coordinate minimization:
    biases are 1-D ridge problems, factor rows are rank-dim ridge
    regressions p_u = (Q_u^T Q_u + lam I)^{-1} Q_u^T r_u."""
    u_tr, i_tr, r_tr = train
    rng = np.random.default_rng(seed)
    mu = r_tr.mean()
    bu = np.zeros(M_USERS)
    bi = np.zeros(N_ITEMS)
    P = 0.1 * rng.standard_normal((M_USERS, rank))
    Q = 0.1 * rng.standard_normal((N_ITEMS, rank))

    # Precompute each user's rated items and each item's raters.
    items_of = [np.where(u_tr == u)[0] for u in range(M_USERS)]
    users_of = [np.where(i_tr == i)[0] for i in range(N_ITEMS)]
    I = np.eye(rank)

    for _ in range(n_sweeps):
        # --- user biases: b_u = sum(residual) / (n_u + lam_b), exact ridge ---
        interact = np.einsum("tk,tk->t", P[u_tr], Q[i_tr])
        resid = r_tr - (mu + bi[i_tr] + interact)
        bu = np.bincount(u_tr, weights=resid, minlength=M_USERS) / (
            np.bincount(u_tr, minlength=M_USERS) + lam_b)
        # --- item biases ---
        interact = np.einsum("tk,tk->t", P[u_tr], Q[i_tr])
        resid = r_tr - (mu + bu[u_tr] + interact)
        bi = np.bincount(i_tr, weights=resid, minlength=N_ITEMS) / (
            np.bincount(i_tr, minlength=N_ITEMS) + lam_b)
        # --- user factors: ridge regression per user ---
        for u in range(M_USERS):
            t = items_of[u]
            if len(t) == 0:
                continue
            Qu = Q[i_tr[t]]                                   # (n_u, rank)
            ru = r_tr[t] - (mu + bu[u] + bi[i_tr[t]])
            P[u] = np.linalg.solve(Qu.T @ Qu + lam * I, Qu.T @ ru)
        # --- item factors: ridge regression per item ---
        for i in range(N_ITEMS):
            t = users_of[i]
            if len(t) == 0:
                continue
            Pu = P[u_tr[t]]
            ri = r_tr[t] - (mu + bu[u_tr[t]] + bi[i])
            Q[i] = np.linalg.solve(Pu.T @ Pu + lam * I, Pu.T @ ri)
    return mu, bu, bi, P, Q


def predict(model, u, i):
    mu, bu, bi, P, Q = model
    return mu + bu[u] + bi[i] + np.einsum("tk,tk->t", P[u], Q[i])


# ----------------------------------------------------------------------------
# Train both at the true rank and report
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print(f"MF WITH BIASES AT RANK {TRUE_RANK} vs BASELINES")
print("=" * 70)

sgd_model = mf_sgd(train, rank=TRUE_RANK)
als_model = mf_als(train, rank=TRUE_RANK)

rows = baseline_rows + [
    ("MF + biases (SGD)", rmse(predict(sgd_model, u_tr, i_tr), r_tr),
     rmse(predict(sgd_model, u_te, i_te), r_te)),
    ("MF + biases (ALS)", rmse(predict(als_model, u_tr, i_tr), r_tr),
     rmse(predict(als_model, u_te, i_te), r_te)),
]
print(f"\n{'model':<22} {'train RMSE':>12} {'test RMSE':>12}")
for name, tr_r, te_r in rows:
    print(f"{name:<22} {tr_r:>12.4f} {te_r:>12.4f}")

noise_floor = NOISE_STD
print(f"\n(noise std = {noise_floor}: no model can beat ~{noise_floor} test RMSE)")
best_baseline = min(baseline_rows, key=lambda r: r[2])[2]
for name, _, te_r in rows[2:]:
    assert te_r < best_baseline, f"{name} failed to beat the best baseline"
print("PASS: both MF variants beat both baselines on test RMSE.")


# ----------------------------------------------------------------------------
# Rank sweep: test RMSE identifies the true rank
# ----------------------------------------------------------------------------
print()
print("=" * 70)
print("RANK SWEEP (ALS): test RMSE vs r")
print("=" * 70)
print(f"\n{'rank':>6} {'train RMSE':>12} {'test RMSE':>12}")
sweep = {}
for r in [1, 2, 5, 10, 20]:
    model = mf_als(train, rank=r)
    tr_r = rmse(predict(model, u_tr, i_tr), r_tr)
    te_r = rmse(predict(model, u_te, i_te), r_te)
    sweep[r] = te_r
    marker = "  <-- true rank" if r == TRUE_RANK else ""
    print(f"{r:>6} {tr_r:>12.4f} {te_r:>12.4f}{marker}")

best_rank = min(sweep, key=sweep.get)
print(f"\nbest test RMSE at rank {best_rank} (true rank {TRUE_RANK})")
print("read the pattern: train RMSE only falls with rank (more capacity), "
      "but test RMSE is U-shaped --")
print("underfit below the true rank, fitting noise above it.")
assert best_rank == TRUE_RANK, "rank sweep did not identify the true rank"
print("PASS: rank sweep identifies the true rank.")
