"""Training/serving skew and leakage, demonstrated concretely.

Two experiments on a toy fraud dataset with a time column:

  Demo 1 -- Preprocessing leakage: fitting a scaler on ALL data (including the
            future test period) before a temporal split inflates offline AUC
            relative to the honest pipeline (scaler fit on train only). The
            honest number is what production would deliver, because at serving
            time the scaler can only have seen the past.

  Demo 2 -- Point-in-time leakage: a feature aggregated over the WHOLE day of
            a transaction (which includes events *after* the prediction moment)
            inflates offline AUC. The point-in-time-correct version (events
            strictly before the transaction) does not. Worse: a model trained
            on the leaky feature collapses when served the only feature
            production can actually compute (the past-only one).

numpy + sklearn only. Seed fixed. No downloads, no plotting.

Run:  uv run python ml/06_ml_systems/feature_pipeline_demo.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RNG = np.random.default_rng(seed=7)


def fit_lr(X, y):
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model


def auc(model, X, y):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


# ---------------------------------------------------------------------------
# Demo 1: scaler fit on all data vs fit on train only (temporal drift)
# ---------------------------------------------------------------------------
def demo_scaler_leak():
    """Fraud amounts drift upward over time (adversaries escalate). A MinMax
    scaler with clipping fit on ALL data has seen the future range; the honest
    scaler (train only) clips unseen-future values at serving time, which is
    exactly what would happen in production."""
    n = 40_000
    t = np.sort(RNG.uniform(0.0, 1.0, size=n))          # time in [0, 1]
    y = (RNG.uniform(size=n) < 0.05).astype(int)         # 5% fraud

    # Informative feature: an "amount anomaly" score. Its overall scale drifts
    # upward over time -- and the drift accelerates (8*t^2), so the test
    # period reaches values the training period never saw. Fraud sits a fixed
    # offset above legit at any moment in time.
    drift = 10.0 * t**3
    noise = RNG.uniform(-2.0, 2.0, size=n)               # bounded noise
    x_amt = drift + 1.5 * y + noise                      # fraud offset: +1.5
    x_noise1 = RNG.normal(0.0, 1.0, size=n)
    x_noise2 = RNG.normal(0.0, 1.0, size=n)
    X = np.column_stack([x_amt, x_noise1, x_noise2])

    train = t < 0.70                                     # temporal split
    test = ~train
    X_tr, y_tr = X[train], y[train]
    X_te, y_te = X[test], y[test]

    # LEAKY: scaler fit on all rows, past and future, THEN split.
    leaky_scaler = MinMaxScaler(clip=True).fit(X)
    m_leaky = fit_lr(leaky_scaler.transform(X_tr), y_tr)
    auc_leaky = auc(m_leaky, leaky_scaler.transform(X_te), y_te)

    # HONEST: scaler fit on the training period only. At serving time the
    # future has drifted past the training range and gets clipped -- fraud
    # and legit tails collapse onto the same boundary value.
    honest_scaler = MinMaxScaler(clip=True).fit(X_tr)
    m_honest = fit_lr(honest_scaler.transform(X_tr), y_tr)
    auc_honest = auc(m_honest, honest_scaler.transform(X_te), y_te)

    print("=" * 72)
    print("Demo 1: scaler fit on ALL data vs fit on TRAIN only (temporal split)")
    print("=" * 72)
    print(f"  train rows: {train.sum():>6}   test rows: {test.sum():>6}   "
          f"fraud rate: {y.mean():.3f}")
    frac_clip = (X_te[:, 0] > X_tr[:, 0].max()).mean()
    print(f"  test rows whose amount exceeds the training-period max: "
          f"{frac_clip:.1%}")
    print(f"  offline test AUC, LEAKY  scaler (fit on all data) : "
          f"{auc_leaky:.4f}")
    print(f"  offline test AUC, HONEST scaler (fit on train)    : "
          f"{auc_honest:.4f}")
    print(f"  inflation from the leak                            : "
          f"+{auc_leaky - auc_honest:.4f}")
    print("  The leaky number is unachievable in production: the serving-time")
    print("  scaler cannot have been fit on data from the future.\n")
    return auc_leaky, auc_honest


# ---------------------------------------------------------------------------
# Demo 2: point-in-time-incorrect aggregate feature (future leakage)
# ---------------------------------------------------------------------------
def make_transactions():
    """Cards transact over 30 days. Some cards get compromised at a random
    time; from that moment the fraudster fires a rapid burst of transactions.
    So a card's FULL-DAY transaction count on a fraud day is inflated by the
    burst itself -- including transactions that happen AFTER the one we are
    scoring. Using it as a feature is using the future."""
    n_cards, n_days = 4000, 30
    rows = []  # (card, time, y, amount, mismatch)
    for card in range(n_cards):
        compromised = RNG.uniform() < 0.06
        t_comp = RNG.uniform(0, n_days) if compromised else np.inf
        # normal activity: ~0.4 transactions/day
        n_legit = RNG.poisson(0.4 * n_days)
        for time in RNG.uniform(0, n_days, size=n_legit):
            amount = RNG.lognormal(3.0, 0.7)
            mismatch = RNG.uniform() < 0.05
            rows.append((card, time, 0, amount, mismatch))
        # fraud burst: 4-12 transactions within ~half a day of compromise
        if compromised:
            n_burst = RNG.integers(4, 13)
            for time in t_comp + RNG.uniform(0, 0.5, size=n_burst):
                if time < n_days:
                    amount = RNG.lognormal(3.4, 0.7)   # only slightly larger
                    mismatch = RNG.uniform() < 0.25    # only mildly telling
                    rows.append((card, time, 1, amount, mismatch))
    rows.sort(key=lambda r: r[1])
    card = np.array([r[0] for r in rows])
    time = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    amount = np.array([r[3] for r in rows])
    mismatch = np.array([r[4] for r in rows], dtype=float)
    return card, time, y, amount, mismatch


def day_count_features(card, time):
    """For each transaction, two versions of 'transactions by this card today':
       leaky   = count over the WHOLE calendar day (includes the future),
       correct = count over the day STRICTLY BEFORE this transaction."""
    day = np.floor(time).astype(int)
    n = len(card)
    leaky = np.zeros(n)
    correct = np.zeros(n)
    from collections import defaultdict
    per_key = defaultdict(list)          # (card, day) -> row indices
    for i in range(n):
        per_key[(card[i], day[i])].append(i)
    for idx in per_key.values():
        total = len(idx)
        # rows were globally time-sorted, so idx is time-ordered within the key
        for rank, i in enumerate(idx):
            leaky[i] = total - 1         # all other txns today, past AND future
            correct[i] = rank            # txns earlier today only
    return leaky, correct


def demo_point_in_time():
    card, time, y, amount, mismatch = make_transactions()
    leaky_cnt, correct_cnt = day_count_features(card, time)

    base = np.column_stack([np.log(amount), mismatch])
    X_leaky = np.column_stack([base, leaky_cnt])
    X_correct = np.column_stack([base, correct_cnt])

    train = time < 24.0                                  # temporal split
    test = ~train
    y_tr, y_te = y[train], y[test]

    def run(X_train_feats, X_test_feats):
        scaler = StandardScaler().fit(X_train_feats[train])  # honest scaler
        m = fit_lr(scaler.transform(X_train_feats[train]), y_tr)
        return auc(m, scaler.transform(X_test_feats[test]), y_te)

    # Honest: point-in-time-correct feature at train AND test.
    auc_honest = run(X_correct, X_correct)
    # Leaky offline eval: full-day aggregate at train AND test. Offline you
    # CAN compute this (the whole log exists), which is exactly the trap.
    auc_leaky = run(X_leaky, X_leaky)
    # Reality check: model trained on the leaky feature, but served the only
    # feature production can compute at decision time (the past-only count).
    auc_deployed = run(X_leaky, X_correct)
    # Baseline without any aggregate, for reference.
    auc_base = run(base.reshape(len(y), -1), base.reshape(len(y), -1))

    print("=" * 72)
    print("Demo 2: point-in-time-incorrect aggregate ('card's txn count today')")
    print("=" * 72)
    print(f"  transactions: {len(y)}   fraud rate: {y.mean():.3f}   "
          f"train rows: {train.sum()}   test rows: {test.sum()}")
    print(f"  offline test AUC, base features only               : "
          f"{auc_base:.4f}")
    print(f"  offline test AUC, HONEST past-only count           : "
          f"{auc_honest:.4f}")
    print(f"  offline test AUC, LEAKY full-day count             : "
          f"{auc_leaky:.4f}   <- looks best; it is a lie")
    print(f"  DEPLOYED: trained on leaky, served past-only count : "
          f"{auc_deployed:.4f}   <- what production would see")
    print(f"  inflation of the leaky offline number vs honest    : "
          f"+{auc_leaky - auc_honest:.4f}")
    print(f"  deployed model vs honest model                     : "
          f"{auc_deployed - auc_honest:+.4f}")
    print("  Why: the fraud burst happens AFTER the transaction being scored,")
    print("  so the full-day count is partly an encoding of the label itself.")
    print("  At serving time only the past exists, so the signal evaporates --")
    print("  and the model, trained to lean on it, does worse than the honest")
    print("  model that never saw the future.\n")
    return auc_base, auc_honest, auc_leaky, auc_deployed


def main():
    d1_leaky, d1_honest = demo_scaler_leak()
    d2_base, d2_honest, d2_leaky, d2_deployed = demo_point_in_time()

    print("=" * 72)
    print("Summary: honest vs leaky offline AUC")
    print("=" * 72)
    print(f"  {'pipeline':<44}{'offline AUC':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Demo 1  scaler fit on train only (honest)':<44}"
          f"{d1_honest:>12.4f}")
    print(f"  {'Demo 1  scaler fit on all data (leaky)':<44}"
          f"{d1_leaky:>12.4f}")
    print(f"  {'Demo 2  base features, no aggregate':<44}"
          f"{d2_base:>12.4f}")
    print(f"  {'Demo 2  past-only aggregate (honest)':<44}"
          f"{d2_honest:>12.4f}")
    print(f"  {'Demo 2  full-day aggregate (leaky)':<44}"
          f"{d2_leaky:>12.4f}")
    print(f"  {'Demo 2  leaky-trained model in production':<44}"
          f"{d2_deployed:>12.4f}")
    print()
    print("  Moral: both leaks make OFFLINE numbers better while making the")
    print("  PRODUCTION model the same or worse. If a new feature or pipeline")
    print("  change improves offline metrics suspiciously much, first ask:")
    print("  could this computation have been done at serving time, using only")
    print("  data from strictly before the prediction moment?")


if __name__ == "__main__":
    main()
