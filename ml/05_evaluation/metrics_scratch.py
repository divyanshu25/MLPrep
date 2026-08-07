"""Classification metrics from scratch (numpy only; sklearn used only to verify).

Implements: confusion matrix, precision/recall/F1, ROC curve + AUC (threshold sweep
AND rank-statistic formula), PR curve + average precision. Then demonstrates that
AUC is blind to class imbalance while precision/AP collapses.

Run: uv run python ml/05_evaluation/metrics_scratch.py
"""

import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Confusion matrix and derived metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """Return (tp, fp, fn, tn) for binary labels in {0, 1}."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def precision_score(y_true, y_pred):
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    tp, _, fn, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# ROC curve + AUC, two independent ways
# ---------------------------------------------------------------------------

def roc_curve(y_true, scores):
    """Threshold sweep. Returns (fpr, tpr, thresholds), thresholds descending.

    Sort by score descending; lowering the threshold past each distinct score
    admits one more batch of predictions as positive. Positives move TPR up,
    negatives move FPR right.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    # cumulative TP/FP as threshold sweeps down
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    # keep only the last index of each run of tied scores
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, y_sorted.size - 1]

    tps, fps = tps[idx], fps[idx]
    n_pos, n_neg = tps[-1], fps[-1]
    tpr = np.r_[0.0, tps / max(n_pos, 1)]
    fpr = np.r_[0.0, fps / max(n_neg, 1)]
    thresholds = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    """Area under the (fpr, tpr) staircase via the trapezoid rule."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed trapz
    return float(trapezoid(tpr, fpr))


def auc_rank_statistic(y_true, scores):
    """AUC = P(score of random positive > score of random negative), ties = 1/2.

    Computed via rank sums (Mann-Whitney U), O(n log n).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)

    # midranks (average rank for ties), 1-based
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average of ranks i+1..j+1
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# PR curve + average precision
# ---------------------------------------------------------------------------

def pr_curve(y_true, scores):
    """Returns (precision, recall) over the descending-threshold sweep."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    tps = np.cumsum(y_sorted)
    preds = np.arange(1, y_sorted.size + 1)

    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, y_sorted.size - 1]

    precision = tps[idx] / preds[idx]
    recall = tps[idx] / max(int(y_true.sum()), 1)
    return precision, recall


def average_precision(y_true, scores):
    """AP = sum over recall steps of precision, i.e. sum_k (R_k - R_{k-1}) P_k."""
    precision, recall = pr_curve(y_true, scores)
    recall = np.r_[0.0, recall]
    return float(np.sum(np.diff(recall) * precision))


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_hand_example():
    print("=" * 72)
    print("1) The 6-example table from notes.md, by hand")
    print("=" * 72)
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.40, 0.20])
    labels = np.array([1, 1, 0, 1, 0, 0])
    fpr, tpr, thr = roc_curve(labels, scores)
    print(f"{'thresh':>8} {'FPR':>6} {'TPR':>6}")
    for t, f, tp in zip(thr, fpr, tpr):
        print(f"{t:>8.2f} {f:>6.3f} {tp:>6.3f}")
    a1 = auc_trapezoid(fpr, tpr)
    a2 = auc_rank_statistic(labels, scores)
    print(f"AUC (trapezoid over sweep) : {a1:.6f}")
    print(f"AUC (rank statistic)       : {a2:.6f}   (expected 8/9 = {8/9:.6f})")
    assert abs(a1 - a2) < 1e-12 and abs(a1 - 8 / 9) < 1e-12


def make_scores(n_pos, n_neg, rng):
    """Synthetic scores: positives ~ N(1.5, 1), negatives ~ N(0, 1).

    Same conditional score distributions regardless of class ratio -> the ROC
    curve / AUC is identical in expectation; only the mix changes.
    """
    s_pos = rng.normal(1.5, 1.0, n_pos)
    s_neg = rng.normal(0.0, 1.0, n_neg)
    scores = np.r_[s_pos, s_neg]
    labels = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    return labels, scores


def demo_metrics_and_verify():
    print()
    print("=" * 72)
    print("2) Larger synthetic set: verify everything against sklearn")
    print("=" * 72)
    rng = np.random.default_rng(42)
    labels, scores = make_scores(500, 500, rng)
    preds = (scores >= 0.75).astype(int)

    tp, fp, fn, tn = confusion_matrix(labels, preds)
    p, r, f1 = precision_score(labels, preds), recall_score(labels, preds), f1_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, scores)
    a_sweep = auc_trapezoid(fpr, tpr)
    a_rank = auc_rank_statistic(labels, scores)
    ap = average_precision(labels, scores)

    print(f"confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={p:.6f}  recall={r:.6f}  F1={f1:.6f}")
    print(f"AUC sweep={a_sweep:.6f}  AUC rank={a_rank:.6f}  (diff={abs(a_sweep-a_rank):.2e})")
    print(f"average precision={ap:.6f}")
    assert abs(a_sweep - a_rank) < 1e-10, "two AUC computations must agree"

    try:
        from sklearn import metrics as skm
    except ImportError:
        print("[sklearn not installed -- skipping verification]")
        return
    sk_p = skm.precision_score(labels, preds)
    sk_r = skm.recall_score(labels, preds)
    sk_f = skm.f1_score(labels, preds)
    sk_auc = skm.roc_auc_score(labels, scores)
    sk_ap = skm.average_precision_score(labels, scores)
    print("sklearn check:")
    print(f"  precision {sk_p:.6f}  recall {sk_r:.6f}  F1 {sk_f:.6f}  AUC {sk_auc:.6f}  AP {sk_ap:.6f}")
    assert abs(p - sk_p) < 1e-10
    assert abs(r - sk_r) < 1e-10
    assert abs(f1 - sk_f) < 1e-10
    assert abs(a_rank - sk_auc) < 1e-10
    assert abs(ap - sk_ap) < 1e-10
    print("  all match to 1e-10. OK")


def demo_imbalance():
    print()
    print("=" * 72)
    print("3) Imbalance demo: AUC stays flat, precision/AP collapses")
    print("=" * 72)
    print("Same score distributions (pos ~ N(1.5,1), neg ~ N(0,1)); only the")
    print("class ratio changes. Threshold fixed at 1.0 for precision/recall.")
    print()
    rng = np.random.default_rng(42)
    n_pos = 2000
    header = f"{'neg:pos':>10} {'prevalence':>11} {'AUC':>8} {'AP':>8} {'prec@t=1':>9} {'rec@t=1':>8}"
    print(header)
    print("-" * len(header))
    for ratio in [1, 10, 100]:
        labels, scores = make_scores(n_pos, n_pos * ratio, rng)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = auc_trapezoid(fpr, tpr)
        ap = average_precision(labels, scores)
        preds = (scores >= 1.0).astype(int)
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)
        prev = n_pos / (n_pos * (1 + ratio))
        print(f"{ratio:>9}:1 {prev:>11.4f} {auc:>8.4f} {ap:>8.4f} {p:>9.4f} {r:>8.4f}")
    print()
    print("Reading: 100x more negatives -> AUC barely moves (it is prevalence-free),")
    print("but precision at the same operating point collapses, because")
    print("precision = pi*TPR / (pi*TPR + (1-pi)*FPR) and (1-pi)*FPR now dominates.")


if __name__ == "__main__":
    demo_hand_example()
    demo_metrics_and_verify()
    demo_imbalance()
