"""Decision tree and random forest from scratch (numpy only).

What this script does:
  1. A binary decision-tree classifier built recursively with entropy-based
     (information gain) splits and a max_depth cap.
  2. A random forest on top of it: bootstrap sampling of rows + random
     feature subsampling at every split, majority-vote prediction.
  3. Evaluation on synthetic data with a comparison against sklearn's
     DecisionTreeClassifier and RandomForestClassifier.

Run:  uv run python ml/02_classical_ml/trees_and_ensembles_scratch.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# Decision tree
# ----------------------------------------------------------------------------
def entropy(y):
    """H(y) = -sum_c p_c log2 p_c, in bits."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "prediction", "proba")

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 prediction=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction   # set on leaves
        self.proba = proba             # P(class=1) at the leaf


class DecisionTree:
    """Entropy-based CART-style binary classification tree."""

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # None = use all features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """Return (feature, threshold, info_gain) of the best split, or None."""
        n, d = X.shape
        parent_H = entropy(y)
        best = None  # (gain, feature, threshold)

        if self.max_features is None:
            features = np.arange(d)
        else:
            k = min(self.max_features, d)
            features = self.rng.choice(d, size=k, replace=False)

        for f in features:
            values = np.unique(X[:, f])
            if len(values) < 2:
                continue
            # Candidate thresholds: midpoints between consecutive unique values.
            thresholds = (values[:-1] + values[1:]) / 2.0
            for t in thresholds:
                mask = X[:, f] <= t
                n_l = int(mask.sum())
                n_r = n - n_l
                if n_l == 0 or n_r == 0:
                    continue
                H_l = entropy(y[mask])
                H_r = entropy(y[~mask])
                gain = parent_H - (n_l / n) * H_l - (n_r / n) * H_r
                if best is None or gain > best[0]:
                    best = (gain, f, t)

        if best is None or best[0] <= 1e-12:
            return None
        return best[1], best[2], best[0]

    def _build(self, X, y, depth):
        # Stopping conditions -> leaf with majority class + class-1 fraction.
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return self._leaf(y)
        split = self._best_split(X, y)
        if split is None:
            return self._leaf(y)
        f, t, _gain = split
        mask = X[:, f] <= t
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return Node(feature=f, threshold=t, left=left, right=right)

    @staticmethod
    def _leaf(y):
        p1 = float(np.mean(y)) if len(y) else 0.0
        return Node(prediction=int(p1 >= 0.5), proba=p1)

    def _predict_one(self, x):
        node = self.root
        while node.prediction is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def depth(self):
        def _d(node):
            if node.prediction is not None:
                return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root)


# ----------------------------------------------------------------------------
# Random forest
# ----------------------------------------------------------------------------
class RandomForest:
    """Bagging + per-split feature subsampling over DecisionTree above."""

    def __init__(self, n_trees=25, max_depth=6, max_features="sqrt", seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        n, d = X.shape
        if self.max_features == "sqrt":
            mf = max(1, int(np.sqrt(d)))
        else:
            mf = self.max_features
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, size=n)          # bootstrap sample (with replacement)
            tree = DecisionTree(max_depth=self.max_depth, max_features=mf,
                                rng=rng)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        votes = np.stack([t.predict(X) for t in self.trees])  # (n_trees, n)
        return (votes.mean(axis=0) >= 0.5).astype(int)


# ----------------------------------------------------------------------------
# Synthetic data: two interleaving structures a linear model cannot separate
# ----------------------------------------------------------------------------
def make_data(n=1200, d_noise=3):
    """XOR-ish problem in 2 informative dims + pure-noise dims."""
    X_inf = np.random.uniform(-2, 2, size=(n, 2))
    y = ((X_inf[:, 0] * X_inf[:, 1]) > 0).astype(int)   # XOR of the signs
    # flip 5% of labels for realism
    flip = np.random.rand(n) < 0.05
    y[flip] = 1 - y[flip]
    X_noise = np.random.randn(n, d_noise)
    X = np.hstack([X_inf, X_noise])
    return X, y


def main():
    X, y = make_data()
    n_train = 800
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    print("=" * 70)
    print("DECISION TREE (from scratch, entropy splits)")
    print("=" * 70)
    print(f"data: {Xtr.shape[0]} train / {Xte.shape[0]} test, "
          f"{X.shape[1]} features (2 informative, 3 noise)")
    print(f"baseline (majority class) test acc = "
          f"{max(np.mean(yte), 1 - np.mean(yte)):.4f}")

    for depth in (2, 4, 8):
        tree = DecisionTree(max_depth=depth).fit(Xtr, ytr)
        acc_tr = float(np.mean(tree.predict(Xtr) == ytr))
        acc_te = float(np.mean(tree.predict(Xte) == yte))
        print(f"  max_depth={depth}: train acc = {acc_tr:.4f}, "
              f"test acc = {acc_te:.4f}, actual depth = {tree.depth()}")

    tree = DecisionTree(max_depth=8).fit(Xtr, ytr)
    acc_tree = float(np.mean(tree.predict(Xte) == yte))

    from sklearn.tree import DecisionTreeClassifier
    sk_tree = DecisionTreeClassifier(max_depth=8, criterion="entropy",
                                     random_state=42).fit(Xtr, ytr)
    sk_tree_acc = float(sk_tree.score(Xte, yte))
    print(f"\nsklearn DecisionTree (depth 8, entropy) test acc = {sk_tree_acc:.4f} "
          f"(ours: {acc_tree:.4f})")
    assert abs(sk_tree_acc - acc_tree) <= 0.05, "tree accuracy far from sklearn"
    print("PASS: within 5% of sklearn's tree.")

    print("\n" + "=" * 70)
    print("RANDOM FOREST (bootstrap + feature subsampling)")
    print("=" * 70)
    forest = RandomForest(n_trees=40, max_depth=8, max_features=3,
                          seed=42).fit(Xtr, ytr)
    acc_rf_tr = float(np.mean(forest.predict(Xtr) == ytr))
    acc_rf = float(np.mean(forest.predict(Xte) == yte))
    print(f"  40 trees, depth 8, 3 of {X.shape[1]} features per split")
    print(f"  train acc = {acc_rf_tr:.4f}, test acc = {acc_rf:.4f}")
    print(f"  single deep tree test acc was {acc_tree:.4f} "
          f"-> forest gain: {acc_rf - acc_tree:+.4f}")

    from sklearn.ensemble import RandomForestClassifier
    sk_rf = RandomForestClassifier(n_estimators=40, max_depth=8,
                                   max_features=3, criterion="entropy",
                                   random_state=42).fit(Xtr, ytr)
    sk_rf_acc = float(sk_rf.score(Xte, yte))
    print(f"\nsklearn RandomForest test acc = {sk_rf_acc:.4f} (ours: {acc_rf:.4f})")
    assert abs(sk_rf_acc - acc_rf) <= 0.05, "forest accuracy far from sklearn"
    print("PASS: within 5% of sklearn's forest.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
