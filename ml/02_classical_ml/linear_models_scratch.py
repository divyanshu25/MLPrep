"""Linear and logistic regression from scratch (numpy only).

What this script does:
  1. Linear regression solved two ways -- the closed-form normal equation and
     batch gradient descent -- and verifies they agree with each other and
     with sklearn.
  2. Logistic regression trained by gradient descent on synthetic 2-class
     data, with optional L2 regularization, verified against sklearn.

Run:  uv run python ml/02_classical_ml/linear_models_scratch.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def add_bias(X):
    """Prepend a column of ones so the bias is just another weight."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


# ----------------------------------------------------------------------------
# Linear regression
# ----------------------------------------------------------------------------
def linreg_normal_equation(X, y, l2=0.0):
    """Solve w = (X^T X + l2*I)^{-1} X^T y.

    With l2 > 0 this is ridge regression (we do not penalize the bias term:
    the first diagonal entry of the regularizer is zeroed).
    """
    Xb = add_bias(X)
    d = Xb.shape[1]
    reg = l2 * np.eye(d)
    reg[0, 0] = 0.0  # never penalize the intercept
    # Solve the linear system instead of forming an explicit inverse.
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def linreg_gradient_descent(X, y, lr=0.1, n_iters=5000, l2=0.0, verbose_every=1000):
    """Minimize (1/n)||Xw - y||^2 + l2*||w_{1:}||^2 by batch gradient descent."""
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    for it in range(n_iters):
        resid = Xb @ w - y                       # (n,)
        grad = (2.0 / n) * (Xb.T @ resid)        # gradient of the mean squared error
        grad[1:] += 2.0 * l2 * w[1:] / n         # match the normal-equation scaling
        w -= lr * grad
        if verbose_every and (it % verbose_every == 0 or it == n_iters - 1):
            print(f"    iter {it:5d}  MSE = {mse(y, Xb @ w):.6f}")
    return w


# ----------------------------------------------------------------------------
# Logistic regression
# ----------------------------------------------------------------------------
def sigmoid(z):
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logreg_gradient_descent(X, y, lr=0.5, n_iters=3000, l2=0.0, verbose_every=500):
    """Minimize mean cross-entropy + (l2/n)*||w_{1:}||^2.

    Gradient of the mean cross-entropy wrt w is (1/n) X^T (p - y):
    the famous (p - y) x form, identical in shape to linear regression's
    residual gradient -- no sigmoid derivative left over.
    """
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    for it in range(n_iters):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        grad[1:] += 2.0 * l2 * w[1:] / n
        w -= lr * grad
        if verbose_every and (it % verbose_every == 0 or it == n_iters - 1):
            print(f"    iter {it:5d}  CE = {cross_entropy(y, p):.6f}")
    return w


def logreg_predict_proba(X, w):
    return sigmoid(add_bias(X) @ w)


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def run_linear_regression():
    print("=" * 70)
    print("LINEAR REGRESSION")
    print("=" * 70)

    n, d = 500, 5
    X = np.random.randn(n, d)
    true_w = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
    true_b = 1.5
    y = X @ true_w + true_b + 0.5 * np.random.randn(n)

    # 1) Normal equation
    w_ne = linreg_normal_equation(X, y)
    print(f"\nNormal equation weights (bias first):\n  {np.round(w_ne, 4)}")
    print(f"  MSE = {mse(y, add_bias(X) @ w_ne):.6f}")

    # 2) Gradient descent
    print("\nGradient descent:")
    w_gd = linreg_gradient_descent(X, y, lr=0.1, n_iters=5000)
    print(f"  GD weights:\n  {np.round(w_gd, 4)}")

    diff = float(np.max(np.abs(w_ne - w_gd)))
    print(f"\nMax |normal_eq - GD| over weights: {diff:.2e}")
    assert diff < 1e-6, "normal equation and gradient descent disagree!"
    print("PASS: normal equation and gradient descent agree.")

    # 3) sklearn check
    from sklearn.linear_model import LinearRegression

    sk = LinearRegression().fit(X, y)
    sk_w = np.concatenate([[sk.intercept_], sk.coef_])
    sk_diff = float(np.max(np.abs(w_ne - sk_w)))
    print(f"Max |ours - sklearn| : {sk_diff:.2e}")
    assert sk_diff < 1e-8, "disagrees with sklearn!"
    print("PASS: matches sklearn LinearRegression.")

    # 4) Ridge: shrinkage sanity check
    w_ridge = linreg_normal_equation(X, y, l2=100.0)
    print(f"\nRidge (l2=100) weights:\n  {np.round(w_ridge, 4)}")
    print(
        f"||w||_2 unregularized = {np.linalg.norm(w_ne[1:]):.4f}, "
        f"ridge = {np.linalg.norm(w_ridge[1:]):.4f}  (ridge shrinks weights)"
    )
    assert np.linalg.norm(w_ridge[1:]) < np.linalg.norm(w_ne[1:])


def run_logistic_regression():
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION")
    print("=" * 70)

    # Two Gaussian blobs, linearly separable-ish.
    n_per = 400
    X0 = np.random.randn(n_per, 2) + np.array([-2.0, -1.0])
    X1 = np.random.randn(n_per, 2) + np.array([1.5, 1.0])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    n_train = 600
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    print("\nGradient descent (no regularization):")
    w = logreg_gradient_descent(Xtr, ytr, lr=0.5, n_iters=3000)
    p_tr = logreg_predict_proba(Xtr, w)
    p_te = logreg_predict_proba(Xte, w)
    acc_tr = float(np.mean((p_tr > 0.5) == ytr))
    acc_te = float(np.mean((p_te > 0.5) == yte))
    print(f"  weights (bias first): {np.round(w, 4)}")
    print(f"  train CE = {cross_entropy(ytr, p_tr):.4f}, acc = {acc_tr:.4f}")
    print(f"  test  CE = {cross_entropy(yte, p_te):.4f}, acc = {acc_te:.4f}")

    print("\nWith L2 regularization (l2=1.0):")
    w_l2 = logreg_gradient_descent(Xtr, ytr, lr=0.5, n_iters=3000, l2=1.0,
                                   verbose_every=0)
    p_te_l2 = logreg_predict_proba(Xte, w_l2)
    acc_te_l2 = float(np.mean((p_te_l2 > 0.5) == yte))
    print(f"  weights: {np.round(w_l2, 4)}")
    print(f"  ||w|| unreg = {np.linalg.norm(w[1:]):.4f}, "
          f"L2-reg = {np.linalg.norm(w_l2[1:]):.4f}  (shrunk)")
    print(f"  test acc = {acc_te_l2:.4f}")

    # sklearn check (C = 1/(2*l2_effective); use near-unregularized to compare)
    from sklearn.linear_model import LogisticRegression

    sk = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr)
    sk_acc = float(sk.score(Xte, yte))
    print(f"\nsklearn LogisticRegression test acc = {sk_acc:.4f} (ours: {acc_te:.4f})")
    assert abs(sk_acc - acc_te) <= 0.02, "accuracy differs from sklearn by > 2%"
    print("PASS: within 2% of sklearn's test accuracy.")


if __name__ == "__main__":
    run_linear_regression()
    run_logistic_regression()
    print("\nAll checks passed.")
