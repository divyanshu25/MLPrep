"""Gradients from scratch: numerical vs analytical derivatives, and the chain rule.

The single most useful debugging tool in ML: the gradient checker.
The derivative answers "if I wiggle x by a tiny h, how much does f wiggle?"
We can ALWAYS answer that question numerically, with no calculus at all:

    centered difference:  df/dx ~= (f(x + h) - f(x - h)) / (2h)

(centered is O(h^2) accurate vs O(h) for the one-sided version, because the
even-order Taylor terms cancel). So the workflow is: derive the gradient
analytically, then verify it against finite differences. If they match to
~1e-6 relative error, your math is right. This script does exactly that for
scalar functions, a vector quadratic form, a linear-regression loss w.r.t.
its weights, and a composed function to demonstrate the chain rule.

Run:  python gradients_from_scratch.py   (numpy only, seeded, no plotting)
"""

import numpy as np

np.random.seed(42)

H = 1e-5  # wiggle size for finite differences


# ---------------------------------------------------------------------------
# Numerical differentiation
# ---------------------------------------------------------------------------

def numerical_derivative(f, x, h=H):
    """Centered finite difference for a scalar function of a scalar."""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x, h=H):
    """Centered finite difference gradient for f: R^n -> R.

    Wiggle each coordinate independently and record how much f wiggles.
    This is n*2 evaluations of f -- expensive, but it needs zero calculus,
    which is what makes it the perfect referee for analytical gradients.
    """
    x = x.astype(float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old  # restore
        grad[idx] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    return grad


def relative_error(a, b):
    """Scale-free comparison; ~1e-7 or less means 'matches'."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return np.max(np.abs(a - b) / denom)


def report(name, numerical, analytical):
    err = relative_error(numerical, analytical)
    status = "MATCH" if err < 1e-6 else "MISMATCH"
    print(f"  {name}")
    print(f"    numerical : {np.round(np.asarray(numerical), 6)}")
    print(f"    analytical: {np.round(np.asarray(analytical), 6)}")
    print(f"    max relative error = {err:.2e}  [{status}]")
    assert err < 1e-6, f"gradient check failed for {name}"


# ---------------------------------------------------------------------------
# Part 1: scalar functions
# ---------------------------------------------------------------------------

def part1_scalar_functions():
    print("=" * 70)
    print("PART 1: scalar derivatives, numerical vs analytical")
    print("=" * 70)
    print(f"(centered differences with h = {H})\n")

    cases = [
        # (name, f, f', test point)
        ("f(x) = x^2         f'(x) = 2x        at x=3.0",
         lambda x: x ** 2, lambda x: 2 * x, 3.0),
        ("f(x) = sin(x)      f'(x) = cos(x)    at x=1.0",
         np.sin, np.cos, 1.0),
        ("f(x) = exp(x)      f'(x) = exp(x)    at x=0.5",
         np.exp, np.exp, 0.5),
        ("f(x) = 1/(1+e^-x)  f'(x) = s(1-s)    at x=2.0  (sigmoid)",
         lambda x: 1 / (1 + np.exp(-x)),
         lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
         2.0),
    ]
    for name, f, fprime, x0 in cases:
        report(name, numerical_derivative(f, x0), fprime(x0))
        print()


# ---------------------------------------------------------------------------
# Part 2: vector inputs -- the quadratic form x^T A x
# ---------------------------------------------------------------------------

def part2_quadratic_form():
    print("=" * 70)
    print("PART 2: gradient of the quadratic form f(x) = x^T A x")
    print("=" * 70)
    print("Claimed identity: grad = (A + A^T) x   (reduces to 2ax in 1-D)\n")

    A = np.array([[2.0, 1.0],
                  [0.0, 3.0]])  # deliberately NOT symmetric
    x0 = np.array([1.0, 2.0])
    print(f"  A =\n{A}\n  x = {x0},  f(x) = x^T A x = {x0 @ A @ x0}\n")

    f = lambda x: x @ A @ x
    report("grad of x^T A x  (non-symmetric A!)",
           numerical_gradient(f, x0.copy()),
           (A + A.T) @ x0)
    print("\n  Note: the naive guess '2Ax' would give",
          np.round(2 * A @ x0, 6), "-- wrong unless A is symmetric.\n")


# ---------------------------------------------------------------------------
# Part 3: linear regression loss w.r.t. the weights
# ---------------------------------------------------------------------------

def part3_linear_regression():
    print("=" * 70)
    print("PART 3: linear regression loss  L(w) = ||Xw - y||^2")
    print("=" * 70)
    print("Claimed gradient: grad_w L = 2 X^T (Xw - y)\n")

    n, d = 6, 3
    X = np.random.randn(n, d)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 0.1 * np.random.randn(n)  # targets with a little noise
    w0 = np.random.randn(d)  # some arbitrary weight vector

    loss = lambda w: np.sum((X @ w - y) ** 2)
    analytical = 2 * X.T @ (X @ w0 - y)

    print(f"  X shape {X.shape}, w = {np.round(w0, 4)}, "
          f"L(w) = {loss(w0):.6f}\n")
    report("grad_w ||Xw - y||^2", numerical_gradient(loss, w0.copy()), analytical)

    # Bonus sanity check: at the least-squares solution the gradient vanishes.
    w_star = np.linalg.solve(X.T @ X, X.T @ y)  # normal equations
    print(f"\n  Normal-equations solution w* = {np.round(w_star, 4)} "
          f"(true w = {true_w})")
    print(f"  ||grad at w*|| = {np.linalg.norm(2 * X.T @ (X @ w_star - y)):.2e} "
          "(~0: w* is a stationary point, as setting the gradient to 0 predicts)\n")


# ---------------------------------------------------------------------------
# Part 4: the chain rule, numerically
# ---------------------------------------------------------------------------

def part4_chain_rule():
    print("=" * 70)
    print("PART 4: chain rule -- wiggle rates MULTIPLY through a pipeline")
    print("=" * 70)
    print("Pipeline:  x --g--> z = x^2 + 1 --f--> y = sin(z)")
    print("Claim:     dy/dx = f'(z) * g'(x) = cos(x^2+1) * 2x\n")

    g = lambda x: x ** 2 + 1
    f = lambda z: np.sin(z)
    composed = lambda x: f(g(x))

    x0 = 0.7
    z0 = g(x0)

    dz_dx = numerical_derivative(g, x0)          # inner wiggle rate
    dy_dz = numerical_derivative(f, z0)          # outer wiggle rate
    dy_dx_direct = numerical_derivative(composed, x0)  # whole-pipeline rate

    print(f"  at x = {x0}:  z = g(x) = {z0:.6f}")
    print(f"  dz/dx (numerical, inner)          = {dz_dx:.8f}")
    print(f"  dy/dz (numerical, outer)          = {dy_dz:.8f}")
    print(f"  product dy/dz * dz/dx             = {dy_dz * dz_dx:.8f}")
    print(f"  dy/dx differentiating composition = {dy_dx_direct:.8f}")
    print(f"  analytical cos(x^2+1) * 2x        = {np.cos(x0**2 + 1) * 2 * x0:.8f}")
    err = relative_error(dy_dz * dz_dx, dy_dx_direct)
    print(f"  product vs direct: relative error = {err:.2e}  "
          f"[{'MATCH' if err < 1e-5 else 'MISMATCH'}]")
    assert err < 1e-5

    print("\n  This locality is ALL backprop needs: each op only knows its own")
    print("  local derivative; multiplying them along the graph (and summing")
    print("  where paths merge) yields the end-to-end gradient.\n")

    # A 3-deep composition, the shape of a tiny neural net:
    # x -> a = 3x -> b = sigmoid(a) -> y = b^2
    print("  Deeper pipeline: x -> a=3x -> b=sigmoid(a) -> y=b^2")
    sig = lambda t: 1 / (1 + np.exp(-t))
    x0 = 0.2
    a0 = 3 * x0
    b0 = sig(a0)
    local = [
        ("da/dx = 3", 3.0),
        ("db/da = s(a)(1-s(a))", b0 * (1 - b0)),
        ("dy/db = 2b", 2 * b0),
    ]
    prod = 1.0
    for name, val in local:
        prod *= val
        print(f"    local {name:24s} = {val:.8f}")
    direct = numerical_derivative(lambda x: sig(3 * x) ** 2, x0)
    print(f"    product of the three locals    = {prod:.8f}")
    print(f"    numerical dy/dx on composition = {direct:.8f}")
    err = relative_error(prod, direct)
    print(f"    relative error = {err:.2e}  "
          f"[{'MATCH' if err < 1e-5 else 'MISMATCH'}]")
    assert err < 1e-5
    print()


if __name__ == "__main__":
    part1_scalar_functions()
    part2_quadratic_form()
    part3_linear_regression()
    part4_chain_rule()
    print("All gradient checks passed.")
