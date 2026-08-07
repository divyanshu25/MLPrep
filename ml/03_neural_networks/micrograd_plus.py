"""micrograd_plus.py - a scalar autograd engine from scratch, Karpathy-style.

The whole idea in three sentences:

  1. Every arithmetic op builds a node in a computation graph and REMEMBERS its
     inputs (its "children") plus a closure `_backward` that knows the op's
     LOCAL derivative.
  2. Calling .backward() on the final scalar (the loss) topologically sorts the
     graph and runs each node's `_backward` in reverse order, so by the time a
     node fires, its own .grad (dLoss/dNode) is already fully accumulated.
  3. Each `_backward` implements exactly one instance of the chain rule:
         child.grad += (local derivative) * (this node's .grad)
     The `+=` is load-bearing: if a value is used in two places, gradients from
     both paths must SUM (multivariate chain rule).

Everything else - the MLP, the training loop, the gradient check - is built on
top of that one mechanism.

Run:  uv run python ml/03_neural_networks/micrograd_plus.py
Only dependency: numpy (for data generation and the finite-difference check).
"""

import math

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# The Value class: one scalar + its gradient + graph bookkeeping
# ----------------------------------------------------------------------------
class Value:
    """A scalar with autograd.

    Attributes:
      data:      the forward-pass value (a plain float)
      grad:      dLoss/dThis, filled in by .backward(); starts at 0
      _prev:     the Values this one was computed from (graph edges)
      _backward: closure that pushes this node's grad to its children
      _op:       op name, for debugging/printing only
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None  # leaves have nothing to propagate
        self._op = _op

    # ---- binary ops --------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # c = a + b  =>  dc/da = 1, dc/db = 1.
            # Addition is a "gradient router": whatever gradient arrives at c
            # flows UNCHANGED to both a and b. Intuition: nudging a by h nudges
            # c by exactly h.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # c = a * b  =>  dc/da = b, dc/db = a  ("swap and multiply").
            # Nudging a by h changes c by b*h, so a's sensitivity is b's
            # FORWARD value - this is why backward needs the forward pass
            # cached: local gradients are functions of forward values.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        # Only constant (non-Value) exponents: covers x**2, x**-1 (division).
        assert isinstance(k, (int, float)), "only supports int/float powers"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # c = a^k  =>  dc/da = k * a^(k-1). Plain calculus power rule.
            self.grad += (k * self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    # ---- unary nonlinearities ---------------------------------------------

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d/dz tanh(z) = 1 - tanh(z)^2. Note we reuse the OUTPUT t rather
            # than recomputing from the input - cheaper, and it is why
            # frameworks keep activations alive until backward completes.
            # Since t in (-1, 1), the factor 1 - t^2 is in (0, 1]: every tanh
            # SHRINKS the gradient, and near saturation (|t| ~ 1) it ~kills it.
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # ReLU is a gradient GATE: derivative is 1 where the unit was
            # active, 0 where it was off. Gradient passes through untouched or
            # not at all - no shrinking, which is why deep ReLU nets train.
            # (At exactly z=0 we choose subgradient 0; measure-zero event.)
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # d/dz e^z = e^z - exp is its own derivative, so the local grad is
            # simply the forward OUTPUT again.
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, "log needs a positive input"
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d/dz log(z) = 1/z. Note the blow-up as z -> 0+: this is exactly
            # what gives cross-entropy its huge gradient on confidently-wrong
            # predictions (and why naive log(softmax) is numerically fragile).
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---- convenience: everything below reduces to +, *, ** ----------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1  # a/b = a * b^-1: division is free once we have pow

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ---- the backward pass -------------------------------------------------

    def backward(self):
        """Run backprop from this node (typically the loss).

        Why topological order matters: a node's _backward multiplies by
        `out.grad`, so out.grad must ALREADY hold the full dLoss/dOut,
        i.e. every path from the loss down to `out` must have been processed
        first. Reverse topological order guarantees exactly that, and it also
        means each node/edge is visited ONCE - this single-visit property is
        the dynamic-programming reuse that makes backprop O(graph) instead of
        O(paths) (paths can be exponential in depth).
        """
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)  # post-order: children always precede parents

        build(self)

        self.grad = 1.0  # seed: dLoss/dLoss = 1
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


# ----------------------------------------------------------------------------
# A tiny neural net library on top of Value
# ----------------------------------------------------------------------------
class Neuron:
    """One neuron: w . x + b, optionally squashed by tanh.

    Init scale: weights ~ U[-1,1] scaled by 1/sqrt(fan_in). This is the
    variance argument from the notes: Var(w.x) = n_in * Var(w) * Var(x), so
    Var(w) must scale like 1/n_in or activations grow/shrink with layer width.
    """

    def __init__(self, n_in, nonlin=True):
        scale = 1.0 / math.sqrt(n_in)
        self.w = [Value(np.random.uniform(-1, 1) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # sum() with a Value start builds the graph: act = w1*x1 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-layer perceptron; last layer is linear (it emits a score/logit  - 
    squashing belongs to the loss, not the architecture)."""

    def __init__(self, n_in, layer_sizes):
        sizes = [n_in] + layer_sizes
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(layer_sizes) - 1))
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ----------------------------------------------------------------------------
# Part 1: gradient check - trust nothing, verify against finite differences
# ----------------------------------------------------------------------------
def gradient_check():
    """Build a random gnarly scalar expression using EVERY op, backprop it,
    and compare each leaf's .grad to a centered finite difference
        (f(x+h) - f(x-h)) / (2h)
    Centered differences have O(h^2) truncation error (the odd terms of the
    Taylor expansion cancel), vs O(h) for one-sided - that's why we use them.
    If the relative error is ~1e-8-ish in float64, the engine is correct.
    """
    print("=" * 70)
    print("PART 1: gradient check of the Value engine vs finite differences")
    print("=" * 70)

    def expr(a, b, c, d):
        # Deliberately uses every op (+ - * / ** tanh relu exp log) and reuses
        # variables on multiple paths, so the += accumulation is exercised too.
        e = (a * b + c).tanh()
        f = (a - d).relu() + (b * c).exp() * 0.1
        g = (e * f + a / d) ** 2
        h = (f + 2.0).log() + g
        return h * c + e

    leaf_vals = {"a": 0.7, "b": -0.4, "c": 1.3, "d": 2.1}

    # Analytic gradients via our engine
    leaves = {k: Value(v) for k, v in leaf_vals.items()}
    out = expr(**leaves)
    out.backward()

    # Numeric gradients via centered differences
    h = 1e-6
    print(f"{'leaf':>4} | {'analytic (backprop)':>20} | {'numeric (fin diff)':>20} | {'rel err':>10}")
    max_rel = 0.0
    for name in leaf_vals:
        vals_p = dict(leaf_vals)
        vals_m = dict(leaf_vals)
        vals_p[name] += h
        vals_m[name] -= h
        fp = expr(**{k: Value(v) for k, v in vals_p.items()}).data
        fm = expr(**{k: Value(v) for k, v in vals_m.items()}).data
        numeric = (fp - fm) / (2 * h)
        analytic = leaves[name].grad
        rel = abs(analytic - numeric) / (abs(analytic) + abs(numeric) + 1e-12)
        max_rel = max(max_rel, rel)
        print(f"{name:>4} | {analytic:>20.10f} | {numeric:>20.10f} | {rel:>10.2e}")

    assert max_rel < 1e-6, f"gradient check FAILED (max rel err {max_rel:.2e})"
    print(f"max relative error: {max_rel:.2e}  -> PASS (engine gradients are correct)\n")


# ----------------------------------------------------------------------------
# Part 2: train an MLP on a tiny 2-class dataset
# ----------------------------------------------------------------------------
def make_moons_ish(n_per_class=30):
    """Two interleaving arcs - linearly INSEPARABLE on purpose, so the tanh
    hidden layers are actually doing something (a linear model cannot solve it).
    Labels are +1 / -1 to pair with the max-margin-style hinge-ish loss below."""
    t = np.random.uniform(0, math.pi, n_per_class)
    x0 = np.stack([np.cos(t), np.sin(t)], axis=1) + np.random.randn(n_per_class, 2) * 0.1
    x1 = np.stack([1 - np.cos(t), -np.sin(t) + 0.3], axis=1) + np.random.randn(n_per_class, 2) * 0.1
    X = np.concatenate([x0, x1])
    y = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    return X, y


def train_mlp():
    print("=" * 70)
    print("PART 2: train a 2-16-16-1 MLP on a tiny 2-class 'two moons' dataset")
    print("=" * 70)

    X, y = make_moons_ish(30)
    model = MLP(2, [16, 16, 1])
    print(f"dataset: {len(X)} points, model: {len(model.parameters())} scalar parameters")

    n_steps = 300
    for step in range(n_steps + 1):
        # Simple linear learning-rate decay: big exploratory steps early,
        # small settling steps late (see the schedules section of notes.md).
        lr = 0.5 * (1.0 - 0.9 * step / n_steps)
        # ---- forward: build one big graph over the whole dataset ----
        scores = [model([Value(xi) for xi in xrow]) for xrow in X]

        # Squared hinge loss: max(0, 1 - y*score)^2 via relu - pushes each
        # score to the correct side of the margin, smooth where active.
        losses = [(1.0 + (-yi) * si).relu() ** 2 for si, yi in zip(scores, y)]
        data_loss = sum(losses, Value(0.0)) * (1.0 / len(losses))
        # small L2: sum w^2 (regularization; also exercises ** in the graph)
        reg_loss = sum((p * p for p in model.parameters()), Value(0.0)) * 1e-4
        loss = data_loss + reg_loss

        # ---- backward ----
        # Zeroing grads is essential: .backward() ACCUMULATES (+=) by design,
        # so stale gradients from the previous step would silently add in.
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # ---- SGD update: step AGAINST the gradient ----
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 30 == 0:
            acc = np.mean([(s.data > 0) == (yi > 0) for s, yi in zip(scores, y)])
            print(f"step {step:3d} | lr {lr:.4f} | loss {loss.data:.4f} | accuracy {acc*100:5.1f}%")

    final_acc = np.mean([(model([Value(xi) for xi in xrow]).data > 0) == (yi > 0)
                         for xrow, yi in zip(X, y)])
    print(f"final training accuracy: {final_acc*100:.1f}%")
    assert final_acc >= 0.95, "training failed to fit the tiny dataset"
    print("-> the scalar engine trains a real (tiny) neural net. No magic anywhere.\n")


if __name__ == "__main__":
    gradient_check()
    train_mlp()
