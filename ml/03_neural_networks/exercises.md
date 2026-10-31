# Neural Networks: Exercises

Twelve exercises, easy → hard. Do them on paper first; where code is asked for,
extend the files in this directory. Solutions are collapsed below each problem;
try each problem before opening its solution.

---

## Exercise 1 (easy): Linear collapse, concretely

Let $W_1 = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$, $b_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$,
$W_2 = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}$, $b_2 = \begin{pmatrix} 0 \\ 3 \end{pmatrix}$.
Find the single layer $(W', b')$ such that $W_2(W_1 x + b_1) + b_2 = W'x + b'$ for
all $x$, and verify on $x = (1, 1)^\top$.

<details><summary>Solution</summary>

$W' = W_2 W_1 = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 2 & 4 \end{pmatrix}$,
$\quad b' = W_2 b_1 + b_2 = \begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 0 \\ 3 \end{pmatrix} = \begin{pmatrix} 1 \\ 5 \end{pmatrix}$.

Check with $x = (1,1)^\top$: two-layer path gives $W_1x + b_1 = (4, 1)^\top$,
then $W_2 (4,1)^\top + b_2 = (3, 11)^\top$. Collapsed: $W'x + b' = (1{+}1{+}1,\; 2{+}4{+}5)^\top = (3, 11)^\top$. ✓

Moral: no nonlinearity ⇒ any depth collapses to one affine map. This is why
activations are not optional.
</details>

---

## Exercise 2 (easy): Hand-backprop a tiny graph

For $a = 2$, $b = -3$, $c = 10$: let $d = a \cdot b$, $e = d + c$, $L = e^2$.
Compute the forward values and then $\partial L/\partial a$, $\partial L/\partial b$,
$\partial L/\partial c$ by hand, node by node (micrograd style).

<details><summary>Solution</summary>

Forward: $d = -6$, $e = 4$, $L = 16$.

Backward (seed $\text{grad}(L) = 1$):
- $L = e^2 \Rightarrow \text{grad}(e) = 2e = 8$
- $e = d + c$: addition routes gradients unchanged ⇒ $\text{grad}(d) = 8$, $\text{grad}(c) = 8$
- $d = ab$: swap-and-multiply ⇒ $\text{grad}(a) = b \cdot 8 = -24$, $\text{grad}(b) = a \cdot 8 = 16$

Sanity check on $a$: $L = (ab + c)^2$, so $\partial L/\partial a = 2(ab+c)\,b = 2 \cdot 4 \cdot (-3) = -24$. ✓
</details>

---

## Exercise 3 (easy): The `+=` bug

A student writes an autograd engine but uses `child.grad = local * out.grad`
(assignment) instead of `+=`. Construct the *smallest* expression where this
gives a wrong gradient, and compute both the wrong and correct answers.

<details><summary>Solution</summary>

Take $L = a \cdot a$ (or $a + a$): any expression where one Value is used twice.
With $a = 3$: correct is $\partial L/\partial a = 2a = 6$. The multiply node runs
its backward for the left operand (`a.grad = a.data * 1 = 3`) and then for the
right operand; with `=` it *overwrites*: `a.grad = 3`. Wrong by a factor 2.

With `+=`, the two path contributions $3 + 3 = 6$ accumulate. The multivariate
chain rule says gradients over multiple paths **sum**:
$\frac{dL}{da} = \frac{\partial L}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial L}{\partial v}\frac{\partial v}{\partial a}$
where $u, v$ are the two uses of $a$. This is also why every training loop must
**zero gradients** each step; accumulation is by design.
</details>

---

## Exercise 4 (easy-medium): Sigmoid's max derivative, and depth

(i) Prove $\max_z \sigma'(z) = 1/4$. (ii) A 15-layer network uses sigmoid
activations everywhere; give an upper bound on the factor the activation
derivatives alone contribute to $\partial L/\partial h_0$, and interpret.

<details><summary>Solution</summary>

(i) $\sigma' = \sigma(1-\sigma)$. Let $s = \sigma(z) \in (0,1)$ and maximize
$g(s) = s - s^2$: $g'(s) = 1 - 2s = 0 \Rightarrow s = 1/2$ (max since $g'' = -2 < 0$),
so $g_{\max} = 1/4$, attained at $z = 0$.

(ii) The backward product contains one $\mathrm{diag}(\sigma'(z_k))$ per layer,
each entry $\le 1/4$, so the activation contribution is at most
$(1/4)^{15} \approx 9.3 \times 10^{-10}$. Unless the weight matrices amplify by
$\sim 4\times$ per layer (which would cause other problems, such as exploding
activations), gradients at the input end are about a billion times smaller than at the
output end, so the early layers receive near-zero updates. This is the vanishing gradient problem;
`training_dynamics.ipynb` EXPT 2 measures it.
</details>

---

## Exercise 5 (medium): Softmax shift invariance and log-sum-exp

(i) Prove $\mathrm{softmax}(z + c\mathbb{1}) = \mathrm{softmax}(z)$ for any scalar $c$.
(ii) For $z = (1000, 1001, 1002)$, naive softmax overflows in float64. Compute
the actual probabilities with the log-sum-exp trick, by hand.

<details><summary>Solution</summary>

(i) $\frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^c e^{z_i}}{e^c \sum_j e^{z_j}} = p_i$; the $e^c$ cancels.

(ii) Subtract $m = 1002$: shifted logits $(-2, -1, 0)$, exponentials
$(e^{-2}, e^{-1}, 1) \approx (0.1353, 0.3679, 1)$, sum $\approx 1.5032$, so
$p \approx (0.0900, 0.2447, 0.6652)$. No number ever exceeded 1, yet these are
the *exact* probabilities of the original logits (by part i). Naively,
$e^{1000}$ overflows ($e^{709.78}$ is the float64 limit) giving `inf/inf = nan`.
Bonus: $\log p_i = z_i - m - \log(1.5032)$ avoids computing `log(softmax)` where
underflow to 0 would give `-inf`.
</details>

---

## Exercise 6 (medium): Why dropout scaling preserves expectation (and what it does to variance)

Inverted dropout keeps a unit with probability $q = 1 - p$ and scales survivors
by $1/q$: $\tilde a = \frac{m}{q} a$ with $m \sim \mathrm{Bernoulli}(q)$.
(i) Show $E[\tilde a] = a$. (ii) Compute $\mathrm{Var}(\tilde a)$ (treating $a$
fixed). (iii) Why do we want the expectation preserved?

<details><summary>Solution</summary>

(i) $E[\tilde a] = \frac{a}{q} E[m] = \frac{a}{q} \cdot q = a$. ✓

(ii) $E[\tilde a^2] = \frac{a^2}{q^2}E[m^2] = \frac{a^2}{q^2} \cdot q = \frac{a^2}{q}$
(since $m^2 = m$ for Bernoulli). So
$\mathrm{Var}(\tilde a) = \frac{a^2}{q} - a^2 = a^2 \frac{1-q}{q} = a^2\frac{p}{1-p}$.
Dropout preserves the mean but **injects variance**; this injected noise is what
regularizes.

(iii) At test time dropout is turned off, so each downstream unit sees exactly
$a$. If training-time expectation weren't also $a$, every downstream layer would
be calibrated to inputs $\frac{1}{q}\times$ different from what it receives at
inference: a systematic train/test mismatch in activation scale at every layer.
Matching in expectation makes test-time "identity" a sensible approximation of
the $2^n$-subnetwork ensemble average.
</details>

---

## Exercise 7 (medium): Derive Adam's bias correction

Adam's first moment is $m_t = \beta m_{t-1} + (1-\beta) g_t$ with $m_0 = 0$.
(i) Unroll to a closed form. (ii) Assuming $E[g_i] = g$ for all $i$, show
$E[m_t] = g(1 - \beta^t)$. (iii) Explain numerically why the correction matters
most for the second moment ($\beta_2 = 0.999$) at $t = 1$.

<details><summary>Solution</summary>

(i) $m_t = (1-\beta)\sum_{i=1}^{t} \beta^{\,t-i} g_i$ (each step's gradient is
weighted by $\beta$ once per subsequent step; the $(1-\beta)$ enters at
insertion).

(ii) $E[m_t] = (1-\beta) g \sum_{i=1}^{t}\beta^{t-i} = (1-\beta) g \cdot \frac{1 - \beta^{t}}{1-\beta} = g(1-\beta^t)$
using the finite geometric series $\sum_{k=0}^{t-1}\beta^k = \frac{1-\beta^t}{1-\beta}$.
Since $\beta^t \ne 0$ for small $t$, the raw EMA underestimates $g$; dividing by
$(1-\beta^t)$ makes it unbiased. As $t \to \infty$ the correction $\to 1$ (no-op).

(iii) At $t=1$: $v_1 = (1-\beta_2)g_1^2 = 0.001\,g_1^2$, so
$\sqrt{v_1} \approx 0.0316\,|g_1|$. The uncorrected update
$\eta\,m_1/\sqrt{v_1} \approx \eta \cdot 0.1|g_1| / (0.0316|g_1|) \approx 3.2\,\eta$, and with corrected $m$ but uncorrected $v$ it would be $\approx 31.6\,\eta$,
about 30× the intended step, applied on the first update. Bias-correcting both gives $\hat m_1/\sqrt{\hat v_1} = g_1/|g_1| = \pm 1$,
a well-scaled step.
</details>

---

## Exercise 8 (medium): Kaiming's factor of 2

Let $z \sim$ symmetric around 0 with variance $\sigma^2$, and $a = \max(0, z)$.
(i) Show $E[a^2] = \sigma^2/2$. (ii) Use this to derive He initialization
$\mathrm{Var}(w) = 2/n_{in}$ for ReLU networks. (iii) Predict what happens to
activation scale in a 30-layer ReLU net initialized with Xavier
$\mathrm{Var}(w) = 1/n_{in}$ (then check against EXPT 1 of `training_dynamics.ipynb`).

<details><summary>Solution</summary>

(i) $E[a^2] = E[z^2 \mathbb{1}[z>0]]$. By symmetry of $z$'s density,
$E[z^2\mathbb{1}[z>0]] = E[z^2\mathbb{1}[z<0]]$ and they sum to $E[z^2] = \sigma^2$
(mean 0), so each equals $\sigma^2/2$.

(ii) For the next layer, $z' = \sum_{j=1}^{n_{in}} w_j a_j$ with independent
mean-0 weights: $\mathrm{Var}(z') = n_{in}\,\mathrm{Var}(w)\,E[a^2] = n_{in}\,\mathrm{Var}(w)\,\sigma^2/2$.
(Note we use $E[a^2]$, not $\mathrm{Var}(a)$: with mean-0 weights it's the
second moment of the input that propagates.) Requiring
$\mathrm{Var}(z') = \sigma^2$: $n_{in}\mathrm{Var}(w)/2 = 1 \Rightarrow \mathrm{Var}(w) = 2/n_{in}$.

(iii) Xavier leaves a per-layer factor of $1/2$ on the second moment, i.e. the
std shrinks by $1/\sqrt{2}$ per layer: after 30 layers,
$(1/\sqrt 2)^{30} = 2^{-15} \approx 3 \times 10^{-5}$, so activations effectively
vanish. EXPT 1 shows this (L30 std ≈ $1.8 \times 10^{-5}$ for relu+Xavier,
≈ 0.59 for relu+Kaiming).
</details>

---

## Exercise 9 (medium-hard): Momentum's effective learning rate

With update $v_t = \beta v_{t-1} + g$, $\theta_t = \theta_{t-1} - \eta v_t$, and
a *constant* gradient $g$: (i) find $\lim_{t\to\infty} v_t$; (ii) conclude the
effective learning rate; (iii) for a gradient that alternates $+g, -g, +g, \dots$
find the steady-state oscillation of $v_t$ and compare magnitudes. What does
this mean geometrically in a ravine-shaped loss?

<details><summary>Solution</summary>

(i) Unrolling: $v_t = g(1 + \beta + \dots + \beta^{t-1}) \to \frac{g}{1-\beta}$
(geometric series).

(ii) Steady-state step: $\eta v_\infty = \frac{\eta}{1-\beta} g$: momentum
multiplies the effective LR by $\frac{1}{1-\beta}$ ($10\times$ at $\beta = 0.9$)
along *persistent* gradient directions.

(iii) For alternating $\pm g$, look for a period-2 steady state $v^+ , v^-$ with
$v^+ = \beta v^- + g$ and $v^- = \beta v^+ - g$. Substituting:
$v^+ = \beta(\beta v^+ - g) + g \Rightarrow v^+ = \frac{g(1-\beta)}{1-\beta^2} = \frac{g}{1+\beta}$.
So oscillating components are *attenuated* to $\frac{1}{1+\beta} \approx 0.53\times$
(at $\beta = 0.9$) while persistent components are amplified $10\times$: an
$\sim 19\times$ ratio. Geometrically: in a ravine, the across-the-valley gradient
alternates sign (damped), the along-the-valley gradient is consistent
(amplified), so momentum reduces the zig-zag across the valley and takes larger steps
along it.
</details>

---

## Exercise 10 (hard): Derive the BatchNorm backward pass

Forward, per feature, over a batch of size $B$:
$\mu = \frac{1}{B}\sum_i x_i$, $\sigma^2 = \frac{1}{B}\sum_i (x_i - \mu)^2$,
$\hat x_i = (x_i - \mu)/\sqrt{\sigma^2 + \epsilon}$, $y_i = \gamma \hat x_i + \beta$.
Given upstream $g_i = \partial L/\partial y_i$, derive
$\partial L/\partial \gamma$, $\partial L/\partial \beta$, and the full
$\partial L/\partial x_i$ (the hard one: $x_i$ affects $y_i$ directly *and*
through $\mu$ and $\sigma^2$).

<details><summary>Solution</summary>

Easy ones: $\frac{\partial L}{\partial \gamma} = \sum_i g_i \hat x_i$, 
$\frac{\partial L}{\partial \beta} = \sum_i g_i$, and define
$\frac{\partial L}{\partial \hat x_i} = g_i \gamma =: \hat g_i$.

Write $s = \sqrt{\sigma^2 + \epsilon}$. Three paths from $x_i$ to the loss:

**Path via $\sigma^2$:** since $\hat x_j = (x_j - \mu)(\sigma^2+\epsilon)^{-1/2}$,
$\frac{\partial \hat x_j}{\partial \sigma^2} = -\frac{1}{2}(x_j - \mu)\,s^{-3}$, so
$$\frac{\partial L}{\partial \sigma^2} = -\frac{1}{2s^3}\sum_j \hat g_j (x_j-\mu) = -\frac{1}{2s^2}\sum_j \hat g_j \hat x_j \qquad (\text{using } x_j - \mu = s\,\hat x_j).$$

**Path via $\mu$:** $\frac{\partial \hat x_j}{\partial \mu} = -\frac{1}{s}$, and
$\sigma^2$ also depends on $\mu$ with $\frac{\partial \sigma^2}{\partial \mu} = -\frac{2}{B}\sum_j (x_j - \mu) = 0$
at the batch mean (the deviations sum to zero), so
$$\frac{\partial L}{\partial \mu} = -\frac{1}{s}\sum_j \hat g_j.$$

**Direct path:** $\frac{\partial \hat x_i}{\partial x_i}\big|_{\mu,\sigma^2\text{ fixed}} = \frac{1}{s}$.

Combine with $\frac{\partial \mu}{\partial x_i} = \frac{1}{B}$ and
$\frac{\partial \sigma^2}{\partial x_i} = \frac{2(x_i - \mu)}{B}$:

$$\frac{\partial L}{\partial x_i} = \frac{\hat g_i}{s} + \frac{\partial L}{\partial \sigma^2}\cdot\frac{2(x_i-\mu)}{B} + \frac{\partial L}{\partial \mu}\cdot\frac{1}{B}
= \frac{1}{B s}\left(B\,\hat g_i - \sum_j \hat g_j - \hat x_i \sum_j \hat g_j \hat x_j\right)$$

Interpretation of the three terms: pass the gradient through ($B\hat g_i$),
subtract the batch-mean of gradients (because shifting all $x$ doesn't change
$\hat x$), and subtract the component correlated with $\hat x$ (because
rescaling doesn't change $\hat x$ either). So BN's backward removes from the
gradient its batch mean and its component along $\hat x$; this is one proposed reason
BN stabilizes training. Verify numerically with a finite-difference check
(recommended).
</details>

---

## Exercise 11 (hard): Residual identity path, quantitatively

Consider a deep chain $h_{k+1} = h_k + F_k(h_k)$ where each block's Jacobian is
$J_k^F = \partial F_k/\partial h_k$ with $\|J_k^F\| \le \delta < 1$.
(i) Write $\partial h_d / \partial h_0$ and show it equals $I + E$ where
$\|E\| \le (1+\delta)^d - 1$ and, to first order, $E \approx \sum_k J_k^F$.
(ii) Show that a *plain* chain $h_{k+1} = F_k(h_k)$ with the same bound gives
$\|\partial h_d/\partial h_0\| \le \delta^d$. (iii) Take $\delta = 0.9$, $d = 50$:
compare the guaranteed gradient scale in both architectures.

<details><summary>Solution</summary>

(i) By the chain rule, $\frac{\partial h_d}{\partial h_0} = \prod_{k=0}^{d-1}(I + J_k^F)$.
Expanding the product: $I + \sum_k J_k^F + \sum_{k<l} J_l^F J_k^F + \cdots = I + E$.
Bounding: $\|\prod (I + J_k^F)\| \le (1+\delta)^d$, and $E$ is everything except
$I$, so $\|E\| \le (1+\delta)^d - 1$; for small $\delta$, $E \approx \sum_k J_k^F$.
The point: the **identity term survives no matter how small $\delta$ is**; the
skip path transmits gradient with factor exactly 1. (The bound also shows the
other risk: $(1+\delta)^d$ is large when $\delta$ is not small, so the
correction terms can make the gradient explode. Deep residual nets scale the
blocks down at init to prevent this.)

(ii) Plain chain: $\frac{\partial h_d}{\partial h_0} = \prod_k J_k^F$, and
submultiplicativity gives $\|\prod_k J_k^F\| \le \delta^d$.

(iii) $\delta = 0.9$, $d = 50$: plain chain gradient $\le 0.9^{50} \approx 5.2\times 10^{-3}$, and with $\delta = 0.5$ it's $0.5^{50} \approx 10^{-15}$, effectively zero. The residual
chain *always* delivers the $I$ component, so the gradient reaching layer 0
never vanishes, whatever $\delta$ is. That $+I$ term in the Jacobian is the main reason
residual networks of 50 to 1000 layers train, and why every transformer block is
`x = x + f(norm(x))`.
</details>

---

## Exercise 12 (hard): Implement a new op in the Value engine

Add a `sigmoid()` op to the `Value` class in `micrograd_plus.ipynb`:
(i) derive the backward rule, expressing it in terms of the forward **output**;
(ii) write the code; (iii) explain why `x.sigmoid()` computed as a primitive is
numerically/computationally preferable to composing it as `1 / (1 + (-x).exp())`
even though both give correct gradients; (iv) verify with a finite-difference
check.

<details><summary>Solution</summary>

(i) $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (derived in notes.md, section 4).
Expressed via the output $s = \sigma(z)$: local gradient is simply $s(1-s)$; no
need to keep the input or recompute any `exp`.

(ii)

```python
def sigmoid(self):
    # numerically stable in both tails: never exponentiate a large positive number
    if self.data >= 0:
        s = 1.0 / (1.0 + math.exp(-self.data))
    else:
        e = math.exp(self.data)
        s = e / (1.0 + e)
    out = Value(s, (self,), "sigmoid")

    def _backward():
        # d(sigma)/dz = s * (1 - s), reusing the forward output s
        self.grad += s * (1.0 - s) * out.grad

    out._backward = _backward
    return out
```

(iii) Three reasons. **Stability**: the composed version computes
`exp(-x)`, which overflows for large negative `x` (e.g. $x = -800$), while the
two-branch primitive never exponentiates a positive number. **Graph size**: in this
engine the composition creates 5 op nodes (multiply by $-1$, exp, add, power $-1$,
multiply), each with its own closure and stored value; the primitive creates 1. Fused
ops in real frameworks apply the same idea (as does fused softmax-cross-entropy).
**Cheaper backward**: one multiply using the cached output vs. chaining five local
gradients.

(iv) Check (add to `gradient_check()` or run standalone):

```python
x = Value(0.7); y = x.sigmoid() * 3.0; y.backward()
h = 1e-6
num = ((Value(0.7 + h).sigmoid() * 3.0).data
     - (Value(0.7 - h).sigmoid() * 3.0).data) / (2 * h)
assert abs(x.grad - num) / (abs(x.grad) + abs(num)) < 1e-8
```

Analytically: $\sigma(0.7) \approx 0.66819$, so
$x.\text{grad} = 3\,\sigma'(0.7) = 3 \cdot 0.66819 \cdot 0.33181 \approx 0.66514$.
The finite difference should agree to about 8 digits.
</details>
