# Neural Networks from First Principles

> Style note: every formula is derived, backprop is worked by hand once on
> actual numbers, and each design choice (init, normalization, Adam, residuals)
> is motivated by a concrete failure. The failures of bad initialization,
> saturating activations and badly chosen learning rates are reproduced
> numerically in `training_dynamics.ipynb`. The model for this style is
> Karpathy's "Zero to Hero" series, in written form.

Companion code in this directory:

- `micrograd_plus.ipynb`: scalar autograd engine + MLP + training loop + gradient check
- `mlp_numpy.ipynb`: vectorized 2-layer MLP, manual backward, SGD vs momentum vs Adam
- `training_dynamics.ipynb`: numeric experiments: init variance, vanishing gradients, learning rates

---

## 1. What a neuron computes, and why nonlinearity is the entire point

A single "neuron" takes a vector $x \in \mathbb{R}^n$,
owns a weight vector $w \in \mathbb{R}^n$ and a bias $b$, and computes

$$a = \sigma(w^\top x + b)$$

That is: a **dot product** (weighted sum of the inputs), a **shift** ($b$ moves
the decision threshold), and a **squashing nonlinearity** $\sigma$. The pre-activation
$z = w^\top x + b$ is a linear function of $x$; geometrically, $z = 0$ defines a
hyperplane, $w$ is its normal vector, and $z$ measures signed distance to that plane
(scaled by $\|w\|$). So a neuron outputs a smooth function of which side of this
hyperplane $x$ lies on, and how far from it. A layer of $m$ neurons is just $m$ of these stacked into a matrix:

$$z = Wx + b, \qquad W \in \mathbb{R}^{m \times n}$$

### Why the nonlinearity is not optional

Suppose we stack layers **without** nonlinearities:

$$h_1 = W_1 x + b_1, \qquad h_2 = W_2 h_1 + b_2$$

Substitute the first into the second (this is the whole proof):

$$h_2 = W_2 (W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{W'} x + \underbrace{(W_2 b_1 + b_2)}_{b'}$$

$$\Rightarrow\; h_2 = W' x + b'$$

A stack of any number of linear layers **collapses to a single linear layer**, because a
composition of affine maps is an affine map. A 100-layer linear "deep network" has
exactly the expressive power of logistic/linear regression: it can only carve the input
space with a single hyperplane per output. It cannot represent XOR, a circle, a spiral,
anything with curvature in its decision boundary.

(Note the collapse can even *lose* rank: if $W_1$ is $\mathbb{R}^{n \to 2}$ and $W_2$
is $\mathbb{R}^{2 \to m}$, then $W'=W_2W_1$ has rank $\le 2$. Depth without
nonlinearity adds no expressive power and can remove some.)

The nonlinearity $\sigma$ breaks the collapse: $W_2\,\sigma(W_1 x)$ does **not**
simplify, because $\sigma$ is not a matrix. What you gain: with ReLU, each hidden
neuron is linear on each side of its hyperplane, so the network is a piecewise-linear
function, and the number of linear regions can grow exponentially with depth.
Two separate facts follow. The universal approximation theorem is about *width*: one
hidden layer with enough neurons can approximate any continuous function on a bounded
domain. Depth buys *efficiency*: composing layers multiplies the number of linear
regions, so some functions need exponentially fewer neurons in a deep network than in
a shallow one.

**One-sentence summary you should be able to say in an interview:** a neuron is
`dot product + bias + nonlinearity`; the nonlinearity is the entire point, because
composition of linear maps is linear ($W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2)$).

---

## 2. Backpropagation = chain rule + bookkeeping + reuse

### The chain rule, stated the useful way

If $L$ depends on $z$, and $z$ depends on $x$, then

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$

In words: *the gradient flowing into a node equals the gradient flowing out of it,
times the node's own local derivative.* Every backward rule in every autograd
engine is one instance of this.

### The computational-graph view

Any forward computation is a DAG: leaves are inputs/parameters, internal nodes are
primitive ops (`+`, `*`, `tanh`, ...), and there is one scalar output, the loss $L$.
The backward step at each node needs two quantities:

1. **Local gradients**: $\partial(\text{out})/\partial(\text{each input})$; e.g. for
   $c = a \cdot b$: $\partial c/\partial a = b$ and $\partial c/\partial b = a$.
2. **The upstream gradient** $\partial L / \partial(\text{out})$, already computed
   earlier in the backward pass.

The node multiplies them and adds each product to the corresponding input's gradient. If a value feeds **two**
downstream nodes, the gradients arriving from both paths **add** (multivariate chain
rule: $\frac{\partial L}{\partial a} = \sum_{\text{paths } p} \frac{\partial L}{\partial p}\frac{\partial p}{\partial a}$).
That's why every autograd engine does `a.grad += ...`, never `a.grad = ...`.

### Why this is dynamic programming

Naively, $\partial L/\partial w$ for each parameter $w$ sums over *every path* from
$w$ to $L$, and their number grows exponentially with depth. Backprop visits nodes in **reverse
topological order**, and by the time it reaches a node, that node's
$\partial L/\partial(\text{out})$ already aggregates *all* downstream paths into one
cached number. Each edge is processed once. That's memoization: backprop computes all
$N$ parameter gradients for roughly the cost of **two** extra forward passes, instead of
$N$ forward passes (the cost of finite differences; so finite differences are fine for
gradient *checking* but too expensive for gradient *computation*).

### A tiny 2-layer network, by hand, on actual numbers

Network: 1 input $\to$ 1 hidden tanh neuron $\to$ 1 linear output, squared-error loss.

$$z_1 = w_1 x + b_1, \quad h = \tanh(z_1), \quad \hat{y} = w_2 h + b_2, \quad L = (\hat{y} - y)^2$$

Numbers: $x = 1.0$, target $y = 1.0$, and parameters
$w_1 = 0.5,\; b_1 = 0.1,\; w_2 = -0.3,\; b_2 = 0.2$.

**Forward pass** (compute and *write down* every intermediate; they are reused):

| node | expression | value |
|---|---|---|
| $z_1$ | $0.5 \cdot 1.0 + 0.1$ | $0.6$ |
| $h$ | $\tanh(0.6)$ | $0.537050$ |
| $\hat{y}$ | $-0.3 \cdot 0.537050 + 0.2$ | $0.038885$ |
| $d$ | $\hat{y} - y$ | $-0.961115$ |
| $L$ | $d^2$ | $0.923742$ |

**Backward pass**, node by node, exactly like micrograd. Notation: `grad(v)` means
$\partial L/\partial v$. Seed: `grad(L) = 1`.

1. $L = d^2 \Rightarrow \text{grad}(d) = 2d \cdot \text{grad}(L) = 2(-0.961115) = -1.922230$
2. $d = \hat{y} - y \Rightarrow \text{grad}(\hat{y}) = 1 \cdot \text{grad}(d) = -1.922230$
3. $\hat{y} = w_2 h + b_2$, three fan-ins, three local gradients:
   - $\text{grad}(w_2) = h \cdot \text{grad}(\hat{y}) = 0.537050 \cdot (-1.922230) = -1.032333$
   - $\text{grad}(h) = w_2 \cdot \text{grad}(\hat{y}) = (-0.3)(-1.922230) = 0.576669$
   - $\text{grad}(b_2) = 1 \cdot \text{grad}(\hat{y}) = -1.922230$
4. $h = \tanh(z_1)$, local gradient $1 - h^2 = 1 - 0.537050^2 = 0.711578$:
   - $\text{grad}(z_1) = 0.711578 \cdot 0.576669 = 0.410345$
5. $z_1 = w_1 x + b_1$:
   - $\text{grad}(w_1) = x \cdot \text{grad}(z_1) = 1.0 \cdot 0.410345 = 0.410345$
   - $\text{grad}(b_1) = 0.410345$
   - $\text{grad}(x) = w_1 \cdot \text{grad}(z_1) = 0.205172$ (not needed for training, but backprop gives it for free)

Sanity checks you should habitually run in your head:
- **Signs**: $\hat{y} \approx 0.04$ is far below target $1$, so we want $\hat{y}$ to
  increase. $\text{grad}(b_2) < 0$, and SGD moves $b_2 \leftarrow b_2 - \eta\,\text{grad}(b_2)$,
  i.e. $b_2$ increases. Correct.
- **Reuse**: $\text{grad}(\hat{y})$ was computed once and used three times in step 3.
  $h$ from the *forward* pass was reused in the backward pass (both as a local gradient
  for $w_2$ and inside $1-h^2$). This is why frameworks store activations: backward
  needs forward's intermediates, and it is why activations are often the largest
  memory cost of training, on top of the parameters, gradients and optimizer state.

One gradient-descent step with $\eta = 0.1$: $w_2 \leftarrow -0.3 - 0.1(-1.032333) = -0.196767$,
etc. Re-run forward: $L$ drops from $0.9237$ to about $0.49$. Training is this loop
repeated many times on larger graphs.

### The three backward rules that generate everything

- **Add** $c = a + b$: local grads are $1,1$ → gradient is **distributed** unchanged to both inputs (a "gradient router").
- **Mul** $c = a b$: local grads are $b, a$ → gradient to each input is scaled by *the other* input's forward value ("swap and multiply").
- **Any unary $f$**: gradient scaled by $f'$ evaluated at the forward value.

Matrix versions ($Z = XW$, upstream $G = \partial L/\partial Z$:
$\partial L/\partial X = G W^\top$, $\partial L/\partial W = X^\top G$) are these rules
batched. Element-wise, $Z_{nc} = \sum_h X_{nh}W_{hc}$, so the mul rule plus summing over
every output an entry touches gives
$\partial L/\partial W_{hc} = \sum_n X_{nh}G_{nc} = (X^\top G)_{hc}$ and
$\partial L/\partial X_{nh} = \sum_c G_{nc}W_{hc} = (GW^\top)_{nh}$. A useful mnemonic:
shapes must match, and there's exactly one transpose arrangement that makes them match.

---

## 3. Losses: softmax + cross-entropy, and why $\nabla = p - y$

### Setup

For $C$-way classification, the network emits **logits** $z \in \mathbb{R}^C$
(unnormalized scores). Softmax converts them to a probability distribution:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Cross-entropy against the true class $y$ (one-hot vector $y$, true index $t$):

$$L = -\sum_i y_i \log p_i = -\log p_t$$

i.e. "negative log-probability assigned to the correct class"; minimizing it is
maximum likelihood. $L = 0$ iff $p_t = 1$, and the loss of a confidently wrong
prediction is unbounded (as $p_t \to 0$, $L \to \infty$). The gradient on the correct
logit, $p_t - 1$ (derived next), stays near its largest magnitude $1$ while the
prediction is badly wrong.

### The full derivation of $\partial L/\partial z = p - y$

Work with $L = -z_t + \log\sum_j e^{z_j}$ (just expanded $-\log p_t$). Differentiate
w.r.t. an arbitrary logit $z_k$:

$$\frac{\partial L}{\partial z_k} = -\frac{\partial z_t}{\partial z_k} + \frac{\partial}{\partial z_k}\log\sum_j e^{z_j}
= -\mathbb{1}[k = t] + \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k - y_k$$

That's the entire derivation once you write $L$ in log-sum-exp form. If you instead
grind through softmax's Jacobian, you need the two cases

$$\frac{\partial p_i}{\partial z_k} = p_i(\mathbb{1}[i=k] - p_k)$$

(derive via quotient rule: for $i = k$, $\frac{e^{z_k}S - e^{z_k}e^{z_k}}{S^2} = p_k(1-p_k)$;
for $i \ne k$, $\frac{0 - e^{z_i}e^{z_k}}{S^2} = -p_i p_k$, with $S = \sum_j e^{z_j}$),
then chain: $\frac{\partial L}{\partial z_k} = -\frac{1}{p_t}\frac{\partial p_t}{\partial z_k}
= -\frac{1}{p_t} p_t(\mathbb{1}[t=k] - p_k) = p_k - y_k$. Same answer, more work.

**Why the form $p - y$ matters:** it's bounded in $[-1, 1]$ per class, it's zero exactly
when the prediction is perfect, and the "error signal" is literally
*predicted minus true probability*. The same form appears for sigmoid + binary
cross-entropy and for linear + MSE; all are canonical link/loss pairings from
exponential-family maximum likelihood, and the $\sigma'$ term that would cause
saturation cancels against the loss's derivative. This is why you **never** pair
softmax with MSE: the cancellation doesn't happen, gradients carry a $p(1-p)$ factor,
and a confidently-wrong network ($p_t \approx 0$) gets a gradient near zero exactly
when its error is largest.

### Numerical stability: the log-sum-exp trick

Softmax needs $e^{z_i}$. In float32, $e^{89} = \infty$; so logits around $100$
(easy to hit mid-training) give `inf/inf = nan`, and every later value is `nan`. The fix exploits
softmax's **shift invariance**: for any constant $m$,

$$\frac{e^{z_i - m}}{\sum_j e^{z_j - m}} = \frac{e^{z_i}e^{-m}}{e^{-m}\sum_j e^{z_j}} = p_i$$

Choose $m = \max_j z_j$. Then every exponent is $\le 0$, so $e^{z_i - m} \in (0, 1]$
(no overflow), and at least one term equals $1$, so the denominator is $\ge 1$ (it
cannot underflow to $0$). Similarly compute the *log*-probability directly instead of
`log(softmax(z))`:

$$\log p_i = z_i - m - \log\sum_j e^{z_j - m}$$

because `log(tiny)` where `tiny` underflowed to `0.0` gives `-inf`. Framework
`cross_entropy` functions fuse softmax, log and the negative log-likelihood (NLL) into
one op for this reason (see `stable_logsumexp` in `mlp_numpy.ipynb`).

---

## 4. Activation functions and saturation

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

(Derive $\sigma'$: write $\sigma = (1+e^{-z})^{-1}$, so
$\sigma' = -(1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}
= \sigma \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(1-\sigma)$.)

**Maximum of $\sigma'$:** let $s = \sigma(z) \in (0,1)$ and maximize $g(s) = s(1-s)$.
$g'(s) = 1 - 2s = 0 \Rightarrow s = \tfrac12$, i.e. $z = 0$, giving
$g_{\max} = \tfrac12 \cdot \tfrac12 = \boxed{0.25}$.

So **every** sigmoid in the backward path multiplies the gradient by *at most* $0.25$, and typically far less: at $|z| = 5$, $\sigma' \approx 0.0066$. In a depth-$d$ stack
of sigmoid layers, the activation-path factor is bounded by $0.25^d$ times weight
terms: $0.25^{10} \approx 10^{-6}$, $0.25^{20} \approx 10^{-12}$. The gradient that
reaches layer 1 of a deep sigmoid net is close to zero. That is the **vanishing
gradient problem**. EXPT 2 in `training_dynamics.ipynb` prints the per-layer mean
absolute gradient of a 20-layer sigmoid net; it shrinks by a roughly constant factor
(about $0.23$) per layer.

Two further drawbacks of sigmoid: outputs are **not zero-centered** (always positive,
so the next layer's weight gradients $\delta \cdot x_j$, where
$\delta = \partial L/\partial z$ is the gradient at that neuron's pre-activation, all
share the sign of $\delta$; in one step a neuron's weights all increase or all decrease,
which produces zig-zag optimization paths), and `exp` is relatively expensive.

### Tanh

$$\tanh(z) = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z)$$

Zero-centered (removes the first drawback) and $\tanh'(0) = 1$ (much better than $0.25$), but it
still **saturates**: $|z|$ large $\Rightarrow \tanh' \to 0$. Fine for shallow nets and
gates (LSTMs), but the gradient still vanishes with depth.

### ReLU and dead neurons

$$\mathrm{ReLU}(z) = \max(0, z), \qquad \mathrm{ReLU}'(z) = \mathbb{1}[z > 0]$$

The derivative is exactly $1$ on the active side, so repeated multiplication by it
does not shrink the gradient; this is a major reason deep nets became trainable.
Also: no `exp`, and the induced sparsity (~half the units at 0) is cheap.

The failure mode: a **dead neuron**. If a neuron's pre-activation is negative for
*every* input in the dataset (e.g. after a large gradient step pushed its bias to a
large negative value, or from bad init), then its output is always $0$, its local
gradient is always $0$, so its weights receive zero gradient and never change again.
The unit is permanently unused. Mitigations: sensible init + moderate learning rates,
LeakyReLU ($\max(\alpha z, z)$, $\alpha \approx 0.01$, slope $\alpha$ for $z<0$),
or smooth variants below. (Note ReLU is non-differentiable at exactly $z=0$; we just
pick a subgradient, conventionally $0$; for continuous inputs this has probability zero.)

### GELU / SiLU (briefly)

$$\mathrm{GELU}(z) = z \cdot \Phi(z) \approx z\,\sigma(1.702 z), \qquad \mathrm{SiLU/Swish}(z) = z\,\sigma(z)$$

Both are "ReLU-shaped but smooth": they gate the input by a soft probability of being
positive instead of a hard threshold, are slightly negative for small negative $z$
(no hard zero → no truly dead units), and empirically train slightly better in
transformers: GELU is used in BERT and GPT-2/3 MLP blocks, and SiLU in gated form
(SwiGLU) in LLaMA-family models. There is no strong theory for the gain; treat them as
smoothed ReLUs that did slightly better in many ablations.

**Rule of thumb:** hidden layers → ReLU family (GELU in transformers); output layer →
whatever the loss demands (linear for regression, logits into softmax-CE for
classification). Sigmoid's $(0,1)$ output range is not a reason to use it in hidden
layers.

---

## 5. Initialization: keeping variance constant across depth

### The problem, from first principles

Consider one linear layer $z = Wx$ (ignore bias, initialized 0), with $n_{in}$ inputs,
$W_{ij}$ i.i.d. mean-0 with variance $\sigma_w^2$, inputs $x_j$ i.i.d. mean-0 with
variance $\mathrm{Var}(x)$, weights independent of inputs. One output coordinate is
$z_i = \sum_{j=1}^{n_{in}} W_{ij} x_j$. Variance of a sum of independent mean-0 terms
adds, and $\mathrm{Var}(W_{ij}x_j) = \mathrm{Var}(W_{ij})\mathrm{Var}(x_j)$ for
independent mean-0 factors (since $E[(Wx)^2]=E[W^2]E[x^2]$), so:

$$\mathrm{Var}(z_i) = \sum_{j=1}^{n_{in}} \mathrm{Var}(W_{ij})\,\mathrm{Var}(x_j) = n_{in}\,\sigma_w^2\,\mathrm{Var}(x)$$

**Every layer multiplies the activation variance by $g = n_{in}\sigma_w^2$.** After
$d$ layers, variance is scaled by $g^d$. Three regimes:

- $g < 1$: activations shrink geometrically → after ~20 layers everything is
  $\approx 0$, all neurons output values near $0$, and gradients (which contain
  activation factors) also vanish.
- $g > 1$: activations explode geometrically → tanh-like outputs sit at $\pm 1$,
  where their derivatives are near 0, so gradients vanish even though activations are
  large; with ReLU, values overflow to `inf/nan`.
- $g = 1$: variance is preserved through depth. This is the **only** regime in which
  a deep net at initialization has activations and gradients of usable size at every
  layer.

Requiring $\mathrm{Var}(z) = \mathrm{Var}(x)$, i.e. $g = 1$:

$$n_{in}\,\sigma_w^2 = 1 \;\Rightarrow\; \sigma_w^2 = \frac{1}{n_{in}}$$

### Xavier/Glorot init

The same argument run on the **backward** pass ($\partial L/\partial x = W^\top \delta$,
where $\delta = \partial L/\partial z$ is the gradient arriving from the layer above,
a sum over $n_{out}$ terms) demands $\sigma_w^2 = 1/n_{out}$. You can't satisfy both
unless $n_{in} = n_{out}$, so Glorot & Bengio use the harmonic mean of the two
requirements $1/n_{in}$ and $1/n_{out}$:

$$\boxed{\sigma_w^2 = \frac{2}{n_{in} + n_{out}}} \qquad \text{(Xavier; for tanh-like activations, } \approx \text{identity near 0)}$$

Equivalently, uniform on $[-\sqrt{6/(n_{in}+n_{out})}, +\sqrt{6/(n_{in}+n_{out})}]$
(since $\mathrm{Var}(U[-a,a]) = a^2/3$).

### Kaiming/He init: accounting for ReLU

Xavier assumed the nonlinearity is roughly identity around 0. ReLU is not: it **zeroes
half the distribution**. For $z$ symmetric mean-0, $a = \max(0, z)$ has

$$E[a^2] = E[z^2 \cdot \mathbb{1}[z>0]] = \tfrac12 E[z^2] = \tfrac12 \mathrm{Var}(z)$$

(by symmetry of $z$'s distribution about 0). So each ReLU halves the second moment,
and to keep $g = 1$ the weights must compensate with an extra factor of 2:

$$n_{in} \cdot \sigma_w^2 \cdot \tfrac12 = 1 \;\Rightarrow\; \boxed{\sigma_w^2 = \frac{2}{n_{in}}} \qquad \text{(Kaiming/He, for ReLU)}$$

Use Xavier init with ReLU in a 30-layer net and the missing $2^{d/2}$ factor shrinks
activations by $\sim (1/\sqrt2)^{30} \approx 3\times10^{-5}$; EXPT 1 in
`training_dynamics.ipynb` shows this numerically. Biases: init to 0 (they don't feed the variance
argument). Never init weights to 0 or any constant: all neurons in a layer then
compute identical outputs and receive identical gradients forever. This is the
**symmetry problem**; random init exists to break symmetry, and the *scale* of that randomness is
what Xavier/Kaiming pin down.

---

## 6. Optimization

### SGD and why minibatches

Full-batch gradient descent: $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$
where $L$ averages over all $N$ examples. For $N$ in the millions this is one very
expensive, very accurate step. Minibatch SGD instead uses $B \ll N$ random examples:
the minibatch gradient is an **unbiased estimator** of the full gradient
($E[\hat{g}] = g$) with variance $\propto 1/B$. Why it wins:

1. **Compute**: the standard error of the gradient estimate falls like $1/\sqrt{B}$
   while cost grows like $B$, so returns diminish quickly, and many cheap noisy steps
   usually make more progress than a few exact ones.
2. **Hardware**: matrix-matrix ops at batch size 32–1024 keep a GPU's arithmetic units
   busy; at batch size 1 most of them sit idle.
3. **Noise as a feature**: gradient noise helps escape saddle points and acts as
   implicit regularization; small/medium batches often generalize better than full
   batch. A popular explanation is that noise steers training toward flatter minima,
   but "flat generalizes better" is debated: reparameterizing a ReLU net can make a
   minimum sharp without changing its function (Dinh et al., 2017).

### Momentum: EMA of gradients, and the effective step

$$v_t = \beta v_{t-1} + g_t, \qquad \theta \leftarrow \theta - \eta\, v_t \qquad (\beta \approx 0.9)$$

Unroll the recursion: $v_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots$, an
exponentially-weighted sum of past gradients (Polyak's "heavy-ball" method; the
exponential-moving-average (EMA) variant $v_t = \beta v_{t-1} + (1-\beta)g_t$ is the same thing with
$\eta$ rescaled). **Effective step size**: if the gradient were constant at $g$, the
velocity converges to the geometric-series limit

$$v_\infty = g\,(1 + \beta + \beta^2 + \cdots) = \frac{g}{1 - \beta}$$

so with $\beta = 0.9$ momentum takes steps up to $\frac{1}{1-\beta} = 10\times$ larger
than plain SGD **along directions where gradients consistently agree**, while
directions where the gradient oscillates (+,−,+,−, e.g. across a narrow ravine in the loss surface)
cancel in the sum. Net effect: larger steps along the consistent direction, smaller
oscillation across it.

### RMSProp: per-parameter learning rates

Keep an EMA of the **squared** gradient and divide by its root:

$$s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t$$

Parameters with persistently large gradients get their steps shrunk; parameters with
tiny, rare gradients get boosted. When a gradient's sign is consistent, the update is
approximately $\eta \cdot \mathrm{sign}(g)$, making one global $\eta$ work across
parameters whose gradient scales differ by orders of magnitude (embeddings vs. biases).

### Adam = momentum + RMSProp + bias correction

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
\theta \leftarrow \theta - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta \sim 10^{-3}$.

**Deriving the bias correction** (a common interview question). $m_0 = 0$, so unrolling:

$$m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{\,t-i} g_i$$

Take expectations assuming $E[g_i] \approx g$ (stationary gradient):

$$E[m_t] = (1-\beta_1)\, g \sum_{i=1}^{t} \beta_1^{\,t-i} = (1-\beta_1)\, g \cdot \frac{1-\beta_1^t}{1-\beta_1} = g\,(1 - \beta_1^t)$$

So the raw EMA is biased toward zero by exactly the factor $(1-\beta_1^t)$, because
it starts at 0 and its weights $(1-\beta_1)\beta_1^{t-i}$ sum to only $1-\beta_1^t$. Dividing by
$1-\beta_1^t$ makes it unbiased: $E[\hat m_t] = g$. Same for $v_t$ with $\beta_2$.
Why it *matters*: with $\beta_2 = 0.999$, after one step $v_1 = 0.001\,g_1^2$, so
uncorrected $\sqrt{v_1} \approx 0.0316|g_1|$. If $m$ is corrected but $v$ is not,
the update $g_1/\sqrt{v_1}$ is about $32\times$ too large. If neither is corrected,
$m_1 = 0.1\,g_1$ and the update is $0.1/0.0316 \approx 3.2\times$ too large. Either
way the first steps are mis-scaled, and correcting both gives a step of exactly
$\pm 1$ (times $\eta$). As $t \to \infty$,
$\beta^t \to 0$ and the correction fades to a no-op; it only matters for early steps.

### Learning-rate schedules

$\eta$ is usually the most important hyperparameter to tune (EXPT 3 in
`training_dynamics.ipynb` shows slow progress, convergence and divergence for three
values). Common schedules:

- **Step decay**: drop $\eta$ by 10× at fixed milestones.
- **Cosine**: $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1 + \cos(\pi t/T))$: smooth decay from $\eta_{\max}$ to $\eta_{\min}$ over $T$ steps; a common default.
- **Warmup**: linearly ramp $\eta$ from 0 over the first few hundred/thousand steps.
  Rationale: even after bias correction, $\hat v$ early on averages only a few
  squared gradients, so it is a noisy scale estimate; early gradients are also large
  and atypical, and the loss surface near initialization is sharp, so a full-size
  step can diverge. Warmup keeps steps small until these settle. Standard for
  transformers.

Big picture: a large $\eta$ early gives fast progress, with the iterates fluctuating
over a wide region; a small $\eta$ late lets them settle into a minimum.

### Weight decay vs. L2, and AdamW

For **plain SGD** the two are identical: L2 loss $\frac{\lambda}{2}\|\theta\|^2$ adds
$\lambda\theta$ to the gradient, giving $\theta \leftarrow (1 - \eta\lambda)\theta - \eta g$, which is exactly "decay the weights, then step." For **Adam they differ**: L2's $\lambda\theta$
enters $m_t$ and $v_t$ and gets *divided by* $\sqrt{\hat v}$, so parameters with large
gradient history get *less* regularization, which is not what weight decay is meant to
do and gives worse results empirically (Loshchilov & Hutter, 2019).
**AdamW decouples**: keep $\lambda\theta$ out of the moments and apply decay directly
in the update,

$$\theta \leftarrow \theta - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta\right)$$

Every parameter is decayed at the same rate regardless of its gradient statistics.
This is why "Adam + L2" and "AdamW" are different optimizers, and why AdamW is the
default for transformers.

---

## 7. Normalization

### BatchNorm: the forward math

For each feature (channel) independently, over a minibatch of size $B$:

$$\mu = \frac{1}{B}\sum_{i=1}^{B} x_i, \qquad \sigma^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma\,\hat{x}_i + \beta$$

Standardize to mean 0 / variance 1 using **batch statistics**, then let learned
per-feature $\gamma, \beta$ set the output's scale and shift to whatever values
training finds best. So BN does not restrict what the layer can represent; it *reparameterizes* it,
making the mean/scale of a layer's activations (now directly set by $\beta, \gamma$)
independent of the combined effect of all upstream weights.

### The two stories of why it works

**Story 1 (the original paper): internal covariate shift.** As layer $k$'s weights
update, the *distribution* of layer $k{+}1$'s inputs shifts, so each layer's weights
are fit to an input distribution that changes after every update; BN fixes the mean
and variance of each layer's inputs. This explanation is widely repeated but not
supported by later experiments: Santurkar et al. (2018) injected random noise with
changing mean and variance *after* BN, reintroducing covariate shift, and the network
still trained about as well as with plain BN.

**Story 2 (the more recent story): loss-surface smoothing.** The same paper measured
that, at points along the training path, the loss and its gradient change less when
moving along the gradient direction in BN networks (a smaller empirical Lipschitz constant), and proved bounds of this kind for
one BN layer under simplifying assumptions. A smoother loss permits **larger learning
rates** without divergence and makes training less sensitive to init. One concrete
mechanism: BN makes the layer's output invariant to the weight scale,
$f(aW) = f(W)$ for $a>0$ (where $f$ is the loss as a function of the weights feeding
BN). Differentiating gives $\nabla f(aW) = \nabla f(W)/a$, so if $W$ grows, its
gradient shrinks and the effective step size falls automatically. Which mechanism
matters most is still debated. Interview-safe phrasing: "the covariate-shift story is
the historical motivation; the evidence favors BN smoothing the optimization
landscape, enabling higher LR and reducing init sensitivity, though the full
explanation is still open."

### Why BN needs running statistics at inference

At inference you may get **one** example. A "batch" mean over one sample is the
sample itself, so $\hat x = 0$ for every input and the output carries no information
about it; with larger batches, predictions would depend on whatever else happened to be
in the batch (non-deterministic, and one request could affect another's output). So during training BN maintains EMAs

$$\mu_{run} \leftarrow (1-\alpha)\,\mu_{run} + \alpha\,\mu_{batch}, \qquad \sigma^2_{run} \leftarrow (1-\alpha)\,\sigma^2_{run} + \alpha\,\sigma^2_{batch}$$

and at eval time uses these **frozen** population estimates, making BN a fixed affine
transform per feature. This train/eval difference is the source of the classic
"model.eval() bug": if eval mode is not set, inference uses batch statistics and gives
batch-dependent outputs without raising any error.

### LayerNorm, and why transformers use it

LayerNorm normalizes across the **feature dimension of a single example**:

$$\mu_i = \frac{1}{D}\sum_{d} x_{i,d}, \qquad \sigma_i^2 = \frac{1}{D}\sum_{d} (x_{i,d} - \mu_i)^2, \qquad y_{i,d} = \gamma_d\,\frac{x_{i,d} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_d$$

Same example-in, example-out statistics at train and test: **no batch dependence, no
running stats, no train/eval mode**. Transformers use LN instead of BN because:

1. **Variable-length sequences + padding** make per-position batch statistics ill-defined.
2. **Batch sizes are small / sharded** in large-model training; BN stats get noisy and
   require cross-device sync (expensive all-reduces).
3. **Autoregressive decoding** processes one token at a time, so there is no batch.
4. Normalizing each token separately matches the transformer's per-token
   computation: each position's residual-stream vector (the running sum
   $x \leftarrow x + f(x)$ carried through the blocks) is processed by the MLP on its
   own. Modern LLMs use the pre-norm placement, often with RMSNorm (LayerNorm without
   the mean subtraction; cheaper, and works about as well).

---

## 8. Regularization

### Dropout

Training: each hidden unit is independently zeroed with probability $p$; surviving
units are scaled by $\frac{1}{1-p}$ (**inverted dropout**). Inference: identity, no-op.

**Why the scaling:** it preserves the expectation. Let $m \sim \mathrm{Bernoulli}(1-p)$ and
$\tilde{a} = \frac{m \cdot a}{1-p}$. Then

$$E[\tilde{a}] = \frac{E[m]\,a}{1-p} = \frac{(1-p)\,a}{1-p} = a$$

so the *expected* input each downstream unit sees is identical at train and test time. Without the scaling, test-time activations would be systematically $\frac{1}{1-p}$×
larger than what the network trained on, so every downstream layer would receive inputs
on a different scale than in training.
(Original paper scaled by $1-p$ at *test* time instead; inverted dropout is the same
idea, moved to training so inference stays a clean no-op.)

**Ensemble interpretation**: a net with $n$ droppable units defines $2^n$ subnetworks
sharing weights; each training step samples one and trains it. Test time approximates
the geometric-mean ensemble of all $2^n$ members with a single forward pass. Practical
effect: a unit's output cannot depend on any specific other unit being present, since
that unit is zeroed with probability $p$, so each feature must be useful on its own,
which gives redundant representations.

### Early stopping

Monitor validation loss; stop (or restore the checkpoint from) when it stops
improving, even as training loss keeps falling. It's regularization because it limits
how far $\theta$ travels from init; for gradient descent on linear least squares it
provably behaves like L2 regularization with strength $\sim 1/(\eta t)$. It costs
nothing extra and also serves as model selection, so it is used almost always, with a
`patience` parameter (evaluations without improvement before stopping) to tolerate
noisy validation curves.

### Data augmentation

Expand the training distribution with label-preserving transforms: crops, flips,
color jitter, cutout/mixup for images; noise/time-shift for audio; back-translation
for text. It injects **invariances you know** (a flipped cat is a cat) directly as
data instead of requiring the net to learn them from limited examples. It is often
the most effective regularizer because it adds information (known invariances) that
the training set does not contain. Caution: transforms must respect the label semantics (don't horizontally flip
digits: a flipped "3" is not a "3").

---

## 9. Vanishing/exploding gradients, and residual connections

### The depth problem, stated with Jacobians

Backprop through $d$ layers multiplies $d$ Jacobians:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_d}\; \prod_{k=1}^{d} J_k, \qquad J_k = \frac{\partial h_k}{\partial h_{k-1}} = \mathrm{diag}(\sigma'(z_k))\, W_k$$

If the typical factor magnitude (roughly $\sigma'_{typ} \cdot \|W\|$) is $c$, the
product scales like $c^{\,d}$: $c<1 \Rightarrow$ gradients **vanish** (early layers
get near-zero updates, so only the layers near the output train); $c>1 \Rightarrow$
gradients **explode** (loss spikes and `nan`s; gradient clipping mitigates this by
rescaling $g$ to norm equal to the threshold when $\|g\| >$ threshold). Sigmoid's $\sigma' \le 0.25$ makes vanishing near-certain
(section 4); nothing keeps $c$ pinned at exactly 1 through training even with perfect
init (section 5). The effect is geometric in $d$, so it grows with depth.

### Residual connections: an identity path for the gradient

He et al.'s fix: make each block compute a **perturbation of identity**:

$$h_{k+1} = h_k + F(h_k)$$

Differentiate. The Jacobian now contains a **$+\,I$** term:

$$\frac{\partial h_{k+1}}{\partial h_k} = I + \frac{\partial F}{\partial h_k}$$

Chain through $d$ blocks:

$$\frac{\partial h_d}{\partial h_0} = \prod_{k=0}^{d-1}\left(I + \frac{\partial F_k}{\partial h_k}\right) = I + \sum_k \frac{\partial F_k}{\partial h_k} + (\text{higher-order products})$$

Even if every $\partial F_k/\partial h_k$ is tiny, the product contains the pure-$I$
term: **gradient reaches layer 0 with factor exactly 1 along the skip path**, plus
corrections. The corrections are not automatically harmless: if each block Jacobian
$J_k = \partial F_k/\partial h_k$ has norm up to $\delta$, the product of $(I + J_k)$
can grow like $(1+\delta)^d$ when the blocks are not small. That is why deep residual
nets scale each block's output down at init (for example by $1/\sqrt{d}$, or by
zero-initializing the last layer of each block, as in Fixup and DeepNorm). The identity
term is not multiplied by any block Jacobian, so no single block can make it vanish. In
a plain net the gradient *must* pass through every $J_k$, and one saturated layer
($J_k \approx 0$) makes the gradient near zero for every layer below it. Equivalently,
in the forward direction: the net starts as approximately the identity function (if
$F \approx 0$ at init, which good init gives you) and learns residual corrections, an
easier starting point than learning an arbitrary deep map from scratch. This one-line
change ($+\,h_k$) is what made 50/100/1000-layer nets trainable, and it's why *every*
transformer block wraps attention and MLP in `x = x + f(norm(x))`: residual connections
plus per-block normalization is the standard recipe, with gradient clipping as a guard
against explosion.

---

## Common interview questions

**Q1. Why do neural networks need nonlinear activation functions?**
Take two scalar linear layers, $h = 2x + 1$ and $y = 3h - 4$. Substituting gives
$y = 6x - 1$, a single linear layer. The same substitution works for matrices,
$W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$, so any stack of linear layers
equals one linear layer and draws one hyperplane per output. One hyperplane cannot
solve XOR: no line puts $(0,0)$ and $(1,1)$ on one side and $(0,1)$ and $(1,0)$ on the
other. A nonlinearity between the layers prevents the collapse, because
$W_2\,\mathrm{ReLU}(W_1x)$ is not a fixed matrix times $x$; with ReLU each hidden unit
folds the input space along its hyperplane, and the output is a piecewise-linear
function with many linear regions.

**If asked:** *Then why use depth instead of one wide layer?* The universal approximation theorem says one hidden layer with enough units approximates any continuous function on a bounded domain; depth makes this cheaper, because composing folds multiplies the number of linear regions, so some functions need exponentially fewer units in a deep network.

**Q2. Explain backpropagation in one minute.**
In the chapter's example, $L = (\hat y - y)^2$ with $\hat y = 0.0389$ and $y = 1$, so
$\partial L/\partial\hat y = 2(\hat y - y) = -1.922$. Since $\hat y = w_2 h + b_2$ with
$h = 0.537$ and $w_2 = -0.3$, one step back gives
$\partial L/\partial w_2 = h \cdot (-1.922) = -1.032$ and
$\partial L/\partial h = w_2 \cdot (-1.922) = 0.577$. The general procedure: write the
computation as a graph of primitive operations, run the forward pass and store every
node's value, then visit nodes in reverse topological order, multiplying each node's
incoming gradient by its local derivative and adding the result to its inputs'
gradients. The gradients add because a value used by two downstream nodes receives the
sum over both paths (multivariate chain rule). Each edge is processed once, so the
backward pass costs a small constant multiple of the forward pass (about 2x for a
matrix multiply: one product for the input gradient, one for the weight gradient).

**If asked:** *Why does training need so much memory?* The backward pass reuses forward values (here $h$ appears in $\partial L/\partial w_2$ and in the tanh derivative $1 - h^2$), so every layer's activations are stored until the backward pass reaches that layer.

**Q3. Why is backprop efficient compared to computing each parameter's gradient separately?**
A model with $N = 10^6$ parameters needs $2 \times 10^6$ forward passes to get its
gradient by centered finite differences, two per parameter. Backprop gets all $10^6$
partial derivatives from one forward pass and one backward pass, roughly the cost of 3
forward passes. The reason is that the output is one scalar: reverse-mode
differentiation propagates $\partial L/\partial(\text{node})$ from the loss backward,
and when it reaches a node, that number already sums every path from the node to the
loss. Each parameter's gradient is then its local derivative times one stored number,
instead of a sum over paths whose count grows exponentially with depth. This reuse of
stored partial results is dynamic programming over the graph.

**If asked:** *When is forward mode better?* Forward mode costs one pass per input and reverse mode one pass per output, so forward mode is cheaper for functions with few inputs and many outputs.

**Q4. Derive the gradient of softmax + cross-entropy.**
Take logits $z = (2, 1, 0)$ with true class 1. Softmax gives
$p = (7.389, 2.718, 1)/11.107 = (0.665, 0.245, 0.090)$, and the gradient is
$p - y = (-0.335, 0.245, 0.090)$: a gradient step raises the true logit, lowers the
others, and the components sum to zero. To derive it, write the loss in log-sum-exp
form and differentiate with respect to logit $z_k$:

$$L = -\log p_t = -z_t + \log\sum_j e^{z_j} \;\Rightarrow\; \frac{\partial L}{\partial z_k} = -\mathbb{1}[k=t] + \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k - y_k$$

Each component lies in $[-1, 1]$, and all are zero exactly when $p = y$.

**If asked:** *Do you get the same result through the softmax Jacobian?* Yes: $\partial p_t/\partial z_k = p_t(\mathbb{1}[t=k] - p_k)$, and multiplying by $\partial L/\partial p_t = -1/p_t$ gives $p_k - y_k$.

**Q5. Why not use MSE for classification?**
Take a sigmoid output with target $y = 1$ and logit $z = -5$, so
$p = \sigma(-5) = 0.0067$, a confident wrong prediction. With MSE, $L = (p - y)^2$ and
$\partial L/\partial z = 2(p - y)\,p(1-p) = -0.0132$. With cross-entropy,
$L = -\log p$ and $\partial L/\partial z = p - y = -0.993$, about 75 times larger. In
general, MSE through a sigmoid or softmax keeps the factor $p(1-p)$ from the sigmoid
derivative, which goes to 0 when the model is confidently wrong, the case that needs
the largest correction; the $\log$ in cross-entropy cancels that factor. MSE on a
sigmoid output is also non-convex in the logit, while cross-entropy is convex in the
logit and is the negative log-likelihood of a Bernoulli or categorical model.

**Q6. What is the vanishing gradient problem and what causes it?**
The sigmoid derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ is at most $0.25$ (at
$z = 0$) and is $0.0066$ at $|z| = 5$. Backprop through 10 sigmoid layers multiplies
the gradient by 10 such factors, so the activation terms alone contribute at most
$0.25^{10} \approx 10^{-6}$, and the first layers receive almost no gradient. In
general the gradient at the first layer is a product of one Jacobian per layer,
$J_k = \mathrm{diag}(\sigma'(z_k))\,W_k$; if the typical scale of each factor is
$c < 1$, the product shrinks like $c^d$ with depth $d$. Causes: saturating
activations, a weight scale with $n_{in}\sigma_w^2 \ne 1$, and depth itself. Fixes:
ReLU-family activations (derivative 1 on the active side), Xavier or Kaiming init,
normalization layers, and residual connections.

**If asked:** *And exploding gradients?* The same product with $c > 1$ grows like $c^d$; gradient clipping (rescaling $g$ when $\|g\|$ exceeds a threshold) bounds the resulting step.

**Q7. What is a dead ReLU and how do you prevent it?**
Suppose a large update sets a ReLU unit's bias to $b = -10$ while $w^\top x$ stays
within $[-3, 3]$ on the training data. Then $z = w^\top x + b < 0$ for every input,
the output is always 0, and the local derivative $\mathbb{1}[z > 0]$ is always 0, so
$w$ and $b$ receive zero gradient on every example and never change again. That unit is
a dead ReLU. The usual causes are a learning rate high enough that one step moves the
bias far negative, and poor initialization. Prevent it with Kaiming init, a moderate
learning rate with warmup, or an activation with nonzero slope for $z < 0$, such as
LeakyReLU ($\max(0.01z, z)$) or GELU.

**If asked:** *How would you detect it?* For each unit, measure the fraction of examples in a large batch where its output is nonzero; units at 0% are dead, and a count that grows during training points to the learning rate.

**Q8. Derive Xavier and Kaiming initialization.**
In a 30-layer ReLU network of width 256, weights with variance $1/256$ halve the
activation variance at every layer, so the output std is
$(1/\sqrt2)^{30} = 2^{-15} \approx 3 \times 10^{-5}$ of the input std; variance
$2/256$ keeps it constant. The derivation: for $z = Wx$ with $n_{in}$ inputs, i.i.d.
mean-0 weights of variance $\sigma_w^2$, and independent mean-0 inputs, each $z_i$ is a
sum of $n_{in}$ independent terms, so

$$\mathrm{Var}(z_i) = n_{in}\,\sigma_w^2\,\mathrm{Var}(x)$$

Keeping the variance fixed requires $\sigma_w^2 = 1/n_{in}$, and the same argument on
the backward pass requires $1/n_{out}$; Xavier uses the compromise
$\sigma_w^2 = 2/(n_{in}+n_{out})$, which is $1/256$ in the example. ReLU sets the
negative half of a symmetric $z$ to zero, so $E[\max(0,z)^2] = \tfrac12\mathrm{Var}(z)$,
and Kaiming doubles the weight variance to $\sigma_w^2 = 2/n_{in}$.

**Q9. Why initialize randomly at all? Why not zeros?**
Take a hidden layer of two tanh units whose incoming weights all equal one constant
and whose outgoing weights all equal another. For every input both units compute the
same value, and each unit's weight gradient is (its outgoing weight) $\times$
$\tanh'(z)$ $\times$ (upstream gradient) $\times$ (input), so both receive the same
gradient and the same update. They stay identical at every later step, and the layer
behaves as one unit whatever its width. All zeros is the extreme case: $h = \tanh(0) = 0$
gives the outgoing weights zero gradient, and outgoing weights of 0 give the incoming
weights zero gradient, so only the output bias trains. Random init makes the units
differ from the start, and Xavier or Kaiming set the scale of that randomness so
activation and gradient variance are preserved through depth.

**If asked:** *Can biases start at zero?* Yes; the random weights already make the units differ, and biases do not enter the variance calculation.

**Q10. Explain Adam. Why the bias correction?**
With $\beta_2 = 0.999$, after one step $v_1 = 0.001\,g^2$, so
$\sqrt{v_1} \approx 0.032|g|$. With $m$ corrected ($\hat m_1 = g$) but $v$ not, the
first step is $g/(0.032|g|) \approx 32$ times the intended size of 1 (in units of
$\eta$); with neither corrected, $m_1 = 0.1\,g$ and the step is
$0.1/0.032 \approx 3.2$ times too large. Adam keeps an exponential moving average of
gradients, $m_t$ ($\beta_1 = 0.9$, the momentum part), and of squared gradients, $v_t$
($\beta_2 = 0.999$, the per-parameter scaling from RMSProp), and updates with

$$\theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}, \qquad \hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

Both averages start at 0, so for a stationary gradient $E[m_t] = g(1-\beta_1^t)$, and
dividing by $1-\beta_1^t$ removes that bias (the same holds for $v$ with $\beta_2$).
The divisor approaches 1 as $t$ grows, though for $v$ this takes a few thousand steps:
$1 - 0.999^{1000} \approx 0.63$.

**If asked:** *Why use warmup if bias correction exists?* In the first steps $\hat v$ averages only a few squared gradients, so it is a noisy scale estimate, and early gradients are large; warmup keeps the steps small until both settle.

**Q11. Weight decay vs. L2 regularization: same thing?**
Under plain SGD they are the same: adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss
adds $\lambda\theta$ to the gradient, and
$\theta - \eta(g + \lambda\theta) = (1-\eta\lambda)\theta - \eta g$, which is weight
decay. Under Adam they differ. Take $\lambda = 0.01$, two weights with $\theta = 1$,
and gradient histories with $\sqrt{\hat v} = 10$ for one and $0.1$ for the other: the
L2 term passes through the moments and is divided by $\sqrt{\hat v}$, so the decay per
step is about $0.001\eta$ for the first and $0.1\eta$ for the second, a factor of 100
apart, and the weights with large gradients are regularized least. AdamW applies the
decay outside the adaptive scaling,
$\theta \leftarrow \theta - \eta\big(\hat m_t/(\sqrt{\hat v_t}+\epsilon) + \lambda\theta\big)$,
so every weight decays by $\eta\lambda\theta$ per step regardless of its gradient
history. AdamW is the default optimizer for transformers.

**Q12. What does BatchNorm do, and why does it help?**
For one feature with batch values $(1, 2, 3, 6)$, BN computes mean 3 and variance 3.5
(std 1.87), standardizes to $(-1.07, -0.53, 0, 1.60)$, then outputs
$\gamma\hat x + \beta$ with learned per-feature $\gamma$ and $\beta$, so the layer can
still produce any mean and scale. The original paper attributed the benefit to reducing
internal covariate shift, the change in a layer's input distribution as earlier layers
update. Santurkar et al. (2018) found that adding random noise after BN, which restores
that shift, leaves the benefit intact, and measured that BN makes the loss and its
gradient smoother (smaller Lipschitz constants). That smoothness allows much larger
learning rates and reduces sensitivity to initialization; it is related to BN's scale
invariance, $\mathrm{BN}(aWx) = \mathrm{BN}(Wx)$ for $a > 0$. Batch statistics also add
noise that acts as a mild regularizer.

**Q13. Why does BatchNorm behave differently at train vs. test time?**
At inference with a batch of one example, each feature's batch mean equals the
example's own value, so $\hat x = 0$ and the layer outputs $\beta$ whatever the input;
with larger batches, an example's prediction would depend on which other examples
share its batch. So during training BN also keeps exponential moving averages of each
feature's mean and variance, $\mu_{run} \leftarrow (1-\alpha)\mu_{run} + \alpha\mu_{batch}$
(and likewise for $\sigma^2$), and at test time normalizes with these fixed values. BN
at inference is then a fixed affine map per feature, identical for every example. The
classic bug is running inference with the model still in training mode, which uses
batch statistics and gives batch-dependent outputs.

**If asked:** *What goes wrong with very small training batches?* With 2 to 4 examples per batch the batch statistics are noisy and differ from the running averages used at test time; GroupNorm, LayerNorm, or BN with statistics synchronized across devices avoids this.

**Q14. Why do transformers use LayerNorm instead of BatchNorm?**
Pad a batch of two sequences, of lengths 10 and 3, to length 10. At position 7 only one
sequence has a real token, so a per-position batch statistic comes from one value or
mixes in padding; during autoregressive decoding there is one token at a time and no
batch at all. LayerNorm computes the mean and variance over the $D$ features of each
token separately (the 768 numbers of one token in a model with $D = 768$), so the
result does not depend on other examples, the computation is identical in training and
inference, and there are no running statistics. It also avoids the cross-device
communication BN needs when a batch is split across GPUs, the usual setting for large
models. Modern LLMs typically use RMSNorm in the pre-norm position: it divides by the
root mean square of the features without subtracting the mean.

**If asked:** *What is pre-norm?* The normalization is applied to the input of each attention or MLP branch, $x + f(\mathrm{norm}(x))$, so the residual path itself is never normalized.

**Q15. How does dropout work, and why scale by $1/(1-p)$ during training?**
With drop probability $p = 0.5$ and an activation $a = 2$, training outputs 0 with
probability 0.5 and $2/(1-0.5) = 4$ with probability 0.5, so the expected output is 2,
equal to the test-time output with dropout off. Without the $1/(1-p)$ factor the
training expectation would be 1, and at test time downstream layers would receive
inputs twice as large as during training. In general each unit is zeroed with
probability $p$ and survivors are divided by $1-p$, so
$E[\tilde a] = (1-p)a/(1-p) = a$ and inference needs no change. Dropout regularizes
because no unit can rely on a specific other unit being present, and it can be viewed
as training $2^n$ weight-sharing subnetworks ($n$ droppable units) whose ensemble the
full network approximates at test time.

**If asked:** *Does the scaling match train and test exactly?* Only in mean; the variance still differs, which is why dropout placed before a BatchNorm layer leaves BN's running variance mismatched at test time (Li et al., 2019).

**Q16. Why do residual connections help train very deep networks?**
In a 50-layer plain network where each layer's Jacobian scales the gradient by 0.5,
the gradient at the first layer is multiplied by $0.5^{50} \approx 9 \times 10^{-16}$.
A residual block computes $h_{k+1} = h_k + F(h_k)$, so its Jacobian is $I + J_k$ with
$J_k = \partial F/\partial h_k$, and the product over blocks expands to
$I + \sum_k J_k + (\text{products of two or more } J_k)$. The $I$ term is a path along
which the gradient reaches the first layer unchanged; if every $\|J_k\| \le 0.01$, the
singular values of the 50-block product stay between $0.99^{50} \approx 0.61$ and
$1.01^{50} \approx 1.64$. This requires the branches to be small: with larger $J_k$ the
product can grow like $(1+\delta)^d$, which is why deep residual networks scale each
branch down at init (zero-initializing the branch's last layer, or multiplying the
branch by a depth-dependent factor such as $1/\sqrt{d}$; Fixup and DeepNorm are schemes
of this kind). In the forward direction, the network starts close to the identity map
and learns corrections to it.

**Q17. Your training loss is NaN after a few hundred steps. Debug it.**
If logits reach 100 in float32, $e^{100}$ overflows (the largest float32 is about
$3.4 \times 10^{38} \approx e^{88.7}$), and a naive softmax computes
$\infty/\infty$ = NaN. Check, roughly in this order: (1) a learning rate that is too
high, visible as loss rising for some steps before the NaN; lower it 10x or add warmup;
(2) a numerically unstable loss: use the fused cross-entropy that applies the
log-sum-exp trick (subtract the max logit) instead of taking the log of a softmax
output; (3) exploding gradients: log the global gradient norm, add clipping, check the
init scale; (4) bad data: NaN or inf in inputs, labels outside the class range;
(5) division by or log of a number near zero, such as a missing $\epsilon$ in a
normalization layer or in Adam, or fp16 overflow above 65504. To localize the fault,
log the loss, gradient norm, and per-layer activation statistics every step, and find
the first step and layer where a non-finite value appears.

**Q18. Why train with minibatches rather than the full dataset?**
Going from $B = 100$ to $B = 10{,}000$ multiplies the compute per step by 100 but
divides the gradient's standard error only by $\sqrt{100} = 10$. A minibatch gradient
is an unbiased estimate of the full gradient with variance proportional to $1/B$, so
its standard error falls like $1/\sqrt{B}$ while its cost grows like $B$; beyond a
moderate batch size, extra compute gives more progress as additional steps than as
more accurate steps. Batches of 32 to 1024 examples also keep the GPU's parallel units
busy, which batch size 1 does not. The gradient noise helps escape saddle points and
acts as an implicit regularizer; one explanation is a bias toward flat minima, but
whether flatness causes generalization is debated, since reparameterizing a ReLU
network can make a minimum sharp without changing the function it computes (Dinh et
al., 2017).

**Q19. How would you verify a hand-written backward pass?**
For $f(\theta) = \theta^3$ at $\theta = 2$ the true derivative is 12. With
$h = 10^{-3}$, the one-sided difference $(f(2.001) - f(2))/0.001 = 12.006$ has error
$6 \times 10^{-3}$, while the centered difference
$(f(2.001) - f(1.999))/0.002 = 12.000001$ has error $10^{-6}$: error $O(h)$ versus
$O(h^2)$. To check a backward pass, compute for each parameter the centered difference

$$g_n = \frac{L(\theta_i + h) - L(\theta_i - h)}{2h}, \qquad h \approx 10^{-5}$$

and compare it to the analytic gradient $g_a$ with the relative error
$|g_a - g_n|/(|g_a| + |g_n| + \epsilon)$, which should be below about $10^{-6}$ in
float64. Use a tiny model and batch so the $2N$ forward passes are affordable, and
avoid points where a ReLU input lies within $h$ of 0, where the difference straddles
the kink.

**If asked:** *Why relative rather than absolute error?* Gradients span many orders of magnitude, so an absolute error of $10^{-6}$ is negligible for a gradient of 10 and as large as the gradient itself for a gradient of $10^{-6}$.

**Q20. Momentum: what does it actually do to the update?**
With $\beta = 0.9$ and a constant gradient $g = 1$, the velocity
$v_t = \beta v_{t-1} + g_t$ goes $1, 1.9, 2.71, \dots$ toward 10; with a gradient
alternating $+1, -1$, it settles to $\pm 1/(1+\beta) \approx \pm 0.53$. Unrolled,
$v_t = \sum_i \beta^{t-i} g_i$, an exponentially weighted sum of past gradients, and
for a constant gradient

$$v_\infty = g\,(1 + \beta + \beta^2 + \cdots) = \frac{g}{1-\beta}$$

So along directions where gradients agree, the step is up to 10 times the plain SGD
step at $\beta = 0.9$, while components that change sign each step (across a narrow
valley) partly cancel. The result is faster progress along the valley floor and less
zig-zag across it.

**If asked:** *What about the form $v_t = \beta v_{t-1} + (1-\beta)g_t$?* It is the same method with $\eta$ multiplied by $1-\beta$; the constant-gradient limit becomes $g$ instead of $g/(1-\beta)$.

## Exit checklist

- [ ] Cold, on paper: derive the softmax + cross-entropy gradient to p − y (5 min)
- [ ] Backprop a 2-layer MLP by hand, writing every forward/backward shape and the three backward rules (6 min)
- [ ] Derive Xavier and Kaiming init variances, including where ReLU's factor of 2 comes from (4 min)
- [ ] Write Adam's update equations with bias correction, and state AdamW vs L2 weight decay (4 min)
- [ ] BatchNorm vs LayerNorm: forward math, why BN needs running stats at inference, why transformers use LN (4 min)
- [ ] Explain vanishing/exploding gradients via products of Jacobians, and how residual connections fix it (3 min)
- [ ] Dead ReLUs: cause, how to detect, two fixes; plus the log-sum-exp stability trick (3 min)

Check every box from memory before moving on; add any miss to practice/queue.tsv.
