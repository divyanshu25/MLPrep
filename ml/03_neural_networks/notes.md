# Neural Networks from First Principles

> Style note: every formula is derived, every gradient is
> computed by hand at least once on actual numbers, and every design choice
> (init, normalization, Adam, residuals) is motivated by a concrete failure you
> can reproduce in `training_dynamics.ipynb`. If a claim seems hand-wavy, it's a
> bug - the goal is Karpathy's "Zero to Hero" in written form.

Companion code in this directory:

- `micrograd_plus.ipynb` - scalar autograd engine + MLP + training loop + gradient check
- `mlp_numpy.ipynb` - vectorized 2-layer MLP, manual backward, SGD vs momentum vs Adam
- `training_dynamics.ipynb` - numeric experiments: init variance, vanishing gradients, learning rates

---

## (a) What a neuron computes, and why nonlinearity is the entire point

A single "neuron" is embarrassingly simple. It takes a vector $x \in \mathbb{R}^n$,
owns a weight vector $w \in \mathbb{R}^n$ and a bias $b$, and computes

$$a = \sigma(w^\top x + b)$$

That's it: a **dot product** (weighted vote over the inputs), a **shift** ($b$ moves
the decision threshold), and a **squashing nonlinearity** $\sigma$. The pre-activation
$z = w^\top x + b$ is a linear function of $x$; geometrically, $z = 0$ defines a
hyperplane, $w$ is its normal vector, and $z$ measures signed distance to that plane
(scaled by $\|w\|$). So a neuron is a soft "which side of this hyperplane are you on?"
detector. A layer of $m$ neurons is just $m$ of these stacked into a matrix:

$$z = Wx + b, \qquad W \in \mathbb{R}^{m \times n}$$

### Why the nonlinearity is not optional

Suppose we get greedy and stack layers **without** nonlinearities, hoping depth alone
buys us power:

$$h_1 = W_1 x + b_1, \qquad h_2 = W_2 h_1 + b_2$$

Substitute the first into the second - this is the whole proof, two lines:

$$h_2 = W_2 (W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{W'} x + \underbrace{(W_2 b_1 + b_2)}_{b'}$$

$$\Rightarrow\; h_2 = W' x + b'$$

A stack of any number of linear layers **collapses to a single linear layer** - matrix
multiplication is closed under composition. A 100-layer linear "deep network" has
exactly the expressive power of logistic/linear regression: it can only carve the input
space with a single hyperplane per output. It cannot represent XOR, a circle, a spiral,
anything with curvature in its decision boundary.

(Note the collapse can even *lose* rank: if $W_1$ is $\mathbb{R}^{n \to 2}$ and $W_2$
is $\mathbb{R}^{2 \to m}$, then $W'=W_2W_1$ has rank $\le 2$. Depth without
nonlinearity is worse than useless.)

The nonlinearity $\sigma$ breaks the collapse: $W_2\,\sigma(W_1 x)$ does **not**
simplify, because $\sigma$ is not a matrix. Intuition for what you gain: with ReLU,
each hidden neuron folds the input space along its hyperplane; a network composes many
folds into a piecewise-linear function with exponentially many linear regions in depth.
That's the actual content of "deep networks are universal approximators" - enough
folds approximate any continuous function.

**One-sentence summary you should be able to say in an interview:** a neuron is
`dot product + bias + nonlinearity`; the nonlinearity is the entire point, because
composition of linear maps is linear ($W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2)$).

---

## (b) Backpropagation = chain rule + bookkeeping + reuse

### The chain rule, stated the useful way

If $L$ depends on $z$, and $z$ depends on $x$, then

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$

Read it as a recipe: *"the gradient flowing into a node equals the gradient flowing
out of it, times the node's own local derivative."* Every backward rule in every
autograd engine is one instance of this sentence.

### The computational-graph view

Any forward computation is a DAG: leaves are inputs/parameters, internal nodes are
primitive ops (`+`, `*`, `tanh`, ...), and there is one scalar output, the loss $L$.
Each node needs to know only two things:

1. **Local gradients**: $\partial(\text{out})/\partial(\text{each input})$ - e.g. for
   $c = a \cdot b$: $\partial c/\partial a = b$ and $\partial c/\partial b = a$.
2. **The upstream gradient** $\partial L / \partial(\text{out})$, handed to it during
   the backward pass.

It multiplies them and hands the products to its inputs. If a value feeds **two**
downstream nodes, the gradients arriving from both paths **add** (multivariate chain
rule: $\frac{\partial L}{\partial a} = \sum_{\text{paths } p} \frac{\partial L}{\partial p}\frac{\partial p}{\partial a}$).
That's why every autograd engine does `a.grad += ...`, never `a.grad = ...`.

### Why this is dynamic programming

Naively, $\partial L/\partial w$ for each parameter $w$ sums over *every path* from
$w$ to $L$ - exponentially many in depth. Backprop visits nodes in **reverse
topological order**, and by the time it reaches a node, that node's
$\partial L/\partial(\text{out})$ already aggregates *all* downstream paths into one
cached number. Each edge is processed once. That's memoization: backprop computes all
$N$ parameter gradients for roughly the cost of **one** extra forward pass, instead of
$N$ forward passes (which is what finite differences costs - hence gradient *checking*
with finite differences is fine, gradient *computation* with them is not).

### A tiny 2-layer network, by hand, on actual numbers

Network: 1 input $\to$ 1 hidden tanh neuron $\to$ 1 linear output, squared-error loss.

$$z_1 = w_1 x + b_1, \quad h = \tanh(z_1), \quad \hat{y} = w_2 h + b_2, \quad L = (\hat{y} - y)^2$$

Numbers: $x = 1.0$, target $y = 1.0$, and parameters
$w_1 = 0.5,\; b_1 = 0.1,\; w_2 = -0.3,\; b_2 = 0.2$.

**Forward pass** (compute and *write down* every intermediate - these get reused):

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
  needs forward's intermediates. Memory cost of training ≈ storing all activations.

One gradient-descent step with $\eta = 0.1$: $w_2 \leftarrow -0.3 - 0.1(-1.032333) = -0.196767$,
etc. Re-run forward: $L$ drops from $0.9237$ to about $0.49$. That is all of training  - 
this loop, a million times, on bigger graphs.

### The three backward rules that generate everything

- **Add** $c = a + b$: local grads are $1,1$ → gradient is **distributed** unchanged to both inputs (a "gradient router").
- **Mul** $c = a b$: local grads are $b, a$ → gradient to each input is scaled by *the other* input's forward value ("swap and multiply").
- **Any unary $f$**: gradient scaled by $f'$ evaluated at the forward value.

Matrix versions ($Z = XW$: $\partial L/\partial X = G W^\top$, $\partial L/\partial W = X^\top G$)
are just these rules batched - derived element-wise in `mlp_numpy.ipynb`'s comments. A
useful mnemonic: shapes must match, and there's exactly one transpose arrangement that
makes them match.

---

## (c) Losses: softmax + cross-entropy, and why $\nabla = p - y$

### Setup

For $C$-way classification, the network emits **logits** $z \in \mathbb{R}^C$
(unnormalized scores). Softmax converts them to a probability distribution:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Cross-entropy against the true class $y$ (one-hot vector $y$, true index $t$):

$$L = -\sum_i y_i \log p_i = -\log p_t$$

i.e. "negative log-probability assigned to the correct class" - this is exactly
maximum likelihood. $L = 0$ iff $p_t = 1$; confidently wrong predictions are punished
without bound (as $p_t \to 0$, $L \to \infty$), which is what generates large
corrective gradients exactly when the model is badly wrong.

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

**Why $p - y$ is beautiful:** it's bounded in $[-1, 1]$ per class, it's zero exactly
when the prediction is perfect, and the "error signal" is literally
*predicted minus true probability*. The same form appears for sigmoid + binary
cross-entropy and for linear + MSE - all are canonical link/loss pairings from
exponential-family maximum likelihood, and the $\sigma'$ term that would cause
saturation cancels against the loss's derivative. This is why you **never** pair
softmax with MSE: the cancellation doesn't happen, gradients carry a $p(1-p)$ factor,
and a confidently-wrong network ($p_t \approx 0$) gets a *tiny* gradient exactly when
it needs a huge one.

### Numerical stability: the log-sum-exp trick

Softmax needs $e^{z_i}$. In float32, $e^{89} = \infty$; so logits around $100$
(easy to hit mid-training) give `inf/inf = nan` and your run dies. The fix exploits
softmax's **shift invariance**: for any constant $m$,

$$\frac{e^{z_i - m}}{\sum_j e^{z_j - m}} = \frac{e^{z_i}e^{-m}}{e^{-m}\sum_j e^{z_j}} = p_i$$

Choose $m = \max_j z_j$. Then every exponent is $\le 0$, so $e^{z_i - m} \in (0, 1]$  - 
no overflow - and at least one term equals $1$, so the denominator is $\ge 1$ - no
division blow-up. Similarly compute the *log*-probability directly instead of
`log(softmax(z))`:

$$\log p_i = z_i - m - \log\sum_j e^{z_j - m}$$

because `log(tiny)` where `tiny` underflowed to `0.0` gives `-inf`. Every serious
framework's `cross_entropy` fuses softmax + log + NLL for exactly this reason; you
should too (see `stable_logsumexp` in `mlp_numpy.ipynb`).

---

## (d) Activation functions: saturation and its discontents

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

(Derive $\sigma'$: write $\sigma = (1+e^{-z})^{-1}$, so
$\sigma' = -(1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}
= \sigma \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(1-\sigma)$.)

**Maximum of $\sigma'$:** let $s = \sigma(z) \in (0,1)$ and maximize $g(s) = s(1-s)$.
$g'(s) = 1 - 2s = 0 \Rightarrow s = \tfrac12$, i.e. $z = 0$, giving
$g_{\max} = \tfrac12 \cdot \tfrac12 = \boxed{0.25}$.

So **every** sigmoid in the backward path multiplies the gradient by *at most* $0.25$ - and typically far less: at $|z| = 5$, $\sigma' \approx 0.0066$. In a depth-$d$ stack
of sigmoid layers, the activation-path factor is bounded by $0.25^d$ times weight
terms: $0.25^{10} \approx 10^{-6}$, $0.25^{20} \approx 10^{-12}$. Layer 1 of a deep
sigmoid net receives essentially zero signal about the loss. That is the **vanishing
gradient problem**, and it's not folklore - `training_dynamics.ipynb` prints the actual
per-layer gradient norms and you can watch them decay geometrically.

Two further sigmoid sins: outputs are **not zero-centered** (always positive → the
gradient of a layer's weights, which is `upstream × input`, has all components with
the same sign pattern → zig-zag optimization paths), and `exp` is relatively
expensive.

### Tanh

$$\tanh(z) = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z)$$

Zero-centered (fixes one sin) and $\tanh'(0) = 1$ (much better than $0.25$), but it
still **saturates**: $|z|$ large $\Rightarrow \tanh' \to 0$. Fine for shallow nets and
gates (LSTMs), still vanishes with depth.

### ReLU and dead neurons

$$\mathrm{ReLU}(z) = \max(0, z), \qquad \mathrm{ReLU}'(z) = \mathbb{1}[z > 0]$$

The derivative is exactly $1$ on the active side - repeated multiplication by $1$
doesn't shrink anything, which is the single biggest reason deep nets became trainable.
Also: no `exp`, and the induced sparsity (~half the units at 0) is cheap.

The failure mode: a **dead neuron**. If a neuron's pre-activation is negative for
*every* input in the dataset (e.g. after a large gradient step knocked its bias very
negative, or from bad init), then its output is always $0$, its local gradient is
always $0$, so no gradient ever flows to its weights, so it can never recover. It's
permanently dead capacity. Mitigations: sensible init + moderate learning rates,
LeakyReLU ($\max(\alpha z, z)$, $\alpha \approx 0.01$, keeps a trickle of gradient),
or smooth variants below. (Note ReLU is non-differentiable at exactly $z=0$; we just
pick a subgradient, conventionally $0$ - a measure-zero event nobody worries about.)

### GELU / SiLU (briefly)

$$\mathrm{GELU}(z) = z \cdot \Phi(z) \approx z\,\sigma(1.702 z), \qquad \mathrm{SiLU/Swish}(z) = z\,\sigma(z)$$

Both are "ReLU-shaped but smooth": they gate the input by a soft probability of being
positive instead of a hard threshold, are slightly negative for small negative $z$
(no hard zero → no truly dead units), and empirically train slightly better in
transformers - GELU is the default in BERT/GPT-family MLP blocks. There's no deep
theory here; treat them as smoothed ReLUs that won a lot of ablations.

**Rule of thumb:** hidden layers → ReLU family (GELU in transformers); output layer →
whatever the loss demands (linear for regression, logits into softmax-CE for
classification - never a hidden-layer sigmoid "because probabilities").

---

## (e) Initialization: why variance is the whole game

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
  $\approx 0$, all neurons compute the same nothing, gradients (which contain
  activation factors) also vanish.
- $g > 1$: activations explode geometrically → saturating nonlinearities pin at
  $\pm 1$ (and their local grads hit 0 - vanishing gradients *via explosion*), or with
  ReLU you overflow to `inf/nan`.
- $g = 1$: variance is preserved through depth. This is the **only** regime where a
  deep net starts life in a trainable state.

Requiring $\mathrm{Var}(z) = \mathrm{Var}(x)$, i.e. $g = 1$:

$$n_{in}\,\sigma_w^2 = 1 \;\Rightarrow\; \sigma_w^2 = \frac{1}{n_{in}}$$

### Xavier/Glorot init

The same argument run on the **backward** pass ($\partial L/\partial x = W^\top \delta$,
a sum over $n_{out}$ terms) demands $\sigma_w^2 = 1/n_{out}$. You can't satisfy both
unless $n_{in} = n_{out}$, so Glorot & Bengio split the difference (harmonic-mean
compromise):

$$\boxed{\sigma_w^2 = \frac{2}{n_{in} + n_{out}}} \qquad \text{(Xavier - for tanh/sigmoid-ish, i.e. } \approx \text{identity near 0)}$$

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
activations by $\sim (1/\sqrt2)^{30} \approx 3\times10^{-5}$ - watch it happen
numerically in `training_dynamics.ipynb`. Biases: init to 0 (they don't feed the variance
argument). Never init weights to 0 or any constant: all neurons in a layer then
compute identical outputs and receive identical gradients forever - the **symmetry
problem**; random init exists to break symmetry, and the *scale* of that randomness is
what Xavier/Kaiming pin down.

---

## (f) Optimization

### SGD and why minibatches

Full-batch gradient descent: $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$
where $L$ averages over all $N$ examples. For $N$ in the millions this is one very
expensive, very accurate step. Minibatch SGD instead uses $B \ll N$ random examples:
the minibatch gradient is an **unbiased estimator** of the full gradient
($E[\hat{g}] = g$) with variance $\propto 1/B$. Why it wins:

1. **Compute**: gradient quality improves like $\sqrt{B}$ (standard error) but cost
   grows like $B$ - heavily diminishing returns; better to take many noisy steps.
2. **Hardware**: matrix-matrix ops at batch size 32–1024 saturate GPUs; batch size 1
   wastes the machine.
3. **Noise as a feature**: gradient noise helps escape saddle points and sharp minima
   and acts as implicit regularization; small/medium batches often generalize better
   than full batch.

### Momentum: EMA of gradients, and the effective step

$$v_t = \beta v_{t-1} + g_t, \qquad \theta \leftarrow \theta - \eta\, v_t \qquad (\beta \approx 0.9)$$

Unroll the recursion: $v_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots$ - an
exponentially-weighted sum of past gradients ("a heavy ball with friction $1-\beta$";
the EMA-flavored variant $v_t = \beta v_{t-1} + (1-\beta)g_t$ is the same thing with
$\eta$ rescaled). **Effective step size**: if the gradient were constant at $g$, the
velocity converges to the geometric-series limit

$$v_\infty = g\,(1 + \beta + \beta^2 + \cdots) = \frac{g}{1 - \beta}$$

so with $\beta = 0.9$ momentum takes steps up to $\frac{1}{1-\beta} = 10\times$ larger
than plain SGD **along directions where gradients consistently agree**, while
directions where the gradient oscillates (+,−,+,−, the across-the-ravine direction)
cancel in the sum. Net effect: accelerate along the valley floor, damp the sideways
zig-zag. That's the entire trick.

### RMSProp: per-parameter learning rates

Keep an EMA of the **squared** gradient and divide by its root:

$$s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t$$

Parameters with persistently large gradients get their steps shrunk; parameters with
tiny, rare gradients get boosted. It approximately normalizes the update to
$\pm\eta \cdot \mathrm{sign\text{-}ish}(g)$, making one global $\eta$ work across
parameters whose gradient scales differ by orders of magnitude (embeddings vs. biases).

### Adam = momentum + RMSProp + bias correction

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
\theta \leftarrow \theta - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta \sim 10^{-3}$.

**Deriving the bias correction** (interviewers love this). $m_0 = 0$, so unrolling:

$$m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{\,t-i} g_i$$

Take expectations assuming $E[g_i] \approx g$ (stationary gradient):

$$E[m_t] = (1-\beta_1)\, g \sum_{i=1}^{t} \beta_1^{\,t-i} = (1-\beta_1)\, g \cdot \frac{1-\beta_1^t}{1-\beta_1} = g\,(1 - \beta_1^t)$$

So the raw EMA is biased toward zero by exactly the factor $(1-\beta_1^t)$ - because
it was initialized at 0 and hasn't accumulated enough mass yet. Dividing by
$1-\beta_1^t$ makes it unbiased: $E[\hat m_t] = g$. Same for $v_t$ with $\beta_2$.
Why it *matters*: with $\beta_2 = 0.999$, after one step $v_1 = 0.001\,g_1^2$, so
uncorrected $\sqrt{v_1} \approx 0.03|g_1|$ and the update $g_1/\sqrt{v_1}$ is
$\sim 30\times$ too large - early training blows up. As $t \to \infty$,
$\beta^t \to 0$ and the correction fades to a no-op; it only matters for early steps.

### Learning-rate schedules

$\eta$ is the single most important hyperparameter (see the divergence/slow/converge
experiment in `training_dynamics.ipynb`). Common schedules:

- **Step decay**: drop $\eta$ by 10× at fixed milestones.
- **Cosine**: $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1 + \cos(\pi t/T))$ - smooth decay to ~0, the modern default.
- **Warmup**: linearly ramp $\eta$ from 0 over the first few hundred/thousand steps.
  Rationale: Adam's $\hat v$ statistics are garbage at $t$ small and early gradients
  are huge/atypical; warmup keeps steps tiny until statistics stabilize. Essential for
  transformers.

Big picture: large $\eta$ early = fast progress + exploration (noise ball); small
$\eta$ late = settle into a minimum. Decay trades exploration for convergence.

### Weight decay vs. L2, and AdamW

For **plain SGD** the two are identical: L2 loss $\frac{\lambda}{2}\|\theta\|^2$ adds
$\lambda\theta$ to the gradient, giving $\theta \leftarrow (1 - \eta\lambda)\theta - \eta g$ - exactly "decay the weights, then step." For **Adam they differ**: L2's $\lambda\theta$
enters $m_t$ and $v_t$ and gets *divided by* $\sqrt{\hat v}$ - so parameters with large
gradient history get *less* regularization, which is backwards and empirically hurts.
**AdamW decouples**: keep $\lambda\theta$ out of the moments and apply decay directly
in the update,

$$\theta \leftarrow \theta - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta\right)$$

Every parameter is decayed at the same rate regardless of its gradient statistics.
This is why "Adam + L2" and "AdamW" are different optimizers, and why AdamW is the
default for transformers.

---

## (g) Normalization

### BatchNorm: the forward math

For each feature (channel) independently, over a minibatch of size $B$:

$$\mu = \frac{1}{B}\sum_{i=1}^{B} x_i, \qquad \sigma^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma\,\hat{x}_i + \beta$$

Standardize to mean 0 / variance 1 using **batch statistics**, then let learned
per-feature $\gamma, \beta$ scale and shift back to wherever the network actually
wants - so BN restricts nothing; it *reparameterizes*, decoupling the mean/scale of a
layer's activations (now directly set by $\beta, \gamma$) from the tangled interaction
of all upstream weights.

### The two stories of why it works

**Story 1 (the original paper): internal covariate shift.** As layer $k$'s weights
update, the *distribution* of layer $k{+}1$'s inputs shifts, so every layer is chasing
a moving target; BN pins each layer's input distribution (mean/var) so layers learn
against a stationary interface. Intuitive, widely repeated - and probably not the real
mechanism: later work (Santurkar et al. 2018) showed you can inject non-stationary
noise *after* BN, reintroducing covariate shift, and BN still helps just as much.

**Story 2 (the better modern story): loss-surface smoothing.** BN demonstrably reduces
the Lipschitz constant of the loss and of its gradient (makes the landscape smoother
and gradients more predictable), which permits **much larger learning rates** without
divergence and makes training far less sensitive to init. Mechanically: the
normalization makes the layer's output scale-invariant to the weight scale
($\mathrm{BN}(aWx) = \mathrm{BN}(Wx)$ for $a>0$), which kills a whole axis of
pathological curvature. Interview-safe phrasing: "the covariate-shift story is the
historical motivation; the empirical story is that BN smooths the optimization
landscape, enabling higher LR and reducing init sensitivity."

### Why BN needs running statistics at inference

At inference you may get **one** example - a "batch" mean over one sample would be the
sample itself ($\hat x = 0$: nonsense), and predictions would depend on whatever else
happened to be in your batch (non-deterministic, a serving bug and even a privacy
leak). So during training BN maintains EMAs

$$\mu_{run} \leftarrow (1-\alpha)\,\mu_{run} + \alpha\,\mu_{batch}, \qquad \sigma^2_{run} \leftarrow (1-\alpha)\,\sigma^2_{run} + \alpha\,\sigma^2_{batch}$$

and at eval time uses these **frozen** population estimates, making BN a fixed affine
transform per feature. This train/eval discrepancy is the classic source of the
"model.eval() bug" - forgetting to switch modes silently corrupts inference.

### LayerNorm, and why transformers use it

LayerNorm normalizes across the **feature dimension of a single example**:

$$\mu_i = \frac{1}{D}\sum_{d} x_{i,d}, \qquad y_{i,d} = \gamma_d\,\frac{x_{i,d} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_d$$

Same example-in, example-out statistics at train and test - **no batch dependence, no
running stats, no train/eval mode**. Transformers use LN instead of BN because:

1. **Variable-length sequences + padding** make per-position batch statistics ill-defined.
2. **Batch sizes are small / sharded** in large-model training; BN stats get noisy and
   require cross-device sync (expensive all-reduces).
3. **Autoregressive decoding** processes one token at a time - there is no batch.
4. Each token independently normalized is exactly the granularity a per-token
   residual-stream architecture wants (modern LLMs use the pre-norm placement, often
   RMSNorm - LayerNorm minus the mean-centering, cheaper, works as well).

---

## (h) Regularization

### Dropout

Training: each hidden unit is independently zeroed with probability $p$; surviving
units are scaled by $\frac{1}{1-p}$ (**inverted dropout**). Inference: identity, no-op.

**Why the scaling** - preserve expectation. Let $m \sim \mathrm{Bernoulli}(1-p)$ and
$\tilde{a} = \frac{m \cdot a}{1-p}$. Then

$$E[\tilde{a}] = \frac{E[m]\,a}{1-p} = \frac{(1-p)\,a}{1-p} = a$$

so the *expected* input each downstream unit sees is identical at train and test time - without the scaling, test-time activations would be systematically $\frac{1}{1-p}$×
larger than what the network trained on, mis-calibrating every downstream layer.
(Original paper scaled by $1-p$ at *test* time instead; inverted dropout is the same
idea, moved to training so inference stays a clean no-op.)

**Ensemble interpretation**: a net with $n$ droppable units defines $2^n$ subnetworks
sharing weights; each training step samples one and trains it. Test time approximates
the geometric-mean ensemble of all $2^n$ members with a single forward pass. Practical
effect: no unit can rely on a specific co-adapted partner (the partner might be gone),
so features must be individually useful - redundant, robust representations.

### Early stopping

Monitor validation loss; stop (or restore the checkpoint from) when it stops
improving, even as training loss keeps falling. It's regularization because it limits
how far $\theta$ travels from init - for linear models this is provably akin to L2
regularization with strength $\sim 1/(\eta t)$. Free, effective, and doubles as your
model-selection criterion; use it basically always (with a `patience` parameter to
tolerate noisy validation curves).

### Data augmentation

Expand the training distribution with label-preserving transforms: crops, flips,
color jitter, cutout/mixup for images; noise/time-shift for audio; back-translation
for text. It injects **invariances you know** (a flipped cat is a cat) directly as
data instead of hoping the net infers them - often the highest-value regularizer
because it attacks generalization at the data level, where the information actually
is. Caution: transforms must respect the label semantics (don't horizontally flip
digits: a flipped "3" is not a "3").

---

## (i) Vanishing/exploding gradients, and residual connections

### The depth problem, stated with Jacobians

Backprop through $d$ layers multiplies $d$ Jacobians:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_d}\; \prod_{k=1}^{d} J_k, \qquad J_k = \frac{\partial h_k}{\partial h_{k-1}} = \mathrm{diag}(\sigma'(z_k))\, W_k$$

If the typical factor magnitude (roughly $\sigma'_{typ} \cdot \|W\|$-ish) is $c$, the
product scales like $c^{\,d}$: $c<1 \Rightarrow$ gradients **vanish** (early layers
frozen, only the top of the net trains); $c>1 \Rightarrow$ gradients **explode**
(loss spikes, `nan`s - mitigated by gradient clipping, which rescales $g$ when
$\|g\| >$ threshold). Sigmoid's $\sigma' \le 0.25$ makes vanishing near-certain
(section d); nothing keeps $c$ pinned at exactly 1 through training even with perfect
init (section e). Depth itself is the enemy: geometric in $d$.

### Residual connections: the gradient highway

He et al.'s fix - make each block compute a **perturbation of identity**:

$$h_{k+1} = h_k + F(h_k)$$

Differentiate. Here is the crucial Jacobian - note the **$+\,I$**:

$$\frac{\partial h_{k+1}}{\partial h_k} = I + \frac{\partial F}{\partial h_k}$$

Chain through $d$ blocks:

$$\frac{\partial h_d}{\partial h_0} = \prod_{k=0}^{d-1}\left(I + \frac{\partial F_k}{\partial h_k}\right) = I + \sum_k \frac{\partial F_k}{\partial h_k} + (\text{higher-order products})$$

Even if every $\partial F_k/\partial h_k$ is tiny, the product contains the pure-$I$
term: **gradient reaches layer 0 with factor exactly 1 along the skip path**, plus
corrections. The identity path is a "gradient highway" that no block can block  - 
contrast the plain net, where the gradient *must* pass through every $J_k$ and one
saturated layer kills everything below it. Equivalently forward: the net starts as
approximately the identity function (if $F \approx 0$ at init, which good init gives
you) and learns residual corrections - a far easier starting point than learning an
arbitrary deep map from scratch. This one-line change ($+\,h_k$) is what made
50/100/1000-layer nets trainable, and it's why *every* transformer block wraps
attention and MLP in `x = x + f(norm(x))`: residual highway + per-block normalization
is the modern recipe (plus clipping for the exploding side).

---

## Common interview questions

**Q1. Why do neural networks need nonlinear activation functions?**
Because composition of linear maps is linear: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$.
Any stack of linear layers collapses to one linear layer, so without nonlinearities a
deep net has exactly the capacity of linear/logistic regression and can't represent
XOR, circles, or any curved decision boundary. The nonlinearity blocks the collapse
and lets depth compose folds of the input space.

**Q2. Explain backpropagation in one minute.**
Write the computation as a DAG of primitive ops. Forward pass computes and caches every
node's value. Backward pass walks nodes in reverse topological order; each node
multiplies the upstream gradient $\partial L/\partial(\text{out})$ by its local
derivative and adds the result into each input's gradient (adds, because fan-out paths
sum by the multivariate chain rule). It's the chain rule plus memoization: every edge
is touched once, so all parameter gradients cost about one extra forward pass.

**Q3. Why is backprop efficient compared to computing each parameter's gradient separately?**
Finite differences needs one forward pass *per parameter* ($O(N)$ passes). Backprop is
reverse-mode autodiff: because the output (loss) is a single scalar, one backward
sweep computes $\partial L/\partial\theta$ for **all** parameters simultaneously,
reusing each node's aggregated downstream gradient - dynamic programming over the
graph. Cost $\approx$ 2–3 forward passes total.

**Q4. Derive the gradient of softmax + cross-entropy.**
$L = -\log p_t = -z_t + \log\sum_j e^{z_j}$. Then
$\partial L/\partial z_k = -\mathbb{1}[k{=}t] + e^{z_k}/\sum_j e^{z_j} = p_k - y_k$.
Clean, bounded, zero at a perfect prediction; the softmax Jacobian's messy two-case
structure cancels against the $-1/p_t$ from the log.

**Q5. Why not use MSE for classification?**
With a softmax/sigmoid output, MSE's gradient retains a $\sigma'$-type factor
(e.g. $p(1-p)$), which $\to 0$ when the model is *confidently wrong* - tiny gradients
exactly when you need the largest correction, plus a non-convex loss in the logits.
Cross-entropy cancels that factor, giving gradient $p - y$: large error → large
gradient. (It's also the maximum-likelihood loss for a categorical model.)

**Q6. What is the vanishing gradient problem and what causes it?**
Backprop multiplies one Jacobian per layer; each contains the activation derivative.
Sigmoid's derivative is at most $0.25$ (max of $s(1-s)$ at $s=\frac12$), so $d$
sigmoid layers scale gradients by at most $0.25^d$ - geometrically small, so early
layers get no learning signal. Causes: saturating activations, bad init scale
($n\sigma_w^2 \ne 1$), and raw depth. Fixes: ReLU-family activations (derivative 1 on
the active side), Xavier/Kaiming init, normalization layers, residual connections.

**Q7. What is a dead ReLU and how do you prevent it?**
A unit whose pre-activation is negative for every input: output always 0, local
gradient always 0, so its weights receive no gradient and it can never recover  - 
typically caused by a huge update (high LR) knocking the bias very negative. Prevent
with sane init, moderate/warmup LR, or LeakyReLU/GELU which keep nonzero gradient for
negative inputs.

**Q8. Derive Xavier and Kaiming initialization.**
For $z = Wx$ with i.i.d. mean-0 weights, $\mathrm{Var}(z) = n_{in}\sigma_w^2\,\mathrm{Var}(x)$.
Depth multiplies this factor per layer, so we need $n_{in}\sigma_w^2 = 1 \Rightarrow \sigma_w^2 = 1/n_{in}$
(and $1/n_{out}$ for the backward pass; Xavier compromises with $2/(n_{in}{+}n_{out})$).
ReLU zeroes half a symmetric distribution, halving the second moment
($E[\max(0,z)^2] = \frac12\mathrm{Var}(z)$), so He init compensates with
$\sigma_w^2 = 2/n_{in}$.

**Q9. Why initialize randomly at all - why not zeros?**
Constant init makes every neuron in a layer compute the same output and receive the
same gradient, so they stay identical forever (symmetry never breaks); the layer
behaves as one neuron. Random init breaks symmetry; Xavier/Kaiming set the *scale* of
that randomness so activation/gradient variance is preserved through depth.

**Q10. Explain Adam. Why the bias correction?**
Adam keeps an EMA of gradients ($m$, momentum) and of squared gradients ($v$,
RMSProp-style per-parameter scaling), updating with $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$.
Both EMAs start at 0, so early on they underestimate: unrolling gives
$E[m_t] = g(1-\beta_1^t)$, hence dividing by $1-\beta_1^t$ makes the estimate
unbiased. It matters mostly for $v$: with $\beta_2 = 0.999$, $v_1 = 0.001g^2$, so the
uncorrected step is $\sim\!\sqrt{1000}\approx 30\times$ too large on step 1. The
correction decays to a no-op as $t$ grows.

**Q11. Weight decay vs. L2 regularization - same thing?**
Identical under plain SGD ($\theta \leftarrow (1-\eta\lambda)\theta - \eta g$).
Different under Adam: L2's $\lambda\theta$ term flows through $m$ and $v$ and gets
divided by $\sqrt{\hat v}$, so high-gradient parameters are under-regularized. AdamW
decouples the decay from the adaptive machinery, applying $-\eta\lambda\theta$
directly - uniform regularization, better generalization; it's the transformer
default.

**Q12. What does BatchNorm do, and why does it help?**
Per feature, standardize using minibatch mean/variance, then apply learned $\gamma,\beta$.
Historical story: fixes internal covariate shift. Better-supported story: it smooths
the loss landscape (and makes the layer invariant to weight scale), so you can use
much larger learning rates and are far less sensitive to initialization. It also adds
a mild noise regularization via batch statistics.

**Q13. Why does BatchNorm behave differently at train vs. test time?**
Test batches may have size 1 and predictions must not depend on batch-mates, so at
inference BN uses running (EMA) estimates of mean/variance collected during training,
becoming a fixed per-feature affine map. Forgetting `model.eval()` - i.e. still using
batch stats at inference - is the classic silent bug.

**Q14. Why do transformers use LayerNorm instead of BatchNorm?**
LN normalizes across features within a single example - no batch dependence. That
suits transformers because sequences are variable-length/padded (batch stats
ill-defined per position), batches are small and sharded across devices (BN would need
sync), and autoregressive decoding has no batch at all. Same computation train/test,
no running stats. (Modern LLMs typically use pre-norm RMSNorm.)

**Q15. How does dropout work, and why scale by $1/(1-p)$ during training?**
Each unit is zeroed w.p. $p$ during training; survivors are scaled by $1/(1-p)$ so
$E[\tilde a] = \frac{(1-p)a}{1-p} = a$ - downstream layers see the same expected input
at train and test, letting inference be an exact no-op. Interpretations: prevents
co-adaptation (no unit can rely on a partner that might vanish) and approximates an
ensemble of $2^n$ weight-sharing subnetworks in one forward pass.

**Q16. Why do residual connections help train very deep networks?**
The block is $h_{k+1} = h_k + F(h_k)$, so its Jacobian is $I + \partial F/\partial h$.
The product of block Jacobians therefore contains a pure identity term: gradient
reaches the earliest layers with factor exactly 1 along the skip path, regardless of
how small each $\partial F/\partial h$ is - a gradient highway. Forward view: the net
starts near the identity map and learns small corrections, which is much easier than
learning an arbitrary deep map.

**Q17. Your training loss is NaN after a few hundred steps. Debug it.**
Usual suspects in order: (1) learning rate too high → loss spikes then overflows
(try 10× lower); (2) unstable softmax/log - must use the log-sum-exp trick / fused
cross-entropy, never `log(softmax(z))`; (3) exploding gradients → add gradient
clipping, check init scale; (4) bad data (NaN/inf features, labels out of range);
(5) division by ~0 somewhere ($\epsilon$ missing in a norm or Adam). Bisect by logging
$\|g\|$ and activation stats per layer and finding the first step/layer that blows up.

**Q18. Why train with minibatches rather than the full dataset?**
Minibatch gradients are unbiased estimates whose standard error shrinks like
$1/\sqrt{B}$ while cost grows like $B$ - past a modest $B$, compute is better spent on
more steps than better steps. Batched ops also saturate parallel hardware. And the
gradient noise itself helps: it escapes saddle points and biases toward flatter minima
that generalize better.

**Q19. How would you verify a hand-written backward pass?**
Finite-difference gradient check: for each parameter $\theta_i$, compare the analytic
gradient with the centered difference
$\frac{L(\theta_i + h) - L(\theta_i - h)}{2h}$, $h \approx 10^{-5}$, using **relative**
error $\frac{|g_a - g_n|}{|g_a| + |g_n| + \epsilon} \lesssim 10^{-6}$ in float64.
Centered differences have $O(h^2)$ error vs. $O(h)$ for one-sided. Do it on a tiny
model/batch, ideally at a point where ReLUs aren't near their kink. Both
`micrograd_plus.ipynb` and `mlp_numpy.ipynb` implement this.

**Q20. Momentum: what does it actually do to the update?**
It's an exponentially-decayed sum of past gradients: $v_t = \sum_i \beta^{t-i} g_i$.
For a persistent gradient direction it converges to $g/(1-\beta)$ - an effective step
up to $10\times$ larger at $\beta=0.9$ - while oscillating components (across a ravine)
cancel. So: acceleration along consistent directions, damping of zig-zag, and some
ability to roll through small bumps and saddle regions.

## Exit checklist

- [ ] Cold, on paper: derive the softmax + cross-entropy gradient to p − y - 5 min
- [ ] Backprop a 2-layer MLP by hand, writing every forward/backward shape and the three backward rules - 6 min
- [ ] Derive Xavier and Kaiming init variances, including where ReLU's factor of 2 comes from - 4 min
- [ ] Write Adam's update equations with bias correction, and state AdamW vs L2 weight decay - 4 min
- [ ] BatchNorm vs LayerNorm: forward math, why BN needs running stats at inference, why transformers use LN - 4 min
- [ ] Explain vanishing/exploding gradients via products of Jacobians, and how residual connections fix it - 3 min
- [ ] Dead ReLUs: cause, how to detect, two fixes; plus the log-sum-exp stability trick - 3 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
