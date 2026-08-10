# Foundations: Linear Algebra, Calculus, Probability - from first principles

The goal of this note: you should be able to *re-derive* everything here on a whiteboard,
not just recite it. Every ML system - a linear model, a transformer, a diffusion model  - 
is built out of exactly three ingredients:

1. **Linear algebra**: how we *represent* data and transform it (matrices).
2. **Calculus**: how we *improve* the transformation (gradients).
3. **Probability**: how we *justify* the loss function we're improving (likelihood).

If you deeply understand these three, "backprop through a transformer" stops being magic
and becomes bookkeeping.

---

## Part 1: Linear algebra for ML

### 1.1 Vectors are data points

Forget arrows in space for a second. In ML, a vector is just **a list of measurements
about one thing**. A house might be $x = [1400, 3, 1985]$ - square feet, bedrooms,
year built. A word in a transformer is a vector of 768 learned numbers. That's it.

The "arrow" picture is still useful for one reason: it gives us geometry. Once data
points are vectors, we can ask geometric questions: *how long is it? how similar are
two points? what direction does it point?* Those questions turn out to be the entire
substrate of ML.

### 1.2 Dot product = similarity = projection

Take two tiny vectors: $a = [1, 2]$, $b = [3, 1]$.

$$a \cdot b = 1\cdot 3 + 2 \cdot 1 = 5$$

Mechanically: multiply matching entries, add them up. But *why* is this number
meaningful? Two views, both essential:

**View 1: similarity.** The identity (provable from the law of cosines):

$$a \cdot b = \|a\| \, \|b\| \cos\theta$$

where $\theta$ is the angle between them. If the vectors point the same way,
$\cos\theta = 1$ and the dot product is large and positive. Orthogonal → 0.
Opposite → negative. So the dot product is an (unnormalized) **agreement score**.
Divide out the lengths and you get **cosine similarity** - the thing every embedding
search engine computes. Attention scores in a transformer are literally
$q \cdot k$ - "how much does this query agree with this key?"

**View 2: projection.** $\frac{a \cdot b}{\|b\|}$ is the length of the shadow $a$ casts
onto the direction of $b$. So a dot product asks: *how much of $a$ lives along $b$'s
direction?* A linear model's prediction $w \cdot x$ is exactly this: project the data
point onto the learned direction $w$; points far along $w$ get big scores.
**One learned direction = one feature detector.** That single sentence explains
what every neuron in a network's linear layer is doing.

Check with numbers: $\|a\| = \sqrt{5} \approx 2.24$, $\|b\| = \sqrt{10} \approx 3.16$,
so $\cos\theta = 5 / (2.24 \cdot 3.16) \approx 0.707$, i.e. $\theta = 45°$. Draw the two
arrows - it's visibly 45°. The formula isn't abstract; it just works.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 320 240" role="img" aria-label="Vectors a=[1,2] and b=[3,1] drawn from the origin with a 45 degree angle between them" style="max-width:340px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="arr-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="arr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.12">
    <line x1="110" y1="40" x2="110" y2="200"/><line x1="180" y1="40" x2="180" y2="200"/><line x1="250" y1="40" x2="250" y2="200"/>
    <line x1="20" y1="130" x2="300" y2="130"/><line x1="20" y1="60" x2="300" y2="60"/>
  </g>
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="20" y1="200" x2="300" y2="200"/>
    <line x1="40" y1="220" x2="40" y2="40"/>
  </g>
  <g fill="currentColor" fill-opacity="0.55">
    <text x="110" y="214" text-anchor="middle">1</text>
    <text x="180" y="214" text-anchor="middle">2</text>
    <text x="250" y="214" text-anchor="middle">3</text>
    <text x="30" y="134" text-anchor="end">1</text>
    <text x="30" y="64" text-anchor="end">2</text>
  </g>
  <path d="M 85.5 184.8 A 48 48 0 0 0 61.5 157.1" fill="none" stroke="currentColor" stroke-opacity="0.7"/>
  <text x="90" y="163" fill="currentColor">45&#176;</text>
  <line x1="40" y1="200" x2="106" y2="68" stroke="#6366f1" stroke-width="2.5" marker-end="url(#arr-a)"/>
  <line x1="40" y1="200" x2="244" y2="132" stroke="#10b981" stroke-width="2.5" marker-end="url(#arr-b)"/>
  <text x="112" y="52" fill="#6366f1" font-weight="600">a = [1, 2]</text>
  <text x="252" y="126" fill="#10b981" font-weight="600">b = [3, 1]</text>
</svg>
</div>

### 1.3 Matrix multiply, view 1: a batch of dot products

Take $W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $x = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$.

$$Wx = \begin{bmatrix} 1\cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4\cdot 6\end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

Row 1 of $W$ dotted with $x$, then row 2 of $W$ dotted with $x$. So $Wx$ is just
**many similarity scores computed at once**: each row of $W$ is a feature detector,
and $Wx$ asks "how much does $x$ activate each detector?" A linear layer with 768
inputs and 3072 outputs is 3072 dot products. Stack a batch of inputs as rows of $X$
and $XW^\top$ does *batch × detectors* dot products in one shot. This is why GPUs +
matmuls = deep learning: the whole field runs on "many dot products at once."

### 1.4 Matrix multiply, view 2: a linear map (columns tell you everything)

Same numbers, different lens. Forget the rows for a moment and ask a simpler
question: what does $W$ do to the two easiest possible inputs, the unit axis
vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$?

$$W e_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix} \qquad W e_2 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}2\\4\end{bmatrix}$$

Do the arithmetic and you'll see the pattern: a one-hot input zeroes out every
column except one. **Feeding in $e_1$ reads off column 1 of $W$; feeding in $e_2$
reads off column 2.** That's step one.

Step two: any input is just a recipe built from those two ingredients.

$$x = \begin{bmatrix}5\\6\end{bmatrix} = 5\,e_1 + 6\,e_2$$

Read it as "5 parts $e_1$, 6 parts $e_2$."

Step three: matrix multiplication is **linear**, meaning $W$ processes a recipe
ingredient by ingredient: $W(5\,e_1 + 6\,e_2) = 5\,(W e_1) + 6\,(W e_2)$. We already
know what $W$ does to each ingredient (it hands back a column), so substitute:

$$Wx = 5 \begin{bmatrix}1\\3\end{bmatrix} + 6\begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}17\\39\end{bmatrix}$$

Same $[17, 39]$ as the row picture, but now it says something different:
**$Wx$ is a blend of $W$'s columns, and the entries of $x$ are the blend weights.**

Two consequences worth internalizing:

1. **The columns tell you everything.** Once you know where $W$ sends $e_1$ and
   $e_2$, linearity pins down where it sends *every* vector. A matrix is nothing
   more than a lookup table of "where do the axes land."
2. **Geometry.** $W$ warps all of space - rotating, stretching, shearing - and the
   columns are literally the warped images of the unit axes: the square spanned by
   $e_1, e_2$ gets mapped to the parallelogram spanned by $[1,3]$ and $[2,4]$.

Why hold both views? View 1 (rows = detectors) explains what a layer *computes*.
View 2 (columns = where basis goes) explains *composition*: $ABx$ means "warp by $B$,
then warp by $A$", which is why matrix multiplication is associative and why a stack
of linear layers **without nonlinearities collapses to a single linear layer**
($W_2 W_1$ is just another matrix - a classic interview question).

### 1.5 Eigenvectors and eigenvalues: the directions a matrix doesn't rotate

A matrix warps space. Are there directions it merely *stretches*? A vector $v \ne 0$
with

$$A v = \lambda v$$

is an **eigenvector**: the map doesn't rotate it, just scales it by the **eigenvalue**
$\lambda$. Concrete: $A = \begin{bmatrix} 2 & 1 \\ 1 & 2\end{bmatrix}$.
Try $v = [1, 1]$: $Av = [3, 3] = 3v$. Eigenvalue 3. Try $v = [1, -1]$:
$Av = [1, -1] = 1 \cdot v$. Eigenvalue 1. Along the diagonal direction this matrix
stretches 3×; along the anti-diagonal it does nothing. Those two directions are
orthogonal - not an accident: **symmetric matrices always have a full set of
orthogonal eigenvectors with real eigenvalues** (the spectral theorem). In the
eigenbasis, $A$ is just a diagonal matrix - the map decomposes into independent
per-axis stretches.

Why care in ML:

- **PCA**: the covariance matrix is symmetric; its top eigenvectors are the directions
  of maximum variance in your data. PCA = "rotate into the eigenbasis, keep the axes
  with big $\lambda$."
- **Repeated application**: $A^k v$ blows up along the largest-$|\lambda|$ eigenvector.
  This is why RNNs explode/vanish (Jacobian eigenvalues $>1$ or $<1$ compounded over
  time), why power iteration finds the top eigenvector, and roughly how PageRank works.
- **Optimization**: the eigenvalues of the loss's Hessian are the curvatures along its
  eigen-directions; the ratio $\lambda_{max}/\lambda_{min}$ (condition number) governs
  how miserable gradient descent will be (zig-zagging across a narrow valley).

### 1.6 SVD: every matrix is rotate → stretch → rotate

Eigendecomposition needs square (and, to be nice, symmetric) matrices. Real data
matrices are $n \times d$, rectangular. The **singular value decomposition** works for
*any* matrix:

$$A = U \Sigma V^\top$$

where $U, V$ have orthonormal columns (rotations/reflections) and $\Sigma$ is diagonal
with non-negative **singular values** $\sigma_1 \ge \sigma_2 \ge \dots$ Read right to
left: rotate the input ($V^\top$), stretch each new axis by $\sigma_i$ ($\Sigma$),
rotate into output space ($U$). **Every linear map, no matter how gnarly, is just
rotate–stretch–rotate.** The unit circle always maps to an ellipse; $\sigma_i$ are the
ellipse's radii.

The killer application: truncate. Keep only the top $k$ singular values:

$$A \approx \sum_{i=1}^{k} \sigma_i \, u_i v_i^\top$$

This is the **best rank-$k$ approximation** of $A$ (Eckart–Young theorem). Each term
$u_i v_i^\top$ is a rank-1 "pattern"; $\sigma_i$ says how much of $A$ that pattern
explains. This one idea powers: PCA (SVD of the centered data matrix), classic
recommender systems (user × item matrix ≈ low rank), model compression, and **LoRA**
(fine-tuning updates $\Delta W$ are assumed low-rank: $\Delta W = BA$ with tiny inner
dimension). Connection to eigen: $V$ holds eigenvectors of $A^\top A$, and
$\sigma_i^2$ are its eigenvalues.

### 1.7 Norms: what "big" means

A norm is a formalized notion of length. The ones that matter:

- **L2**: $\|x\|_2 = \sqrt{\sum_i x_i^2}$. Ordinary Euclidean length. Squared-L2 shows
  up everywhere because it's smooth and its gradient is linear:
  $\nabla \|x\|_2^2 = 2x$. Weight decay penalizes $\|w\|_2^2$, shrinking all weights
  smoothly toward 0.
- **L1**: $\|x\|_1 = \sum_i |x_i|$. Its unit ball is a diamond with corners *on the
  axes*, so optimization constrained by it lands on corners → exact zeros → **sparsity**
  (lasso). L2's ball is round, no corners, so it shrinks but rarely zeros.
- **L∞**: $\max_i |x_i|$ - worst-case coordinate; the norm of adversarial-example fame.
- **Frobenius** (matrices): $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$ - L2 with the matrix
  flattened.

Norms also define distance ($\|a - b\|$) and normalization ($x / \|x\|$, as in
LayerNorm's core idea and cosine similarity). Interview-relevant contrast in one line:
*L2 shrinks, L1 selects.*

---

## Part 2: Calculus for ML

### 2.1 The derivative is local sensitivity

Forget limits-as-ritual. The derivative answers one concrete question:
**if I wiggle the input a tiny bit, how much does the output wiggle?**

$$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad \text{for tiny } h$$

Take $f(x) = x^2$ at $x = 3$: $f(3) = 9$, $f(3.001) = 9.006001$. Output moved by
$0.006001$ for an input wiggle of $0.001$ → sensitivity $\approx 6.001 \approx 6 = 2x$.
The derivative is the **exchange rate between input wiggles and output wiggles**.
Equivalently: zoom in far enough and every smooth function is a straight line;
$f'$ is that line's slope: $f(x + h) \approx f(x) + f'(x) h$.

This wiggle view is *the* mental model for training: the gradient of the loss w.r.t. a
weight says "if I nudge this weight up by $\epsilon$, the loss changes by (gradient)
$\times \epsilon$." Gradient descent just nudges every weight in whichever direction
lowers the loss, proportional to its sensitivity.

Numerics aside: in code, prefer the **centered** difference
$\frac{f(x+h) - f(x-h)}{2h}$ - its error is $O(h^2)$ instead of $O(h)$ (the even-order
error terms cancel; Taylor-expand both sides to see it). This is the standard gradient
checker, implemented in `gradients_from_scratch.ipynb`.

### 2.2 The chain rule is the ONLY thing backprop needs

Compose two functions: $z = g(x)$, then $y = f(z)$. Wiggle $x$ by $\epsilon$; $z$
wiggles by $g'(x)\,\epsilon$; that wiggle in $z$ makes $y$ wiggle by
$f'(z) \cdot g'(x)\, \epsilon$. So:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}$$

**Wiggle rates multiply through a pipeline.** That's the whole theorem. Concrete:
$y = (3x)^2$ at $x = 2$: inner rate 3, outer rate $2z = 12$, total $36$. Check
directly: $y = 9x^2$, $y' = 18x = 36$. ✓

Backprop is *nothing but* this, applied to a graph and organized efficiently:

1. **Forward pass**: compute and cache every intermediate value.
2. **Backward pass**: walk the graph in reverse, and at each node multiply the
   incoming "wiggle rate of the loss w.r.t. my output" (the upstream gradient) by the
   node's **local derivative**, then pass it on. Where paths merge (a variable used
   twice), **add** the contributions - because independent wiggle paths superpose.

There is no other math in backprop. Every exotic layer - attention, layernorm,
convolutions - just needs its local derivative; the chain rule does the rest.
Why backward and not forward? With many parameters ($n$ inputs, 1 scalar loss),
reverse mode computes all $n$ sensitivities in one backward sweep at roughly the cost
of one forward pass; forward mode would need $n$ sweeps. That asymmetry is the entire
reason deep learning is computationally feasible.

### 2.3 Gradients of vector/matrix expressions

When $f : \mathbb{R}^n \to \mathbb{R}$ (a loss!), the **gradient** $\nabla f$ collects
all $n$ partial sensitivities into a vector. Key fact: $\nabla f$ points in the
direction of **steepest ascent**. Why: the wiggle from stepping in unit direction $u$
is $\nabla f \cdot u = \|\nabla f\| \cos\theta$ - maximized when $u$ aligns with
$\nabla f$ (dot product as projection, again!). Hence *descent* steps along
$-\nabla f$.

The three identities that cover 90% of ML derivations (all provable by writing out
components - do it once by hand):

| Expression | Gradient | Sanity check (1-D) |
|---|---|---|
| $f(x) = a^\top x$ | $\nabla f = a$ | $ax \to a$ |
| $f(x) = x^\top A x$ | $\nabla f = (A + A^\top)x$ | $ax^2 \to 2ax$ |
| $f(x) = \|x\|^2 = x^\top x$ | $\nabla f = 2x$ | $x^2 \to 2x$ |

The 1-D sanity check column is the trick: matrix calculus identities are the scalar
rules with transposes arranged so shapes work out. Worked example - linear regression,
$L(w) = \|Xw - y\|^2$. Let $r = Xw - y$ (residuals). Chain rule:
sensitivity of $L$ to $r$ is $2r$; sensitivity of $r$ to $w$ is $X$; combine
(transposing to route the wiggle backward through the map):

$$\nabla_w L = 2 X^\top (Xw - y)$$

Shape check: $X^\top$ is $d \times n$, $r$ is $n \times 1$, result $d \times 1$ = shape
of $w$. ✓ **The gradient of a scalar loss always has the same shape as the thing you're
differentiating with respect to** - the single most useful debugging rule in matrix
calculus. Setting $\nabla_w L = 0$ gives the normal equations
$X^\top X w = X^\top y$, i.e. the closed-form least-squares solution.

### 2.4 Jacobians: the full wiggle table

When $f : \mathbb{R}^n \to \mathbb{R}^m$ (vector in, vector out - i.e., a layer), one
number can't describe sensitivity. The **Jacobian** is the $m \times n$ table of all of
them: $J_{ij} = \partial f_i / \partial x_j$ - "how much does output $i$ wiggle when I
wiggle input $j$?" To first order, $f(x + \delta) \approx f(x) + J\delta$: **the
Jacobian is the local linear approximation of the layer** - the matrix that the
function locally *is*. (Full circle to Part 1: nonlinear functions are locally linear
maps.)

Chain rule for vectors: Jacobians **multiply**: $J_{f \circ g}(x) = J_f(g(x))\, J_g(x)$.
Backprop propagates the row vector $\partial L / \partial \text{output}$ through each
layer by multiplying by that layer's Jacobian (a *vector–Jacobian product* - the
Jacobian itself is usually never materialized; for a hidden layer of width 4096 it
would be 4096², per example). Examples worth knowing cold:

- Elementwise ReLU: Jacobian is diagonal, entries $\mathbb{1}[x_i > 0]$ - backprop
  just masks the upstream gradient.
- Linear layer $f(x) = Wx$: Jacobian w.r.t. $x$ is exactly $W$; backprop multiplies
  the upstream gradient by $W^\top$.
- Softmax: $J_{ij} = p_i(\delta_{ij} - p_j)$ - dense, which is why softmax is always
  fused with cross-entropy (see exercises: the fused gradient collapses to $p - y$).

Vanishing/exploding gradients, restated precisely: the backward signal is a product of
many Jacobians; if their singular values are systematically $< 1$ or $> 1$, the product
shrinks or blows up exponentially with depth. Residual connections fix this by making
each layer's Jacobian $\approx I + \text{small}$, so products stay near $I$.

---

## Part 3: Probability & statistics

### 3.1 Random variables, expectation, variance - from first principles

A **random variable** is just a number attached to a random outcome: die roll $X \in
\{1..6\}$, each with probability $1/6$. The **expectation** is the probability-weighted
average - the long-run mean if you repeat forever:

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

Two properties you use constantly: **linearity**,
$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$, which holds *even when $X, Y$
are dependent* (it's just rearranging a weighted sum) - this is what makes so many ML
proofs one-liners. And $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ only under
independence.

**Variance** is the expected squared distance from the mean - the natural "spread":

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mu^2$$

(The identity: expand $(X-\mu)^2 = X^2 - 2\mu X + \mu^2$, apply linearity, note
$\mathbb{E}[X] = \mu$.) Die: $\mathbb{E}[X^2] = 91/6 \approx 15.17$, so
$\mathrm{Var} = 15.17 - 12.25 \approx 2.92$. Scaling: $\mathrm{Var}(aX) = a^2
\mathrm{Var}(X)$ (squared units!), and for **independent** variables variances add:
$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$. That last fact is the seed
of the CLT, of $1/\sqrt{n}$ error bars, and of every variance-scaling weight-init
scheme (Xavier/He: pick weight variance so activations keep unit variance layer to
layer).

### 3.2 Bayes rule - derived, not memorized

Conditional probability is a *definition*: restrict attention to the world where $B$
happened, and renormalize:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Write it twice: $P(A \cap B) = P(A \mid B) P(B)$ and also $P(A \cap B) = P(B \mid A)
P(A)$ - same quantity, two factorizations. Set them equal, divide by $P(B)$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

That's the whole derivation - two lines from the definition. The point of Bayes rule is
that it converts **likelihood** ("probability of the evidence given the hypothesis",
often easy to model) into the **posterior** ("probability of the hypothesis given the
evidence", the thing you actually want), weighted by the **prior** $P(A)$.

The classic trap (run it with raw counts, as in `probability_demos.ipynb`): a disease has
prevalence 1%, the test has 99% sensitivity and 95% specificity. You test positive  - 
what's the chance you're sick? Take 10,000 people: 100 sick → 99 true positives;
9,900 healthy → 5% false positive rate → 495 false positives. Positive tests: $99 +
495 = 594$; sick among them: $99 / 594 \approx 16.7\%$. Not 99%. The prior (rarity of
the disease) dominates because false positives come from a *huge* pool. In ML terms:
this is precision under class imbalance - exactly why accuracy is a useless metric for
rare-event detection and why a 99%-accurate fraud model can be garbage.

### 3.3 MLE, and why minimizing cross-entropy IS maximum likelihood

**Maximum likelihood estimation**: given data and a model family $p_\theta$, choose
the $\theta$ that makes the observed data most probable:

$$\hat\theta = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i \mid x_i)
             = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y_i \mid x_i)$$

(Log is monotonic so the argmax is unchanged; it turns the product into a sum  - 
numerically stable and differentiable term by term.)

Now the punchline. A classifier with softmax output assigns probabilities
$p_\theta(k \mid x_i)$ to each class. The negative log-likelihood of the data is:

$$-\sum_i \log p_\theta(y_i \mid x_i) = -\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)$$

where $y_{ik}$ is the one-hot label (the inner sum has one surviving term - the
correct class). The right-hand side is *literally the definition of the cross-entropy
loss*. So: **minimizing cross-entropy = maximizing the log-probability the model
assigns to the true labels.** Cross-entropy isn't an arbitrary "good loss"; it's MLE
for a categorical output. Same story elsewhere: MSE is MLE under a Gaussian noise
model ($-\log$ of a Gaussian density is a squared error plus constants), and L1 loss
is MLE under Laplace noise. When an interviewer asks "why this loss?", the deep answer
is always "it's the negative log-likelihood under this noise/output model."

Simplest concrete MLE (verified numerically in `probability_demos.ipynb`): coin with
unknown $p$, observe $h$ heads in $n$ flips. $\log L(p) = h \log p + (n - h)\log(1-p)$.
Differentiate, set to zero: $h/p = (n-h)/(1-p) \Rightarrow \hat p = h/n$. The empirical
frequency - MLE formalizes the obvious.

### 3.4 Bernoulli and Gaussian - where they come from

**Bernoulli($p$)**: the atom of randomness - one binary event. $P(X=1) = p$,
$\mathbb{E}[X] = p$, $\mathrm{Var} = p(1-p)$ (maximal at $p = 1/2$: a fair coin is the
most unpredictable coin). Sum $n$ independent Bernoullis → **Binomial**. Every
binary-classification output and every dropout mask is a Bernoulli.

**Gaussian** $\mathcal{N}(\mu, \sigma^2)$, density $\frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. It is not chosen for convenience alone; it *emerges*:

1. **CLT** (next section): sums of many small independent effects are Gaussian  - 
   which is why measurement noise so often is.
2. **Maximum entropy**: among all distributions with a given mean and variance, the
   Gaussian assumes the least extra structure - the honest default.
3. Its $-\log$ density is quadratic - the bridge from "Gaussian noise" to
   "least-squares loss" (§3.3).

### 3.5 Bias and variance of estimators

An estimator $\hat\theta$ is a formula applied to a random sample, so it's a random
variable itself. Two failure modes:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

Bias: wrong *on average* (systematic). Variance: right on average but *jumpy* across
samples. They combine exactly (derivation: add and subtract $\mathbb{E}[\hat\theta]$
inside the square, cross term vanishes):

$$\mathbb{E}[(\hat\theta - \theta)^2] = \mathrm{Bias}^2 + \mathrm{Var}$$

Canonical example: sample variance with $\frac{1}{n}$ is biased low - the sample mean
$\bar x$ chases the data, so deviations from $\bar x$ are systematically smaller than
deviations from the true $\mu$; the exact correction factor is $\frac{n-1}{n}$, fixed
by **Bessel's correction** $\frac{1}{n-1}$. The ML version of the tradeoff: a linear
model is high-bias/low-variance (systematically misses curvature, but stable across
datasets); a deep net or unpruned tree is low-bias/high-variance (can fit anything,
including this particular sample's noise). Regularization, more data, and ensembling
(bagging averages away variance) are all moves on this dial. Test error decomposes as
$\text{Bias}^2 + \text{Variance} + \text{irreducible noise}$.

### 3.6 Central limit theorem - the $\sqrt{n}$ intuition

Average $n$ i.i.d. draws with mean $\mu$, variance $\sigma^2$:
$\bar X_n = \frac1n \sum X_i$. Two facts, both from §3.1:

- $\mathbb{E}[\bar X_n] = \mu$ (linearity) - unbiased.
- $\mathrm{Var}(\bar X_n) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$
  (independence → variances add; the $\frac1n$ comes out squared).

So the spread of the mean shrinks like $\sigma/\sqrt{n}$ - the **law of large
numbers** with a rate. To halve your error bar you need 4× the data. The CLT adds the
shape: not only does $\bar X_n$ concentrate, its distribution becomes **Gaussian**,
*regardless of the original distribution*:

$$\frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

Intuition for the shape: summing convolves distributions, and convolution repeatedly
smooths - idiosyncratic features (skew, bumps) wash out, and the Gaussian is the fixed
point of this smoothing (only skew/kurtosis die slowly; mean and variance are
preserved by design of the normalization). Start with something as lopsided as an
exponential distribution: means of 100 draws are already visibly symmetric and bell-
shaped (see `probability_demos.ipynb`, which tracks the skewness → 0 and std →
$\sigma/\sqrt{n}$ numerically).

Why care in ML: minibatch gradients are sample means of per-example gradients (SGD
noise is $\propto 1/\sqrt{\text{batch size}}$); test-set accuracy is a sample mean, so
its error bar is $\sqrt{p(1-p)/n}$ - with $n = 1000$ test examples, accuracy has a
±1.5% standard error, i.e. a "0.5% improvement" is pure noise; A/B tests and
confidence intervals all lean on CLT normality.

---

## Common interview questions

**Q1. Why does the dot product measure similarity?**
Because $a \cdot b = \|a\|\|b\|\cos\theta$: for fixed lengths it's maximal when the
vectors align, zero when orthogonal, negative when opposed. Normalized, it's cosine
similarity; unnormalized, it's the projection of one vector on the other scaled by
its length - "how much of $a$ points along $b$."

**Q2. What happens if you stack linear layers with no nonlinearity?**
$W_2(W_1 x) = (W_2 W_1) x$ - the composition of linear maps is a single linear map, so
depth adds no expressive power. Nonlinearities are what break this collapse.

**Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?**
Directions a matrix scales without rotating ($Av = \lambda v$); the top eigenvectors of
the data covariance matrix are the PCA axes (directions of maximal variance).

**Q4. What's the difference between eigendecomposition and SVD?**
Eigendecomposition needs a square matrix (and orthogonal eigenvectors only if
symmetric); SVD ($U\Sigma V^\top$) exists for *any* rectangular matrix and reads as
rotate–stretch–rotate. They connect via $A^\top A = V \Sigma^2 V^\top$. Truncated SVD
gives the best low-rank approximation (basis of PCA, recommenders, LoRA intuition).

**Q5. Why does L1 regularization give sparsity but L2 doesn't?**
The L1 ball is a diamond with corners on the coordinate axes; the constrained optimum
tends to land on a corner where some coordinates are exactly 0. Equivalently, L1's
gradient is a constant-magnitude push toward 0 that can win at $w=0$, while L2's push
$\propto w$ fades near 0, shrinking weights without zeroing them.

**Q6. Explain backprop in two sentences.**
Forward pass computes and caches all intermediates; backward pass walks the graph in
reverse applying the chain rule - multiply the upstream gradient by each node's local
derivative, summing where paths merge. It's reverse-mode autodiff, chosen because one
backward sweep gets the gradient w.r.t. *all* parameters at roughly the cost of one
forward pass.

**Q7. What is a Jacobian and what's its role in backprop?**
The $m \times n$ matrix of all partials $\partial f_i / \partial x_j$ - the local
linear approximation of a vector-valued function. Backprop propagates gradients via
vector–Jacobian products; the full Jacobian is rarely materialized.

**Q8. Why do gradients vanish or explode in deep/recurrent nets?**
The backward signal is a product of per-layer Jacobians; if their singular values are
consistently below/above 1, the product decays/grows exponentially with depth.
Fixes: residual connections (Jacobian ≈ $I$), careful init, normalization layers,
gradient clipping, gating (LSTM).

**Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?**
$\nabla_w = 2X^\top(Xw - y)$; setting it to zero gives the normal equations
$X^\top X w = X^\top y$, the closed-form least-squares solution.

**Q10. Derive Bayes rule.**
From the definition $P(A|B) = P(A\cap B)/P(B)$: the joint factors both ways,
$P(A|B)P(B) = P(B|A)P(A)$; divide by $P(B)$ to get
$P(A|B) = P(B|A)P(A)/P(B)$.

**Q11. Why is minimizing cross-entropy the same as maximum likelihood?**
For a probabilistic classifier, the data's negative log-likelihood is
$-\sum_i \log p_\theta(y_i|x_i)$, which with one-hot labels is exactly the
cross-entropy loss. Minimizing one is minimizing the other. Similarly MSE = MLE under
Gaussian noise, L1 = MLE under Laplace noise.

**Q12. A test is 99% accurate for a disease with 1% prevalence; you test positive.
Roughly what's the chance you're sick?**
Around 17%, not 99% (with 95% specificity): per 10,000 people, 99 true positives vs
495 false positives, so $99/594 \approx 0.167$. The false positives swamp the rare
true positives - same reason accuracy misleads on imbalanced classes.

**Q13. Why divide by $n-1$ in the sample variance?**
Deviations are measured from the sample mean, which itself chases the data, so
$\frac1n\sum(x_i - \bar x)^2$ underestimates $\sigma^2$ by exactly the factor
$\frac{n-1}{n}$; dividing by $n-1$ makes the estimator unbiased (Bessel's correction).

**Q14. Why does averaging reduce noise, and at what rate?**
For i.i.d. terms, variances add while the $1/n$ prefactor comes out squared:
$\mathrm{Var}(\bar X) = \sigma^2/n$, so the standard error shrinks as $1/\sqrt{n}$  - 
4× the data for half the error. CLT adds that the distribution of the mean is
approximately Gaussian regardless of the source distribution.

**Q15. Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?**
Standard error of accuracy is $\sqrt{p(1-p)/n} \approx \sqrt{0.09 \cdot 0.91/1000}
\approx 0.9$–$1.5\%$ per model - the 0.5% gap is within noise. You'd need a bigger
test set, paired comparison (McNemar), or multiple seeds to claim a real improvement.

## Exit checklist

- [ ] Cold, on paper: derive the projection of b onto a and explain dot product as similarity - 3 min
- [ ] State SVD as rotate → stretch → rotate and give the geometric meaning of eigenvectors/eigenvalues - 3 min
- [ ] Write the chain rule for a two-stage composition and give the Jacobian shape for f: R^n → R^m - 3 min
- [ ] Derive Bayes rule from the definition of conditional probability, then apply it to a base-rate problem - 4 min
- [ ] Show that minimizing cross-entropy is maximum likelihood for a Bernoulli model - 5 min
- [ ] Define bias and variance of an estimator and explain why the 1/n variance estimator is biased - 4 min
- [ ] Use standard error to decide whether a 0.5% accuracy gap between two models on n=1000 is real - 4 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
