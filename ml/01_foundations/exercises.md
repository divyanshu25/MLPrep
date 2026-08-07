# Foundations: exercises (easy → hard)

Do each one on paper before opening the solution. Every derivation should be
complete — if a step feels like hand-waving, that's the step to slow down on.

---

### Exercise 1 (easy): dot product geometry

Let $a = [3, 4]$ and $b = [4, 3]$. Compute $a \cdot b$, $\|a\|$, $\|b\|$, and the
angle between $a$ and $b$. Then give a unit vector orthogonal to $a$.

<details><summary>Solution</summary>

$a \cdot b = 3\cdot4 + 4\cdot3 = 24$. $\|a\| = \sqrt{9+16} = 5$, $\|b\| = 5$.

$$\cos\theta = \frac{a \cdot b}{\|a\|\|b\|} = \frac{24}{25} = 0.96
\;\Rightarrow\; \theta = \arccos(0.96) \approx 16.26°$$

Orthogonal to $a = [3,4]$: swap and negate one component, $[4, -3]$ (check:
$3\cdot4 + 4\cdot(-3) = 0$). Normalize by its length 5: $[0.8, -0.6]$.

Interview relevance: this is the entire mechanical core of cosine similarity and
attention scores — you should be able to do it in your head for small vectors.
</details>

---

### Exercise 2 (easy): stacked linear layers collapse

A network computes $y = W_2 (W_1 x)$ with no nonlinearity, where
$W_1 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$,
$W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$.
Find the single matrix $W$ with $y = Wx$, and explain in one sentence why this
means depth without nonlinearities is useless.

<details><summary>Solution</summary>

$$W = W_2 W_1 = \begin{bmatrix} 2 & 0 \\ 0 & 3\end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 0 & 1\end{bmatrix}
= \begin{bmatrix} 2 & 2 \\ 0 & 3 \end{bmatrix}$$

(Row $i$ of $W_2$ dotted with column $j$ of $W_1$.) Composition of linear maps is a
linear map, so any depth-$k$ stack of linear layers computes some single matrix
$W_k \cdots W_1$ — exactly the hypothesis class of a 1-layer linear model. The
nonlinearity between layers is what prevents this collapse and gives depth its
expressive power.
</details>

---

### Exercise 3 (easy): expectation and variance by hand

$X$ takes value $0$ with probability $0.5$, $1$ with probability $0.3$, $10$ with
probability $0.2$. Compute $\mathbb{E}[X]$ and $\mathrm{Var}(X)$. Then compute
$\mathbb{E}[3X + 2]$ and $\mathrm{Var}(3X + 2)$ *without* re-enumerating outcomes.

<details><summary>Solution</summary>

$\mathbb{E}[X] = 0(0.5) + 1(0.3) + 10(0.2) = 2.3$.

$\mathbb{E}[X^2] = 0(0.5) + 1(0.3) + 100(0.2) = 20.3$, so

$$\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = 20.3 - 5.29 = 15.01$$

By linearity of expectation: $\mathbb{E}[3X+2] = 3(2.3) + 2 = 8.9$.
Shifts don't change spread and scaling enters squared:
$\mathrm{Var}(3X+2) = 9 \cdot 15.01 = 135.09$.

Note how the rare large value ($10$ w.p. $0.2$) dominates the variance — the same
reason outliers dominate MSE.
</details>

---

### Exercise 4 (easy–medium): eigenvectors of a 2×2 by inspection

Find the eigenvalues and eigenvectors of
$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$, and describe geometrically what
$A$ does to the plane.

<details><summary>Solution</summary>

For symmetric 2×2 matrices with equal diagonal, guess the diagonal directions.
$v_1 = [1, 1]$: $Av_1 = [4, 4] = 4v_1$ → $\lambda_1 = 4$.
$v_2 = [1, -1]$: $Av_2 = [2, -2] = 2v_2$ → $\lambda_2 = 2$.

Systematic route: solve $\det(A - \lambda I) = (3-\lambda)^2 - 1 = 0$, so
$3 - \lambda = \pm 1$, $\lambda \in \{4, 2\}$. Plug back to get the eigenvectors.

Geometrically: $A$ stretches the plane by 4× along the $45°$ diagonal and by 2×
along the $-45°$ diagonal — no rotation in the eigenbasis. Note the eigenvectors are
orthogonal, as guaranteed for any symmetric matrix (spectral theorem). This is the
picture behind PCA: a covariance matrix stretches space most along its top
eigenvector.
</details>

---

### Exercise 5 (medium): gradient of $x^\top A x$

Derive $\nabla_x \, (x^\top A x)$ for a general (not necessarily symmetric) square
$A$, from components. Verify on $A = \begin{bmatrix} 2 & 1 \\ 0 & 3\end{bmatrix}$,
$x = [1, 2]^\top$.

<details><summary>Solution</summary>

Write it as a double sum: $f(x) = \sum_i \sum_j A_{ij} x_i x_j$. Differentiate w.r.t.
a particular coordinate $x_k$. A term contributes iff $i = k$ or $j = k$ (the term
$A_{kk} x_k^2$ hits both cases and correctly contributes twice via the product rule):

$$\frac{\partial f}{\partial x_k}
= \sum_j A_{kj} x_j + \sum_i A_{ik} x_i
= (Ax)_k + (A^\top x)_k$$

Stacking over $k$:

$$\nabla_x\, x^\top A x = (A + A^\top)\, x$$

which reduces to the familiar $2Ax$ only when $A = A^\top$ (and to $2ax$ in 1-D).

Check: $f(x) = x^\top A x$ with the given numbers: $Ax = [4, 6]^\top$, so
$f = 1\cdot4 + 2\cdot6 = 16$. And
$(A + A^\top)x = \begin{bmatrix} 4 & 1 \\ 1 & 6 \end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix} = [6, 13]^\top$.
Numerical check (see `gradients_from_scratch.py`): wiggling $x_1$ by $10^{-5}$
changes $f$ at rate $\approx 6$; wiggling $x_2$, at rate $\approx 13$. ✓

This form is everywhere: quadratic losses, Gaussians ($x^\top \Sigma^{-1} x$ in the
exponent), second-order Taylor expansions ($\frac12 \delta^\top H \delta$).
</details>

---

### Exercise 6 (medium): gradient of the least-squares loss

Derive $\nabla_w L$ for $L(w) = \|Xw - y\|^2$, state the closed-form minimizer, and
say when that closed form fails and what you do about it.

<details><summary>Solution</summary>

Expand: $L = (Xw - y)^\top (Xw - y) = w^\top X^\top X w - 2 y^\top X w + y^\top y$.

Differentiate term by term using the identities
$\nabla_w (w^\top M w) = (M + M^\top) w$ and $\nabla_w (b^\top w) = b$:

- $\nabla_w\, w^\top (X^\top X) w = 2 X^\top X w$ (since $X^\top X$ is symmetric),
- $\nabla_w\, (-2 y^\top X w) = -2 X^\top y$,
- constant term → $0$.

$$\nabla_w L = 2 X^\top X w - 2 X^\top y = 2 X^\top (Xw - y)$$

Set to zero → normal equations $X^\top X \, w^* = X^\top y$, so
$w^* = (X^\top X)^{-1} X^\top y$ when $X^\top X$ is invertible.

It fails when $X^\top X$ is singular (or nearly so): fewer samples than features
($n < d$), or collinear/duplicated features. Fixes: ridge regularization —
minimize $\|Xw - y\|^2 + \lambda \|w\|^2$, giving
$w^* = (X^\top X + \lambda I)^{-1} X^\top y$, always invertible for $\lambda > 0$ —
or the pseudo-inverse via SVD, or just gradient descent (which is also the only
practical option when $d$ is large, since inversion costs $O(d^3)$).
</details>

---

### Exercise 7 (medium): variance of the sample mean

Let $X_1, \dots, X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$, and
$\bar X = \frac{1}{n}\sum_i X_i$. Derive $\mathbb{E}[\bar X]$ and
$\mathrm{Var}(\bar X)$, stating exactly where independence is used. Then: how many
times more data halves the standard error?

<details><summary>Solution</summary>

Mean, by linearity of expectation (needs **no** independence):

$$\mathbb{E}[\bar X] = \frac1n \sum_{i=1}^n \mathbb{E}[X_i] = \frac1n \cdot n\mu = \mu$$

Variance: pull out the constant (squared!), then use that variances of
**independent** variables add — this is the only step needing independence
(uncorrelatedness suffices; the cross-covariance terms vanish):

$$\mathrm{Var}(\bar X) = \frac{1}{n^2} \mathrm{Var}\Big(\sum_i X_i\Big)
= \frac{1}{n^2} \sum_i \mathrm{Var}(X_i) = \frac{1}{n^2}\, n \sigma^2
= \frac{\sigma^2}{n}$$

Standard error $= \sigma/\sqrt{n}$: halving it requires **4×** the data.

Why interviewers love this: it is the quantitative core of "more data reduces
variance", of test-metric error bars, of minibatch-gradient noise
($\propto 1/\sqrt{\text{batch}}$), and of ensembling (averaging $k$ de-correlated
models divides the variance term by up to $k$).
</details>

---

### Exercise 8 (medium–hard): Bessel's correction

Show that $\hat\sigma^2 = \frac1n \sum_i (X_i - \bar X)^2$ has
$\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n} \sigma^2$, i.e. it is biased low, and
conclude why we divide by $n - 1$.

<details><summary>Solution</summary>

Key algebraic step — add and subtract $\mu$ inside each square:

$$\sum_i (X_i - \bar X)^2 = \sum_i \big((X_i - \mu) - (\bar X - \mu)\big)^2
= \sum_i (X_i - \mu)^2 - 2(\bar X - \mu)\sum_i (X_i - \mu) + n(\bar X - \mu)^2$$

But $\sum_i (X_i - \mu) = n(\bar X - \mu)$, so the middle term is
$-2n(\bar X - \mu)^2$ and:

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

Take expectations: $\mathbb{E}[(X_i - \mu)^2] = \sigma^2$ for each $i$, and
$\mathbb{E}[(\bar X - \mu)^2] = \mathrm{Var}(\bar X) = \sigma^2/n$ (Exercise 7). So

$$\mathbb{E}\Big[\sum_i (X_i - \bar X)^2\Big] = n\sigma^2 - n \cdot \frac{\sigma^2}{n}
= (n-1)\sigma^2
\;\Rightarrow\;
\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$$

Intuition for the bias: $\bar X$ is chosen by the data to sit in its middle, so
deviations from $\bar X$ are systematically smaller than deviations from the true
$\mu$ — we "spent" one degree of freedom estimating the mean. Dividing by $n-1$
instead of $n$ exactly cancels the factor, giving the unbiased sample variance.
</details>

---

### Exercise 9 (hard): softmax + cross-entropy gradient is $p - y$

Let $z \in \mathbb{R}^K$ be logits, $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ (softmax),
and $L = -\sum_k y_k \log p_k$ with one-hot $y$. Show that
$\frac{\partial L}{\partial z_k} = p_k - y_k$, deriving the softmax Jacobian on the way.

<details><summary>Solution</summary>

**Step 1: softmax Jacobian.** Let $S = \sum_j e^{z_j}$, so $p_i = e^{z_i}/S$.

Case $i = k$ (quotient rule; note $\partial S/\partial z_k = e^{z_k}$):

$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2}
= p_i - p_i^2 = p_i(1 - p_i)$$

Case $i \ne k$ (numerator doesn't depend on $z_k$):

$$\frac{\partial p_i}{\partial z_k} = e^{z_i} \cdot \frac{-e^{z_k}}{S^2} = -p_i p_k$$

Both cases in one formula: $\frac{\partial p_i}{\partial z_k} = p_i(\delta_{ik} - p_k)$.

**Step 2: chain rule through the loss.** Since $L = -\sum_i y_i \log p_i$,
$\frac{\partial L}{\partial p_i} = -y_i / p_i$. Then:

$$\frac{\partial L}{\partial z_k}
= \sum_i \frac{\partial L}{\partial p_i} \frac{\partial p_i}{\partial z_k}
= \sum_i \Big(-\frac{y_i}{p_i}\Big) p_i (\delta_{ik} - p_k)
= -\sum_i y_i (\delta_{ik} - p_k)
= -y_k + p_k \sum_i y_i$$

With one-hot $y$, $\sum_i y_i = 1$, hence

$$\boxed{\frac{\partial L}{\partial z_k} = p_k - y_k}$$

Sanity check with numbers: $z = [2, 0]$, true class 1. Then
$p \approx [0.881, 0.119]$, gradient $\approx [-0.119, 0.119]$: push the correct
logit up, the wrong one down, in proportion to how wrong we are. When $p = y$ the
gradient is exactly 0.

Why it matters: this is the cleanest gradient in deep learning — "prediction minus
target" — and it is *why* frameworks fuse softmax with cross-entropy: the dense
softmax Jacobian and the $1/p_i$ (numerically explosive as $p_i \to 0$) cancel
analytically, leaving something simple and stable. Logistic regression's
$(\sigma(z) - y)$ is the $K = 2$ special case.
</details>

---

### Exercise 10 (hard): why high dimensions make points equidistant

Let $x, y \in \mathbb{R}^d$ have i.i.d. coordinates with mean 0 and variance 1
(e.g. standard Gaussians), independent of each other. Show that
$\|x - y\|^2$ has mean $2d$ and standard deviation $O(\sqrt{d})$, so the *relative*
spread of distances shrinks like $1/\sqrt{d}$. State two practical consequences
for ML.

<details><summary>Solution</summary>

Write the squared distance as a sum of $d$ i.i.d. terms:

$$\|x - y\|^2 = \sum_{i=1}^d (x_i - y_i)^2 = \sum_{i=1}^d Z_i^2,
\qquad Z_i := x_i - y_i$$

Each $Z_i$ has mean 0 and variance $\mathrm{Var}(x_i) + \mathrm{Var}(y_i) = 2$
(independence), so $\mathbb{E}[Z_i^2] = 2$ and by linearity:

$$\mathbb{E}\big[\|x - y\|^2\big] = 2d$$

The terms $Z_i^2$ are i.i.d. with some finite variance $c$ (for Gaussians,
$Z_i \sim \mathcal{N}(0,2)$, so $Z_i^2$ has variance $2 \cdot 2^2 = 8$). Variances
of independent terms add:

$$\mathrm{Var}\big(\|x - y\|^2\big) = c\,d
\;\Rightarrow\; \mathrm{std} = \sqrt{c d} = O(\sqrt d)$$

So the ratio of spread to typical value is

$$\frac{\mathrm{std}}{\mathrm{mean}} = \frac{O(\sqrt d)}{2d} = O\!\Big(\frac{1}{\sqrt d}\Big) \to 0$$

By the same argument as the law of large numbers, $\frac{1}{d}\|x-y\|^2$
concentrates at 2: in $d = 10{,}000$ dimensions, essentially **every pair of random
points is at distance $\approx \sqrt{2d}$**, give or take about 1%. "Nearest" and
"farthest" neighbors are nearly the same distance away.

Consequences:

1. **k-NN and distance-based methods degrade**: distance contrast vanishes, so
   nearest-neighbor labels carry little signal in raw high-dimensional spaces;
   you need learned embeddings that put semantic structure back into the geometry
   (this is what metric learning / contrastive training does).
2. **Intuition from 2-D/3-D fails**: e.g. random high-dimensional vectors are
   nearly orthogonal ($\cos\theta \approx 0$ with high probability — same
   concentration argument applied to the dot product $\sum_i x_i y_i$, mean 0,
   std $\sqrt d$, vs lengths $\approx \sqrt d$), volume concentrates in thin shells,
   and Gaussian samples live near a sphere of radius $\sqrt d$, not near the mode.
   This near-orthogonality is also why you can cram huge numbers of
   almost-uncorrelated feature directions into embedding spaces.
</details>
