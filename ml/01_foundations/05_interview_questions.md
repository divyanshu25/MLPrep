# Interview questions & exit checklist

Fifteen standard questions, grouped by topic. Work them as a drill: read the
question, answer out loud or on paper as if the interviewer just asked it,
and only then open the model answer. Any answer you could not produce cold
goes into `practice/queue.tsv`.

Each model answer is what to say in about a minute, plus the one line of
math or numbers that backs it. The full derivations live in the chapters.

## Linear algebra

### Q1. Why does the dot product measure similarity?

<details><summary>Model answer</summary>

$a \cdot b = \|a\|\,\|b\|\cos\theta$. With the lengths fixed, it is largest
when the vectors point the same way, zero when they are perpendicular, and
negative when they point in opposite directions. Dividing by both lengths
gives cosine similarity.

Check: $(3,4)\cdot(4,3) = 24 = 5 \cdot 5 \cdot 0.96$, and
$(3,4)\cdot(4,-3) = 0$.

**If asked:**

- *Where does the cosine come from?* Expand
  $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\,a \cdot b$ and compare with the law
  of cosines.
- *Why normalize?* The raw dot product grows with length, so long vectors
  score high against everything.
</details>

---

### Q2. What happens if you stack linear layers with no nonlinearity?

<details><summary>Model answer</summary>

They collapse into one linear layer, so depth adds nothing the model can
represent. A nonlinearity between the layers prevents the collapse.

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2)$$

**If asked:** *Something no linear model can do?* XOR. Positives
$(0,1), (1,0)$ require $w_1 + w_2 + 2b > 0$; negatives $(0,0), (1,1)$
require $w_1 + w_2 + 2b < 0$. One ReLU unit fixes it.
</details>

---

### Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?

<details><summary>Model answer</summary>

An eigenvector of $A$ is a direction that $A$ only scales: $Av = \lambda v$.
The eigenvectors of the data covariance matrix are the PCA directions, and
each eigenvalue is the variance of the data along its eigenvector.

Why: maximizing the projected variance $u^\top \Sigma u$ subject to
$\|u\| = 1$ gives $\Sigma u = \lambda u$ (Lagrange multiplier), and the
variance achieved is $u^\top \Sigma u = \lambda$.

**If asked:** *Another place?* Gradient descent on a quadratic converges
only if the learning rate satisfies $\eta < 2/\lambda_{\max}$ of the Hessian.
</details>

---

### Q4. What's the difference between eigendecomposition and SVD?

<details><summary>Model answer</summary>

Eigendecomposition $A = Q\Lambda Q^{-1}$ needs a square matrix and fails
even for some square ones. SVD $A = U\Sigma V^\top$ exists for every matrix,
with orthogonal $U, V$ and nonnegative singular values: rotate, stretch,
rotate. They meet in $A^\top A = V\Sigma^2 V^\top$, so the singular values
of $A$ are the square roots of the eigenvalues of $A^\top A$. Keeping the
top $k$ singular values gives the best rank-$k$ approximation, which is what
PCA, matrix factorization and LoRA rely on.

**If asked:**

- *A square matrix with no eigendecomposition?*
  $\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$: only eigenvalue 0, only
  one eigenvector direction.
- *When are they the same?* For symmetric positive semidefinite matrices,
  such as covariance matrices.
</details>

---

### Q5. Why does L1 regularization give sparsity but L2 doesn't?

<details><summary>Model answer</summary>

Near zero, the L1 penalty $\lambda|w|$ still has slope $\lambda$, while the
L2 penalty $\lambda w^2$ has slope $2\lambda w$, which goes to 0. So with
L1, $w = 0$ stays the minimum unless the loss's slope there exceeds
$\lambda$; with L2, any nonzero loss slope moves the minimum off zero.
Geometrically, the L1 constraint region is a diamond with corners on the
axes, and the loss contours usually touch it first at a corner, where some
weights are exactly 0.

With one weight and loss $(w - a)^2$:

$$\text{L2: } w^* = \frac{a}{1 + \lambda} \qquad\qquad \text{L1: } w^* = \operatorname{sign}(a)\,\max\!\big(0,\ |a| - \tfrac{\lambda}{2}\big)$$

At $\lambda = 1$, $a = 0.3$: L2 gives 0.15, L1 gives exactly 0. (Derivation
and figure: module 2, §3.2.)

**If asked:**

- *Bayesian view?* L2 is a Gaussian prior on the weights, L1 a Laplace
  prior.
- *Does SGD with an L1 term give exact zeros?* No, subgradient steps
  oscillate around 0. Exact zeros need a proximal (soft-threshold) step.
</details>

## Calculus

### Q6. Explain backprop.

<details><summary>Model answer</summary>

The forward pass computes every node and stores its value. The backward
pass starts from $\partial L/\partial L = 1$ and visits the nodes in
reverse: each node multiplies the gradient arriving at its output by its
local derivative and adds the result into each input's gradient (adds,
because a variable used in several places gets one term per path). This is
reverse-mode autodiff. With a single scalar loss, one backward pass gives
the gradient for every parameter at about twice the cost of the forward
pass, where finite differences would need one pass per parameter.

Check: $L = (wx + b)^2$ at $w = 2, x = 3, b = 1$. Forward $z = 7$.
Backward $\partial L/\partial z = 2z = 14$, then
$\partial L/\partial w = 14 \cdot x = 42$ and $\partial L/\partial b = 14$.
The step for $w$ used the stored $x$ and $z$.

**If asked:**

- *Why about 2× forward?* For $Y = XW$ with upstream gradient $G$, backward
  is two matrix multiplies of the same size, $X^\top G$ and $G W^\top$. A
  training step is then about 3× forward: the $6N$ in $C = 6ND$.
- *What limits it?* Memory for the stored activations. Checkpointing
  recomputes them instead of storing them.
</details>

---

### Q7. What is a Jacobian and what's its role in backprop?

<details><summary>Model answer</summary>

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the $m \times n$
matrix $J_{ij} = \partial f_i/\partial x_j$, the linear approximation
$f(x + \delta) \approx f(x) + J\delta$. Backprop only needs the
vector-Jacobian product $\nabla_x L = J^\top g$, where
$g = \partial L/\partial f$, and each layer computes it without building
$J$.

Example: for ReLU, $J$ is diagonal with 0/1 entries, so the backward step is
$g \odot \mathbb{1}[x > 0]$. On 4,096 activations the full $J$ would have
16.8 million entries, only 4,096 of them nonzero.
</details>

---

### Q8. Why do gradients vanish or explode in deep/recurrent nets?

<details><summary>Model answer</summary>

The backward pass multiplies the gradient by one Jacobian per layer, so the
gradient at the first layer is a product of one factor per layer. Factors
consistently below 1 shrink it exponentially with depth, factors above 1
grow it: $0.9^{50} \approx 0.005$, $1.1^{50} \approx 117$. A sigmoid's
derivative is at most 0.25, so ten sigmoid layers multiply the gradient by
at most $10^{-6}$. An RNN applies the same $W$ at every time step, so the
effect compounds over the sequence.

Fixes:

- residual connections: $J = I + \partial F/\partial h$ keeps a path with factor 1
- Xavier/Kaiming initialization and normalization layers: factors near 1
- ReLU instead of sigmoid
- gradient clipping (explosion only)
- LSTM gating: a forget gate near 1 carries the gradient along the cell state
</details>

---

### Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?

<details><summary>Model answer</summary>

$\nabla_w \|Xw - y\|^2 = 2X^\top(Xw - y)$. Setting it to zero gives the
normal equations $X^\top X w = X^\top y$, so
$w = (X^\top X)^{-1} X^\top y$ when $X^\top X$ is invertible.

Why: expand to $w^\top X^\top X w - 2y^\top X w + y^\top y$, then use
$\nabla(w^\top A w) = 2Aw$ and $\nabla(c^\top w) = c$.

**If asked:**

- *Geometric meaning?* $X^\top(y - Xw) = 0$: the residual is perpendicular
  to every column of $X$, so $Xw$ is the projection of $y$ onto their span.
- *What if $X^\top X$ is singular?* Collinear features. Ridge solves
  $(X^\top X + \lambda I)\,w = X^\top y$, invertible for any $\lambda > 0$.
</details>

## Probability & statistics

### Q10. Derive Bayes rule.

<details><summary>Model answer</summary>

Write the joint probability both ways and divide by $P(B)$:

$$P(A \mid B)\,P(B) = P(B \mid A)\,P(A) \quad\Longrightarrow\quad P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

When $P(B)$ is not given, expand it:
$P(B) = P(B \mid A)P(A) + P(B \mid \lnot A)P(\lnot A)$. Q12 applies it.
</details>

---

### Q11. Why is minimizing cross-entropy the same as maximum likelihood?

<details><summary>Model answer</summary>

Maximum likelihood maximizes $\prod_i p_\theta(y_i \mid x_i)$. Taking
$-\log$ gives $\sum_i -\log p_\theta(y_i \mid x_i)$, and for a classifier
$-\log p_\theta(y \mid x)$ is exactly the cross-entropy loss. For binary
labels with $\hat y = p_\theta(1 \mid x)$ it is
$-[\,y\log\hat y + (1 - y)\log(1 - \hat y)\,]$. The same argument gives MSE
under Gaussian noise and absolute error under Laplace noise.

**If asked:** *Information-theory view?* The expected loss is
$H(q, p_\theta) = H(q) + \mathrm{KL}(q \,\|\, p_\theta)$, and $H(q)$ does not
depend on $\theta$, so training minimizes the KL divergence from the data
to the model (Probability §3.7).
</details>

---

### Q12. Positive test, rare disease

A test detects the disease in 99% of sick people and gives a false positive
for 5% of healthy people. 1% of the population has the disease. You test
positive. Roughly what's the chance you're sick?

<details><summary>Model answer</summary>

About 17%. Out of 10,000 people, 100 are sick and 99 of them test positive;
9,900 are healthy and 5% of them, 495, test positive. So
$P(\text{sick} \mid +) = 99/(99 + 495) \approx 0.17$. The false positives
from the large healthy group outnumber the true positives from the small
sick group, which is also why accuracy misleads on imbalanced classes.

**If asked:**

- *99% accurate for both groups?* $99/(99 + 99) = 50\%$.
- *A second independent positive test?* With prior 0.167:
  $0.99 \cdot 0.167 / (0.99 \cdot 0.167 + 0.05 \cdot 0.833) \approx 0.80$.
</details>

---

### Q13. Why divide by $n-1$ in the sample variance?

<details><summary>Model answer</summary>

The deviations are measured from $\bar X$, which is computed from the same
sample and minimizes the sum of squared deviations. So
$\sum_i (X_i - \bar X)^2$ is smaller than $\sum_i (X_i - \mu)^2$, and its
expected value is $(n - 1)\sigma^2$ instead of $n\sigma^2$. Dividing by
$n - 1$ makes the estimate unbiased.

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

Taking expectations, with $\mathbb{E}[(\bar X - \mu)^2] = \sigma^2/n$:

$$\mathbb{E}\Big[\sum_i (X_i - \bar X)^2\Big] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n - 1)\,\sigma^2$$

**If asked:** *Is $s$ then unbiased for $\sigma$?* No. The square root is
concave, so $\mathbb{E}[s] \le \sigma$ (Jensen).
</details>

---

### Q14. Why does averaging reduce noise, and at what rate?

<details><summary>Model answer</summary>

For independent draws, $\mathrm{Var}(\bar X) = \frac{1}{n^2} \cdot n\sigma^2
= \sigma^2/n$: the $1/n$ comes out squared, and independence lets the
variances add. So the standard error is $\sigma/\sqrt{n}$, and halving it
takes 4× the data. The CLT adds that $\bar X$ is approximately normal for
large $n$.

**If asked:** *Correlated draws?* With pairwise correlation $\rho$,
$\mathrm{Var}(\bar X) = \sigma^2/n + \frac{n-1}{n}\rho\sigma^2 \to
\rho\sigma^2$. Averaging stops helping, which is why random forests
decorrelate their trees.
</details>

---

### Q15. Is a 0.5% accuracy gain real?

Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?

<details><summary>Model answer</summary>

Probably not. One accuracy on 1,000 examples has standard error
$\sqrt{0.91 \cdot 0.09/1000} \approx 0.9$ points, larger than the 0.5-point
gap. Both models are scored on the same examples, so use a paired test
(McNemar): if $b$ examples are right only for the new model and $c$ only for
the old one, then $b - c = 5$ and $z = (b - c)/\sqrt{b + c}$. Reaching
$z \ge 1.96$ needs $b + c \le 6$, so the gap is real only if the two models
disagree on almost nothing.

**If asked:** *What else would you do?* Retrain with several seeds (seed
spread is often about 0.5 points) and use a larger test set (10,000
examples gives 0.29 points).
</details>

## Exit checklist

Seven timed derivations, about 26 minutes total. Cold, on paper, before
moving on - a miss goes into `practice/queue.tsv`.

- [ ] Derive the projection of $b$ onto $a$ and explain the dot product as similarity - 3 min
- [ ] State SVD as rotate → stretch → rotate and give the geometric meaning of eigenvectors/eigenvalues - 3 min
- [ ] Write the chain rule for a two-stage composition and give the Jacobian shape for $f: \mathbb{R}^n \to \mathbb{R}^m$ - 3 min
- [ ] Derive Bayes rule from the definition of conditional probability, then apply it to a base-rate problem - 4 min
- [ ] Show that minimizing cross-entropy is maximum likelihood for a Bernoulli model - 5 min
- [ ] Define bias and variance of an estimator and explain why the 1/n variance estimator is biased - 4 min
- [ ] Use standard error to decide whether a 0.5% accuracy gap between two models on n=1000 is real - 4 min
