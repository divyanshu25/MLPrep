# Interview questions & exit checklist

Fifteen standard questions, grouped by topic. Work them as a drill: read the
question, answer out loud or on paper as if the interviewer just asked it,
and only then open the model answer. Any answer you could not produce cold
goes into `practice/queue.tsv`.

Each model answer starts from a tiny concrete case, often a few lines of
numpy you could type into a notebook, and states the general rule after.
That is also a good way to answer in the room: show the small case, then
generalize. The full derivations live in the chapters.

## Linear algebra

### Q1. Why does the dot product measure similarity?

<details><summary>Model answer</summary>

Take $a = (3, 4)$ and $b = (4, 3)$. Then $a \cdot b = 3 \cdot 4 + 4 \cdot 3
= 24$. Both vectors have length 5, so if they pointed in exactly the same
direction the dot product would be $5 \cdot 5 = 25$. It is 24, a ratio of
0.96: nearly the same direction. Now take $c = (4, -3)$:
$a \cdot c = 12 - 12 = 0$, and the vectors are perpendicular.

In general $a \cdot b = \|a\|\,\|b\|\cos\theta$: largest when the vectors
align, zero when perpendicular, negative when opposite. Dividing out the
lengths leaves $\cos\theta$, the cosine similarity:

```python
a @ b / (np.linalg.norm(a) * np.linalg.norm(b))   # 0.96
```

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

Write two layers with no activation between them and expand:

```python
h = W1 @ x + b1
y = W2 @ h + b2        # = (W2 @ W1) @ x + (W2 @ b1 + b2)
```

`W2 @ W1` is just another matrix and `W2 @ b1 + b2` is just another vector,
so the two layers compute exactly what one layer can. The same step
collapses any number of layers. Depth only adds functions the model can
represent when a nonlinearity sits between the layers.

**If asked:** *Something no linear model can do?* XOR. Positives
$(0,1), (1,0)$ require $w_1 + w_2 + 2b > 0$; negatives $(0,0), (1,1)$
require $w_1 + w_2 + 2b < 0$. One ReLU unit fixes it.
</details>

---

### Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?

<details><summary>Model answer</summary>

Multiply a few vectors by $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$:

```python
A @ [1, 1]     # [3, 3]    same direction, 3x longer
A @ [1, -1]    # [1, -1]   same direction, 1x
A @ [1, 0]     # [2, 1]    direction changed
```

The first two are eigenvectors: directions that $A$ only scales,
$Av = \lambda v$, with eigenvalues 3 and 1.

In ML: the eigenvectors of the data covariance matrix $\Sigma$ are the PCA
directions, and each eigenvalue is the variance of the data along its
eigenvector. The reason: maximizing the projected variance
$u^\top \Sigma u$ subject to $\|u\| = 1$ gives $\Sigma u = \lambda u$
(Lagrange multiplier), and the variance achieved is $\lambda$.

**If asked:** *Another place?* Gradient descent on a quadratic converges
only if the learning rate satisfies $\eta < 2/\lambda_{\max}$ of the Hessian.
</details>

---

### Q4. What's the difference between eigendecomposition and SVD?

<details><summary>Model answer</summary>

$Av = \lambda v$ needs $v$ and $Av$ in the same space, so eigenvectors only
exist for square matrices, and even some square matrices do not have
enough of them ($\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ has one
eigenvector direction). SVD works for every matrix: $A = U\Sigma V^\top$
rotates ($V^\top$), stretches along the axes ($\Sigma$), and rotates again
($U$). The two meet in $A^\top A = V\Sigma^2 V^\top$: the singular values of
$A$ are the square roots of the eigenvalues of $A^\top A$.

Why it matters: keeping the top $k$ singular values gives the best rank-$k$
approximation of $A$, which is what PCA, matrix factorization and LoRA
rely on.

```python
U, S, Vt = np.linalg.svd(A)
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]     # best rank-k approximation
```

**If asked:** *When are they the same?* For symmetric positive
semidefinite matrices, such as covariance matrices.
</details>

---

### Q5. Why does L1 regularization give sparsity but L2 doesn't?

<details><summary>Model answer</summary>

Take one weight whose unregularized best value is $a$, and minimize
$(w - a)^2$ plus each penalty. Both have closed forms:

```python
w_l2 = a / (1 + lam)                            # L2: divide
w_l1 = np.sign(a) * max(0, abs(a) - lam / 2)    # L1: subtract, clip at 0
```

With `lam = 1` and `a = 0.3`: L2 gives 0.15, L1 gives exactly 0.0.

The reason is the slope of each penalty near zero. The L2 penalty
$\lambda w^2$ has slope $2\lambda w$, which goes to 0 as $w$ does, so any
nonzero slope of the loss moves the minimum off zero. The L1 penalty
$\lambda|w|$ has slope $\lambda$ all the way down to 0, so when the loss's
slope at zero ($2|a|$) is smaller than $\lambda$, the minimum stays at
exactly $w = 0$. In 2D, the L1 constraint region is a diamond with corners
on the axes, and the loss contours usually touch it first at a corner
(figure in module 2, §3.2).

**If asked:**

- *Bayesian view?* L2 is a Gaussian prior on the weights, L1 a Laplace
  prior.
- *Does SGD with an L1 term give exact zeros?* No, subgradient steps
  oscillate around 0. Exact zeros need the soft-threshold step above
  applied after each gradient step (a proximal step).
</details>

## Calculus

### Q6. Explain backprop.

<details><summary>Model answer</summary>

Take $L = (wx + b)^2$ with $w = 2$, $x = 3$, $b = 1$. The question is: if
$w$ increases by a tiny amount, how much does $L$ change? That rate is
$\partial L/\partial w$, and backprop computes it for every input at once.

The forward pass computes each intermediate value and keeps it:

```python
z = w * x + b      # 7
L = z ** 2         # 49
```

The backward pass starts at the end with $\partial L/\partial L = 1$ and
walks back, multiplying by each operation's local derivative (the chain
rule):

```python
dL = 1.0
dz = dL * 2 * z    # 14   local derivative of z**2 is 2z
dw = dz * x        # 42   local derivative of w*x + b with respect to w is x
db = dz * 1        # 14
```

Two things to notice. The backward lines use `z` and `x`, values from the
forward pass, which is why the forward pass stores them. And if a variable
feeds into two operations, its gradient is the sum of the two
contributions, which is why real implementations write `grad +=`.

Why walk backward? The direct alternative is to measure each derivative:
add a small $h$ to one parameter, rerun the forward pass, and compute
$(L(w + h) - L(w))/h$. Here that gives 42.00001. But it gives the
derivative for one parameter per forward pass, so a model with $10^9$
parameters would need $10^9$ forward passes for a single gradient. The
backward pass produces all $10^9$ derivatives in one sweep, at about twice
the cost of a forward pass. This is reverse-mode autodiff, and it is
efficient because the loss is a single number.

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

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J$ is the
$m \times n$ table of partial derivatives $J_{ij} = \partial f_i/\partial
x_j$: row $i$ says how output $i$ changes with each input. For $f(x) = Wx$,
$J = W$. For an elementwise ReLU, $J$ is diagonal with 1 where $x > 0$ and
0 elsewhere.

Backprop never builds $J$. Given $g = \partial L/\partial f$, each layer
only needs $J^\top g$, and computes it directly:

```python
dx = W.T @ g        # linear layer
dx = g * (x > 0)    # ReLU
```

For a ReLU on 4,096 activations, $J$ would hold 16.8 million numbers; the
backward line above does 4,096 multiplications.
</details>

---

### Q8. Why do gradients vanish or explode in deep/recurrent nets?

<details><summary>Model answer</summary>

Take a chain of layers that each hold one number, $z_l = w_l h_{l-1}$ and
$h_l = f(z_l)$. The chain rule back to the input multiplies one factor per
layer:

$$\frac{\partial h_D}{\partial h_0} = \prod_{l=1}^{D} w_l\, f'(z_l)$$

A product of many factors is exponential in depth:

```python
0.9 ** 50      # 0.005   vanishes
1.1 ** 50      # 117     explodes
0.25 ** 10     # 1e-6    ten sigmoids: sigmoid's slope is at most 0.25
```

In a real network, a layer is $z = W h_{\text{prev}}$, $h = f(z)$, and the
backward step through it is `W.T @ (g * f'(z))`: take the upstream
gradient $g = \partial L/\partial h$, multiply elementwise by the
activation's slope $f'(z)$, then by $W^\top$. That is the matrix version
of $w_l f'(z_l)$, and the gradient's length behaves the same way. An RNN reuses exactly the same $W$
at every time step; the tanh slope varies per step but is at most 1, so if
$W$'s largest singular value is below 1, the gradient is guaranteed to
vanish over a long sequence.

Fixes, each keeping the factor near 1:

- Kaiming/Xavier init: weight scale chosen so each factor is about 1 at the start
- ReLU: slope 1 where active, instead of sigmoid's 0.25 at most
- LayerNorm/BatchNorm: activations stay at unit scale during training
- Residual connections: $h + F(h)$ has factor $I + \partial F/\partial h$, so an identity path always exists
- LSTM forget gate near 1: the same identity path, through time
- Gradient clipping: limits explosion only

**If asked:** *How do you detect it?* Log each layer's gradient norm.
Early layers orders of magnitude below late ones means vanishing; loss
spikes or NaN mean exploding.
</details>

---

### Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?

<details><summary>Model answer</summary>

Start with one feature: fit $y \approx wx$ through the points $(1, 1)$,
$(2, 2)$, $(3, 2)$. The loss is the sum of squared errors, and its
derivative is ordinary calculus:

$$L(w) = \sum_i (w x_i - y_i)^2, \qquad \frac{dL}{dw} = 2\sum_i x_i\,(w x_i - y_i)$$

Set it to zero and solve:

$$w = \frac{\sum_i x_i y_i}{\sum_i x_i^2} = \frac{1 + 4 + 6}{1 + 4 + 9} = \frac{11}{14} \approx 0.786$$

The matrix formula is the same computation. Stack the $x_i$ as the rows of
$X$ and the $y_i$ into $y$:

```python
err  = X @ w - y        # one error per data point
grad = 2 * X.T @ err    # sum of x_i * error_i: one number per weight
```

So $\nabla_w \|Xw - y\|^2 = 2X^\top(Xw - y)$, and with many features $X$
simply has more columns. Setting it to zero gives the normal equations,
$X^\top X\, w = X^\top y$, the matrix version of
$w = \sum x_i y_i / \sum x_i^2$:

```python
w = np.linalg.solve(X.T @ X, X.T @ y)    # 0.786
```

**If asked:** *What if $X^\top X$ is not invertible?* That happens when
features are copies or combinations of each other. Ridge adds $\lambda I$:
$(X^\top X + \lambda I)\,w = X^\top y$ always has a solution for
$\lambda > 0$.
</details>

## Probability & statistics

### Q10. Derive Bayes rule.

<details><summary>Model answer</summary>

The probability that $A$ and $B$ both happen can be computed in two orders:
first $B$ then $A$ given $B$, or first $A$ then $B$ given $A$. Set the two
equal and divide by $P(B)$:

$$P(A \mid B)\,P(B) = P(B \mid A)\,P(A) \quad\Longrightarrow\quad P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

When $P(B)$ is not given, add up both ways $B$ can happen:
$P(B) = P(B \mid A)P(A) + P(B \mid \lnot A)P(\lnot A)$. Q12 is this with
numbers.
</details>

---

### Q11. Why is minimizing cross-entropy the same as maximum likelihood?

<details><summary>Model answer</summary>

A classifier outputs a probability for each label. The likelihood of the
dataset is the product of the probabilities it assigned to the correct
labels. Take the log to turn the product into a sum, and a minus sign to
turn maximizing into minimizing. For one binary example with predicted
$p = p_\theta(1 \mid x)$:

```python
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))   # = -log(prob of the correct label)
```

That line is the binary cross-entropy loss. So cross-entropy is the
negative log-likelihood, term for term. The same steps with Gaussian noise
on a regression target give MSE, and with Laplace noise give absolute
error.

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

Count 10,000 people. 100 are sick, and 99 of them test positive. 9,900 are
healthy, and 5% of them, 495, test positive. Among the 594 positives, 99
are sick:

$$P(\text{sick} \mid +) = \frac{99}{99 + 495} \approx 0.17$$

About 17%, not 99%. The healthy group is so much larger that its small
false-positive rate produces five times more positives than the sick group
does. The same arithmetic is why accuracy misleads on imbalanced classes.

**If asked:**

- *99% accurate for both groups?* $99/(99 + 99) = 50\%$.
- *A second independent positive test?* With prior 0.167:
  $0.99 \cdot 0.167 / (0.99 \cdot 0.167 + 0.05 \cdot 0.833) \approx 0.80$.
</details>

---

### Q13. Why divide by $n-1$ in the sample variance?

<details><summary>Model answer</summary>

Try it: draw many samples of size 2 from a distribution with variance 1 and
average the two estimators.

```python
x = np.random.randn(100000, 2)       # true variance 1
x.var(axis=1).mean()                 # 0.50   divide by n
x.var(axis=1, ddof=1).mean()         # 0.99   divide by n - 1
```

Dividing by $n$ comes out low, at $\frac{n-1}{n}\sigma^2$ on average. The
reason: the deviations are measured from $\bar X$, which is computed from
the same sample and is the point that makes the sum of squared deviations
as small as possible. So the sum is smaller than it would be around the
true mean $\mu$, by exactly $n(\bar X - \mu)^2$:

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

Taking expectations, $n\sigma^2 - n \cdot \sigma^2/n = (n - 1)\sigma^2$.
Dividing by $n - 1$ removes the bias.

**If asked:** *Is $s$ then unbiased for $\sigma$?* No. The square root is
concave, so $\mathbb{E}[s] \le \sigma$ (Jensen).
</details>

---

### Q14. Why does averaging reduce noise, and at what rate?

<details><summary>Model answer</summary>

Try it: average $n$ draws with standard deviation 1, many times over, and
measure the spread of the averages.

```python
np.random.randn(100000, 100).mean(axis=1).std()   # 0.10 = 1/sqrt(100)
np.random.randn(100000, 400).mean(axis=1).std()   # 0.05 = 1/sqrt(400)
```

Four times the data halves the spread. The algebra:
$\mathrm{Var}(\bar X) = \frac{1}{n^2} \cdot n\sigma^2 = \sigma^2/n$. The
$1/n$ comes out squared, and independence lets the variances add. So the
standard error is $\sigma/\sqrt{n}$. The CLT adds that $\bar X$ is
approximately normal for large $n$.

**If asked:** *Correlated draws?* With pairwise correlation $\rho$,
$\mathrm{Var}(\bar X) = \sigma^2/n + \frac{n-1}{n}\rho\sigma^2 \to
\rho\sigma^2$. Averaging stops helping, which is why random forests
decorrelate their trees.
</details>

---

### Q15. Is a 0.5% accuracy gain real?

Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?

<details><summary>Model answer</summary>

First measure the noise. Each test example is right or wrong, so the
accuracy is an average of 1,000 coin flips with $p \approx 0.91$:

```python
np.sqrt(0.91 * 0.09 / 1000)     # 0.009, so about 0.9 points of noise
```

The gap is 0.5 points, smaller than the noise in either number. Probably
not real.

A sharper test uses the fact that both models were scored on the same
examples. 0.5 points is 5 more examples right. Let $b$ be the examples only
the new model gets right and $c$ those only the old model gets right, so
$b - c = 5$. McNemar's test: $z = (b - c)/\sqrt{b + c}$. Reaching
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
