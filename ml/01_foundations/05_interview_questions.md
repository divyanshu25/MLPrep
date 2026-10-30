# Interview questions & exit checklist

Fifteen standard questions, grouped by topic. Work them as a drill: read the
question, answer out loud or on paper as if the interviewer just asked it,
and only then open the model answer. Any answer you could not produce cold
goes into `practice/queue.tsv`.

Each model answer starts from a tiny concrete case with numbers and states
the general rule after. That is also a good way to answer in the room: show
the small case, then generalize. The full derivations live in the chapters.

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
lengths leaves $\cos\theta$, the cosine similarity, 0.96 here.

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

$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2)$$

$W_2 W_1$ is just another matrix and $W_2 b_1 + b_2$ is just another vector,
so the two layers compute exactly what one layer can. The same step
collapses any number of layers. Depth only adds functions the model can
represent when a nonlinearity sits between the layers.

**If asked:** *Something no linear model can do?* XOR. Positives
$(0,1), (1,0)$ require $w_1 + w_2 + 2b > 0$; negatives $(0,0), (1,1)$
require $w_1 + w_2 + 2b < 0$. Two hidden ReLUs fix it:
$y = \mathrm{ReLU}(x_1 + x_2) - 2\,\mathrm{ReLU}(x_1 + x_2 - 1)$ gives 0, 1,
1, 0.
</details>

---

### Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?

<details><summary>Model answer</summary>

Multiply a few vectors by $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$:
$A[1, 1] = [3, 3]$, same direction and 3 times longer; $A[1, -1] = [1, -1]$,
unchanged; $A[1, 0] = [2, 1]$, a new direction. The first two are
eigenvectors: directions that $A$ only scales, $Av = \lambda v$, with
eigenvalues 3 and 1.

In ML: the eigenvectors of the data covariance matrix $\Sigma$ are the PCA
directions, and each eigenvalue is the variance of the data along its
eigenvector. The reason: write a unit direction $u$ in $\Sigma$'s
eigenvectors with weights $c_i$ ($\sum c_i^2 = 1$). The variance along it is
$u^\top \Sigma u = \sum_i \lambda_i c_i^2$, a weighted average of the
eigenvalues, so it is largest when all the weight is on the top eigenvector.

**If asked:** *Another place?* Gradient descent on a quadratic multiplies
the error along each Hessian eigenvector by $1 - \eta\lambda$ per step, so it
converges only if $|1 - \eta\lambda| < 1$ for every eigenvalue:
$\eta < 2/\lambda_{\max}$.
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
approximation of $A$, which is what PCA and matrix factorization use. LoRA
rests on the related assumption that a fine-tuning update has low rank.

**If asked:** *When are they the same?* For symmetric positive
semidefinite matrices, such as covariance matrices.
</details>

---

### Q5. Why does L1 regularization give sparsity but L2 doesn't?

<details><summary>Model answer</summary>

Take one weight whose unregularized best value is $a$, and minimize
$(w - a)^2$ plus each penalty. Both have closed forms: with
$\lambda w^2$ the minimum is $w = a/(1 + \lambda)$, and with $\lambda|w|$ it
is $w = \mathrm{sign}(a)\max(0, |a| - \lambda/2)$. With $\lambda = 1$ and
$a = 0.3$: L2 gives 0.15, L1 gives exactly 0.

The reason is the slope of each penalty near zero. The L2 penalty
$\lambda w^2$ has slope $2\lambda w$, which goes to 0 as $w$ does, so any
nonzero slope of the loss moves the minimum off zero. The L1 penalty
$\lambda|w|$ has slope $\lambda$ all the way down to 0, so when the loss's
slope at zero ($2|a|$) is smaller than $\lambda$, the minimum stays at
exactly $w = 0$. In 2D, the L1 constraint region is a diamond with corners
on the axes, and the loss contours usually touch it first at a corner
(figure in §1.7 of the linear algebra notes).

**If asked:**

- *Bayesian view?* Minimizing loss plus penalty is maximizing likelihood
  times a prior. $-\log$ of a Gaussian prior is a squared penalty (L2), and
  $-\log$ of a Laplace prior is an absolute-value penalty (L1).
- *Does SGD with an L1 term give exact zeros?* No, subgradient steps
  oscillate around 0. Exact zeros need the clip-at-zero step above applied
  after each gradient step (a proximal step).
</details>

## Calculus

### Q6. Explain backprop.

<details><summary>Model answer</summary>

Take $L = (wx + b)^2$ with $w = 2$, $x = 3$, $b = 1$. The forward pass
computes and keeps each intermediate value: $z = wx + b = 7$, then
$L = z^2 = 49$. The backward pass starts at $\partial L/\partial L = 1$ and
walks back, multiplying by each operation's local derivative (the chain
rule): $\partial L/\partial z = 2z = 14$, then
$\partial L/\partial w = 14 \cdot x = 42$ and $\partial L/\partial b = 14$.

The backward step used $z$ and $x$, values from the forward pass, which is
why the forward pass stores them. And if a variable feeds into two
operations, its gradient is the sum of the two contributions.

Why backward? Measuring one derivative directly, $(L(w + h) - L(w))/h$ with
$h = 10^{-6}$, gives 42.000009, but it takes one forward pass per parameter:
$10^9$ passes for a model with $10^9$ parameters. The backward pass gives all
$10^9$ derivatives in one sweep at about twice the cost of a forward pass,
because the loss is a single number. This is reverse-mode autodiff.

**If asked:**

- *Why about 2× forward?* For $Y = XW$ with upstream gradient $G$, backward
  is two matrix multiplies of the same size, $X^\top G$ and $G W^\top$. A
  training step is then about 3× forward, which is where the estimate of 6
  FLOPs per parameter per training token comes from.
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
only needs $J^\top g$, and computes it directly: $W^\top g$ for a linear
layer, and $g$ times 1 or 0 entry by entry for a ReLU. For a ReLU on 4,096
activations, $J$ would hold 16.8 million numbers; the direct backward step
does 4,096 multiplications.
</details>

---

### Q8. Why do gradients vanish or explode in deep/recurrent nets?

<details><summary>Model answer</summary>

Take a chain of layers that each hold one number, $z_l = w_l h_{l-1}$ and
$h_l = f(z_l)$. The chain rule back to the input multiplies one factor per
layer:

$$\frac{\partial h_D}{\partial h_0} = \prod_{l=1}^{D} w_l\, f'(z_l)$$

A product of many factors is exponential in depth: $0.9^{50} \approx 0.005$
vanishes, $1.1^{50} \approx 117$ explodes, and ten sigmoids, whose slope is
at most 0.25, give at most $0.25^{10} \approx 10^{-6}$.

In a real network the backward step through a layer is
$W^\top(g \odot f'(z))$, the matrix version of $w_l f'(z_l)$, and the
gradient's length behaves the same way. An RNN reuses the same $W$ at every
time step, so if $W$'s largest singular value is below 1 the gradient
vanishes over a long sequence.

Fixes, each keeping the factor near 1: initialization scaled so each factor
starts near 1; ReLU (slope 1 where active); normalization layers; residual
connections, whose Jacobian $I + \partial F/\partial h$ contains an identity
path; the LSTM's gated memory, the same idea through time; and gradient
clipping, which only limits explosion.

**If asked:** *How do you detect it?* Log each layer's gradient norm.
Early layers orders of magnitude below late ones means vanishing; loss
spikes or NaN mean exploding.
</details>

---

### Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?

<details><summary>Model answer</summary>

Start with one feature: fit $y \approx wx$ through the points $(1, 1)$,
$(2, 2)$, $(3, 2)$. Differentiate the sum of squared errors and set it to
zero:

$$\frac{d}{dw}\sum_i (w x_i - y_i)^2 = 2\sum_i x_i\,(w x_i - y_i) = 0 \;\Rightarrow\; w = \frac{\sum_i x_i y_i}{\sum_i x_i^2} = \frac{11}{14} \approx 0.786$$

The matrix formula is the same computation. With the $x_i$ as the rows of
$X$, $X^\top(Xw - y)$ is the sum of each feature times each error, so
$\nabla_w \|Xw - y\|^2 = 2X^\top(Xw - y)$. Setting it to zero gives the
normal equations $X^\top X\, w = X^\top y$, the matrix version of
$w = \sum x_i y_i / \sum x_i^2$.

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
$p = p_\theta(1 \mid x)$, that gives $-[y \log p + (1 - y)\log(1 - p)]$, the
binary cross-entropy: $-\log 0.9 = 0.105$ if $y = 1$ and $p = 0.9$. So
cross-entropy is the negative log-likelihood, term for term. The same steps
with Gaussian noise on a regression target give MSE, and with Laplace noise
give absolute error.

**If asked:** *Information-theory view?* The expected loss is
$H(q, p_\theta) = H(q) + \mathrm{KL}(q \,\|\, p_\theta)$, and $H(q)$ does not
depend on $\theta$, so training minimizes the KL divergence from the data
to the model (Probability §3.5).
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
does. The same arithmetic is why accuracy misleads on imbalanced classes:
this test is right for 9,504 of the 10,000 people (95%), while a test that
calls everyone healthy is right for 9,900 (99%).

**If asked:**

- *99% accurate for both groups?* $99/(99 + 99) = 50\%$.
- *A second independent positive test?* With prior 0.167:
  $0.99 \cdot 0.167 / (0.99 \cdot 0.167 + 0.05 \cdot 0.833) \approx 0.80$.
</details>

---

### Q13. Why divide by $n-1$ in the sample variance?

<details><summary>Model answer</summary>

Draw many samples of size 2 from a distribution with variance 1. Dividing
by $n$ averages 0.50; dividing by $n - 1$ averages 1.00.

Dividing by $n$ comes out low, at $\frac{n-1}{n}\sigma^2$ on average. The
reason: the deviations are measured from $\bar X$, which is computed from
the same sample and is the point that makes the sum of squared deviations
as small as possible. So the sum is smaller than it would be around the
true mean $\mu$, by exactly $n(\bar X - \mu)^2$:

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

Taking expectations, $n\sigma^2 - n \cdot \sigma^2/n = (n - 1)\sigma^2$.
Dividing by $n - 1$ removes the bias.

**If asked:** *Is $s$ then unbiased for $\sigma$?* No. For a concave
function like the square root, the average of the function is at most the
function of the average (Jensen's inequality), so
$\mathbb{E}[s] \le \sqrt{\mathbb{E}[s^2]} = \sigma$.
</details>

---

### Q14. Why does averaging reduce noise, and at what rate?

<details><summary>Model answer</summary>

Average 100 draws with standard deviation 1, many times over: the averages
spread with standard deviation 0.10. Average 400 draws: 0.05. Four times
the data halves the spread.

The algebra: $\mathrm{Var}(\bar X) = \frac{1}{n^2} \cdot n\sigma^2 =
\sigma^2/n$. The $1/n$ comes out squared, and independence lets the
variances add. So the standard error is $\sigma/\sqrt{n}$. The CLT adds that
$\bar X$ is approximately normal for large $n$.

**If asked:** *Correlated draws?* With pairwise correlation $\rho$,
$\mathrm{Var}(\bar X) = \sigma^2/n + \frac{n-1}{n}\rho\sigma^2 \to
\rho\sigma^2$. Averaging stops helping, which is why random forests
decorrelate their trees.
</details>

---

### Q15. Is a 0.5% accuracy gain real?

Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?

<details><summary>Model answer</summary>

Probably not. 0.5 points on 1,000 examples is 5 more examples right.

Each test example is right or wrong, so the number of correct answers is
Binomial with $p \approx 0.91$. Its variance is
$1000 \times 0.91 \times 0.09 \approx 82$, so its standard error is
$\sqrt{82} \approx 9$ examples. The gain, 5 examples, is less than one
standard error, and significance needs about two (1.96). It's noise. (This
uses one score's noise as the yardstick. A careful test compares the models
on the examples where they disagree, McNemar's test in Probability §3.8, and
gives the same verdict unless the models disagree on only a handful of
examples.)

**If asked:** *What else would you do?* Check the training noise too. A
**random seed** sets the starting weights and the order of the training
data, and the same model trained with two different seeds often differs by
about 0.5 points of test accuracy. Train each model with several seeds and
compare the averages, and use a bigger test set.
</details>

## Exit checklist

Seven timed derivations, about 26 minutes total. Cold, on paper, before
moving on. A miss goes into `practice/queue.tsv`.

- [ ] Derive the projection of $b$ onto $a$ and explain the dot product as similarity - 3 min
- [ ] State SVD as rotate → stretch → rotate and give the geometric meaning of eigenvectors/eigenvalues - 3 min
- [ ] Write the chain rule for a two-stage composition and give the Jacobian shape for $f: \mathbb{R}^n \to \mathbb{R}^m$ - 3 min
- [ ] Derive Bayes rule from the definition of conditional probability, then apply it to a base-rate problem - 4 min
- [ ] Show that minimizing cross-entropy is maximum likelihood for a Bernoulli model - 5 min
- [ ] Define bias and variance of an estimator and explain why the 1/n variance estimator is biased - 4 min
- [ ] Use standard error to decide whether a 0.5% accuracy gap between two models on n=1000 is real - 4 min
