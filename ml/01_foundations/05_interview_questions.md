# Interview questions & exit checklist

Fifteen standard questions, grouped by topic. Answer each one out loud before
opening the model answer. Any answer you could not give cold goes into
`practice/queue.tsv`.

Each model answer gives the short answer first, then a small example with
numbers, then the general rule. The full derivations are in the chapters.

## Linear algebra

### Q1. Why does the dot product measure similarity?

<details><summary>Model answer</summary>

**It is large when two vectors point the same way, zero when they are
perpendicular, and negative when they point opposite ways.**

Take $a = (3, 4)$ and $b = (4, 3)$:

- $a \cdot b = 12 + 12 = 24$.
- Both have length 5. Pointing exactly the same way would give
  $5 \times 5 = 25$.
- $24 / 25 = 0.96$, so they point almost the same way.
- With $c = (4, -3)$: $a \cdot c = 12 - 12 = 0$, so $a$ and $c$ are
  perpendicular.

The rule is $a \cdot b = \|a\|\,\|b\|\cos\theta$. Dividing by the two lengths
leaves $\cos\theta$, the cosine similarity.

**If asked:**

- *Why divide by the lengths?* Otherwise long vectors score high against
  everything.
- *Where does the cosine come from?* Expand $\|a - b\|^2$ and compare it with
  the law of cosines.
</details>

---

### Q2. What happens if you stack linear layers with no nonlinearity?

<details><summary>Model answer</summary>

**They collapse into one linear layer, so the extra depth adds nothing.**

$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2)$$

$W_2 W_1$ is one matrix and $W_2 b_1 + b_2$ is one vector, so two layers
compute the same thing as one. The same step collapses any number of layers.
A nonlinearity between the layers stops the collapse.

**If asked:** *Something a linear model can't do?* XOR: no straight line
separates $(0,1), (1,0)$ from $(0,0), (1,1)$. Two ReLUs can:
$\mathrm{ReLU}(x_1 + x_2) - 2\,\mathrm{ReLU}(x_1 + x_2 - 1)$ gives 0, 1, 1, 0.
</details>

---

### Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?

<details><summary>Model answer</summary>

**An eigenvector is a direction the matrix only stretches and does not turn:
$Av = \lambda v$, where $\lambda$ is the stretch.**

Take $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$:

- $A[1, 1] = [3, 3]$: same direction, 3 times longer. Eigenvector, $\lambda = 3$.
- $A[1, -1] = [1, -1]$: unchanged. Eigenvector, $\lambda = 1$.
- $A[1, 0] = [2, 1]$: the direction changed. Not an eigenvector.

In ML: PCA. The eigenvectors of the data's covariance matrix are the
directions of spread, and each eigenvalue is the variance along its
direction. The top eigenvector is the first principal component.

**If asked:**

- *Why is the top eigenvector the direction of most variance?* Write a unit
  direction as $c_i$ parts of each eigenvector. The variance along it is
  $\sum_i \lambda_i c_i^2$, a weighted average of the eigenvalues, so it is
  largest with all the weight on $\lambda_1$.
- *Another place?* Gradient descent on a quadratic multiplies the error along
  each Hessian eigenvector by $1 - \eta\lambda$ per step, so it converges only
  if $\eta < 2/\lambda_{\max}$.
</details>

---

### Q4. What's the difference between eigendecomposition and SVD?

<details><summary>Model answer</summary>

**Eigendecomposition works only for some square matrices. SVD works for every
matrix.**

- $Av = \lambda v$ needs $v$ and $Av$ to be the same size, so $A$ must be
  square.
- Even some square matrices fail:
  $\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ has only one eigenvector
  direction, not two.
- SVD writes any matrix as $A = U\Sigma V^\top$: rotate ($V^\top$), stretch
  along the axes ($\Sigma$), rotate again ($U$).
- The two connect through $A^\top A$: the singular values of $A$ are the
  square roots of the eigenvalues of $A^\top A$.

Why it matters: keeping the top $k$ singular values gives the best rank-$k$
approximation of $A$. PCA and matrix factorization use this.

**If asked:** *When are they the same?* For symmetric matrices with no
negative eigenvalues, such as covariance matrices.
</details>

---

### Q5. Why does L1 regularization give sparsity but L2 doesn't?

<details><summary>Model answer</summary>

**L1 pushes a small weight toward zero with the same force all the way down.
L2's push shrinks as the weight shrinks, so it never quite reaches zero.**

Take one weight whose best value without a penalty is $a = 0.3$, and a
penalty strength $\lambda = 1$. Minimize $(w - a)^2$ plus the penalty:

- L2, adding $\lambda w^2$: the best $w$ is $a / (1 + \lambda) = 0.15$.
  Smaller, but not zero.
- L1, adding $\lambda |w|$: the best $w$ is exactly 0.

Why: near zero, the slope of the L2 penalty is $2\lambda w$, which goes to 0.
The slope of the L1 penalty stays at $\lambda$. If the loss pulls with less
than $\lambda$, the weight stays at 0. Here the loss pulls with
$2a = 0.6 < 1$.

**If asked:**

- *Picture?* In 2D the L1 region is a diamond with corners on the axes. The
  loss contours usually touch it at a corner, where one weight is 0.
- *Bayesian view?* L2 is a Gaussian prior on the weights; L1 is a Laplace
  prior.
</details>

## Calculus

### Q6. Explain backprop.

<details><summary>Model answer</summary>

**Backprop applies the chain rule from the loss back to the inputs, and gets
the gradient for every parameter in one backward pass.**

Take $L = (wx + b)^2$ with $w = 2$, $x = 3$, $b = 1$:

- Forward, saving each value: $z = wx + b = 7$, then $L = z^2 = 49$.
- Backward, multiplying local derivatives: $\partial L/\partial z = 2z = 14$,
  then $\partial L/\partial w = 14 \cdot x = 42$ and
  $\partial L/\partial b = 14$.

The backward pass used $z$ and $x$, which is why the forward pass saves them.

Why backward: the loss is one number, so one backward pass gives all the
gradients, at about twice the cost of a forward pass. Nudging each parameter
separately would take one forward pass per parameter: a billion passes for a
billion-parameter model.

**If asked:**

- *Why about 2× forward?* For $Y = XW$, backward is two matrix multiplies of
  the same size, one for $W$ and one for $X$. A training step is then about
  3× forward: 6 FLOPs per parameter per token.
- *What limits it?* Memory for the saved activations. Checkpointing
  recomputes them instead of saving them.
</details>

---

### Q7. What is a Jacobian and what's its role in backprop?

<details><summary>Model answer</summary>

**The Jacobian is the table of every output's derivative with respect to
every input. Backprop never builds it; it only multiplies by it.**

- For $f: \mathbb{R}^n \to \mathbb{R}^m$, $J$ is $m \times n$ with
  $J_{ij} = \partial f_i / \partial x_j$.
- For $f(x) = Wx$, $J = W$.
- For a ReLU, $J$ is diagonal: 1 where $x > 0$, 0 elsewhere.

Each layer receives the gradient $g$ from the layer above and needs
$J^\top g$. It computes that directly: $W^\top g$ for a linear layer, and $g$
with entries zeroed where $x \le 0$ for a ReLU. For a ReLU on 4,096 values,
$J$ would hold 16.8 million numbers; the direct step does 4,096
multiplications.
</details>

---

### Q8. Why do gradients vanish or explode in deep/recurrent nets?

<details><summary>Model answer</summary>

**The gradient at an early layer is a product with one factor per layer, and
a product of many factors shrinks to zero or blows up.**

For a chain of one-number layers $h_l = f(w_l h_{l-1})$:

$$\frac{\partial h_D}{\partial h_0} = \prod_{l=1}^{D} w_l\, f'(w_l h_{l-1})$$

- Every factor 0.9, 50 layers: $0.9^{50} \approx 0.005$. Vanishes.
- Every factor 1.1: $1.1^{50} \approx 117$. Explodes.
- A sigmoid's slope is at most 0.25, so ten sigmoids give at most
  $0.25^{10} \approx 10^{-6}$.

An RNN multiplies by the same $W$ at every time step, so long sequences hit
this hard.

The fixes keep each factor near 1: careful initialization, ReLU (slope 1
where active), normalization layers, residual connections (a direct identity
path for the gradient), LSTM gates (the same idea through time), and
gradient clipping (stops explosion only).

**If asked:** *How do you detect it?* Log the gradient norm of each layer.
Early layers far smaller than late ones means vanishing; loss spikes or NaN
mean exploding.
</details>

---

### Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?

<details><summary>Model answer</summary>

**The gradient is $2X^\top(Xw - y)$. Setting it to zero gives the normal
equations $X^\top X\, w = X^\top y$, whose solution is the least-squares
fit.**

With one feature, fit $y \approx wx$ to the points $(1, 1)$, $(2, 2)$,
$(3, 2)$:

$$\frac{d}{dw}\sum_i (w x_i - y_i)^2 = 2\sum_i x_i\,(w x_i - y_i) = 0 \;\Rightarrow\; w = \frac{\sum_i x_i y_i}{\sum_i x_i^2} = \frac{11}{14} \approx 0.79$$

The matrix version is the same sum: $X^\top(Xw - y)$ adds up each feature
times each error.

**If asked:** *What if $X^\top X$ is not invertible?* That happens when
features duplicate each other. Ridge adds $\lambda I$:
$(X^\top X + \lambda I)\,w = X^\top y$ always has a solution for
$\lambda > 0$.
</details>

## Probability & statistics

### Q10. Derive Bayes rule.

<details><summary>Model answer</summary>

**Write the probability of "A and B" in two ways, set them equal, and
divide.**

$$P(A \mid B)\,P(B) = P(B \mid A)\,P(A) \quad\Longrightarrow\quad P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

Both sides are $P(A \text{ and } B)$: "B happens, then A given B" or "A
happens, then B given A". If $P(B)$ is not given, add up the two ways B can
happen: $P(B) = P(B \mid A)P(A) + P(B \mid \lnot A)P(\lnot A)$. Q12 uses
this.
</details>

---

### Q11. Why is minimizing cross-entropy the same as maximum likelihood?

<details><summary>Model answer</summary>

**Cross-entropy is the negative log of the likelihood, so minimizing one
maximizes the other.**

- The likelihood of the data is the product of the probabilities the model
  gave the correct labels.
- Take the log: the product becomes a sum.
- Put a minus sign in front: maximizing becomes minimizing.
- That sum is the cross-entropy loss.

Example: if the model gives the correct label probability 0.9, that example
adds $-\log 0.9 = 0.105$ to the loss. At probability 0.5 it adds 0.69.

The same steps with Gaussian noise on a regression target give MSE.

**If asked:** *Information-theory view?* Cross-entropy equals
$H(q) + \mathrm{KL}(q \,\|\, p_\theta)$. $H(q)$ does not depend on the model,
so training minimizes the KL divergence from the data to the model.
</details>

---

### Q12. Positive test, rare disease

A test detects the disease in 99% of sick people and gives a false positive
for 5% of healthy people. 1% of the population has the disease. You test
positive. Roughly what's the chance you're sick?

<details><summary>Model answer</summary>

**About 17%.**

Count 10,000 people:

- 100 are sick, and 99 of them test positive.
- 9,900 are healthy, and 5% of them, 495, test positive.
- Of the 594 positives, 99 are sick.

$$P(\text{sick} \mid +) = \frac{99}{99 + 495} \approx 0.17$$

Healthy people are 99 times more common, so even a 5% false-positive rate
gives five times more false positives than true positives.

**If asked:**

- *99% accurate for both groups?* $99 / (99 + 99) = 50\%$.
- *A second independent positive test?* Start from 17%:
  $0.99 \cdot 0.17 / (0.99 \cdot 0.17 + 0.05 \cdot 0.83) \approx 80\%$.
</details>

---

### Q13. Why divide by $n-1$ in the sample variance?

<details><summary>Model answer</summary>

**Dividing by $n$ underestimates the variance, because deviations are
measured from the sample mean instead of the true mean. Dividing by $n - 1$
fixes it exactly.**

Example: draw many samples of size 2 from a distribution with variance 1.
Dividing by $n$ averages 0.5. Dividing by $n - 1$ averages 1.0.

Why it is low: the sample mean $\bar X$ is the point closest to the sample's
own values. So the squared deviations from $\bar X$ are smaller than those
from the true mean $\mu$, by exactly $n(\bar X - \mu)^2$:

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

On average the right side is $n\sigma^2 - \sigma^2 = (n - 1)\sigma^2$, so
divide by $n - 1$.

**If asked:** *Is $s$ then unbiased for $\sigma$?* No. The square root is
concave, so $\mathbb{E}[s] \le \sqrt{\mathbb{E}[s^2]} = \sigma$ (Jensen's
inequality).
</details>

---

### Q14. Why does averaging reduce noise, and at what rate?

<details><summary>Model answer</summary>

**Averaging $n$ independent draws divides the variance by $n$, so the noise
(standard deviation) falls as $1/\sqrt{n}$.**

Example: draws with standard deviation 1. The average of 100 draws has
standard deviation 0.1. The average of 400 has 0.05. Four times the data,
half the noise.

Why: $\mathrm{Var}(\bar X) = \frac{1}{n^2} \cdot n\sigma^2 = \sigma^2/n$. The
variances of independent draws add, and the $1/n$ comes out squared.

**If asked:** *Correlated draws?* With correlation $\rho$ between draws, the
variance of the average goes to $\rho\sigma^2$, not 0, so averaging stops
helping. That is why random forests decorrelate their trees.
</details>

---

### Q15. Is a 0.5% accuracy gain real?

Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?

<details><summary>Model answer</summary>

**Probably not. The gain is 5 examples, and the noise is about 9.**

- 0.5% of 1,000 is 5 more examples right.
- The number correct is Binomial with $p \approx 0.91$.
- Its variance is $1000 \times 0.91 \times 0.09 \approx 82$, so its standard
  error is $\sqrt{82} \approx 9$.
- 5 is under one standard error. Significance needs about two. So it is
  noise.

**If asked:**

- *A sharper test?* Both models saw the same examples, so compare them only
  where they disagree (McNemar's test, Probability §3.8).
- *What else?* Train each model with a few random seeds. The seed alone
  often moves accuracy by about 0.5 points.
</details>

## Exit checklist

Seven timed derivations, about 26 minutes total. Do them cold, on paper,
before moving on. A miss goes into `practice/queue.tsv`.

- [ ] Derive the projection of $b$ onto $a$ and explain the dot product as similarity - 3 min
- [ ] State SVD as rotate → stretch → rotate and give the geometric meaning of eigenvectors/eigenvalues - 3 min
- [ ] Write the chain rule for a two-stage composition and give the Jacobian shape for $f: \mathbb{R}^n \to \mathbb{R}^m$ - 3 min
- [ ] Derive Bayes rule from the definition of conditional probability, then apply it to a base-rate problem - 4 min
- [ ] Show that minimizing cross-entropy is maximum likelihood for a Bernoulli model - 5 min
- [ ] Define bias and variance of an estimator and explain why the 1/n variance estimator is biased - 4 min
- [ ] Use standard error to decide whether a 0.5% accuracy gap between two models on n=1000 is real - 4 min
