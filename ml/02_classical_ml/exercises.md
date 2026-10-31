# Classical ML: Exercises

Ordered easy → hard. Attempt each on paper before opening the solution.
Notation matches `notes.md`: $X \in \mathbb{R}^{n\times d}$, rows are
examples, bias absorbed unless stated.

---

## Exercise 1 (warm-up). Least squares by hand

Fit $y = w x + b$ to the three points $(0, 1), (1, 2), (2, 2)$ by minimizing
squared error. Compute $w$ and $b$ exactly, and verify the residuals sum
to zero.

<details><summary>Solution</summary>

With $X = \begin{pmatrix}1 & 0\\ 1 & 1\\ 1 & 2\end{pmatrix}$ (bias column
first), the normal equations $X^\top X \binom{b}{w} = X^\top y$ are

$$\begin{pmatrix}3 & 3\\ 3 & 5\end{pmatrix}\begin{pmatrix}b\\ w\end{pmatrix} = \begin{pmatrix}5\\ 6\end{pmatrix}.$$

Solve: from row 1, $b = (5 - 3w)/3$. Substitute into row 2:
$3\cdot\frac{5-3w}{3} + 5w = 6 \Rightarrow 5 - 3w + 5w = 6 \Rightarrow w = \tfrac12$,
$b = (5 - 1.5)/3 = \tfrac{7}{6}$.

Predictions: $\hat y = (7/6,\; 5/3,\; 13/6)$. Residuals:
$(1 - 7/6, 2 - 5/3, 2 - 13/6) = (-1/6, +1/3, -1/6)$; sum $= 0$. This is
guaranteed whenever a bias column is present: the residual is orthogonal to
the all-ones column, i.e. $\mathbf{1}^\top r = 0$.
</details>

---

## Exercise 2. Derive the ridge closed form

Show that $\hat w = (X^\top X + \lambda I)^{-1} X^\top y$ minimizes
$L(w) = \|Xw - y\|^2 + \lambda \|w\|^2$, and explain why the inverse always
exists for $\lambda > 0$.

<details><summary>Solution</summary>

Expand: $L(w) = w^\top X^\top X w - 2y^\top X w + y^\top y + \lambda w^\top w$.
Gradient (using $\nabla w^\top A w = 2Aw$ for symmetric $A$):

$$\nabla L = 2X^\top X w - 2X^\top y + 2\lambda w = 0 \;\Rightarrow\; (X^\top X + \lambda I) w = X^\top y.$$

The Hessian is $2(X^\top X + \lambda I) \succ 0$, so this stationary point is
the unique global minimum (strictly convex).

Existence of the inverse: $X^\top X$ is symmetric PSD, so it has eigenvalues
$\mu_i \ge 0$. The eigenvalues of $X^\top X + \lambda I$ are
$\mu_i + \lambda \ge \lambda > 0$, all strictly positive, hence invertible,
*even when* columns of $X$ are collinear or $d > n$. Bonus insight: in the
eigenbasis, ridge shrinks each component by $\mu_i / (\mu_i + \lambda)$;
low-variance directions (small $\mu_i$) are shrunk hardest, which is exactly
where OLS variance is largest.
</details>

---

## Exercise 3. Entropy and information gain by hand

A node holds 8 positive and 8 negative examples. Split A yields children
$(6+, 2-)$ and $(2+, 6-)$. Split B yields $(8+, 4-)$ and $(0+, 4-)$. Compute
the information gain of both (in bits, 3 decimals) and pick the better split.

<details><summary>Solution</summary>

Parent: $p = 1/2$, $H = 1$ bit.

**Split A**: each child has $p = 3/4$ (or $1/4$; entropy is symmetric).
$H = -\tfrac34\log_2\tfrac34 - \tfrac14\log_2\tfrac14 = 0.75(0.415) + 0.25(2) = 0.811$.
Both children weight $8/16 = 1/2$:
$IG_A = 1 - 0.811 = 0.189$ bits.

**Split B**: left child $(8+,4-)$: $p = 2/3$,
$H = -\tfrac23\log_2\tfrac23 - \tfrac13\log_2\tfrac13 = \tfrac23(0.585) + \tfrac13(1.585) = 0.918$,
weight $12/16 = 0.75$. Right child pure: $H = 0$, weight $0.25$.
$IG_B = 1 - 0.75(0.918) - 0 = 1 - 0.689 = 0.311$ bits.

**B wins** (0.311 > 0.189): carving out a pure chunk beats making both sides
mildly cleaner, even though B is unbalanced. This is the arithmetic a tree
does at every node.
</details>

---

## Exercise 4. Gradient of logistic loss (full chain)

Derive $\nabla_w L$ for $L = -[y\log p + (1-y)\log(1-p)]$,
$p = \sigma(w^\top x)$, showing every step, including the proof that
$\sigma'(z) = \sigma(z)(1-\sigma(z))$.

<details><summary>Solution</summary>

**Sigmoid derivative.** $\sigma(z) = (1 + e^{-z})^{-1}$. By the chain rule,

$$\sigma'(z) = -(1+e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2} = \underbrace{\frac{1}{1+e^{-z}}}_{\sigma}\cdot\underbrace{\frac{e^{-z}}{1+e^{-z}}}_{1-\sigma} = \sigma(1-\sigma).$$

**Loss wrt $p$:**
$\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{-y(1-p) + (1-y)p}{p(1-p)} = \frac{p - y}{p(1-p)}$.

**Chain through $z = w^\top x$:**

$$\frac{\partial L}{\partial z} = \frac{p-y}{p(1-p)}\cdot p(1-p) = p - y, \qquad \nabla_w L = (p - y)\,x.$$

The $p(1-p)$ cancellation is the key result: with the *matched* loss
(cross-entropy for a sigmoid), the sigmoid's flat tails do not appear in the
gradient. Had you used MSE, $\partial L/\partial z = (p - y)\,p(1-p)$, which
$\to 0$ when the model is confidently wrong, so the update is near zero
exactly when the error is largest. Full-batch:
$\nabla_w L = \frac{1}{n}X^\top(p - y)$.
</details>

---

## Exercise 5. Soft-thresholding: the 1-D lasso

Solve $\min_w \frac12 (w - a)^2 + \lambda |w|$ exactly for all $a$ and
$\lambda > 0$. Then explain in one sentence why the same phenomenon gives
sparsity in $d$ dimensions.

<details><summary>Solution</summary>

Case $w > 0$: objective is smooth; derivative $w - a + \lambda = 0 \Rightarrow
w = a - \lambda$, consistent (positive) iff $a > \lambda$.
Case $w < 0$: $w = a + \lambda$, consistent iff $a < -\lambda$.
Case $w = 0$: the subdifferential is $\{-a + \lambda t : t \in [-1, 1]\}$;
$0$ belongs to it iff $|a| \le \lambda$.

Together:

$$w^* = \operatorname{sign}(a)\max(0, |a| - \lambda) \quad (\text{soft threshold}).$$

Any coordinate whose "unregularized pull" $|a|$ is below $\lambda$ is set to
**exactly zero**. This is possible only because $|w|$ has a kink at 0 whose
subgradients form a whole interval, so for $|a| \le \lambda$ some subgradient
cancels the data gradient $-a$. In $d$ dimensions the
same kink exists along every coordinate axis (the corners/edges of the L1
ball), so weakly-useful features are zeroed while L2's smooth penalty
($w^* = a/(1+2\lambda)$ in the same 1-D setup) only ever shrinks.
</details>

---

## Exercise 6. Variance of a mean of correlated estimators

Let $f_1,\dots,f_B$ be identically distributed with variance $\sigma^2$ and
pairwise correlation $\rho$. Derive $\operatorname{Var}(\bar f)$ for
$\bar f = \frac1B \sum_b f_b$, and use the result to justify (a) bagging and
(b) random forests' feature subsampling.

<details><summary>Solution</summary>

$$\operatorname{Var}(\bar f) = \frac{1}{B^2}\operatorname{Var}\Big(\sum_b f_b\Big) = \frac{1}{B^2}\Big[\sum_b \operatorname{Var}(f_b) + \sum_{b\ne b'}\operatorname{Cov}(f_b, f_{b'})\Big].$$

There are $B$ variance terms ($\sigma^2$ each) and $B(B-1)$ covariance terms
($\rho\sigma^2$ each):

$$\operatorname{Var}(\bar f) = \frac{B\sigma^2 + B(B-1)\rho\sigma^2}{B^2} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$

As $B \to \infty$ the second term vanishes but $\rho\sigma^2$ remains: an
irreducible floor set by correlation.

(a) **Bagging**: bootstrap resampling produces diverse trees, making
$\rho < 1$ so averaging genuinely reduces variance; since deep trees have
huge $\sigma^2$ and low bias, averaging reduces exactly the term that is large
for them.
(b) **Random forests**: bagged trees still correlate (they all split on the same
dominant features near the root), keeping the floor high. Random feature
subsets per split lower $\rho$ directly: a small loss in per-tree accuracy
gives a lower floor, which is the term that actually matters at large $B$.
</details>

---

## Exercise 7. k-means objective decreases every step (proof)

Prove that each iteration of Lloyd's algorithm does not increase
$J(z,\mu) = \sum_i \|x_i - \mu_{z_i}\|^2$, and conclude that the algorithm
terminates in finitely many iterations. Why is the result not necessarily a
global minimum?

<details><summary>Solution</summary>

**Assignment step.** With centers fixed at $\mu^t$, each point independently
switches to its nearest center:
$z_i^{t+1} = \arg\min_j \|x_i - \mu_j^t\|^2$, so term-by-term
$\|x_i - \mu^t_{z_i^{t+1}}\|^2 \le \|x_i - \mu^t_{z_i^{t}}\|^2$, hence
$J(z^{t+1}, \mu^t) \le J(z^t, \mu^t)$.

**Update step.** With assignments fixed, for each cluster $j$ the map
$\mu \mapsto \sum_{i: z_i = j}\|x_i - \mu\|^2$ is a strictly convex quadratic
whose unique minimizer is the mean: setting the gradient
$-2\sum_{i:z_i=j}(x_i - \mu) = 0$ gives $\mu = \frac{1}{n_j}\sum x_i$. So
replacing each center by its cluster mean gives
$J(z^{t+1}, \mu^{t+1}) \le J(z^{t+1}, \mu^t)$.

Chaining: $J$ is non-increasing across iterations, and $J \ge 0$. Moreover
$J$ after the update step is a function of the assignment alone, and there
are at most $k^n$ assignments, finitely many. A strictly decreasing sequence
over a finite set of attainable values must stop changing in finite time;
once the assignment repeats, the algorithm is at a fixed point and halts.

Not global: the argument only shows that $J$ never increases. The fixed point
reached depends on initialization; e.g. two initial centers inside one true
blob can converge to splitting that blob while merging two others: locally
stable, globally bad. Hence k-means++ and random restarts.
</details>

---

## Exercise 8. When is naive Bayes optimal?

State precisely the condition under which naive Bayes achieves the Bayes
error rate, and give (a) a concrete distribution where it is optimal, and
(b) one where correlated features mislead it. Explain why NB's *probability
estimates* degrade before its *decisions* do.

<details><summary>Solution</summary>

The Bayes-optimal classifier is $\arg\max_y p(y)p(x\mid y)$. NB computes
$\arg\max_y p(y)\prod_j p(x_j \mid y)$. Therefore NB is optimal when its
model is correct: **features conditionally independent given the class**
(with correctly estimated marginals); or, more weakly, whenever the
independence-factored posterior happens to have the same argmax as the true
posterior for (almost) all $x$.

(a) Optimal: class $y \sim \text{Bernoulli}(1/2)$; given $y$, features
$x_1, x_2$ drawn independently, e.g.
$x_j \mid y \sim \mathcal N(\mu_y, 1)$ independently across $j$. The true
likelihood factorizes, so NB *is* the Bayes classifier.

(b) Misled: duplicate a feature, $x_2 = x_1$ exactly. The true model should
count $x_1$'s evidence once; NB multiplies $p(x_1\mid y)$ twice, squaring the
likelihood ratio. With a third, genuinely informative feature $x_3$ pointing
the other way, the double-counted $x_1$ can outvote $x_3$ and flip decisions
the Bayes classifier gets right.

Decisions vs. probabilities: the argmax only needs the *sign* of
$\log\frac{p(y{=}1\mid x)}{p(y{=}0\mid x)}$, which survives monotone
distortions; double counting typically *stretches* the log-odds (pushing
posteriors toward 0/1, i.e. overconfidence) while often preserving its sign. So
calibration fails first, accuracy later.
</details>

---

## Exercise 9. Perceptron convergence theorem

The perceptron update: on a mistake for example $(x_i, y_i)$,
$y_i \in \{-1,+1\}$, set $w \leftarrow w + y_i x_i$ (start $w_0 = 0$). Assume
the data is linearly separable with margin $\gamma > 0$: there exists a unit
vector $w^*$ with $y_i \langle w^*, x_i\rangle \ge \gamma$ for all $i$, and
$\|x_i\| \le R$. Prove the perceptron makes at most $R^2/\gamma^2$ mistakes.

<details><summary>Solution</summary>

Let $w_k$ be the weight vector after the $k$-th mistake. Track two
quantities.

**(1) Alignment with $w^*$ grows linearly.** On the $k$-th mistake (example
$i$):

$$\langle w_k, w^*\rangle = \langle w_{k-1}, w^*\rangle + y_i\langle x_i, w^*\rangle \ge \langle w_{k-1}, w^*\rangle + \gamma.$$

By induction from $w_0 = 0$: $\langle w_k, w^*\rangle \ge k\gamma$.

**(2) Norm grows at most like $\sqrt{k}$.**

$$\|w_k\|^2 = \|w_{k-1}\|^2 + 2 y_i \langle w_{k-1}, x_i\rangle + \|x_i\|^2 \le \|w_{k-1}\|^2 + R^2,$$

because the middle term is $\le 0$: we only update on a **mistake**, which
means $y_i\langle w_{k-1}, x_i\rangle \le 0$. By induction:
$\|w_k\|^2 \le kR^2$.

**Combine** with Cauchy–Schwarz ($\|w^*\| = 1$):

$$k\gamma \le \langle w_k, w^*\rangle \le \|w_k\| \le \sqrt{k}\,R \;\Rightarrow\; \sqrt{k} \le R/\gamma \;\Rightarrow\; k \le R^2/\gamma^2. \qquad\blacksquare$$

Interpretation: every mistake increases the alignment $\langle w, w^*\rangle$ by
at least $\gamma$, while $\|w\|$ grows only like $\sqrt{k}$; since the alignment
cannot exceed $\|w\|$, the number of mistakes is bounded. Note the bound is independent of
$n$ and $d$; only the *geometric* quantities $R$ and $\gamma$ matter. This is
the earliest form of margin-based generalization arguments (SVMs maximize
exactly this $\gamma$).
</details>

---

## Exercise 10. Bias–variance decomposition

Let $y = f(x) + \varepsilon$ with $\mathbb E[\varepsilon] = 0$,
$\operatorname{Var}(\varepsilon) = \sigma^2$, and let $\hat f$ be a model
trained on a random dataset $D$. Derive, for a fixed test point $x$:

$$\mathbb E_{D,\varepsilon}\big[(y - \hat f(x))^2\big] = \underbrace{\big(f(x) - \mathbb E_D[\hat f(x)]\big)^2}_{\text{bias}^2} + \underbrace{\mathbb E_D\big[(\hat f(x) - \mathbb E_D[\hat f(x)])^2\big]}_{\text{variance}} + \sigma^2.$$

<details><summary>Solution</summary>

Write $\bar f = \mathbb E_D[\hat f(x)]$ (average prediction over training
sets). Insert $\pm f(x)$ then $\pm \bar f$ and expand.

**Step 1: separate the noise.**

$$\mathbb E[(y - \hat f)^2] = \mathbb E[(f + \varepsilon - \hat f)^2] = \mathbb E[(f - \hat f)^2] + 2\,\mathbb E[\varepsilon(f - \hat f)] + \mathbb E[\varepsilon^2].$$

The cross term vanishes because $\varepsilon$ (test noise) is independent of
$\hat f$ (a function of the training set) and $\mathbb E[\varepsilon] = 0$.
The last term is $\sigma^2$.

**Step 2: split model error around $\bar f$.**

$$\mathbb E_D[(f - \hat f)^2] = \mathbb E_D[(f - \bar f + \bar f - \hat f)^2] = (f - \bar f)^2 + 2(f - \bar f)\,\mathbb E_D[\bar f - \hat f] + \mathbb E_D[(\bar f - \hat f)^2].$$

The middle term vanishes because $\mathbb E_D[\bar f - \hat f] = \bar f -
\bar f = 0$ ($f - \bar f$ is a constant wrt $D$). What remains is
$\text{bias}^2 + \text{variance}$.

Together: expected test MSE $= \text{bias}^2 + \text{variance} + \sigma^2$.
Interpretation checklist: $\sigma^2$ is irreducible; regularization,
pruning, and averaging trade the two model terms against each other
(ridge: bias ↑ variance ↓; deeper tree: bias ↓ variance ↑; bagging: variance
↓ at ~constant bias). Note the decomposition is for *squared* loss;
for 0-1 loss no such clean additive split exists.
</details>

---

## Exercise 11. PCA: two derivations meet

(a) Show that the unit vector maximizing the projected variance
$u^\top \Sigma u$ is the top eigenvector of $\Sigma$.
(b) Show that minimizing reconstruction error
$\sum_i \|x_i - uu^\top x_i\|^2$ over unit $u$ gives the same answer
(centered data).
(c) Show the minimal reconstruction error with $k$ components equals
$(n-1)\sum_{j>k}\lambda_j$.

<details><summary>Solution</summary>

**(a)** Maximize $u^\top \Sigma u$ s.t. $u^\top u = 1$. Lagrangian
$\mathcal L = u^\top \Sigma u - \lambda(u^\top u - 1)$;
$\nabla_u \mathcal L = 2\Sigma u - 2\lambda u = 0 \Rightarrow \Sigma u =
\lambda u$: stationary points are eigenvectors, with objective value
$u^\top \Sigma u = \lambda u^\top u = \lambda$. The maximum over eigenvectors
is the largest eigenvalue $\lambda_1$, attained at its eigenvector $u_1$.

**(b)** For unit $u$, using $u^\top u = 1$:

$$\|x - uu^\top x\|^2 = x^\top x - 2x^\top u u^\top x + x^\top u u^\top u u^\top x = \|x\|^2 - (u^\top x)^2.$$

Summing over (centered) data:
$\sum_i \|x_i - uu^\top x_i\|^2 = \sum_i \|x_i\|^2 - \sum_i (u^\top x_i)^2 =
\text{const} - (n-1)\, u^\top \Sigma u$. Minimizing the left side maximizes
$u^\top \Sigma u$, the identical problem as (a). Same eigenvector.

**(c)** Take orthonormal $U = [u_1 \cdots u_k]$ spanning the top-$k$
eigenspace. Total energy in the eigenbasis:
$\sum_i \|x_i\|^2 = \operatorname{tr}\big(\sum_i x_i x_i^\top\big) = (n-1)\operatorname{tr}(\Sigma) = (n-1)\sum_{j=1}^d \lambda_j$.
Captured energy:
$\sum_i \|U^\top x_i\|^2 = (n-1)\sum_{j\le k} u_j^\top \Sigma u_j = (n-1)\sum_{j \le k}\lambda_j$.
By the Pythagorean split from (b), the error is total − captured:

$$\min \sum_i \|x_i - UU^\top x_i\|^2 = (n-1)\sum_{j > k} \lambda_j.$$

(That the top-$k$ eigenspace is optimal among all $k$-dim subspaces is Ky
Fan's inequality: for any $U$ with $k$ orthonormal columns,
$\operatorname{tr}(U^\top \Sigma U) \le \lambda_1 + \dots + \lambda_k$.) So the
reconstruction-error curve is the tail sum of eigenvalues, which is why it
drops sharply until $k$ reaches the true latent dimension and is nearly flat
after it.
</details>

---

## Exercise 12 (hard). Gradient boosting from scratch, on paper

Consider gradient boosting for binary classification with log-loss
$\ell(y, F) = \log(1 + e^{-\tilde y F})$ where $\tilde y \in \{-1,+1\}$ and
$F$ is the model's raw score (log-odds).
(a) Derive the pseudo-residual $r_i = -\partial \ell/\partial F$ and show it
equals $y_i - p_i$ with $y \in \{0,1\}$, $p = \sigma(F)$.
(b) Derive the optimal constant initial model $F_0$.
(c) For a regression tree fit to the pseudo-residuals, derive the optimal
leaf value under a second-order (Newton) approximation, which is the XGBoost
leaf formula with $\lambda = 0$.

<details><summary>Solution</summary>

**(a)**

$$\frac{\partial \ell}{\partial F} = \frac{-\tilde y e^{-\tilde y F}}{1 + e^{-\tilde y F}} = -\tilde y\,\sigma(-\tilde y F) = -\tilde y (1 - \sigma(\tilde y F)).$$

For $\tilde y = +1$ ($y = 1$): $\partial\ell/\partial F = -(1 - \sigma(F)) =
-(1 - p)$, so $r = 1 - p = y - p$. For $\tilde y = -1$ ($y = 0$):
$\partial\ell/\partial F = 1 - \sigma(-F) = \sigma(F) = p$, so
$r = -p = y - p$. In both cases $r_i = y_i - p_i$: the same "residual on the
probability scale" as logistic regression (notes §2.3), now serving as the
regression target of the next tree.

**(b)** $F_0 = \arg\min_c \sum_i \ell(y_i, c)$. Set the derivative to zero
using (a): $\sum_i (p_i - y_i) = 0$ with $p_i = \sigma(c)$ for all $i$, so
$\sigma(c) = \bar y$ (the positive rate), giving

$$F_0 = \log\frac{\bar y}{1 - \bar y},$$

the log-odds of the base rate. Sanity: 90% positives → $F_0 = \log 9
\approx 2.197$, $\sigma(F_0) = 0.9$. ✓

**(c)** Fix the tree's structure; for one leaf $S$ we choose a single value
$v$ added to every $F_i$, $i \in S$. Second-order Taylor of the leaf's loss
around $v = 0$, with $g_i = \partial_F \ell = p_i - y_i$ and
$h_i = \partial^2_F \ell = p_i(1 - p_i)$ (differentiate $p_i - y_i$ wrt $F$
using $\sigma' = \sigma(1-\sigma)$):

$$\sum_{i\in S} \ell(y_i, F_i + v) \approx \text{const} + v\sum_{i\in S} g_i + \frac{v^2}{2}\sum_{i \in S} h_i.$$

A quadratic in $v$; minimize: $\sum g_i + v \sum h_i = 0$, so

$$v^* = -\frac{\sum_{i\in S} g_i}{\sum_{i\in S} h_i} = \frac{\sum_{i\in S} (y_i - p_i)}{\sum_{i \in S} p_i(1 - p_i)}.$$

This is one Newton step per leaf, and precisely XGBoost's leaf weight
$-\sum g / (\sum h + \lambda)$ with $\lambda = 0$; plugging $v^*$ back gives
the leaf's contribution to the split score
$\frac{(\sum g)^2}{2\sum h}$, which XGBoost maximizes when choosing splits.
Note the mechanism: the numerator is total residual (how wrong we are), the
denominator is total confidence-weighted curvature (how much a unit of score
actually moves probabilities). Leaves full of confident predictions
($p$ near 0/1, tiny $h$) take *larger* corrective steps per unit of
residual.
</details>

---

## Exercise 13. Derive the M-step update for GMM means

The M-step maximizes the expected complete-data log-likelihood
$Q(\theta) = \sum_{i,j} \gamma_{ij} \big[\log \pi_j + \log \mathcal N(x_i; \mu_j, \Sigma_j)\big]$
with responsibilities $\gamma_{ij}$ held fixed. Derive the update for
$\mu_j$, and explain why it is a *weighted* mean and why the answer does not
depend on $\Sigma_j$.

<details><summary>Solution</summary>

Only the Gaussian's quadratic term involves $\mu_j$:

$$Q(\mu_j) = -\frac12 \sum_i \gamma_{ij}\,(x_i - \mu_j)^\top \Sigma_j^{-1} (x_i - \mu_j) + \text{terms free of } \mu_j.$$

Gradient wrt $\mu_j$ (using $\nabla_\mu\, (x-\mu)^\top A (x-\mu) = -2A(x-\mu)$
for symmetric $A$):

$$\frac{\partial Q}{\partial \mu_j} = \sum_i \gamma_{ij}\, \Sigma_j^{-1}(x_i - \mu_j) = \Sigma_j^{-1} \sum_i \gamma_{ij}(x_i - \mu_j) = 0.$$

$\Sigma_j^{-1}$ is invertible, so multiply both sides by $\Sigma_j$; this is why
the update is covariance-free:

$$\sum_i \gamma_{ij} x_i = \Big(\sum_i \gamma_{ij}\Big)\mu_j \;\Rightarrow\; \boxed{\;\mu_j = \frac{\sum_i \gamma_{ij}\, x_i}{\sum_i \gamma_{ij}} = \frac{1}{N_j}\sum_i \gamma_{ij}\, x_i.\;}$$

It's a weighted mean because each point contributes to component $j$ in
proportion to the posterior probability that $j$ generated it, the
soft-label analogue of "the cluster mean", and the Hessian
$-\sum_i \gamma_{ij}\Sigma_j^{-1} \prec 0$ confirms a maximum. Sanity check
the limits: if $\gamma_{ij} \in \{0,1\}$ (hard assignments) this is exactly
the k-means center update (notes §8.1); if $\gamma_{ij} = 1/k$ for all
points (totally uncertain), every component's mean collapses to the global
mean, which is what happens for the first EM iteration from a
symmetric initialization.
</details>

---

## Exercise 14. One EM iteration by hand (1-D, two components)

Data: $x \in \{0, 1, 3, 4\}$. Model: two 1-D Gaussians with **fixed**
$\sigma^2 = 1$ and **fixed** $\pi_1 = \pi_2 = \tfrac12$; only the means are
learned. Initialize $\mu_1 = 0$, $\mu_2 = 4$. Compute all responsibilities in
closed form (show they reduce to a sigmoid), then the updated means. Use
$\sigma(4) \approx 0.9820$, $\sigma(8) \approx 0.99966$.

<details><summary>Solution</summary>

**E-step.** With equal weights and variances, the normalizing constants
cancel:

$$\gamma_1(x) = \frac{e^{-(x-\mu_1)^2/2}}{e^{-(x-\mu_1)^2/2} + e^{-(x-\mu_2)^2/2}} = \frac{1}{1 + \exp\!\big(\tfrac{(x-\mu_1)^2 - (x-\mu_2)^2}{2}\big)}.$$

Plug in $\mu_1 = 0$, $\mu_2 = 4$:
$(x-0)^2 - (x-4)^2 = 8x - 16$, so
$\gamma_1(x) = \sigma(8 - 4x)$, so the responsibility is a **sigmoid of a
linear function of $x$**, with decision boundary at the midpoint $x = 2$
(same structure as logistic regression, and systematically so: two equal-
covariance Gaussians induce a logistic posterior).

| $x$ | $\gamma_1(x) = \sigma(8-4x)$ | $\gamma_2(x)$ |
|-----|------------------------------|----------------|
| 0   | $\sigma(8) = 0.99966$        | 0.00034        |
| 1   | $\sigma(4) = 0.98201$        | 0.01799        |
| 3   | $\sigma(-4) = 0.01799$       | 0.98201        |
| 4   | $\sigma(-8) = 0.00034$       | 0.99966        |

Effective counts: $N_1 = 0.99966 + 0.98201 + 0.01799 + 0.00034 = 2.000$
(exactly, by the symmetry of the data about $x=2$), $N_2 = 2.000$.

**M-step** (weighted means, Exercise 13):

$$\mu_1 = \frac{0(0.99966) + 1(0.98201) + 3(0.01799) + 4(0.00034)}{2} = \frac{0.98201 + 0.05397 + 0.00136}{2} \approx 0.519,$$

$$\mu_2 = \frac{0(0.00034) + 1(0.01799) + 3(0.98201) + 4(0.99966)}{2} = \frac{0.01799 + 2.94603 + 3.99864}{2} \approx 3.481.$$

(Check: by the data's symmetry about 2, $\mu_2 = 4 - \mu_1$. ✓)

Interpretation: each mean moves toward the *center of mass of its soft
cluster*. $\mu_1$ ends above $\tfrac{0+1}{2} = 0.5$ because $x=3$ has
responsibility 0.018 for component 1 and $x=4$ has 0.00034. Iterating does not return to
$(0.5, 3.5)$: the next steps give $(0.555, 3.445)$, then $(0.559, 3.441)$, and
the means settle at $(\mu_1, \mu_2) \approx (0.559, 3.441)$, where the
responsibilities and means are mutually consistent (the fixed point of notes
§11.1's coupled equations). The small cross-component responsibilities keep both
means closer to the middle, $x = 2$, than $(0.5, 3.5)$. Had we run *hard* EM (k-means with these inits), the
assignment would be $\{0,1\}$ vs. $\{3,4\}$ and the means would jump to
exactly $(0.5, 3.5)$ in one step: hard assignment sets the 2%
cross-component responsibilities to zero.
</details>

---

## Exercise 15. Derive the ALS update as ridge regression

In matrix factorization with objective
$\sum_{(u,i)\in\Omega} (r_{ui} - p_u^\top q_i)^2 + \lambda\big(\sum_u \|p_u\|^2 + \sum_i \|q_i\|^2\big)$,
fix all item factors $Q$ and derive the optimal $p_u$ for a single user $u$.
Show the subproblem *is* ridge regression, verify invertibility, and check
the formula on this tiny case: $r = 1$ (scalar factors), user $u$ rated two
items with $q_1 = 2, q_2 = 1$, ratings $r_{u1} = 4, r_{u2} = 2$,
$\lambda = 1$.

<details><summary>Solution</summary>

With $Q$ fixed, the objective splits across users: $p_u$ appears only in

$$L(p_u) = \sum_{i \in \Omega_u} (r_{ui} - q_i^\top p_u)^2 + \lambda \|p_u\|^2.$$

Stack the factor vectors of the items $u$ rated as rows of
$Q_u \in \mathbb{R}^{|\Omega_u| \times r}$ and the ratings as
$r_u \in \mathbb{R}^{|\Omega_u|}$:

$$L(p_u) = \|r_u - Q_u p_u\|^2 + \lambda \|p_u\|^2.$$

This is *literally* ridge regression (Exercise 2) with design matrix $Q_u$,
targets $r_u$, weights $p_u$: the item factors act as the "features" of each
rating. By the identical derivation ($\nabla L = -2Q_u^\top(r_u - Q_u p_u) +
2\lambda p_u = 0$):

$$\boxed{\; p_u = (Q_u^\top Q_u + \lambda I)^{-1} Q_u^\top r_u. \;}$$

Invertibility: $Q_u^\top Q_u$ is PSD, so $Q_u^\top Q_u + \lambda I$ has
eigenvalues $\ge \lambda > 0$, so it is solvable even for a user who rated fewer
items than $r$ (the common case; this is why ALS regularization is not
optional). The item update $q_i = (P_i^\top P_i + \lambda I)^{-1} P_i^\top
r_i$ is symmetric. Since each half-sweep exactly minimizes its block, the
objective is monotonically non-increasing, by the same alternating-minimization
argument as for Lloyd's algorithm (Exercise 7).

**Tiny case.** $Q_u = \binom{2}{1}$, $r_u = \binom{4}{2}$:
$Q_u^\top Q_u = 4 + 1 = 5$, $Q_u^\top r_u = 8 + 2 = 10$, so
$p_u = 10/(5+1) = 5/3 \approx 1.667$. Predictions: $(3.33, 1.67)$ vs.
targets $(4, 2)$: shrunk below the unregularized fit ($\lambda = 0$ gives
$p_u = 2$, which predicts perfectly), because ridge trades a little data fit
for a smaller weight. Push $\lambda \to \infty$ and $p_u \to 0$: in the full model with biases, a
user with little data gets the prediction $\mu + b_u + b_i$, which is the
desired behavior for near-cold-start users.
</details>
