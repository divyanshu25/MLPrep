# Classical ML from First Principles

These notes derive everything. No "it can be shown". If a formula appears, we
build it. Companion code: `linear_models_scratch.ipynb`,
`trees_and_ensembles_scratch.ipynb`, `kmeans_pca_scratch.ipynb`.

Notation: $n$ = number of examples, $d$ = number of features,
$X \in \mathbb{R}^{n \times d}$ (rows are examples), $y \in \mathbb{R}^n$,
$w \in \mathbb{R}^d$. Bias is absorbed into $w$ by appending a constant-1
feature unless stated otherwise.

---

## 1. Linear regression

### 1.1 Where does squared loss come from? (Gaussian MLE)

Don't start with "minimize squared error". Start with a *model of how the data
was generated*, and squared error will fall out.

Assume each label is a linear function of the input plus Gaussian noise:

$$y_i = w^\top x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2) \text{ i.i.d.}$$

Then the likelihood of one observation is

$$p(y_i \mid x_i, w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - w^\top x_i)^2}{2\sigma^2}\right).$$

Maximum likelihood says: pick the $w$ that makes the observed data most
probable. Products of exponentials are awkward, so take the log (monotone, so
the argmax doesn't change):

$$\log p(y \mid X, w) = \sum_{i=1}^n \left[ -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right].$$

The first term doesn't depend on $w$. Maximizing the rest is exactly

$$\hat{w}_{\text{MLE}} = \arg\min_w \sum_i (y_i - w^\top x_i)^2.$$

**So "least squares" is not an arbitrary choice - it is the MLE under Gaussian
noise.** This is the master pattern of these notes: *loss = negative
log-likelihood of a noise model*. Swap the noise model and the loss changes:
Laplace noise $\Rightarrow$ absolute error (L1 loss, robust regression);
Bernoulli labels $\Rightarrow$ cross-entropy (§2).

**Interview trap:** "Why not absolute error?" Answer: both are fine; they
correspond to different noise assumptions. Squared error is MLE for Gaussian
noise and heavily punishes outliers (a residual of 10 costs 100); absolute
error is MLE for Laplace noise and is robust. Squared error also happens to be
differentiable everywhere and has a closed form.

### 1.2 The normal equation, derived

Write the total loss in matrix form:

$$L(w) = \|Xw - y\|^2 = (Xw - y)^\top (Xw - y) = w^\top X^\top X w - 2 y^\top X w + y^\top y.$$

This is a quadratic bowl in $w$ (convex, because $X^\top X \succeq 0$). Take
the gradient and set it to zero. Two facts we use:
$\nabla_w (w^\top A w) = 2Aw$ for symmetric $A$, and
$\nabla_w (b^\top w) = b$. So

$$\nabla L = 2 X^\top X w - 2 X^\top y = 0 \quad\Longrightarrow\quad \boxed{X^\top X \, w = X^\top y}$$

If $X^\top X$ is invertible (columns of $X$ linearly independent),
$\hat{w} = (X^\top X)^{-1} X^\top y$.

Tiny concrete example: fit $y = wx$ (no bias) to points $(1, 2), (2, 3)$.
$X^\top X = 1^2 + 2^2 = 5$, $X^\top y = 1\cdot 2 + 2 \cdot 3 = 8$, so
$\hat w = 8/5 = 1.6$. Check: residuals are $2 - 1.6 = 0.4$ and
$3 - 3.2 = -0.2$; loss $= 0.16 + 0.04 = 0.2$. Nudge $w$ to $1.7$: residuals
$0.3, -0.4$, loss $0.09 + 0.16 = 0.25 > 0.2$. Nudge to $1.5$: loss
$0.25 + 0 = 0.25$. Indeed 1.6 is the bottom of the bowl.

**When does $X^\top X$ fail to be invertible?** When columns of $X$ are
linearly dependent: duplicated features, one-hot columns that sum to the
all-ones bias column, or simply $d > n$. Then infinitely many $w$ achieve the
same minimum. Ridge (§3) fixes this: $X^\top X + \lambda I$ is always
invertible for $\lambda > 0$ because its eigenvalues are $e_i + \lambda > 0$,
where $e_i \ge 0$ are the eigenvalues of $X^\top X$.

### 1.3 Gradient descent on the same loss

With the mean loss $L(w) = \frac{1}{n}\|Xw - y\|^2$, the gradient is
$\nabla L = \frac{2}{n} X^\top (Xw - y)$, and the update is

$$w \leftarrow w - \eta \cdot \frac{2}{n} X^\top (Xw - y).$$

Read it as English: for each feature $j$, the gradient component is
(residual) × (feature value), summed over examples. If the model's prediction
is too high wherever feature $j$ is large, weight $j$ gets pushed down.
That's the entire mechanism.

Why bother when a closed form exists? Cost. The normal equation needs
$X^\top X$ ($O(nd^2)$) plus a $d \times d$ solve ($O(d^3)$). For $d$ in the
millions that's dead on arrival; GD is $O(nd)$ per step and SGD is $O(d)$ per
example. Also GD generalizes to any differentiable loss - normal equations
don't.

Convergence detail worth knowing: for a quadratic, GD converges iff
$\eta < 2/\lambda_{\max}$ where $\lambda_{\max}$ is the largest eigenvalue of
the Hessian $\frac{2}{n}X^\top X$; the convergence *rate* is governed by the
condition number $\lambda_{\max}/\lambda_{\min}$. Ill-conditioned (elongated
bowl) $\Rightarrow$ zig-zagging. This is why feature standardization speeds up
GD: it round-ifies the bowl.

### 1.4 Geometric view: projection

Forget calculus for a moment. The set of all possible predictions
$\{Xw : w \in \mathbb{R}^d\}$ is the **column space** of $X$ - a
$d$-dimensional plane inside $\mathbb{R}^n$ (one axis per *example*). The
target $y$ is a point in $\mathbb{R}^n$, generally not on that plane. Least
squares asks: which point on the plane is closest to $y$? Answer: the
**orthogonal projection** of $y$ onto the plane.

"Closest = orthogonal" means the residual $y - X\hat w$ must be perpendicular
to every column of $X$:

$$X^\top (y - X\hat w) = 0 \quad\Longleftrightarrow\quad X^\top X \hat w = X^\top y.$$

Same normal equation, zero calculus. The prediction is
$\hat y = X(X^\top X)^{-1}X^\top y = Hy$, where $H$ is the "hat matrix" - a
projection matrix ($H^2 = H$, $H^\top = H$). Two consequences interviewers
like: residuals are orthogonal to predictions, and residuals sum to zero
whenever a bias column is present (the residual is orthogonal to the all-ones
column).

---

## 2. Logistic regression

### 2.1 Why not just use MSE on labels?

You could run linear regression on $y \in \{0, 1\}$. Problems:

1. **Predictions escape $[0,1]$** - "probability 1.7" is meaningless, and far
   easy points keep pulling the boundary (a point at $x=+100$, correctly
   classified, still contributes huge squared error and drags the line).
2. **Wrong noise model.** Gaussian noise on a binary label is absurd; the
   right generative story is Bernoulli.
3. If you bolt a sigmoid on and *then* use MSE, the loss becomes
   **non-convex** in $w$, and the gradient contains a factor $\sigma'(z) =
   p(1-p)$ which vanishes when the model is confidently *wrong* ($p \approx 0$
   or $1$) - learning stalls exactly when it should be correcting hardest.
   Cross-entropy cancels this factor (derived in §2.3).

### 2.2 Sigmoid from log-odds

Model the **log-odds** as linear (odds = $p/(1-p)$, the natural scale on which
"evidence adds up"):

$$\log \frac{p}{1 - p} = w^\top x.$$

Why is this a sensible starting point? Because probability lives in $[0,1]$
but a linear function lives in $(-\infty, \infty)$; log-odds is a monotone
bijection between them. Each unit of $w^\top x$ multiplies the odds by $e$.
Now solve for $p$. Let $z = w^\top x$:

$$\frac{p}{1-p} = e^z \;\Rightarrow\; p = e^z - p e^z \;\Rightarrow\; p(1 + e^z) = e^z \;\Rightarrow\; p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} \equiv \sigma(z).$$

The sigmoid is not chosen for its S-shape; it's the *inverse of the log-odds*.
Concrete anchor points: $\sigma(0) = 0.5$, $\sigma(2) \approx 0.88$,
$\sigma(-2) \approx 0.12$, $\sigma(4) \approx 0.982$. And the identity you'll
use constantly: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. (Proof:
$\sigma = (1+e^{-z})^{-1}$, so $\sigma' = e^{-z}(1+e^{-z})^{-2} =
\sigma \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(1-\sigma)$.)

### 2.3 Cross-entropy and its gradient, by hand

Bernoulli likelihood of one example: $p(y_i \mid x_i) = p_i^{y_i}(1-p_i)^{1-y_i}$
where $p_i = \sigma(w^\top x_i)$. Negative log-likelihood over the dataset:

$$L(w) = -\sum_i \left[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \right].$$

That's binary cross-entropy - again, *loss = NLL of the noise model*.

Now the gradient, every step shown. Work with one example, drop the subscript
$i$. Chain: $L \to p \to z \to w$ where $z = w^\top x$, $p = \sigma(z)$.

**Step 1** - $\partial L / \partial p$:

$$\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1 - y}{1 - p} = \frac{-y(1-p) + (1-y)p}{p(1-p)} = \frac{p - y}{p(1 - p)}.$$

**Step 2** - $\partial p / \partial z = \sigma'(z) = p(1 - p)$ (from §2.2).

**Step 3** - multiply; the $p(1-p)$ **cancels**:

$$\frac{\partial L}{\partial z} = \frac{p - y}{p(1-p)} \cdot p(1-p) = p - y.$$

**Step 4** - $\partial z / \partial w = x$, so

$$\boxed{\nabla_w L = (p - y)\,x.}$$

Sanity check with numbers: $y = 1$, $p = 0.9$ → gradient $= -0.1 x$ (tiny
correction, nearly right already). $y = 1$, $p = 0.1$ → gradient $= -0.9 x$
(big push toward positive $z$). The error signal is exactly proportional to
how wrong the probability is. No vanishing.

**The deep pattern:** linear regression's gradient is $(\hat y - y)x$;
logistic regression's is $(p - y)x$; softmax regression's is
$(p_k - y_k)x$. All identical in form, and systematically so: squared
loss + identity link, and cross-entropy + sigmoid/softmax link, are all
*matched pairs* (canonical links of exponential-family models), and matched
pairs always yield residual-times-input gradients. If an interviewer asks
"why does cross-entropy pair with softmax?", this is the answer.

Also: cross-entropy in $w$ is **convex** (its Hessian is
$\sum_i p_i(1-p_i) x_i x_i^\top \succeq 0$), so GD finds the global optimum.

**Interview trap:** on perfectly linearly separable data, the MLE does not
exist - $\|w\| \to \infty$ drives the loss to 0 and probabilities to hard
0/1. Regularization (or early stopping) fixes this. This is why sklearn's
`LogisticRegression` regularizes by default.

---

## 3. Regularization

### 3.1 MAP: L2 = Gaussian prior, L1 = Laplace prior

MLE asks "which $w$ makes the data likely?" MAP adds "…while $w$ itself stays
plausible under a prior":

$$\hat w_{\text{MAP}} = \arg\max_w \; p(y \mid X, w)\, p(w) = \arg\min_w \; \underbrace{-\log p(y \mid X, w)}_{\text{data loss}} \; \underbrace{- \log p(w)}_{\text{regularizer}}.$$

**Gaussian prior** $w_j \sim \mathcal{N}(0, \tau^2)$:
$-\log p(w) = \sum_j \frac{w_j^2}{2\tau^2} + \text{const} = \frac{1}{2\tau^2}\|w\|_2^2 + \text{const}$.
With Gaussian noise likelihood you get exactly **ridge regression**:

$$\hat w = \arg\min_w \|Xw - y\|^2 + \lambda \|w\|_2^2, \qquad \lambda = \sigma^2/\tau^2.$$

Read $\lambda$: noisy data (big $\sigma^2$) or a tight prior (small $\tau^2$)
⇒ trust the data less, shrink more. Closed form (same derivation as §1.2):
$\hat w = (X^\top X + \lambda I)^{-1} X^\top y$ - and note this is invertible
for any $\lambda > 0$, curing collinearity.

**Laplace prior** $p(w_j) \propto e^{-|w_j|/b}$:
$-\log p(w) = \frac{1}{b}\|w\|_1 + \text{const}$ ⇒ **lasso**:

$$\hat w = \arg\min_w \|Xw - y\|^2 + \lambda \|w\|_1.$$

The Laplace density has a sharp peak at 0 - it puts much more mass exactly
near zero than a Gaussian does. That peak is where sparsity comes from.

### 3.2 Why L1 gives sparsity: the corner argument

**Geometric picture (constrained form).** Penalized and constrained problems
are equivalent: minimizing loss $+\lambda\|w\|$ is minimizing loss subject to
$\|w\| \le t$ for some $t$. Picture 2D weight space $(w_1, w_2)$: the
unregularized optimum sits somewhere off-origin, surrounded by elliptical
loss contours (level sets of the quadratic bowl). The constraint set for L2 is
a **disk** ($w_1^2 + w_2^2 \le t$); for L1 it is a **diamond**
($|w_1| + |w_2| \le t$) with its four corners *on the axes*. The solution is
the point where the smallest possible loss ellipse first touches the
constraint set. A growing ellipse touching a smooth disk can make first
contact anywhere on the circle - the touch point is generically off-axis, so
both weights stay nonzero (just shrunk). Touching a diamond, first contact is
overwhelmingly likely to happen **at a corner**, because corners poke out
toward the ellipse - and a corner has $w_1 = 0$ or $w_2 = 0$. Sparsity is
literally the geometry of corners. In $d$ dimensions the L1 ball's corners
and edges lie on coordinate subspaces (many weights exactly zero), and
contact on them is the norm, not the exception.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 580 240" role="img" aria-label="Two panels. Left: the L1 budget is a diamond with corners on the axes; the smallest loss contour around the optimum (2, 0.8) touches it at the corner (1, 0), where w2 is zero. Right: the L2 budget is a disk; the contour touches it at (0.93, 0.37), where both weights are nonzero." style="max-width:600px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <defs>
    <marker id="q5-ax" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity="0.5"/></marker>
  </defs>
  <g fill="currentColor" text-anchor="middle" font-size="12">
    <text x="140" y="14">L1 budget: |w&#8321;| + |w&#8322;| &#8804; 1</text>
    <text x="430" y="14">L2 budget: w&#8321;&#178; + w&#8322;&#178; &#8804; 1</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.5">
    <line x1="15" y1="150" x2="268" y2="150" marker-end="url(#q5-ax)"/>
    <line x1="85" y1="212" x2="85" y2="30" marker-end="url(#q5-ax)"/>
    <line x1="305" y1="150" x2="558" y2="150" marker-end="url(#q5-ax)"/>
    <line x1="375" y1="212" x2="375" y2="30" marker-end="url(#q5-ax)"/>
  </g>
  <g fill="currentColor" fill-opacity="0.7">
    <text x="258" y="166">w&#8321;</text>
    <text x="93" y="38">w&#8322;</text>
    <text x="548" y="166">w&#8321;</text>
    <text x="383" y="38">w&#8322;</text>
  </g>
  <polygon points="140,150 85,95 30,150 85,205" fill="#6366f1" fill-opacity="0.15" stroke="#6366f1" stroke-width="1.5"/>
  <circle cx="375" cy="150" r="55" fill="#6366f1" fill-opacity="0.15" stroke="#6366f1" stroke-width="1.5"/>
  <g fill="none" stroke="#10b981" stroke-width="1.5">
    <circle cx="195" cy="106" r="70.4"/>
    <circle cx="485" cy="106" r="63.5"/>
  </g>
  <g fill="none" stroke="#10b981" stroke-dasharray="3 3">
    <circle cx="195" cy="106" r="33"/>
    <circle cx="485" cy="106" r="33"/>
  </g>
  <g fill="currentColor">
    <circle cx="195" cy="106" r="3.5"/>
    <circle cx="485" cy="106" r="3.5"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="195" y="96">(2, 0.8)</text>
    <text x="485" y="96">(2, 0.8)</text>
  </g>
  <g stroke="#f43f5e" stroke-opacity="0.7">
    <line x1="140" y1="156" x2="140" y2="216"/>
    <line x1="426.1" y1="136" x2="426.1" y2="216"/>
  </g>
  <g fill="#f43f5e">
    <circle cx="140" cy="150" r="4.5"/>
    <circle cx="426.1" cy="129.6" r="4.5"/>
  </g>
  <g fill="#f43f5e" text-anchor="middle">
    <text x="140" y="230">(1, 0): w&#8322; = 0 exactly</text>
    <text x="426" y="230">(0.93, 0.37): both nonzero</text>
  </g>
</svg>
</div>

*The simplest case: loss $\|w - (2, 0.8)\|^2$, whose contours are circles
around $(2, 0.8)$ (green; dashed is a smaller level), with budget $t = 1$.
Left: the smallest contour that reaches the diamond touches it at the corner
$(1, 0)$, so $w_2 = 0$ exactly. Right: it touches the disk at
$(2, 0.8)/\|(2, 0.8)\| = (0.93, 0.37)$, both weights shrunk, neither zero.*

**Calculus view (1-D, concrete).** Minimize $f(w) = (w - a)^2 + \lambda|w|$.
For $w > 0$: $f' = 2(w - a) + \lambda = 0 \Rightarrow w = a - \lambda/2$.
Valid only if $a > \lambda/2$. Symmetrically for $w<0$. Otherwise the optimum
is $w = 0$ exactly. So the lasso solution is **soft-thresholding**:

$$w^* = \operatorname{sign}(a)\,\max(0, |a| - \lambda/2).$$

Numbers: $a = 3, \lambda = 2 \Rightarrow w^* = 2$. $a = 0.5, \lambda = 2
\Rightarrow w^* = 0$, *exactly* zero. Compare ridge in 1-D:
$f' = 2(w-a) + 2\lambda w = 0 \Rightarrow w^* = a/(1+\lambda)$ - for
$a=0.5, \lambda=1$, $w^* = 0.25$: shrunk, never zero. L2 shrinks
multiplicatively; L1 subtracts a constant and clips at zero. The kink of
$|w|$ at 0 (nonzero subgradient interval $[-\lambda, \lambda]$) is what lets
zero be a genuine optimum.

### 3.3 Effect on bias and variance

Regularization deliberately **adds bias to remove variance**. Shrinking
weights toward 0 makes the estimator less sensitive to which particular
training set you drew (variance ↓) at the cost of systematically
underestimating true effects (bias ↑). Test error
$\approx \text{bias}^2 + \text{variance} + \text{noise}$, so a small
$\lambda$ often lowers total test error even though training error rises.
Sweep $\lambda$: at $\lambda = 0$ you have the (unbiased, high-variance)
OLS/MLE; as $\lambda \to \infty$, $w \to 0$ (maximal bias, zero variance).
Somewhere in between lies the sweet spot; find it with a validation set or CV.

**Interview traps:** (1) Don't regularize the bias term - it just centers
predictions, penalizing it adds bias with no variance benefit. (2) Standardize
features first: the penalty $\|w\|$ is scale-dependent, so a feature measured
in millimeters vs. meters gets a 1000× different effective penalty.
(3) "Elastic net" = L1 + L2, useful when correlated features make lasso
arbitrarily pick one of a group.

---

## 4. Support vector machines

### 4.1 Maximum margin

Many hyperplanes separate two separable classes. Which one? Intuition: the
one farthest from the closest points on either side generalizes best - small
perturbations of the data won't flip predictions. Formalize: for a hyperplane
$w^\top x + b = 0$, the distance from point $x_i$ is
$|w^\top x_i + b| / \|w\|$. Scale $(w,b)$ so the closest points satisfy
$|w^\top x_i + b| = 1$ (allowed - the hyperplane is scale-invariant). Then
the margin (distance to the closest point) is $1/\|w\|$, and maximizing the
margin becomes the clean convex program

$$\min_{w, b} \tfrac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^\top x_i + b) \ge 1 \; \forall i, \qquad y_i \in \{-1, +1\}.$$

Only the points sitting exactly on the margin (constraint tight) determine
the solution - the **support vectors**. Delete any other point and nothing
changes. That's the sense in which SVMs are "sparse in examples".

### 4.2 Soft margin, hinge loss, and C

Real data isn't separable. Introduce slack $\xi_i \ge 0$ measuring how badly
each constraint is violated: $y_i(w^\top x_i + b) \ge 1 - \xi_i$, and pay for
slack:

$$\min_{w,b,\xi} \tfrac{1}{2}\|w\|^2 + C \sum_i \xi_i.$$

At the optimum, $\xi_i = \max(0, 1 - y_i(w^\top x_i + b))$ - so the whole
thing collapses to an unconstrained loss:

$$\min_{w,b} \; \sum_i \underbrace{\max(0,\, 1 - y_i(w^\top x_i + b))}_{\text{hinge loss}} + \tfrac{1}{2C}\|w\|^2.$$

**SVM = hinge loss + L2 regularization.** Hinge is flat (zero) once the
margin exceeds 1: correctly-and-confidently classified points contribute
nothing (unlike logistic loss, which never quite reaches zero - logistic
regression keeps caring about all points, SVM ignores the easy ones).

**C is inverse regularization.** Big $C$: violations are expensive → fit
training data hard, narrow margin, low bias / high variance, risk of
overfitting. Small $C$: wide margin, tolerate misclassifications, high bias /
low variance. (Trap: people say "C is the regularization parameter" - it's
the *inverse*; $\lambda \sim 1/C$.)

### 4.3 The kernel trick, from the dual

Form the Lagrangian of the hard-margin problem with multipliers
$\alpha_i \ge 0$, set $\partial/\partial w = 0$ and $\partial/\partial b = 0$:

$$w = \sum_i \alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0.$$

Substituting back gives the **dual**:

$$\max_{\alpha \ge 0} \; \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \, \langle x_i, x_j \rangle \quad \text{s.t.} \; \textstyle\sum_i \alpha_i y_i = 0.$$

Look hard at this: the data enters **only through inner products**
$\langle x_i, x_j \rangle$. Prediction too:
$f(x) = \sum_i \alpha_i y_i \langle x_i, x \rangle + b$. That's the door the
kernel trick walks through.

**What the trick actually buys you.** Suppose you want a nonlinear boundary
by mapping to features $\phi(x)$ - e.g. all degree-2 monomials of $d$
features, which has dimension $O(d^2)$; degree-$p$ has $O(d^p)$. Computing
and storing $\phi(x)$ explicitly is exponential-in-$p$ pain. But the dual
only ever needs $\langle \phi(x_i), \phi(x_j)\rangle$. If a function
$K(x, x') = \langle \phi(x), \phi(x')\rangle$ can be computed *directly*,
you never materialize $\phi$ at all. Concrete: in $d=2$ with
$\phi(x) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)$,

$$\langle \phi(x), \phi(z)\rangle = x_1^2 z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2 = (x_1z_1 + x_2z_2)^2 = \langle x, z\rangle^2.$$

One multiplication and a square replaces building 3-D (or, for degree 10 in
100 dims, ~$10^{13}$-D) feature vectors. The RBF kernel
$K(x,z) = \exp(-\gamma\|x-z\|^2)$ corresponds to an *infinite-dimensional*
$\phi$ - you get to run a linear method in a space you could never construct.
Any symmetric positive semi-definite $K$ is a valid kernel (Mercer).

Cost of admission: the dual works with the $n \times n$ Gram matrix, so
kernel methods scale with the *number of examples* ($O(n^2)$ memory,
$O(n^2)$–$O(n^3)$ time) rather than feature dimension. That's precisely why
they faded for web-scale data and neural nets (which learn $\phi$ instead of
fixing it) took over.

---

## 5. Decision trees

### 5.1 Entropy from first principles - what is a "bit"?

The information content of an event with probability $p$ is defined as
$\log_2(1/p)$ bits. Why this formula and not something else? Three axioms pin
it down: (1) certain events carry zero information ($p{=}1 \to 0$ bits);
(2) rarer = more informative (decreasing in $p$); (3) independent events'
information **adds** ($p_1 p_2 \to \log\frac{1}{p_1p_2} = \log\frac{1}{p_1} +
\log\frac{1}{p_2}$) - and additivity-under-multiplication forces a logarithm.
Base 2 makes the unit "bits": one fair coin flip = $\log_2 2 = 1$ bit - the
answer to exactly one yes/no question.

**Entropy** is the *expected* information of a draw:

$$H(Y) = -\sum_c p_c \log_2 p_c.$$

Operationally, it is a lower bound on the average number of yes/no questions
per draw needed to identify the classes of many draws coded together (one
draw alone always needs at least one question). Numbers:
a 50/50 label is $H = 1$ bit (you must ask one question). A 90/10 label:
$H = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.137 + 0.332 = 0.469$ bits  - 
you're barely uncertain. Pure node: $H = 0$. Uniform over 4 classes:
$H = 2$ bits.

### 5.2 Information gain

A split sends $n_L$ examples left, $n_R$ right. **Information gain** is the
expected reduction in entropy from learning which side an example falls on:

$$IG = H(\text{parent}) - \frac{n_L}{n} H(\text{left}) - \frac{n_R}{n} H(\text{right}).$$

Concrete: parent has 10 examples, 5/5 → $H = 1$. A split gives left = (4+, 1−)
and right = (1+, 4−). Each child: $H = -0.8\log_2 0.8 - 0.2 \log_2 0.2
\approx 0.722$. $IG = 1 - 0.5(0.722) - 0.5(0.722) = 0.278$ bits. A useless
split producing two 50/50 children has $IG = 0$. Tree building is greedy:
at each node, scan every feature and every threshold (midpoints of sorted
unique values suffice), pick the split with max $IG$, recurse; stop at
max depth / pure node / min samples. Greedy, not optimal - finding the
globally optimal tree is NP-hard.

**Gini vs. entropy.** Gini impurity $G = 1 - \sum_c p_c^2$ is the probability
of misclassifying a random draw if you label it by a random draw from the
node's distribution. For binary: $G = 2p(1-p)$, maxed at 0.5 with $G = 0.5$
(entropy maxes at 1). Both are concave, both zero at purity, they choose the
same split the vast majority of the time. Gini skips the $\log$, so it's
marginally cheaper; entropy penalizes mixed nodes slightly more. In practice:
an implementation detail, not a modeling decision. (Interview answer: "they
almost always agree; Gini is the default because it's cheaper.")

### 5.3 Why trees overfit

A tree with enough depth can carve the feature space into one box per training
example: 100% training accuracy on any dataset, i.e. **zero bias, huge
variance**. Deeper splits are made on ever-smaller samples - a leaf holding 3
examples is essentially fitting noise, and one changed training point can
restructure the entire subtree below the node where it altered the best split
(instability compounds down the tree). Axis-aligned boxes are also a jagged,
high-frequency function class. Controls: `max_depth`, `min_samples_leaf`,
`min_impurity_decrease`, or cost-complexity pruning. But the modern answer to
tree variance is not a better single tree - it's ensembles.

---

## 6. Ensembles

### 6.1 Bagging: why averaging reduces variance (derived)

Take $B$ identically distributed estimators $f_1, \dots, f_B$, each with
variance $\sigma^2$, with pairwise correlation $\rho$. Variance of the
average $\bar f = \frac{1}{B}\sum_b f_b$:

$$\operatorname{Var}(\bar f) = \frac{1}{B^2}\left[ \sum_b \operatorname{Var}(f_b) + \sum_{b \ne b'} \operatorname{Cov}(f_b, f_{b'}) \right] = \frac{1}{B^2}\left[ B\sigma^2 + B(B-1)\rho\sigma^2 \right]$$

$$\boxed{\operatorname{Var}(\bar f) = \rho \sigma^2 + \frac{1 - \rho}{B}\, \sigma^2.}$$

Read it: the second term dies as $B \to \infty$ - averaging kills the
*independent* part of the noise for free. But the first term,
$\rho\sigma^2$, is a **floor** you cannot average away. Numbers:
$\sigma^2 = 1$, $B = 100$: if $\rho = 0$, variance $= 0.01$; if $\rho = 0.5$,
variance $= 0.505$ - fifty times worse. Correlation, not ensemble size, is
the binding constraint.

**Bagging** (bootstrap aggregating) manufactures approximately-independent
estimators from one dataset: train each model on a bootstrap sample ($n$
draws *with replacement*; each sample omits $\approx 1/e \approx 37\%$ of
points - since $P(\text{point never drawn}) = (1 - 1/n)^n \to e^{-1}$), then
average (regression) or majority-vote (classification). It targets variance
only; bias stays roughly that of a single model. Hence: bag **low-bias,
high-variance** learners - deep trees are the perfect client. Bagging linear
regression is pointless (already low variance, and the average of linear
models is linear). The held-out 37% gives **out-of-bag error** - free
cross-validation.

### 6.2 Random forests: what decorrelation adds

Bagged trees are still correlated: if one feature is dominant, every tree
splits on it at the root and the trees look alike - $\rho$ stays high and the
variance floor $\rho\sigma^2$ stays high. Random forests attack $\rho$
directly: **at every split, consider only a random subset of
$m \approx \sqrt{d}$ features**. Strong features are forcibly benched in many
splits, so different trees discover different structure, $\rho$ drops, and by
the boxed formula the achievable ensemble variance drops. The trade: each
individual tree gets slightly worse, so $\sigma^2$ rises a little. The floor
is $\rho\sigma^2$, and subsampling lowers $\rho$ by more than it raises
$\sigma^2$, so the product falls. That single formula is the whole design
rationale of random forests.

The flip side: with only a few informative features, too-aggressive
subsampling leaves many splits with no useful feature to choose, so
$\sigma^2$ rises faster than $\rho$ falls. $m$ is a real hyperparameter, not
folklore.

### 6.3 Boosting: gradient descent in function space (derived)

Bagging builds independent models in parallel and averages; **boosting**
builds models *sequentially, each one fixing the errors of the current
ensemble*. It reduces bias (weak, shallow learners get combined into a strong
one), whereas bagging reduces variance. Here is the derivation that makes
"gradient boosting" a literal description, not a brand name.

We want to minimize $L(F) = \sum_i \ell(y_i, F(x_i))$ over functions $F$.
Treat the vector of predictions on the training set,
$\mathbf{F} = (F(x_1), \dots, F(x_n))$, as the "parameters". Gradient descent
on these parameters would update

$$F(x_i) \leftarrow F(x_i) - \eta \, \frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} \quad \text{for each } i.$$

Define the **pseudo-residuals** $r_i = -\partial \ell(y_i, F(x_i)) / \partial F(x_i)$.
Problem: the update above is only defined at the $n$ training points; we need
a function defined everywhere. Solution: **fit a weak learner $h_m$ (a small
tree) to the pseudo-residuals** $\{(x_i, r_i)\}$ - $h_m$ is the best
representable approximation of the negative gradient direction - then take
the step in function space:

$$F_m = F_{m-1} + \eta\, h_m.$$

The algorithm is complete. Check it against the standard special case:
for squared loss $\ell = \frac12 (y - F)^2$, $r_i = y_i - F(x_i)$ = the plain
residual - "fit a tree to the residuals and add it," the classic description.
For log-loss, $r_i = y_i - p_i$ (§2.3 again!). Boosting with a generic loss
is *exactly* gradient descent where each step direction is a tree. The
learning rate $\eta$ ("shrinkage", ~0.01–0.3) trades number of trees for
generalization: smaller steps, more trees, better test error.

Contrast worth stating in interviews: boosting keeps adding capacity, so
*more rounds can overfit* (use early stopping on a validation set); bagging
essentially cannot overfit by adding more trees (variance only goes down).

### 6.4 XGBoost's key ideas

XGBoost is gradient boosting plus engineering and second-order math:

- **Second-order Taylor expansion** of the loss: with $g_i = \partial_F \ell$,
  $h_i = \partial^2_F \ell$, the loss of adding tree $f$ is approximated as
  $\sum_i [g_i f(x_i) + \tfrac12 h_i f(x_i)^2] + \Omega(f)$. For a fixed tree
  structure with leaf values $w_j$ this is a per-leaf quadratic, solved in
  closed form: $w_j^* = -\frac{\sum_{i \in \text{leaf } j} g_i}{\sum_{i \in \text{leaf } j} h_i + \lambda}$,
  giving each candidate structure an exact quality score
  $-\tfrac12 \sum_j \frac{(\sum g_i)^2}{\sum h_i + \lambda} + \gamma T$ - so
  split decisions optimize the *actual* objective, not a proxy.
- **Built-in regularization** $\Omega(f) = \gamma T + \tfrac{\lambda}{2}\|w\|^2$
  (number of leaves + L2 on leaf values), plus shrinkage and row/column
  subsampling.
- **Systems**: histogram/approximate split finding, sparsity-aware default
  directions (native missing-value handling), cache-aware and parallel split
  evaluation. (LightGBM and CatBoost are siblings with different split/
  categorical strategies.)

Why gradient-boosted trees still win on tabular data: they natively handle
mixed feature types, scales, missing values, and non-smooth interactions,
with strong regularization - things MLPs need heavy tuning to match.

---

## 7. kNN and the curse of dimensionality

kNN has no training: predict by majority vote (or average) of the $k$ nearest
training points. $k$ controls bias/variance directly - $k{=}1$: zero training
error, jagged boundary, high variance; $k{=}n$: predict the global majority,
maximal bias. It's a *local* method: it works iff "near in feature space ⇒
similar label" and iff you have enough data for neighborhoods to be small.

**The curse, quantified.** In $[0,1]^d$ with uniformly spread data, to
capture a fraction $r$ of the data in a sub-cube you need side length
$s = r^{1/d}$. Numbers, for $r = 1\%$ of the data: $d{=}2 \Rightarrow s = 0.1$
(nicely local); $d{=}10 \Rightarrow s = 0.01^{0.1} \approx 0.63$;
$d{=}100 \Rightarrow s \approx 0.955$. In 100 dimensions your "neighborhood"
spans 95% of the range of every coordinate - the word "nearest" has lost its
meaning. Two more symptoms: (1) points concentrate near the *boundary* (the
fraction of the unit cube within 0.05 of a face is $1 - 0.9^d$: 10% in 1-D,
99.997% in 100-D); (2) distances concentrate - the ratio of farthest to
nearest neighbor distance → 1, so rankings by distance become noise. Sample
complexity to keep neighborhoods local grows **exponentially in $d$**.

Escape hatches: dimensionality reduction (PCA, §9), metric learning, or the
empirical fact that real data lies near low-dimensional manifolds. Practical
traps: **scale your features** (distance is scale-sensitive - a feature in
grams dominates one in kilograms 1000:1), use odd $k$ to break ties, and note
prediction is $O(nd)$ per query without KD-trees/LSH (which themselves
degrade in high $d$).

---

## 8. k-means

### 8.1 The objective and Lloyd's algorithm as alternating minimization

Objective (inertia / within-cluster sum of squares) over assignments
$z_i \in \{1..k\}$ and centers $\mu_1..\mu_k$:

$$J(z, \mu) = \sum_{i=1}^n \|x_i - \mu_{z_i}\|^2.$$

Joint minimization is NP-hard, but each block is easy with the other fixed:

- **Fix $\mu$, minimize over $z$:** the terms are independent per point;
  each point picks its nearest center: $z_i = \arg\min_j \|x_i - \mu_j\|^2$.
- **Fix $z$, minimize over $\mu$:** for cluster $j$,
  $\frac{\partial}{\partial \mu_j} \sum_{i: z_i = j} \|x_i - \mu_j\|^2 =
  -2\sum_{i: z_i=j}(x_i - \mu_j) = 0 \Rightarrow \mu_j = \frac{1}{n_j}\sum_{i: z_i=j} x_i$
  - the **mean** minimizes summed squared distance (this is *why* it's called
  k-*means*; with L1 distance the minimizer is the median → k-medians).

Lloyd's algorithm just alternates these two exact minimizations.

### 8.2 Why it converges

Each step minimizes $J$ over one block with the other held fixed, so
$J(z^{t+1}, \mu^t) \le J(z^t, \mu^t)$ and
$J(z^{t+1}, \mu^{t+1}) \le J(z^{t+1}, \mu^t)$: **$J$ never increases.** $J$ is
bounded below by 0, and there are finitely many assignments ($k^n$), so the
algorithm reaches a state where the assignment stops changing - a fixed
point - in finite time. (`kmeans_pca_scratch.ipynb` prints the trace; watch it
be monotone.) What is *not* guaranteed: a global optimum. Lloyd's halts at a
local minimum that depends entirely on initialization.

### 8.3 Failure modes (know these cold)

1. **Bad init → bad local optimum**: two centers land in one true cluster,
   another true cluster gets split or merged. Fixes: multiple restarts keep
   the lowest $J$ (sklearn `n_init`); **k-means++** seeding - pick each new
   center with probability ∝ squared distance from existing centers, which
   gives an $O(\log k)$ expected approximation guarantee.
2. **Wrong shapes**: the model is implicitly "spherical clusters, equal
   variance, similar sizes" (it's the hard-assignment limit of a Gaussian
   mixture with shared isotropic covariance). Elongated, ring-shaped, or
   very-different-density clusters get butchered. Use GMMs or spectral/
   density-based clustering (DBSCAN).
3. **You must choose $k$**, and $J$ always decreases with $k$ (at $k = n$,
   $J = 0$), so you can't pick $k$ by minimizing $J$ - use the elbow
   heuristic, silhouette score, or domain knowledge.
4. **Scale sensitivity**: it's squared Euclidean distance, so features must
   be standardized, and outliers drag means hard (means, unlike medians, are
   not robust).

---

## 9. PCA

Setup: center the data ($X$ has column means subtracted; call it $X_c$).
Sample covariance $\Sigma = \frac{1}{n-1} X_c^\top X_c \in \mathbb{R}^{d\times d}$.

### 9.1 Derivation 1: variance maximization

Find the unit direction $u$ maximizing the variance of the projections
$u^\top x_i$:

$$\operatorname{Var}(u^\top x) = \frac{1}{n-1}\sum_i (u^\top x_{c,i})^2 = u^\top \Sigma u, \qquad \text{maximize s.t. } u^\top u = 1.$$

Lagrangian: $u^\top \Sigma u - \lambda(u^\top u - 1)$; gradient zero gives

$$\Sigma u = \lambda u.$$

The stationary points are exactly the **eigenvectors** of $\Sigma$, and the
attained variance is $u^\top \Sigma u = \lambda$ - the eigenvalue itself. So:
top eigenvector = direction of max variance, its eigenvalue = that variance;
the second component maximizes variance orthogonal to the first, etc.
"Explained variance ratio" is $\lambda_j / \sum_l \lambda_l$.

### 9.2 Derivation 2: reconstruction-error minimization

Alternative goal: find the $k$-dim subspace (orthonormal basis $U \in
\mathbb{R}^{d\times k}$, $U^\top U = I$) minimizing squared reconstruction
error when each point is replaced by its projection $UU^\top x$:

$$\min_U \sum_i \|x_i - UU^\top x_i\|^2.$$

Expand one term using $\|a\|^2 = a^\top a$ and $U^\top U = I$:

$$\|x - UU^\top x\|^2 = x^\top x - 2x^\top UU^\top x + x^\top UU^\top UU^\top x = \|x\|^2 - \|U^\top x\|^2.$$

$\sum_i \|x_i\|^2$ is fixed, so **minimizing reconstruction error ≡
maximizing $\sum_i \|U^\top x_i\|^2$ ≡ maximizing projected variance**. The
two derivations are the same problem - Pythagoras splits each point's energy
into "kept" (variance along the subspace) and "lost" (reconstruction error),
and the lost energy is exactly the sum of the discarded eigenvalues,
$(n-1)\sum_{j > k} \lambda_j$. That's why the reconstruction-error-vs-$k$
curve in `kmeans_pca_scratch.ipynb` falls off a cliff once $k$ reaches the true
latent dimension.

### 9.3 SVD connection

Write the SVD $X_c = U S V^\top$. Then
$\Sigma = \frac{1}{n-1} V S^\top S V^\top$ - so the right singular vectors
$V$ **are** the eigenvectors of $\Sigma$ (the principal components), and
$\lambda_j = s_j^2 / (n-1)$. Libraries compute PCA via SVD of $X_c$ instead
of forming $\Sigma$: it's numerically better (forming $\Sigma$ squares the
condition number) and cheaper when $d \gg n$. (Our notebook forms $\Sigma$
for clarity, which is fine at its small size.) Projected coordinates
("scores") are $X_c V = US$.

**Traps:** center the data (uncentered "PCA" finds the direction of the mean,
not of variance); standardize if features have incomparable units (else PCA
is dominated by whatever has big numbers); components are only defined up to
sign; PCA is linear and unsupervised - the top-variance direction can be
*useless for a downstream classification* (variance ≠ relevance).

---

## 10. Naive Bayes

Bayes rule for classification:
$p(y \mid x) \propto p(y)\, p(x \mid y)$. Modeling the full joint
$p(x_1..x_d \mid y)$ needs exponentially many parameters (for binary
features, $2^d - 1$ per class). The **naive assumption**: features are
conditionally independent given the class,

$$p(x \mid y) = \prod_{j=1}^d p(x_j \mid y),$$

collapsing this to $O(d)$ parameters per class - each estimated by trivial
counting (Bernoulli/multinomial NB) or per-class univariate Gaussians
(Gaussian NB). Prediction:
$\hat y = \arg\max_y \log p(y) + \sum_j \log p(x_j \mid y)$ (always sum logs;
products of hundreds of small probabilities underflow).

Tiny spam example: $p(\text{spam}) = 0.5$;
$p(\text{"free"}\mid\text{spam}) = 0.6$ vs. $0.05$ given ham;
$p(\text{"meeting"}\mid\text{spam}) = 0.01$ vs. $0.3$. Email = {"free",
"meeting"}: spam score $0.5 \cdot 0.6 \cdot 0.01 = 0.003$; ham score
$0.5 \cdot 0.05 \cdot 0.3 = 0.0075$ → ham. Fast, interpretable, done.

**Laplace smoothing** is mandatory: a word never seen in spam gives
$p(x_j\mid\text{spam}) = 0$ and one factor vetoes the entire product. Add
pseudo-counts: $p(x_j \mid y) = \frac{\text{count} + \alpha}{\text{total} + \alpha \cdot |\text{vocab}|}$
(this is the posterior mean under a Dirichlet($\alpha$) prior, or
equivalently MAP under Dirichlet($\alpha + 1$): the same MLE→MAP move as §3).

Why it works despite a false assumption: classification only needs the
*argmax* to be right, not the probabilities. Correlated features (e.g.
repeated words) make NB **overconfident** - probabilities pushed toward 0/1  - 
but often leave the ranking intact. So: decent classifier, badly calibrated
probability estimator. It's a **generative** model (models $p(x, y)$) vs.
logistic regression's **discriminative** $p(y \mid x)$; the classic
Ng–Jordan result: NB reaches its (higher) asymptotic error faster - better in
tiny-data regimes - while logistic regression wins with enough data.
Gaussian NB with shared class covariance actually *induces* a linear
(sigmoid) decision rule - NB and logistic regression are the
generative/discriminative pair of the same family.

---

## 11. Gaussian mixture models and EM

Companion code: `gmm_em_scratch.ipynb`.

### 11.1 The latent-variable model, and why direct MLE breaks

Generative story: to produce a point, first roll a $k$-sided die to pick a
component, then draw from that component's Gaussian:

$$z_i \sim \text{Categorical}(\pi_1, \dots, \pi_k), \qquad x_i \mid z_i = j \;\sim\; \mathcal{N}(\mu_j, \Sigma_j).$$

We never observe $z_i$ - it's **latent**. Marginalizing it out gives the
density we actually fit:

$$p(x_i \mid \theta) = \sum_{j=1}^k \pi_j \, \mathcal{N}(x_i; \mu_j, \Sigma_j), \qquad \theta = \{\pi_j, \mu_j, \Sigma_j\}.$$

Tiny numbers to fix intuition. 1-D, two components, $\pi_1 = \pi_2 = 0.5$,
$\sigma = 1$ both, $\mu_1 = 0$, $\mu_2 = 4$. Point $x = 1$: unnormalized
densities $\propto e^{-1/2} = 0.607$ and $e^{-9/2} = 0.011$, so the posterior
over which die face produced it is $0.607/0.618 = 0.982$ vs. $0.018$. Point
$x = 2$ (equidistant): $0.5$ / $0.5$. These posteriors are the
**responsibilities** - hold that thought.

Now try MLE directly. The log-likelihood is

$$\ell(\theta) = \sum_{i=1}^n \log \underbrace{\sum_{j=1}^k \pi_j \,\mathcal{N}(x_i; \mu_j, \Sigma_j)}_{\text{a sum!}}$$

**The log of a sum.** The log cannot reach past the sum to the individual
Gaussians, so nothing factorizes. Compare the counterfactual where the labels
$z_i$ *were* observed: the complete-data log-likelihood
$\sum_i [\log \pi_{z_i} + \log \mathcal{N}(x_i; \mu_{z_i}, \Sigma_{z_i})]$
splits into per-component problems, each solved by counting and per-cluster
sample means/covariances. Missing labels are the entire difficulty.

Setting the gradient of $\ell$ to zero anyway is instructive:

$$\frac{\partial \ell}{\partial \mu_j} = \sum_i \underbrace{\frac{\pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}{\sum_l \pi_l \mathcal{N}(x_i; \mu_l, \Sigma_l)}}_{\equiv\, \gamma_{ij}} \Sigma_j^{-1}(x_i - \mu_j) = 0 \;\Rightarrow\; \mu_j = \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}.$$

The responsibilities $\gamma_{ij}$ appear on their own - but they *depend on*
$\mu_j$. A fixed-point equation, not a closed form. EM is exactly the natural
"iterate this fixed point" idea, made rigorous.

### 11.2 EM derived from Jensen's inequality (the ELBO)

Take any distribution $q_i(j)$ over the $k$ components for each point
(nonnegative, sums to 1). Multiply and divide inside the sum, then apply
**Jensen's inequality** - $\log$ is concave, so
$\log \mathbb{E}[A] \ge \mathbb{E}[\log A]$:

$$\log p(x_i \mid \theta) = \log \sum_j q_i(j)\, \frac{\pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}{q_i(j)} \;\ge\; \sum_j q_i(j) \log \frac{\pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}{q_i(j)} \;\equiv\; \mathcal{L}_i(q_i, \theta).$$

Summing over $i$ gives a lower bound on the log-likelihood - the **ELBO**
(evidence lower bound). How loose is it? Compute the gap exactly, using
$p(x_i, z_i{=}j) = \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)$ and
$p(z_i{=}j \mid x_i) = p(x_i, j)/p(x_i)$:

$$\log p(x_i) - \mathcal{L}_i = \sum_j q_i(j)\log p(x_i) - \sum_j q_i(j) \log\frac{p(x_i, j)}{q_i(j)} = \sum_j q_i(j) \log \frac{q_i(j)\, p(x_i)}{p(x_i, j)} = \mathrm{KL}\big(q_i \,\|\, p(z_i \mid x_i)\big) \ge 0.$$

(First equality: $\sum_j q_i(j) = 1$.) So the bound identity is exact:

$$\boxed{\;\log p(x_i \mid \theta) = \mathcal{L}_i(q_i, \theta) + \mathrm{KL}\big(q_i \,\|\, p(z_i \mid x_i, \theta)\big).\;}$$

EM is coordinate ascent on the ELBO:

- **E-step** (maximize over $q$, $\theta$ fixed): the KL is the only
  $q$-dependent slack; it is zero iff $q_i = p(z_i \mid x_i, \theta)$ - the
  posterior. Setting $q_i(j) = \gamma_{ij}$ makes the bound **tight**:
  ELBO $= \ell(\theta)$ at the current $\theta$.
- **M-step** (maximize over $\theta$, $q$ fixed): drop the $\theta$-free
  entropy term $-\sum q \log q$; what's left is the **expected complete-data
  log-likelihood**
  $Q(\theta) = \sum_i \sum_j \gamma_{ij} \big[\log \pi_j + \log \mathcal{N}(x_i; \mu_j, \Sigma_j)\big]$  - 
  and unlike $\ell$, the log now sits *inside* the sum over components, so it
  factorizes and has closed-form maximizers (§11.3).

**Monotonicity, proved in one line.** With $q^t$ the E-step posterior at
$\theta^t$:

$$\ell(\theta^{t+1}) \;\ge\; \mathcal{L}(q^t, \theta^{t+1}) \;\ge\; \mathcal{L}(q^t, \theta^t) \;=\; \ell(\theta^t).$$

First inequality: ELBO is always a lower bound. Second: the M-step maximized
over $\theta$. Equality: the E-step made the bound tight. **The
log-likelihood never decreases** - the same alternating-minimization shape as
Lloyd's algorithm (§8.2), and `gmm_em_scratch.ipynb` asserts this trace is
monotone. What is *not* guaranteed: a global optimum. EM climbs to a
stationary point that depends on initialization (in practice: init with
k-means, use restarts).

### 11.3 E-step and M-step updates, derived

**E-step** is just Bayes' rule per point:

$$\gamma_{ij} = p(z_i = j \mid x_i) = \frac{\pi_j \,\mathcal{N}(x_i; \mu_j, \Sigma_j)}{\sum_{l} \pi_l \,\mathcal{N}(x_i; \mu_l, \Sigma_l)}.$$

(Implementation note: compute in log-space and normalize with logsumexp;
products of $d$-dimensional Gaussian densities underflow - same lesson as
naive Bayes, §10.)

**M-step.** Maximize
$Q(\theta) = \sum_{i,j} \gamma_{ij}\big[\log \pi_j - \tfrac12 \log|\Sigma_j| - \tfrac12 (x_i - \mu_j)^\top \Sigma_j^{-1} (x_i - \mu_j)\big] + \text{const}$.
Write $N_j = \sum_i \gamma_{ij}$ (the "effective count" of component $j$;
$\sum_j N_j = n$).

*Means:* $\partial Q/\partial \mu_j = \sum_i \gamma_{ij} \Sigma_j^{-1}(x_i - \mu_j) = 0$.
$\Sigma_j^{-1}$ is invertible, so it cancels:

$$\boxed{\;\mu_j = \frac{1}{N_j}\sum_i \gamma_{ij}\, x_i\;}$$
 - the responsibility-**weighted** mean.

*Covariances:* using the two matrix-calculus facts
$\partial \log|\Sigma| / \partial \Sigma = \Sigma^{-1}$ and
$\partial\, (a^\top \Sigma^{-1} a) / \partial \Sigma = -\Sigma^{-1} a a^\top \Sigma^{-1}$,
setting $\partial Q/\partial \Sigma_j = 0$ gives
$\sum_i \gamma_{ij}\big[\Sigma_j^{-1} - \Sigma_j^{-1}(x_i-\mu_j)(x_i-\mu_j)^\top \Sigma_j^{-1}\big] = 0$;
multiply by $\Sigma_j$ on both sides:

$$\Sigma_j = \frac{1}{N_j} \sum_i \gamma_{ij} (x_i - \mu_j)(x_i - \mu_j)^\top.$$

*Mixing weights:* maximize $\sum_j N_j \log \pi_j$ subject to
$\sum_j \pi_j = 1$. Lagrangian $\sum_j N_j \log \pi_j + \lambda(1 - \sum_j\pi_j)$;
derivative: $N_j/\pi_j = \lambda \Rightarrow \pi_j \propto N_j$; the
constraint fixes $\lambda = n$:

$$\pi_j = \frac{N_j}{n}.$$

The pattern to say out loud in an interview: **every M-step update is the
obvious labeled-data estimator (cluster fraction, cluster mean, cluster
covariance) with hard counts replaced by responsibility weights.** EM =
"guess soft labels from current parameters; refit parameters as if those soft
labels were data; repeat."

**Trap - the unbounded likelihood.** Let one component's mean sit exactly on
a data point and shrink its variance: $\mathcal{N}(x_i; x_i, \sigma^2)
\propto 1/\sigma \to \infty$. The GMM likelihood has singular spikes, so the
"global MLE" doesn't even exist in the unconstrained problem. Fix: add
$\varepsilon I$ to covariances (sklearn's `reg_covar`) or put a prior on
$\Sigma$ (MAP - the §3 move again).

### 11.4 k-means is EM with hard assignments

Freeze the covariances to a shared spherical $\Sigma_j = \varepsilon I$ and
the weights to $\pi_j = 1/k$, and learn only the means. The responsibilities
become

$$\gamma_{ij} = \frac{\exp(-\|x_i - \mu_j\|^2 / 2\varepsilon)}{\sum_l \exp(-\|x_i - \mu_l\|^2 / 2\varepsilon)},$$

a softmax over negative squared distances with temperature $\varepsilon$.
Look at the ratio for two components:

$$\frac{\gamma_{ij}}{\gamma_{il}} = \exp\!\left(\frac{\|x_i - \mu_l\|^2 - \|x_i - \mu_j\|^2}{2\varepsilon}\right) \;\xrightarrow[\varepsilon \to 0]{}\; \begin{cases} \infty & \text{if } \mu_j \text{ is strictly nearer} \\ 0 & \text{if } \mu_l \text{ is} \end{cases}$$

Numbers: squared distances 1 and 2 to two centers. $\varepsilon = 1$:
$\gamma = \sigma(0.5) = 0.62$ - genuinely soft. $\varepsilon = 0.1$:
$\sigma(5) = 0.993$. $\varepsilon = 0.01$: $\sigma(50) \approx 1 - 2\cdot
10^{-22}$ - one-hot for all practical purposes. So as $\varepsilon \to 0$:

- **E-step → k-means assignment step**: responsibilities collapse to a
  one-hot on the nearest center, $z_i = \arg\min_j \|x_i - \mu_j\|^2$.
- **M-step → k-means update step**: the weighted mean
  $\sum_i \gamma_{ij}x_i / N_j$ becomes the plain mean of the assigned
  points.
- **Objective → inertia**: the expected complete log-likelihood reduces to
  $-\frac{1}{2\varepsilon}\sum_i \|x_i - \mu_{z_i}\|^2 + \text{const}$  - 
  maximizing it *is* minimizing the k-means objective $J$ from §8.1.

`gmm_em_scratch.ipynb` demonstrates this numerically: EM with tiny frozen
spherical covariances produces assignments identical to Lloyd's algorithm.
This is also the honest explanation of k-means' failure modes (§8.3): you
silently assumed spherical, equal-covariance, equal-weight clusters.

**When GMM over k-means:** you need *soft* memberships (a point can be 60/40
between clusters); clusters are elliptical or differently sized (full
$\Sigma_j$ learns a per-cluster Mahalanobis metric); you want an actual
**density** - score $p(x)$ for anomaly detection, sample new points, or
compare $k$ via likelihood-based criteria (BIC/AIC), none of which k-means
can do. When k-means: you just need a fast, robust partition - it's cheaper
($O(nkd)$ per iter, no $d\times d$ solves), has fewer parameters to estimate
($k$ full covariances cost $kd(d{+}1)/2$ parameters - hungry in high $d$),
and is the standard *initializer* for GMM anyway.

---

## 12. t-SNE and UMAP vs. PCA

### 12.1 What PCA can't do

PCA (§9) finds a **linear** map maximizing retained variance - equivalently,
it best preserves *large* (global) squared distances. Two things follow. (1)
Structure that isn't linearly separable in any projection is lost: two
concentric rings of radius 1 and 3 are perfectly separated in 2-D, but *every*
linear projection to 1-D overlaps them - no eigenvector can help, the classes
are radially, not directionally, organized. Same for a Swiss-roll manifold:
PCA squashes the roll flat and points far apart *along* the sheet land on top
of each other. (2) Variance ≠ neighborhood structure: PCA happily spends its
budget on a high-variance direction that separates nothing. For
visualization, the question is usually "**who is whose neighbor?**" - a local
question that calls for a method built from neighborhoods, not from global
variance.

### 12.2 t-SNE from first principles

**Step 1 - turn distances into probabilities.** For each point $i$, define
the probability that $i$ would "pick $j$ as its neighbor" under a Gaussian
centered at $x_i$:

$$p_{j\mid i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \ne i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \qquad p_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2n}.$$

(The symmetrization guarantees every point contributes; a point in a dense
blob and an outlier get comparable total mass.)

**Step 2 - the perplexity knob.** Each point gets its *own* bandwidth
$\sigma_i$, chosen by binary search so that the entropy of $p_{\cdot\mid i}$
hits a user-set target: $\text{Perp}(P_i) = 2^{H(P_i)}$ = the **effective
number of neighbors** (typical 5–50). Dense regions get small $\sigma_i$,
sparse regions large - the method adapts to local density. This is *the*
hyperparameter: small perplexity → fragmented micro-clusters (can shatter
real structure, or manufacture clusters out of noise); large → blurred toward
global structure.

**Step 3 - low-dimensional similarities with a heavy tail.** In the 2-D map
$\{y_i\}$, define

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}$$
 - a **Student-t with 1 degree of freedom** (Cauchy), *not* a Gaussian. This
asymmetry is the "t" in t-SNE and it is load-bearing. **The crowding
problem:** in $d$ dimensions the amount of room at distance $r$ from a point
grows like $r^{d-1}$ - there are vastly more "moderately far" pairs than a
2-D map has area to place them. With a Gaussian on both sides, all those
moderately-far pairs would need to be crammed into a thin 2-D annulus,
crushing the map's center. The fix: give the low-dim kernel a **heavy tail**
so a pair can sit *far apart* in the map while still keeping the small-but-
nonzero $q_{ij}$ that its moderate $p_{ij}$ demands. Numbers: compare
unnormalized kernels at distance $u$ - Gaussian $e^{-u^2/2}$ vs. t
$(1+u^2)^{-1}$. At $u = 2$: $0.135$ vs. $0.200$ (similar). At $u = 4$:
$0.00034$ vs. $0.059$ - the t-kernel carries **~175× more mass**. Moderate
high-D dissimilarity can be represented by a *large* map distance at low
cost, so clusters spread out and white space appears between them.

**Step 4 - the objective.** Match the two distributions by minimizing

$$\mathrm{KL}(P \,\|\, Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

with gradient descent on the $y_i$ (the map coordinates are the parameters).
The gradient has a beautifully mechanical form:

$$\frac{\partial\, \mathrm{KL}}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})\, \frac{y_i - y_j}{1 + \|y_i - y_j\|^2}$$
 - springs between all pairs, with force proportional to the similarity
mismatch (attract if $p > q$, repel if $q > p$), damped at long range by the
t-kernel. **Why KL and not something symmetric:** KL$(P\|Q)$ charges
$p \log(p/q)$ - putting *neighbors far apart* (big $p$, tiny $q$) costs a
fortune, while putting *non-neighbors close-ish* (tiny $p$) costs almost
nothing. t-SNE therefore fights hardest to preserve **local** structure and
treats global arrangement as negotiable. That's a design choice, and it's
also the source of every caveat below.

### 12.3 Honest limitations (interviewers probe these)

1. **Non-parametric.** t-SNE learns coordinates for *these* $n$ points, not a
   map $f: \mathbb{R}^d \to \mathbb{R}^2$. A new point cannot be embedded
   without rerunning (or training a separate regressor). So: no
   train/test-consistent features - **never feed t-SNE coordinates to a
   downstream model.**
2. **Between-cluster distances and cluster sizes are meaningless.** The
   per-point $\sigma_i$ expands dense clusters and shrinks sparse ones, and
   the KL objective doesn't price global layout - so "these two clusters are
   close in the map" and "this cluster is bigger" are artifacts, not
   findings. Only *neighborhood membership* is (approximately) trustworthy.
3. **Stochastic and hyperparameter-sensitive.** Different seeds give
   different maps; different perplexities give different cluster counts; low
   perplexity can conjure clusters from pure noise. Sanity-check any claimed
   structure across seeds and perplexities.
4. **Cost.** Naive $O(n^2)$; Barnes–Hut approximation gets $O(n \log n)$.
   Standard practice: PCA down to ~50 dims first (denoise + speed), then
   t-SNE.

### 12.4 UMAP in one paragraph

UMAP keeps the same skeleton - high-dim neighborhood graph, low-dim layout,
attract/repel optimization - with different machinery: it builds a $k$-NN
graph and converts distances to fuzzy edge weights (theory framed in
Riemannian-geometry/fuzzy-simplicial-set language, but operationally: a local,
adaptive kernel like t-SNE's), then minimizes a **cross-entropy** between
high- and low-dim edge weights by SGD with **negative sampling** - no global
normalization over all $n^2$ pairs, which is the main reason it's much
**faster** and scales to millions of points. Its repulsion structure tends to
preserve somewhat more global/inter-cluster geometry, and it provides a
`transform()` for new points (fit a reusable graph). The caveats, however,
transfer almost verbatim: distances between clusters and cluster densities in
a UMAP plot are still not quantitative evidence.

### 12.5 The standard question: "PCA or t-SNE - when?"

**PCA** when you need: features for a downstream model (deterministic,
train/test-consistent linear map), compression/denoising, speed on huge data,
interpretable directions (loadings), preserved global/variance structure, or
a preprocessing step. **t-SNE/UMAP** when you need: a 2-D *picture* of
cluster structure for human eyes - exploratory analysis only. They are not
competitors; the standard pipeline is PCA → 50 dims → t-SNE → plot. Trap
follow-ups: "Can t-SNE embed new points?" (no - non-parametric; UMAP can.)
"Two clusters are far apart in your t-SNE plot - what does that tell you?"
(Nothing quantitative.) "Why does t-SNE use a t-distribution?" (Crowding  - 
heavy tail lets moderate dissimilarities map to large distances; see the
175× number above.)

---

## 13. Collaborative filtering and matrix factorization

Companion code: `matrix_factorization_scratch.ipynb`. This is the recsys
baseline every interviewer expects before any deep-learning talk; the
production side (retrieval/ranking, serving) is module 6
(`../06_ml_systems/`).

### 13.1 The user–item matrix

Users rate (or click, watch, buy) items. Arrange into
$R \in \mathbb{R}^{m \times n}$, users × items, **mostly missing**: the
Netflix Prize matrix was ~480k users × 17.8k movies with 100M ratings - 98.8%
empty. Let $\Omega$ = the set of observed $(u, i)$ pairs. Two tasks: predict
missing entries (rating prediction) or rank each user's unseen items (top-$k$
recommendation - usually what production actually wants).

Tiny running example:

|       | M1 | M2 | M3 | M4 |
|-------|----|----|----|----|
| Alice | 5  | 4  | ?  | 1  |
| Bob   | 4  | 5  | 1  | ?  |
| Carol | 1  | ?  | 5  | 4  |
| Dave  | ?  | 1  | 4  | 5  |

Two taste groups hiding in plain sight: {Alice, Bob} love {M1, M2}; {Carol,
Dave} love {M3, M4}. Every method below is a way of discovering and
exploiting that structure.

### 13.2 Neighborhood CF: "people like you liked..."

**User–user.** Represent each user by their (sparse) row; measure similarity
by cosine over co-rated items,
$\text{sim}(u, v) = \frac{\sum_{i \in \Omega_u \cap \Omega_v} r_{ui} r_{vi}}{\|r_u\| \|r_v\|}$,
with the norms also taken over the co-rated items $\Omega_u \cap \Omega_v$
(Pearson = cosine after subtracting each user's mean rating - corrects for
harsh vs. generous raters). Numbers from the tiny matrix: Alice–Bob co-rated
{M1, M2}: $\frac{5\cdot4 + 4\cdot5}{\sqrt{41}\sqrt{41}} = 40/41 = 0.976$.
Alice–Carol co-rated {M1, M4}: $\frac{5\cdot1 + 1\cdot4}{\sqrt{26}\sqrt{17}}
= 9/21.0 = 0.43$. Bob is Alice's neighbor; predict Alice's M3 from Bob's
$r_{\text{Bob,M3}} = 1$: low, correctly. General prediction, mean-centered:

$$\hat r_{ui} = \bar r_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v)\,(r_{vi} - \bar r_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}.$$

**Item–item** is the same idea on columns ("people who liked X also liked
Y") and usually wins in practice: item neighborhoods are more stable over
time, items are often fewer than users, and the similarity matrix can be
precomputed offline (this was Amazon's classic recommender). Weaknesses of
both: with 99% sparsity, co-rating overlaps are tiny and similarities noisy;
$O(m^2)$ or $O(n^2)$ pairs; and no generalization - two users with disjoint
histories have similarity 0 even if their tastes are identical.

### 13.3 Matrix factorization: low-rank structure, and why plain SVD fails

The tiny matrix is morally **rank ~2**: a user is described by "how much do I
like taste-group A vs. B", an item by "how much am I an A vs. B movie".
Generalize: give each user a latent vector $p_u \in \mathbb{R}^r$ and each
item $q_i \in \mathbb{R}^r$, and model

$$\hat r_{ui} = p_u^\top q_i \qquad \Longleftrightarrow \qquad \hat R = P Q^\top \;\; (\text{rank} \le r).$$

A rating is the *alignment* of a taste vector with an item vector - the same
"dot product in a learned latent space" that §9.3's SVD provides. So why not
just SVD the ratings matrix? **Because Eckart–Young answers the wrong
question.** SVD gives the best rank-$r$ approximation in Frobenius norm  - 
a sum over **all** $mn$ entries. Our matrix has 99% of entries *missing*.
Treat them as zeros and the objective is dominated by reconstructing zeros:
the model learns "everyone is indifferent to almost everything" and drags all
predictions toward 0. Impute the mean instead, and 99% of your "data" is your
own imputation - the SVD dutifully reconstructs the fill, not the signal.
The honest objective sums over observed entries only:

$$\min_{P, Q} \sum_{(u,i) \in \Omega} (r_{ui} - p_u^\top q_i)^2 + \lambda\big(\textstyle\sum_u \|p_u\|^2 + \sum_i \|q_i\|^2\big),$$

and with a mask inside the sum, the SVD's closed form is gone - the problem
is non-convex in $(P, Q)$ jointly. (Regularization is not optional: a user
with 3 ratings and $r = 20$ free parameters is §3's overfitting story in
miniature.)

**Add biases before factors.** Most rating variance isn't interaction - it's
"this movie is widely loved" and "this user grades harshly". Model them
explicitly:

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i,$$

with $b_u, b_i$ regularized too. Netflix-Prize lore worth repeating: baseline
biases alone captured a large share of the achievable error reduction; the
factors then model genuine taste rather than popularity. (This is the same
"don't make the interesting part of the model explain the boring variance"
move as centering data before PCA, §9.)

### 13.4 ALS: alternating least squares, derived

The objective is non-convex jointly but **biconvex**: fix $Q$ and it is
convex in $P$; fix $P$ and convex in $Q$. Better: with $Q$ fixed, the
objective **decouples per user** - user $u$'s terms are

$$\sum_{i \in \Omega_u} (r_{ui} - p_u^\top q_i)^2 + \lambda \|p_u\|^2.$$

Stare at it: this is exactly **ridge regression** (§3.1) with design matrix
$Q_u$ (rows = factor vectors of the items $u$ rated), targets $r_u$, and
weights $p_u$. Closed form, straight from Exercise 2's derivation:

$$\boxed{\; p_u = (Q_u^\top Q_u + \lambda I)^{-1} Q_u^\top r_u \;}$$
 - an $r \times r$ solve per user, always invertible for $\lambda > 0$, and
embarrassingly parallel across users. Then fix $P$ and update every $q_i$
symmetrically from the users who rated it. Each half-step *exactly* minimizes
the objective over its block, so the loss is monotonically non-increasing  - 
the third appearance of the alternating-minimization pattern in these notes
(Lloyd's §8.2, EM §11.2, ALS here). Biases fold in via closed-form updates
too (a bias is a ridge regression on a constant feature; the scratch script
does exactly this).

**ALS vs. SGD.** SGD updates per observed rating
($e_{ui} = r_{ui} - \hat r_{ui}$; $p_u \mathrel{+}= \eta(e_{ui} q_i - \lambda
p_u)$, symmetric for $q_i$) - cheap per epoch, trivially handles biases,
needs a learning rate. ALS - no learning rate, parallelizes per row, and is
the standard choice for implicit-feedback weighting where the "sum over all
entries" trick makes SGD infeasible. Both are in
`matrix_factorization_scratch.ipynb`; they reach essentially the same RMSE.

### 13.5 Implicit feedback and BPR

Real logs rarely contain ratings - they contain clicks, watches, purchases:
**positive-only** signal where absence is ambiguous (didn't like it, or never
saw it?). Regressing on the 1s alone is degenerate (predict 1 everywhere).
Two standard fixes: (1) **implicit ALS** (Hu–Koren–Volinsky): treat all
unobserved entries as 0 but with low confidence and observed ones as 1 with
high confidence, and let ALS handle the now-dense weighted problem. (2)
**BPR** (Bayesian Personalized Ranking): stop predicting values, model the
*ranking*. Sample triples $(u, i, j)$ where $u$ interacted with $i$ but not
$j$, and maximize

$$\sum_{(u,i,j)} \log \sigma\big(\hat x_{ui} - \hat x_{uj}\big) - \lambda\|\Theta\|^2,$$

i.e. logistic regression (§2!) on score *differences*: "the probability $u$
prefers $i$ over $j$". SGD with sampled negatives; optimizes a smooth proxy
for AUC. This pairwise-ranking idea is the ancestor of most modern
recommender losses.

### 13.6 Cold start

A new item has no interactions, so its $q_i$ is whatever the regularizer
left it - zero. Same for a new user. Pure CF is structurally blind here: it
generalizes across the interaction graph, and a new node isn't connected to
it. Mitigations, roughly in order of deployment reality: popularity fallback;
**content features** (build $q_i$ from metadata/text/image embeddings so an
unseen item lands near similar seen ones); hybrid models mixing CF and
content; onboarding questionnaires (buy a few interactions cheaply);
exploration/bandits (show new items deliberately to gather signal fast).

### 13.7 Bridge to two-tower models

Matrix factorization *is* a two-tower model in its most degenerate form: the
"user tower" is an embedding lookup $u \mapsto p_u$, the "item tower" is
$i \mapsto q_i$, and the score is their dot product. Replace the lookups with
neural networks over **features** - user history, demographics, item
content - and you keep everything that makes MF deployable (a dot-product
score means candidate retrieval is a nearest-neighbor search over
precomputed item vectors, served by an ANN index) while fixing cold start:
an unseen item gets a vector from its features, not from interactions it
doesn't have. Module 6 (`../06_ml_systems/`) picks this up on the production
side: retrieval → ranking pipelines, ANN serving, and training two-tower
models at scale.

---

## Common interview questions

**Q1. Why does least squares correspond to Gaussian noise?**
Take Gaussian noise with $\sigma = 1$ and two residuals, 1 and 3. The
log-density of a residual $r$ is $-\tfrac12\log(2\pi) - r^2/2$, so the
negative log-likelihood is a constant plus $0.5 + 4.5 = 5$, which is half the
sum of squared residuals. In general, for $y_i = w^\top x_i + \varepsilon_i$
with $\varepsilon_i \sim \mathcal N(0, \sigma^2)$ i.i.d., the negative
log-likelihood is $\frac{1}{2\sigma^2}\sum_i (y_i - w^\top x_i)^2$ plus a term
that does not depend on $w$, so the maximum-likelihood $w$ is the
least-squares $w$ (§1.1). The loss is the negative log-density of the assumed
noise.

**If asked:** *What if the noise is Laplace?* Its log-density is $-|r|/b$
plus a constant, so the MLE minimizes absolute error, which grows linearly in
the residual and is less sensitive to outliers.

**Q2. When would you use the normal equation vs. gradient descent?**
With $d = 100$ features, $X^\top X$ is $100 \times 100$ and the solve costs on
the order of $d^3 = 10^6$ operations, so the normal equation gives the exact
answer with no learning rate to tune. With $d = 10^6$ features, $X^\top X$ has
$10^{12}$ entries (8 TB in float64) and the solve costs on the order of
$10^{18}$ operations, which is infeasible. A full gradient step costs $O(nd)$
and an SGD step $O(d)$ per example, and SGD can stream data that does not fit
in memory. Rule: closed form for small-to-medium $d$ with squared loss;
gradient methods for large $d$ or $n$, streaming data, or any loss without a
closed form, such as logistic loss. When $X^\top X$ is invertible both reach
the same minimizer, because the loss is a convex quadratic.

**If asked:** *What if $X^\top X$ is singular?* The columns are linearly
dependent (duplicate features, or $d > n$), so infinitely many $w$ fit
equally well; ridge adds $\lambda I$, which makes the matrix invertible.

**Q3. Derive the gradient of logistic loss.**
Take $y = 1$ and a confidently wrong prediction $p = 0.1$: the gradient of
the loss with respect to the logit $z$ is $p - y = -0.9$, a large correction.
With $L = -[y\log p + (1-y)\log(1-p)]$, $p = \sigma(z)$ and $z = w^\top x$,
the chain rule gives

$$\frac{\partial L}{\partial p} = \frac{p-y}{p(1-p)}, \qquad \frac{\partial p}{\partial z} = p(1-p), \qquad \frac{\partial z}{\partial w} = x \quad\Rightarrow\quad \nabla_w L = (p - y)\,x.$$

The factor $p(1-p)$ cancels. With squared loss $\tfrac12(p-y)^2$ on top of
the sigmoid it would not cancel, and the gradient on $z$ would be
$(p-y)\,p(1-p) = -0.9 \times 0.09 = -0.081$, about 11 times smaller exactly
when the prediction is most wrong.

**If asked:** *Is the loss convex in $w$?* Yes: its Hessian is
$\sum_i p_i(1-p_i)\,x_i x_i^\top$, which is positive semidefinite.

**Q4. Why does L1 produce exact zeros but L2 doesn't?**
Take one weight whose unpenalized best value is $a = 0.3$, so the data loss
is $(w - 0.3)^2$, and add a penalty with $\lambda = 1$. Ridge,
$(w - 0.3)^2 + w^2$, has its minimum at $0.15$; lasso, $(w - 0.3)^2 + |w|$,
has its minimum at exactly $0$. The difference is the slope of the penalty
at zero. The slope of $w^2$ is 0 there, so the total slope at $w = 0$ is
$-2a \ne 0$ and zero is never the minimum; ridge only rescales,
$w^* = a/(1+\lambda)$. The slope of $|w|$ is $\pm 1$ all the way to zero,
so moving off zero adds $\lambda = 1$ of penalty per unit while the data loss
falls by only $2|a| = 0.6$ per unit, and the weight stays at 0. In general
lasso gives soft thresholding,
$w^* = \operatorname{sign}(a)\max(0, |a| - \lambda/2)$ (§3.2).

**If asked:** *Geometric version?* The L1 constraint $|w_1| + |w_2| \le t$
is a diamond with corners on the axes, and the loss contours usually first
touch it at a corner, where one weight is 0; the L2 disk has no corners.

**Q5. What's the Bayesian interpretation of ridge and lasso?**
Take noise variance $\sigma^2 = 1$ and a prior $w_j \sim \mathcal N(0, \tau^2)$
with $\tau^2 = 0.25$. With residuals $r_i = y_i - w^\top x_i$, the negative
log-posterior is $\tfrac12\sum_i r_i^2 + \|w\|^2/(2 \cdot 0.25)$ plus a
constant, which equals $\tfrac12\big(\sum_i r_i^2 + 4\|w\|^2\big)$:
ridge with $\lambda = \sigma^2/\tau^2 = 4$. In general the MAP estimate
minimizes the negative log-likelihood plus $-\log p(w)$, so the regularizer
is $-\log p(w)$ (§3.1). A Gaussian prior gives the L2 penalty (ridge); a
Laplace prior $p(w_j) \propto e^{-|w_j|/b}$ gives $\|w\|_1/b$ (lasso). Noisier
data (larger $\sigma^2$) or a tighter prior (smaller $\tau^2$) gives a larger
$\lambda$.

**If asked:**

- *Why does the Laplace prior give zeros?* Its density has a kink at 0, which
  becomes the kink of $|w|$ in the loss; the Gaussian is smooth at 0.
- *Is this fully Bayesian?* No: MAP returns the posterior mode, a single
  point, while a full Bayesian treatment keeps the whole posterior.

**Q6. What is C in an SVM and how does it relate to over/underfitting?**
The soft-margin objective is $\tfrac12\|w\|^2 + C\sum_i \xi_i$, where the
slack $\xi_i$ measures how far point $i$ falls inside the margin. One
violation of $\xi = 0.5$ adds 50 to the objective at $C = 100$ and 0.005 at
$C = 0.01$. With large $C$ the minimum has few violations and a large
$\|w\|$, which is a narrow margin $1/\|w\|$: low bias, high variance,
overfitting risk. With small $C$ the minimum has a small $\|w\|$ and more
violations: wide margin, more bias. Dividing the objective by $C$ gives
hinge loss plus $\frac{1}{2C}\|w\|^2$, so $C$ plays the role of $1/\lambda$:
it is inverse regularization strength.

**If asked:** *What is $\xi_i$ at the optimum?*
$\max(0, 1 - y_i(w^\top x_i + b))$, the hinge loss of point $i$.

**Q7. What does the kernel trick buy you, concretely?**
Take $x = (1, 2)$, $z = (3, 1)$ and the degree-2 feature map
$\phi(x) = (x_1^2, \sqrt2\,x_1x_2, x_2^2)$. Explicitly,
$\phi(x) \cdot \phi(z) = 9 + 12 + 4 = 25$; the kernel
$K(x, z) = (x \cdot z)^2 = 5^2 = 25$ gives the same number from one 2-D dot
product. The SVM dual and its prediction rule use the data only through
inner products $\langle x_i, x_j\rangle$, so replacing each one by $K$
trains a linear classifier in $\phi$-space without computing $\phi$, which
for degree 10 in 100 dimensions has about $4 \times 10^{13}$ coordinates and
for the RBF kernel $\exp(-\gamma\|x - z\|^2)$ is infinite-dimensional. The
price is the $n \times n$ Gram matrix: $O(n^2)$ memory and $O(n^2)$ to
$O(n^3)$ training time, so kernel SVMs scale poorly with the number of
examples.

**If asked:** *Which functions are valid kernels?* Any symmetric $K$ whose
Gram matrix is positive semidefinite for every finite set of points
(Mercer's condition).

**Q8. Gini vs. entropy: does the choice matter?**
Rarely. For a binary node that is 90/10, Gini impurity $2p(1-p)$ is 0.18 and
entropy is 0.469 bits. Divided by their maxima at 50/50 (0.5 and 1), these
are 0.36 and 0.47: two curves of nearly the same shape, both concave, zero
at a pure node and largest at a uniform one. So they rank candidate splits
the same way in almost all cases. Gini needs no logarithm, so it is slightly
cheaper and is the usual default.

**If asked:** *When can they disagree?* Entropy assigns relatively more
impurity to slightly mixed nodes (0.47 vs. 0.36 at 90/10 after scaling), so
it can prefer a split with purer children; depth and leaf-size limits change
the tree far more than this choice.

**Q9. Why do random forests use feature subsampling on top of bagging?**
Average $B = 100$ trees, each with variance $\sigma^2 = 1$ and pairwise
correlation $\rho = 0.5$: the average has variance
$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 = 0.5 + 0.005 = 0.505$. More trees
shrink only the second term, so $\rho\sigma^2 = 0.5$ is a floor (§6.1).
Bagged trees are correlated because a dominant feature gives the best root split
in nearly every bootstrap sample, so the trees are similar. Restricting each
split to a random subset of about $\sqrt d$ features makes different trees
split on different features, which lowers $\rho$. Each tree gets somewhat
worse, so $\sigma^2$ rises; for example, if $\rho$ falls to 0.2 while
$\sigma^2$ rises to 1.2, the variance is $0.24 + 0.0096 \approx 0.25$, half
of before.

**If asked:** *Can subsampling go too far?* Yes: with few informative
features, a small subset often contains none of them, so $\sigma^2$ rises
faster than $\rho$ falls; the subset size is tuned.

**Q10. Bagging vs. boosting in one line each?**
Bagging: train many deep trees on bootstrap samples, in parallel, and average
them; this reduces variance, and adding trees does not increase overfitting.
Boosting: train shallow trees one after another, each fit to the current
ensemble's residuals (more generally, the negative gradient of the loss),
and add each with a small step; this reduces bias, and too many rounds
overfit, so the number of rounds is set by early stopping on validation
data. Concretely, if the true value of a point is 10 and the
ensemble predicts 7, the next boosting tree (with squared loss) is trained on
target 3 for that point; a bagged tree is trained on the original target 10.

**If asked:** *Why deep trees for bagging and shallow for boosting?* Bagging
reduces only variance, so it needs low-bias, high-variance members; boosting
reduces bias over many rounds, so each member can be a high-bias,
low-variance shallow tree.

**Q11. In what sense is gradient boosting "gradient descent"?**
Treat the training-set predictions $F(x_1), \dots, F(x_n)$ as the
parameters. For squared loss $\tfrac12(y - F)^2$, a point with $y = 10$ and
current prediction $F = 7$ has gradient $F - y = -3$, so a gradient step
moves that prediction in the direction $+3$, its residual. A gradient step
is defined only at the $n$ training points, so boosting fits a small tree
$h_m$ to the negative gradients $r_i = -\partial \ell(y_i, F(x_i))/\partial
F(x_i)$ (the pseudo-residuals), which gives a function defined everywhere,
and updates $F \leftarrow F + \eta h_m$ with learning rate $\eta$ (§6.3).
Each tree is the best tree approximation of one gradient step in function
space. For log loss with $F$ the logit, the pseudo-residual is $y - p$: a
positive example predicted at $p = 0.8$ gets target 0.2.

**If asked:** *What does XGBoost add?* A second-order expansion of the loss:
each leaf's value is minus the sum of the leaf's gradients divided by the
sum of its second derivatives plus $\lambda$, a regularized Newton step.

**Q12. Why does k-means always converge, and to what?**
Points 0, 2, 10, 12 on a line, $k = 2$, centers started at 0 and 2. The
objective $J = \sum_i \|x_i - \mu_{z_i}\|^2$ takes the values 164 (assign
each point to its nearest center), 56 (move the centers to the means, 0 and
8), 24 (point 2 switches to center 0), and 4 (centers move to 1 and 11);
then no assignment changes. Each step minimizes $J$ exactly over one block
with the other fixed: the nearest center is optimal for fixed centers, and
the mean minimizes summed squared distance for fixed assignments. So $J$
never increases, it is bounded below by 0, and there are finitely many
assignments ($k^n$), so the algorithm stops after finitely many steps at a
fixed point (§8.2). That fixed point is a local minimum that depends on the
initialization, not the global minimum, which is why k-means++ seeding and
multiple restarts are used.

**If asked:** *What does k-means++ do?* It picks each new initial center with
probability proportional to its squared distance from the nearest center
already chosen, which gives an expected $O(\log k)$ approximation to the
optimal $J$.

**Q13. Why must you standardize features for kNN, k-means, SVM, and
regularized models, but not for trees?**
Take income in dollars and age in years. Two people who differ by 1,000
dollars in income and 40 years in age have squared distance
$1000^2 + 40^2 = 1{,}001{,}600$, of which age contributes 0.16%, so the
distance is almost entirely income. kNN, k-means and RBF-kernel SVMs use
Euclidean distance, so they need features on comparable scales. Penalized
models are also scale-sensitive: measuring a feature in kilometers instead of meters
multiplies its fitted weight by 1000 and its L2 penalty by $10^6$. A tree
only compares one feature against a threshold, and any increasing transform
of a feature (rescaling, log) keeps the order of its values, so the same
partitions are available and the tree is unchanged.

**If asked:** *Fit the scaler on what data?* On the training set only, then
apply the same means and standard deviations to validation and test data;
fitting on all data leaks test information into training.

**Q14. State the curse of dimensionality with one number.**
To enclose 1% of points spread uniformly in the unit cube $[0,1]^d$, a
sub-cube needs side $0.01^{1/d}$: 0.1 in 2 dimensions, 0.63 in 10, and 0.955
in 100. So in 100 dimensions the neighborhood holding the nearest 1% of the
data spans 95.5% of the range of every feature, and it is not local.
Distances also concentrate: the ratio of the farthest to the nearest
neighbor distance approaches 1 as $d$ grows, so ranking points by distance
carries little information. Keeping neighborhoods at a fixed side $s$ with
$k$ points in each needs about $k/s^d$ samples, which grows exponentially in
$d$.

**If asked:** *Why does kNN still work on some high-dimensional data?* Real
data such as images lies near a low-dimensional manifold, so its effective
dimension is far below the number of raw features.

**Q15. PCA: variance maximization or reconstruction minimization?**
Both, and they are the same problem. Project the point $x = (3, 4)$ onto the
unit direction $u = (1, 0)$: the projection is $(3, 0)$, the kept squared
length is 9, the reconstruction error is $\|(0, 4)\|^2 = 16$, and
$9 + 16 = 25 = \|x\|^2$. In general, for an orthonormal basis $U$,
$\|x - UU^\top x\|^2 = \|x\|^2 - \|U^\top x\|^2$, and $\sum_i \|x_i\|^2$ is
fixed by the data, so minimizing total reconstruction error is maximizing
total projected squared length, which for centered data is the projected
variance (§9.2). Both are solved by the top $k$ eigenvectors of the
covariance matrix, equal to the top right singular vectors of the centered
data matrix, and the remaining error is $(n-1)$ times the sum of the
discarded eigenvalues.

**If asked:** *Why center first?* Without centering, the top direction
points roughly at the data mean instead of along the spread of the data.

**Q16. Why is naive Bayes often fine for classification but bad for
probabilities?**
Take a spam filter where the word "free" has likelihood ratio
$p(\text{free} \mid \text{spam}) / p(\text{free} \mid \text{ham}) = 0.6/0.05
= 12$ and the prior is 50/50, so seeing "free" gives spam probability
$12/13 = 0.92$. Now add a second feature that is an exact copy of the first.
Naive Bayes treats the two as independent given the class and multiplies by
12 twice: odds 144, probability $144/145 = 0.993$, although the copy adds no
information. Correlated features are counted more than once, which pushes
posteriors toward 0 or 1. The predicted class is spam either way, so the
argmax, and with it accuracy, often survives while the probabilities are
overconfident. If the features are conditionally independent given the
class, the model is correctly specified and, with the true per-feature
distributions, it is the Bayes-optimal classifier.

**If asked:** *How do you fix the probabilities?* Recalibrate on held-out
data with Platt scaling (a logistic regression on the model's score) or
isotonic regression.

**Q17. Why does logistic regression fail on perfectly separable data?**
Two points in 1-D: $x = -1$ with label 0 and $x = +1$ with label 1, model
$p = \sigma(wx)$. Each point's loss is $\log(1 + e^{-w})$: 0.313 at $w = 1$,
0.0067 at $w = 5$, $4.5 \times 10^{-5}$ at $w = 10$. The loss decreases with
every increase in $w$ and approaches 0 without reaching it, so no finite $w$
minimizes it: the MLE does not exist, and gradient descent increases $w$
without bound. The same happens whenever a hyperplane separates the classes:
scaling up its $w$ moves every $p_i$ toward its label. An L2 penalty
$\lambda\|w\|^2$ grows with $\|w\|$ while the data loss falls toward 0, so
the sum has a finite minimizer; early stopping also returns a finite $w$.

**If asked:** *Does the direction of $w$ settle?* Yes: under gradient
descent $w/\|w\|$ converges to the hard-margin SVM direction, while $\|w\|$
grows like $\log t$ in the number of steps $t$.

**Q18. Hinge vs. logistic loss?**
Write the margin of a point as $m = y\,f(x)$ with $y \in \{-1, +1\}$. At
$m = 2$, a correct and confident point, hinge loss $\max(0, 1 - m)$ is 0 and
logistic loss $\log(1 + e^{-m})$ is 0.127; at $m = -1$, a misclassified
point, they are 2 and 1.31. Hinge is exactly zero for every point with
margin at least 1, so the solution depends only on the points on or inside
the margin (the support vectors), and its output $f(x)$ is a score, not a
probability. Logistic loss is smooth and never zero, so every point
contributes to the gradient, and $\sigma(f(x))$ is a probability estimate,
calibrated when the model is well specified. Both are convex.

**If asked:** *Are they upper bounds on 0-1 loss?* Hinge is (it equals 1 at
$m = 0$); logistic is after dividing by $\ln 2$, since
$\log_2(1 + e^{0}) = 1$.

**Q19. Where does the 37% out-of-bag figure come from?**
With $n = 10$ points, a bootstrap sample makes 10 draws with replacement; a
given point is missed by one draw with probability 0.9, so by all ten with
probability $0.9^{10} = 0.349$. For $n = 100$ it is $0.99^{100} = 0.366$,
and as $n \to \infty$, $(1 - 1/n)^n \to e^{-1} \approx 0.368$. So each tree
is trained on about 63% of the distinct points, and the other 37% are out of
bag for that tree. Predicting each point with only the trees that did not
train on it gives the out-of-bag error, an estimate of test error that needs
no separate validation set.

**If asked:** *Is the OOB estimate biased?* Slightly pessimistic: each
point's OOB prediction averages only about 37% of the trees, and a smaller
ensemble has higher variance than the full forest.

**Q20. Why do gradient-boosted trees still beat neural nets on tabular
data?**
Take a fraud feature, account age in days, from 0 to 10,000, where risk
drops sharply after day 30. A tree represents this with one split,
"age < 30", whatever the scale or skew of the feature, while an MLP needs the
feature rescaled and several units to approximate the step with smooth
functions. The same holds for missing values (XGBoost learns a default
branch for them at each split), mixed feature types, and sharp interactions
between features: trees handle them without preprocessing. Boosting reduces
bias tree by tree, and shrinkage, depth limits, row and column subsampling,
and XGBoost's penalty $\gamma T + \tfrac{\lambda}{2}\|w\|^2$ on the number of
leaves $T$ and the leaf values $w$ control variance. On small-to-medium
tabular datasets, MLPs need heavy preprocessing and tuning to match this.

**If asked:** *When do neural nets win on tabular data?* With very large
datasets, or when some columns are text, images, or sequences that a network
can embed.

**Q21. What convergence guarantee does EM actually have?**
In a 1-D two-component GMM, the log-likelihood never goes down from one EM
iteration to the next, yet runs from different starting means can stop at
different values, and a run can diverge to $+\infty$: if a component's mean
lands on one data point and its $\sigma$ shrinks, that point's density term
$\pi_j/(\sqrt{2\pi}\,\sigma)$ (with $\pi_j = 0.5$) is about 20 at
$\sigma = 0.01$ and 200 at $\sigma = 0.001$. The guarantee is monotone
ascent:

$$\ell(\theta^{t+1}) \;\ge\; \mathcal{L}(q^t, \theta^{t+1}) \;\ge\; \mathcal{L}(q^t, \theta^t) \;=\; \ell(\theta^t).$$

The ELBO $\mathcal L$ is a lower bound on $\ell$ for every $q$ (first
inequality), the M-step maximizes it over $\theta$ (second), and the E-step
set $q^t$ to the posterior, which makes the bound tight at $\theta^t$ (the
equality) (§11.2). If $\ell$ is bounded above, its values converge, but only
to a stationary point that depends on the initialization, not to the global
maximum. In practice: add a small constant to each covariance diagonal
(which keeps $\ell$ bounded), initialize with k-means, and use several
restarts.

**If asked:** *How fast does it converge?* Linearly; the rate depends on how
much information the missing labels carry, so heavily overlapping components
converge slowly.

**Q22. GMM vs. k-means: relationship and when to prefer which?**
Take a point at squared distance 1 from one center and 2 from another, in a
GMM with shared covariance $\varepsilon I$ and equal weights. Its
responsibility for the nearer center is $\sigma\big(\frac{2 - 1}{2\varepsilon}\big)$:
0.62 at $\varepsilon = 1$, 0.993 at $\varepsilon = 0.1$, and within
$2 \times 10^{-22}$ of 1 at $\varepsilon = 0.01$. As $\varepsilon \to 0$ the
E-step becomes nearest-center assignment and the weighted-mean M-step
becomes the plain mean, so k-means is EM on this constrained GMM (§11.4).
Prefer a GMM for soft memberships, clusters that are elongated or of
different sizes (a full covariance per cluster), or a density $p(x)$ for
anomaly scores, sampling, or choosing $k$ by BIC (log-likelihood penalized by
parameter count). Prefer k-means for speed ($O(nkd)$ per iteration, no
covariances to estimate or to collapse), and as the initializer for a GMM.

**If asked:** *What do full covariances cost?* $k\,d(d+1)/2$ parameters: 10
clusters in 100 dimensions need 50,500, which takes a lot of data to
estimate.

**Q23. PCA vs. t-SNE: when do you use each?**
On scikit-learn's 8×8 digits dataset (1,797 images, 64 pixels), the top two
principal components keep 28.5% of the variance and the ten digit classes
overlap in the 2-D PCA plot, while a t-SNE plot of the same data shows ten
mostly separate clusters. Use PCA, a linear, deterministic map that applies
the same matrix to new points, for compression, denoising, and features for
downstream models. Use t-SNE, which is nonlinear, stochastic, and fits
coordinates for these $n$ points rather than a function, only to look at
neighborhood and cluster structure: neighbors are preserved, but distances
between clusters and cluster sizes in the plot carry no quantitative
information, and a new point cannot be embedded without rerunning it.
Standard pipeline: PCA to about 50 dimensions, then t-SNE to 2 for the plot.

**If asked:** *Why a t-distribution in the map?* At map distance 4 the
t-kernel $(1 + u^2)^{-1}$ is 0.059 versus 0.00034 for the Gaussian
$e^{-u^2/2}$, about 175 times more, so moderately dissimilar points can sit
far apart in the map; this relieves the crowding problem (§12.2).

**Q24. Why can't you just run SVD on a ratings matrix?**
The average Netflix Prize user rated about 200 of 17,770 movies. Fill the
missing entries with 0, and the SVD objective, squared error over all
entries, spends about 99% of that user's terms on reconstructing zeros, so the
fitted values for unseen movies are close to 0, a rating nobody gave.
Fill with the mean instead, and 99% of the objective is your own fill value,
which the SVD then reconstructs. Eckart-Young's optimality is for error over
all entries, the wrong objective here; the right one sums over the observed
entries $\Omega$ only, with L2 penalties on the factors:

$$\min_{P, Q} \sum_{(u,i) \in \Omega} (r_{ui} - p_u^\top q_i)^2 + \lambda\Big(\sum_u \|p_u\|^2 + \sum_i \|q_i\|^2\Big).$$

It has no closed form and is non-convex jointly, so it is solved by SGD or
ALS; in ALS each user update with item factors fixed is a ridge regression,
$p_u = (Q_u^\top Q_u + \lambda I)^{-1} Q_u^\top r_u$, where $Q_u$ stacks the
factor vectors of the items $u$ rated and $r_u$ holds those ratings.

**If asked:** *Why add user and item biases?* Much of the rating variance is
"this user rates harshly" or "this movie is widely liked"; the model
$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$ leaves the factors to model
taste.

**Q25. What is cold start and how do you handle it?**
A movie added today has no ratings, so in the matrix factorization objective
its vector $q_i$ appears only in the penalty $\lambda\|q_i\|^2$, which is
minimized at $q_i = 0$. Its score $p_u^\top q_i$ is then 0 for every user,
and collaborative filtering cannot rank it; a new user has the same problem
with $p_u$. Fixes, roughly in order of how often they are deployed: fall
back to popularity; build the vector from content features (genre, text,
image embeddings) so a new item starts near similar existing items; combine
CF and content in a hybrid model; ask new users a few onboarding questions;
and show new items to some users on purpose (exploration) to collect
interactions quickly. Content-based vectors are the idea behind two-tower
models: two networks map user features and item features to vectors and
the score is their dot product, so an unseen item still gets a vector.

**If asked:** *How do you test a cold-start fix offline?* Hold out whole
items (or users), not individual ratings, so the test set contains only
items the model never saw in training.

## Exit checklist

- [ ] Cold, on paper: derive squared loss from Gaussian MLE, then the normal equation - 5 min
- [ ] Derive the logistic-regression gradient: sigmoid + cross-entropy → (p − y)x - 5 min
- [ ] State L2 = Gaussian prior, L1 = Laplace prior, and give the corner argument for why L1 sparsifies - 4 min
- [ ] Explain the kernel trick from the SVM dual: where K(x, x') enters and why you never form φ(x) - 4 min
- [ ] Write the bagging variance formula with correlation ρ and say what random-forest decorrelation adds - 4 min
- [ ] Explain boosting as gradient descent in function space: what each new tree fits and why - 3 min
- [ ] Derive PCA both ways (variance maximization, reconstruction error) and connect to SVD - 5 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
