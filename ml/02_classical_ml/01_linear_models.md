# Linear models: linear regression, logistic regression, regularization

Companion notebook: `linear_models_scratch.ipynb`.

Notation: $n$ examples, $d$ features, $X \in \mathbb{R}^{n \times d}$ with one
example per row, labels $y \in \mathbb{R}^n$, weights $w \in \mathbb{R}^d$.
The bias is folded into $w$ by adding a feature that is always 1.

---

## 1. Linear regression

### 1.1 The model and the loss

Linear regression predicts a number as a weighted sum of features,
$\hat y = w^\top x$. For a house, $\text{price} = 50 + 2 \times \text{size}$
is a linear model with weights 50 (the bias) and 2.

The weights are chosen to minimize the **squared error**
$\sum_i (y_i - w^\top x_i)^2$. Squared error is maximum likelihood when the
errors are Gaussian (Probability §3.4.3). It punishes large misses heavily:
a miss of 10 costs 100. If the errors have occasional outliers, absolute
error (L1) is the maximum-likelihood loss instead and is less affected by
them.

### 1.2 The normal equation

The squared error is a bowl-shaped function of the weights, and the best
weights sit at the bottom, where the slope is zero in every direction. The
gradient of $\|Xw - y\|^2$ is $2X^\top(Xw - y)$ (Calculus §2.3.3). Setting
it to zero gives the **normal equation**:

$$\boxed{\color{#6366f1}{X^\top X\, w = X^\top y}}$$

When $X^\top X$ can be inverted, solving it gives the weights directly:

$$\hat w = (X^\top X)^{-1} X^\top y$$

Example: fit $y = wx$ to the points $(1, 2)$ and $(2, 3)$. Then
$X^\top X = 1^2 + 2^2 = 5$ and $X^\top y = 1 \cdot 2 + 2 \cdot 3 = 8$, so
$\hat w = 1.6$. The squared error is $0.4^2 + 0.2^2 = 0.2$; at $w = 1.5$ or
$w = 1.7$ it is 0.25, so 1.6 is the minimum.

$X^\top X$ has no inverse when one feature can be built exactly from the
others (the columns are **linearly dependent**). Example: height in
centimeters and height in inches as two features. Weights $(1, 0)$, $(0, 2.54)$
and $(2, -2.54)$ all make the same predictions, because 1 inch = 2.54 cm,
so $2.54 \times \text{inches} = \text{cm}$. There is no single best answer. The same happens with one-hot columns that add up to the bias
column, or with more features than examples ($d > n$). Ridge regression
(§3.1) fixes this.

### 1.3 Gradient descent

With the mean loss $\frac1n\|Xw - y\|^2$, each step is

$$w \leftarrow w - \eta \cdot \frac{2}{n} X^\top (Xw - y)$$

For weight $j$, the gradient is the average of (prediction error) × (feature
$j$). If predictions are too high on examples where feature $j$ is large,
weight $j$ goes down.

This uses the same gradient as the normal equation, but differently. The
normal equation sets the gradient to zero and solves for $w$ in one step.
Gradient descent takes many small steps downhill; the steps stop changing
$w$ when the gradient reaches zero, which is exactly when
$X^\top X\, w = X^\top y$. So both end at the same weights. In the example
of §1.2, starting from $w = 0$, gradient descent moves toward 1.6 a little
each step.

#### Why not always use the normal equation?

Count the work in its two steps:

1. **Build $X^\top X$.** It is a $d \times d$ table. Each entry is a dot
   product of two feature columns, each of length $n$, so $n$
   multiplications. $d^2$ entries times $n$ each is $nd^2$.
2. **Solve $X^\top X\, w = X^\top y$.** Solving $d$ equations in $d$
   unknowns costs $O(d^3)$.

So the total is about $nd^2 + d^3$. With $d = 10^6$ features,
$d^3 = 10^{18}$, which is infeasible.

A gradient step is much cheaper. Computing $X^\top(Xw - y)$ takes two steps:

1. **Predictions, $Xw$:** each of the $n$ examples needs a dot product of
   its $d$ features with $w$, so $nd$ multiplications. Subtract $y$ to get
   the $n$ errors.
2. **Gradient, $X^\top(\text{errors})$:** each of the $d$ weights needs
   the sum over $n$ examples of (error × that feature), again $nd$
   multiplications.

Together that is $nd + nd = 2 \times n \times d$ multiplications per step.

#### Why feature scale matters

Suppose one feature is house size in square
meters (around 100) and another is number of bedrooms (around 3). Raise
each weight by 1:

- size weight +1: each predicted price goes up by about 100 (the size);
- bedrooms weight +1: each predicted price goes up by about 3.

The same nudge moves the predictions about 30 times more through the size
weight, so the loss is far more sensitive to it. The loss bowl is steep in
the size direction and almost flat in the bedrooms direction: a long,
narrow valley.

- The learning rate has to be small enough for the steep direction, or the
  steps overshoot and blow up. Precisely, gradient descent converges only if
  $\eta < 2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue
  of $\frac2n X^\top X$, the curvature in the steepest direction.
- With that small learning rate, progress along the flat direction is very
  slow, and the steps zig-zag across the valley (Linear algebra §1.5). The
  ratio of the steepest to the flattest curvature is the **condition
  number**; the larger it is, the slower gradient descent goes.

Standardizing every feature to mean 0 and variance 1 makes the bowl closer
to round, so one learning rate suits every direction.

### 1.4 Least squares as a projection

There is a second way to see the normal equation, with geometry instead of
calculus. Use the example from §1.2: the feature values are $x = (1, 2)$
and the targets are $y = (2, 3)$, one entry per example.

- Whatever $w$ you pick, the predictions are $w \cdot (1, 2)$: a point on the
  line through $(1, 2)$. You can't get any prediction vector off that line.
- The targets $(2, 3)$ are not on the line, so the best you can do is the
  point on the line closest to $(2, 3)$.
- The closest point is where the miss, $y - \hat y$, is perpendicular to the
  line. Here $\hat y = 1.6 \cdot (1, 2) = (1.6, 3.2)$, the miss is
  $(0.4, -0.2)$, and $(0.4, -0.2) \cdot (1, 2) = 0.4 - 0.4 = 0$.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 360 290" role="img" aria-label="Plane with one axis per example. The line through the origin and (1, 2) holds every possible prediction w times (1, 2). The target y = (2, 3) is off the line. The closest point on the line is y-hat = (1.6, 3.2), and the miss from y-hat to y, (0.4, -0.2), meets the line at a right angle." style="max-width:380px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="lp-ax" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity="0.5"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="50" y1="270" x2="330" y2="270" marker-end="url(#lp-ax)"/>
    <line x1="50" y1="270" x2="50" y2="12" marker-end="url(#lp-ax)"/>
  </g>
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="110" y1="267" x2="110" y2="273"/><line x1="170" y1="267" x2="170" y2="273"/><line x1="230" y1="267" x2="230" y2="273"/>
    <line x1="47" y1="210" x2="53" y2="210"/><line x1="47" y1="150" x2="53" y2="150"/><line x1="47" y1="90" x2="53" y2="90"/><line x1="47" y1="30" x2="53" y2="30"/>
  </g>
  <g fill="currentColor" fill-opacity="0.6" font-size="10" text-anchor="middle">
    <text x="110" y="285">1</text><text x="170" y="285">2</text><text x="230" y="285">3</text>
    <text x="40" y="214">1</text><text x="40" y="154">2</text><text x="40" y="94">3</text><text x="40" y="34">4</text>
  </g>
  <g fill="currentColor" fill-opacity="0.7" font-size="10.5">
    <text x="326" y="262" text-anchor="end">example 1</text>
    <text x="58" y="18">example 2</text>
  </g>
  <line x1="50" y1="270" x2="179" y2="12" stroke="#6366f1" stroke-width="2"/>
  <text x="186" y="28" fill="#6366f1" font-size="11">all predictions w · (1, 2)</text>
  <circle cx="110" cy="150" r="3.5" fill="#6366f1"/>
  <text x="102" y="154" text-anchor="end" fill="#6366f1" font-size="11">(1, 2)</text>
  <line x1="146" y1="78" x2="170" y2="90" stroke="#f43f5e" stroke-width="2"/>
  <path d="M 154 82 L 150 90.1 L 142 86" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
  <circle cx="170" cy="90" r="4" fill="currentColor"/>
  <circle cx="146" cy="78" r="4" fill="#10b981"/>
  <text x="178" y="86" fill="currentColor" font-size="11">target y = (2, 3)</text>
  <text x="138" y="70" text-anchor="end" fill="#10b981" font-size="11">ŷ = (1.6, 3.2)</text>
  <text x="166" y="112" fill="#f43f5e" font-size="11">miss (0.4, −0.2)</text>
</svg>
</div>

*One axis per example. Every possible prediction lies on the blue line
$w \cdot (1, 2)$. The target $y$ is off the line, and the closest point on it
is $\hat y$, where the miss meets the line at a right angle.*

With more features, the predictions can be any combination of the columns
of $X$, and the miss must be perpendicular to every column:

$$X^\top (y - X\hat w) = 0$$

which is the normal equation again. One consequence: when $X$ has a bias
column of ones, the misses are perpendicular to $(1, 1, \dots, 1)$, which
means they sum to zero.

---

## 2. Logistic regression

### 2.1 The model

Logistic regression predicts a probability for a yes/no label, such as "is
this email spam?". It computes a score $z = w^\top x$ and turns it into a
probability with the **sigmoid**:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

<div style="margin:1.25rem 0">
<svg viewBox="0 0 380 235" role="img" aria-label="The sigmoid curve: probability p against score z from -6 to 6. It is 0.5 at z = 0, rises in an S shape, and flattens toward 0 on the left and 1 on the right. Marked points: sigma(-2) = 0.12, sigma(0) = 0.5, sigma(2) = 0.88, sigma(4) = 0.98." style="max-width:400px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="40" y1="200" x2="345" y2="200"/>
    <line x1="190" y1="205" x2="190" y2="30"/>
  </g>
  <g stroke="currentColor" stroke-opacity="0.25" stroke-dasharray="4 4">
    <line x1="40" y1="40" x2="340" y2="40"/>
    <line x1="40" y1="120" x2="340" y2="120"/>
  </g>
  <g fill="currentColor" fill-opacity="0.6" font-size="10">
    <text x="346" y="44">1</text><text x="346" y="124">0.5</text><text x="346" y="204">0</text>
    <text x="40" y="216" text-anchor="middle">−6</text><text x="115" y="216" text-anchor="middle">−3</text><text x="190" y="216" text-anchor="middle">0</text><text x="265" y="216" text-anchor="middle">3</text><text x="340" y="216" text-anchor="middle">6</text>
  </g>
  <text x="190" y="230" text-anchor="middle" fill="currentColor" fill-opacity="0.75" font-size="11">score z = wᵀx</text>
  <text x="196" y="26" fill="currentColor" fill-opacity="0.75" font-size="11">probability p = σ(z)</text>
  <polyline points="40.0,199.6 45.0,199.5 50.0,199.4 55.0,199.3 60.0,199.1 65.0,198.9 70.0,198.7 75.0,198.4 80.0,198.1 85.0,197.6 90.0,197.1 95.0,196.5 100.0,195.7 105.0,194.8 110.0,193.7 115.0,192.4 120.0,190.8 125.0,188.9 130.0,186.7 135.0,184.0 140.0,180.9 145.0,177.3 150.0,173.1 155.0,168.3 160.0,163.0 165.0,157.0 170.0,150.4 175.0,143.3 180.0,135.8 185.0,128.0 190.0,120.0 195.0,112.0 200.0,104.2 205.0,96.7 210.0,89.6 215.0,83.0 220.0,77.0 225.0,71.7 230.0,66.9 235.0,62.7 240.0,59.1 245.0,56.0 250.0,53.3 255.0,51.1 260.0,49.2 265.0,47.6 270.0,46.3 275.0,45.2 280.0,44.3 285.0,43.5 290.0,42.9 295.0,42.4 300.0,41.9 305.0,41.6 310.0,41.3 315.0,41.1 320.0,40.9 325.0,40.7 330.0,40.6 335.0,40.5 340.0,40.4" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <line x1="140.0" y1="200" x2="140.0" y2="180.9" stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 3"/>
  <circle cx="140.0" cy="180.9" r="3.5" fill="#f43f5e"/>
  <text x="132.0" y="184.9" text-anchor="end" fill="#f43f5e" font-size="11">σ(−2) = 0.12</text>
  <line x1="190.0" y1="200" x2="190.0" y2="120.0" stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 3"/>
  <circle cx="190.0" cy="120.0" r="3.5" fill="#f43f5e"/>
  <text x="182.0" y="114.0" text-anchor="end" fill="#f43f5e" font-size="11">σ(0) = 0.5</text>
  <line x1="240.0" y1="200" x2="240.0" y2="59.1" stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 3"/>
  <circle cx="240.0" cy="59.1" r="3.5" fill="#f43f5e"/>
  <text x="248.0" y="71.1" text-anchor="start" fill="#f43f5e" font-size="11">σ(2) = 0.88</text>
  <line x1="290.0" y1="200" x2="290.0" y2="42.9" stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 3"/>
  <circle cx="290.0" cy="42.9" r="3.5" fill="#f43f5e"/>
  <text x="296.0" y="58.9" text-anchor="start" fill="#f43f5e" font-size="11">σ(4) = 0.98</text>
</svg>
</div>

*The sigmoid squashes any score into a probability between 0 and 1. A score
of 0 means 0.5; each step away from 0 pushes the probability toward 0 or 1,
quickly at first and then more and more slowly.*

**Where the sigmoid comes from.** The score $z$ can be any number, so we
need a way to turn any number into a probability. The **log-odds** do the
reverse, turning a probability into any number:

| $p$ | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| odds $p/(1-p)$ | 0.25 | 1 | 4 |
| log-odds $\log\frac{p}{1-p}$ | $-1.39$ | 0 | $1.39$ |

Logistic regression assumes the score is the log-odds. Solving
$\log\frac{p}{1-p} = z$ for $p$ gives the sigmoid:

$$p = \frac{1}{1 + e^{-z}}$$

Check: $z = 1.39$ gives $p = 1/(1 + e^{-1.39}) = 0.8$, as in the table.

Since $z = b + w_1 x_1 + w_2 x_2 + \dots$, each feature that is present adds
its weight to the log-odds. Example, a spam model with bias $-2$:

| email contains | log-odds $z$ | $p(\text{spam})$ |
|---|---|---|
| neither word | $-2$ | 0.12 |
| "free" (weight 1.39) | $-2 + 1.39 = -0.61$ | 0.35 |
| "free" and "winner" (weight 1.1) | $-0.61 + 1.1 = 0.49$ | 0.62 |

Each word pushes the log-odds up by its own weight, whatever else the email
contains. In odds terms, "free" multiplies the odds of spam by
$e^{1.39} = 4$.

The derivative is $\sigma'(z) = p(1-p)$. To see it, differentiate
$(1 + e^{-z})^{-1}$ and split the result into two factors:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \underbrace{\frac{1}{1 + e^{-z}}}_{p} \cdot \underbrace{\frac{e^{-z}}{1 + e^{-z}}}_{1 - p} = p(1-p)$$

The second factor is $1 - p$ because the two fractions add up to 1. At
$z = 0$, $p = 0.5$ and the slope is $0.25$, the steepest point of the curve.

### 2.2 The loss and its gradient

The model outputs $p$, its probability that the label is 1. A good model
gives high probability to the label that actually happened:

- if the email is spam ($y = 1$), the model gave the true label probability
  $p$;
- if it isn't ($y = 0$), the model gave the true label probability $1 - p$.

Maximum likelihood (Probability §3.4.2) picks the weights that make these
probabilities as large as possible over all examples. Maximizing a product
of probabilities is the same as minimizing the sum of their $-\log$, so each
example's loss is $-\log p$ when $y = 1$ and $-\log(1 - p)$ when $y = 0$.
Writing both cases in one line gives **binary cross-entropy**:

$$L = -\big[y \log p + (1 - y)\log(1 - p)\big]$$

When $y = 1$ the second term is zero, and when $y = 0$ the first is. For a
spam email: $p = 0.9$ costs $-\log 0.9 = 0.11$, but $p = 0.1$ costs
$-\log 0.1 = 2.3$. Confident wrong answers are charged heavily.

Its gradient has the same simple form as linear regression's:

$$\nabla_w L = (p - y)\,x$$

By the chain rule, $\frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}$,
$\frac{\partial p}{\partial z} = p(1-p)$ and $\frac{\partial z}{\partial w} = x$.
The $p(1-p)$ cancels.

Example with $y = 1$: if $p = 0.9$ the gradient is $-0.1x$, a small
correction; if $p = 0.1$ it is $-0.9x$, a large one. The correction is
proportional to how wrong the probability is.

**The same gradient shape appears in three models:**

| model | output | loss | gradient |
|---|---|---|---|
| linear regression | $\hat y = w^\top x$ | squared error | $(\hat y - y)\,x$ |
| logistic regression | $p = \sigma(w^\top x)$ | binary cross-entropy | $(p - y)\,x$ |
| softmax regression | $p_k = \text{softmax}(z)_k$ | cross-entropy | $(p_k - y_k)\,x$ |

In each row the gradient is (prediction − target) × input. This is not a
coincidence: each loss is the negative log-likelihood that matches its
output, and that pairing cancels the extra factors in the chain rule.

**The loss is convex.** Its curvature (second derivative) is

$$\sum_i p_i(1 - p_i)\, x_i x_i^\top$$

which is never negative in any direction, since $p_i(1-p_i) \ge 0$. So the
loss is a single bowl with no false minima, and gradient descent finds the
global minimum.

### 2.3 Why not linear regression on the labels?

The labels are just the numbers 0 and 1, and §1 already has a method for
predicting numbers: linear regression. Interviewers often ask why
classification doesn't just reuse it.

The outputs aren't probabilities. Example: predict whether a student passes
an exam (1) or fails (0) from the hours they studied. Four students studied
1, 2, 3 and 4 hours; the first two failed and the last two passed. A
straight line fit to these points is $\hat y = 0.4x - 0.5$. It predicts 1.9
for a student who studies 6 hours and $-0.5$ for one who studies 0 hours: a
"190%" and a "$-50\%$" chance of passing. Logistic regression passes the
score through a sigmoid (§2.1), so every output is between 0 and 1.

**Interview trap:** if the classes can be separated perfectly by a line, the
weights grow without limit, because larger weights always push the
probabilities closer to 0 and 1 and lower the loss. Regularization (§3) or
early stopping fixes this, which is why scikit-learn's `LogisticRegression`
regularizes by default.

---

## 3. Regularization

### 3.1 Ridge and lasso

**Why weights need limiting.** Predict 20 students' exam scores from hours
studied, plus 30 useless features such as shoe size and birth month. With
more features than students, least squares can always find weights that fit
all 20 scores exactly, by giving shoe size and birth month whatever weights
make the numbers come out right. Those weights fit the noise in these 20
scores, so the model fails on the next student.

Regularization adds a penalty on the size of the weights to the loss. Now a
weight stays large only if it lowers the error enough to pay for itself.
Hours studied does; shoe size doesn't, so its weight shrinks toward 0:

$$\text{ridge: } \|Xw - y\|^2 + \lambda\|w\|_2^2 \qquad\qquad \text{lasso: } \|Xw - y\|^2 + \lambda\|w\|_1$$

$\lambda$ sets how much weight size costs compared with error.

- **Ridge** (L2) shrinks all weights toward zero. It has a closed form,
  $\hat w = (X^\top X + \lambda I)^{-1} X^\top y$, and the matrix is always
  invertible for $\lambda > 0$: adding $\lambda I$ raises every eigenvalue
  by $\lambda$, so none is zero.
- **Lasso** (L1) sets some weights to exactly zero, so it also selects
  features. It has no closed form.

**Where the penalty comes from.** §1.1 showed that squared error is
maximum likelihood: it picks the weights $w$ that make the data most likely,
$p(\text{data} \mid w)$. That ignores anything you believed about $w$
before seeing the data. Suppose you believe small weights are more likely
than large ones. That belief is a **prior** $p(w)$, and Bayes rule
(Probability §3.3) combines it with the data:

$$p(w \mid \text{data}) \propto p(\text{data} \mid w)\; p(w)$$

Picking the $w$ that maximizes this is the **MAP** (maximum a posteriori)
estimate. Taking $-\log$ turns the product into a sum to minimize:

$$\underbrace{-\log p(\text{data} \mid w)}_{\text{squared error}} \ \underbrace{-\ \log p(w)}_{\text{penalty}}$$

**Gaussian prior gives ridge.** Believe each weight is a bell curve centered
at 0 with variance $\tau^2$: $p(w_j) \propto e^{-w_j^2/(2\tau^2)}$. The
$-\log$ of that is $w_j^2/(2\tau^2)$, so the penalty is
$\|w\|_2^2/(2\tau^2)$. The squared-error term, written in full, is
$\|Xw - y\|^2/(2\sigma^2)$, where $\sigma^2$ is the variance of the noise in
$y$. Multiply both terms by $2\sigma^2$:

$$\|Xw - y\|^2 + \frac{\sigma^2}{\tau^2}\,\|w\|_2^2$$

This is ridge with $\lambda = \sigma^2/\tau^2$. So $\lambda$ is large when:

- the data are noisy (large $\sigma^2$), so you trust them less;
- you strongly believe the weights are small (small $\tau^2$).

**Laplace prior gives lasso.** The Laplace prior
$p(w_j) \propto e^{-|w_j|/b}$ has a sharp peak at 0. Its $-\log$ is
$|w_j|/b$, so the penalty is $\|w\|_1/b$.

### 3.2 Why L1 gives exact zeros

Lasso sets useless weights to exactly 0; ridge only makes them small. The
reason is how hard each penalty pulls a weight toward 0.

**Penalty alone.** Gradient descent moves each weight against the slope of
the penalty:

- Ridge's penalty $\lambda w^2$ has slope $2\lambda w$, which is
  proportional to the weight. So each step removes a fixed **percentage**.
  Removing 10% per step takes a weight of 1 to 0.9, 0.81, 0.73, ...; after 50
  steps it is 0.005. It gets tiny but never reaches 0.
- Lasso's penalty $\lambda |w|$ has slope $\lambda$ (with the sign of $w$),
  the same at any size. So each step removes a fixed **amount**. Removing
  0.1 per step takes a weight of 1 to 0.9, 0.8, ..., 0.1, 0 in ten steps.
  Lasso stops a weight at 0 rather than letting it cross to the other side.

**With the data.** The data's gradient pulls each weight toward the value
that fits best, and the weight settles where the two pulls balance. For a
useless feature like shoe size in §3.1, the data's pull is weak.

- Ridge's pull gets weaker as the weight shrinks and vanishes at 0, so near
  0 even a weak data pull wins. The weight stops at a small nonzero value.
- Lasso's pull stays $\lambda$ all the way to 0. If the data's pull is weaker
  than $\lambda$, nothing stops the weight, and it ends at exactly 0.

**The same idea in a picture.** Take two weights. With no penalty, least
squares would pick $(w_1, w_2) = (2, 0.8)$, the black dot below. Think of
the penalty as a budget: the weights must stay inside a region. The L1
region $|w_1| + |w_2| \le 1$ is a diamond with corners on the axes; the L2
region $w_1^2 + w_2^2 \le 1$ is a disk. The answer is the point of the
region closest to the black dot, where the smallest error circle (green)
around the dot touches the region. The diamond sticks out at its corners,
so the circle usually touches a corner, where one weight is 0. The disk has
no corners.

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

*Left: the circle touches the diamond at
the corner $(1, 0)$, so $w_2 = 0$ exactly. Right: it touches the disk at
$(0.93, 0.37)$, where both weights are smaller but neither is 0.*

### 3.3 Effect on bias and variance

A penalty **adds bias to remove variance** (Probability §3.7). Shrinking the
weights makes them depend less on which training set you happened to draw
(less variance), but pulls true effects toward zero (more bias). At
$\lambda = 0$ you get plain least squares; as $\lambda \to \infty$ every
weight goes to 0. The best $\lambda$ is in between; choose it on a
validation set or with cross-validation.

Three practical points:

- **Don't penalize the bias term.** It only shifts every prediction by a
  constant, so shrinking it adds bias without reducing variance.
- **Standardize features first.** The penalty depends on scale. A feature
  measured in millimeters instead of meters needs a weight 1000 times
  smaller for the same effect, so it is penalized 1000 times less (L1) or
  $10^6$ times less (L2).
- **Elastic net** adds both penalties, $\lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$.
  It fixes lasso's problem with near-duplicate features. Say two features
  are almost the same, such as house size from the listing and house size
  measured by an inspector, and together they need a total weight of 2. The
  predictions are the same for $(2, 0)$ and $(1, 1)$, and so is the L1
  penalty: $|2| + |0| = |1| + |1| = 2$. So lasso has no reason to prefer
  sharing. It puts all the weight on whichever feature fits slightly better,
  and small noise in the data decides which one that is. The L2 penalty
  prefers sharing: $2^2 + 0^2 = 4$ but $1^2 + 1^2 = 2$. Elastic net's L2
  part splits the weight across the group, and its L1 part still sets
  useless features to 0.
