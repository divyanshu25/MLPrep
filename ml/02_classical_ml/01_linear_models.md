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
it to zero gives

$$X^\top X\, w = X^\top y \qquad\Longrightarrow\qquad \hat w = (X^\top X)^{-1} X^\top y$$

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

**Why not always use the normal equation?** It costs about $nd^2 + d^3$
operations. With $d = 10^6$ features, $d^3 = 10^{18}$, which is infeasible.
A gradient step costs $nd$, and gradient descent also works for losses with
no closed-form solution, such as logistic loss.

**Why feature scale matters.** Suppose one feature is house size in square
meters (around 100) and another is number of bedrooms (around 3). A small
change in the size weight changes the predictions a lot; the same change in
the bedrooms weight barely changes them. So the loss bowl is steep along one
weight and almost flat along the other: a long, narrow valley.

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

$\sigma(0) = 0.5$, $\sigma(2) = 0.88$, $\sigma(-2) = 0.12$, $\sigma(4) = 0.98$.

**Where the sigmoid comes from.** A probability lies between 0 and 1, but
$w^\top x$ can be any number, so we need a translation between the two. The
**odds** are $p/(1-p)$: a probability of 0.8 is odds of 4 ("4 to 1"). Odds
run from 0 to infinity, and their log, the **log-odds**, runs over all
numbers, just like $z$. So the model sets the log-odds equal to the score:

$$\log\frac{p}{1-p} = z \qquad\Longrightarrow\qquad p = \frac{1}{1 + e^{-z}}$$

So each feature adds its weight to the log-odds, and each extra 1 in $z$
multiplies the odds by $e \approx 2.7$.

The derivative is $\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) = p(1-p)$.

### 2.2 The loss and its gradient

The loss is **binary cross-entropy**, the negative log-likelihood of the
label (Probability §3.4.2):

$$L = -\big[y \log p + (1 - y)\log(1 - p)\big]$$

Its gradient has the same simple form as linear regression's:

$$\nabla_w L = (p - y)\,x$$

By the chain rule, $\frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}$,
$\frac{\partial p}{\partial z} = p(1-p)$ and $\frac{\partial z}{\partial w} = x$.
The $p(1-p)$ cancels.

Example with $y = 1$: if $p = 0.9$ the gradient is $-0.1x$, a small
correction; if $p = 0.1$ it is $-0.9x$, a large one. The correction is
proportional to how wrong the probability is.

The same form, (prediction − target) × input, appears for linear
regression, $(\hat y - y)x$, and for softmax regression with several classes,
$(p_k - y_k)x$. It isn't a coincidence: in each case the loss is the negative
log-likelihood that matches the output (squared error for a plain number,
cross-entropy for a sigmoid or softmax probability), and that matching
always cancels the extra factors in the chain rule.

The loss is **convex** in $w$: its curvature, $\sum_i p_i(1-p_i)\,x_i x_i^\top$,
is never negative in any direction. So gradient descent finds the global
minimum.

### 2.3 Why not squared error on the labels?

- **Predictions leave $[0, 1]$** if you fit plain linear regression to 0/1
  labels.
- **With a sigmoid plus squared error, the gradient vanishes when the model
  is most wrong.** The gradient on $z$ becomes $(p - y)\,p(1-p)$. For $y = 1$,
  $p = 0.1$: $-0.9 \times 0.09 = -0.081$, 11 times smaller than
  cross-entropy's $-0.9$. The loss is also no longer convex.

**Interview trap:** if the classes can be separated perfectly by a line, the
weights grow without limit, because larger weights always push the
probabilities closer to 0 and 1 and lower the loss. Regularization (§3) or
early stopping fixes this, which is why scikit-learn's `LogisticRegression`
regularizes by default.

---

## 3. Regularization

### 3.1 Ridge and lasso

**Why weights need limiting.** Suppose two features are nearly identical,
say two sensors measuring the same temperature. Weights $(0.5, 0.5)$ and
$(1000, -999)$ make almost the same predictions on the training data, but
the second pair amplifies any tiny difference between the sensors, which is
noise. With many features and little data, plain least squares often picks
large weights like these that chase noise and fail on new data.
Regularization adds a penalty on the size of the weights, so it prefers
$(0.5, 0.5)$:

$$\text{ridge: } \|Xw - y\|^2 + \lambda\|w\|_2^2 \qquad\qquad \text{lasso: } \|Xw - y\|^2 + \lambda\|w\|_1$$

- **Ridge** (L2) shrinks all weights toward zero. It has a closed form,
  $\hat w = (X^\top X + \lambda I)^{-1} X^\top y$, and the matrix is always
  invertible for $\lambda > 0$: adding $\lambda I$ raises every eigenvalue
  by $\lambda$, so none is zero.
- **Lasso** (L1) sets some weights to exactly zero, so it also selects
  features. It has no closed form.

**The probability view.** A penalty is the same as a **prior**: a belief,
before seeing data, that weights are probably small. Instead of the weights
that make the data most likely, pick the weights that are most likely given
both the data and the prior (the **MAP** estimate). Taking $-\log$ turns
that into

$$\text{minimize} \quad \underbrace{-\log(\text{likelihood})}_{\text{squared error}} \ \underbrace{-\ \log(\text{prior})}_{\text{penalty}}$$

- A Gaussian prior $w_j \sim \mathcal N(0, \tau^2)$ has
  $-\log(\text{prior}) = \|w\|_2^2/(2\tau^2)$ plus a constant: ridge, with
  $\lambda = \sigma^2/\tau^2$, where $\sigma^2$ is the noise variance.
- A Laplace prior, which is sharply peaked at 0, gives lasso.

So $\lambda$ grows when the data are noisier (larger $\sigma^2$: trust the
data less) or when you believe more strongly that weights are small
(smaller $\tau^2$).

### 3.2 Why L1 gives exact zeros

Take one weight with loss $(w - a)^2$, where $a$ is the unpenalized best
value, and add each penalty:

$$\text{lasso: } w^* = \operatorname{sign}(a)\,\max\big(0,\ |a| - \tfrac{\lambda}{2}\big) \qquad\qquad \text{ridge: } w^* = \frac{a}{1 + \lambda}$$

| | $a = 3,\ \lambda = 2$ | $a = 0.5,\ \lambda = 2$ |
|---|---|---|
| lasso | 2 | **0 exactly** |
| ridge | 1 | 0.17, never 0 |

Lasso subtracts a fixed amount and stops at zero; ridge divides, so it never
reaches zero.

**Why, in terms of slopes.** Start at $w = 0$ and ask whether moving a
little toward $a$ helps:

- The data loss $(w - a)^2$ has slope $-2a$ at 0, so moving toward $a$
  lowers it by $2|a|$ per unit.
- The lasso penalty $\lambda|w|$ has slope $\lambda$ as soon as $w$ moves
  off zero, so moving costs $\lambda$ per unit. If $2|a| < \lambda$, the
  move costs more than it gains, and the weight stays at exactly 0. With
  $a = 0.5$, $\lambda = 2$: gain 1, cost 2.
- The ridge penalty $\lambda w^2$ has slope 0 at $w = 0$, so the first small
  move is free. Any $a \ne 0$ pulls the weight off zero.

The same picture in two dimensions (Linear algebra §1.7): the L1 budget is a
diamond with corners on the axes, and the loss contours usually first touch
it at a corner, where one weight is zero.

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

*Loss $\|w - (2, 0.8)\|^2$ with budget 1. Left: the contour touches the
diamond at the corner $(1, 0)$, so $w_2 = 0$ exactly. Right: it touches the
disk at $(0.93, 0.37)$, both weights shrunk, neither zero.*

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
- **Elastic net** (L1 + L2) handles groups of correlated features. Lasso
  alone keeps one feature from the group and zeros the rest, and which one
  it keeps can change with small changes in the data.
