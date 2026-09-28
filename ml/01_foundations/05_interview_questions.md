# Interview questions & exit checklist

Fifteen standard questions, grouped by topic. Work them as a drill: read the
question, answer out loud or on paper as if the interviewer just asked it,
and only then open the model answer. Any answer you could not produce cold
goes into `practice/queue.tsv`.

Each model answer opens with **the short answer**, the version to say out
loud in the first thirty seconds. The derivation and the follow-ups below it
are what the interviewer's next question will ask for.

## Linear algebra

### Q1. Why does the dot product measure similarity?

<details><summary>Model answer</summary>

**Short answer.** $a \cdot b = \|a\|\,\|b\|\cos\theta$, where $\theta$ is the
angle between the vectors. With the lengths fixed, the dot product is largest
when the vectors point the same way ($\theta = 0$), zero when they are
perpendicular, and negative when they point in opposite directions. Dividing
by both lengths leaves $\cos\theta$, the cosine similarity.

**Where the cosine comes from.** Write the squared length of $a - b$ two ways.

- Expand with the dot product:

$$\|a - b\|^2 = (a - b)\cdot(a - b) = \|a\|^2 + \|b\|^2 - 2\,a \cdot b$$

- Apply the law of cosines to the triangle with sides $a$, $b$, $a - b$:

$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\,\|a\|\,\|b\|\cos\theta$$

Setting the two right sides equal gives $a \cdot b = \|a\|\,\|b\|\cos\theta$.

**Numbers.**

- $a = (3, 4)$, $b = (4, 3)$: $a \cdot b = 12 + 12 = 24$, $\|a\| = \|b\| = 5$,
  so $\cos\theta = 24/25 = 0.96$. Nearly the same direction.
- $a = (3, 4)$, $c = (4, -3)$: $a \cdot c = 12 - 12 = 0$. Perpendicular.

**Projection reading.** $\|a\|\cos\theta$ is the length of the projection of
$a$ onto the direction of $b$, so $a \cdot b$ equals that projected length
times $\|b\|$.

**Follow-ups.**

- *Why normalize?* The raw dot product grows with vector length, so a long
  embedding scores high against every query. Cosine similarity removes
  length. Attention keeps the raw dot product $q \cdot k$ and divides by
  $\sqrt{d_k}$ instead (module 4).
- *Dot product or Euclidean distance for retrieval?* For unit vectors,
  $\|a - b\|^2 = 1 + 1 - 2\,a \cdot b$, so ranking by largest dot product
  and ranking by smallest distance give the same order.
</details>

---

### Q2. What happens if you stack linear layers with no nonlinearity?

<details><summary>Model answer</summary>

**Short answer.** The stack reduces to a single linear layer, so the extra
layers add no functions the model can represent. A nonlinearity between the
layers prevents the reduction.

**Derivation, with biases.**

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\,x + (W_2 b_1 + b_2) = W x + b$$

with $W = W_2 W_1$ and $b = W_2 b_1 + b_2$. Applying the same step
repeatedly reduces any number of layers to one.

**A function no linear model computes: XOR.** A linear classifier outputs 1
when $w_1 x_1 + w_2 x_2 + b > 0$. XOR needs:

- $(0, 1)$ and $(1, 0)$ positive: $w_2 + b > 0$ and $w_1 + b > 0$. Adding
  them: $w_1 + w_2 + 2b > 0$.
- $(0, 0)$ and $(1, 1)$ negative: $b < 0$ and $w_1 + w_2 + b < 0$. Adding
  them: $w_1 + w_2 + 2b < 0$.

The two sums contradict each other, so no $(w_1, w_2, b)$ works. One ReLU
unit fixes it: with $h = \max(0,\ x_1 + x_2 - 1)$, the output
$x_1 + x_2 - 2h$ gives $0, 1, 1, 0$ on $(0,0), (0,1), (1,0), (1,1)$.

**Follow-up: is a linear stack ever useful?** It adds no functions, but it
can restrict them. If $W_1$ is $r \times d$ with small $r$, the product
$W_2 W_1$ has rank at most $r$. The LoRA update $\Delta W = BA$ (module 7)
uses exactly this: two linear factors, chosen to reduce the parameter count,
not to add expressiveness.
</details>

---

### Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?

<details><summary>Model answer</summary>

**Short answer.** An eigenvector of a square matrix $A$ is a nonzero vector
$v$ that $A$ maps to a multiple of itself, $Av = \lambda v$, and the
eigenvalue $\lambda$ is that multiple. The eigenvectors of a data covariance
matrix are the PCA directions, and each eigenvalue is the variance of the
data along its eigenvector.

**Numbers.** $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$.

- $v = (1, 1)$: $Av = (3, 3) = 3v$. Eigenvalue 3.
- $v = (1, -1)$: $Av = (1, -1) = 1 \cdot v$. Eigenvalue 1.
- $v = (1, 0)$: $Av = (2, 1)$, not a multiple of $(1, 0)$. Not an eigenvector.

To find them without guessing, solve
$\det(A - \lambda I) = (2 - \lambda)^2 - 1 = 0$, which gives
$\lambda = 1, 3$.

**The PCA link, derived.** For centered data with covariance matrix
$\Sigma$, the variance of the projection onto a unit vector $u$ is
$u^\top \Sigma u$. Maximize it subject to $u^\top u = 1$ with a Lagrange
multiplier $\lambda$:

$$\nabla_u\big[\,u^\top \Sigma u - \lambda(u^\top u - 1)\,\big] = 2\Sigma u - 2\lambda u = 0 \quad\Longrightarrow\quad \Sigma u = \lambda u$$

So the best direction is an eigenvector, and the variance it achieves is
$u^\top \Sigma u = \lambda\, u^\top u = \lambda$. The largest eigenvalue's
eigenvector is the first principal component.

**Follow-up: eigenvalues and learning rate.** Gradient descent on a
quadratic loss with Hessian $H$ multiplies the error along each eigenvector
by $1 - \eta\lambda$ per step. It converges only if $|1 - \eta\lambda| < 1$
for every eigenvalue, that is $\eta < 2/\lambda_{\max}$.
</details>

---

### Q4. What's the difference between eigendecomposition and SVD?

<details><summary>Model answer</summary>

**Short answer.** Eigendecomposition $A = Q \Lambda Q^{-1}$ needs a square
matrix and does not exist for every square matrix. SVD $A = U \Sigma V^\top$
exists for every matrix, square or rectangular, with $U$ and $V$ orthogonal
and $\Sigma$ diagonal with nonnegative entries. They are linked by
$A^\top A = V \Sigma^2 V^\top$: the right singular vectors of $A$ are
eigenvectors of $A^\top A$, and the singular values are the square roots of
its eigenvalues.

**Where eigendecomposition fails.**

- Rectangular $A$: $Av = \lambda v$ requires $v$ and $Av$ to live in the
  same space.
- Some square $A$: for $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$,
  $\det(A - \lambda I) = \lambda^2$, so the only eigenvalue is 0, and
  $Av = 0$ has only the multiples of $(1, 0)$ as solutions. One eigenvector
  direction for a $2 \times 2$ matrix is not enough to build $Q$.

**Deriving the link.** Use $U^\top U = I$:

$$A^\top A = (U \Sigma V^\top)^\top (U \Sigma V^\top) = V \Sigma^\top U^\top U \Sigma V^\top = V (\Sigma^\top \Sigma) V^\top$$

$\Sigma^\top \Sigma$ is diagonal with entries $\sigma_i^2$, so this is an
eigendecomposition of $A^\top A$. For a symmetric positive semidefinite
matrix such as a covariance matrix, the two decompositions coincide.

**Reading $U \Sigma V^\top x$ right to left.** $V^\top$ rotates $x$ into the
input singular basis, $\Sigma$ scales coordinate $i$ by $\sigma_i$ (and
changes the dimension), $U$ rotates into the output space.

**Why it matters: the best low-rank approximation.** Keeping the $k$ largest
singular values,

$$A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top,$$

gives the closest rank-$k$ matrix to $A$ (Eckart-Young), with Frobenius
error $\sqrt{\sum_{i > k} \sigma_i^2}$. PCA is the SVD of the centered data
matrix, recommender matrix factorization is a low-rank fit, and LoRA assumes
the fine-tuning update is well approximated at low rank.
</details>

---

### Q5. Why does L1 regularization give sparsity but L2 doesn't?

<details><summary>Model answer</summary>

**Short answer.** The L1 penalty $\lambda|w|$ has slope of magnitude
$\lambda$ at every $w \neq 0$, including values arbitrarily close to 0. So
at $w = 0$ the loss's slope must exceed $\lambda$ in magnitude for the
optimum to move away from zero; when it does not, the optimum is exactly
$w = 0$. The L2 penalty $\lambda w^2$ has slope $2\lambda w$, which is 0 at
$w = 0$, so any nonzero slope of the loss moves the optimum off zero. L2
makes weights small, L1 makes some of them exactly zero.

**Derivation with one weight.** Take the simplest loss, $(w - a)^2$, whose
minimum is at $a$: the value the data alone would choose.

L2:

$$f(w) = (w - a)^2 + \lambda w^2, \qquad f'(w) = 2(w - a) + 2\lambda w = 0 \;\Longrightarrow\; w^* = \frac{a}{1 + \lambda}$$

$w^*$ is zero only when $a$ is. Every weight is divided by the same factor
$1 + \lambda$, and none reaches zero.

L1: $|w|$ has no derivative at 0, so solve on each side separately.

- $w > 0$: $f'(w) = 2(w - a) + \lambda = 0$ gives $w = a - \lambda/2$,
  which is on this side only if $a > \lambda/2$.
- $w < 0$: $f'(w) = 2(w - a) - \lambda = 0$ gives $w = a + \lambda/2$,
  which is on this side only if $a < -\lambda/2$.
- $|a| \le \lambda/2$: neither side has a solution. At $w = 0$ the slope
  just to the right is $-2a + \lambda \ge 0$ and just to the left is
  $-2a - \lambda \le 0$, so $f$ increases in both directions and
  $w^* = 0$.

The three cases combine into the soft-threshold formula:

$$w^* = \operatorname{sign}(a)\,\max\!\big(0,\ |a| - \tfrac{\lambda}{2}\big)$$

**Numbers, $\lambda = 1$.**

| Data optimum $a$ | L2: $a/(1+\lambda)$ | L1: soft threshold |
|---|---|---|
| 0.3 | 0.15 | 0 |
| 2 | 1 | 1.5 |

L2 divides every weight by 2. L1 subtracts 0.5 from every magnitude and
sets to zero any weight whose magnitude is below 0.5. (For the large
weight, L1 shrinks less than L2.)

**The picture interviewers expect.** Minimizing loss $+\ \lambda\|w\|$ is
equivalent to minimizing the loss subject to $\|w\| \le t$ for some budget
$t$. Draw both budgets in two dimensions and grow a loss contour from the
unconstrained optimum until it first touches the budget region:

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

*Loss $\|w - (2, 0.8)\|^2$, whose contours are circles around $(2, 0.8)$
(green; dashed is a smaller level). The constrained solution is where the
smallest contour meets the budget region. Left: first contact at the corner
$(1, 0)$, so $w_2 = 0$; this is the soft threshold applied to each
coordinate with threshold 1: $2 - 1 = 1$, and $0.8 < 1$ goes to 0.
Right: first contact at $(2, 0.8)/\|(2, 0.8)\| = (0.93, 0.37)$, both
weights shrunk, neither zero.*

In $d$ dimensions the L1 ball has $2d$ corners, one on each half-axis, and
its edges and faces lie in coordinate subspaces where some weights are zero.
Loss contours usually make first contact there. The L2 ball is smooth and
has no such points.

**Follow-ups.**

- *Bayesian view:* L2 is MAP estimation with a Gaussian prior on the
  weights, L1 with a Laplace prior, whose density has a sharp peak at 0
  (module 2, §3.1).
- *Correlated features:* lasso tends to keep one feature of a correlated
  group and zero the rest, and which one it keeps can change between
  resamples. Elastic net (L1 + L2) keeps the group together.
- *Does SGD with an L1 term give exact zeros?* No. Subgradient steps
  overshoot 0 and oscillate around it. Exact zeros need a proximal step:
  after each gradient step on the loss, apply the soft threshold above (as
  in ISTA or coordinate descent).
</details>

## Calculus

### Q6. Explain backprop.

<details><summary>Model answer</summary>

**Short answer.** Backprop computes $\partial L / \partial \theta$ for every
parameter $\theta$ with two passes over the computation graph. The forward
pass evaluates every node and stores its value. The backward pass starts
from $\partial L / \partial L = 1$ and visits the nodes in reverse order:
each node multiplies the gradient arriving at its output by its local
derivative, and adds the product into the gradient of each of its inputs.
It is reverse-mode automatic differentiation, and because the loss is a
single number, one backward pass, costing about two forward passes, gives
the gradient for all parameters at once.

**The rule at one node.** If a variable $v$ feeds into later nodes
$u_1, \dots, u_k$:

$$\frac{\partial L}{\partial v} = \sum_{j=1}^{k} \frac{\partial L}{\partial u_j}\,\frac{\partial u_j}{\partial v}$$

- $\partial L / \partial u_j$ is the **upstream gradient**. It is already
  known because $u_j$ comes after $v$ in the graph, and the backward pass
  visited it first.
- $\partial u_j / \partial v$ is the **local derivative**, the one thing
  each operation must know how to compute.
- The sum has one term per path from $v$ to $L$ (the multivariable chain
  rule). In code this is `v.grad += ...`, not `v.grad = ...`.

**Run it on numbers.** $L = (wx + b)^2$ at $w = 2$, $x = 3$, $b = 1$.

Forward, storing every value:

$$z = wx + b = 2 \cdot 3 + 1 = 7, \qquad L = z^2 = 49$$

Backward, one chain-rule step per line:

$$\frac{\partial L}{\partial L} = 1$$

$$\frac{\partial L}{\partial z} = 1 \cdot 2z = 14$$

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x = 14 \cdot 3 = 42, \qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot 1 = 14$$

Check by differentiating directly: $\partial L / \partial w = 2(wx + b)\,x =
2 \cdot 7 \cdot 3 = 42$. The backward steps used $z = 7$ and $x = 3$, both
values from the forward pass, which is why the forward pass stores them.
(Calculus §2.2 draws this graph.)

**Why backward and not forward.** The chain rule can be evaluated in either
direction.

- **Forward mode** carries $\partial v / \partial \theta_i$ for one chosen
  input $\theta_i$ through the graph. One pass gives the derivative of every
  node with respect to that one parameter; $n$ parameters need $n$ passes.
- **Reverse mode** carries $\partial L / \partial v$ for one output $L$. One
  pass gives the derivative of $L$ with respect to every node, all $n$
  parameters included.

A network has one loss and $n \approx 10^9$ parameters: one reverse pass
versus $10^9$ forward-mode passes (or $10^9$ extra forward passes for finite
differences). Forward mode is the cheaper choice in the opposite shape, few
inputs and many outputs.

**What the backward pass costs in compute.** For a linear layer $Y = XW$,
with upstream gradient $G = \partial L / \partial Y$: since
$Y_{ij} = \sum_k X_{ik} W_{kj}$,

$$\frac{\partial L}{\partial W_{kj}} = \sum_i G_{ij}\, X_{ik} = (X^\top G)_{kj}, \qquad \frac{\partial L}{\partial X_{ik}} = \sum_j G_{ij}\, W_{kj} = (G W^\top)_{ik}$$

The forward pass is one matrix multiply; the backward pass is two of the
same size. So backward costs about $2\times$ forward, and a training step
about $3\times$ forward. At $2N$ FLOPs per token forward for $N$ parameters,
that is $6N$ per token: the $C = 6ND$ of the scaling laws (module 7).

**What it costs in memory.** Every stored activation stays in memory until
the backward pass uses it, so activation memory grows with depth, batch size
and sequence length, and for large batches it can exceed the memory of the
weights. Activation checkpointing stores only some layers' outputs and
recomputes the rest during the backward pass, trading about one extra
forward pass of compute for a large memory saving.

**Follow-ups.**

- *What must a new layer implement?* A forward function and a
  vector-Jacobian product: given $\partial L / \partial(\text{output})$,
  return $\partial L / \partial(\text{each input})$ (Q7).
- *Why do gradients vanish?* The backward pass multiplies one local
  Jacobian per layer (Q8).
- *Does backprop find the minimum?* No. It only computes the gradient; the
  optimizer (SGD, Adam) chooses the step.
</details>

---

### Q7. What is a Jacobian and what's its role in backprop?

<details><summary>Model answer</summary>

**Short answer.** For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J$
is the $m \times n$ matrix with entries $J_{ij} = \partial f_i / \partial
x_j$. It is the linear approximation of $f$ near $x$:
$f(x + \delta) \approx f(x) + J\delta$. Backprop never builds it: each layer
computes the vector-Jacobian product $J^\top g$ directly, where
$g = \partial L / \partial f$.

**Why a vector-Jacobian product.** For a scalar loss $L(f(x))$, the chain
rule sums over the $m$ outputs:

$$\frac{\partial L}{\partial x_j} = \sum_{i=1}^{m} \frac{\partial L}{\partial f_i}\,\frac{\partial f_i}{\partial x_j} \quad\Longrightarrow\quad \nabla_x L = J^\top g$$

**Two Jacobians worked out.**

- Linear map $f(x) = Wx$: $f_i = \sum_j W_{ij} x_j$, so $J_{ij} = W_{ij}$,
  $J = W$, and the backward step is $W^\top g$.
- Elementwise ReLU $f_i = \max(0, x_i)$: $f_i$ depends only on $x_i$, so $J$
  is diagonal with $J_{ii} = 1$ if $x_i > 0$ and 0 otherwise. The backward
  step is $g \odot \mathbb{1}[x > 0]$, an elementwise product with no matrix.

**Why it is never materialized.** A ReLU on 4,096 activations has a
$4096 \times 4096$ Jacobian: 16,777,216 entries, of which all but 4,096 are
zero. The vector-Jacobian product needs 4,096 multiplications.
</details>

---

### Q8. Why do gradients vanish or explode in deep/recurrent nets?

<details><summary>Model answer</summary>

**Short answer.** The backward pass multiplies the gradient by one Jacobian
per layer, so the gradient at layer 1 of a network with $D$ layers is a
product of $D$ matrices applied to the output gradient. If the factors shrink
vectors, the product shrinks exponentially with depth; if they stretch
vectors, it grows exponentially.

**The product.** With $h_l = f_l(h_{l-1})$ and
$J_l = \partial h_l / \partial h_{l-1}$:

$$\frac{\partial L}{\partial h_0} = J_1^\top J_2^\top \cdots J_D^\top\, \frac{\partial L}{\partial h_D}$$

**Numbers, scalar case with every factor equal.**

- Factor 0.9, 50 layers: $0.9^{50} \approx 0.005$.
- Factor 1.1, 50 layers: $1.1^{50} \approx 117$.
- Sigmoid: $\sigma'(z) \le 0.25$, so ten sigmoid layers with unit weights
  multiply the gradient by at most $0.25^{10} \approx 9.5 \times 10^{-7}$.

In a recurrent net the same weight matrix $W$ appears in every time step's
factor, so over $T$ steps the gradient scales roughly like the $T$-th power
of $W$'s largest singular value (module 4, BPTT).

**Fixes, each tied to the product.**

- *Residual connections:* $h_l = h_{l-1} + F(h_{l-1})$ gives
  $J_l = I + \partial F / \partial h_{l-1}$, so the identity term carries
  the gradient through every layer unchanged.
- *Initialization (Xavier, Kaiming):* choose the weight variance so each
  layer preserves the variance of its input, making the factors average
  magnitude 1 (module 3).
- *Normalization layers:* keep activations at unit scale so local
  derivatives do not drift with depth.
- *ReLU instead of sigmoid:* derivative 1 on the active side instead of at
  most 0.25.
- *Gradient clipping:* rescale $g$ when $\|g\|$ exceeds a threshold. Treats
  explosion only.
- *Gating (LSTM):* the cell update $c_t = f_t \odot c_{t-1} + \dots$ has
  Jacobian $\operatorname{diag}(f_t)$ along the cell path; a forget gate near
  1 carries the gradient across many steps.
</details>

---

### Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?

<details><summary>Model answer</summary>

**Short answer.** $\nabla_w \|Xw - y\|^2 = 2X^\top(Xw - y)$. Setting it to
zero gives the normal equations $X^\top X w = X^\top y$, and when
$X^\top X$ is invertible, $w = (X^\top X)^{-1} X^\top y$: the closed-form
least-squares solution.

**Derivation.** Expand the square:

$$\|Xw - y\|^2 = (Xw - y)^\top (Xw - y) = w^\top X^\top X w - 2\,y^\top X w + y^\top y$$

(the cross terms $w^\top X^\top y$ and $y^\top X w$ are the same scalar).
Apply $\nabla_w (w^\top A w) = 2Aw$ for symmetric $A$ and
$\nabla_w (c^\top w) = c$:

$$\nabla_w = 2X^\top X w - 2X^\top y = 2X^\top (Xw - y)$$

**What the zero-gradient condition says.** $X^\top (y - Xw) = 0$: the
residual $y - Xw$ is perpendicular to every column of $X$. The fitted values
$Xw$ are the orthogonal projection of $y$ onto the column space of $X$.

**Numbers.** Fit $y = wx$ through $(1, 1), (2, 2), (3, 2)$:
$X^\top X = 1 + 4 + 9 = 14$, $X^\top y = 1 + 4 + 6 = 11$, so
$w = 11/14 \approx 0.786$. Residuals $(0.214, 0.429, -0.357)$, and
$X^\top r = 0.214 + 0.857 - 1.071 = 0$.

**Follow-ups.**

- *When is $X^\top X$ singular?* When columns of $X$ are linearly dependent
  (duplicated or collinear features). Ridge solves
  $(X^\top X + \lambda I)\,w = X^\top y$, invertible for any $\lambda > 0$.
- *Do you compute the inverse?* No. Solve the system with a Cholesky or QR
  factorization; at large scale, use gradient descent.
</details>

## Probability & statistics

### Q10. Derive Bayes rule.

<details><summary>Model answer</summary>

**Short answer.** Write the probability that both $A$ and $B$ happen in two
ways, then divide by $P(B)$:

$$P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A) \quad\Longrightarrow\quad P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

**The denominator.** $P(B)$ is usually not given. Split $B$ by whether $A$
happened (law of total probability):

$$P(B) = P(B \mid A)\,P(A) + P(B \mid \lnot A)\,P(\lnot A)$$

**Names.** $P(A)$ is the prior, $P(B \mid A)$ the likelihood,
$P(A \mid B)$ the posterior. Q12 applies it with numbers.

**Follow-up: where it appears in ML.** MAP estimation maximizes
$p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta)\,p(\theta)$;
after taking $-\log$, the prior term becomes the regularizer (Q5). Naive
Bayes classifiers apply the rule directly with the assumption that features
are independent given the class.
</details>

---

### Q11. Why is minimizing cross-entropy the same as maximum likelihood?

<details><summary>Model answer</summary>

**Short answer.** Maximum likelihood chooses $\theta$ to maximize the
probability the model assigns to the observed labels,
$\prod_i p_\theta(y_i \mid x_i)$. Taking $-\log$ turns the product into a
sum without moving the optimum, and $-\log p_\theta(y_i \mid x_i)$ for a
classifier is exactly the cross-entropy loss on example $i$. The two
objectives are the same function.

**Binary case, line by line.** With $\hat y_i = p_\theta(1 \mid x_i)$, the
probability of the observed label is
$p_\theta(y_i \mid x_i) = \hat y_i^{\,y_i} (1 - \hat y_i)^{1 - y_i}$ (it
equals $\hat y_i$ when $y_i = 1$ and $1 - \hat y_i$ when $y_i = 0$). So

$$-\log \prod_i p_\theta(y_i \mid x_i) = -\sum_i \big[\,y_i \log \hat y_i + (1 - y_i)\log(1 - \hat y_i)\,\big]$$

which is the binary cross-entropy summed over the data. With $K$ classes and
a one-hot label, $-\sum_k y_k \log p_k = -\log p_{\text{true class}}$.

**The same step for regression.**

- Gaussian noise, $y = f_\theta(x) + \varepsilon$ with
  $\varepsilon \sim \mathcal{N}(0, \sigma^2)$:

$$-\log p_\theta(y \mid x) = \frac{\big(y - f_\theta(x)\big)^2}{2\sigma^2} + \tfrac12 \log(2\pi\sigma^2)$$

  The second term does not depend on $\theta$, so maximum likelihood is
  least squares.
- Laplace noise with density $\frac{1}{2b} e^{-|\varepsilon|/b}$:
  $-\log p = |y - f_\theta(x)|/b + \log 2b$, so maximum likelihood minimizes
  absolute error (the L1 *loss*, not the L1 *penalty* of Q5).

**The information-theory reading (Probability §3.7).** Averaged over the
data distribution $q$, the loss is the cross-entropy
$H(q, p_\theta) = H(q) + \mathrm{KL}(q \,\|\, p_\theta)$. $H(q)$ does not
depend on $\theta$, so minimizing cross-entropy minimizes the KL divergence
from the data distribution to the model.
</details>

---

### Q12. Positive test, rare disease

A test detects the disease in 99% of sick people and gives a false positive
for 5% of healthy people. 1% of the population has the disease. You test
positive. Roughly what's the chance you're sick?

<details><summary>Model answer</summary>

**Short answer.** About 17%, not 99%. The healthy group is 99 times larger
than the sick group, so its 5% false-positive rate produces five times as
many positives as the sick group does.

**Count 10,000 people.**

- Sick: 100 people, of whom 99 test positive.
- Healthy: 9,900 people, of whom $0.05 \times 9{,}900 = 495$ test positive.
- Positive tests: $99 + 495 = 594$, so
  $P(\text{sick} \mid +) = 99/594 \approx 0.167$.

**The same with Bayes rule (Q10).**

$$P(S \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = \frac{0.0099}{0.0594} \approx 0.167$$

**Variations to have ready.**

- If "99% accurate" means 99% in both groups: false positives are
  $0.01 \times 9{,}900 = 99$, so $P = 99/198 = 50\%$.
- A second, independent positive test: the prior is now 0.167, and

$$P(S \mid +, +) = \frac{0.99 \times 0.167}{0.99 \times 0.167 + 0.05 \times 0.833} \approx 0.80$$

- The ML version: a fraud classifier with a 1% base rate has the same
  arithmetic in its precision, which is why accuracy misleads on imbalanced
  classes and precision and recall are reported instead.
</details>

---

### Q13. Why divide by $n-1$ in the sample variance?

<details><summary>Model answer</summary>

**Short answer.** The deviations are measured from the sample mean
$\bar X$, not the true mean $\mu$, and $\bar X$ is the point that minimizes
the sum of squared deviations for that sample. So
$\sum_i (X_i - \bar X)^2$ is never larger than $\sum_i (X_i - \mu)^2$, and
its expected value is $(n - 1)\sigma^2$ rather than $n\sigma^2$. Dividing by
$n - 1$ makes the estimator unbiased (Bessel's correction).

**Derivation.** Write $X_i - \mu = (X_i - \bar X) + (\bar X - \mu)$, square,
and sum. The cross term vanishes because $\sum_i (X_i - \bar X) = 0$:

$$\sum_{i=1}^n (X_i - \mu)^2 = \sum_{i=1}^n (X_i - \bar X)^2 + n(\bar X - \mu)^2$$

Rearrange and take expectations, using $\mathbb{E}[(X_i - \mu)^2] = \sigma^2$
and $\mathbb{E}[(\bar X - \mu)^2] = \mathrm{Var}(\bar X) = \sigma^2/n$ (Q14):

$$\mathbb{E}\Big[\sum_{i=1}^n (X_i - \bar X)^2\Big] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n - 1)\,\sigma^2$$

**Numbers.** With $n = 2$, the $1/n$ estimator has expected value
$\sigma^2/2$, half the true variance. With $n = 100$ it is $0.99\,\sigma^2$,
and the correction barely matters.

**Degrees of freedom.** The $n$ deviations $X_i - \bar X$ sum to zero, so
only $n - 1$ of them can vary freely. (Full treatment: Probability §3.5.)

**Follow-up: is the sample standard deviation unbiased too?** No. Even with
$\mathbb{E}[s^2] = \sigma^2$, the square root is concave, so
$\mathbb{E}[s] \le \sqrt{\mathbb{E}[s^2]} = \sigma$ (Jensen's inequality).
</details>

---

### Q14. Why does averaging reduce noise, and at what rate?

<details><summary>Model answer</summary>

**Short answer.** For $n$ independent draws with variance $\sigma^2$,
$\mathrm{Var}(\bar X) = \sigma^2/n$, so the standard deviation of the mean
(the standard error) is $\sigma/\sqrt{n}$. Halving the error takes four
times the data. The central limit theorem adds that $\bar X$ is
approximately normal for large $n$, whatever the distribution of each draw.

**Derivation.**

$$\mathrm{Var}(\bar X) = \mathrm{Var}\Big(\frac1n \sum_{i=1}^n X_i\Big) = \frac{1}{n^2} \sum_{i=1}^n \mathrm{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$$

- The $1/n$ comes out squared because $\mathrm{Var}(cX) = c^2\,\mathrm{Var}(X)$.
- The variance of the sum is the sum of the variances only because
  independent draws have zero covariance.

**Numbers.** $\sigma = 10$: $n = 100$ gives standard error 1; $n = 400$
gives 0.5.

**Follow-up: correlated draws.** If every pair of draws has correlation
$\rho$, the variance of the sum gains $n(n - 1)$ covariance terms
$\rho\sigma^2$:

$$\mathrm{Var}(\bar X) = \frac{1}{n^2}\Big[\,n\sigma^2 + n(n - 1)\,\rho\sigma^2\,\Big] = \frac{\sigma^2}{n} + \frac{n - 1}{n}\,\rho\,\sigma^2 \;\longrightarrow\; \rho\,\sigma^2$$

The variance stops falling at $\rho\sigma^2$ no matter how many draws are
averaged. This is why an ensemble of similar models gains less than $1/n$,
and why random forests sample features to decorrelate their trees.
</details>

---

### Q15. Is a 0.5% accuracy gain real?

Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?

<details><summary>Model answer</summary>

**Short answer.** Probably not. The standard error of one accuracy measured
on 1,000 examples is about 0.9 percentage points, larger than the 0.5-point
gap. Because both models are scored on the same examples, the right test is
a paired one (McNemar's), which counts only the examples where the models
disagree, and it can call the gap real only if they disagree on six or fewer.

**Standard error of one accuracy.** Each example is a Bernoulli draw (correct
or not) with $p \approx 0.91$ and variance $p(1 - p)$. The accuracy is the
mean of 1,000 draws, so by Q14:

$$\mathrm{SE} = \sqrt{\frac{p(1 - p)}{n}} = \sqrt{\frac{0.91 \times 0.09}{1000}} = \sqrt{0.0000819} \approx 0.0090$$

That is 0.9 percentage points.

**Treating the two accuracies as independent.** The standard error of the
difference is $\sqrt{2} \times 0.9 \approx 1.28$ points, so
$z = 0.5 / 1.28 \approx 0.39$ and the two-sided $p \approx 0.70$. Nowhere
near significant.

**The paired test.** A 0.5-point gain on 1,000 examples is 5 more examples
correct. Let $b$ be the number of examples only the new model gets right and
$c$ the number only the old model gets right, so $b - c = 5$. Under "no
difference", each disagreement is equally likely to go either way
(Probability §3.8):

$$z = \frac{b - c}{\sqrt{b + c}}$$

- $b = 15$, $c = 10$: $z = 5/\sqrt{25} = 1.0$, $p \approx 0.32$. Not
  significant.
- $b = 5$, $c = 0$: $z = 5/\sqrt{5} \approx 2.24$, $p \approx 0.025$.
  Significant.

$z \ge 1.96$ requires $b + c \le 6$: the same 0.5-point gap is real only if
the new model fixes a handful of examples and breaks almost none.

**Other checks to name.**

- *Seeds:* retrain both models with several random seeds; seed-to-seed
  spread is often as large as 0.5 points.
- *More test data:* with 10,000 examples the standard error of one accuracy
  falls to about 0.29 points.
- *Many comparisons:* if this was the best of 20 variants tried, the chance
  that at least one clears $p < 0.05$ by noise alone is
  $1 - 0.95^{20} \approx 0.64$.
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
