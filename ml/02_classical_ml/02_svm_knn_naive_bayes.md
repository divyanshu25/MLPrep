# Other classifiers: SVMs, kNN, naive Bayes

Notation: $n$ examples, $d$ features, $X \in \mathbb{R}^{n \times d}$ with one
example per row, labels $y \in \mathbb{R}^n$, weights $w \in \mathbb{R}^d$.
The bias is folded into $w$ by adding a feature that is always 1.

---

## 4. Support vector machines

### 4.1 Maximum margin

When two classes can be separated by a line, many lines work. A support
vector machine (SVM) picks the one with the widest gap, the **margin**, to
the nearest points of each class. A wide gap means small changes to the data
are less likely to flip a prediction.

**Turning "widest gap" into a formula.** Label the classes $y = +1$ and
$y = -1$. The line is $w^\top x + b = 0$, and a point's distance from it is
$|w^\top x + b|/\|w\|$. Multiplying $w$ and $b$ by the same number doesn't
move the line, so we can choose the scale where the nearest points have
$|w^\top x + b| = 1$. Then:

- the nearest points are at distance $1/\|w\|$, so a smaller $\|w\|$ means
  a wider margin;
- every point must be on its correct side and at least that far out, which
  is $y_i(w^\top x_i + b) \ge 1$ (for a $+1$ point the score must be at least
  1, for a $-1$ point at most $-1$).

$$\min_{w, b}\ \tfrac12\|w\|^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \ge 1 \text{ for all } i$$

The points exactly on the margin are the **support vectors**. They alone
decide the line: delete any other point and the solution doesn't change.

### 4.2 Soft margin: hinge loss plus L2

Real data usually can't be separated perfectly: some points end up inside
the margin or on the wrong side. So allow these violations and charge for
each one in proportion to how far it is from where it should be. This turns
the SVM into an ordinary loss plus a penalty, like the models in §1 to §3:

$$\min_{w, b}\ \sum_i \underbrace{\max\big(0,\ 1 - y_i f(x_i)\big)}_{\text{hinge loss}} + \frac{1}{2C}\|w\|^2, \qquad f(x) = w^\top x + b$$

The hinge loss depends on $y f(x)$, which is positive when the prediction
is on the correct side:

| $y\,f(x)$ | 2 | 1 | 0.5 | 0 | $-1$ |
|---|---|---|---|---|---|
| hinge loss | 0 | 0 | 0.5 | 1 | 2 |
| logistic loss $\log(1 + e^{-yf})$ | 0.13 | 0.31 | 0.47 | 0.69 | 1.31 |

Points beyond the margin ($yf \ge 1$) have zero hinge loss and zero
gradient, so they have no effect. Logistic loss is positive for every point,
so every point pulls on the weights a little.

**$C$ is the inverse of the regularization strength.** The L2 penalty is
$1/(2C)$. Large $C$ makes violations expensive: the model fits the training
data closely with a narrow margin (low bias, high variance). Small $C$
allows violations and keeps a wide margin (high bias, low variance).

### 4.3 The kernel trick

**The problem.** Suppose one class sits inside a circle and the other
outside it. No straight line separates them. But add the feature
$x_1^2 + x_2^2$, the squared distance from the center: in that feature the
inside points are small and the outside points are large, and a straight
cut separates them. In general, mapping the inputs to new features $\phi(x)$
(squares, products of features, and so on) lets a linear model draw curved
boundaries in the original space.

**The catch.** The number of new features explodes. All products of up to
10 of 100 features is about $4 \times 10^{13}$ features per example, far too
many to compute or store.

**The way out.** Rewriting the SVM problem with Lagrange multipliers
$\alpha_i \ge 0$, one per training point, gives $w = \sum_i \alpha_i y_i x_i$
and the prediction

$$f(x) = \sum_i \alpha_i\, y_i\, \langle x_i, x \rangle + b$$

Only support vectors have $\alpha_i > 0$, and the training problem (the
**dual**) likewise needs the data only as dot products
$\langle x_i, x_j \rangle$. So the features themselves are never needed,
only dot products between them. A **kernel** is a function that computes
the dot product of the new features directly from the original inputs,
$K(x, z) = \langle \phi(x), \phi(z) \rangle$. Example in two dimensions with
$\phi(x) = (x_1^2,\ \sqrt2\, x_1 x_2,\ x_2^2)$:

$$\langle \phi(x), \phi(z) \rangle = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2 = (x_1 z_1 + x_2 z_2)^2 = \langle x, z\rangle^2$$

One ordinary dot product and a square give the dot product in the 3-feature
space, without building it. Replace every $\langle x_i, x \rangle$ in the
SVM with $K(x_i, x)$ and you get the curved boundary at the cost of the
original features.

**The RBF kernel** $K(x, z) = e^{-\gamma\|x - z\|^2}$ is the most used. Read
it as a similarity: 1 when $x = z$, falling toward 0 as they move apart. The
prediction then becomes a vote of the support vectors, each weighted by how
similar it is to the new point. It corresponds to infinitely many features,
which no explicit $\phi$ could handle. A function is a valid kernel when its
matrix of values $K(x_i, x_j)$ on any set of points is symmetric with no
negative eigenvalues (Mercer's theorem).

**The cost:** the kernel matrix is $n \times n$, so memory grows as $n^2$ and
training time as $n^2$ to $n^3$ in the number of examples. That is why
kernel SVMs lost to neural networks on large datasets: a network learns its
features $\phi$ and scales to millions of examples.

---

## 5. kNN and the curse of dimensionality

**k-nearest neighbors** has no training: to predict, find the $k$ closest
training points and take their majority label (or average value). $k$ sets
the bias-variance tradeoff: $k = 1$ fits the training data perfectly with a
jagged boundary (high variance); $k = n$ always predicts the overall
majority (high bias). kNN works only if nearby points have similar labels
and there is enough data for "nearby" to be close.

**The curse of dimensionality.** Spread data uniformly in the unit cube
$[0,1]^d$ and ask how big a box around a point must be to hold its nearest
1% of the data. A box covering a fraction $s$ of every feature's range holds
$s^d$ of the data, so $s^d = 0.01$ and $s = 0.01^{1/d}$. In 10 dimensions,
$0.63^{10} = 0.01$: the box must cover 63% of every feature's range.

| dimensions $d$ | 2 | 10 | 100 |
|---|---|---|---|
| side to hold 1% of data | 0.1 | 0.63 | 0.955 |

In 100 dimensions the "neighborhood" covers 95% of every feature's range, so
the nearest neighbors are not near. Distances also bunch together: the
nearest and farthest points end up at almost the same distance, so ranking
by distance tells you little. The data needed to keep neighborhoods small
grows exponentially with $d$.

What helps: reduce dimensions first ([§10](04_clustering_and_dim_reduction.md#10)), learn the distance function from
labels, and rely on real data lying near a low-dimensional surface. Practical
points: **scale features** (a mass in grams instead of kilograms dominates
every distance), use odd $k$ to avoid ties, and expect $O(nd)$ cost per
prediction without an index such as a KD-tree.

---

## 6. Naive Bayes

Example: is an email with the words "free" and "meeting" spam?

| | $p(\text{class})$ | $p(\text{"free"} \mid \text{class})$ | $p(\text{"meeting"} \mid \text{class})$ | product |
|---|---|---|---|---|
| spam | 0.5 | 0.6 | 0.01 | 0.003 |
| not spam | 0.5 | 0.05 | 0.3 | 0.0075 |

Not spam wins. This is Bayes rule, $p(y \mid x) \propto p(y)\,p(x \mid y)$,
with the **naive assumption** that the words are independent given the
class:

$$p(x \mid y) = \prod_{j=1}^{d} p(x_j \mid y)$$

Why the assumption is needed: without it, you would need the probability of
every combination of words, $2^d - 1$ numbers per class for $d$ yes/no
words, far more than any dataset can estimate. With it, you need $d$
numbers, each a simple count: "what fraction of spam emails contain
'free'?". In practice add logs
instead of multiplying, since a product of hundreds of small probabilities
underflows to 0.

**Laplace smoothing.** A word never seen in spam gets
$p(\text{word} \mid \text{spam}) = 0$, which zeroes the whole product
whatever the other words say. Add a pseudo-count $\alpha$ (often 1):

$$p(x_j \mid y) = \frac{\text{count} + \alpha}{\text{total} + \alpha \cdot |\text{vocabulary}|}$$

**Why it works although words aren't independent.** Classification needs
only the right winner, not accurate probabilities. Correlated words get
counted twice, which pushes probabilities toward 0 and 1 (overconfident) but
often keeps the ranking right. So naive Bayes is a decent classifier with
poorly calibrated probabilities.

Naive Bayes is **generative**: it models how each class produces its
features, $p(x \mid y)$, and gets the class from Bayes rule. Logistic
regression is **discriminative**: it models $p(y \mid x)$ directly and never
describes the features. Naive Bayes reaches its best accuracy with less
data; logistic regression is usually better once data is plentiful, because
it doesn't rely on the independence assumption.
