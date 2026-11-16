# Trees and ensembles: decision trees, bagging, random forests, boosting

Companion notebook: `trees_and_ensembles_scratch.ipynb`.

---

## 7. Decision trees

### 7.1 Measuring how mixed a node is

A decision tree is a flowchart of yes/no questions about features. To
classify an email: "contains 'free'?" If yes, "has more than 3 links?", and
so on, until you reach a final group, a **leaf**, which predicts the most
common label of the training examples that ended there.

Training chooses the questions. A good question splits a mixed group into
groups that are each mostly one label. So the tree needs a measure of how
mixed a group's labels are. The
usual one is **entropy** (Probability §3.5.1):

$$H = -\sum_c p_c \log_2 p_c$$

where $p_c$ is the fraction of the group in class $c$. A 50/50 group has
$H = 1$ bit, a 90/10 group 0.47 bits, a pure group 0.

### 7.2 Information gain: choosing a split

Example: 10 emails, 5 spam and 5 not, so $H = 1$. Splitting on "contains
'free'" gives a left group with 4 spam and 1 not, and a right group with 1
spam and 4 not. Each group has $H = 0.722$. The **information gain** is the
parent's entropy minus the size-weighted entropy of the children:

$$IG = H(\text{parent}) - \tfrac{n_L}{n}H(\text{left}) - \tfrac{n_R}{n}H(\text{right}) = 1 - 0.5 \times 0.722 - 0.5 \times 0.722 = 0.278 \text{ bits}$$

A split that leaves both children 50/50 has gain 0. Building a tree is
greedy: at each node, try every feature and every threshold, keep the split
with the highest gain, and repeat on each child until a stopping rule (max
depth, too few examples, pure node). Finding the best possible tree overall
is NP-hard, so the greedy method is used.

**Gini impurity** $G = 1 - \sum_c p_c^2$ is the common alternative
($2p(1-p)$ for two classes). Gini and entropy choose the same split almost
always; Gini is the default in many libraries because it needs no logarithm.

### 7.3 Why trees overfit

A deep enough tree can give every training example its own leaf and reach
100% training accuracy on any data: **zero bias, high variance**. Deep
splits are chosen from a handful of examples, so they follow noise: a leaf
with 3 examples predicts from 3 noisy labels. And changing one training
point can change a split near the top, which changes everything below it.

Limits on growth help: `max_depth`, `min_samples_leaf`, or pruning. The
stronger fix is to average many trees (§8).

---

## 8. Ensembles

### 8.1 Bagging: averaging removes variance

A model's **variance** is how much its predictions would change if you
trained it on a different sample of data (Probability §3.7). A deep tree has
high variance: retrain it on slightly different data and many predictions
change.

**Averaging cancels errors, but only independent ones.** Ask 100 people to
guess a cow's weight: each guess is off, but the errors point in different
directions, so the average is much closer than a typical guess. If all 100
copied one person, averaging would gain nothing. The same holds for models.
For $B$ models, each with variance $\sigma^2$ and pairwise correlation
$\rho$ (how much their errors move together), the average has

$$\operatorname{Var}(\bar f) = \rho\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

The second term goes to 0 with more models; the first, $\rho\sigma^2$, is a
floor. With $\sigma^2 = 1$ and $B = 100$: if $\rho = 0$ the variance is
0.01, if $\rho = 0.5$ it is 0.505, about 50 times larger. Correlation, not the
number of models, is the limit.

**Bagging** (bootstrap aggregating) makes different models from one dataset:
train each on a **bootstrap sample** ($n$ draws with replacement), then
average the predictions (or take a majority vote). Each bootstrap sample
leaves out about 37% of the examples ($(1 - 1/n)^n \to 1/e$); scoring each
model on its left-out examples gives the **out-of-bag error**, a validation
score with no separate holdout set.

Bagging reduces variance, not bias, so it helps models with low bias and high
variance, such as deep trees. Bagging linear regression gains almost
nothing.

### 8.2 Random forests: lowering the correlation

Bagged trees are still similar: if one feature is strongest, every tree
splits on it first, so $\rho$ stays high. A **random forest** considers only
a random subset of about $\sqrt d$ features at each split. The strong feature
is often unavailable, trees split on different features, and $\rho$ drops.
Each tree gets slightly worse ($\sigma^2$ rises), but the floor $\rho\sigma^2$
falls. In `trees_and_ensembles_scratch.ipynb`, going from all 6 features
(plain bagging) to 2 per split cuts $\rho$ from 0.167 to 0.055, raises
$\sigma^2$ from 0.061 to 0.101, and lowers the floor from 0.0102 to 0.0056.

Too few features per split hurts: with only 1, many splits have no useful
feature to choose from and accuracy is worst. The subset size is a
hyperparameter to tune.

### 8.3 Boosting: gradient descent with trees

Bagging trains trees independently and averages them. **Boosting** trains
small trees one after another, each fixing what the trees so far still get
wrong. Each small tree is too simple on its own (high bias), but their sum
is a strong model, so boosting reduces bias.

**Example.** Three houses sold for 200, 300 and 400 (thousands).

1. Start with the average: predict 300 for all three. The misses
   (**residuals**) are $-100$, $0$, $+100$.
2. Fit a small tree to the residuals. Say it learns "small house: $-100$,
   medium: 0, large: $+100$". Add it with a step size $\eta = 0.5$: the
   predictions become 250, 300, 350.
3. The new residuals are $-50$, $0$, $+50$. Fit another tree to those, add
   half of it, and so on. Each round closes part of the remaining gap.

**Why it's called gradient boosting.** For squared loss
$\frac12(y - F)^2$, the residual $y - F(x)$ is exactly the negative gradient
of the loss with respect to the prediction $F(x)$. So "fit a tree to the
residuals and take a step" is gradient descent, where each step is a tree.
For other losses, fit the tree to the negative gradient instead, called the
**pseudo-residual**:

$$r_i = -\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}, \qquad F \leftarrow F + \eta\, h \quad (h \text{ fit to the } r_i)$$

For log-loss, $r_i = y_i - p_i$, the same $y - p$ as logistic regression
([§2.2](01_linear_models.md#2.2)). A smaller step $\eta$ (typically 0.01 to 0.3)
needs more trees but usually gives lower test error.

Boosting keeps adding capacity, so too many rounds overfit: stop early on a
validation set. Bagging does not overfit by adding trees, since that only
lowers variance.

### 8.4 XGBoost

XGBoost is gradient boosting with three additions:

- **Second-order steps.** Plain gradient boosting uses only the slope of
  the loss. XGBoost also uses its curvature, like Newton's method. With
  $g_i$ and $h_i$ the first and second derivatives of the loss for example
  $i$, the best value for a leaf is
  $w^* = -\sum g_i / (\sum h_i + \lambda)$, summing over the examples in the
  leaf. For squared loss every $h_i = 1$, so this is the leaf's average
  residual, shrunk toward 0 by $\lambda$. The same sums give each candidate
  split an exact score, so splits are chosen on the real objective.
- **Regularization:** a cost $\gamma$ per leaf and an L2 penalty $\lambda$ on
  leaf values, plus shrinkage and row and column subsampling.
- **Speed:** histogram-based split finding, a learned default direction for
  missing values, and parallel split search. (LightGBM and CatBoost are
  similar libraries.)

Gradient-boosted trees are still the best default for tabular data: they
handle mixed feature types, different scales, missing values and sharp
interactions without feature engineering.
