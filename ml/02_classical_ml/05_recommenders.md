# Recommenders: collaborative filtering and matrix factorization

---

## 13. Collaborative filtering and matrix factorization

Companion notebook: `matrix_factorization_scratch.ipynb`. This is the baseline
recommender interviewers expect before deep learning; serving and two-stage
retrieval are in module 6 (`../06_ml_systems/`).

### 13.1 The user-item matrix

Put users in rows, items in columns, and ratings (or clicks) in the cells.
Most cells are empty: the Netflix Prize matrix had 480k users, 17.8k movies
and 100M ratings, so 98.8% was missing. Write $\Omega$ for the set of observed
(user, item) pairs. The task is to fill in the missing cells, or to rank each
user's unseen items.

Running example:

|       | M1 | M2 | M3 | M4 |
|-------|----|----|----|----|
| Alice | 5  | 4  | ?  | 1  |
| Bob   | 4  | 5  | 1  | ?  |
| Carol | 1  | ?  | 5  | 4  |
| Dave  | ?  | 1  | 4  | 5  |

There are two taste groups: Alice and Bob like M1 and M2; Carol and Dave
like M3 and M4. Each method below finds and uses this structure.

### 13.2 Neighborhood methods: "people like you liked..."

**User-user.** Measure how similar two users are with the cosine of their
ratings on items both rated:

- Alice and Bob (both rated M1, M2): $\frac{5\cdot4 + 4\cdot5}{\sqrt{41}\sqrt{41}} = 0.98$.
- Alice and Carol (both rated M1, M4): $\frac{5\cdot1 + 1\cdot4}{\sqrt{26}\sqrt{17}} = 0.43$.

Bob is Alice's nearest neighbor and rated M3 a 1, so predict that Alice
dislikes M3. The general prediction is a similarity-weighted average of the
neighbors' ratings, each measured from that neighbor's own average (some
people rate everything high):

$$\hat r_{ui} = \bar r_u + \frac{\sum_{v} \text{sim}(u,v)\,(r_{vi} - \bar r_v)}{\sum_{v} |\text{sim}(u,v)|}$$

**Item-item** does the same with columns ("people who liked X also liked
Y"). It usually works better: item similarities change slowly and can be
precomputed (Amazon's classic recommender).

Weaknesses: with 99% missing, users share few rated items, so similarities
are noisy; and two users with no items in common get similarity 0 even if
their tastes match.

### 13.3 Matrix factorization

The example matrix is roughly **rank 2**: each user can be described by how
much they like group A versus group B, and each item by how much it belongs
to A versus B. In general, give each user a vector $p_u$ and each item a
vector $q_i$, both of length $r$, and predict

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$$

where $\mu$ is the overall average rating, $b_u$ the user's bias (harsh or
generous) and $b_i$ the item's (widely liked or not). The biases alone
explain a large part of the ratings; the vectors then capture real taste.

Fit by minimizing squared error over **observed entries only**, with L2 on
all parameters:

$$\min \sum_{(u,i) \in \Omega} (r_{ui} - \hat r_{ui})^2 + \lambda\Big(\sum_u \|p_u\|^2 + \sum_i \|q_i\|^2 + \text{biases}^2\Big)$$

**Why not just take the SVD?** The SVD gives the best low-rank fit to every
cell, but 99% of the cells are missing. Filling them with zeros makes the fit
mostly about reproducing zeros; filling them with averages makes it mostly
about your own guesses. The sum must run over observed cells only, and then
the SVD formula no longer applies.

**Why the L2 penalty is needed:** a user with 3 ratings and $r = 20$ factors
has more unknowns than equations, so without it the 3 ratings are fit
exactly in infinitely many ways.

### 13.4 Training: ALS and SGD

**Alternating least squares (ALS).** With the item vectors fixed, each
user's vector is a separate ridge regression ([§3.1](01_linear_models.md#3.1)): the features are the
vectors of the items the user rated, and the targets are the ratings.

$$p_u = (Q_u^\top Q_u + \lambda I)^{-1} Q_u^\top r_u$$

Solve this for every user (independently, so in parallel), then fix the user
vectors and solve for every item the same way, and repeat. Each half-step is
an exact minimum for its half, so the loss never increases: the same
alternating pattern as k-means ([§9.2](04_clustering_and_dim_reduction.md#9.2)) and EM ([§11.2](04_clustering_and_dim_reduction.md#11.2)).

**SGD** instead updates on one rating at a time: with error
$e = r_{ui} - \hat r_{ui}$, set $p_u \mathrel{+}= \eta(e\, q_i - \lambda p_u)$ and
the same for $q_i$. It is cheap per pass but needs a learning rate. ALS needs
none, and it is the standard choice for implicit feedback (§13.5), where the
loss covers every cell. In `matrix_factorization_scratch.ipynb` both reach
about the same test error (RMSE 0.52 for SGD, 0.54 for ALS, against a noise
floor of 0.4).

### 13.5 Implicit feedback and BPR

Real logs mostly have clicks, views and purchases, not ratings. These are
positive-only: a missing click could mean "didn't like it" or "never saw
it". Two standard approaches:

- **Implicit ALS:** treat every unobserved cell as 0 with low confidence and
  every observed cell as 1 with high confidence.
- **BPR** (Bayesian personalized ranking): learn a ranking instead of values.
  Sample triples $(u, i, j)$ where user $u$ interacted with item $i$ but not
  $j$, and maximize
  $\sum \log \sigma(\hat x_{ui} - \hat x_{uj})$ minus an L2 penalty. This is
  logistic regression ([§2](01_linear_models.md#2)) on score differences, and it optimizes a smooth
  version of AUC. Most modern recommender losses descend from it.

### 13.6 Cold start

A new item has no interactions, so nothing pushes its vector away from 0.
Pure collaborative filtering cannot recommend it. Common fixes: fall back to
popularity; build the item's vector from its **content** (text, images,
metadata) so it lands near similar items; ask new users a few onboarding
questions; and deliberately show new items to some users to collect data.

### 13.7 Bridge to two-tower models

Matrix factorization is the simplest **two-tower model**: one "tower" maps a
user to $p_u$, the other maps an item to $q_i$, and the score is their dot
product. Replace the lookup tables with neural networks over features (user
history, item content) and two things follow. Retrieval stays a
nearest-neighbor search over precomputed item vectors, served by an
approximate nearest neighbor (ANN) index. And cold start is fixed, because a
new item gets a vector from its features. Module 6 covers the production
side.
