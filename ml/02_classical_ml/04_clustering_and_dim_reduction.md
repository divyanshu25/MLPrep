# Clustering and dimensionality reduction: k-means, PCA, GMM and EM, t-SNE

Companion notebooks: `kmeans_pca_scratch.ipynb` (§9, §10) and `gmm_em_scratch.ipynb` (§11).

---

## 9. k-means

### 9.1 The algorithm

k-means splits unlabeled data into $k$ groups of nearby points. Example: the
six numbers 1, 2, 3, 10, 11, 12 with $k = 2$, starting from centers 1 and 2.

| round | assign each point to the nearest center | move each center to its points' mean |
|---|---|---|
| 1 | {1} and {2, 3, 10, 11, 12} | 1 and 7.6 |
| 2 | {1, 2, 3} and {10, 11, 12} | 2 and 11 |
| 3 | no change: stop | |

Even from a poor start, it finds the two obvious groups in two rounds.

Formally, k-means minimizes the total squared distance from each point to
its cluster's center:

$$J = \sum_i \|x_i - \mu_{z_i}\|^2$$

where $z_i$ is point $i$'s cluster and $\mu_j$ is cluster $j$'s center.
Minimizing over both at once is NP-hard, but each half is easy with the other
fixed, so **Lloyd's algorithm** alternates:

1. **Assign:** put each point in the cluster with the nearest center.
2. **Update:** move each center to the mean of its points. The mean is the
   point with the smallest total squared distance to a set of points (set
   the derivative of $\sum \|x_i - \mu\|^2$ to zero), hence the name
   k-*means*.

### 9.2 Why it converges

Each step can only lower $J$ or leave it unchanged: assigning to the nearest
center lowers each point's distance, and moving to the mean lowers each
cluster's total. $J$ can't go below 0 and there are finitely many
assignments, so the algorithm stops. It stops at a **local** minimum that
depends on the starting centers, not necessarily the best clustering.
`kmeans_pca_scratch.ipynb` checks that $J$ never increases.

### 9.3 Failure modes

1. **Bad starting centers** give a bad local minimum, such as two centers in
   one true cluster. Fixes: several restarts, keeping the lowest $J$; and
   **k-means++**, which picks each new starting center with probability
   proportional to its squared distance from the existing centers.
2. **Wrong shapes.** Nearest-center assignment cuts space with straight
   boundaries, which suits round clusters of similar size. Long, ring-shaped
   or uneven-density clusters get split wrongly. Use Gaussian mixtures (§11)
   for elliptical clusters, or DBSCAN (which grows clusters from points with
   many close neighbors) for arbitrary shapes.
3. **Choosing $k$.** $J$ always falls as $k$ grows (it is 0 when every point
   is its own cluster), so minimizing $J$ can't choose $k$. Use the elbow of
   the $J$-versus-$k$ curve, the silhouette score, or domain knowledge.
4. **Scale and outliers.** Distances need standardized features, and one
   outlier drags its cluster's mean toward it.

---

## 10. PCA

Height and weight are strongly related: tall people tend to be heavier. So
most of the variation between people lies along one direction, "overall
size" (more height and more weight together), and much less along "heavy
for one's height". Describing each person by their position along the size
direction alone loses little. **PCA** (principal component analysis) finds
these directions automatically: the directions along which the data varies
most, so you can keep a few and drop the rest.

Example with numbers: if two features have covariance matrix
$\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$, the direction $(1, 1)/\sqrt2$
(both up together) holds variance 3, and the direction $(1, -1)/\sqrt2$ (one
up, one down) holds variance 1. Keeping only the first keeps $3/4$ of the
variance with half the numbers.

Throughout, center the data first (subtract each feature's mean) and call the
covariance matrix $\Sigma = \frac{1}{n-1} X_c^\top X_c$.

### 10.1 Directions of maximum variance

Projecting each point onto a unit direction $u$ (taking the dot product
$u^\top x$) gives one number per point, and those numbers have variance
$u^\top \Sigma u$. We want the direction that makes this as large as
possible. Maximizing $u^\top \Sigma u$ subject to $\|u\| = 1$ (with a
Lagrange multiplier) gives

$$\Sigma u = \lambda u$$

So the best directions are the **eigenvectors** of $\Sigma$, and the variance
along each is its eigenvalue $\lambda$ (Linear algebra §1.5). The first
principal component is the top eigenvector, the second is the next one
(perpendicular to the first), and so on. The **explained variance ratio** of
component $j$ is $\lambda_j / \sum_l \lambda_l$; in the example, $3/4$.

### 10.2 The same answer from reconstruction error

A second goal gives the same answer: compress each point to $k$ numbers and
rebuild it with as little error as possible. Keep $k$ perpendicular
directions (the columns of $U$); the $k$ numbers are $U^\top x$, and the
rebuilt point is $UU^\top x$. By Pythagoras, each point's squared length
splits into the part along the kept directions and the part lost:

$$\|x\|^2 = \underbrace{\|U^\top x\|^2}_{\text{kept}} + \underbrace{\|x - UU^\top x\|^2}_{\text{lost}}$$

The left side is fixed, so making the lost part small is the same as making
the kept part (the projected variance) large: the same problem as §10.1. With
the top $k$ eigenvectors, the total lost is proportional to the sum of the
eigenvalues left out, $\sum_{j > k}\lambda_j$. That's why the
reconstruction error drops sharply until $k$ reaches the data's true
dimension and flattens after it (`kmeans_pca_scratch.ipynb`).

### 10.3 Computing it with the SVD

With the SVD $X_c = U S V^\top$ (Linear algebra §1.6), the columns of $V$
are the principal components and $\lambda_j = s_j^2/(n-1)$. Libraries use
the SVD of $X_c$ instead of forming $\Sigma$, for two reasons. Forming
$\Sigma$ squares the singular values, so small ones become tiny and get lost
in rounding error. And the SVD is cheaper when there are more features than
examples ($d \gg n$). The projected data is $X_c V = US$.

**Traps:**

- **Center first.** Without centering, the first component points at the
  mean, not along the spread.
- **Standardize features with different units**, or the feature with the
  largest numbers dominates.
- **Signs are arbitrary:** $u$ and $-u$ are the same component.
- **High variance is not the same as useful.** PCA ignores labels, so the
  top direction can be useless for a classification task.

---

## 11. Gaussian mixture models and EM

### 11.1 The model

Suppose you measure the heights of 1,000 adults but don't record who is a
woman or a man. The histogram has two overlapping bells, one per group. A
**Gaussian mixture model** (GMM) fits exactly this: several bells, how big
each one is, and for each person how likely they came from each bell, all
without the labels. Unlike k-means, a point can belong partly to two
clusters, and clusters can be stretched in any direction.

The model says each point was made in two steps:
pick a cluster $j$ with probability $\pi_j$, then draw the point from that
cluster's Gaussian $\mathcal N(\mu_j, \Sigma_j)$. The cluster label $z_i$ is
never observed (it is **latent**), so the density of a point sums over the
clusters:

$$p(x) = \sum_{j=1}^{k} \pi_j\, \mathcal N(x;\ \mu_j, \Sigma_j)$$

**Responsibilities.** Given a point, Bayes rule says how likely each
cluster is to have produced it:

$$\gamma_{ij} = p(z_i = j \mid x_i) = \frac{\pi_j\, \mathcal N(x_i; \mu_j, \Sigma_j)}{\sum_l \pi_l\, \mathcal N(x_i; \mu_l, \Sigma_l)}$$

Example in 1-D: two clusters with equal weights, standard deviation 1,
centers 0 and 4. The point $x = 1$ has Gaussian heights proportional to
$e^{-1/2} = 0.607$ and $e^{-9/2} = 0.011$, so responsibilities 0.98 and
0.02. The point $x = 2$, halfway, gets 0.5 and 0.5.

**Why maximum likelihood is hard.** The log-likelihood is
$\sum_i \log \sum_j \pi_j \mathcal N(x_i; \mu_j, \Sigma_j)$: a log of a sum,
which doesn't split into one problem per cluster. If the labels were known,
it would split, and each cluster's parameters would just be its fraction,
mean and covariance. The missing labels are the whole difficulty.

### 11.2 The EM algorithm

The idea: if you knew the labels, fitting the bells would be easy; if you
knew the bells, guessing the labels would be easy. So start with a guess for
the bells and alternate. For the heights: guess two bells, compute for each
person how likely each bell is (a person of 160 cm is mostly in the shorter
bell), refit each bell from the people weighted by those numbers, and
repeat until nothing changes.

**EM** (expectation-maximization) is this loop:

1. **E-step:** compute the responsibilities $\gamma_{ij}$ from the current
   parameters: soft labels.
2. **M-step:** refit each cluster's parameters as if the soft labels were
   real, weighting each point by its responsibility (§11.3).

**Why the likelihood never goes down.** For any guess $q_i$ of the
probabilities of point $i$'s cluster, the log-likelihood splits into a lower
bound plus a KL divergence (Probability §3.5), which is never negative:

$$\log p(x_i \mid \theta) = \underbrace{\mathcal L_i(q_i, \theta)}_{\text{lower bound (ELBO)}} + \underbrace{\mathrm{KL}\big(q_i \,\|\, p(z_i \mid x_i, \theta)\big)}_{\ge\, 0}$$

The E-step sets $q_i$ to the responsibilities, which makes the KL zero, so
the bound equals the log-likelihood. The M-step raises the bound by changing
$\theta$, and the true log-likelihood is at least the bound. So

$$\ell(\theta^{t+1}) \ \ge\ \mathcal L(q^t, \theta^{t+1}) \ \ge\ \mathcal L(q^t, \theta^t) \ =\ \ell(\theta^t)$$

The bound itself comes from Jensen's inequality ($\log$ of an average is at
least the average of the $\log$). EM reaches a local optimum that depends on
the start, so initialize with k-means and use several restarts.
`gmm_em_scratch.ipynb` checks that the log-likelihood never decreases.

### 11.3 The update formulas

With $N_j = \sum_i \gamma_{ij}$, the effective number of points in cluster
$j$, setting the M-step derivatives to zero gives

$$\pi_j = \frac{N_j}{n}, \qquad \mu_j = \frac{1}{N_j}\sum_i \gamma_{ij}\, x_i, \qquad \Sigma_j = \frac{1}{N_j}\sum_i \gamma_{ij}\,(x_i - \mu_j)(x_i - \mu_j)^\top$$

Each one is the ordinary estimate from labeled data (cluster fraction,
mean, covariance) with each point counted by its responsibility instead of
0 or 1. Compute the E-step in log space with log-sum-exp, since Gaussian
densities in many dimensions underflow.

**Trap: the likelihood has no maximum.** Put one cluster's mean exactly on a
data point and shrink its variance toward 0: that point's density grows
without limit. Fix: add a small $\varepsilon I$ to every covariance
(scikit-learn's `reg_covar`), or put a prior on $\Sigma$.

### 11.4 k-means is EM with hard assignments

Fix every covariance to $\varepsilon I$ and every weight to $1/k$. The
responsibilities become a softmax over negative squared distances divided by
$2\varepsilon$. For a point at squared distances 1 and 2 from two centers,
the nearer center's responsibility is

| $\varepsilon$ | 1 | 0.1 | 0.01 |
|---|---|---|---|
| responsibility | 0.62 | 0.993 | $\approx 1$ |

As $\varepsilon \to 0$ the E-step becomes "assign to the nearest center",
the M-step becomes "take the mean", and EM becomes Lloyd's algorithm (§9.1).
So k-means is a mixture of equal-sized round clusters, which is why it
struggles with other shapes (§9.3).

**GMM or k-means?** Use a GMM when you need soft memberships, elliptical or
different-sized clusters, or an actual density $p(x)$ (for anomaly
detection, sampling, or comparing $k$ by likelihood). Use k-means when a fast
hard partition is enough: it is cheaper, needs no covariance estimates
($kd(d+1)/2$ numbers for full covariances), and is the usual way to
initialize a GMM anyway.

---

## 12. t-SNE and UMAP vs. PCA

### 12.1 What PCA can't do

PCA is a linear projection that keeps the directions of largest variance. Two
consequences:

- **Curved structure is lost.** Two rings of radius 1 and 3 around the same
  center are clearly separate in 2-D, but every straight-line projection to
  1-D overlaps them, because they differ in radius, not direction.
- **Largest variance is not the same as neighborhoods.** For a picture of
  clusters, the question is "which points are near which?", a local
  question. PCA answers a global one.

t-SNE and UMAP are built to keep neighbors near each other in a 2-D map.

### 12.2 How t-SNE works

1. **Neighbor probabilities in the original space.** For each point $i$,
   turn distances to the others into probabilities with a Gaussian centered
   at $x_i$:
   $p_{j\mid i} \propto \exp\!\big(-\|x_i - x_j\|^2 / 2\sigma_i^2\big)$, then
   symmetrize, $p_{ij} = (p_{j\mid i} + p_{i\mid j})/2n$.
2. **Perplexity sets each width $\sigma_i$.** It is roughly the number of
   neighbors each point pays attention to (typically 5 to 50). Points in
   dense regions get a small $\sigma_i$, sparse ones a large one. Small
   perplexity gives many small clusters (even from noise); large perplexity
   blurs them together.
3. **Neighbor probabilities in the map, with a heavy tail.** For map points
   $y_i$, use a Student-t kernel instead of a Gaussian:
   $q_{ij} \propto (1 + \|y_i - y_j\|^2)^{-1}$.
4. **Match them.** Move the map points by gradient descent to minimize
   $\mathrm{KL}(P \,\|\, Q) = \sum_{i \ne j} p_{ij}\log(p_{ij}/q_{ij})$. Each
   pair pulls together if $p_{ij} > q_{ij}$ and pushes apart if
   $q_{ij} > p_{ij}$.

**Why the heavy tail (the "t").** High-dimensional space has far more room at
moderate distances than a 2-D map. With a Gaussian in the map, all those
moderately distant points get squeezed together (the **crowding problem**).
The t kernel falls off much more slowly:

| map distance | 2 | 4 |
|---|---|---|
| Gaussian $e^{-u^2/2}$ | 0.135 | 0.0003 |
| t kernel $1/(1 + u^2)$ | 0.200 | 0.059 (175 times more) |

So a pair that is moderately far apart can be placed far apart in the map at
little cost, and clusters separate with clear gaps between them.

**Why KL in this direction.** $\mathrm{KL}(P\|Q)$ charges $p\log(p/q)$:
placing true neighbors far apart (large $p$, tiny $q$) costs a lot, while
placing non-neighbors close (tiny $p$) costs almost nothing. So t-SNE keeps
local structure and largely ignores global layout, which explains the
limitations below.

### 12.3 Limitations

1. **No mapping for new points.** t-SNE learns coordinates for these $n$
   points only. A new point needs a rerun, so never use t-SNE coordinates as
   features for a model.
2. **Distances between clusters and cluster sizes mean nothing.** The
   per-point widths expand dense clusters and shrink sparse ones, and the
   loss ignores global layout. Only "these points are neighbors" is
   reliable.
3. **Results vary with seed and perplexity**, and low perplexity can make
   clusters out of noise. Check any structure across several runs.
4. **Cost** is $O(n^2)$, or $O(n \log n)$ with the Barnes-Hut
   approximation. In practice, reduce to about 50 dimensions with PCA
   first.

### 12.4 UMAP

UMAP has the same structure: a neighbor graph in the original space and a
2-D layout that keeps neighbors close. It builds a $k$-nearest-neighbor graph
with edge weights between 0 and 1, then minimizes a cross-entropy between
the original and map edge weights with SGD. Each step pushes a point away
from a few random points (**negative sampling**) instead of normalizing
over all $n^2$ pairs. That makes it much faster, it keeps somewhat more
global structure, and it can map new points (`transform()`). The caveats
about cluster distances and sizes still apply.

### 12.5 PCA or t-SNE?

- **PCA** for features fed to a model (a fixed linear map that works the same
  on new data), compression, denoising, speed, interpretable directions, or
  preprocessing.
- **t-SNE or UMAP** for a 2-D picture of clusters for people to look at, in
  exploratory analysis only.

They work together: PCA to 50 dimensions, then t-SNE, then plot.
