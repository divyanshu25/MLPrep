# Interview questions & exit checklist

**Q1. Why does least squares correspond to Gaussian noise?**

**Because the negative log of a Gaussian density is a squared error plus a
constant.**

- With noise $\mathcal N(0, \sigma^2)$, a residual $r$ has
  $-\log p = r^2/(2\sigma^2) + \text{const}$.
- Summed over examples, maximizing the likelihood is minimizing
  $\sum_i r_i^2$ ([§1.1](01_linear_models.md#1.1)).

**If asked:** *What if the noise is Laplace?* Then $-\log p = |r|/b$ plus a
constant, and you minimize absolute error, which is less affected by
outliers.

**Q2. Normal equation or gradient descent?**

**Normal equation for a modest number of features with squared loss;
gradient descent for many features, streaming data, or other losses.**

- The normal equation costs about $d^3$: $10^6$ operations at $d = 100$, but
  $10^{18}$ at $d = 10^6$.
- A gradient step costs $nd$ and works for any differentiable loss, such as
  logistic loss.

**If asked:** *What if $X^\top X$ is singular?* Columns are dependent
(duplicates, or $d > n$); ridge's $\lambda I$ makes it invertible.

**Q3. Derive the gradient of the logistic loss.**

**The gradient is $\nabla_w L = (p - y)\,x$.**

$$\frac{\partial L}{\partial p} = \frac{p-y}{p(1-p)}, \quad \frac{\partial p}{\partial z} = p(1-p), \quad \frac{\partial z}{\partial w} = x$$

- The $p(1-p)$ cancels.
- For $y = 1$, $p = 0.1$: the gradient on $z$ is $-0.9$, a large correction
  when the model is badly wrong.

**Q4. Why does L1 give exact zeros but L2 doesn't?**

**L1's slope at zero is $\pm\lambda$, not 0, so a weight stays at zero unless
the data pulls harder than $\lambda$.**

- One weight with data loss $(w - a)^2$: lasso gives
  $\max(0, |a| - \lambda/2)$ with the sign of $a$; ridge gives $a/(1+\lambda)$.
- $a = 0.5$, $\lambda = 2$: lasso gives exactly 0, ridge 0.17.

**If asked:** *Geometrically?* The L1 ball is a diamond with corners on the
axes, and the loss contours usually touch it first at a corner.

**Q5. What is the Bayesian interpretation of ridge and lasso?**

**Ridge is MAP with a Gaussian prior on the weights; lasso is MAP with a
Laplace prior.**

- MAP minimizes $-\log(\text{likelihood}) - \log(\text{prior})$, and the
  negative log of the prior is the penalty.
- $\lambda = \sigma^2/\tau^2$: noisier data or a tighter prior means more
  shrinkage.

**Q6. What is $C$ in an SVM?**

**The cost of margin violations, the inverse of the regularization
strength.**

- Large $C$: fit the training data closely, narrow margin, risk of
  overfitting.
- Small $C$: wide margin, more violations allowed, risk of underfitting.

**If asked:** *Where is the L2 term?* In the hinge form, the penalty is
$\|w\|^2/(2C)$.

**Q7. What does the kernel trick buy you?**

**A linear method in a huge feature space at the cost of a dot product in
the original space.**

- $\langle \phi(x), \phi(z) \rangle = \langle x, z \rangle^2$ for all degree-2
  products of two features; degree 10 on 100 features would need about
  $4 \times 10^{13}$ features.
- The SVM only needs dot products, so it never builds $\phi$.

**If asked:** *The cost?* An $n \times n$ kernel matrix: $n^2$ memory, $n^2$ to
$n^3$ time.

**Q8. Gini or entropy for tree splits?**

**It rarely matters: they pick the same split almost always.**

- Both are 0 for a pure node and largest for a 50/50 node.
- Gini avoids the logarithm, so it is the usual default.

**Q9. Why do random forests subsample features on top of bagging?**

**To make the trees less correlated, which lowers the variance floor.**

- Averaging $B$ trees gives variance $\rho\sigma^2 + (1-\rho)\sigma^2/B$; more
  trees can't remove $\rho\sigma^2$.
- If one feature dominates, every bagged tree splits on it first. Hiding it
  from some splits makes the trees differ.

**If asked:** *The downside?* Each tree gets a little worse; with too few
features per split, accuracy drops.

**Q10. Bagging vs. boosting?**

**Bagging averages independent deep trees to cut variance; boosting adds
shallow trees in sequence to cut bias.**

- Bagging: parallel, more trees never overfits.
- Boosting: sequential, too many rounds overfit, so stop early on a
  validation set.

**Q11. In what sense is gradient boosting gradient descent?**

**Each tree is fit to the negative gradient of the loss at the training
predictions, then added with a small step.**

- For squared loss the negative gradient is the residual $y - F(x)$, so each
  tree fits what the model still gets wrong.
- For log-loss it is $y - p$.

**Q12. Why does k-means always converge, and to what?**

**Each step lowers or keeps the objective, and there are finitely many
assignments, so it stops, at a local minimum.**

- Assigning to the nearest center and moving centers to means each lower
  the total squared distance.
- The result depends on the starting centers: use k-means++ and several
  restarts.

**Q13. Why standardize features for kNN, k-means, SVMs and regularized
models?**

**They depend on distances or weight sizes, so a feature with larger numbers
dominates.**

- A mass in grams instead of kilograms makes its differences 1000 times
  larger in every distance.
- Under a penalty, it needs a 1000 times smaller weight, so it is barely
  penalized.

**If asked:** *Which models don't care?* Trees: a split threshold works the
same in any units.

**Q14. State the curse of dimensionality with one number.**

**In 100 dimensions, a cube holding 1% of uniformly spread data has sides
95% as long as the whole range.**

- The side is $0.01^{1/d}$: 0.1 at $d = 2$, 0.63 at $d = 10$, 0.955 at
  $d = 100$.
- So the "nearest" neighbors are not near, and distances barely differ.

**Q15. PCA: variance maximization or reconstruction error?**

**Both: they give the same answer.**

- By Pythagoras, $\|x\|^2$ = kept part + lost part, and $\|x\|^2$ is fixed,
  so less lost means more kept.
- Both are solved by the top eigenvectors of the covariance; the lost part
  is the sum of the dropped eigenvalues.

**Q16. Why is naive Bayes a fine classifier but a poor probability
estimator?**

**Correlated features are counted as independent evidence, which pushes
probabilities toward 0 and 1, but often leaves the winning class right.**

- Classification needs only the right winner.
- So use it for ranking or quick baselines, not calibrated probabilities.

**Q17. Why does logistic regression fail on perfectly separable data?**

**The weights grow without limit, because larger weights always lower the
loss.**

- Scaling $w$ up pushes every probability closer to its correct 0 or 1, and
  the loss toward 0.
- Fix: L2 regularization or early stopping (scikit-learn regularizes by
  default).

**Q18. Hinge loss vs. logistic loss?**

**Hinge is exactly zero once a point is past the margin; logistic is always
positive.**

- So an SVM depends only on support vectors, while every point pulls on
  logistic regression a little.
- Logistic gives probabilities; hinge gives only a side of the line.

**Q19. Where does the 37% out-of-bag figure come from?**

**A point is missed by one draw with probability $1 - 1/n$, so by all $n$
draws with probability $(1 - 1/n)^n \to 1/e \approx 0.37$.**

- Each tree's left-out 37% gives a free validation score, the out-of-bag
  error.

**Q20. Why do gradient-boosted trees still beat neural networks on tabular
data?**

**Trees handle mixed types, different scales, missing values and sharp
thresholds directly; networks need careful preprocessing and tuning.**

- A split like "income > 50k" is one node; a network has to approximate the
  step.
- Tabular datasets are often small, where trees' built-in limits help.

**Q21. What convergence guarantee does EM have?**

**The log-likelihood never decreases; it reaches a local optimum, not
necessarily the global one.**

- The E-step makes the lower bound equal the log-likelihood; the M-step
  raises the bound.
- The result depends on the start: initialize with k-means and restart.

**If asked:** *Can the likelihood be unbounded?* Yes: a cluster collapsing
onto one point; add $\varepsilon I$ to the covariances.

**Q22. GMM vs. k-means?**

**k-means is the limit of a GMM with equal round clusters and hard
assignments.**

- Use a GMM for soft memberships, elliptical clusters, or a density.
- Use k-means when a fast hard partition is enough.

**Q23. PCA vs. t-SNE?**

**PCA for features and compression; t-SNE for a 2-D picture of clusters.**

- PCA is a fixed linear map that applies to new data.
- t-SNE has no map for new points, and its cluster distances and sizes mean
  nothing.

**If asked:** *Why the "t"?* Its heavy tail lets moderately distant points
sit far apart in the map, which fixes crowding.

**Q24. Why not just run the SVD on a ratings matrix?**

**The SVD fits every cell, but most cells are missing.**

- Zeros in the missing cells make the fit mostly about reproducing zeros.
- Fit only the observed ratings, with L2, using ALS or SGD.

**Q25. What is cold start and how do you handle it?**

**A new user or item has no interactions, so collaborative filtering can't
learn its vector.**

- Fall back to popularity, or build the vector from content features.
- Collect data quickly: onboarding questions, or deliberately showing new
  items.

**Q26. Random forest or gradient boosting?**

**Gradient boosting for the best accuracy with tuning; a random forest for
a strong result with almost no tuning.**

- Random forests are hard to overfit with more trees and train in parallel.
- Boosting needs a learning rate, depth and early stopping, but usually wins
  when tuned.

---

## Exit checklist

- [ ] Say why squared loss is the Gaussian maximum-likelihood loss, and write the normal equation (3 min)
- [ ] Derive the logistic-regression gradient $(p - y)x$ (3 min)
- [ ] Show with the one-weight formulas why L1 gives exact zeros and L2 doesn't (3 min)
- [ ] Explain the kernel trick: where $K(x, z)$ enters and why $\phi$ is never built (3 min)
- [ ] Write the bagging variance formula and say what random forests change (3 min)
- [ ] Explain boosting as gradient descent: what each new tree fits (2 min)
- [ ] Derive PCA as an eigenvector problem and connect it to the SVD (4 min)
- [ ] Explain EM's two steps and why the likelihood never decreases (3 min)
- [ ] Explain why plain SVD fails on a ratings matrix and how ALS fixes it (3 min)

All boxes cold before moving on; a miss goes into `practice/queue.tsv`.
