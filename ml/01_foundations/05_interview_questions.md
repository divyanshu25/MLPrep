# Interview questions & exit checklist

Fifteen standard questions, grouped by topic. Work them as a drill: read the
question, answer out loud or on paper as if the interviewer just asked it,
and only then open the model answer. Any answer you could not produce cold
goes into `practice/queue.tsv`.

## Linear algebra

### Q1. Why does the dot product measure similarity?

<details><summary>Model answer</summary>

Because $a \cdot b = \|a\|\|b\|\cos\theta$: for fixed lengths it's maximal when the
vectors align, zero when orthogonal, negative when opposed. Normalized, it's cosine
similarity; unnormalized, it's the projection of one vector on the other scaled by
its length - "how much of $a$ points along $b$."
</details>

---

### Q2. What happens if you stack linear layers with no nonlinearity?

<details><summary>Model answer</summary>

$W_2(W_1 x) = (W_2 W_1) x$ - the composition of linear maps is a single linear map, so
depth adds no expressive power. Nonlinearities are what break this collapse.
</details>

---

### Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?

<details><summary>Model answer</summary>

Directions a matrix scales without rotating ($Av = \lambda v$); the top eigenvectors of
the data covariance matrix are the PCA axes (directions of maximal variance).
</details>

---

### Q4. What's the difference between eigendecomposition and SVD?

<details><summary>Model answer</summary>

Eigendecomposition needs a square matrix (and orthogonal eigenvectors only if
symmetric); SVD ($U\Sigma V^\top$) exists for *any* rectangular matrix and reads as
rotate → stretch → rotate. They connect via $A^\top A = V \Sigma^2 V^\top$. Truncated SVD
gives the best low-rank approximation (basis of PCA, recommenders, LoRA intuition).
</details>

---

### Q5. Why does L1 regularization give sparsity but L2 doesn't?

<details><summary>Model answer</summary>

The L1 ball is a diamond with corners on the coordinate axes; the constrained optimum
tends to land on a corner where some coordinates are exactly 0. Equivalently, L1's
gradient is a constant-magnitude push toward 0 that can win at $w=0$, while L2's push
$\propto w$ fades near 0, shrinking weights without zeroing them.
</details>

## Calculus

### Q6. Explain backprop in two sentences.

<details><summary>Model answer</summary>

Forward pass computes and caches all intermediates; backward pass walks the graph in
reverse applying the chain rule - multiply the upstream gradient by each node's local
derivative, summing where paths merge. It's reverse-mode autodiff, chosen because one
backward sweep gets the gradient w.r.t. *all* parameters at roughly the cost of one
forward pass.
</details>

---

### Q7. What is a Jacobian and what's its role in backprop?

<details><summary>Model answer</summary>

The $m \times n$ matrix of all partials $\partial f_i / \partial x_j$ - the local
linear approximation of a vector-valued function. Backprop propagates gradients via
vector-Jacobian products; the full Jacobian is rarely materialized.
</details>

---

### Q8. Why do gradients vanish or explode in deep/recurrent nets?

<details><summary>Model answer</summary>

The backward signal is a product of per-layer Jacobians; if their singular values are
consistently below/above 1, the product decays/grows exponentially with depth. Fixes:

- residual connections (Jacobian $\approx I$)
- careful initialization
- normalization layers
- gradient clipping
- gating (LSTM)
</details>

---

### Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?

<details><summary>Model answer</summary>

$\nabla_w = 2X^\top(Xw - y)$; setting it to zero gives the normal equations
$X^\top X w = X^\top y$, the closed-form least-squares solution.
</details>

## Probability & statistics

### Q10. Derive Bayes rule.

<details><summary>Model answer</summary>

From the definition $P(A|B) = P(A\cap B)/P(B)$: the joint factors both ways,
$P(A|B)P(B) = P(B|A)P(A)$; divide by $P(B)$ to get
$P(A|B) = P(B|A)P(A)/P(B)$.
</details>

---

### Q11. Why is minimizing cross-entropy the same as maximum likelihood?

<details><summary>Model answer</summary>

For a probabilistic classifier, the data's negative log-likelihood is
$-\sum_i \log p_\theta(y_i|x_i)$, which with one-hot labels is exactly the
cross-entropy loss. Minimizing one is minimizing the other. Similarly MSE = MLE under
Gaussian noise, L1 = MLE under Laplace noise.
</details>

---

### Q12. Positive test, rare disease

A test is 99% accurate for a disease with 1% prevalence; you test positive.
Roughly what's the chance you're sick?

<details><summary>Model answer</summary>

Around 17%, not 99% (with 95% specificity): per 10,000 people, 99 true positives vs
495 false positives, so $99/594 \approx 0.167$. The false positives swamp the rare
true positives - same reason accuracy misleads on imbalanced classes.
</details>

---

### Q13. Why divide by $n-1$ in the sample variance?

<details><summary>Model answer</summary>

Deviations are measured from the sample mean, which itself chases the data, so
$\frac1n\sum(x_i - \bar x)^2$ underestimates $\sigma^2$ by exactly the factor
$\frac{n-1}{n}$; dividing by $n-1$ makes the estimator unbiased (Bessel's correction).
</details>

---

### Q14. Why does averaging reduce noise, and at what rate?

<details><summary>Model answer</summary>

For i.i.d. terms, variances add while the $1/n$ prefactor comes out squared:
$\mathrm{Var}(\bar X) = \sigma^2/n$, so the standard error shrinks as $1/\sqrt{n}$ -
4× the data for half the error. CLT adds that the distribution of the mean is
approximately Gaussian regardless of the source distribution.
</details>

---

### Q15. Is a 0.5% accuracy gain real?

Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?

<details><summary>Model answer</summary>

Standard error of accuracy is $\sqrt{p(1-p)/n} \approx \sqrt{0.09 \cdot 0.91/1000}
\approx 0.9$ to $1.5\%$ per model - the 0.5% gap is within noise. To claim a real
improvement you'd need one of:

- a bigger test set
- a paired comparison (McNemar's test)
- multiple training seeds
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
