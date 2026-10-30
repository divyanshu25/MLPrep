# Rapid-Fire Quiz - DeepMind Style

40 questions. The format this simulates: an interviewer firing definitions at
you for 20–30 minutes, expecting a crisp, correct answer in one breath  - 
then occasionally drilling one answer three levels deep. So practice out loud:
read the question, answer in ≤ 2 minutes *speaking*, then unfold and compare.
An answer that is correct but takes 5 minutes fails this round.

Each answer notes the module where the full derivation lives - if the folded
answer surprises you, that module is tonight's reading, not this file.

Scoring: 35+/40 fluent → ready. Anything you hesitated on goes on a card and
comes back tomorrow.

---

## Linear algebra

**1. What are eigenvalues and eigenvectors?**
<details><summary>Answer</summary>

For square A, an eigenvector v ≠ 0 satisfies Av = λv: a direction the map
only stretches, never rotates, and λ is the stretch factor. They diagonalize
the map - in the eigenbasis, A acts as independent 1-D scalings. Repeated
application scales by λⁿ, which is why eigenvalues govern stability,
power iteration, and gradient explosion. *[Module 1; RNN connection in 3–4]*
</details>

**2. What's special about symmetric matrices' eigenstructure?**
<details><summary>Answer</summary>

Spectral theorem: a real symmetric matrix has all-real eigenvalues and an
orthonormal basis of eigenvectors, so A = QΛQᵀ. Covariance matrices and
Hessians are symmetric, which is why PCA directions are orthogonal and why
curvature decomposes into independent axes. *[Modules 1–2]*
</details>

**3. Define the SVD.**
<details><summary>Answer</summary>

Any real m×n matrix factors as A = UΣVᵀ with U, V orthogonal and Σ
non-negative diagonal (the singular values). Geometrically: every linear map
is rotate → axis-scale → rotate. It exists for *any* matrix - no symmetry or
squareness needed - which is what makes it the workhorse over
eigendecomposition. Singular values are the square roots of the eigenvalues
of AᵀA. *[Module 2, PCA section]*
</details>

**4. What is rank, and how does the SVD expose it?**
<details><summary>Answer</summary>

Rank is the dimension of the image - the number of linearly independent
columns (equivalently rows). It equals the number of nonzero singular values.
Numerically we use effective rank: singular values above a tolerance, since
float noise makes exact zeros rare. *[Module 2]*
</details>

**5. Best low-rank approximation - and the LoRA connection?**
<details><summary>Answer</summary>

Eckart–Young: the best rank-k approximation of A (in Frobenius or spectral
norm) is the truncated SVD - keep the top k singular triples; the error is the
discarded singular values. LoRA bets that a fine-tuning *update* is
approximately low-rank: it learns ΔW = BA with B: d×r, A: r×d, r ≪ d, cutting
trainable parameters from d² to 2dr while W stays frozen. *[Module 2; module 4
notes touch the parameterization]*
</details>

**6. What does positive semi-definite mean, and where does it show up?**
<details><summary>Answer</summary>

A symmetric matrix with xᵀAx ≥ 0 for all x - equivalently, all eigenvalues
≥ 0. Covariance matrices are always PSD (xᵀΣx is a variance), and the Hessian
at a local minimum is PSD (curvature can't be negative in any direction).
PSD is what makes quadratic forms bowls rather than saddles. *[Modules 1–2]*
</details>

**7. What is a condition number and why should an ML person care?**
<details><summary>Answer</summary>

κ = σ_max/σ_min: the ratio of the most- to least-stretched directions. For
optimization it measures valley elongation - gradient descent needs on the
order of κ iterations on a quadratic because the stable learning rate is set
by the steep direction while progress is set by the shallow one. Feature
normalization and momentum/Adam are all condition-number medicine.
*[Modules 2–3]*
</details>

---

## Calculus & autodiff

**8. Gradient vs Jacobian?**
<details><summary>Answer</summary>

For f: ℝⁿ→ℝ the gradient is the n-vector of partials - the steepest-ascent
direction. For f: ℝⁿ→ℝᵐ the Jacobian is the m×n matrix of all partials - the
best local linear approximation. The gradient is the m = 1 special case
(Jacobian's single row, transposed). *[Module 1]*
</details>

**9. State the chain rule for compositions of vector functions.**
<details><summary>Answer</summary>

The Jacobian of a composition is the product of the Jacobians:
J_{f∘g}(x) = J_f(g(x)) · J_g(x). All of backprop is this one fact plus a
choice of multiplication *order*. *[Modules 1, 3]*
</details>

**10. What does forward-mode autodiff compute, and at what cost?**
<details><summary>Answer</summary>

It pushes a tangent alongside the values: one pass computes a Jacobian-vector
product Jv - the directional derivative - at roughly the cost of one function
evaluation, with no storage of intermediates. Getting the full Jacobian of
f: ℝⁿ→ℝᵐ takes n forward passes (one per input direction, a column each).
*[Module 1]*
</details>

**11. Reverse mode - and why does deep learning use it?**
<details><summary>Answer</summary>

It runs the function forward, stores intermediates, then pulls a cotangent
backward, computing a vector-Jacobian product vᵀJ - a *row* of the Jacobian
per pass. A loss has one output and millions of inputs, so one reverse pass
yields the entire gradient; forward mode would need one pass per parameter.
The price is memory: activations must be kept (or recomputed - checkpointing).
*[Modules 1, 3; `micrograd_plus.ipynb` is reverse mode built by hand]*
</details>

**12. When does forward mode win?**
<details><summary>Answer</summary>

When outputs outnumber inputs (m ≫ n) - few parameters, many outputs - or
when you only need directional derivatives: JVPs, and Hessian-vector products
via forward-over-reverse. It also wins on memory, since nothing is stored.
Rule of thumb: cost scales with n forward, with m reverse. *[Module 1]*
</details>

**13. How do you check a gradient numerically?**
<details><summary>Answer</summary>

Central differences: (f(x+h) − f(x−h)) / 2h per coordinate, h ≈ 1e-5, compare
by *relative* error. Central is O(h²)-accurate vs O(h) one-sided. Pitfalls:
too-small h drowns in float cancellation, too-large h in truncation; use
float64 and perturb one coordinate at a time. *[Module 1,
`gradients_from_scratch.ipynb`]*
</details>

---

## Probability

**14. State Bayes' rule and name the pieces.**
<details><summary>Answer</summary>

P(H|D) = P(D|H) P(H) / P(D): posterior = likelihood × prior / evidence, where
the evidence P(D) = Σ_h P(D|h)P(h) normalizes over hypotheses. It's just the
two factorizations of P(H, D) set equal. The classic trap it fixes: confusing
P(D|H) with P(H|D) - a 99%-accurate test on a rare disease still yields mostly
false positives. *[Module 1]*
</details>

**15. Likelihood vs probability?**
<details><summary>Answer</summary>

Same expression P(data | θ), two readings: fixing θ and varying data gives a
probability distribution (sums to 1 over data); fixing the observed data and
varying θ gives the likelihood function of θ - which is *not* a distribution
over θ and needn't integrate to 1. MLE maximizes the second reading.
*[Module 1]*
</details>

**16. MLE vs MAP - and the regularization connection?**
<details><summary>Answer</summary>

MLE maximizes P(data|θ); MAP maximizes P(data|θ)P(θ) - likelihood times
prior. In log space the prior becomes an additive penalty: a Gaussian prior
on weights is exactly L2 regularization, a Laplace prior is exactly L1. So
"regularized loss" and "MAP with a prior" are the same computation.
*[Modules 1–2]*
</details>

**17. Central limit theorem in one breath?**
<details><summary>Answer</summary>

The mean of n i.i.d. samples with finite variance converges in distribution
to a Gaussian: √n (x̄ − μ) → N(0, σ²), regardless of the underlying
distribution's shape. Practically: averages have √n-shrinking, bell-shaped
error - the basis of confidence intervals on test metrics and of why batch
gradient noise is roughly Gaussian. *[Module 1]*
</details>

**18. Relate entropy, cross-entropy, and KL divergence.**
<details><summary>Answer</summary>

H(p) = −Σ p log p is the optimal average code length for p; cross-entropy
H(p, q) = −Σ p log q is the cost of coding p while believing q; and
H(p, q) = H(p) + KL(p‖q), so KL is the excess - always ≥ 0, zero iff p = q.
Minimizing cross-entropy in q therefore *is* minimizing KL(p‖q), since H(p)
is constant. *[Module 1]*
</details>

**19. Variance of a sum?**
<details><summary>Answer</summary>

Var(X+Y) = Var(X) + Var(Y) + 2Cov(X, Y); the covariance term vanishes under
independence. This additivity is the engine behind CLT scaling, variance
tracking through network layers for initialization, and why ensembles of
*decorrelated* models cut variance while correlated ones don't.
*[Modules 1, 3, 5]*
</details>

---

## Numerics & losses

**20. What is the log-sum-exp trick?**
<details><summary>Answer</summary>

log Σ exp(zᵢ) = m + log Σ exp(zᵢ − m) with m = max z - an exact identity,
since shifting by m factors out eᵐ. After the shift the largest exponent is
exactly 0, so nothing overflows and at least one term is 1, so the log never
sees 0. It's the safe way to normalize log-probabilities. *[Module 1]*
</details>

**21. How do you compute softmax stably?**
<details><summary>Answer</summary>

Subtract the row max before exponentiating: softmax is shift-invariant, so
the output is unchanged, but exp now maps into (0, 1] instead of overflowing
at z ≈ 89 in float32. For log-softmax, fuse it - z − max − logΣexp - rather
than taking log of a softmax that may have underflowed to exact zero.
*[Modules 1, 3; drill 5]*
</details>

**22. Why cross-entropy and not MSE for classification?**
<details><summary>Answer</summary>

Two reasons. Statistically, CE is the negative log-likelihood of a categorical
model - minimizing it is MLE; MSE corresponds to a Gaussian likelihood, the
wrong noise model for labels. Optimization-wise, softmax+CE gives gradient
p − y, which stays large when the model is confidently wrong, while MSE
through a sigmoid/softmax multiplies in p(1−p) and vanishes exactly where you
most need signal. *[Modules 1, 3]*
</details>

**23. What should the loss be at initialization for a V-way classifier?**
<details><summary>Answer</summary>

ln V - a fresh model should be uniform, and −log(1/V) = ln V (e.g. 3.61 for
V = 37, ~10.8 for a 50k vocab). Much higher means your logits are too large
at init (scaling bug); much lower at step 0 means the target is leaking into
the input. It's the single cheapest sanity check in deep learning.
*[Module 3; used throughout module 8]*
</details>

**24. float16 vs bfloat16 - why does fp16 need loss scaling?**
<details><summary>Answer</summary>

fp16 has a 5-bit exponent: its smallest normal value is ~6e-5, and subnormals
reach down to ~6e-8 with fewer and fewer bits of precision, so small gradients
lose precision or flush to zero - loss scaling multiplies the loss by ~2¹⁰ to
push gradients into range, then unscales before the update. bf16 keeps
float32's 8-bit exponent (same range, less precision), so it trains without
scaling, which is why it's the modern default. *[Module 7, mixed-precision memory arithmetic]*
</details>

---

## Optimization & training

**25. What causes vanishing gradients, and what fixed them?**
<details><summary>Answer</summary>

Backprop multiplies one Jacobian per layer; if their typical scale is below 1 - saturating activations, bad init - the product shrinks exponentially with
depth, so early layers stop learning. Fixes attack the multiplication itself:
residual connections (gradient flows through the addition untouched),
ReLU-family activations, variance-preserving init, and normalization layers.
*[Module 3; RNN case in module 4]*
</details>

**26. And exploding gradients?**
<details><summary>Answer</summary>

Same product with scale above 1 - exponential growth, common in RNNs where
the same weight matrix multiplies in every timestep, so its top singular
value compounds. Standard treatment: global-norm gradient clipping (rescale
when ‖g‖ exceeds a threshold), plus careful init; spikes in grad-norm plots
are the telltale. *[Modules 3–4; `src/gradient_explosion.py`]*
</details>

**27. BatchNorm vs LayerNorm?**
<details><summary>Answer</summary>

BatchNorm normalizes each channel over the *batch* dimension; LayerNorm
normalizes each example over its *feature* dimension. BN therefore couples
examples, breaks at small batch sizes, and needs running statistics at
inference; LN is per-example, identical train/test, and indifferent to batch
size - which is why transformers use LN (and BN survives mainly in convnets).
*[Module 3]*
</details>

**28. Adam in one breath?**
<details><summary>Answer</summary>

Keep exponential moving averages of the gradient (m, momentum) and its square
(v, per-parameter scale); update by m̂/(√v̂ + ε) after bias-correcting both for
their zero init. Effect: each parameter gets a step size normalized by its own
gradient scale - approximately scale-invariant, hence robust default learning
rates. AdamW decouples weight decay from the adaptive scaling. *[Module 3]*
</details>

**29. L1 vs L2 regularization one-liner?**
<details><summary>Answer</summary>

L2 shrinks all weights proportionally (gradient λw - a Gaussian prior);
L1 applies a constant-magnitude pull (gradient λ·sign(w) - a Laplace prior)
that drives small weights to *exactly* zero, giving sparsity. Geometrically
the L1 ball's corners sit on the axes, so the constrained optimum lands on
coordinates being zero. *[Module 2]*
</details>

**30. Bias–variance tradeoff one-liner?**
<details><summary>Answer</summary>

Expected test error decomposes as bias² + variance + irreducible noise: bias
is being systematically wrong (model too simple), variance is being
fit-to-the-sample wrong (model too flexible, answers change with the training
draw). Capacity, regularization, and data size move you along the tradeoff;
more data cuts variance for free. *[Module 5, full derivation]*
</details>

**31. What is dropout and what happens at inference?**
<details><summary>Answer</summary>

During training, zero each unit with probability p and scale survivors by
1/(1−p) (inverted dropout); at inference, do nothing - the scaling already
matched expectations. It prevents co-adaptation and acts like training an
implicit ensemble of subnetworks. Forgetting `model.eval()` in PyTorch leaves
it on and quietly degrades eval metrics. *[Module 3]*
</details>

**32. Why not initialize all weights to zero - and what's the right scale?**
<details><summary>Answer</summary>

Zero (or any constant) init makes units in a layer compute identical outputs
and receive identical gradients - symmetry never breaks and the layer
collapses to one effective unit. The right scale keeps activation variance
constant across layers: Var(w) = 1/fan_in (LeCun; Xavier's 2/(fan_in + fan_out)
balances the forward and backward passes, for tanh) or 2/fan_in (Kaiming, for
ReLU, which halves variance). Too big saturates; too small
vanishes. *[Module 3, derived by variance tracking]*
</details>

---

## Transformers & LLMs

**33. Why divide attention scores by √d_k?**
<details><summary>Answer</summary>

A dot product of two d_k-dimensional vectors with unit-variance entries has
variance d_k, so raw scores grow with head size; dividing by √d_k restores
unit variance. Without it, softmax inputs are large, the distribution
saturates to near-one-hot, and gradients through softmax vanish at init.
Same variance-tracking logic as Xavier init. *[Module 4, derived]*
</details>

**34. Why multiple attention heads instead of one big one?**
<details><summary>Answer</summary>

Each head gets its own learned projections into a d/h-dimensional subspace,
so different heads can attend by different criteria simultaneously (syntax
here, an earlier entity there) - h independent attention patterns instead of
one averaged compromise, at the same total cost. W_o then mixes the
concatenated heads back into a common space. *[Module 4]*
</details>

**35. KV cache in one line?**
<details><summary>Answer</summary>

During autoregressive decoding, past tokens' keys and values never change
(causality), so cache them and compute Q/K/V only for the new token  - 
per-step attention cost drops from O(T²) to O(T). The cost is memory,
2·layers·T·d per sequence, which is what actually caps serving batch size.
*[Module 8, worked implementation in buggy_transformer_solutions.md]*
</details>

**36. Why is attention O(n²), and name a mitigation.**
<details><summary>Answer</summary>

Every query scores every key: an n×n score matrix per head, quadratic in
sequence length in both compute and (naively) memory. Mitigations attack
different terms: FlashAttention keeps O(n²) compute but avoids materializing
the matrix (tiled, IO-aware - exact); sliding-window/sparse attention and
linear-attention variants cut compute to O(n·w) or O(n) by restricting or
approximating who attends to whom. *[Modules 4, 7]*
</details>

**37. What exactly does the causal mask do?**
<details><summary>Answer</summary>

It sets scores for j > i to −∞ *before* softmax, so position i's attention
distribution puts zero weight on the future - enforcing the autoregressive
factorization p(x) = Π p(x_t|x_{<t}). This is what lets one forward pass
train all T positions in parallel, each seeing only its prefix. Applied after
softmax it's wrong: rows no longer sum to 1. *[Module 4; bug 2 of the
flagship drill]*
</details>

**38. What does sampling temperature do?**
<details><summary>Answer</summary>

Divides logits by T before softmax. T < 1 sharpens the distribution toward
the argmax (T→0 is greedy); T > 1 flattens it toward uniform, raising
diversity and error rate together. It rescales log-probability *ratios*, so
relative preferences are preserved, only their contrast changes.
*[Module 4; drill 3]*
</details>

**39. PPO vs DPO in one line each - in the RLHF context?**
<details><summary>Answer</summary>

RLHF pipeline: supervised fine-tune, fit a reward model on human preference
pairs, then optimize the policy against it. PPO does that last step with
reinforcement learning - sample generations, score with the reward model,
take clipped policy-gradient steps under a KL leash to the reference model.
DPO skips the reward model and RL entirely: a change of variables turns the
same preference objective into a direct classification-style loss on
preference pairs - simpler and more stable, at the cost of being offline.
*[Module 7 derives DPO; the one-liners are what's expected]*
</details>

**40. What is perplexity?**
<details><summary>Answer</summary>

exp of the average per-token cross-entropy: PPL = exp(−(1/N) Σ log p(x_t |
x_{<t})). Interpretation: the model is as uncertain as if choosing uniformly
among PPL tokens each step - a uniform model over V tokens has PPL = V.
Lower is better, and it's only comparable across models sharing a tokenizer,
since it's per-*token*. *[Modules 1, 4]*
</details>
