# Rapid-Recall Flashcards

~160 cards distilled from the interview-question banks at the end of each
`ml/` module's notes plus the invariants in `dsa/` - the answers you must
produce in ≤15 seconds, because in a live round anything slower reads as
not-known. Protocol: answer **out loud** before unfolding; a hesitation counts
as a miss. Track deck passes in the queue (`cards:<section>` items). The same
deck in plain text: [`anki.tsv`](anki.tsv) (File → Import in Anki, tab-separated).

---

## 1. Foundations (ml/01)

**Q:** Why does the dot product measure similarity?
<details><summary>A</summary>

a·b = |a||b|cos θ - maximal when aligned, zero when orthogonal, negative when opposed; normalized it's cosine similarity.
</details>

**Q:** What happens if you stack linear layers with no nonlinearity?
<details><summary>A</summary>

W2(W1x) = (W2W1)x - composition of linear maps is one linear map; depth adds no expressive power.
</details>

**Q:** Eigenvectors in one sentence, and one ML place they appear?
<details><summary>A</summary>

Directions a matrix scales without rotating (Av = λv); top eigenvectors of the covariance matrix are the PCA axes.
</details>

**Q:** Eigendecomposition vs SVD?
<details><summary>A</summary>

Eigen needs square (orthogonal eigenvectors only if symmetric); SVD exists for any rectangular matrix - rotate–stretch–rotate; truncated SVD = best low-rank approximation.
</details>

**Q:** Why does L1 give sparsity but L2 doesn't?
<details><summary>A</summary>

L1 ball has corners on the axes, so optima land at exact zeros; L1's pull toward 0 is constant-magnitude while L2's fades ∝ w near 0.
</details>

**Q:** Backprop in two sentences?
<details><summary>A</summary>

Forward computes and caches intermediates; backward walks the graph in reverse, multiplying upstream gradient by each local derivative, summing where paths merge. One backward sweep gets all parameter gradients for ~one forward pass's cost.
</details>

**Q:** What is a Jacobian and its role in backprop?
<details><summary>A</summary>

The m×n matrix of partials ∂fi/∂xj - the local linear map; backprop propagates vector–Jacobian products without materializing it.
</details>

**Q:** Why do gradients vanish or explode in deep nets?
<details><summary>A</summary>

Backward signal is a product of per-layer Jacobians; singular values consistently below/above 1 make it decay/grow exponentially with depth.
</details>

**Q:** Gradient of ||Xw − y||², and what does setting it to 0 give?
<details><summary>A</summary>

2Xᵀ(Xw − y); zero gives the normal equations XᵀXw = Xᵀy - closed-form least squares.
</details>

**Q:** Derive Bayes rule in one line.
<details><summary>A</summary>

Joint factors both ways: P(A|B)P(B) = P(B|A)P(A); divide by P(B).
</details>

**Q:** Why is minimizing cross-entropy the same as maximum likelihood?
<details><summary>A</summary>

NLL of a probabilistic classifier with one-hot labels is exactly the cross-entropy loss. (MSE = MLE under Gaussian noise; L1 under Laplace.)
</details>

**Q:** 99%-accurate test, 1% prevalence, you test positive - roughly P(sick)?
<details><summary>A</summary>

~17%, not 99%: false positives from the healthy 99% swamp the rare true positives (base-rate neglect).
</details>

**Q:** Why divide by n−1 in sample variance?
<details><summary>A</summary>

Deviations are from the sample mean, which chases the data - /n underestimates σ² by (n−1)/n; Bessel's correction unbiases it.
</details>

**Q:** Why does averaging reduce noise, and at what rate?
<details><summary>A</summary>

Var(mean) = σ²/n, so standard error shrinks as 1/√n - 4× the data for half the error; CLT makes the mean ~Gaussian.
</details>

**Q:** Accuracy 91.0% → 91.5% on 1,000 test examples. Real?
<details><summary>A</summary>

No - SE ≈ √(p(1−p)/n) ≈ 1%, so 0.5% is within noise; need a bigger test set, paired test, or multiple seeds.
</details>

---

## 2. Classical ML (ml/02)

**Q:** Why does least squares correspond to Gaussian noise?
<details><summary>A</summary>

NLL of y = wᵀx + ε, ε∼N(0,σ²) is const + Σ(y−wᵀx)²/2σ² - minimizing NLL is minimizing squared error.
</details>

**Q:** Normal equation vs gradient descent - when each?
<details><summary>A</summary>

Normal equation: small/medium d (O(nd² + d³)), exact, no LR. GD/SGD: large n or d, streaming, or no closed form.
</details>

**Q:** Gradient of logistic loss?
<details><summary>A</summary>

(p − y)x - the sigmoid's p(1−p) cancels against the loss's derivative; no vanishing gradient from the sigmoid.
</details>

**Q:** Bayesian interpretation of ridge and lasso?
<details><summary>A</summary>

MAP with Gaussian prior (ridge) or Laplace prior (lasso); regularizer = −log p(w).
</details>

**Q:** What is C in an SVM?
<details><summary>A</summary>

Penalty on margin violations - inverse regularization (λ ~ 1/C). Large C: narrow margin, overfit risk; small C: wide margin, more bias.
</details>

**Q:** What does the kernel trick buy you?
<details><summary>A</summary>

The dual touches data only via inner products; K(x,z) = ⟨φ(x),φ(z)⟩ runs the linear algorithm in a huge feature space without computing φ. Price: O(n²) Gram matrix.
</details>

**Q:** Gini vs entropy - does it matter?
<details><summary>A</summary>

Rarely - both concave impurity measures picking nearly identical splits; Gini skips the log, so it's the cheap default.
</details>

**Q:** Why do random forests subsample features on top of bagging?
<details><summary>A</summary>

Var(mean) = ρσ² + (1−ρ)σ²/B - trees kill the second term but the correlation floor ρσ² remains; feature subsampling decorrelates trees, lowering ρ.
</details>

**Q:** Bagging vs boosting in one line each?
<details><summary>A</summary>

Bagging: parallel deep trees averaged to cut variance; more trees never overfit. Boosting: sequential shallow trees fit to pseudo-residuals to cut bias; can overfit - early-stop.
</details>

**Q:** In what sense is gradient boosting "gradient descent"?
<details><summary>A</summary>

Descent in function space: predictions are the parameters, pseudo-residuals are the negative gradient, each weak learner approximates that vector, F ← F + ηh is the step.
</details>

**Q:** Why does k-means always converge, and to what?
<details><summary>A</summary>

Each phase exactly minimizes J over one block (assignments, then means), so J is non-increasing and bounded - finite convergence to a local optimum; hence k-means++/restarts.
</details>

**Q:** Why standardize for kNN/k-means/SVM/regularized models but not trees?
<details><summary>A</summary>

Those depend on Euclidean distance or penalized weight magnitude - scale-sensitive; trees only threshold one feature at a time, invariant to monotone transforms.
</details>

**Q:** Curse of dimensionality with one number?
<details><summary>A</summary>

Capturing 1% of uniform data in [0,1]¹⁰⁰ needs a cube of side 0.01^(1/100) ≈ 0.955 - "local" spans 95% of every axis; distances concentrate.
</details>

**Q:** PCA: variance maximization or reconstruction minimization?
<details><summary>A</summary>

Both - provably identical: ||x − UUᵀx||² = ||x||² − ||Uᵀx||²; solution = top covariance eigenvectors; lost energy = discarded eigenvalues.
</details>

**Q:** Why does logistic regression fail on perfectly separable data?
<details><summary>A</summary>

MLE doesn't exist - scaling w up forever keeps increasing likelihood, weights diverge; regularization or early stopping restores a finite solution.
</details>

**Q:** Hinge vs logistic loss?
<details><summary>A</summary>

Both convex upper bounds on 0–1 loss. Hinge: exactly zero past margin 1, only support vectors matter, no probabilities. Logistic: smooth, never zero, calibrated probabilities.
</details>

**Q:** Where does the 37% out-of-bag figure come from?
<details><summary>A</summary>

P(point missed by all n bootstrap draws) = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368 - free held-out error estimate per tree.
</details>

**Q:** Why can't you run plain SVD on a ratings matrix?
<details><summary>A</summary>

It's ~99% missing; SVD optimizes error over ALL entries, so zeros-as-missing fits "everyone dislikes everything". Sum error over observed entries only + regularization → SGD or ALS.
</details>

---

## 3. Neural networks & training (ml/03)

**Q:** Why do neural nets need nonlinear activations?
<details><summary>A</summary>

Composition of linear maps is linear - without nonlinearities any depth collapses to one linear layer; can't represent XOR or curved boundaries.
</details>

**Q:** Why is backprop efficient vs per-parameter finite differences?
<details><summary>A</summary>

Finite differences: one forward pass per parameter. Backprop: the loss is scalar, so one reverse sweep gets all gradients - dynamic programming over the graph, ~2–3 forward passes total.
</details>

**Q:** Derive the softmax + cross-entropy gradient.
<details><summary>A</summary>

L = −z_t + log Σe^{z_j}, so ∂L/∂z_k = p_k − y_k. The softmax Jacobian's mess cancels against −1/p_t from the log.
</details>

**Q:** Why not MSE for classification?
<details><summary>A</summary>

MSE keeps a σ′-type factor (p(1−p)) that → 0 when confidently wrong - tiny gradient exactly when you need the biggest correction; CE cancels it, giving p − y.
</details>

**Q:** Sigmoid's maximum derivative, and the consequence for depth?
<details><summary>A</summary>

0.25 (at s = ½) - d sigmoid layers scale gradients by ≤ 0.25^d: vanishing gradients; hence ReLU, good init, normalization, residuals.
</details>

**Q:** What is a dead ReLU and how do you prevent it?
<details><summary>A</summary>

Pre-activation negative for every input → output 0, gradient 0, never recovers (usually a huge update knocked the bias negative). Prevent: sane init, moderate LR/warmup, LeakyReLU/GELU.
</details>

**Q:** Derive Xavier and Kaiming init.
<details><summary>A</summary>

Var(z) = n_in σ_w² Var(x); keep the factor 1 per layer → σ_w² = 1/n_in (Xavier: 2/(n_in+n_out)); ReLU halves the second moment → He's σ_w² = 2/n_in.
</details>

**Q:** Why not initialize with zeros?
<details><summary>A</summary>

Every neuron computes the same output and gets the same gradient - symmetry never breaks; the layer acts as one neuron. Random init breaks it; Xavier/Kaiming set its scale.
</details>

**Q:** Adam in one line, and why bias correction?
<details><summary>A</summary>

EMA of gradients (m) and squared gradients (v), step η·m̂/(√v̂+ε). EMAs start at 0 so early estimates are low; dividing by 1−βᵗ unbiases - without it, step 1 is ~30× too large via v.
</details>

**Q:** Weight decay vs L2 - same thing?
<details><summary>A</summary>

Under plain SGD yes; under Adam no - L2's λθ gets divided by √v̂, under-regularizing high-gradient params. AdamW decouples the decay; transformer default.
</details>

**Q:** What does BatchNorm do and why does it help?
<details><summary>A</summary>

Standardize per feature with batch stats, then learned γ,β. Best-supported story: smooths the loss landscape → larger LRs, less init sensitivity; plus mild noise regularization.
</details>

**Q:** Why does BatchNorm differ train vs test?
<details><summary>A</summary>

Inference can't depend on batch-mates (or batch size 1), so it uses running EMA stats - a fixed affine map. Forgetting `model.eval()` is the classic silent bug.
</details>

**Q:** Why LayerNorm instead of BatchNorm in transformers?
<details><summary>A</summary>

LN normalizes across features within one example - no batch dependence: handles variable-length/padded sequences, tiny sharded batches, and batchless autoregressive decoding; identical train/test.
</details>

**Q:** How does dropout work, and why scale by 1/(1−p)?
<details><summary>A</summary>

Zero each unit w.p. p in training, scale survivors by 1/(1−p) so the expected input is unchanged - inference becomes an exact no-op. Prevents co-adaptation; approximates a 2ⁿ subnetwork ensemble.
</details>

**Q:** Why do residual connections help deep training?
<details><summary>A</summary>

Block Jacobian is I + ∂F/∂h - gradient reaches early layers with factor exactly 1 along the skip path; forward view: start at identity, learn small corrections.
</details>

**Q:** Training loss is NaN after a few hundred steps - first suspects?
<details><summary>A</summary>

(1) LR too high; (2) unstable log(softmax) - use log-sum-exp; (3) exploding gradients - clip, check init; (4) bad data (NaN/inf); (5) missing ε in a division. Bisect by logging ‖g‖ per layer.
</details>

**Q:** Why minibatches rather than full-batch?
<details><summary>A</summary>

Gradient SE shrinks 1/√B while cost grows ∝ B - past modest B, compute buys more steps, not better steps; batches saturate hardware; gradient noise escapes saddles and finds flatter minima.
</details>

**Q:** How do you verify a hand-written backward pass?
<details><summary>A</summary>

Centered finite differences (h ≈ 1e−5, O(h²) error) vs analytic gradient, relative error ≲ 1e−6 in float64, on a tiny model/batch.
</details>

---

## 4. Deep learning & transformers (ml/04)

**Q:** Why do CNNs use so many fewer parameters than MLPs on images?
<details><summary>A</summary>

Weight sharing + locality: conv params C_out(C_in k² + 1) - independent of image size; the architecture hard-codes translation equivariance the MLP would have to learn from data.
</details>

**Q:** Conv output size and parameter count?
<details><summary>A</summary>

H_out = ⌊(H + 2p − k)/s⌋ + 1; params = C_out(C_in k² + 1).
</details>

**Q:** Why did ResNet skips enable 150+ layers?
<details><summary>A</summary>

Plain deep nets underfit (optimization failure - stacked layers can't represent identity); y = x + F(x) makes identity the default and puts I in every Jacobian.
</details>

**Q:** Derive why RNN gradients vanish/explode.
<details><summary>A</summary>

∂h_t/∂h_{t−1} = diag(tanh′)W_hh; over n steps the norm behaves like (γσ_max(W))ⁿ - geometric decay below 1, growth above; nothing pins it at 1.
</details>

**Q:** How does the LSTM fix vanishing gradients?
<details><summary>A</summary>

Additive cell state: c_t = f⊙c_{t−1} + i⊙c̃, so ∂c_t/∂c_{t−1} ≈ diag(f) - no repeated W, no squashing; forget gate ≈ 1 is a data-dependent gradient highway. Doesn't fix exploding (still clip).
</details>

**Q:** Write down attention and gloss each part.
<details><summary>A</summary>

softmax(QKᵀ/√d_k)V - queries (what a token seeks) dot keys (what tokens advertise), scale, softmax to a distribution, convex combination of values. A soft differentiable dictionary lookup.
</details>

**Q:** Why divide by √d_k?
<details><summary>A</summary>

qᵀk with unit-variance entries has variance d_k (std 8 at d_k=64) - softmax saturates one-hot, Jacobian → 0, gradients vanish at init; scaling restores unit variance.
</details>

**Q:** Why does attention need positional encodings?
<details><summary>A</summary>

Pure content-based dot products are permutation equivariant - "dog bites man" ≡ "man bites dog"; RNNs get order from sequence, CNNs from kernel layout. Fix: absolute embeddings or RoPE/relative bias.
</details>

**Q:** What do multiple heads buy over one big head?
<details><summary>A</summary>

h independent attention distributions per token, each in its own learned subspace (dims split, so same compute) - one head can track position while another tracks coreference.
</details>

**Q:** Explain the KV cache. Why doesn't it help training?
<details><summary>A</summary>

Causality means past K/V never change - cache them; each new token computes only its own q,k,v: O(td) per layer instead of O(t²d). Training computes all positions in one parallel pass and weights change every step - nothing to reuse.
</details>

**Q:** Complexity of self-attention?
<details><summary>A</summary>

O(T²d) compute, O(T²) attention-matrix memory per head (FlashAttention removes the memory, not the FLOPs); MLP adds O(Td²); quadratic dominates once T ≳ d.
</details>

**Q:** Pre-LN vs post-LN - why does it matter?
<details><summary>A</summary>

Post-LN normalizes the residual path - no identity path survives depth, needs warmup, unstable. Pre-LN (x + f(LN(x))) keeps the stream addition-only, I in every Jacobian - stable at depth.
</details>

**Q:** What does perplexity 20 mean?
<details><summary>A</summary>

PPL = e^(mean CE in nats): the model is as uncertain as a uniform choice among 20 tokens - effective branching factor. Only comparable under the same tokenizer.
</details>

**Q:** Temperature vs top-k vs top-p?
<details><summary>A</summary>

Temperature rescales logits (reshapes the whole distribution); top-k keeps the k most likely (fixed nucleus); top-p keeps the smallest set with mass ≥ p (adaptive). Compose: temperature then top-p.
</details>

**Q:** What is BPE and why not characters or words?
<details><summary>A</summary>

Iteratively merge the most frequent adjacent pair from bytes up to a vocab budget. Characters: long sequences (quadratic pain); words: huge vocab + OOV. BPE interpolates, byte fallback means nothing is OOV.
</details>

**Q:** Why has RoPE replaced learned absolute positions?
<details><summary>A</summary>

Rotating q,k by position-proportional angles makes the score depend exactly on the gap n−m by rotation algebra - relative structure shared across positions, defined at any length, extrapolates better.
</details>

**Q:** Why grouped-query attention?
<details><summary>A</summary>

KV-cache size scales with KV heads; sharing one K,V per group of query heads cuts cache h/g× (4× at 32 heads, g=8) while keeping all h query subspaces - faster bandwidth-bound decode, bigger batches.
</details>

**Q:** An "8×7B" MoE - what runs per token and why bother?
<details><summary>A</summary>

Router picks top-2 of 8 expert MLPs: own ~47B params, spend ~13B of FLOPs. Loss scales with params, cost with FLOPs - conditional compute buys params without per-token compute. Costs: all params in memory, routing comms, load-balancing loss.
</details>

---

## 5. Evaluation & metrics (ml/05)

**Q:** Precision and recall - define, and one case where each is priority.
<details><summary>A</summary>

Precision = TP/(TP+FP): of the flagged, how many real - spam filtering. Recall = TP/(TP+FN): of the real, how many flagged - cancer screening.
</details>

**Q:** Why is F1 the harmonic mean, not arithmetic?
<details><summary>A</summary>

Harmonic mean is dominated by the smaller value - predict-all-positive at 1% prevalence gets P=.01, R=1: arithmetic says 0.505, harmonic says ≈0.02 (correctly terrible).
</details>

**Q:** What exactly does AUC = 0.9 mean?
<details><summary>A</summary>

A random positive is scored above a random negative 90% of the time - a ranking statement, independent of threshold, prevalence, and calibration.
</details>

**Q:** Fraud at 1:1000, AUC 0.98 - done?
<details><summary>A</summary>

No - ROC axes are prevalence-free. With TPR=.9, FPR=.01, π=.001, precision ≈ 8%: most alerts false. Report PR curve / precision at required recall.
</details>

**Q:** Can AUC be 1.0 with terrible calibration?
<details><summary>A</summary>

Yes - 0.51 for every positive, 0.49 for every negative: perfect ranking, meaningless probabilities. AUC sees order; calibration is about values.
</details>

**Q:** Temperature scaling - why doesn't it change accuracy?
<details><summary>A</summary>

Dividing logits by T > 0 is monotone - argmax and the whole ranking (hence AUC) unchanged; only probability values move.
</details>

**Q:** State the bias–variance decomposition.
<details><summary>A</summary>

E[(y − f̂)²] = bias² + variance + σ² - systematic error of the family, fit's sensitivity to the sample, and irreducible label noise.
</details>

**Q:** Does more data reduce bias or variance? Regularization?
<details><summary>A</summary>

More data: variance. Regularization: variance down, some bias added.
</details>

**Q:** Test MSE below σ² - what does it mean?
<details><summary>A</summary>

A bug or leakage - σ² is irreducible label noise, present regardless of model.
</details>

**Q:** Three forms of data leakage?
<details><summary>A</summary>

(1) Preprocessing fit on full data before splitting; (2) target leakage - feature caused by the label; (3) temporal leakage - random splits on time series. (Bonus: same user in train and test.)
</details>

**Q:** Why nested CV?
<details><summary>A</summary>

Reporting the best hyperparameter's CV score selects the max of noisy estimates - winner's curse. Inner loop selects, outer loop scores on folds never used for selection.
</details>

**Q:** A: 84.7%, B: 84.2%, same test set - ship A?
<details><summary>A</summary>

Not yet - use a paired test (McNemar / paired differences) to remove shared example-difficulty variance, and multiple seeds: seed variance alone often exceeds 0.5%.
</details>

**Q:** What does a class weight do to the gradient?
<details><summary>A</summary>

∂L/∂z = w_y(p̂ − y) - minority examples push w× harder (≈ oversampling in expectation); side effect: calibration destroyed, recalibrate or move the threshold instead.
</details>

**Q:** Threshold moving vs resampling - when each?
<details><summary>A</summary>

If probabilities are fine and only the decision is miscosted: keep natural training, set t* = c_FP/(c_FP + c_FN). Resample only when the minority is too rare to learn at all - then recalibrate.
</details>

**Q:** Prove the constant minimizing MAE is the median.
<details><summary>A</summary>

h′(c) = P(y<c) − P(y>c); zero where both are ½ - the median. (MSE's zero gradient gives the mean.)
</details>

**Q:** Offline AUC up 2%, A/B flat - name causes.
<details><summary>A</summary>

Ranking gains on pairs users never see; training/serving skew; offline label ≠ online goal; feedback loops (logs collected under the old policy). Trust the experiment, audit the proxy.
</details>

**Q:** LLM-judge biases and mitigations?
<details><summary>A</summary>

Position (judge both orders), verbosity (length-control), self-preference (other-family judge/panel); then validate the judge against a human-labeled sample before trusting deltas.
</details>

---

## 6. ML systems (ml/06)

**Q:** Why do recommenders use two stages instead of one great model?
<details><summary>A</summary>

Arithmetic: even 1µs/item over 10⁹ items is 10³ s/request. Cheap ANN retrieval cuts 10⁹ → 10³ in ~10ms; the expressive model runs only there.
</details>

**Q:** What structural constraint defines a two-tower model, and the trade?
<details><summary>A</summary>

Score must be f(u)·g(i) - no interaction before the dot product. Buys precomputed item embeddings + ANN retrieval; costs cross features (ranking stage handles those).
</details>

**Q:** Data drift vs concept drift?
<details><summary>A</summary>

Data drift: P(x) moves, P(y|x) fixed - retrain on recent data. Concept drift: P(y|x) moves (fraud tactics shift) - retraining helps only after new-concept labels arrive; need proxies + rules.
</details>

**Q:** Training/serving skew - most common cause?
<details><summary>A</summary>

The feature implemented twice (batch pipeline vs serving service) and diverging. Fix: single feature definition materialized to both stores + log served features and diff.
</details>

**Q:** What is point-in-time correctness?
<details><summary>A</summary>

Every training feature computed only from data timestamped before that example's prediction time - else label leaks in, offline inflates, online craters. Enforced by as-of joins.
</details>

**Q:** Why is accuracy wrong for fraud, and what's right?
<details><summary>A</summary>

At 0.2% fraud, always-legit is 99.8% accurate. Use precision/recall at threshold, PR-AUC, money-weighted recall, calibration (the decision is an expected-value computation).
</details>

**Q:** How do you handle position bias in click labels?
<details><summary>A</summary>

Clicks confound relevance with exposure. Position as a training feature frozen at serving; inverse-propensity weighting from randomization; or examination × relevance factorization.
</details>

**Q:** Shadow vs canary vs A/B - what does each uniquely tell you?
<details><summary>A</summary>

Shadow: latency/feature-availability at real traffic, zero outcome signal. Canary (1–5%): guardrail harms with fast rollback. A/B: the only stage that measures improvement.
</details>

**Q:** Why does bigger batch call for bigger LR?
<details><summary>A</summary>

k× batch cuts gradient variance k× and gives k× fewer updates - take proportionally larger steps (linear scaling), with warmup because the rule fails early and at extreme k.
</details>

**Q:** Batch vs online inference - when batch?
<details><summary>A</summary>

When inputs are enumerable in advance and hours of staleness is fine: precompute nightly, serve from a KV store. Online is forced when the input exists only at request time.
</details>

**Q:** Why does RAG exist instead of fine-tuning or paste-everything?
<details><summary>A</summary>

Fine-tuning: slow per update, unreliable for verbatim facts, can't cite. Paste-all: context limits, linear cost, attention degradation. Retrieval: knowledge updates = index updates, answers carry citations.
</details>

**Q:** How do you evaluate a RAG system?
<details><summary>A</summary>

Decompose: retrieval recall@k against gold passages; generation faithfulness-to-context + correctness on a golden set + abstention on unanswerables; online deflection/feedback.
</details>

**Q:** What do you monitor when labels arrive 60 days late?
<details><summary>A</summary>

Label-free layers: per-feature distributions (nulls, PSI, unseen categories) and prediction distributions, sliced by segment; fast proxy labels; true metrics by maturity cohort.
</details>

**Q:** Features stable but online performance decaying - what and why?
<details><summary>A</summary>

Concept drift signature: P(x) stable, P(y|x) moving. Confirm via calibration decay on freshest labels; respond with faster labels, recent-window retrains, rules meanwhile.
</details>

**Q:** PM asks for "just one metric" - answer?
<details><summary>A</summary>

One primary online metric + guardrails that must not regress + a counter-metric for the primary's known failure mode; a single unguarded metric gets Goodharted.
</details>

---

## 7. LLM training & post-training (ml/07)

**Q:** 10²³ FLOPs - what model do you train?
<details><summary>A</summary>

Chinchilla: C = 6ND, D ≈ 20N → C = 120N², N = √(C/120) ≈ 29B params, D ≈ 580B tokens. Caveat: heavy serving traffic → overtrain a smaller model instead.
</details>

**Q:** Derive C = 6ND.
<details><summary>A</summary>

Forward: 2 FLOPs per param per token; backward: two matmuls per forward matmul → 4N; total 6N per token × D tokens.
</details>

**Q:** Training memory for 70B with Adam, mixed precision?
<details><summary>A</summary>

16 B/param (2 fp16 weights + 2 grads + 4 fp32 master + 4 m + 4 v) → 1.12 TB of state + activations; ≥14 H100s, only with ZeRO-3/FSDP.
</details>

**Q:** Why fp32 master weights?
<details><summary>A</summary>

The update ηg (~1e−6) is below the fp16 ulp near weight magnitude - it rounds to zero; accumulate in fp32, round to fp16 for matmuls.
</details>

**Q:** ZeRO stages - what does each shard?
<details><summary>A</summary>

1: optimizer state (12B/param); 2: + gradients; 3: + weights (16Ψ/G - the only stage fitting 70B on 80 GB GPUs, at ~1.5× DP communication). FSDP ≈ ZeRO-3.
</details>

**Q:** Where do all-reduces land in tensor parallelism?
<details><summary>A</summary>

Column-split first matmul (comm-free), row-split second → one all-reduce per MLP block and per attention block, forward and backward; frequent and unhideable → TP stays inside a node's NVLink.
</details>

**Q:** Pipeline bubble fraction?
<details><summary>A</summary>

(p−1)/(m+p−1) for p stages, m microbatches - p=8, m=32 gives 18%; raise m to amortize; 1F1B caps live activations.
</details>

**Q:** How does gradient checkpointing get O(√L) memory?
<details><summary>A</summary>

Store k checkpoints, recompute L/k-layer segments in backward: peak ∝ k + L/k, minimized at k=√L; cost ≈ +33% FLOPs (one extra forward of the 6N).
</details>

**Q:** Batch 256 → 2048 - what happens to LR?
<details><summary>A</summary>

Linear scaling: 8× with warmup (fails early in training); watch the critical batch size; for Adam √8× is often better.
</details>

**Q:** Derive DPO from the RLHF objective.
<details><summary>A</summary>

KL-regularized reward max has closed form π* ∝ π_ref e^{r/β}; invert for r, substitute into Bradley–Terry - Z cancels, leaving a supervised loss on preference pairs. RM + RL collapse into one loss.
</details>

**Q:** Why the KL term in PPO-RLHF? Why the clip?
<details><summary>A</summary>

KL-to-reference: the RM is only valid near the SFT distribution - unpenalized optimization reward-hacks it. Clip: bounds each update from stale samples (trust-region-lite).
</details>

**Q:** GRPO vs PPO?
<details><summary>A</summary>

GRPO deletes the value network: sample G completions per prompt, advantage = (reward − group mean)/std - fits sparse verifiable rewards, saves an LM-sized critic.
</details>

**Q:** LoRA - why so cheap, and inference cost after training?
<details><summary>A</summary>

Freeze W, learn ΔW = (α/r)BA with r ≪ d: r(d+k) params instead of dk (~250× fewer at 4096/r=8); B=0 init starts at the pretrained model; merge W + BA → zero added latency.
</details>

**Q:** How would you detect benchmark contamination?
<details><summary>A</summary>

13-gram overlap with training corpus, canary strings, verbatim-completion probes, accuracy drop on fresh matched-difficulty items (strongest), paraphrase sensitivity.
</details>

---

## 8. LLM inference (ml/07)

**Q:** Why is single-stream decode memory-bound?
<details><summary>A</summary>

Decode is GEMV: ~1 FLOP per byte of weights read vs ridge ~156 (A100) - every token streams all weights; batching/speculation raise intensity, quantization shrinks bytes.
</details>

**Q:** Tokens/sec ceiling for 7B fp16 on 2 TB/s?
<details><summary>A</summary>

bandwidth / weight bytes = 2e12 / 14e9 ≈ 143 tok/s.
</details>

**Q:** KV-cache size formula?
<details><summary>A</summary>

2 (K and V) × n_layers × d_model × bytes/value × seq × batch. 7B, batch 32, 4k ctx, fp16: 512 KB/token → ~69 GB - 5× the weights.
</details>

**Q:** Three ways to shrink the KV cache?
<details><summary>A</summary>

GQA (÷ group factor, e.g. 8), KV quantization to int8/fp8 (÷2), PagedAttention (reclaims 60–80% fragmentation waste).
</details>

**Q:** Why does speculative decoding not change the output distribution?
<details><summary>A</summary>

Accept draft x∼q w.p. min(1, p/q), else resample from (p−q)⁺ normalized: P(x) = min(p,q) + max(0, p−q) = p(x) exactly. Acceptance = 1 − TV(p,q).
</details>

**Q:** Is FlashAttention approximate?
<details><summary>A</summary>

No - exact. IO reorganization: tile K,V through SRAM with online softmax, never materializing the T×T matrix in HBM; attention is bandwidth-bound, so ~10× less HBM traffic gives 2–4× speed and O(s) memory.
</details>

**Q:** What breaks in naive int8 quantization of LLMs?
<details><summary>A</summary>

Outlier activation channels 20–100× larger dominate the absmax scale, crushing everyone else's resolution; fix with per-channel/group scales, mixed decomposition (LLM.int8), or scale migration (AWQ).
</details>

**Q:** Why is generation slow when training is parallel?
<details><summary>A</summary>

Training teacher-forces all T positions in one masked pass; generation is sequential (token t+1 needs sampled token t) - n dependent forward passes, each bandwidth-bound.
</details>

**Q:** Why doesn't the KV cache help training?
<details><summary>A</summary>

Every K/V is used immediately and exactly once per step, and weights change every step - nothing valid to reuse.
</details>

**Q:** Judge win-rate up 10 points - what do you check?
<details><summary>A</summary>

Position/verbosity/self-preference biases, human agreement on a sample, fresh prompts, per-category breakdown, length drift - +10 win-rate with +40% length is a length gain.
</details>

**Q:** Quantization error trick that makes 4-bit GPTQ ≈ fp16?
<details><summary>A</summary>

Quantize weights one at a time and update the remaining weights to absorb each rounding error instead of letting errors accumulate.
</details>

**Q:** Why does batching fix decode's bandwidth problem?
<details><summary>A</summary>

One weight-stream serves B tokens - arithmetic intensity scales with B, moving decode toward the compute-bound regime.
</details>

---

## 9. Numbers that must be automatic

**Q:** Bytes per parameter: fp32 / fp16-bf16 / int8 / int4?
<details><summary>A</summary>

4 / 2 / 1 / 0.5.
</details>

**Q:** Training bytes per parameter, mixed-precision Adam?
<details><summary>A</summary>

16 (2+2+4+4+4); ~20 with fp32 gradient accumulation. Say "16–20 before activations."
</details>

**Q:** Inference bytes per parameter, fp16?
<details><summary>A</summary>

2 - a 7B is ~14 GB, a 70B ~140 GB.
</details>

**Q:** Training compute rule?
<details><summary>A</summary>

C = 6ND FLOPs (2 forward + 4 backward per param per token).
</details>

**Q:** Chinchilla-optimal tokens for N params?
<details><summary>A</summary>

D ≈ 20N - so C ≈ 120N².
</details>

**Q:** A100 ballparks: bf16 compute, HBM bandwidth, ridge point?
<details><summary>A</summary>

~312 TFLOP/s, ~2 TB/s, ridge ≈ 156 FLOPs/byte.
</details>

**Q:** H100 memory per GPU?
<details><summary>A</summary>

80 GB (same as A100-80GB); SRAM ~20 MB at ~19 TB/s is the FlashAttention resource.
</details>

**Q:** KV bytes per token (dense fp16 model)?
<details><summary>A</summary>

2 × n_layers × d_model × 2 bytes - 7B (32 layers, d=4096): 512 KB/token.
</details>

**Q:** ZeRO-3 per-GPU state for 70B on 64 GPUs?
<details><summary>A</summary>

16Ψ/G = 16×70e9/64 ≈ 17.5 GB - the only stage under 80 GB.
</details>

**Q:** Pipeline bubble at p=8, m=32?
<details><summary>A</summary>

(p−1)/(m+p−1) = 7/39 ≈ 18%.
</details>

**Q:** Gradient checkpointing cost?
<details><summary>A</summary>

~+33% FLOPs (one extra forward of the 6N total) for O(√L) activation memory.
</details>

**Q:** Latency: what does ~100ms of added page delay cost?
<details><summary>A</summary>

~1% of revenue/engagement (the classic Amazon/Google findings) - why the latency budget is designed before the model.
</details>

**Q:** Budget 100ms to score 500 items - sketch the split.
<details><summary>A</summary>

~15 candidate fetch + ~20 features (one batched multi-get) + ~25 model (one batched forward) + ~10 re-rank + ~25 headroom.
</details>

**Q:** LoRA trainable params for W_Q, W_V of a 7B at r=16?
<details><summary>A</summary>

2·4096·16 per matrix × 2 matrices × 32 layers ≈ 8.4M ≈ 0.12% of 7B.
</details>

**Q:** Standard error of accuracy on n test examples?
<details><summary>A</summary>

√(p(1−p)/n) - ~1% at n=1000, p≈0.9: half-point gains on 1k examples are noise.
</details>

---

## 10. DSA invariants (dsa/01–06)

**Q:** Sliding window - the invariant and the precondition?
<details><summary>A</summary>

Window [l, r] satisfies predicate P; P must be monotone under shrinking, so the smallest valid l never moves left - l only advances, O(n) amortized.
</details>

**Q:** When does sliding window break, and the fallback?
<details><summary>A</summary>

Negatives with sum predicates - shrinking no longer monotone. Fall back to prefix sums + hashmap (or monotonic deque for shortest-subarray-≥k).
</details>

**Q:** Two pointers - what licenses skipping pairs?
<details><summary>A</summary>

An elimination argument: moving l must provably kill every pair (l, x) - one comparison + monotonicity discards a whole row of the pair matrix.
</details>

**Q:** Prefix-sum identity, and the hashmap trick for "subarray sum = k"?
<details><summary>A</summary>

sum(i..j) = pre[j+1] − pre[i]; count earlier prefixes equal to pre[j+1] − k, hashmap seeded {0:1} for subarrays starting at 0. Query before insert.
</details>

**Q:** Binary search boundary invariant?
<details><summary>A</summary>

Everything < lo is F, everything ≥ hi is T; the first T lies in [lo, hi]; loop `lo < hi`, moves `hi = mid` / `lo = mid + 1`, answer is lo.
</details>

**Q:** The `lo = mid` infinite-loop rule?
<details><summary>A</summary>

The pointer assigned bare `mid` must be unreachable by the rounding: round down when `hi = mid`, round up (`+1` inside) when `lo = mid`.
</details>

**Q:** Binary search on the answer - recognition cue and the script?
<details><summary>A</summary>

"Minimize the maximum" / "smallest value such that ⟨checkable⟩". Name the bracket, prove monotonicity in one sentence, write the (usually greedy) checker, drop in the boundary template.
</details>

**Q:** Rotated sorted array - what survives rotation?
<details><summary>A</summary>

Cut anywhere and at least one side is properly sorted - a sorted half answers "is target in you?" in O(1), so discard half each step. Find-min: compare a[mid] to a[hi].
</details>

**Q:** Why does LIFO fit matching/nesting problems?
<details><summary>A</summary>

In well-nested input, the thing that must close next is the thing that opened most recently - LIFO order IS nesting order.
</details>

**Q:** Monotonic stack - invariant and the O(n) argument?
<details><summary>A</summary>

Stack holds the unanswered, monotone by construction (next-greater ⇒ decreasing); each index pushes and pops at most once, so the nested while is O(n) total.
</details>

**Q:** Floyd cycle detection - why must fast and slow meet?
<details><summary>A</summary>

Inside a cycle of length C the gap changes by exactly 1 per step (mod C) - a quantity decreasing by 1 in a finite cyclic group hits 0; they can't jump over each other.
</details>

**Q:** LRU cache - derive the data structure from O(1) requirements.
<details><summary>A</summary>

O(1) get ⇒ hashmap; O(1) recency reorder + evict-oldest ⇒ doubly linked list with sentinels; map values are the DLL nodes. Each covers the other's blind spot.
</details>

**Q:** The BST invariant, exactly?
<details><summary>A</summary>

For every node, ALL keys in the left subtree < node < ALL keys in the right subtree - whole subtrees, not just children; validate by carrying (lo, hi) bounds down.
</details>

**Q:** The one tree template?
<details><summary>A</summary>

Recurse on both subtrees, combine at the root; design questions: base case, combine, and is the answer at the root or anywhere (→ nonlocal best, return the local quantity).
</details>

**Q:** Heap: the property, the array trick, and Python's quirks?
<details><summary>A</summary>

Every node ≤ children (min at root, siblings unordered); complete tree in an array, parent (i−1)//2, children 2i+1/2i+2; heapq is min-only - negate for max, add a counter to tuples for unorderable items.
</details>

**Q:** Why does BFS with a queue find shortest paths?
<details><summary>A</summary>

FIFO processes all distance-d nodes before any d+1 - the expanding-rings order is the proof, for unweighted graphs.
</details>

**Q:** Kahn's algorithm - and how does it detect cycles for free?
<details><summary>A</summary>

Peel indegree-0 nodes; if fewer than n ever reach indegree 0, the leftovers all have a prerequisite among themselves - a cycle.
</details>

**Q:** Union-find - the two optimizations and the result?
<details><summary>A</summary>

Union by rank (height ≤ log n: a rank-r tree has ≥ 2^r nodes) + path compression → near-constant amortized find/union for incremental connectivity.
</details>

**Q:** Dijkstra - why is greedy safe, and its precondition?
<details><summary>A</summary>

Cut argument: any shorter path to the closest unsettled node would cross the settled boundary at a node with even smaller tentative distance - contradiction. Requires non-negative weights.
</details>

**Q:** The universal DP checklist?
<details><summary>A</summary>

State in words (checkable by hand) → recurrence as a decision ("what was the last move?") → base cases → evaluation order → where the answer lives. Cost = #states × transition cost.
</details>

**Q:** The 1-D knapsack compression bug?
<details><summary>A</summary>

Iterating capacity ascending reads dp entries already updated for THIS item - item taken twice (that's the unbounded variant); 0/1 needs descending capacity.
</details>

**Q:** When is greedy trustworthy?
<details><summary>A</summary>

Only with an exchange argument: any optimal solution can be swapped, one step at a time, into the greedy one without getting worse (e.g. earliest-finish interval scheduling). Guilty until proven safe.
</details>
