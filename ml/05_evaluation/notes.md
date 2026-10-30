# Evaluation: how to know if your model is any good

The theme of this note: **every metric is an answer to a specific question.** If you
don't know what question a metric answers, you don't understand the metric - you're
just pattern-matching to "bigger number good." We'll build every metric from scratch,
with tiny numeric examples first, formulas second, and derivations in full.

Companion code in this directory:

- `metrics_scratch.ipynb` - confusion matrix, precision/recall/F1, ROC/AUC (two ways), PR/AP, imbalance demo
- `bias_variance_experiment.ipynb` - empirical bias²/variance decomposition + k-fold CV
- `calibration_scratch.ipynb` - reliability tables, ECE, temperature scaling
- `../../src/roc_curve.ipynb` - companion notebook with a plotted ROC curve (this note builds the same curve by hand, in words)

---

## 1. The confusion matrix: the atom of classification evaluation

Binary classification. The model says positive or negative; reality says positive or
negative. Two binary variables → four cells. Every classification metric
you've ever heard of is some ratio of these four numbers.

|                    | Predicted + | Predicted − |
|--------------------|-------------|-------------|
| **Actually +**     | TP          | FN          |
| **Actually −**     | FP          | TN          |

Mnemonic: the second letter is what the model *said* (P/N), the first letter is
whether it was *right* (T/F). A **false negative** is a positive example the model
called negative - the model said N, and that was False.

Concrete running example. A cancer screening test on 1000 people, 10 of whom actually
have cancer:

|                    | Predicted + | Predicted − |
|--------------------|-------------|-------------|
| **Actually +** (10)   | TP = 8      | FN = 2      |
| **Actually −** (990)  | FP = 90     | TN = 900    |

Now the metrics, each answering a distinct question:

**Accuracy** - "what fraction of predictions were right?"

$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{8 + 900}{1000} = 0.908$$

Looks decent. But note: a model that predicts "no cancer" for *everyone* gets
$990/1000 = 0.99$. Higher accuracy, completely useless. First interview trap:
**accuracy is dominated by the majority class**. More in §7.

**Precision** - "when the model says positive, how often is it right?" Reads down the
Predicted-+ column.

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{8}{8 + 90} = 0.082$$

Only 8.2% of people flagged actually have cancer. This is the number a *user of the
prediction* cares about: "the test came back positive - should I panic?"

**Recall (sensitivity, true positive rate)** - "of the actual positives, what fraction
did we catch?" Reads across the Actually-+ row.

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{8}{8 + 2} = 0.8$$

We caught 8 of 10 cancers, missed 2. This is the number the *world* cares about:
what slipped through.

**Specificity (true negative rate)** - recall for the negative class: "of the actual
negatives, what fraction did we correctly clear?"

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{900}{990} = 0.909$$

Its complement, $FPR = FP/(FP+TN) = 1 - \text{specificity} = 0.091$, is the x-axis of the
ROC curve (§2).

### Precision vs recall: who pays for which error?

- **Cancer screening**: a FN is a missed cancer (someone dies); a FP is an unnecessary
  follow-up biopsy (unpleasant, costly, survivable). You want **high recall** and will
  tolerate mediocre precision - that's what screening *is*: a cheap high-recall filter
  before an expensive high-precision test.
- **Spam filter**: a FP is a real email (your job offer) buried in spam; a FN is one
  spam message in your inbox (annoying). You want **high precision** on the spam class.
- **Fraud detection**: FN = stolen money, FP = a blocked legitimate customer who churns.
  Both cost real dollars; you need the actual cost ratio to pick an operating point.

Interview trap: there is no "precision is better than recall." The only correct answer
is "which error is more expensive here?"

### F1: why the harmonic mean

We often want one number combining precision and recall. Why not the arithmetic mean?
Because the arithmetic mean rewards degenerate classifiers. Concrete example:

Take a dataset with 1000 examples, 10 positive. Classifier A predicts *everything*
positive:

- Precision $= 10/1000 = 0.01$, Recall $= 10/10 = 1.0$
- Arithmetic mean $= (0.01 + 1.0)/2 = 0.505$. A coin-flip-looking score for a
  classifier that made zero useful distinctions.
- Harmonic mean: $F_1 = \frac{2 \cdot 0.01 \cdot 1.0}{0.01 + 1.0} = \frac{0.02}{1.01} \approx 0.0198$.

The harmonic mean

$$F_1 = \frac{2 \, P R}{P + R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$$

is dominated by the *smaller* of the two numbers (it's an average of reciprocals, and
a tiny $P$ makes $1/P$ huge). You cannot buy a good F1 by maxing one metric and
abandoning the other. Sanity checks: if $P = R$ then $F_1 = P = R$ (all three means
agree when the inputs are equal); if either is 0, $F_1 = 0$.

Generalization when errors have asymmetric cost: $F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P + R}$,
where $\beta > 1$ weights recall more ($F_2$ for screening), $\beta < 1$ weights
precision more.

Interview trap: F1 ignores TN entirely. Two problems with identical TP/FP/FN but
wildly different TN counts have identical F1. That's a feature when the negative class
is a huge boring background (retrieval), a bug when true negatives matter.

---

## 2. ROC and AUC

### The threshold is a free parameter

A classifier usually outputs a *score* $s(x) \in \mathbb{R}$ (a logit, a probability),
and you get a hard decision by thresholding: predict + iff $s(x) \ge t$. Every metric
in §1 is a function of $t$. Slide $t$ from $+\infty$ down to $-\infty$:

- $t = +\infty$: predict nothing positive. $TPR = 0$, $FPR = 0$.
- $t = -\infty$: predict everything positive. $TPR = 1$, $FPR = 1$.

The **ROC curve** is the path $(FPR(t), TPR(t))$ traced between those two corners.

### Build one by hand

Six examples, sorted by model score, with true labels (1 = positive, 3 positives and
3 negatives total):

| example | score | label |
|---------|-------|-------|
| A | 0.95 | 1 |
| B | 0.80 | 1 |
| C | 0.70 | 0 |
| D | 0.60 | 1 |
| E | 0.40 | 0 |
| F | 0.20 | 0 |

Sweep the threshold just below each score in turn. At each step one more example
becomes predicted-positive. If it's a true positive, TPR jumps up by $1/3$ (one of 3
positives); if it's a false positive, FPR steps right by $1/3$ (one of 3 negatives):

| threshold | predicted + | TP | FP | TPR | FPR | move |
|-----------|-------------|----|----|------|------|------|
| $>0.95$ | {} | 0 | 0 | 0 | 0 | start at (0,0) |
| $0.95$ | {A} | 1 | 0 | 1/3 | 0 | **up** |
| $0.80$ | {A,B} | 2 | 0 | 2/3 | 0 | **up** |
| $0.70$ | {A,B,C} | 2 | 1 | 2/3 | 1/3 | **right** |
| $0.60$ | {A,B,C,D} | 3 | 1 | 1 | 1/3 | **up** |
| $0.40$ | {A,B,C,D,E} | 3 | 2 | 1 | 2/3 | **right** |
| $0.20$ | all | 3 | 3 | 1 | 1 | **right** |

So the curve is a staircase from $(0,0)$: up, up, right, up, right, right, ending at
$(1,1)$. Reading the labels in score order (1,1,0,1,0,0), **each positive is a step up,
each negative is a step right**. A perfect ranker (all 1s before all 0s) goes straight
up then straight right - the curve hugs the top-left corner. A random scorer
interleaves 1s and 0s → the diagonal. The `src/roc_curve.ipynb` notebook plots exactly
this kind of staircase; here we did it in words.

**AUC** = area under this staircase. The grid is $3 \times 3 = 9$ cells of area
$1/9$; count the cells under the curve column by column: during C's right-step the
curve height is $2/3$ (2 cells), during E's it's $1$ (3 cells), during F's it's $1$
(3 cells) → $8/9 \approx 0.889$. That per-column count is not an accident - it's the
key to the most important fact about AUC:

### AUC = P(random positive scores above random negative)

**Claim.** $\text{AUC} = \Pr(s(x^+) > s(x^-))$ for a positive $x^+$ and negative $x^-$
drawn uniformly at random (ties counted $\tfrac12$).

**Proof by counting the staircase area.** With $n_+$ positives and $n_-$ negatives,
each negative contributes one "right-step" of width $1/n_-$. The height of the curve
during that step is $k/n_+$, where $k$ = number of positives ranked *above* that
negative (they were the up-steps already taken). So the area is

$$\text{AUC} = \sum_{\text{negatives } j} \frac{1}{n_-} \cdot \frac{k_j}{n_+}
= \frac{1}{n_+ n_-} \sum_j k_j
= \frac{\#\{(i,j) : s_i^+ > s_j^-\}}{n_+ n_-},$$

which is exactly the fraction of positive–negative pairs ranked correctly, i.e., the
probability a random positive outranks a random negative. $\blacksquare$

Check on our example: pairs are $(A,B,D) \times (C,E,F)$, 9 pairs. A beats C,E,F (3).
B beats C,E,F (3). D loses to C (0.70 > 0.60) but beats E,F (2). Total $8/9 \approx 0.889$
= AUC. `metrics_scratch.ipynb` verifies both computations agree (and agree with sklearn).

Consequences of the ranking view:

- AUC is **threshold-free**: it evaluates the *ranking*, not any particular decision.
- AUC is invariant under any **strictly monotone transform** of the scores (sigmoid,
  scaling, adding a constant) - it only sees the order. So AUC tells you *nothing about
  calibration* (§3).
- AUC 0.5 = random ranking; AUC < 0.5 means your model is anti-correlated (flip it).
- This quantity is the normalized Mann–Whitney U statistic - AUC is a rank test in disguise.

### Why ROC lies on imbalanced data

Look at the two axes: $TPR = TP/(TP+FN)$ is computed *within the positives*,
$FPR = FP/(FP+TN)$ *within the negatives*. Neither axis involves the class ratio at all.
Change the prevalence from 1:1 to 1:1000 and the ROC curve **does not move**.

That sounds like a feature until you ask the question a user actually asks:
*"the model flagged this - is it really positive?"* That's precision, and precision
depends hard on the base rate. Derive it. Let $\pi$ = prevalence $= \Pr(y=1)$. Among
$N$ examples, expected counts: $TP = N\pi \cdot TPR$, $FP = N(1-\pi) \cdot FPR$. So

$$\text{Precision} = \frac{\pi \cdot TPR}{\pi \cdot TPR + (1-\pi) \cdot FPR}.$$

Plug in a genuinely good-looking operating point, $TPR = 0.9$, $FPR = 0.01$
(specificity 99%!):

- Balanced, $\pi = 0.5$: precision $= \frac{0.45}{0.45 + 0.005} = 0.989$. 
- Rare, $\pi = 0.001$: precision $= \frac{0.0009}{0.0009 + 0.00999} \approx 0.083$.

Same classifier, same ROC point, same AUC - but at a 1:1000 base rate, 92% of alarms
are false. The $(1-\pi)FPR$ term swamps the $\pi \cdot TPR$ term because there are just
so many negatives available to generate false positives. (This is exactly the
base-rate fallacy from Bayes' rule - precision *is* the posterior $\Pr(y=1 \mid \hat{y}=1)$.)

**PR curves** plot (recall, precision) over the threshold sweep. Precision has $\pi$
baked in, so the PR curve *does* collapse when the data gets imbalanced - which is the
honest behavior. Rules of thumb:

- Rare positive class, and you care about the quality of the flagged set (fraud,
  retrieval, anomaly detection) → **PR curve / average precision**.
- Classes roughly comparable, or you truly care about both error rates symmetrically
  → ROC/AUC is fine and has nicer properties (baseline is always 0.5; PR's baseline
  is $\pi$, so PR-AUCs aren't comparable across datasets with different prevalence).

`metrics_scratch.ipynb` demonstrates this numerically: 100× the imbalance, AUC stays flat,
average precision collapses.

Interview trap: "our fraud model has AUC 0.98" is close to meaningless on 1:10,000
data. Ask for precision at the operating recall, or the PR curve.

---

## 3. Calibration: do the probabilities mean anything?

A model is **calibrated** if, among all examples where it outputs $p = 0.7$, about 70%
are actually positive. Formally: $\Pr(y = 1 \mid \hat{p}(x) = p) = p$ for all $p$.
Calibration is about the *values* of the scores; discrimination (AUC) is about their
*order*. Independent properties:

- Perfectly calibrated, zero discrimination: always predict $\hat{p} = \pi$ (the base
  rate). Calibrated by construction, useless for ranking.
- Perfect discrimination, terrible calibration: a model that outputs 0.51 for every
  positive and 0.49 for every negative. AUC = 1.0, but "0.51" is a lie as a probability.

Why care? Because downstream decisions consume probabilities: expected-cost decisions
(treat if $p \cdot \text{cost}_{FN} > (1-p)\cdot\text{cost}_{FP}$), ranking ads by
$p(\text{click}) \times \text{bid}$, combining models. Modern deep nets are famously
**overconfident** (Guo et al. 2017): they output 0.99 and are right 85% of the time.

**Reliability diagram / table.** Bin predictions by confidence (e.g. 10 bins), and for
each bin compare mean predicted probability vs empirical accuracy. Calibrated → the
points sit on the diagonal. Overconfident → points below the diagonal (accuracy <
confidence). A one-number summary is the **expected calibration error**:

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|,$$

a bin-count-weighted average gap. (Caveat: ECE depends on the binning; it's a
histogram estimate of a conditional expectation.)

**Brier score** = MSE on probabilities: $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$. It
decomposes (Murphy decomposition) as

$$\text{Brier} = \underbrace{\text{reliability}}_{\text{calibration gap}} - \underbrace{\text{resolution}}_{\text{how much bins differ from base rate}} + \underbrace{\pi(1-\pi)}_{\text{uncertainty}}.$$

Intuition: you're penalized for miscalibration (reliability term, want 0), rewarded
for making confident bins that genuinely differ from the base rate (resolution, want
large), and pay an irreducible floor $\pi(1-\pi)$ set by the label noise itself. Brier
is a *proper scoring rule*: it's minimized in expectation by reporting the true
probability - you can't game it by hedging.

**Fixing calibration** (all post-hoc, fit on a held-out validation set, never train):

- **Platt scaling**: fit a logistic regression on the model's score,
  $\hat{p} = \sigma(a s + b)$ - two parameters, learned by minimizing NLL on
  validation data. Parametric, works when the miscalibration is a smooth
  sigmoid-shaped distortion.
- **Isotonic regression**: fit the best *monotone non-decreasing* step function
  mapping score → probability (via pool-adjacent-violators). Nonparametric, more
  flexible, needs more data, can overfit small validation sets.
- **Temperature scaling**: Platt with the intercept dropped and one shared scalar  - 
  divide the logits by $T$ before softmax: $\hat{p} = \text{softmax}(z/T)$. $T > 1$
  softens (fixes overconfidence), $T < 1$ sharpens. One parameter, fit by minimizing
  validation NLL. Crucially, $z/T$ preserves the order of an example's logits, so
  **the argmax, and hence accuracy, is unchanged** - you fix the probabilities
  without touching accuracy. For binary problems the ranking of examples, and so
  AUC, is unchanged too. (With more than two classes, one class's probability can
  reorder across examples, so one-vs-rest AUC can shift slightly.) The standard fix
  for neural nets. `calibration_scratch.ipynb` implements it with golden-section search.

---

## 4. Bias–variance: the full derivation

Setup. Data generated as $y = f(x) + \varepsilon$, with noise
$\mathbb{E}[\varepsilon] = 0$, $\text{Var}(\varepsilon) = \sigma^2$, independent of
everything else. We fit $\hat{f}_D$ on a random training set $D$, and ask: at a fixed
test point $x$, what is the expected squared error over both the draw of $D$ and the
test noise?

$$\text{Err}(x) = \mathbb{E}_{D, \varepsilon}\left[(y - \hat{f}_D(x))^2\right]$$

Define the average prediction over training sets: $\bar{f}(x) = \mathbb{E}_D[\hat{f}_D(x)]$.

Step 1 - peel off the noise. Write $y - \hat{f}_D = \varepsilon + (f - \hat{f}_D)$:

$$\mathbb{E}\left[(y - \hat{f}_D)^2\right]
= \mathbb{E}[\varepsilon^2] + 2\,\mathbb{E}[\varepsilon]\,\mathbb{E}[f - \hat{f}_D] + \mathbb{E}\left[(f - \hat{f}_D)^2\right]
= \sigma^2 + \mathbb{E}_D\left[(f - \hat{f}_D)^2\right].$$

The cross term dies because $\varepsilon$ is independent of $D$ and has mean zero.

Step 2 - peel apart the model error. Insert and subtract $\bar{f}$:
$f - \hat{f}_D = (f - \bar{f}) + (\bar{f} - \hat{f}_D)$:

$$\mathbb{E}_D\left[(f - \hat{f}_D)^2\right]
= (f - \bar{f})^2 + 2(f - \bar{f})\,\mathbb{E}_D[\bar{f} - \hat{f}_D] + \mathbb{E}_D\left[(\bar{f} - \hat{f}_D)^2\right].$$

The middle term dies because $\mathbb{E}_D[\bar{f} - \hat{f}_D] = \bar{f} - \bar{f} = 0$
(that's the definition of $\bar{f}$; note $f - \bar{f}$ is a constant w.r.t. $D$). So:

$$\boxed{\;\text{Err}(x) = \underbrace{(f(x) - \bar{f}(x))^2}_{\text{bias}^2}
\;+\; \underbrace{\mathbb{E}_D\left[(\hat{f}_D(x) - \bar{f}(x))^2\right]}_{\text{variance}}
\;+\; \underbrace{\sigma^2}_{\text{irreducible noise}}\;}$$

No "it can be shown" - that's the whole proof: two completions of the square, two
cross terms killed by independence and by the definition of the mean.

Read it out loud:

- **Bias²**: how wrong is the *average* model - the error you'd make even with
  infinitely many training sets to average over. Systematic error from the model
  family being too rigid. A degree-1 polynomial fit to a sine wave has high bias no
  matter how much data you give it.
- **Variance**: how much does the fit *vary* across training sets - sensitivity to
  which particular sample you got. A degree-15 polynomial through 20 noisy points
  changes wildly with each resample.
- **$\sigma^2$**: label noise. No model beats this in expectation. A finite test set
  can come in a little below $\sigma^2$ by chance, but test MSE far below $\sigma^2$
  on a large test set means you have a bug (or leakage, §5).

**Tie to under/overfitting.** As capacity grows, bias² falls (the family can express
$f$) while variance rises (more knobs chasing noise). Their sum is U-shaped;
underfitting = left of the minimum (bias-dominated), overfitting = right
(variance-dominated). `bias_variance_experiment.ipynb` measures all three terms
empirically for polynomial degrees 1–15 and prints the U.

Two classic knobs restated in this language: **more data** reduces variance (bias
unchanged - same family, better averaged); **regularization** trades variance down for
a little bias up.

**Double descent, honestly.** The U-shape is real but not the whole story. If you keep
increasing capacity past the *interpolation threshold* (the model exactly fits the
training data), test error often falls *again* - worst error near the threshold, then a
second descent in the overparameterized regime. The mechanism, as best understood: at
the threshold there is essentially one interpolating solution and it's forced to be
wild; with far more parameters than data there are many interpolating solutions, and
the training procedure (SGD, minimum-norm least squares) implicitly picks a *smooth*,
low-norm one - variance comes back down. This is an empirical regularity with solid
theory in linear/random-feature models and consistent observations in deep nets, not a
repeal of bias–variance: the decomposition identity still holds; what changes is that
"more parameters ⇒ more variance" fails when the fitting procedure is implicitly
regularized. Don't oversell it in interviews; do know the phrase "interpolation
threshold."

---

## 5. Cross-validation and the leakage sins

### Why one split isn't enough

A single train/test split gives you *one sample* of the random variable "test error of
a model trained on a random 80%." With small data that random variable has high
variance: reshuffle the split and your 0.83 becomes 0.78 or 0.88. You'd be comparing
models on noise.

**k-fold CV**: split the data into $k$ folds; for each fold, train on the other
$k-1$ and evaluate on it; average the $k$ scores. Every example gets used for
validation exactly once, and you get a mean *and a spread* across folds - report both.
Trade-offs in $k$: small $k$ (=2,3) → each model trains on much less data (pessimistic
bias); $k = n$ (leave-one-out) → nearly unbiased but expensive and the $n$ trained
models are highly correlated (they share almost all data), so the variance of the
estimate isn't as low as $n$ folds suggests. $k = 5$ or $10$ is the standard
compromise.

**Stratification**: sample folds so each preserves the class ratio. Essential when a
class is rare - with 3% positives and 5 random folds, some fold may get almost none,
making its score garbage. For regression, you can stratify on binned $y$.

### Leakage: the classic sins

Leakage = information from the validation/test data (or from the future, or from the
label) sneaking into training. It always inflates your offline score and always
deflates in production. The three classics, each with a concrete failure:

1. **Preprocessing before splitting.** You standardize features using the mean/std of
   the *full* dataset, then split. The training fold now knows the test fold's
   statistics. Same sin: choosing features by correlation with the label on the full
   data, fitting PCA on the full data, imputing with the full-data median. The fix is
   mechanical: **fit every transform inside the training fold only**, apply to
   validation (this is exactly what sklearn's `Pipeline` inside `cross_val_score`
   enforces). Scaling leakage is often mild; feature *selection* on the full data can
   be catastrophic (with more features than examples you can "select" pure-noise
   features that correlate with the whole label vector, then "validate" them on data
   they were chosen on).
2. **Target leakage**: a feature that is a downstream consequence of the label.
   Canonical example: predicting pneumonia with a feature `took_antibiotics` - patients
   took antibiotics *because* they were diagnosed. AUC 0.99 offline; at prediction
   time the feature doesn't exist yet. Detection heuristics: a feature that is
   suspiciously predictive alone; ask for each feature "is this value known *at the
   moment the prediction must be made*?"
3. **Temporal leakage**: random-splitting time-series data. Training on Wednesday and
   Friday, testing on Thursday lets the model interpolate the future. Also: computing
   a "user's average purchase amount" feature over the whole history including
   post-split purchases. Fix: split by time (train on past, validate on later),
   rolling-origin / expanding-window CV, and compute every aggregate feature *as of*
   the prediction timestamp.

Related sin worth naming: **grouped leakage** - the same patient/user has rows in both
train and test. The model memorizes the patient, you think it learned medicine. Fix:
group-aware splits (all of a patient's rows in one fold).

### Nested CV: hyperparameter honesty

If you tune hyperparameters by picking whatever maximizes k-fold CV score, that
maximum is an **optimistically biased** estimate of generalization - you selected it
*because* it was the highest of many noisy numbers (winner's curse). The CV score has
become part of training. Honest protocol: an **outer loop** of CV for estimation; within
each outer training fold, an **inner loop** of CV for hyperparameter selection; the
outer validation fold only ever scores a model whose hyperparameters were chosen
without seeing it. Report the outer scores. In practice with big data, a simple
train/val/test split with test touched exactly once serves the same purpose.

---

## 6. Is model A actually better than model B?

Your new model scores 0.847, the old one 0.842. Ship it? The difference may be noise
from (a) the finite test set and (b) training randomness (init, data order, dropout).

**Paired tests intuition.** Both models are evaluated on the *same* test examples, so
don't compare two independent means - compare *per-example (or per-fold) differences*
$d_i = \text{err}_A(x_i) - \text{err}_B(x_i)$. Pairing subtracts out the shared
example-difficulty variance ("this test point is hard for everyone"), leaving a much
lower-variance estimate of the gap; then a paired t-test (or the nonparametric
Wilcoxon signed-rank) asks whether $\bar{d}$ is distinguishable from 0 given its
spread. For a single test set of binary decisions, **McNemar's test** does the paired
logic on the 2×2 table of (A right/wrong × B right/wrong) - only the discordant cells
matter. Caveat for CV: the k fold scores are *not* independent (overlapping training
sets), so the naive paired t-test over folds is anti-conservative; corrections exist
(Nadeau–Bengio), but at minimum know *why* it's a problem.

**Multiple seeds.** For neural nets, training noise alone (seed) moves benchmark
scores by amounts that rival claimed SOTA improvements. One run of A vs one run of B
compares two random draws. Minimum honest practice: 3–5 seeds each; report the mean
and its standard error across seeds (std divided by the square root of the number of
seeds), and compare models with a paired test or a confidence interval on the
difference. Overlapping ±std bands are not a test. And if you compared 20 model
variants, the winner's score is inflated by selection (same winner's curse as §5), so
use a stricter threshold, such as Bonferroni's 0.05/20 (foundations §3.8).

---

## 7. Imbalanced data

At 99:1, "predict majority always" gets 99% accuracy. So accuracy is off the table; use
PR-based metrics (§2) or per-class recalls. What to do about the *model*:

**Resampling.** Oversample the minority (duplicate, or synthesize à la SMOTE  - 
interpolate between minority neighbors) or undersample the majority. Oversampling
risks overfitting the duplicated points (do it *inside* CV folds only - resampling
before splitting is leakage sin #1: identical copies of one example land in both train
and validation); undersampling throws away data.

**Class weights.** Weight the loss: $L = \frac{1}{N}\sum_i w_{y_i} \, \ell(\hat{p}_i, y_i)$
with $w_1 > w_0$, typically $w_c \propto 1/\text{freq}(c)$. Derive the gradient effect
for weighted cross-entropy on a logit $z$, $\hat{p} = \sigma(z)$. Recall
$\partial \ell / \partial z = \hat{p} - y$ for unweighted CE. With weights:

$$\frac{\partial L_i}{\partial z_i} = w_{y_i} (\hat{p}_i - y_i)$$

So every gradient step from a positive example is scaled by $w_1$. Two consequences,
both worth saying in an interview: (1) it's equivalent *in expectation* to oversampling
the positives by $w_1$ (same expected gradient, less variance, no duplicated data);
(2) at the optimum the balance point shifts - with weights $w_1, w_0$ the loss is
stationary where $w_1 \pi_{\text{local}} (1-\hat{p}) = w_0 (1-\pi_{\text{local}})\hat{p}$,
i.e. the model outputs *inflated* probabilities:
$\hat{p} = \frac{w_1 \pi}{w_1 \pi + w_0 (1-\pi)}$ instead of $\pi$. Class weights
buy recall by *destroying calibration* - recalibrate after (§3).

**Threshold moving.** Often the cleanest fix: train normally on the imbalanced data
(the probabilities stay calibrated), then choose the decision threshold from the
cost matrix instead of 0.5. Optimal decision theory: predict positive when expected
cost says so, $\hat{p} \cdot c_{FN} > (1-\hat{p}) \cdot c_{FP}$, i.e. threshold
$t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$. If a FN costs 100× a FP, threshold at ~0.0099,
not 0.5. Interview trap: people reach for SMOTE when the actual problem was just an
unexamined 0.5 threshold.

---

## 8. Regression metrics

**MSE and MAE optimize different summaries.** What constant $c$ minimizes each?

*MSE*: $g(c) = \mathbb{E}[(y - c)^2]$. $g'(c) = -2\,\mathbb{E}[y - c] = 0 \Rightarrow c = \mathbb{E}[y]$.
The **mean**. (Second derivative $2 > 0$: a minimum.)

*MAE*: $h(c) = \mathbb{E}[|y - c|]$. Derivative of $|y-c|$ w.r.t. $c$ is $+1$ when
$c > y$ and $-1$ when $c < y$, so
$h'(c) = \Pr(y < c) - \Pr(y > c)$, which crosses zero exactly where
$\Pr(y < c) = \Pr(y > c) = \tfrac12$: the **median**.

Tiny example: $y \in \{1, 1, 1, 1, 100\}$ (a skewed distribution with an outlier).
MSE-optimal prediction = mean = 20.8; MAE-optimal = median = 1. This is the whole
story of "MAE is robust to outliers": MSE's quadratic penalty makes one 100-sized
error cost as much as ~10,000 unit errors, so the fit gets dragged toward the outlier.
Choose by asking *which summary of the conditional distribution you want to predict*
(and whether big errors are disproportionately bad - for house prices maybe yes, for
delivery-time ETAs maybe you want the median). Huber loss interpolates (quadratic
near 0, linear in the tails). Generalization: pinball/quantile loss with asymmetric
slopes $\tau, 1-\tau$ recovers any quantile.

**R²** - MSE normalized by the dumbest baseline (predict $\bar{y}$):

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}.$$

$R^2 = 1$: perfect. $R^2 = 0$: no better than predicting the mean. **Negative $R^2$ is
possible** (and common on real test sets): it just means you did worse than the
constant baseline - a favorite interview gotcha ("what's the range of R²?" It's
$(-\infty, 1]$ on held-out data). Also: RMSE for interpretability in $y$'s units;
MAPE if relative error matters (undefined at $y=0$, asymmetric - penalizes
over-prediction more).

---

## 9. Offline vs online: metrics meet reality

Everything above is **offline**: computed on logged, held-out data. Production cares
about **online** metrics: click-through rate, conversion, revenue, retention, latency,
user-reported outcomes. The mapping is loose - offline AUC up 2% can be online-neutral
(the ranking improved on pairs users never see) or even negative (the model got better
at predicting clicks on clickbait). Offline metrics are cheap *proxies*; validate the
proxy periodically.

**A/B testing in one paragraph.** Randomly assign units (users, not requests - a user
must have one consistent experience, and randomizing requests leaks one arm's effect
into the other via the shared user) to control (old model) and treatment (new model).
Randomization makes the groups exchangeable, so the difference in the metric estimates
the *causal* effect. Before launching: choose the primary metric and the minimal
effect worth detecting, compute the required sample size (power analysis - small
effects on noisy metrics need surprisingly long runs), and fix the duration; run at
least full weeks to average out day-of-week cycles. Sins to name-check: **peeking**
(testing daily and stopping at the first $p < 0.05$ massively inflates false
positives - use sequential tests if you must peek), underpowered tests read as "no
effect," novelty effects (users click the new thing because it's new), and
interference between arms (marketplaces, social feeds - two-sided effects violate the
independence assumption). Guardrail metrics (latency, crash rate, unsubscribes) run
alongside the primary metric so you don't win the battle and lose the war.

---

## 10. Evaluating generative models & LLMs

### The reference problem: why generative evaluation is fundamentally harder

Everything before this section evaluated models with a **single ground truth per
example**: the label is 1, you said 0.83, we can score that. Generation breaks the
premise. "Summarize this article" has an astronomically large set of *equally correct*
outputs, and any two good summaries may share almost no words. There is no ground
truth - at best there is *a* reference, one sample from the set of acceptable answers.
Every generative metric is therefore one of three compromises: **proximity to a
reference** (BLEU/ROUGE - punishes valid answers for being different), **the model's
own probability** (perplexity - measures fit, not output quality), or **judgment**
(human or LLM - expensive, biased, but the only thing aimed at the actual question).
Know which compromise you're making; each fails differently.

### N-gram overlap: BLEU and ROUGE, and how they fail

**BLEU** (machine translation, precision-flavored): the geometric mean of modified
n-gram precisions for $n = 1..4$ - what fraction of the candidate's n-grams appear in
the reference - times a brevity penalty so you can't win by emitting three safe words.
**ROUGE** (summarization, recall-flavored): ROUGE-N is n-gram *recall* - what fraction
of the reference's n-grams the candidate covered; ROUGE-L uses longest common
subsequence. Same idea, opposite direction: BLEU asks "is what you said in the
reference?", ROUGE asks "did you say what's in the reference?".

Failure modes, with tiny examples:

- **Paraphrase blindness.** "The physician arrived late" vs reference "the doctor was
  late": two unigrams out of four overlap ("the" and "late"), and no bigrams do: a low
  score for an identical meaning.
  The metric measures *surface* overlap, and the reference problem guarantees good
  outputs with low overlap.
- **Overlap without correctness.** "Profits fell sharply" vs reference "profits rose
  sharply": 2/3 unigrams match ("profits", "sharply") and 0/2 bigrams, so unigram
  metrics give a respectable score for the *exact opposite claim*. N-grams have no notion of negation, factuality, or salience.
- **Gaming.** Extractive copying inflates ROUGE (the reference's words are literally
  in the source); short hedged outputs inflate precision-style scores until the
  brevity penalty bites.

Verdict: usable as a cheap regression alarm on reference-heavy tasks (MT-style, where
system-level correlation with humans is fair), close to useless per-example and for
open-ended generation. Interview trap: "ROUGE went up" is not evidence the summaries
got better - say what it actually measures.

### Perplexity is not a quality metric

Module 4 §5.3 derived $\operatorname{PPL} = e^{H}$: the model's effective branching
factor *on a fixed corpus*. It measures how well the model's distribution fits that
text - a modeling metric, ideal for tracking pretraining. Its limits as a *quality*
metric: (1) only comparable under the same tokenizer (per-token PPL changes when the
tokens change); (2) it scores probability mass on the reference continuation, not the
quality of what the model *samples* - decoding settings (§ module 4, temperature/top-p)
never enter it; (3) instruction-tuned models routinely have *worse* web-text PPL than
their base models while being far more useful - fitting raw internet text is simply not
the goal anymore; (4) it needs logprobs (unavailable for many closed models) and says
nothing about a single output. Use it for training progress and sanity checks; never
answer "is the chatbot good?" with perplexity.

### Human evaluation: pairwise preference → Bradley–Terry / Elo

Absolute human scores ("rate this 1–10") are noisy, rater-dependent, and drift over a
session. The reliable primitive is the **pairwise comparison**: show two outputs for
the same prompt, ask which is better - the same reason RLHF trains reward models on
comparisons, not scores (module 4 §6.3). To turn many pairwise outcomes into a
ranking, fit **Bradley–Terry**: give each model a strength $\beta_i$ and model

$$P(i \text{ beats } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \sigma(\beta_i - \beta_j),$$

then maximize the likelihood of the observed battles (it's logistic regression on
win/loss with model-indicator features). **Elo** is the online, incremental
approximation of the same idea. This is exactly the machinery behind arena-style
leaderboards (Chatbot Arena): crowdsourced blind battles → Bradley–Terry fit →
ratings with confidence intervals. Protocol hygiene: hide model identities, randomize
which side each model appears on, allow ties, and report CIs - a 5-point Elo gap on
overlapping intervals is noise, not a ranking.

### LLM-as-judge: the three biases and their mitigations

Human eval doesn't scale to every experiment, so use a strong LLM as the grader  - 
pairwise verdicts or rubric scores. It works well enough to be everywhere, and it is
a *proxy metric*, so audit it like one (§9). The three biases to name in an interview:

1. **Position bias** - judges favor the first (sometimes last) answer shown.
   Mitigation: judge every pair in both orders; keep only consistent verdicts (or
   average), and report the swap-agreement rate.
2. **Verbosity bias** - longer answers win regardless of content. Mitigation: control
   or penalize length (compare length-matched pairs; put "do not reward padding" with
   explicit criteria in the rubric).
3. **Self-preference** - a judge rates its own family's outputs higher. Mitigation:
   judge from a different model family, or a panel of judges.

General mitigations: make the judge state rubric-based reasons before its verdict,
anchor with few-shot examples of good/bad, and - non-negotiable - **validate the judge
against a small human-labeled set** (agreement rate) before trusting its deltas.

### Benchmarks and contamination, at overview level

Static benchmarks (MMLU, GSM8K, HumanEval, …) are fixed test sets, and the training
corpus is the internet - so test items leak into pretraining data. **Contamination**
inflates scores via memorization masquerading as capability; a model can ace a
benchmark it has effectively seen. Detection heuristics: n-gram overlap search against
training data, and performance cliffs on freshly written variants of the same tasks.
Practical stance: treat headline benchmark numbers as upper bounds, prefer private,
fresh, or continuously refreshed evals, and remember Goodhart (§9's proxy lesson  - 
optimized-for benchmarks stop measuring). Benchmark design, hallucination measurement,
and LLM eval in depth: module 7.

### Mini-walkthrough: "how would you evaluate a summarization feature?"

The interview wants decomposition, not a metric name. Split "good summary" into named
axes, because they trade off and fail independently: **faithfulness** (every claim in
the summary is supported by the source - the failure that kills the product is
hallucination), **coverage** (the important points made it in), **fluency/concision**
(nearly free with modern LLMs - don't overweight it). Then a protocol:

1. **Offline set:** 200–500 real documents, stratified by what you actually serve
   (length, structure, domain). Not synthetic, not cherry-picked.
2. **Faithfulness:** decompose each summary into atomic claims, verify each claim
   against the source (LLM-judge per claim - per-claim verification beats one holistic
   "is this faithful?" judgment, which misses single fabricated details). Metric:
   % summaries with zero unsupported claims.
3. **Coverage:** a once-per-document human-written key-point list; score % of key
   points present.
4. **System comparison:** pairwise LLM-judge, both orders (position bias),
   length-controlled (verbosity bias), different judge family (self-preference);
   Bradley–Terry if comparing several systems.
5. **Spot-check protocol:** every eval run, sample ~50 judge verdicts for human
   review; if human–judge agreement drops below ~85%, fix the judge before believing
   any model delta. The judge is a proxy; this is its calibration check.
6. **Online:** thumbs up/down, edit rate (did the user rewrite the summary?), and
   downstream task completion - with §9's A/B discipline. ROUGE, if at all, as a
   cheap regression alarm - never as the target.

---

## Common interview questions

**Q1. Precision vs recall: define both and give a case where each is the priority.**
Screen 1000 people, 10 of whom have cancer: the test flags 98 people, 8 of them with
cancer, and misses 2. Precision $= TP/(TP+FP) = 8/98 \approx 0.08$: of the flagged, the
fraction that is real. Recall $= TP/(TP+FN) = 8/10 = 0.8$: of the real cases, the
fraction flagged. Cancer screening prioritizes recall, because a missed cancer can kill
and a false positive costs one follow-up test. A spam filter prioritizes precision on the
spam class, because a real email sent to spam (a false positive) costs more than one spam
message in the inbox. In general, prioritize the metric whose error type (FN for recall,
FP for precision) costs more.
**If asked:** *What if both errors cost real money, as in fraud?* Get the cost ratio and set the threshold from it (Q15).

**Q2. Why is F1 the harmonic mean and not the arithmetic mean?**
Take 1000 examples with 10 positives and a classifier that predicts positive for
everything: $P = 10/1000 = 0.01$, $R = 1.0$. The arithmetic mean is $0.505$, which reads
as a middling model; the harmonic mean is $F_1 = 2PR/(P+R) = 0.02/1.01 \approx 0.02$,
which reads as a useless one. The harmonic mean is the reciprocal of the average of $1/P$
and $1/R$, so a small $P$ makes $1/P$ large and pulls $F_1$ toward the smaller of the two
values. A classifier therefore cannot reach a high F1 by maximizing one metric and
ignoring the other.
**If asked:** *What does F1 ignore?* True negatives: two problems with the same TP, FP and FN but different TN counts have the same F1.

**Q3. What exactly does AUC = 0.9 mean?**
Take 10 positives and 10 negatives, which form 100 positive-negative pairs. AUC = 0.9
means the model gives the positive the higher score in 90 of those 100 pairs (ties count
as half). In general, AUC is the probability that a randomly drawn positive scores above a
randomly drawn negative, which equals the area under the ROC staircase (§2). It depends
only on the order of the scores, so it involves no threshold, does not depend on the
class ratio, and says nothing about whether the scores are calibrated probabilities.
**If asked:** *What do AUC = 0.5 and AUC < 0.5 mean?* 0.5 is a random ordering; below 0.5 the scores are anti-correlated with the label, and negating them gives $1 - \text{AUC}$.

**Q4. Your fraud dataset is 1:1000 and your AUC is 0.98. Are you done?**
No. Take 1,000,000 transactions with 1,000 frauds and an operating point with TPR = 0.9
and FPR = 0.01. The model catches 900 frauds and also flags 1% of the 999,000 legitimate
transactions, which is 9,990 false alarms, so precision is $900/10{,}890 \approx 8\%$.
TPR is computed within the positives and FPR within the negatives, so the ROC curve and
AUC are the same at 1:1 and at 1:1000, while precision falls with the prevalence $\pi$:

$$\text{Precision} = \frac{\pi\,TPR}{\pi\,TPR + (1-\pi)\,FPR}.$$

Report the PR curve, average precision, or precision at the recall the business requires.
**If asked:** *What is the baseline for average precision?* The prevalence $\pi$ (here 0.001), so AP values are not comparable across datasets with different prevalence.

**Q5. Can a model have AUC 1.0 and be badly calibrated?**
Yes. A model that outputs 0.51 for every positive and 0.49 for every negative ranks every
positive above every negative, so AUC = 1.0, yet every example scored 0.51 is positive,
a calibration gap of 0.49. AUC depends only on the order of the scores; calibration asks
whether, among examples scored $p$, a fraction $p$ are positive. The reverse case also
exists: predicting the base rate $\pi$ for every example is perfectly calibrated and has
AUC 0.5.
**If asked:** *How would you fix this model's probabilities?* Fit Platt scaling, $\sigma(as+b)$, on a held-out validation set; with $a > 0$ it is strictly increasing, so AUC stays 1.0.

**Q6. What is temperature scaling and why doesn't it change accuracy?**
Logits $(2, 1, 0)$ give softmax probabilities $(0.665, 0.245, 0.090)$; dividing by
$T = 2$ gives logits $(1, 0.5, 0)$ and probabilities $(0.506, 0.307, 0.186)$. The top
class is the same and its probability fell from 0.665 to 0.506. Temperature scaling is
post-hoc calibration: output $\text{softmax}(z/T)$ with one scalar $T$ fit by minimizing
NLL on a validation set, where $T > 1$ softens overconfident outputs. Dividing by a
positive constant keeps the order of each example's logits, so the argmax, and therefore
accuracy, does not change. For binary problems the ranking of examples, and so AUC, is
unchanged too.
**If asked:** *Can AUC change with more than two classes?* Yes: one class's probability depends on all the logits, so its order across examples can change and one-vs-rest AUC can shift slightly.

**Q7. State and sketch the bias–variance decomposition.**
A straight line fit to a sine wave is wrong in the same way on every training set (bias);
a degree-15 polynomial through 20 noisy points changes shape with every resample
(variance). At a fixed $x$, with $y = f(x) + \varepsilon$, $\text{Var}(\varepsilon) = \sigma^2$,
and $\bar f = \mathbb{E}_D[\hat f_D]$ the average fit over training sets $D$:

$$\mathbb{E}_{D,\varepsilon}\big[(y-\hat{f}_D(x))^2\big] = \underbrace{(f - \bar{f})^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big[(\hat{f}_D-\bar{f})^2\big]}_{\text{variance}} + \sigma^2.$$

Sketch: write $y - \hat f_D = \varepsilon + (f - \hat f_D)$ and expand the square; the
cross term is zero because $\varepsilon$ has mean zero and is independent of $D$. Then
write $f - \hat f_D = (f - \bar f) + (\bar f - \hat f_D)$ and expand; that cross term is
zero because $\mathbb{E}_D[\hat f_D] = \bar f$. Full derivation in §4.
**If asked:** *How does this connect to under- and overfitting?* As capacity grows, bias² falls and variance rises; underfitting is the bias-dominated side of the U, overfitting the variance-dominated side.

**Q8. Does more training data reduce bias or variance? Does regularization?**
A degree-15 polynomial fit to 20 noisy points changes shape with each new sample; fit to
2,000 points, the fits from different samples nearly agree, so variance falls. A straight
line fit to a sine wave keeps the same systematic error at any sample size, so bias stays
the same: more data reduces variance and leaves bias unchanged. Regularization, such as a
ridge penalty $\lambda\|w\|^2$, shrinks the coefficients toward zero: the fit varies less
across samples (variance down) and the average fit moves away from $f$ (bias up). Choose
$\lambda$ where validation error, which tracks the sum, is smallest.
**If asked:** *How do you tell which one dominates?* High training error close to validation error indicates bias; low training error with a large gap to validation error indicates variance.

**Q9. Why can't test MSE go below $\sigma^2$, and what does it mean if it does?**
Suppose $y = f(x) + \varepsilon$ with Gaussian noise of variance $\sigma^2 = 1$. For any
model $g$ fixed before seeing the test labels, expected test MSE is
$\sigma^2 + \mathbb{E}[(f - g)^2] \ge \sigma^2$, so even the true $f$ has expected MSE 1.
A finite test set gives an average of noisy terms: for the true $f$ on 100 points, the
test MSE has standard deviation $\sqrt{2\sigma^4/100} \approx 0.14$, so a value of 0.9
can occur by chance. On 100,000 points that standard deviation is about 0.0045, so 0.9
is about 22 standard deviations low and indicates a bug or leakage, such as a feature
derived from the label.
**If asked:** *How would you find the leak?* Look for a feature that is suspiciously predictive alone and ask whether its value is known at prediction time (§5).

**Q10. What is double descent?**
In minimum-norm least-squares regression on random features with $n = 100$ training
points, test error typically falls, rises to a peak near $p = 100$ features, then falls
again as $p$ grows well past 100. The peak is at the interpolation threshold, the
capacity at which the model can first fit the training data exactly: there the single
interpolating solution needs large coefficients to pass through every noisy point, so
variance is high. Past it, many solutions interpolate, and the fitting procedure
(minimum-norm least squares, which gradient descent from zero initialization reaches in
linear models) selects the one with the smallest norm, which is smoother, so variance
falls again. The bias-variance identity still holds; what fails is the assumption that
more parameters always means more variance.
**If asked:** *Is it seen in deep nets?* Yes, in test error versus model width and versus training epochs, though the theory is most complete for linear and random-feature models.

**Q11. Name three forms of data leakage with examples.**
(1) Preprocessing before splitting: standardizing with the mean and standard deviation of
the full dataset, or selecting features by correlation with the label on all the data,
gives the training folds statistics of the test fold; fit every transform on the training
fold only. (2) Target leakage: a feature that is a consequence of the label, such as
"took antibiotics" when predicting pneumonia, which gives high offline AUC and does not
exist yet when the prediction is made. (3) Temporal leakage: a random split of
time-series data trains on Friday and tests on Thursday, and a feature such as a user's
average purchase computed over the full history includes purchases made after the
prediction time; split by time and compute every feature as of the prediction timestamp.
A fourth is grouped leakage: rows from the same patient in train and test, which the
model can score well on by memorizing the patient.
**If asked:** *How do you detect target leakage?* A feature that is suspiciously predictive alone, plus asking for each feature whether its value is known at the moment of prediction.

**Q12. Why nested CV? What goes wrong with plain CV for hyperparameter tuning?**
Suppose 50 hyperparameter settings all have true accuracy 80%, and each CV estimate has
independent random error with standard deviation 1 point. The largest of 50 such
estimates is on average about 2.25 standard deviations above the mean, so the best
setting reports about 82.2% while its true accuracy is 80%. The CV score of the selected
setting is optimistically biased because the same score was used to make the selection
(winner's curse). Nested CV separates the two jobs: an inner CV loop inside each outer
training fold picks the hyperparameters, and the outer fold, which played no part in that
choice, scores the result; report the outer scores.
**If asked:** *Is nested CV needed with a large dataset?* No: a train/validation/test split with the test set used once gives the same protection.

**Q13. Model A: 84.7%, model B: 84.2% on the same test set. Ship A?**
Not on this evidence. On 10,000 test examples each accuracy has a standard error of about
$\sqrt{0.845 \times 0.155/10{,}000} \approx 0.36$ points, so if the two scores were
independent the 0.5-point gap would be about one standard error of the difference
($0.36\sqrt{2} \approx 0.51$). They share the same examples, so use a paired test:
McNemar's test uses only the examples where the models disagree, with statistic
$(b-c)^2/(b+c)$ compared against a chi-square distribution with 1 degree of freedom. If
A alone is right on $b = 300$ examples and B alone on $c = 250$, the statistic is about
4.5 and $p \approx 0.03$; with 1,025 versus 975, the same 50-example gap gives 1.25 and
$p \approx 0.26$. For neural nets, also train each model with 3 to 5 seeds, because
seed-to-seed variation can be as large as 0.5 points.
**If asked:** *How do you report the seed runs?* Mean and standard error (std divided by $\sqrt{\text{number of seeds}}$), with a paired test or a confidence interval on the difference; overlapping ±std bands are not a test.

**Q14. Derive what a class weight does to the gradient.**
With 1% positives and inverse-frequency weights $w_1 = 99$, $w_0 = 1$, each positive
example's gradient is 99 times larger. For weighted cross-entropy on a logit $z$ with
$\hat p = \sigma(z)$, the unweighted gradient $\hat p_i - y_i$ is scaled by the class
weight:

$$\frac{\partial L_i}{\partial z_i} = w_{y_i}(\hat p_i - y_i).$$

In expectation this matches oversampling the positives 99 times, without duplicated
rows. The side effect is on calibration: the loss is minimized at
$\hat p = w_1\pi/(w_1\pi + w_0(1-\pi))$ instead of $\pi$; at $\pi = 0.01$ that is
$0.99/(0.99 + 0.99) = 0.5$, so a region with 1% positives gets predicted probability 0.5.
Recalibrate afterwards, or train unweighted and move the threshold.
**If asked:** *Where does that formula come from?* Setting the expected weighted gradient $w_1\pi(\hat p - 1) + w_0(1-\pi)\hat p$ to zero and solving for $\hat p$.

**Q15. When is threshold moving better than resampling?**
If a missed fraud (FN) costs 100 times a false alarm (FP), flagging is worth it when
$100\,\hat p > 1 - \hat p$, i.e. above $t^* = c_{FP}/(c_{FP}+c_{FN}) = 1/101 \approx 0.0099$,
not 0.5. Threshold moving is better when the model trained on the natural class ratio
already gives calibrated probabilities and only the decision rule is wrong for the costs:
it keeps calibration and needs no retraining. Resampling or reweighting changes the
learned probabilities, which helps when positives are so rare that the model barely
learns their features, but the outputs then need recalibration before a cost-based
threshold applies.
**If asked:** *What does threshold moving assume?* That $\hat p$ is calibrated, since $t^*$ treats $\hat p$ as a true probability; check a reliability table first.

**Q16. Prove the constant minimizing MAE is the median.**
Take $y \in \{1, 1, 1, 1, 100\}$ and a candidate constant $c = 10$: four points lie
below $c$ and one above, so raising $c$ by a small $\delta$ adds $4\delta$ to the total
absolute error and removes $\delta$, a net increase of $3\delta$, and lowering $c$
reduces the error. In general, with $h(c) = \mathbb{E}|y - c|$, the derivative of
$|y - c|$ in $c$ is $+1$ when $y < c$ and $-1$ when $y > c$, so
$h'(c) = \Pr(y < c) - \Pr(y > c)$. This is negative below the median, positive above it,
and zero where both probabilities are $\tfrac12$: the median (here 1). For MSE,
$g'(c) = -2\,\mathbb{E}[y - c] = 0$ gives the mean (here 20.8, pulled up by the outlier).
**If asked:** *Which loss gives other quantiles?* The pinball loss with slopes $\tau$ and $1-\tau$ is minimized at the $\tau$-quantile.

**Q17. Can R² be negative?**
Yes. With test labels $\{1, 2, 3\}$ and predictions $\{3, 2, 1\}$,
$\text{SS}_{res} = 4 + 0 + 4 = 8$ and $\text{SS}_{tot} = 1 + 0 + 1 = 2$ (squared
deviations from the test mean 2), so $R^2 = 1 - 8/2 = -3$. In general
$R^2 = 1 - \text{SS}_{res}/\text{SS}_{tot}$ compares the model with the constant
prediction $\bar y$, the mean of the labels being scored, so it is negative whenever the
model has larger squared error than that constant. On held-out data the range is
$(-\infty, 1]$; it is guaranteed to lie in $[0, 1]$ only on the training data of a
least-squares fit with an intercept.
**If asked:** *Does predicting the training-set mean give $R^2 = 0$ on the test set?* Only if it equals the test mean; otherwise $R^2$ is negative.

**Q18. Your offline AUC improved 2% but the A/B test is flat. Explain.**
Suppose the new model orders items 50 to 100 better, but users look only at the top 10:
AUC over all pairs rises and what users see does not change. Offline AUC is a proxy for
the online goal, and the two can differ for several reasons: the improvement is on pairs
users never see; the offline label (click) differs from the online goal (conversion,
retention), so a model can get better at predicting clicks on clickbait; live traffic
has a different distribution from the logged data; and the logged data was collected
under the old model's choices, so it under-represents items the new model would show.
The A/B test measures the causal effect on the real goal, so trust it and find which
assumption behind the offline metric failed.
**If asked:** *What offline metric would track online better?* One restricted to what users see, such as precision in the top 10 positions, computed on a label that matches the online goal.

**Q19. Why do you randomize by user rather than by request in an A/B test, and what is peeking?**
If one user gets the new ranker on one request and the old ranker on the next, what they
saw from one model changes how they behave on the other model's request, so the two arms
are mixed and the observations are not independent; randomizing by user gives each user
one consistent experience. Peeking means testing for significance repeatedly as data
arrives and stopping at the first $p < 0.05$. With no true effect and 10 equally spaced
looks, the chance that at least one look gives $p < 0.05$ is about 19%, not 5%. The fix
is to fix the sample size and duration in advance and test once, or to use a sequential
testing method designed for repeated looks.
**If asked:** *How long should the test run?* Long enough to reach the sample size from a power analysis, in full weeks to cover day-of-week cycles.

**Q20. Why does the bias–variance decomposition not carry over cleanly to 0–1 loss?**
At one test point with true label 1, suppose model A predicts 0 on every training set and
model B predicts 1 on 30% of training sets and 0 on the rest. Both have the same majority
prediction, 0, which is wrong (both are biased), and A has zero variance, yet A's
expected 0–1 loss is 1.0 while B's is 0.7: variance lowered the error. For squared error
the cross terms vanish under expectation and the error splits exactly into
bias² + variance + noise; 0–1 loss is not a squared distance, so that algebra does not
apply and no exact additive split exists. Decompositions for 0–1 loss (for example
Domingos 2000) define bias through the majority prediction and include terms whose sign
depends on whether that prediction is correct.
**If asked:** *What happens where the majority prediction is correct?* Variance increases the error: a model right on 70% of training sets has loss 0.3, versus 0 for one that is always right.

**Q21. Why is evaluating a text generator fundamentally harder than evaluating a classifier?**
Two correct summaries of "Profits fell 10% in Q3 as costs rose" are "Q3 profit dropped
10% on higher costs" and "Rising costs cut third-quarter profits by a tenth", which share
one word ("costs"). A classifier has one correct label per example; a generator has a
large set of acceptable outputs, and a reference is one sample from that set. Every
generative metric is therefore a compromise: overlap with a reference (BLEU, ROUGE)
penalizes correct outputs worded differently; the model's probability on a corpus
(perplexity) measures fit to that text, not the quality of sampled outputs; judgment by
humans or an LLM addresses quality directly but is expensive and has known biases. A
strong answer names which compromise a metric makes and how it fails.
**If asked:** *How would you evaluate a summarizer?* Split quality into faithfulness, coverage and concision, and measure each with claim-level checks and pairwise judgments (§10 walkthrough).

**Q22. Your summarizer's ROUGE improved but users say summaries got worse. Possible?**
Yes. Against the reference "profits rose sharply", the candidate "profits fell sharply"
matches 2 of the 3 reference unigrams (ROUGE-1 recall 0.67) while stating the opposite
fact. ROUGE-N is the fraction of the reference's n-grams that appear in the candidate, so
it rises when a summary copies the reference's wording, and it does not measure factual
correctness, which points matter most, or paraphrase: a better summary in different
words can score lower. Its agreement with human judgment is fair at the system level and
poor per example. Measure faithfulness and user preference directly, and keep ROUGE as a
cheap regression alarm.
**If asked:** *How would you measure faithfulness?* Split each summary into atomic claims, check each claim against the source, and report the fraction of summaries with zero unsupported claims.

**Q23. Why is perplexity a poor headline metric for chatbot quality?**
A model with mean NLL 2.0 nats per token on a corpus has PPL $= e^{2.0} \approx 7.4$; if
a different tokenizer splits the same text into 1.5 times as many tokens and the total
NLL of the text stays the same, per-token NLL becomes 1.33 and PPL becomes
$e^{1.33} \approx 3.8$, with no change in how well the text is predicted. Perplexity is
$e^{\text{mean NLL}}$ on a fixed corpus, a measure of how well the model's distribution
fits that text, and it is comparable only under the same tokenizer. It scores the
probability of reference text, so decoding settings such as temperature and top-p never
enter it. Instruction-tuned models often have worse web-text perplexity than their base
models while being better assistants. Use it to track pretraining; use preference and
task evaluations for quality.
**If asked:** *What else limits it for closed models?* It needs token log-probabilities, which many closed model APIs do not return.

**Q24. You're using GPT-x to judge two models' outputs. What biases and mitigations?**
If the judge prefers answer A when A is shown first and prefers B when B is shown first,
its verdict follows position, not content. Position bias: judges favor the first
(sometimes the last) answer; judge every pair in both orders, keep only consistent
verdicts, and report the swap-agreement rate. Verbosity bias: longer answers win
regardless of content; compare length-matched pairs and use a rubric that tells the judge
not to reward padding. Self-preference: a judge rates outputs from its own model family
higher; use a judge from another family or a panel. The judge is a proxy metric, so check
its agreement with a human-labeled sample before trusting its deltas.
**If asked:** *What agreement is good enough?* Set the bar in advance, for example about 85% agreement with human labels on roughly 50 sampled verdicts per run, and fix the judge if it falls below.

## Exit checklist

- [ ] Cold, on paper: fill a confusion matrix, compute precision/recall/F1, and say who pays for each error type - 4 min
- [ ] Build an ROC curve by hand from 6 scored examples and state AUC's probabilistic meaning - 5 min
- [ ] Explain why ROC flatters on imbalanced data and when precision-recall curves are the right view - 3 min
- [ ] Derive the full bias-variance decomposition of expected squared error - 5 min
- [ ] Name the classic leakage sins (fit-before-split, temporal, group, target) and what nested CV fixes - 3 min
- [ ] Decide via standard error whether model A at 90.5% beats model B at 90% on n=1000 - 3 min
- [ ] LLM-as-judge: name the three biases with mitigations, then walk the summarization-eval answer end to end - 5 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
