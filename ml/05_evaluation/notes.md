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
  late": one unigram overlaps out of four ("late"); near-zero score, identical meaning.
  The metric measures *surface* overlap, and the reference problem guarantees good
  outputs with low overlap.
- **Overlap without correctness.** "Profits fell sharply" vs reference "profits rose
  sharply": 2/3 unigrams and 1/2 bigrams match - a respectable score for the *exact
  opposite claim*. N-grams have no notion of negation, factuality, or salience.
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

**Q1. Precision vs recall - define both and give a case where each is the priority.**
Precision $= TP/(TP+FP)$: of the flagged, how many are real. Recall $= TP/(TP+FN)$:
of the real, how many were flagged. Spam filtering prioritizes precision (a lost real
email is expensive); cancer screening prioritizes recall (a missed cancer is
catastrophic, follow-up tests catch the FPs).

**Q2. Why is F1 the harmonic mean and not the arithmetic mean?**
The harmonic mean is dominated by the smaller value, so a degenerate classifier can't
score well by maxing one metric. Predict-all-positive at 1% prevalence: P=0.01, R=1.0.
Arithmetic mean 0.505 (looks half-decent); harmonic mean ≈0.02 (correctly terrible).

**Q3. What exactly does AUC = 0.9 mean?**
Pick a random positive and a random negative; the model scores the positive higher
90% of the time. It's a statement about ranking, independent of threshold, prevalence,
and calibration.

**Q4. Your fraud dataset is 1:1000 and your AUC is 0.98. Are you done?**
No. ROC axes (TPR, FPR) are prevalence-free, so AUC hides the base rate. Derive
precision $= \pi TPR / (\pi TPR + (1-\pi)FPR)$: with TPR=0.9, FPR=0.01, $\pi=0.001$,
precision ≈ 8% - most alerts are false. Report the PR curve / precision at the
required recall.

**Q5. Can a model have AUC 1.0 and be badly calibrated?**
Yes - output 0.51 for every positive, 0.49 for every negative: perfect ranking,
meaningless probabilities. AUC only sees order; calibration is about the values.
Conversely, always predicting the base rate is perfectly calibrated with AUC 0.5.

**Q6. What is temperature scaling and why doesn't it change accuracy?**
Post-hoc calibration: divide logits by scalar $T$ (fit on validation NLL) before
softmax. $T>1$ softens overconfident outputs. Division by a positive constant is
order-preserving for each example's logits, so the argmax (hence accuracy) is
unchanged - only the probability values move. For binary problems AUC is unchanged
too; with more classes, one-vs-rest AUC can shift slightly.

**Q7. State and sketch the bias–variance decomposition.**
$\mathbb{E}[(y-\hat{f}_D(x))^2] = (f - \bar{f})^2 + \mathbb{E}_D[(\hat{f}_D-\bar{f})^2] + \sigma^2$.
Sketch: split $y - \hat f = \varepsilon + (f - \hat f)$, cross term dies since
$\mathbb{E}[\varepsilon]=0$ and is independent of $D$; then insert $\bar f$ into
$(f-\hat f)$, cross term dies by definition of $\bar f = \mathbb{E}_D[\hat f]$.

**Q8. Does more training data reduce bias or variance? Does regularization?**
More data: variance (the fit fluctuates less across samples; the family's systematic
error is unchanged). Regularization: reduces variance at the cost of some added bias.

**Q9. Why can't test MSE go below $\sigma^2$, and what does it mean if it does?**
$\sigma^2$ is the label noise, present in the decomposition regardless of the model,
so no model beats it in expectation. A small test set can dip slightly below it by
chance; beating it clearly on a large test set signals a bug or leakage (e.g., a
feature derived from the label).

**Q10. What is double descent?**
Past the interpolation threshold (model exactly fits training data), test error can
*decrease* again with more capacity: peak at the threshold (a single, wild
interpolant), then a second descent because among many interpolating solutions,
SGD / min-norm fitting implicitly selects smooth ones - variance falls again. The
decomposition still holds; the "capacity ⇒ variance" intuition is what breaks.

**Q11. Name three forms of data leakage with examples.**
(1) Preprocessing before splitting - scaling/feature-selection/PCA fit on the full
data, so training folds see test statistics. (2) Target leakage - a feature caused by
the label, e.g. `took_antibiotics` when predicting pneumonia. (3) Temporal leakage  - 
random splits on time series let the model train on the future; also features
aggregated over post-prediction data. (Bonus: grouped leakage - same user in train
and test.)

**Q12. Why nested CV? What goes wrong with plain CV for hyperparameter tuning?**
If you report the CV score of the best hyperparameter, you've selected the maximum of
noisy estimates - optimistic bias (winner's curse). Nested CV: inner loop selects
hyperparameters, outer loop scores models on folds never used for any selection.

**Q13. Model A: 84.7%, model B: 84.2% on the same test set. Ship A?**
Not on this evidence. Use a *paired* comparison (same examples): McNemar's test or
paired differences - pairing removes shared example-difficulty variance. For neural
nets, also rerun with several seeds; seed variance alone often exceeds 0.5%.

**Q14. Derive what a class weight does to the gradient.**
Weighted CE: $L_i = w_{y_i}\ell_i$, so $\partial L_i/\partial z_i = w_{y_i}(\hat p_i - y_i)$ - each minority example pushes $w_1\times$ harder, equivalent in expectation to
oversampling by $w_1$. Side effect: predicted probabilities inflate to
$\frac{w_1\pi}{w_1\pi + w_0(1-\pi)}$ - calibration is destroyed; recalibrate or move
the threshold instead.

**Q15. When is threshold moving better than resampling?**
When the base model's probabilities are fine and only the *decision* is miscosted.
Train on natural data (keeps calibration), set $t^* = c_{FP}/(c_{FP}+c_{FN})$ from
the cost matrix. Resampling/reweighting changes the learned probabilities themselves - useful when the minority class is so rare the model barely learns its features, but
it needs recalibration afterwards.

**Q16. Prove the constant minimizing MAE is the median.**
$h(c)=\mathbb{E}|y-c|$; $h'(c) = \Pr(y<c) - \Pr(y>c)$ (each unit of $c$ increase adds
$+1$ per point below $c$, $-1$ per point above). Zero crossing where both
probabilities are $\tfrac12$: the median. For MSE, $g'(c) = -2\mathbb{E}[y-c] = 0$
gives the mean.

**Q17. Can R² be negative?**
Yes, on held-out data: $R^2 = 1 - \text{SS}_{res}/\text{SS}_{tot}$ is negative
whenever the model does worse than predicting the training-mean constant. Range is
$(-\infty, 1]$, not $[0, 1]$.

**Q18. Your offline AUC improved 2% but the A/B test is flat. Explain.**
Offline metrics are proxies. Possible causes: the ranking improved on pairs users
never encounter; distribution shift between logged data and live traffic; the offline
label (click) diverges from the online goal (conversion/retention); feedback loops  - 
logged data was collected under the old model's policy. Trust the experiment; audit
the proxy.

**Q19. Why do you randomize by user rather than by request in an A/B test, and what is peeking?**
Per-request randomization gives one user a mixture of both experiences - carryover
contaminates both arms and violates unit independence. Peeking = repeatedly testing
significance as data accrues and stopping at the first $p<0.05$; under repeated looks
the false-positive rate far exceeds 5%. Fix: fixed-horizon analysis or sequential
testing methods.

**Q20. Why does the bias–variance decomposition not carry over cleanly to 0–1 loss?**
The derivation relies on the algebra of squared error (cross terms vanishing under
expectation). 0–1 loss isn't a squared distance; there's no exact additive split, and
"variance" can even *help*: if the average model is wrong at a point (biased), a
high-variance model sometimes lands on the correct label. Decompositions exist
(e.g. Domingos 2000) but with interaction terms and sign flips - say that, don't
pretend the clean identity holds.

**Q21. Why is evaluating a text generator fundamentally harder than evaluating a classifier?**
A classifier has one ground truth per example; a generator has an enormous set of
equally valid outputs, and a "reference" is just one sample from that set. So every
metric is a compromise: overlap-with-reference (BLEU/ROUGE) punishes valid answers for
being different; model probability (perplexity) measures fit to a corpus, not sampled
output quality; judgment (human/LLM) targets the real question but is expensive and
biased. Strong answers name which compromise a metric makes and how it fails.

**Q22. Your summarizer's ROUGE improved but users say summaries got worse. Possible?**
Entirely. ROUGE is n-gram recall against a reference: it rises for extractive copying
of reference wording and cannot see factuality ("profits fell" vs "profits rose"
shares most n-grams while inverting the claim), salience, or paraphrase (a better
summary in different words scores *lower*). It's a surface-overlap regression alarm,
weakly correlated at system level, bad per-example - check faithfulness and
preference directly.

**Q23. Why is perplexity a poor headline metric for chatbot quality?**
PPL is $e^{\text{mean NLL}}$ on a fixed corpus: how well the distribution fits that
text. It's tokenizer-dependent (only comparable with the same tokenizer), ignores
decoding (it scores probability on references, not what you actually sample), and is
misaligned with usefulness - instruction-tuned models often have *worse* web-text PPL
than their base models while being clearly better assistants. Use it for pretraining
progress; use preference/task evals for quality.

**Q24. You're using GPT-x to judge two models' outputs. What biases and mitigations?**
Position bias (first answer favored - judge both orders, keep consistent verdicts),
verbosity bias (longer wins - length-control the comparison, rubric that penalizes
padding), self-preference (judges favor their own family - use a judge from another
family or a panel). Plus the meta-mitigation: the judge is a proxy metric, so validate
it against a human-labeled sample (agreement rate) before trusting its deltas - same
discipline as any offline proxy (§9, §10).

## Exit checklist

- [ ] Cold, on paper: fill a confusion matrix, compute precision/recall/F1, and say who pays for each error type - 4 min
- [ ] Build an ROC curve by hand from 6 scored examples and state AUC's probabilistic meaning - 5 min
- [ ] Explain why ROC flatters on imbalanced data and when precision-recall curves are the right view - 3 min
- [ ] Derive the full bias-variance decomposition of expected squared error - 5 min
- [ ] Name the classic leakage sins (fit-before-split, temporal, group, target) and what nested CV fixes - 3 min
- [ ] Decide via standard error whether model A at 90.5% beats model B at 90% on n=1000 - 3 min
- [ ] LLM-as-judge: name the three biases with mitigations, then walk the summarization-eval answer end to end - 5 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
