# Probability & statistics for ML

## 3.1 Random variables, expectation, variance - from first principles

A **random variable** is just a number attached to a random outcome: die roll $X \in
\{1..6\}$, each with probability $1/6$. The **expectation** is the probability-weighted
average - the long-run mean if you repeat forever:

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

Two properties you use constantly: **linearity**,
$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$, which holds *even when $X, Y$
are dependent* (it's just rearranging a weighted sum) - this is what makes so many ML
proofs one-liners. And $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ only under
independence.

**Variance** is the expected squared distance from the mean - the natural "spread":

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mu^2$$

(The identity: expand $(X-\mu)^2 = X^2 - 2\mu X + \mu^2$, apply linearity, note
$\mathbb{E}[X] = \mu$.) Die: $\mathbb{E}[X^2] = 91/6 \approx 15.17$, so
$\mathrm{Var} = 15.17 - 12.25 \approx 2.92$. Scaling: $\mathrm{Var}(aX) = a^2
\mathrm{Var}(X)$ (squared units!), and for **independent** variables variances add:
$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$. That last fact is the seed
of the CLT, of $1/\sqrt{n}$ error bars, and of every variance-scaling weight-init
scheme (Xavier/He: pick weight variance so activations keep unit variance layer to
layer).

## 3.2 Bayes rule - derived, not memorized

Conditional probability is a *definition*: restrict attention to the world where $B$
happened, and renormalize:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Write it twice: $P(A \cap B) = P(A \mid B) P(B)$ and also $P(A \cap B) = P(B \mid A)
P(A)$ - same quantity, two factorizations. Set them equal, divide by $P(B)$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

That's the whole derivation - two lines from the definition. The point of Bayes rule is
that it converts **likelihood** ("probability of the evidence given the hypothesis",
often easy to model) into the **posterior** ("probability of the hypothesis given the
evidence", the thing you actually want), weighted by the **prior** $P(A)$.

The classic trap (run it with raw counts, as in `probability_demos.ipynb`): a disease has
prevalence 1%, the test has 99% sensitivity and 95% specificity. You test positive -
what's the chance you're sick? Take 10,000 people: 100 sick → 99 true positives;
9,900 healthy → 5% false positive rate → 495 false positives. Positive tests: $99 +
495 = 594$; sick among them: $99 / 594 \approx 16.7\%$. Not 99%. The prior (rarity of
the disease) dominates because false positives come from a *huge* pool. In ML terms:
this is precision under class imbalance - exactly why accuracy is a useless metric for
rare-event detection and why a 99%-accurate fraud model can be garbage.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 130" role="img" aria-label="Among 10,000 people the positive tests split into 99 true positives from the sick group and 495 false positives from the healthy group, so a positive test means only about 17 percent chance of being sick" style="max-width:430px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="150" y="42" text-anchor="end" fill="currentColor" fill-opacity="0.8">sick &amp; test +</text>
  <rect x="160" y="28" width="35" height="20" fill="#10b981" fill-opacity="0.85"/>
  <text x="201" y="42" fill="currentColor">99</text>
  <text x="150" y="76" text-anchor="end" fill="currentColor" fill-opacity="0.8">healthy &amp; test +</text>
  <rect x="160" y="62" width="173" height="20" fill="#f43f5e" fill-opacity="0.8"/>
  <text x="339" y="76" fill="currentColor">495</text>
  <text x="160" y="112" fill="currentColor" font-weight="600">P(sick | +) = 99 / 594 &#8776; 17%</text>
</svg>
</div>

*The two bars are everyone who tests positive, drawn to scale. The sick pool is
tiny (100 people), so even a 5% false-positive rate on the 9,900 healthy people
buries the 99 true positives.*

## 3.3 MLE, and why minimizing cross-entropy IS maximum likelihood

**Maximum likelihood estimation**: given data and a model family $p_\theta$, choose
the $\theta$ that makes the observed data most probable:

$$\hat\theta = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i \mid x_i)
             = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y_i \mid x_i)$$

(Log is monotonic so the argmax is unchanged; it turns the product into a sum -
numerically stable and differentiable term by term.)

Now the key step. A classifier with softmax output assigns probabilities
$p_\theta(k \mid x_i)$ to each class. The negative log-likelihood of the data is:

$$-\sum_i \log p_\theta(y_i \mid x_i) = -\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)$$

where $y_{ik}$ is the one-hot label (the inner sum has one surviving term - the
correct class). The right-hand side is *literally the definition of the cross-entropy
loss*. So: **minimizing cross-entropy = maximizing the log-probability the model
assigns to the true labels.** Cross-entropy isn't an arbitrary "good loss"; it's MLE
for a categorical output. Same story elsewhere: MSE is MLE under a Gaussian noise
model ($-\log$ of a Gaussian density is a squared error plus constants), and L1 loss
is MLE under Laplace noise. When an interviewer asks "why this loss?", the deep answer
is always "it's the negative log-likelihood under this noise/output model."

Simplest concrete MLE (verified numerically in `probability_demos.ipynb`): coin with
unknown $p$, observe $h$ heads in $n$ flips. $\log L(p) = h \log p + (n - h)\log(1-p)$.
Differentiate, set to zero: $h/p = (n-h)/(1-p) \Rightarrow \hat p = h/n$. The empirical
frequency - MLE formalizes the obvious.

## 3.4 Bernoulli and Gaussian - where they come from

**Bernoulli($p$)**: the atom of randomness - one binary event. $P(X=1) = p$,
$\mathbb{E}[X] = p$, $\mathrm{Var} = p(1-p)$ (maximal at $p = 1/2$: a fair coin is the
most unpredictable coin). Sum $n$ independent Bernoullis → **Binomial**. Every
binary-classification output and every dropout mask is a Bernoulli.

**Gaussian** $\mathcal{N}(\mu, \sigma^2)$, density $\frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. It is not chosen for convenience alone; it *emerges*:

1. **CLT** (next section): sums of many small independent effects are Gaussian -
   which is why measurement noise so often is.
2. **Maximum entropy**: among all distributions with a given mean and variance, the
   Gaussian assumes the least extra structure - the honest default.
3. Its $-\log$ density is quadratic - the bridge from "Gaussian noise" to
   "least-squares loss" (§3.3).

## 3.5 Bias and variance of estimators

An estimator $\hat\theta$ is a formula applied to a random sample, so it's a random
variable itself. Two failure modes:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

Bias: wrong *on average*, a systematic miss. Variance: right on average but
*jumpy* from sample to sample. They combine exactly. The derivation is one move:
inside $\mathbb{E}[(\hat\theta - \theta)^2]$, add and subtract
$\mathbb{E}[\hat\theta]$, expand the square, and the cross term dies because it
contains the factor $\mathbb{E}\big[\hat\theta - \mathbb{E}[\hat\theta]\big] = 0$:

$$\mathbb{E}[(\hat\theta - \theta)^2] = \mathrm{Bias}^2 + \mathrm{Var}$$

<div style="margin:1.25rem 0">
<svg viewBox="0 0 320 220" role="img" aria-label="Four dartboards showing the combinations of low and high bias with low and high variance" style="max-width:340px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="90" y="28" text-anchor="middle" fill="currentColor" fill-opacity="0.7">low variance</text>
  <text x="230" y="28" text-anchor="middle" fill="currentColor" fill-opacity="0.7">high variance</text>
  <text x="30" y="74" text-anchor="middle" fill="currentColor" fill-opacity="0.7">low bias</text>
  <text x="30" y="174" text-anchor="middle" fill="currentColor" fill-opacity="0.7">high bias</text>
  <g fill="none" stroke="currentColor">
    <circle cx="90" cy="70" r="26" stroke-opacity="0.2"/><circle cx="90" cy="70" r="17" stroke-opacity="0.35"/><circle cx="90" cy="70" r="8" stroke-opacity="0.5"/>
    <circle cx="230" cy="70" r="26" stroke-opacity="0.2"/><circle cx="230" cy="70" r="17" stroke-opacity="0.35"/><circle cx="230" cy="70" r="8" stroke-opacity="0.5"/>
    <circle cx="90" cy="170" r="26" stroke-opacity="0.2"/><circle cx="90" cy="170" r="17" stroke-opacity="0.35"/><circle cx="90" cy="170" r="8" stroke-opacity="0.5"/>
    <circle cx="230" cy="170" r="26" stroke-opacity="0.2"/><circle cx="230" cy="170" r="17" stroke-opacity="0.35"/><circle cx="230" cy="170" r="8" stroke-opacity="0.5"/>
  </g>
  <g fill="#6366f1">
    <circle cx="92" cy="67" r="2.5"/><circle cx="86" cy="72" r="2.5"/><circle cx="93" cy="74" r="2.5"/><circle cx="88" cy="66" r="2.5"/><circle cx="90" cy="71" r="2.5"/>
    <circle cx="240" cy="58" r="2.5"/><circle cx="216" cy="76" r="2.5"/><circle cx="236" cy="84" r="2.5"/><circle cx="222" cy="60" r="2.5"/><circle cx="244" cy="74" r="2.5"/>
    <circle cx="105" cy="154" r="2.5"/><circle cx="99" cy="159" r="2.5"/><circle cx="106" cy="161" r="2.5"/><circle cx="101" cy="153" r="2.5"/><circle cx="103" cy="158" r="2.5"/>
    <circle cx="251" cy="146" r="2.5"/><circle cx="230" cy="164" r="2.5"/><circle cx="248" cy="170" r="2.5"/><circle cx="234" cy="148" r="2.5"/><circle cx="254" cy="160" r="2.5"/>
  </g>
</svg>
</div>

*Each dot is the estimate you'd get from one resampled dataset. Bias is how far
the cloud's center sits from the bullseye; variance is how scattered the cloud
is. The decomposition above says total error is exactly those two plus noise.*

Canonical example: sample variance with $\frac{1}{n}$ is biased low - the sample mean
$\bar x$ chases the data, so deviations from $\bar x$ are systematically smaller than
deviations from the true $\mu$; the exact correction factor is $\frac{n-1}{n}$, fixed
by **Bessel's correction** $\frac{1}{n-1}$. The ML version of the tradeoff: a linear
model is high-bias/low-variance (systematically misses curvature, but stable across
datasets); a deep net or unpruned tree is low-bias/high-variance (can fit anything,
including this particular sample's noise). Regularization, more data, and ensembling
(bagging averages away variance) are all moves on this dial. Test error decomposes as
$\text{Bias}^2 + \text{Variance} + \text{irreducible noise}$.

## 3.6 Central limit theorem - the $\sqrt{n}$ intuition

Average $n$ i.i.d. draws with mean $\mu$, variance $\sigma^2$:
$\bar X_n = \frac1n \sum X_i$. Two facts, both from §3.1:

- $\mathbb{E}[\bar X_n] = \mu$ (linearity) - unbiased.
- $\mathrm{Var}(\bar X_n) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$
  (independence → variances add; the $\frac1n$ comes out squared).

So the spread of the mean shrinks like $\sigma/\sqrt{n}$ - the **law of large
numbers** with a rate. To halve your error bar you need 4× the data. The CLT adds the
shape: not only does $\bar X_n$ concentrate, its distribution becomes **Gaussian**,
*regardless of the original distribution*:

$$\frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

Intuition for the shape: summing convolves distributions, and convolution repeatedly
smooths - idiosyncratic features (skew, bumps) wash out, and the Gaussian is the fixed
point of this smoothing (only skew/kurtosis die slowly; mean and variance are
preserved by design of the normalization). Start with something as lopsided as an
exponential distribution: means of 100 draws are already visibly symmetric and bell-
shaped (see `probability_demos.ipynb`, which tracks the skewness → 0 and std →
$\sigma/\sqrt{n}$ numerically).

Why care in ML: minibatch gradients are sample means of per-example gradients (SGD
noise is $\propto 1/\sqrt{\text{batch size}}$); test-set accuracy is a sample mean, so
its error bar is $\sqrt{p(1-p)/n}$ - with $n = 1000$ test examples, accuracy has a
±1.5% standard error, i.e. a "0.5% improvement" is pure noise; A/B tests and
confidence intervals all lean on CLT normality.

