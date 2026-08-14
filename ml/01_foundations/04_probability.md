# Probability & statistics for ML

## 3.1 Random variables, expectation, variance

A **random variable** is a number attached to a random outcome: a die roll is
$X \in \{1..6\}$, each value with probability $1/6$. The **expectation** is
the probability-weighted average - the long-run mean if you repeat forever:

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

#### Linearity

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

This holds *even when $X$ and $Y$ are dependent*, because both sides are the
same weighted sum with its terms grouped differently. It is what makes many
ML proofs one-liners.

Products are different. In general $\mathbb{E}[XY] \neq
\mathbb{E}[X]\,\mathbb{E}[Y]$; equality needs **independence**
($P(X = x, Y = y) = P(X = x)\,P(Y = y)$ for all $x, y$). Derivation in one
line - the joint probability factors, so the double sum splits:

$$\mathbb{E}[XY] = \sum_{x,y} x\,y\,P(x)\,P(y) = \Big(\sum_x x\,P(x)\Big)\Big(\sum_y y\,P(y)\Big) = \mathbb{E}[X]\,\mathbb{E}[Y]$$

#### Variance

**Variance** is the expected squared distance from the mean, the standard
measure of spread ($\mu$ is shorthand for $\mathbb{E}[X]$):

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mu^2$$

The identity: expand $(X-\mu)^2 = X^2 - 2\mu X + \mu^2$, apply linearity to
get $\mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2$, and use
$\mathbb{E}[X] = \mu$ to collapse the last two terms into $-\mu^2$. Die:
$\mathbb{E}[X^2] = 91/6 \approx 15.17$, so
$\mathrm{Var} = 15.17 - 3.5^2 = 15.17 - 12.25 \approx 2.92$.

#### Two scaling rules

**Scaling by a constant.** $\mathrm{Var}(aX) = \mathbb{E}[(aX - a\mu)^2] =
a^2\, \mathbb{E}[(X-\mu)^2] = a^2\, \mathrm{Var}(X)$. Variance carries
squared units, so the constant comes out squared.

**Adding independent variables.** Expand the definition for $X + Y$:

$$\mathrm{Var}(X + Y) = \mathbb{E}\big[((X - \mu_X) + (Y - \mu_Y))^2\big]
= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$$

The last term is the **covariance**. Under independence it is zero: multiply
out $(X-\mu_X)(Y-\mu_Y)$ and apply the product rule above -
$\mathbb{E}[XY] - \mu_X \mu_Y = 0$. So for independent variables,

$$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$

This additivity is the fact behind the $\sigma/\sqrt{n}$ error bars of §3.6
and behind variance-scaling weight initialization (Xavier/He: pick the weight
variance so each layer's outputs keep unit variance).

## 3.2 Bayes rule - derived, not memorized

Notation first: $A \cap B$ is the event "$A$ *and* $B$ both happen".
Conditional probability is a *definition*: restrict attention to the world
where $B$ happened, and renormalize:

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

*The two bars are everyone who tests positive, drawn to scale. The sick pool
is small (100 people), so the 495 false positives from the 9,900 healthy
people outnumber the 99 true positives five to one.*

## 3.3 MLE, and why minimizing cross-entropy IS maximum likelihood

#### The principle

**Maximum likelihood estimation**: given data and a model family $p_\theta$,
choose the $\theta$ that makes the observed data most probable. The notation
$\arg\max_\theta$ means "the value of $\theta$ at which the expression is
largest":

$$\hat\theta = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i \mid x_i)
             = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y_i \mid x_i)$$

The two argmaxes are equal because $\log$ is increasing, so it preserves
which input is largest. The log form is the one used in practice: it turns
the product into a sum, which is numerically stable and differentiable term
by term.

#### Cross-entropy is MLE for a classifier

A classifier with softmax output assigns probabilities
$p_\theta(k \mid x_i)$ to each class $k$. The negative log-likelihood of the data is:

$$-\sum_i \log p_\theta(y_i \mid x_i) = -\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)$$

where $y_{ik}$ is the **one-hot** label: $y_{ik} = 1$ if example $i$'s true
class is $k$, and $0$ otherwise. So the inner sum has exactly one surviving
term, the log-probability of the correct class - the two sides are the same
sum written two ways. The right-hand side is the definition of the
**cross-entropy loss**. So: **minimizing cross-entropy = maximizing the
log-probability the model assigns to the true labels.** Cross-entropy isn't
an arbitrary "good loss"; it's MLE for a categorical output.

#### The same pattern gives the regression losses

Model the target as prediction plus noise, $y = f_\theta(x) + \varepsilon$,
and take the negative log of the noise density:

- Gaussian noise, density $\frac{1}{\sqrt{2\pi\sigma^2}}
  e^{-\varepsilon^2 / 2\sigma^2}$:
  $-\log p(y \mid x) = \frac{(y - f_\theta(x))^2}{2\sigma^2} + \text{const}$,
  a squared error. MLE under Gaussian noise is **MSE**.
- Laplace noise, density $\frac{1}{2b} e^{-|\varepsilon| / b}$:
  $-\log p(y \mid x) = \frac{|y - f_\theta(x)|}{b} + \text{const}$, an
  absolute error. MLE under Laplace noise is **L1 loss**.

When an interviewer asks "why this loss?", the answer is: it is the negative
log-likelihood under this output/noise model.

#### The smallest concrete MLE: a coin

Coin with unknown heads probability $p$: observe $h$ heads in $n$ flips.

$$\log L(p) = h \log p + (n - h)\log(1-p)$$

Differentiate and set to zero: $\frac{h}{p} = \frac{n-h}{1-p}$, so
$\hat p = h/n$ - the empirical frequency. (Verified numerically in
`probability_demos.ipynb`.)

<div style="margin:1.25rem 0">
<svg viewBox="0 0 360 235" role="img" aria-label="Log likelihood of a coin's heads probability after seeing 7 heads in 10 flips: a single-peaked curve with its maximum at p equals 0.7" style="max-width:380px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="40" y1="200" x2="330" y2="200"/>
    <line x1="40" y1="215" x2="40" y2="30"/>
  </g>
  <polyline points="82,153 96,126.3 110,106.4 124,90.9 138,78.4 152,68.3 166,60.1 180,53.5 194,48.4 208,44.7 222,42.4 236,41.6 250,42.5 264,45.7 278,52.1 292,63.9 306,88.7" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <line x1="236" y1="41.6" x2="236" y2="200" stroke="#f59e0b" stroke-dasharray="4 4"/>
  <circle cx="236" cy="41.6" r="4" fill="#f59e0b"/>
  <text x="236" y="30" text-anchor="middle" fill="#f59e0b">max at h/n = 0.7</text>
  <text x="236" y="216" text-anchor="middle" fill="#f59e0b">0.7</text>
  <text x="40" y="216" text-anchor="middle" fill="currentColor" fill-opacity="0.7">0</text>
  <text x="180" y="216" text-anchor="middle" fill="currentColor" fill-opacity="0.7">0.5</text>
  <text x="320" y="216" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
  <text x="336" y="204" fill="currentColor" fill-opacity="0.7">p</text>
  <text x="48" y="40" fill="currentColor" fill-opacity="0.7">log L(p)</text>
</svg>
</div>

*The log-likelihood of every candidate $p$ after observing 7 heads in 10
flips: one peak, at $p = h/n = 0.7$ - the same value the
derivative-equals-zero computation picks out.*

## 3.4 Bernoulli and Gaussian - where they come from

**Bernoulli($p$)**: one binary event. $P(X = 1) = p$, $P(X = 0) = 1 - p$.
Its moments come straight from the definitions: $\mathbb{E}[X] = 1 \cdot p +
0 \cdot (1-p) = p$; since $X^2 = X$ for a 0/1 variable,
$\mathbb{E}[X^2] = p$ too, so $\mathrm{Var}(X) = p - p^2 = p(1-p)$ - largest
at $p = 1/2$, where the outcome is hardest to predict. Summing $n$
independent Bernoullis gives the **Binomial**. Every binary-classification
output and every dropout mask is a Bernoulli.

**Gaussian** $\mathcal{N}(\mu, \sigma^2)$, density $\frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. It is not chosen for convenience alone; it shows
up for three reasons:

1. **CLT** (§3.6): sums of many small independent effects are approximately
   Gaussian - which is why measurement noise so often is.
2. **Maximum entropy**: among all distributions with a given mean and
   variance, the Gaussian adds no structure beyond those two constraints -
   so it is the default assumption when mean and variance are all you know.
3. Its $-\log$ density is quadratic - the bridge from "Gaussian noise" to
   "least-squares loss" (§3.3).

## 3.5 Bias and variance of estimators

An estimator $\hat\theta$ is a formula applied to a random sample, so it's a random
variable itself. Two failure modes:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

Bias: wrong *on average*, a systematic miss. Variance: right on average but
*jumpy* from sample to sample. They combine exactly. The derivation is one
move: inside $\mathbb{E}[(\hat\theta - \theta)^2]$, add and subtract
$\mathbb{E}[\hat\theta]$ and expand the square; the cross term is zero
because it contains the factor
$\mathbb{E}\big[\hat\theta - \mathbb{E}[\hat\theta]\big] = 0$:

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

#### Why the 1/n sample variance is biased (Bessel's correction)

The estimator $\frac{1}{n}\sum_i (x_i - \bar x)^2$ underestimates $\sigma^2$,
and the derivation shows by exactly how much. Start from an algebraic
identity (add and subtract $\bar x$ inside $(x_i - \mu)^2$ and expand; the
cross term vanishes because $\sum_i (x_i - \bar x) = 0$):

$$\sum_i (x_i - \bar x)^2 = \sum_i (x_i - \mu)^2 - n(\bar x - \mu)^2$$

Take expectations of both sides. The first term is $n\sigma^2$ by the
definition of variance. For the second: $\mathbb{E}[(\bar x - \mu)^2] =
\mathrm{Var}(\bar x)$, and by §3.1's two rules (variances add over the $n$
independent draws, then the $1/n$ comes out squared),
$\mathrm{Var}(\bar x) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$. So:

$$\mathbb{E}\Big[\sum_i (x_i - \bar x)^2\Big] = n\sigma^2 - \sigma^2 = (n-1)\,\sigma^2$$

Dividing by $n$ therefore gives $\frac{n-1}{n}\sigma^2$ on average - biased
low - while dividing by $n - 1$ gives exactly $\sigma^2$: **Bessel's
correction**. The source of the bias is visible in the identity: deviations
are measured from $\bar x$, which was computed from the same sample, and the
$-n(\bar x - \mu)^2$ term is the price of that double use.

#### The tradeoff in ML terms

A linear model is high-bias/low-variance: it misses curvature systematically,
but refitting on a new sample gives nearly the same line. A deep net or
unpruned tree is low-bias/high-variance: it can fit anything, including this
particular sample's noise. Regularization trades variance down for some bias;
more data lowers variance directly; bagging averages several high-variance
fits to cancel their variance. Test error decomposes as
$\text{Bias}^2 + \text{Variance} + \text{irreducible noise}$ (derived in the
evaluation module).

## 3.6 Central limit theorem - the $\sqrt{n}$ intuition

Take $n$ **i.i.d.** draws - independent and identically distributed, i.e.
sampled from the same distribution with mean $\mu$ and variance $\sigma^2$,
each draw independent of the rest - and average them:
$\bar X_n = \frac1n \sum X_i$. Two facts, both from §3.1:

- $\mathbb{E}[\bar X_n] = \mu$ (linearity) - unbiased.
- $\mathrm{Var}(\bar X_n) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$
  (independence → variances add; the $\frac1n$ comes out squared).

So the spread of the mean shrinks like $\sigma/\sqrt{n}$ - the **law of large
numbers** with a rate. To halve your error bar you need 4× the data. The CLT adds the
shape: not only does $\bar X_n$ concentrate, its distribution becomes **Gaussian**,
*regardless of the original distribution*:

$$\frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

Where the shape comes from: the distribution of a sum is the **convolution**
of the summands' distributions, and each convolution smooths the result -
after many of them, distribution-specific features (skew, bumps) shrink
toward zero while the normalization holds mean and variance fixed, and the
limiting shape is the Gaussian. Concretely: start from a distribution as
lopsided as the exponential; means of 100 draws are already visibly symmetric
and bell-shaped. `probability_demos.ipynb` tracks this numerically - the
skewness of $\bar X_n$ falls toward 0 and its standard deviation matches
$\sigma/\sqrt{n}$.

Why care in ML:

- Minibatch gradients are sample means of per-example gradients, so SGD noise
  shrinks like $1/\sqrt{\text{batch size}}$.
- Test-set accuracy is a sample mean of 0/1 correctness (a Bernoulli), so its
  standard error is $\sqrt{p(1-p)/n}$. With $n = 1000$ and accuracy near
  90%, that is $\sqrt{0.9 \cdot 0.1 / 1000} \approx 0.9\%$ - a "0.5%
  improvement" on such a test set is within the noise.
- A/B tests and confidence intervals lean on the CLT's Gaussian shape.
