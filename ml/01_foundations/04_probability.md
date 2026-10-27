# Probability & statistics for ML

This chapter answers three questions that come up constantly in ML:

- **Why is the loss what it is?** Cross-entropy, MSE and L1 each follow from
  an assumption about how the labels are generated. Know the assumption and
  you know when the loss is the wrong choice (§3.4).
- **Is this improvement real?** Model B beats model A by 0.5% on 1,000 test
  examples. Accuracy on 1,000 examples varies by about ±0.9% just from which
  examples happened to be in the test set, so a 0.5% gap can be luck (§3.6,
  §3.8).
- **Why does my model miss?** It is too simple for the pattern (bias), or it
  fits the quirks of its particular training set (variance). Regularization,
  more data and bagging each reduce one of the two (§3.7).

It builds in three parts, each using only what came before:

1. **Describing uncertainty** (§3.1–3.2): averages and spread, and the two
   distributions ML uses most.
2. **Learning from data** (§3.3–3.5): Bayes rule, maximum likelihood (where
   the losses come from), and what a loss value means.
3. **Trusting numbers computed from data** (§3.6–3.8): error bars, bias and
   variance, and whether a difference is real.

Every idea comes with a few lines of NumPy you can run. The snippets assume
`import numpy as np`.

## 3.1 Random variables, expectation, variance

A **random variable** is a number that comes out of a random process. Roll a
die and you get $X \in \{1, \dots, 6\}$, each face with probability $1/6$.

Two numbers summarize it. The **expectation** (or mean) is where it lands on
average. The **variance** is how far it typically lands from there.

```python
faces = np.arange(1, 7)                 # 1, 2, ..., 6
p = np.full(6, 1/6)                     # each face equally likely

mean = (faces * p).sum()                # 3.5
var  = ((faces - mean)**2 * p).sum()    # 2.92

rolls = np.random.randint(1, 7, size=100_000)
rolls.mean(), rolls.var()               # ≈ 3.5, ≈ 2.92
```

The first two lines compute the exact values from the probabilities. The
last two simulate 100,000 rolls and get nearly the same numbers.

#### Expectation

**The expectation is each value times its probability, summed:** the
long-run average.

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

**Linearity: the average of a sum is the sum of the averages.**

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

This always holds, even when $X$ and $Y$ depend on each other, because both
sides are the same weighted sum with the terms grouped differently. Two dice
add to 7 on average, $3.5 + 3.5$. Most results in this chapter are one
application of this rule.

Products are different. $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$ only
when $X$ and $Y$ are **independent**: knowing one tells you nothing about
the other, $P(X = x, Y = y) = P(X = x)\,P(Y = y)$. Then the double sum splits:

$$\mathbb{E}[XY] = \sum_{x,y} x\,y\,P(x)\,P(y) = \Big(\sum_x x\,P(x)\Big)\Big(\sum_y y\,P(y)\Big) = \mathbb{E}[X]\,\mathbb{E}[Y]$$

#### Variance

**Variance is the expected squared distance from the true mean**
$\mu = \mathbb{E}[X]$:

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]$$

"Mean" and "average" are the same operation applied to two different things:

- The **true mean** averages over every possible outcome, weighted by its
  probability: 3.5 for the die. It is one fixed number.
- The **sample mean** averages the draws you actually got: rolls $2, 6, 3$
  give $3.67$. Another set of rolls gives another number.

$\mathbb{E}[\,\cdot\,]$ always means the first kind. Estimating a variance
from an actual sample is a different problem, with a twist of its own
(§3.7).

A shortcut for computing it: expand the square and use linearity.

$$\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2 = \tfrac{91}{6} - 3.5^2 \approx 2.92 \quad \text{for the die}$$

Two rules for how variance behaves:

- **Scaling: $\mathrm{Var}(aX) = a^2\,\mathrm{Var}(X)$.** Doubling $X$
  doubles every distance from the mean, so it quadruples every squared
  distance.
- **Adding independent variables: $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$.**
  Expanding $\big((X - \mu_X) + (Y - \mu_Y)\big)^2$ gives the two variances
  plus a cross term, $2\,\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$, called the
  **covariance** term. For independent variables the product rule above
  makes it 0. Two dice: $2.92 + 2.92 = 5.83$.

The second rule is behind the $\sigma/\sqrt{n}$ error bars of §3.6 and behind
weight initialization schemes like Xavier and He, which pick the weight
variance so each layer's outputs keep the same variance as its inputs.

## 3.2 Bernoulli and Gaussian: the two distributions ML leans on

Two distributions cover most of what ML models:

- **One yes/no event** (a coin flip, a binary label, a dropout mask):
  **Bernoulli**.
- **A quantity built from many small effects** (measurement noise, how far a
  prediction misses, an average): **Gaussian**.

§3.4 uses both to derive the standard losses.

#### Bernoulli: one yes/no event

**A Bernoulli($p$) variable is 1 with probability $p$ and 0 otherwise.** One
draw is one coin flip.

```python
flips = (np.random.rand(100_000) < 0.3).astype(float)   # Bernoulli(0.3)
flips.mean(), flips.var()                               # ≈ 0.3, ≈ 0.21
```

The mean is $p$: $1 \cdot p + 0 \cdot (1 - p) = p$. The variance is
$p(1 - p)$, here $0.3 \times 0.7 = 0.21$. (A 0/1 variable has $X^2 = X$, so
$\mathbb{E}[X^2] = p$ and the shortcut gives $p - p^2$.)

<div style="margin:1.25rem 0">
<svg viewBox="0 0 380 200" role="img" aria-label="The Bernoulli variance p times one minus p plotted against p: a parabola that is zero at p equals 0 and p equals 1 and peaks at one quarter when p equals one half" style="max-width:400px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="180" y="22" text-anchor="middle" fill="#6366f1" font-weight="600">Var(X) = p(1 − p)</text>
  <line x1="40" y1="150" x2="332" y2="150" stroke="currentColor" opacity="0.5"/>
  <line x1="40" y1="150" x2="40" y2="30" stroke="currentColor" opacity="0.5"/>
  <line x1="180" y1="150" x2="180" y2="40" stroke="currentColor" opacity="0.35" stroke-dasharray="4 4"/>
  <line x1="40" y1="40" x2="180" y2="40" stroke="currentColor" opacity="0.35" stroke-dasharray="4 4"/>
  <path d="M 40 150 Q 180 -70 320 150" fill="none" stroke="#6366f1" stroke-width="2"/>
  <circle cx="180" cy="40" r="3.5" fill="#6366f1"/>
  <circle cx="40" cy="150" r="3" fill="#f43f5e"/>
  <circle cx="320" cy="150" r="3" fill="#f43f5e"/>
  <text x="40" y="168" text-anchor="middle" fill="currentColor">0</text>
  <text x="180" y="168" text-anchor="middle" fill="currentColor">1/2</text>
  <text x="320" y="168" text-anchor="middle" fill="currentColor">1</text>
  <text x="344" y="154" fill="currentColor">p</text>
  <text x="34" y="44" text-anchor="end" fill="currentColor">1/4</text>
  <text x="186" y="100" fill="#6366f1" font-size="10.5">fair coin</text>
  <text x="52" y="140" fill="#f43f5e" font-size="10.5">nearly certain</text>
  <text x="308" y="140" text-anchor="end" fill="#f43f5e" font-size="10.5">nearly certain</text>
</svg>
</div>

*The variance of one Bernoulli($p$) draw as $p$ goes from 0 to 1: zero at
both ends, maximum $1/4$ at $p = 1/2$.*

A fair coin ($p = 1/2$) is the hardest to predict and has the largest
variance; a nearly certain outcome barely varies.

Three things built from it:

- **Counting successes: the Binomial.** The number of heads in $n$ flips is
  a sum of $n$ Bernoullis, so it has mean $np$ (linearity) and variance
  $np(1 - p)$ (independent variances add).
- **Binary labels.** A classifier that outputs "spam with probability 0.9"
  is modeling the label as Bernoulli($0.9$). §3.4 turns this into the binary
  cross-entropy loss.
- **Dropout.** Each unit is kept with probability $p$, so a layer of $n$
  units keeps $np$ on average. (PyTorch quotes the drop probability:
  `nn.Dropout(0.1)` keeps each unit with $p = 0.9$.)

#### Gaussian: the sum of many small effects

**The Gaussian $\mathcal{N}(\mu, \sigma^2)$ is the bell curve**, centered at
the mean $\mu$, with width set by the standard deviation $\sigma$.

Where the bell comes from: add up many small random effects. Here is the
error of a house-price prediction as 20 small independent pushes of up to
±1k each (the floor, the condition, the buyer's mood, ...):

```python
np.random.seed(0)
pushes = np.random.uniform(-1, 1, size=(100_000, 20))   # 100k houses
errors = pushes.sum(axis=1)                             # each house's total miss
np.histogram(errors, bins=9, range=(-9, 9))[0]          # counts, -9k to +9k
# [  309  2324  9546 22638 30192 22586  9784  2324   267]
```

Each push is flat, any value from −1k to 1k equally likely, but the totals
pile up in the middle and thin out at the edges. A total near zero only
needs the pushes to roughly cancel, which many combinations do; a total of
+9k needs nearly all 20 to push the same way. §3.6 shows this happens for
almost any kind of push: the **central limit theorem**.

The bell's formula:

$$\frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(x-\mu)^2 / 2\sigma^2}$$

The shape comes from the factor $e^{-(x-\mu)^2 / 2\sigma^2}$. It is 1 at the
mean, the same on both sides, and falls off fast. With $\sigma = 10$, a value
10 from the mean gets factor $e^{-0.5} = 0.61$, and a value 30 away gets
$e^{-4.5} = 0.011$. The constant in front only scales the curve so its total
area is 1.

Three reasons it is the default:

1. **Noise usually is a sum of many small effects**, as above, so a bell
   often fits reality.
2. **It assumes the least.** For a given mean and variance it is the most
   spread out, least committal distribution (§3.5 makes this precise).
3. **Its $-\log$ is a squared distance:**

   $$-\log\Big(\tfrac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(x-\mu)^2 / 2\sigma^2}\Big)
   = \frac{(x-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi\sigma^2)$$

   §3.4 uses this to show that assuming Gaussian errors gives least squares.

## 3.3 Bayes rule - derived, not memorized

**Bayes rule works backward from what you can measure to what you want to
know.** A lab can measure how often a test is positive for people known to be
sick. A patient wants the reverse: how often a positive result means sick.
These are very different numbers, and confusing them is the classic mistake.

#### The rare-disease test

A disease affects 1 person in 100. A test for it is good but not perfect:

- If you **are sick**, it comes back positive 99% of the time.
- If you **are healthy**, it still comes back positive 5% of the time, a
  false alarm.

These are two separate numbers, one for each group. Missing a sick person
and flagging a healthy one have different causes, so a test can be very good
at one and only fairly good at the other.

You test positive. What is the chance you are sick? Most people guess about
99%. Count through 10,000 people instead:

```python
sick, healthy = 100, 9_900
true_pos  = 0.99 * sick              # 99 sick people test positive
false_pos = 0.05 * healthy           # 495 healthy people test positive
true_pos / (true_pos + false_pos)    # 0.167
```

About 17%. The test catches nearly every sick person, but healthy people are
so much more common (9,900 vs 100) that even a 5% false-alarm rate produces
five false alarms for every real case.

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

*The two bars are everyone who tests positive, drawn to scale: 99 sick, 495
healthy.*

The three starting numbers have standard names. One describes the people
being tested; two describe the test:

| Name | Value | Describes |
|---|---|---|
| **Prevalence** | 1% | the people: how common the disease is among them |
| **Sensitivity** | 99% | the test: the share of sick people it flags |
| **Specificity** | 95% | the test: the share of healthy people it clears (so 5% false alarms) |

In ML terms this is precision under class imbalance. It is why accuracy is
useless for rare events: a 99%-accurate fraud model can still flag mostly
innocent transactions. (`probability_demos.ipynb` runs it with raw counts.)

#### The rule

The code above is Bayes rule; the formula only adds notation.
$P(\text{sick} \mid +)$, read "probability of sick given positive", means:
among the people who test positive, the fraction who are sick. That is
exactly what the code computed:

$$P(\text{sick} \mid +) = \frac{\text{sick and positive}}{\text{all positive}}$$

Write each count with the starting numbers:

- **Sick and positive:** $P(+ \mid \text{sick})\, P(\text{sick}) = 0.99 \times 0.01 = 0.0099$
- **All positive:** both ways of testing positive,
  $P(+) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0594$

Divide:

$$P(\text{sick} \mid +) = \frac{P(+ \mid \text{sick})\; P(\text{sick})}{P(+)} = \frac{0.0099}{0.0594} \approx 16.7\%$$

In general, for a hypothesis $A$ and evidence $B$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

- **Prior** $P(A)$: belief before the evidence. Here the prevalence, 1%.
- **Likelihood** $P(B \mid A)$: how often this evidence shows up when the
  hypothesis is true. Here the sensitivity, 99%.
- **Posterior** $P(A \mid B)$: belief after the evidence, the number you
  want. Here 17%.
- **Evidence** $P(B)$: how often the evidence shows up at all. Here 5.94%.

**Why the rule is needed:** the test's numbers (the likelihoods) are the same
wherever it is used, but the answer also depends on who is being tested. The
same test on people who already have symptoms, where say 10% are sick:

$$P(\text{sick} \mid +) = \frac{0.99 \times 0.10}{0.99 \times 0.10 + 0.05 \times 0.90} \approx 69\%$$

Same test, same positive result: 17% in the general population, 69% among
people with symptoms.

#### Today's posterior is tomorrow's prior

**Each new piece of evidence updates the current belief, starting from
wherever the last update left it.** The positive patient takes the test
again and is positive again. Writing $+_2$ for "the second test is
positive":

$$P(\text{sick} \mid +_2) = \frac{P(+_2 \mid \text{sick})\, P(\text{sick})}{P(+_2 \mid \text{sick})\, P(\text{sick}) + P(+_2 \mid \text{healthy})\, P(\text{healthy})}$$

The prior is now the first test's posterior, $P(\text{sick}) = 1/6$ (that
is, $99/594$), so $P(\text{healthy}) = 5/6$. The test's numbers are
unchanged, 0.99 and 0.05:

$$P(\text{sick} \mid +_2) = \frac{0.99 \times \tfrac{1}{6}}{0.99 \times \tfrac{1}{6} + 0.05 \times \tfrac{5}{6}} = \frac{0.165}{0.165 + 0.042} \approx 80\%$$

The first test moved the probability from 1% to 17%; the second, starting
from 17%, moved it to 80%.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 240" role="img" aria-label="Three bars showing the probability of being sick rising from 1 percent before any test to 17 percent after one positive test to 80 percent after a second positive test, with the population counts under each bar" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="ch-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.4">
    <rect x="60" y="40" width="44" height="130" rx="3"/>
    <rect x="230" y="40" width="44" height="130" rx="3"/>
    <rect x="400" y="40" width="44" height="130" rx="3"/>
  </g>
  <g fill="#10b981">
    <rect x="60" y="168" width="44" height="2"/>
    <rect x="230" y="148" width="44" height="22"/>
    <rect x="400" y="66" width="44" height="104"/>
  </g>
  <g fill="currentColor" text-anchor="middle" font-weight="600">
    <text x="82" y="30">1%</text>
    <text x="252" y="30">17%</text>
    <text x="422" y="30">80%</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.6">
    <line x1="112" y1="105" x2="222" y2="105" marker-end="url(#ch-f)"/>
    <line x1="282" y1="105" x2="392" y2="105" marker-end="url(#ch-f)"/>
  </g>
  <g fill="currentColor" fill-opacity="0.7" text-anchor="middle" font-size="10.5">
    <text x="167" y="96">test 1: positive</text>
    <text x="337" y="96">test 2: positive</text>
  </g>
  <g fill="currentColor" fill-opacity="0.8" text-anchor="middle" font-size="10.5">
    <text x="82" y="190">before any test</text>
    <text x="252" y="190">after test 1</text>
    <text x="422" y="190">after test 2</text>
    <text x="82" y="206">P(sick) = 100/10,000</text>
    <text x="252" y="206">P(sick | 1st +) = 99/594</text>
    <text x="422" y="206">P(sick | both +) = 98/123</text>
  </g>
  <g fill="#6366f1" text-anchor="middle" font-size="10">
    <text x="82" y="224">prior</text>
    <text x="252" y="224">posterior of test 1 = prior of test 2</text>
    <text x="422" y="224">posterior of test 2</text>
  </g>
</svg>
</div>

*$P(\text{sick})$ at each stage, green portion of each bar. Each update keeps
only the people consistent with the new evidence: 10,000 shrink to the 594
who tested positive, then to the 123 who tested positive twice.*

## 3.4 Maximum likelihood: where the standard losses come from

**A model is a claim about how probable each possible outcome is. Training
means adjusting it until the outcomes that actually happened look as
probable as possible.** Every standard loss is the same quantity,
$-\log$(probability the model gave to what actually happened):

- A **classifier** outputs a probability per class. The loss is $-\log$ of
  the probability on the true class: **cross-entropy**.
- A **regression model** outputs a number, read as the center of a bell of
  likely values. The loss is $-\log$ of the bell's height at the true value,
  which works out to the squared error: **MSE**.

#### The principle, on a coin

**Maximum likelihood (MLE): pick the parameters under which the data you saw
is most probable.**

A coin lands heads 7 times in 10. Each flip is a Bernoulli($p$) draw (§3.2),
so the chance of that sequence is $p^7 (1-p)^3$. Try every $p$:

```python
p = np.linspace(0.01, 0.99, 99)          # candidate heads probabilities
lik = p**7 * (1 - p)**3                  # probability of the data under each
p[lik.argmax()]                          # 0.7
lik[np.isclose(p, 0.5)], lik[np.isclose(p, 0.7)]   # 0.00098, 0.0022
```

$p = 0.7$ makes the data twice as probable as $p = 0.5$, and no $p$ does
better. In general, for $h$ heads in $n$ flips the best $p$ is $h/n$, the
observed frequency. (Setting the derivative of $h \log p + (n-h)\log(1-p)$
to zero gives $h/p = (n-h)/(1-p)$, so $p = h/n$.)

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

*The log-likelihood of every candidate $p$ after 7 heads in 10 flips: one
peak, at $p = 0.7$.*

That probability of the data, viewed as a function of the parameters, is the
**likelihood**: the same word as in Bayes rule (§3.3), the probability of the
evidence given a hypothesis. For a model with parameters $\theta$ and
independent examples it is a product, one factor per example. In practice
you maximize its log: a product of thousands of probabilities rounds to 0 on
a computer, the log turns it into a sum, and the best $\theta$ is the same.

$$\hat\theta = \arg\max_\theta \prod_i p_\theta(y_i \mid x_i) = \arg\max_\theta \sum_i \log p_\theta(y_i \mid x_i)$$

($\arg\max_\theta$ means "the $\theta$ at which this is largest".)

#### Cross-entropy is MLE for a classifier

**Cross-entropy charges each example $-\log$(probability on the true class).
Adding up those charges gives exactly the negative log-likelihood.**

Two photos, three classes (cat, dog, bird):

```python
probs = np.array([[0.7, 0.2, 0.1],     # photo 1, true class: cat
                  [0.5, 0.2, 0.3]])    # photo 2, true class: dog
labels = np.array([0, 1])

p_true = probs[np.arange(2), labels]   # [0.7, 0.2]: probability on the truth
(-np.log(p_true)).sum()                # 1.97  cross-entropy
-np.log(p_true.prod())                 # 1.97  negative log-likelihood
```

The likelihood is $0.7 \times 0.2 = 0.14$, the probability the model gave to
both true labels, and $-\log 0.14 = 1.97$: the same number as the summed
loss. Minimizing cross-entropy is maximizing the likelihood.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 255" role="img" aria-label="Two examples with three-class probability bars: the loss reads only the true class bar of each example, contributing minus log of that probability; the contributions add to the cross-entropy" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g fill="currentColor" fill-opacity="0.85" text-anchor="middle">
    <text x="101" y="34">photo 1: a cat</text>
    <text x="331" y="34">photo 2: a dog</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.35">
    <line x1="44" y1="150" x2="160" y2="150"/>
    <line x1="274" y1="150" x2="390" y2="150"/>
  </g>
  <g fill="currentColor" fill-opacity="0.25">
    <rect x="88" y="130" width="26" height="20"/>
    <rect x="126" y="140" width="26" height="10"/>
    <rect x="280" y="100" width="26" height="50"/>
    <rect x="356" y="120" width="26" height="30"/>
  </g>
  <g fill="#10b981">
    <rect x="50" y="80" width="26" height="70"/>
    <rect x="318" y="130" width="26" height="20"/>
  </g>
  <g fill="currentColor" fill-opacity="0.75" text-anchor="middle" font-size="10.5">
    <text x="63" y="72">0.7</text>
    <text x="101" y="124">0.2</text>
    <text x="139" y="134">0.1</text>
    <text x="293" y="94">0.5</text>
    <text x="331" y="124">0.2</text>
    <text x="369" y="114">0.3</text>
  </g>
  <g text-anchor="middle" font-size="10.5">
    <text x="63" y="166" fill="#10b981" font-weight="600">cat</text>
    <text x="101" y="166" fill="currentColor" fill-opacity="0.6">dog</text>
    <text x="139" y="166" fill="currentColor" fill-opacity="0.6">bird</text>
    <text x="293" y="166" fill="currentColor" fill-opacity="0.6">cat</text>
    <text x="331" y="166" fill="#10b981" font-weight="600">dog</text>
    <text x="369" y="166" fill="currentColor" fill-opacity="0.6">bird</text>
  </g>
  <g fill="#f43f5e" text-anchor="middle" font-size="10.5">
    <text x="101" y="192">term: &#8722;log 0.7 = 0.36</text>
    <text x="331" y="192">term: &#8722;log 0.2 = 1.61</text>
  </g>
  <text x="260" y="222" text-anchor="middle" fill="currentColor" fill-opacity="0.8" font-size="10.5">likelihood = 0.7 &#183; 0.2 = 0.14</text>
  <text x="260" y="242" text-anchor="middle" fill="#f43f5e" font-weight="600" font-size="10.5">cross-entropy = 0.36 + 1.61 = 1.97</text>
</svg>
</div>

*The loss reads only the green bar of each example, the probability on the
true class.*

Why $-\log$ is the right charge: it is small when the model was confident
and right, and grows without limit as the probability on the true class goes
to 0.

| probability the model put on the true class | 0.9 | 0.5 | 0.01 |
|---|---|---|---|
| loss $= -\log(\text{that probability})$ | 0.11 | 0.69 | 4.61 |

So the loss comes mostly from the examples the model found surprising, and
training moves probability toward the labels that actually occurred.

In general, cross-entropy compares a target distribution $q$ with the
model's $p$: $H(q, p) = -\sum_k q_k \log p_k$. With a **one-hot** target
($q_k = 1$ for the true class, 0 otherwise) only the true class's term
survives, so it equals the negative log-likelihood. With soft targets such
as label smoothing, $q = [0.9, 0.05, 0.05]$, it is still a fine loss, but no
longer a likelihood.

**Two classes: binary cross-entropy.** A classifier that outputs
$\hat y = 0.9$ for "spam" models the label as Bernoulli($0.9$). It gives the
true label $y \in \{0, 1\}$ probability $\hat y$ if $y = 1$ and $1 - \hat y$
if $y = 0$. Taking $-\log$, both cases in one line:

$$-\log p_\theta(y \mid x) = -\big[\, y \log \hat y + (1 - y) \log(1 - \hat y) \,\big]$$

#### The regression losses: MSE and L1

**MSE is MLE when prediction errors are bell-shaped (Gaussian). L1 is MLE
when they follow a Laplace distribution.**

**Why an assumption is needed.** A regression model outputs one number, say
a price of 300k. Taken literally, that claims the price is exactly 300k and
gives every other price probability 0. Real sales never land exactly on the
prediction, so every model would give the data probability 0 and MLE could
not compare them. So assume the actual price is the prediction plus random
noise, $y = f_\theta(x) + \varepsilon$, spread as the bell of §3.2 with width
$\sigma = 10$k. A price that misses by $d$ thousand then gets probability
factor $e^{-d^2 / 2\sigma^2} = e^{-d^2/200}$:

- off by 0: factor 1
- off by 10k: $e^{-100/200} = 0.61$
- off by 30k: $e^{-900/200} = 0.011$

**Two candidate models.** Three houses sold for 310k, 290k and 305k. Which
model makes those prices more probable?

```python
y = np.array([310, 290, 305])                 # actual prices (k)
pred_A = np.array([300, 300, 300])
pred_B = np.array([310, 290, 300])

lik = lambda pred: np.exp(-(y - pred)**2 / 200).prod()
lik(pred_A), lik(pred_B)                      # 0.32, 0.88
((y - pred_A)**2).sum(), ((y - pred_B)**2).sum()   # 225, 25
```

Model B makes the actual prices more probable, so MLE picks B. It also has
the smaller sum of squared errors, and that is no coincidence: each factor
is $e$ raised to $-\text{error}^2/200$, and multiplying powers of $e$ adds
their exponents,

$$\text{likelihood} = e^{-(\text{sum of squared errors}) / 200}$$

so $e^{-225/200} = 0.32$ and $e^{-25/200} = 0.88$. A smaller sum of squared
errors always means a larger likelihood. **Maximizing the likelihood is
minimizing the squared error: MSE.**

**L1 is the same with a different shape.** If misses follow a Laplace
distribution, the factor for a miss of $d$ is $e^{-|d|/b}$. The exponents
again add, the likelihood is $e^{-(\text{sum of absolute errors})/b}$, and
MLE minimizes the sum of absolute errors: **L1**.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 220" role="img" aria-label="Left: Gaussian and Laplace noise densities. Right: their minus log curves, a parabola for the Gaussian giving MSE and a V shape for the Laplace giving L1 loss" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g fill="currentColor" fill-opacity="0.7" text-anchor="middle" font-size="10.5">
    <text x="130" y="24">noise density p(&#949;)</text>
    <text x="390" y="24">loss = &#8722;log p(&#949;) + const</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.4">
    <line x1="20" y1="190" x2="240" y2="190"/>
    <line x1="280" y1="190" x2="500" y2="190"/>
  </g>
  <g fill="currentColor" fill-opacity="0.6" text-anchor="middle" font-size="10">
    <text x="130" y="204">0</text>
    <text x="390" y="204">0</text>
    <text x="246" y="194" text-anchor="start">&#949;</text>
    <text x="506" y="194" text-anchor="start">&#949;</text>
  </g>
  <polyline points="20,188.8 38.3,185.2 56.7,175.3 75,154.8 93.3,124.2 111.7,94.2 130,81.5 148.3,94.2 166.7,124.2 185,154.8 203.3,175.3 221.7,185.2 240,188.8" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <polyline points="20,183.2 38.3,178.8 56.7,171.6 75,159.7 93.3,140 111.7,107.5 130,54 148.3,107.5 166.7,140 185,159.7 203.3,171.6 221.7,178.8 240,183.2" fill="none" stroke="#10b981" stroke-width="2.5"/>
  <line x1="24" y1="44" x2="44" y2="44" stroke="#6366f1" stroke-width="2.5"/>
  <text x="48" y="48" fill="#6366f1" font-size="10.5">Gaussian</text>
  <line x1="24" y1="62" x2="44" y2="62" stroke="#10b981" stroke-width="2.5"/>
  <text x="48" y="66" fill="#10b981" font-size="10.5">Laplace</text>
  <polyline points="280,43.3 298.3,88.1 316.7,124.8 335,153.3 353.3,173.7 371.7,185.9 390,190 408.3,185.9 426.7,173.7 445,153.3 463.3,124.8 481.7,88.1 500,43.3" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <polyline points="280,92.2 390,190 500,92.2" fill="none" stroke="#10b981" stroke-width="2.5"/>
  <text x="350" y="54" text-anchor="middle" fill="#6366f1" font-size="10.5">&#949;&#178;/2: MSE</text>
  <text x="437" y="54" text-anchor="middle" fill="#10b981" font-size="10.5">|&#949;|: L1</text>
</svg>
</div>

*Left: the two noise distributions, the Gaussian bell and the Laplace with
its sharper peak and heavier tails. Right: their $-\log$, the loss per
example: a parabola (MSE) and a V (L1).*

**Which to pick.** Picking a loss is picking how you expect misses to be
spread. The Gaussian says big misses almost never happen, so MSE punishes
them quadratically. The Laplace expects occasional big misses, and L1
punishes them only linearly. With errors of 1 on most points and 10 on one
outlier, MSE charges the outlier 100 against 1 for each other point and bends
the whole fit toward it; L1 charges 10 against 1.

In symbols: $p_\theta(y \mid x) = p_\varepsilon\big(y - f_\theta(x)\big)$, the
probability of a price is the probability of the error it takes to produce
it. The gap $y - f_\theta(x)$ measured on data is called the **residual**.

## 3.5 Entropy and KL: what a loss value means

§3.4 wrote every loss as $-\log$(probability of what happened). This section
says what that number measures. The same few definitions explain the floor a
label-smoothed loss never crosses, the KL penalty in RLHF (module 7), and why
the Gaussian assumes the least (§3.2).

#### Surprise and entropy

**The surprise of an outcome with probability $p$ is $-\log p$.** Certain
outcomes have surprise 0, rare ones a lot, and independent surprises add,
since $-\log(pq) = -\log p - \log q$. The cross-entropy charges in §3.4's
table (0.11 for a probability of 0.9, 4.61 for 0.01) are exactly these
surprises. With natural logs the unit is the **nat**, with base 2 the
**bit** (1 nat ≈ 1.44 bits); `torch` losses are in nats.

**Entropy is the average surprise of a distribution**: how unpredictable one
draw is.

```python
H = lambda p: -(p * np.log(p)).sum()     # average surprise, in nats

H(np.full(6, 1/6))           # 1.79   fair die (= log 6)
H(np.array([0.5, 0.5]))      # 0.693  fair coin (= 1 bit)
H(np.array([0.9, 0.1]))      # 0.325  biased coin: more predictable
```

$$H(p) = -\sum_x p(x)\log p(x)$$

<div style="margin:1.25rem 0">
<svg viewBox="0 0 380 200" role="img" aria-label="Entropy of a coin in bits plotted against its heads probability p: zero at p equals 0 and 1, maximum of one bit at p equals one half" style="max-width:400px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="180" y="22" text-anchor="middle" fill="#6366f1" font-weight="600">H(p) of a coin, in bits</text>
  <line x1="40" y1="160" x2="332" y2="160" stroke="currentColor" opacity="0.5"/>
  <line x1="40" y1="160" x2="40" y2="30" stroke="currentColor" opacity="0.5"/>
  <line x1="180" y1="160" x2="180" y2="40" stroke="currentColor" opacity="0.35" stroke-dasharray="4 4"/>
  <line x1="40" y1="40" x2="180" y2="40" stroke="currentColor" opacity="0.35" stroke-dasharray="4 4"/>
  <polyline points="40.0,160.0 45.6,143.0 51.2,130.9 56.8,120.7 62.4,111.7 68.0,103.7 73.6,96.5 79.2,89.9 84.8,83.9 90.4,78.4 96.0,73.4 101.6,68.8 107.2,64.6 112.8,60.8 118.4,57.3 124.0,54.2 129.6,51.5 135.2,49.0 140.8,46.9 146.4,45.0 152.0,43.5 157.6,42.2 163.2,41.2 168.8,40.6 174.4,40.1 180.0,40.0 185.6,40.1 191.2,40.6 196.8,41.2 202.4,42.2 208.0,43.5 213.6,45.0 219.2,46.9 224.8,49.0 230.4,51.5 236.0,54.2 241.6,57.3 247.2,60.8 252.8,64.6 258.4,68.8 264.0,73.4 269.6,78.4 275.2,83.9 280.8,89.9 286.4,96.5 292.0,103.7 297.6,111.7 303.2,120.7 308.8,130.9 314.4,143.0 320.0,160.0" fill="none" stroke="#6366f1" stroke-width="2"/>
  <circle cx="180" cy="40" r="3.5" fill="#6366f1"/>
  <circle cx="292" cy="103.7" r="3.5" fill="#f59e0b"/>
  <text x="286" y="122" text-anchor="end" fill="#f59e0b" font-size="10.5">p = 0.9: 0.47 bits</text>
  <text x="40" y="178" text-anchor="middle" fill="currentColor">0</text>
  <text x="180" y="178" text-anchor="middle" fill="currentColor">1/2</text>
  <text x="320" y="178" text-anchor="middle" fill="currentColor">1</text>
  <text x="344" y="164" fill="currentColor">p</text>
  <text x="34" y="44" text-anchor="end" fill="currentColor">1</text>
</svg>
</div>

*Entropy of one coin flip as $p$ goes from 0 to 1. The fair coin is the
hardest to predict and has the most, exactly one bit.*

#### Cross-entropy and KL divergence

Now two distributions over the same outcomes: $q$, the truth, and $p$, a
model's guess at it. Say a fair coin, $q = [0.5, 0.5]$, and a model that
believes it lands heads 90% of the time, $p = [0.9, 0.1]$.

- **Cross-entropy** $H(q, p) = -\sum_x q(x)\log p(x)$ is the average surprise
  when outcomes come from $q$ but are scored with $p$. This is §3.4's loss,
  with $q$ the label and $p$ the model's output.
- **KL divergence** $\mathrm{KL}(q\|p) = \sum_x q(x)\log\frac{q(x)}{p(x)}$ is
  the extra surprise caused by scoring with the wrong distribution.

```python
q, p = np.array([0.5, 0.5]), np.array([0.9, 0.1])
H(q)                              # 0.693  the truth's own unpredictability
(q * np.log(q / p)).sum()         # 0.511  KL: extra surprise from using p
-(q * np.log(p)).sum()            # 1.204  cross-entropy = 0.693 + 0.511
```

The three are connected:

$$H(q, p) = H(q) + \mathrm{KL}(q \,\|\, p)$$

In words: the loss is the labels' own unpredictability plus the extra caused
by the model being wrong.

#### KL is never negative, and what follows

**$\mathrm{KL}(q\|p) \ge 0$, and it is 0 only when $p = q$: a wrong model can
only add surprise, never remove it.** (This is Gibbs' inequality.) The proof
uses $\log x \le x - 1$, true for every $x > 0$, with $x = p/q$:

$$-\mathrm{KL}(q \,\|\, p) = \sum_x q(x)\log\frac{p(x)}{q(x)}
  \;\le\; \sum_x q(x)\Big(\frac{p(x)}{q(x)} - 1\Big)
  = 1 - 1 = 0$$

Three consequences used daily:

- **The loss has a floor.** Cross-entropy can never go below the labels'
  entropy $H(q)$. With one-hot labels the floor is 0. With label smoothing,
  $q = [0.9, 0.05, 0.05]$, it is $H(q) \approx 0.394$ nats, which is why a
  smoothed training loss levels off above zero even when the model is right
  on every example.
- **Uniform has the most entropy.** Comparing any $q$ over $K$ outcomes with
  the uniform distribution gives $H(q) \le \log K$. The die's $\log 6$ is
  the most any six-outcome variable can have.
- **KL is not symmetric.** Swapping the coin example gives
  $\mathrm{KL}(p\|q) = 0.368 \ne 0.511$. $\mathrm{KL}(q\|p)$ charges heavily
  wherever the truth produces outcomes the model calls unlikely, so
  minimizing it makes the model spread out to cover everything.
  $\mathrm{KL}(p\|q)$ charges wherever the model puts probability the truth
  does not, so minimizing it makes the model play safe and focus on one
  region.

#### Where these appear in ML

- **MLE minimizes a KL.** Averaged over a large dataset, the negative
  log-likelihood is $H(q, p_\theta) = H(q) + \mathrm{KL}(q\|p_\theta)$, with
  $q$ the true data distribution. $H(q)$ does not depend on the model, so
  maximizing likelihood minimizes $\mathrm{KL}(q\|p_\theta)$: the spreading
  direction, which is why an MLE-trained model spreads probability over
  everything that appears in the data.
- **RLHF's KL penalty** (module 7, §3.2) is the other direction,
  $\mathrm{KL}(\pi_\theta\|\pi_{ref})$: the fine-tuned model pays wherever it
  gives probability to outputs the original model would rarely produce, so it
  can become more focused but cannot drift far.
- **Knowledge distillation** trains a student on the teacher's full output
  distribution instead of one-hot labels. The floor becomes the teacher's
  entropy, and the student learns the teacher's uncertainty, not just its top
  choice.

#### Mutual information

**Mutual information is how much knowing one variable tells you about
another.** It is the KL between how often pairs of values actually occur
together, $p(x, y)$, and how often they would if the two were unrelated,
$p(x)\,p(y)$:

$$I(X; Y) = \sum_{x, y} p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)} = H(X) - H(X \mid Y)$$

It is 0 exactly when $X$ and $Y$ are independent, and otherwise equals the
uncertainty in $X$ minus what is left once $Y$ is known. Feature selection
can rank features by it, and the contrastive loss of module 4 is designed to
push it up.

#### Entropy of a Gaussian, and why it is the maximum

For a bell of width $\sigma$, averaging the $-\log$ density from §3.2 gives

$$H = \tfrac{1}{2} + \tfrac{1}{2}\log(2\pi\sigma^2) = \tfrac{1}{2}\log(2\pi e\sigma^2)$$

It grows with the width and does not depend on where the bell is centered.
For $\sigma = 2$, $H \approx 2.11$ nats.

**Why it is the maximum.** Take any distribution $p$ with mean $\mu$ and
variance $\sigma^2$, and the Gaussian $g$ with the same two numbers. Scoring
$p$'s draws with $g$ costs exactly $H(g)$ on average, because $-\log g$ only
involves $(x - \mu)^2$, whose average is $\sigma^2$ for both. Then
$0 \le \mathrm{KL}(p\|g) = H(g) - H(p)$, so no distribution with that mean
and variance is more unpredictable than the Gaussian. Assuming Gaussian noise
when you know only the mean and variance assumes nothing more.

## 3.6 Averages: error bars and the central limit theorem

**Average $n$ independent draws and two things happen: the average's typical
error shrinks like $\sigma/\sqrt{n}$, and its distribution becomes a bell,
whatever distribution the draws came from.**

This starts the chapter's third part: how much to trust a number computed
from data. Almost every such number is an average over a sample: a test
accuracy (an average of 0/1 scores), a minibatch gradient (an average of
per-example gradients). The typical error of such an average, its standard
deviation, is called its **standard error**: the error bar.

#### The $\sigma/\sqrt{n}$ error bar

Average $n$ draws that are **independent** (no draw affects another) and
**identically distributed** (all from the same distribution, mean $\mu$,
variance $\sigma^2$). By §3.1's rules:

- the average $\bar X$ has mean $\mu$ (linearity), and
- variance $\frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$ (the variances
  of the $n$ draws add, and the $\frac{1}{n}$ comes out squared).

So the error bar is $\sigma/\sqrt{n}$.

```python
x = np.random.exponential(size=(100_000, 30))   # 100k samples of 30 draws
x.std()                    # ≈ 1.0    one draw's spread (sigma)
x.mean(axis=1).std()       # ≈ 0.18   spread of a 30-draw average: 1/sqrt(30)
```

It shrinks as $n$ grows, but slowly: halving the error bar takes 4× the
data.

#### The bell shape

The draws above come from the exponential distribution, which is lopsided:
highest at zero, with a long right tail. Their averages are not:

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 205" role="img" aria-label="Three panels showing the distribution of the average of n exponential draws for n equal to 1, 4, and 30: the lopsided exponential becomes a narrow symmetric bell centered on the mean 1" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g fill="currentColor" fill-opacity="0.85" text-anchor="middle">
    <text x="90" y="28">n = 1: one draw</text>
    <text x="265" y="28">n = 4</text>
    <text x="440" y="28">n = 30</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.4">
    <line x1="20" y1="170" x2="160" y2="170"/>
    <line x1="195" y1="170" x2="335" y2="170"/>
    <line x1="370" y1="170" x2="510" y2="170"/>
  </g>
  <g stroke="#f59e0b" stroke-dasharray="4 4">
    <line x1="66.7" y1="170" x2="66.7" y2="44"/>
    <line x1="241.7" y1="170" x2="241.7" y2="44"/>
    <line x1="416.7" y1="170" x2="416.7" y2="44"/>
  </g>
  <polyline points="20.0,48.0 25.0,60.4 30.0,71.5 35.0,81.5 40.0,90.5 45.0,98.6 50.0,105.9 55.0,112.4 60.0,118.2 65.0,123.5 70.0,128.2 75.0,132.5 80.0,136.3 85.0,139.7 90.0,142.8 95.0,145.5 100.0,148.0 105.0,150.3 110.0,152.3 115.0,154.1 120.0,155.7 125.0,157.1 130.0,158.4 135.0,159.6 140.0,160.7 145.0,161.6 150.0,162.5 155.0,163.2 160.0,163.9" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <polyline points="195.0,170.0 200.0,165.3 205.0,145.7 210.0,116.7 215.0,87.7 220.0,65.2 225.0,52.1 230.0,48.0 235.0,51.4 240.0,60.0 245.0,71.7 250.0,84.7 255.0,97.9 260.0,110.3 265.0,121.4 270.0,131.1 275.0,139.2 280.0,145.9 285.0,151.4 290.0,155.8 295.0,159.2 300.0,161.8 305.0,163.9 310.0,165.4 315.0,166.6 320.0,167.5 325.0,168.2 330.0,168.7 335.0,169.0" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <polyline points="370.0,170.0 375.0,170.0 380.0,170.0 385.0,170.0 390.0,169.9 395.0,168.1 400.0,155.3 405.0,118.4 410.0,70.3 415.0,48.0 420.0,65.9 425.0,103.7 430.0,136.8 435.0,156.4 440.0,165.3 445.0,168.6 450.0,169.6 455.0,169.9 460.0,170.0 465.0,170.0 470.0,170.0 475.0,170.0 480.0,170.0 485.0,170.0 490.0,170.0 495.0,170.0 500.0,170.0 505.0,170.0 510.0,170.0" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <g fill="currentColor" fill-opacity="0.6" text-anchor="middle" font-size="10">
    <text x="20" y="184">0</text>
    <text x="195" y="184">0</text>
    <text x="370" y="184">0</text>
  </g>
  <g fill="#f59e0b" text-anchor="middle" font-size="10">
    <text x="66.7" y="184">&#956; = 1</text>
    <text x="241.7" y="184">&#956; = 1</text>
    <text x="416.7" y="184">&#956; = 1</text>
  </g>
</svg>
</div>

*The distribution of the average of $n$ exponential draws (curves scaled to
matching heights). It narrows around the mean (that is $\sigma/\sqrt{n}$) and
becomes a symmetric bell (that is the central limit theorem).*

The **central limit theorem (CLT)**: measure how far the average lands from
$\mu$ in units of its error bar, and that number follows the standard bell
($\mu = 0$, $\sigma = 1$) more and more closely as $n$ grows:

$$\frac{\bar X - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

**Why sums become bells.** Two dice: a total of 7 can happen six ways
(1+6, 2+5, ..., 6+1), but 2 and 12 only one way each. Middle totals have many
combinations, extreme totals few, so one die's flat distribution becomes a
triangle for two. The figure shows this as six shifted copies of one die's
distribution added up, one copy per value of the second die. That
shift-and-add is called a **convolution**.

<div style="margin:1.25rem 0">
<svg viewBox="-32 0 552 250" role="img" aria-label="Six copies of a die's flat distribution, each slid one step further right, stack into a triangle: the total 7 is covered by all six copies while the totals 2 and 12 are covered by one each" style="max-width:560px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <line x1="240" y1="18" x2="240" y2="215" stroke="#f59e0b" stroke-dasharray="4 4" stroke-opacity="0.7"/>
  <g fill="currentColor" fill-opacity="0.3">
    <rect x="22" y="24" width="236" height="9" rx="2"/>
    <rect x="62" y="39" width="236" height="9" rx="2"/>
    <rect x="102" y="54" width="236" height="9" rx="2"/>
    <rect x="142" y="69" width="236" height="9" rx="2"/>
    <rect x="182" y="84" width="236" height="9" rx="2"/>
    <rect x="222" y="99" width="236" height="9" rx="2"/>
  </g>
  <g fill="currentColor" fill-opacity="0.65" font-size="10">
    <text x="18" y="32" text-anchor="end">v = 1</text>
    <text x="58" y="47" text-anchor="end">v = 2</text>
    <text x="98" y="62" text-anchor="end">v = 3</text>
    <text x="138" y="77" text-anchor="end">v = 4</text>
    <text x="178" y="92" text-anchor="end">v = 5</text>
    <text x="218" y="107" text-anchor="end">v = 6</text>
  </g>
  <text x="240" y="136" text-anchor="middle" fill="currentColor" fill-opacity="0.7" font-size="10.5">add the six rows, column by column</text>
  <g fill="#6366f1">
    <rect x="26" y="204" width="28" height="11"/>
    <rect x="66" y="193" width="28" height="22"/>
    <rect x="106" y="182" width="28" height="33"/>
    <rect x="146" y="171" width="28" height="44"/>
    <rect x="186" y="160" width="28" height="55"/>
    <rect x="226" y="149" width="28" height="66"/>
    <rect x="266" y="160" width="28" height="55"/>
    <rect x="306" y="171" width="28" height="44"/>
    <rect x="346" y="182" width="28" height="33"/>
    <rect x="386" y="193" width="28" height="22"/>
    <rect x="426" y="204" width="28" height="11"/>
  </g>
  <g fill="currentColor" fill-opacity="0.6" text-anchor="middle" font-size="10">
    <text x="40" y="200">1</text>
    <text x="80" y="189">2</text>
    <text x="120" y="178">3</text>
    <text x="160" y="167">4</text>
    <text x="200" y="156">5</text>
    <text x="240" y="145">6</text>
    <text x="280" y="156">5</text>
    <text x="320" y="167">4</text>
    <text x="360" y="178">3</text>
    <text x="400" y="189">2</text>
    <text x="440" y="200">1</text>
  </g>
  <g text-anchor="middle" font-size="10.5">
    <text x="40" y="232" fill="currentColor" fill-opacity="0.7">t = 2</text>
    <text x="240" y="232" fill="#f59e0b">t = 7</text>
    <text x="440" y="232" fill="currentColor" fill-opacity="0.7">t = 12</text>
  </g>
</svg>
</div>

*Top: six copies of one die's flat distribution, shifted one step apart.
Bottom: adding the columns counts the ways to make each total (out of 36).
The flat shape has become a triangle.*

Each extra draw repeats the shift-and-add. Every repeat smooths away more of
the starting shape (like the exponential's spike at zero), while the mean and
variance just add. And once the shape is a bell it stays a bell, because
adding Gaussians gives a Gaussian. `probability_demos.ipynb` tracks this
numerically.

#### Where ML uses it

- **SGD noise.** A minibatch gradient is an average of per-example
  gradients, so its noise shrinks like $1/\sqrt{\text{batch size}}$:
  quadrupling the batch halves the noise.
- **Error bars on test accuracy.** Each test example scores 1 or 0, a
  Bernoulli draw with variance $p(1-p)$, so accuracy on $n$ examples has
  error bar $\sqrt{p(1-p)/n}$. With 1,000 examples at 90% accuracy:
  $\sqrt{0.9 \times 0.1 / 1000} \approx 0.009$, about $\pm 0.9\%$. A "0.5%
  improvement" on that test set is within noise.
- **Confidence intervals** report "$\pm 2$ standard errors" because about 95%
  of a bell lies within 2 standard deviations of its center (95.4%; exactly
  95% at 1.96, the value §3.8 uses).

## 3.7 Bias and variance: two ways an estimate misses

**A number computed from a random sample is itself random. It can miss the
truth in two ways: off on average (bias), or jumpy from sample to sample
(variance). Its expected squared error is exactly bias² + variance.** §3.6
measured the jumpiness of an average; this section adds the second way to be
wrong.

Names: the **target** $\theta$ is the true number you want (the die's mean).
An **estimator** is a formula applied to a sample, and its output is the
**estimate** $\hat\theta$. §3.4's $\hat p = h/n$ is an estimator, and so is
every MLE.

Example: estimate the die's mean (3.5) by averaging 3 rolls.

```python
samples = np.random.randint(1, 7, size=(100_000, 3))   # 100k samples of 3 rolls
est = samples.mean(axis=1)                             # one estimate per sample
est[:3]                     # e.g. [3.67, 2.33, 4.67]: each sample disagrees
est.mean() - 3.5            # ≈ 0      bias: right on average
est.var()                   # ≈ 0.97   variance: 2.92 / 3 (§3.6)
```

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

- **Bias** is a systematic miss: averaging estimates from thousands of fresh
  samples still would not land on the truth. The 3-roll average has none.
- **Variance** is the sample-to-sample jumpiness.

Scored by the **mean squared error**, $\mathbb{E}[(\hat\theta - \theta)^2]$,
the two add exactly. Split the miss into "jump around the estimator's own
average" ($a$, random) plus "bias" ($b$, fixed). Then
$(a + b)^2 = a^2 + 2ab + b^2$, and $a$ averages to zero, so the cross term
drops out:

$$\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$$

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

*Each dot is the estimate from one resampled dataset. Bias is how far the
cloud's center sits from the bullseye; variance is how scattered it is.*

#### Estimating a variance from data (Bessel's correction)

**Computing a sample's variance with divisor $n$ underestimates the truth on
average. Dividing by $n - 1$ fixes it exactly.** NumPy exposes the choice as
`ddof`. Check it on every possible 3-roll sample:

```python
import itertools
all_samples = list(itertools.product(range(1, 7), repeat=3))   # all 216
np.mean([np.var(s) for s in all_samples])           # 1.94  divide by n
np.mean([np.var(s, ddof=1) for s in all_samples])   # 2.92  divide by n - 1
```

The true variance is 2.92 (§3.1). Divide by $n$ and you are a third short on
average; divide by $n - 1$ and you are exactly right.

**Why: the sample mean sits too close to its own sample.** The variance
formula measures squared distances from the mean. A sample does not know the
true mean $\mu$, so it measures from its own mean $\bar X$. But $\bar X$ is the
point that makes the sum of squared distances to the sample smallest:

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 220" role="img" aria-label="The sum of squared deviations of the rolls 2, 6 and 3 from a point c, plotted against c: a parabola with its minimum at c equal to x bar, so the value at c equal to mu is higher by n times x bar minus mu squared" style="max-width:420px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="200" y="18" text-anchor="middle" fill="#6366f1" font-weight="600">Σᵢ (xᵢ − c)², for the rolls 2, 6, 3</text>
  <line x1="44" y1="32" x2="44" y2="190" stroke="currentColor" opacity="0.5"/>
  <line x1="44" y1="190" x2="356" y2="190" stroke="currentColor" opacity="0.5"/>
  <path d="M 50 40 Q 190 280 330 40" fill="none" stroke="#6366f1" stroke-width="2"/>
  <line x1="44" y1="124.7" x2="190" y2="124.7" stroke="#f59e0b" opacity="0.6" stroke-dasharray="4 4"/>
  <line x1="44" y1="160" x2="190" y2="160" stroke="#10b981" opacity="0.6" stroke-dasharray="4 4"/>
  <line x1="114" y1="124.7" x2="114" y2="190" stroke="#f59e0b" opacity="0.6" stroke-dasharray="4 4"/>
  <line x1="190" y1="160" x2="190" y2="190" stroke="#10b981" opacity="0.6" stroke-dasharray="4 4"/>
  <line x1="190" y1="124.7" x2="190" y2="160" stroke="#f43f5e" stroke-width="1.5"/>
  <line x1="186" y1="124.7" x2="194" y2="124.7" stroke="#f43f5e" stroke-width="1.5"/>
  <line x1="186" y1="160" x2="194" y2="160" stroke="#f43f5e" stroke-width="1.5"/>
  <text x="190" y="116" text-anchor="middle" fill="#f43f5e" font-size="10.5">n(x̄ − μ)² = 0.08</text>
  <circle cx="114" cy="124.7" r="3.5" fill="#f59e0b"/>
  <circle cx="190" cy="160" r="3.5" fill="#10b981"/>
  <text x="40" y="128.7" text-anchor="end" fill="#f59e0b" font-size="11">8.75</text>
  <text x="40" y="164" text-anchor="end" fill="#10b981" font-size="11">8.67</text>
  <text x="114" y="206" text-anchor="middle" fill="#f59e0b">μ = 3.5</text>
  <text x="190" y="206" text-anchor="middle" fill="#10b981">x̄ ≈ 3.67</text>
  <text x="364" y="194" fill="currentColor">c</text>
</svg>
</div>

*The sum of squared distances from the rolls 2, 6, 3 to a point $c$. It is
smallest at $c = \bar x$, where the estimator measures from, and higher at the
true mean $c = \mu$ by exactly $n(\bar x - \mu)^2$.*

So measuring from $\bar X$ always undercounts the spread. By exactly how
much: writing $X_i - \mu = (X_i - \bar X) + (\bar X - \mu)$ and summing the
squares gives

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

On average the first sum is $n\sigma^2$, and the gap is
$n \cdot \mathrm{Var}(\bar X) = n \cdot \sigma^2/n = \sigma^2$ (§3.6), one
draw's worth. So the sum of squared deviations averages $(n-1)\sigma^2$, and
dividing by $n - 1$ makes it exactly right. This is **Bessel's correction**.

Another way to see it: the $n$ deviations $X_i - \bar X$ always sum to zero,
so once $n - 1$ are known the last is fixed. The sample has only $n - 1$
free deviations. With one draw, the single deviation is 0: dividing by $n$
claims zero spread, while dividing by $n - 1$ gives $0/0$, undefined, which
is the honest answer.

#### The tradeoff in ML terms

Training a model fits the same template. The sample is the training set, the
estimator is the training procedure, and the estimate is the trained model's
prediction at some input $x$. "Redrawing the sample" means collecting a new
training set and retraining.

- **Bias**: average the predictions at $x$ over many training sets; bias is
  how far that average is from the truth. It comes from what the model
  cannot express.
- **Variance**: how much the prediction at $x$ changes between training
  sets. It comes from the fit following the quirks and noise of the one
  sample it saw.

A **linear model** is high-bias, low-variance: if the truth curves, a line
misses it in the same places every time, but refitting gives nearly the same
line. A **deep net or unpruned decision tree** is low-bias, high-variance: it
can fit nearly any shape, but each fit also follows its own training set's
noise, so two fits disagree.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 380 205" role="img" aria-label="Bias squared falls and variance rises as model flexibility grows; their sum is a U shaped curve with a minimum at intermediate flexibility" style="max-width:400px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <line x1="50" y1="170" x2="348" y2="170" stroke="currentColor" opacity="0.5"/>
  <line x1="50" y1="170" x2="50" y2="28" stroke="currentColor" opacity="0.5"/>
  <line x1="195" y1="103.3" x2="195" y2="170" stroke="currentColor" opacity="0.35" stroke-dasharray="4 4"/>
  <path d="M 50 36.7 Q 195 170 340 170" fill="none" stroke="#10b981" stroke-width="2"/>
  <path d="M 50 170 Q 195 170 340 36.7" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <path d="M 50 36.7 Q 195 170 340 36.7" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <circle cx="195" cy="103.3" r="3.5" fill="#6366f1"/>
  <text x="195" y="90" text-anchor="middle" fill="#6366f1" font-weight="600">Bias² + Var</text>
  <text x="72" y="120" fill="#10b981">Bias²</text>
  <text x="318" y="120" text-anchor="end" fill="#f59e0b">Var</text>
  <text x="52" y="18" fill="currentColor" fill-opacity="0.7">error</text>
  <text x="50" y="190" text-anchor="middle" fill="currentColor" fill-opacity="0.7">simple</text>
  <text x="195" y="190" text-anchor="middle" fill="currentColor" fill-opacity="0.7">model flexibility</text>
  <text x="340" y="190" text-anchor="middle" fill="currentColor" fill-opacity="0.7">flexible</text>
</svg>
</div>

*As model flexibility grows, $\mathrm{Bias}^2$ falls and $\mathrm{Var}$
rises. Their sum is smallest in between (the dot).*

What the standard fixes do:

- **Regularization** stops the fit from following the training set so
  closely: variance drops, some bias is accepted in exchange.
- **More data** lowers variance, like $\sigma^2/n$ (§3.6).
- **Bagging** averages several high-variance models trained on resampled
  data: averaging cancels variance while leaving bias about the same.

For test error there is one more term,
$\mathrm{Bias}^2 + \mathrm{Var} + \text{irreducible noise}$: the noise in the
labels themselves (§3.4), which no model can predict. The full derivation is
in the evaluation module.

## 3.8 Hypothesis tests: is the difference real?

**A hypothesis test asks: if there were no real difference, how often would
noise alone produce a gap this large?** If rarely, call the difference real.
Two examples:

- Model A scores 91.2% and model B 90.5% on the same 1,000 test examples. Is
  A better?
- Checkout page B converts 2.3% of 50,000 users and page A 2.1%. Ship B?

#### The procedure

1. **Assume no real difference.** This is the **null hypothesis** $H_0$: the
   gap is sampling noise.
2. **Measure the gap in error bars:** $z = \text{gap} / \text{SE}$. By the
   CLT (§3.6), $z$ follows the standard bell when $H_0$ holds.
3. **Compute the p-value:** the probability, under $H_0$, of a gap at least
   this large in either direction.
4. **Decide** with a threshold fixed in advance, usually 0.05: if
   $p < 0.05$, call the difference **statistically significant**.

```python
from scipy.stats import norm
p_pool = 0.022                                  # A/B test, both groups pooled
se = np.sqrt(p_pool * (1 - p_pool) * 2 / 50_000)   # 0.000928
z = (0.023 - 0.021) / se                        # 2.16
2 * norm.sf(z)                                  # 0.031  the p-value
```

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 190" role="img" aria-label="Standard normal curve with both tails beyond plus and minus 2.16 shaded; the shaded area is the two-sided p-value of 0.031" style="max-width:420px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="200" y="22" text-anchor="middle" fill="#6366f1" font-weight="600">distribution of z under H₀</text>
  <polygon points="291.8,148.4 296.1,150.7 300.3,152.6 304.6,154.2 308.8,155.5 313.1,156.5 317.3,157.3 321.6,158.0 325.8,158.5 330.1,158.9 334.3,159.2 338.6,159.4 342.8,159.6 347.1,159.7 351.3,159.8 355.6,159.9 359.8,159.9 364.1,159.9 368.3,160.0 370.0,160 291.8,160" fill="#f43f5e" fill-opacity="0.35" stroke="none"/>
  <polygon points="108.2,148.4 103.9,150.7 99.7,152.6 95.5,154.2 91.2,155.5 86.9,156.5 82.7,157.3 78.4,158.0 74.2,158.5 69.9,158.9 65.7,159.2 61.4,159.4 57.2,159.6 52.9,159.7 48.7,159.8 44.4,159.9 40.2,159.9 35.9,159.9 31.7,160.0 30.0,160 108.2,160" fill="#f43f5e" fill-opacity="0.35" stroke="none"/>
  <polyline points="30.0,160.0 34.2,159.9 38.5,159.9 42.8,159.9 47.0,159.8 51.2,159.7 55.5,159.6 59.8,159.5 64.0,159.3 68.2,159.0 72.5,158.7 76.8,158.2 81.0,157.6 85.2,156.9 89.5,155.9 93.8,154.7 98.0,153.3 102.3,151.5 106.5,149.4 110.8,146.8 115.0,143.8 119.2,140.3 123.5,136.3 127.8,131.8 132.0,126.7 136.2,121.1 140.5,115.1 144.8,108.6 149.0,101.7 153.2,94.6 157.5,87.4 161.8,80.2 166.0,73.1 170.2,66.3 174.5,60.0 178.8,54.4 183.0,49.5 187.2,45.6 191.5,42.7 195.8,40.9 200.0,40.3 204.2,40.9 208.5,42.7 212.8,45.6 217.0,49.5 221.2,54.4 225.5,60.0 229.8,66.3 234.0,73.1 238.2,80.2 242.5,87.4 246.8,94.6 251.0,101.7 255.2,108.6 259.5,115.1 263.8,121.1 268.0,126.7 272.2,131.8 276.5,136.3 280.8,140.3 285.0,143.8 289.2,146.8 293.5,149.4 297.8,151.5 302.0,153.3 306.2,154.7 310.5,155.9 314.8,156.9 319.0,157.6 323.2,158.2 327.5,158.7 331.8,159.0 336.0,159.3 340.2,159.5 344.5,159.6 348.8,159.7 353.0,159.8 357.2,159.9 361.5,159.9 365.8,159.9 370.0,160.0" fill="none" stroke="#6366f1" stroke-width="2"/>
  <line x1="30" y1="160" x2="370" y2="160" stroke="currentColor" opacity="0.5"/>
  <line x1="291.8" y1="160" x2="291.8" y2="70" stroke="#f43f5e" stroke-dasharray="4 4"/>
  <line x1="108.2" y1="160" x2="108.2" y2="70" stroke="#f43f5e" stroke-dasharray="4 4"/>
  <text x="200" y="176" text-anchor="middle" fill="currentColor">0</text>
  <text x="291.8" y="176" text-anchor="middle" fill="#f43f5e">z = 2.16</text>
  <text x="108.2" y="176" text-anchor="middle" fill="#f43f5e">−2.16</text>
  <text x="300" y="64" fill="#f43f5e" font-size="10.5">shaded: p = 0.031</text>
</svg>
</div>

*The A/B test lands at $z = 2.16$. The p-value is the shaded area: how often
noise alone would push $z$ that far from zero in either direction.*

#### Two proportions, derived

Each conversion rate is an average of 0/1 outcomes, a Bernoulli average with
variance $p(1-p)/n$ (§3.2, §3.6). The two groups are separate users, so the
variance of the difference is the sum (§3.1). Under $H_0$ both rates equal
one $p$, estimated by pooling everyone:

$$\text{SE} = \sqrt{\hat p(1-\hat p)\Big(\frac{1}{n_A} + \frac{1}{n_B}\Big)},
  \qquad z = \frac{\hat p_B - \hat p_A}{\text{SE}}$$

**The A/B test** gives $z = 2.16$, $p = 0.031$, as computed above:
statistically significant, with the caveats below.

**The two models**, treated the same way:

$$\text{SE} = \sqrt{\tfrac{0.912 \times 0.088}{1000} + \tfrac{0.905 \times 0.095}{1000}} \approx 0.0129,
  \qquad z = \frac{0.007}{0.0129} \approx 0.54, \qquad p \approx 0.59$$

Nowhere near significant: a 0.7-point gap on 1,000 examples is within noise.
The computation is also slightly off, because both models were scored on the
*same* examples. The paired version, **McNemar's test**, uses only the
examples where the models disagree. If A is right and B wrong on $b$ of them,
and the reverse on $c$, then under $H_0$ each disagreement is a fair coin and
$z = (b - c)/\sqrt{b + c}$. A 7-example gap is significant only with at most
12 disagreements in total ($7/\sqrt{12} \approx 2.0$). With the hundreds of
disagreements typical in practice, it is noise.

#### What a p-value is not

**A p-value is the probability of data this extreme if there is no effect.
It is not the probability that there is no effect.** Turning one into the
other is §3.3's problem: it needs a prior on how often tested ideas are real.
A team that tests many long shots will find that most of its $p < 0.05$
results are false alarms, just like most positive tests in §3.3's
rare-disease example.

#### Errors, power and sample size

- **False positive (type I error):** calling a difference real when it is
  not. Its rate is the threshold, 5%.
- **Miss (type II error):** failing to detect a real effect. The chance of
  detecting it is the test's **power**.

**Power decides the sample size.** To be significant, the gap must exceed
1.96 SE. For a real effect $\delta$ to clear that bar 80% of the time, it must
sit another 0.84 SE higher (80% of a bell lies below 0.84 standard
deviations above its center). So $\delta \ge 2.8$ SE. With
$\text{SE} \approx \sqrt{2p(1-p)/n}$ for two equal groups:

$$n \;\ge\; \frac{2 \cdot 2.8^2 \cdot p(1-p)}{\delta^2} \quad\text{per group}$$

For the checkout test ($p \approx 0.022$, a lift worth detecting of
$\delta = 0.002$) that is about **84,000 per group**. The test ran 50,000, so
it was underpowered: a real 0.2-point lift sits only 2.16 SE from zero, and
the power is $P(Z > 1.96 - 2.16) \approx 58\%$. Such a lift would be missed
about 4 times in 10. And the lifts an underpowered test does catch are
biased upward, since only the samples that happened to overshoot clear the
bar.

#### Two ways to fool yourself

- **Testing many things.** Test 20 metrics and, even if nothing is real, the
  chance that at least one comes out significant is
  $1 - 0.95^{20} \approx 0.64$. Pick one primary metric in advance, or divide
  the threshold by the number of tests (Bonferroni).
- **Peeking.** Checking the p-value daily and stopping the first time it dips
  below 0.05 is many tests disguised as one. Fix the sample size in advance
  from the power calculation, or use a test designed for repeated looks.

A confidence interval says the same thing and also shows the size: the 95%
interval for the gap is $\text{gap} \pm 1.96$ SE, and it excludes 0 exactly
when $p < 0.05$. For the A/B test it is $0.002 \pm 0.0018$: the lift is
somewhere between 0.02 and 0.38 points. Significant, but its size is still
almost unknown.
