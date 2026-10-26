# Probability & statistics for ML

This chapter answers three questions that come up constantly in ML:

- **Why is the loss what it is?** Cross-entropy, MSE and L1 each follow from
  an assumption about how the labels are generated. Know the assumption and
  you know when the loss is the wrong choice (§3.3, §3.4).
- **Is this improvement real?** Model B beats model A by 0.5% on 1,000 test
  examples. Accuracy measured on 1,000 examples varies by about ±0.9% just
  from which examples happened to be in the test set, so a 0.5% gap can be
  luck (§3.6, §3.8).
- **Why does my model miss?** A model can be wrong because it is too simple
  for the pattern (bias), or because it fits the quirks of its particular
  training set (variance). Regularization, more data and bagging each
  reduce one of the two (§3.5).

## 3.1 Random variables, expectation, variance

A **random variable** is a number attached to a random outcome: a die roll
is $X \in \{1, \dots, 6\}$, each value with probability $1/6$. Its
**expectation** is the long-run average if you repeated the experiment
forever: each value times its probability, summed.

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

#### Linearity

**The average of a sum is the sum of the averages, always:**

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

This holds even when $X$ and $Y$ are dependent, because both sides are the
same weighted sum with the terms grouped differently. The chapter uses it
constantly: the variance shortcut below, the Binomial mean $np$ (§3.4), and
the fact that a sample mean is right on average (§3.5) are each one
application of it.

Products are different. $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$ only
when $X$ and $Y$ are **independent**, meaning
$P(X = x, Y = y) = P(X = x)\,P(Y = y)$ for all $x, y$. Then the double sum
splits into two single sums:

$$\mathbb{E}[XY] = \sum_{x,y} x\,y\,P(x)\,P(y) = \Big(\sum_x x\,P(x)\Big)\Big(\sum_y y\,P(y)\Big) = \mathbb{E}[X]\,\mathbb{E}[Y]$$

#### Variance

**Variance measures spread: the expected squared distance of a draw from
the true mean** $\mu = \mathbb{E}[X]$ (3.5 for the die):

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]$$

"Mean" and "average" are the same operation, but probability applies it to
two different things:

- The **true mean** averages over every possible outcome, weighted by its
  probability: $(1 + 2 + \dots + 6)/6 = 3.5$ for the die. It is one fixed
  number.
- The **sample mean** averages the draws you actually got: rolls $2, 6, 3$
  give $11/3 \approx 3.67$. Another set of rolls gives another number.

$\mathbb{E}[\,\cdot\,]$ always means the first kind, so the variance above
uses the true mean and weights every outcome by its probability. Estimating
a variance from an actual sample is a different problem, with a twist of its
own (§3.5).

An easier form to compute comes from expanding the square and applying
linearity ($\mu$ is a constant, and $\mathbb{E}[X] = \mu$):

$$\mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2
  = \mathbb{E}[X^2] - \mu^2$$

For the die, $\mathbb{E}[X^2] = \tfrac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \tfrac{91}{6} \approx 15.17$,
so $\mathrm{Var}(X) = 15.17 - 3.5^2 \approx 2.92$.

#### Two scaling rules

- **Scaling: $\mathrm{Var}(aX) = a^2\,\mathrm{Var}(X)$.** Multiplying by $a$
  multiplies every deviation from the mean by $a$, so every squared
  deviation by $a^2$. Variance is in squared units, so the constant comes
  out squared.
- **Adding independent variables: $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$.**
  Write $u = X - \mu_X$ and $v = Y - \mu_Y$. The deviation of $X + Y$ is
  $u + v$, and $(u + v)^2 = u^2 + 2uv + v^2$, so

  $$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + 2\,\mathbb{E}[uv] + \mathrm{Var}(Y)$$

  The middle expectation is the **covariance**. Multiplying it out gives
  $\mathbb{E}[uv] = \mathbb{E}[XY] - \mu_X\mu_Y$, which is 0 for independent
  variables by the product rule above. So the variances simply add.

This additivity is behind the $\sigma/\sqrt{n}$ error bars of §3.6 and
behind variance-preserving weight initialization (Xavier/He: choose the
weight variance so each layer's outputs keep unit variance).

## 3.2 Bayes rule - derived, not memorized

**Bayes rule works backward from what you can measure to what you want to
know.** A lab can measure how often a test is positive for people known to be
sick. A patient wants the reverse: how often a positive result means sick.
These two numbers are very different, and confusing them is the classic
mistake.

#### The rare-disease test

A disease affects 1 person in 100. A test for it is good but not perfect:

- If you **are sick**, the test comes back positive 99% of the time.
- If you **are healthy**, it still comes back positive 5% of the time, a
  false alarm.

These are two separate numbers, one for each group. Missing a sick person
and flagging a healthy one have different causes, so a test can be very
good at one and only fairly good at the other.

You test positive. What is the chance you are actually sick? Most people
guess about 99%. Count through 10,000 people instead:

- 100 are sick, and 99% of them test positive: **99 true positives**.
- 9,900 are healthy, and 5% of them test positive: **495 false alarms**.
- So 594 people test positive, and only 99 of them are sick:
  $99 / 594 \approx 16.7\%$.

The test catches nearly every sick person, but healthy people are so much
more common (9,900 vs 100) that even a 5% false-alarm rate produces five
false alarms for every real case.

The three starting numbers have standard names. One describes the people
being tested; the other two describe the test:

| Name | Value | Describes |
|---|---|---|
| **Prevalence** | 1% | the people: how common the disease is among them |
| **Sensitivity** | 99% | the test: the share of sick people it flags |
| **Specificity** | 95% | the test: the share of healthy people it clears (so 5% false alarms) |

In ML terms this is precision under class imbalance. It is why accuracy is
useless for rare events, and why a 99%-accurate fraud model can still flag
mostly innocent transactions. (`probability_demos.ipynb` runs it with raw
counts.)

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

#### The rule

The counting above is Bayes rule; writing it as a formula only needs
notation. $P(\text{sick} \mid +)$, read "probability of sick given
positive", means: among the people who test positive, the fraction who are
sick. The count computed exactly that:

$$P(\text{sick} \mid +) = \frac{\text{sick and positive}}{\text{all positive}} = \frac{99}{594} \approx 16.7\%$$

Now write each count with the numbers you started from:

- **Sick and positive:** 1% are sick, and 99% of those test positive, so
  $P(+ \mid \text{sick})\, P(\text{sick}) = 0.99 \times 0.01 = 0.0099$
  (99 out of 10,000).
- **All positive:** add both ways of testing positive, sick or healthy:
  $P(+) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0594$ (594 out of 10,000).

Divide the first by the second, and that is Bayes rule:

$$P(\text{sick} \mid +) = \frac{P(+ \mid \text{sick})\; P(\text{sick})}{P(+)} = \frac{0.0099}{0.0594} \approx 16.7\%$$

In general, for a hypothesis $A$ and evidence $B$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

Each piece has a name:

- **Prior** $P(A)$: belief before the evidence. $P(\text{sick}) = 1\%$.
- **Likelihood** $P(B \mid A)$: how often this evidence shows up when the
  hypothesis is true. $P(+ \mid \text{sick}) = 99\%$.
- **Posterior** $P(A \mid B)$: belief after the evidence, the number you
  want. $P(\text{sick} \mid +) \approx 17\%$.
- **Evidence** $P(B)$: how often the evidence shows up at all, from every
  cause. $P(+) = 5.94\%$.

So the **prevalence is the prior**, and the test's sensitivity and
false-positive rate are the **likelihoods**.

**Why the rule is needed:** the likelihoods belong to the test, so they are
the same wherever it is used. The posterior also depends on who is being
tested. Use the same test on people who already have symptoms, where say 10%
are sick:

$$P(\text{sick} \mid +) = \frac{0.99 \times 0.10}{0.99 \times 0.10 + 0.05 \times 0.90} \approx 69\%$$

Same test, same positive result: 17% in the general population, 69% among
people with symptoms. Bayes rule turns the test's fixed numbers into the
answer for whichever group you are in.

#### Today's posterior is tomorrow's prior

**Each new piece of evidence updates the current belief, starting from
wherever the last update left it.** Suppose the positive patient takes the
test again, independently, and is positive again. Start from the 594 people
who tested positive once:

- Of the 99 sick, 99% test positive again: about 98.
- Of the 495 healthy, 5% test positive again: about 25.

So $P(\text{sick} \mid \text{two positives}) = 98 / (98 + 25) \approx 80\%$.

The same thing with Bayes rule. Writing $+_2$ for "the second test is
positive":

$$P(\text{sick} \mid +_2) = \frac{P(+_2 \mid \text{sick})\, P(\text{sick})}{P(+_2 \mid \text{sick})\, P(\text{sick}) + P(+_2 \mid \text{healthy})\, P(\text{healthy})}$$

Plug in the values. The prior is now the first test's posterior; the
likelihoods are the test's, unchanged:

- $P(\text{sick}) = 99/594 = 1/6$, so $P(\text{healthy}) = 5/6$
- $P(+_2 \mid \text{sick}) = 0.99$ and $P(+_2 \mid \text{healthy}) = 0.05$

$$P(\text{sick} \mid +_2) = \frac{0.99 \times \tfrac{1}{6}}{0.99 \times \tfrac{1}{6} + 0.05 \times \tfrac{5}{6}} = \frac{0.165}{0.165 + 0.042} \approx 0.798 \approx 80\%$$

The first test moved the probability from 1% to 17%; the second, starting
from 17%, moved it to 80%. The 17% is the posterior of the first test and the
prior of the second.

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

*$P(\text{sick})$ at each stage, green portion of each bar. Every update
keeps only the people consistent with the new evidence: 10,000 shrink to the
594 who tested positive, then to the 123 who tested positive twice, and the
sick share climbs from 1% to 17% to 80%. The middle number is both the
posterior of the first test and the prior of the second.*

## 3.3 MLE, and why minimizing cross-entropy IS maximum likelihood

#### The principle

**Maximum likelihood estimation (MLE): choose the parameters that make the
observed data most probable.** If a coin lands heads 7 times in 10 flips, a
heads probability of 0.7 makes that result more probable than 0.5 does, so
MLE prefers 0.7. "Likelihood" is §3.2's word: the probability of the
evidence (the data) given a hypothesis (a parameter value $\theta$). Written
out, with $\arg\max_\theta$ meaning "the $\theta$ at which this is
largest":

$$\hat\theta = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i \mid x_i)
             = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y_i \mid x_i)$$

Taking the log does not change where the maximum is, because $\log$ is
increasing. It turns the product into a sum, which is numerically stable and
easy to differentiate term by term, so the log form is what gets used.

#### The smallest instance: a coin

A coin has unknown heads probability $p$; you see $h$ heads in $n$ flips.
**MLE says $\hat p = h/n$, the observed frequency.** To derive it, write the
log-likelihood and set its derivative to zero:

$$\log L(p) = h \log p + (n - h)\log(1-p), \qquad
\frac{d}{dp}\log L(p) = \frac{h}{p} - \frac{n-h}{1-p} = 0$$

Multiplying through by $p(1-p)$ gives $h(1-p) = (n-h)\,p$, so $h = np$ and
$\hat p = h/n$. (Checked numerically in `probability_demos.ipynb`.)

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
peak, at $p = h/n = 0.7$.*

#### Cross-entropy is MLE for a classifier

**The cross-entropy loss charges each example $-\log$(the probability the
model put on the true class).** Adding up those charges gives the negative
log of the probability the model gave to all the true labels, so minimizing
the loss is maximizing the likelihood.

Example: three classes (cat, dog, bird) and two training photos.

- Photo 1 is a **cat**; the model outputs $[0.7, 0.2, 0.1]$. Loss:
  $-\log 0.7 = 0.36$.
- Photo 2 is a **dog**; the model outputs $[0.5, 0.2, 0.3]$. Loss:
  $-\log 0.2 = 1.61$.

Total cross-entropy: $0.36 + 1.61 = 1.97$.

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
true class, and charges $-\log(\text{that bar})$. A confident correct answer
(0.7) costs little; a doubtful one (0.2) costs a lot.*

Now compute the likelihood of the same data: the probability the model gave
to both observed labels. With independent examples it is the product,
$L = 0.7 \cdot 0.2 = 0.14$, and

$$-\log L = -\log 0.7 - \log 0.2 = 0.36 + 1.61 = 1.97$$

The same numbers, term for term. In general, cross-entropy compares a
target distribution $q$ with the model's distribution $p$ over the classes,

$$H(q, p) = -\sum_k q_k \log p_k$$

and with a **one-hot** target ($q_k = 1$ for the true class, 0 otherwise)
only the true class's term survives. Written with the one-hot label of
example $i$ as $y_{ik}$:

$$\underbrace{-\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)}_{\text{cross-entropy loss}} \;=\; \underbrace{-\sum_i \log p_\theta(y_i \mid x_i)}_{\text{negative log-likelihood}}$$

So **minimizing cross-entropy is maximizing the likelihood of the observed
labels**: MLE for a categorical output. This needs one-hot targets. With
soft targets such as label smoothing, $q = [0.9, 0.05, 0.05]$, the
cross-entropy formula still works but is no longer the likelihood of any
observed label.

#### The same pattern gives the regression losses

**MSE and L1 are also negative log-likelihoods, once you say how the target
scatters around the prediction: Gaussian scatter gives MSE, Laplace scatter
gives L1.**

The problem first. MLE needs $p_\theta(y \mid x)$, a probability for the
observed target. A classifier outputs one, but a regression model outputs a
single number: predicting an apartment's price at 300k does not say how
likely a sale at 310k or 295k would be. The fix is to model the gap. The
actual price differs from the prediction by everything the input does not
capture (floor, condition, the buyer's mood), so write

$$y = f_\theta(x) + \varepsilon$$

where $\varepsilon$ is the **noise**. (Measured after the fact, the same gap
is the **residual**.) Assume a distribution for $\varepsilon$, say centered
at 0. Then a sale at 310k is as likely as a noise draw of $+10$k, and

$$p_\theta(y \mid x) = p_\varepsilon\big(y - f_\theta(x)\big)$$

(The noise is continuous, so $p_\varepsilon$ is a **density**: not a
probability of one exact value, but a curve that is high where values are
likely and low where they are not.)

Now take $-\log$ per example, exactly as for the classifier. The noise
density decides which loss comes out:

- **Gaussian noise**, density
  $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\varepsilon^2 / 2\sigma^2}$. The
  $-\log$ turns the exponential into its exponent and the front factor into a
  constant:

  $$-\log p_\theta(y \mid x) = \frac{(y - f_\theta(x))^2}{2\sigma^2}
  + \underbrace{\tfrac{1}{2}\log(2\pi\sigma^2)}_{\text{no } \theta \text{: a constant}}$$

  Minimizing it over $\theta$ is minimizing squared error: **MSE**.
- **Laplace noise**, density $\frac{1}{2b} e^{-|\varepsilon| / b}$. The same
  steps give

  $$-\log p_\theta(y \mid x) = \frac{|y - f_\theta(x)|}{b} + \log(2b)$$

  absolute error plus a constant: **L1 loss**.

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

*Left: the two noise distributions on matching scales, the Gaussian bell and
the Laplace with its sharper peak and heavier tails. Right: their $-\log$
densities, which are the per-example losses: the parabola
$\varepsilon^2/2$ (MSE) and the V shape $|\varepsilon|$ (L1).*

So choosing a regression loss is choosing a noise model:

- **MSE** assumes Gaussian noise: large residuals are very rare, so they are
  punished quadratically.
- **L1** assumes Laplace noise: heavier tails, so outliers are expected and
  punished only linearly.

When an interviewer asks "why this loss?", name the assumption: MSE for
Gaussian noise, L1 for Laplace noise, cross-entropy for a categorical label.

## 3.4 Bernoulli and Gaussian: the two distributions ML leans on

Two distributions answer the two commonest modeling questions:

- What does **one yes/no event** look like? **Bernoulli**: the coin flips
  and classifier labels of §3.3.
- What does the **sum of many small independent effects** look like?
  **Gaussian**: regression noise (§3.3) and the limit of averages (§3.6).

#### Bernoulli: one yes/no event

Code the two outcomes as 1 and 0, with $p$ the probability of 1:

$$P(X = 1) = p, \qquad P(X = 0) = 1 - p$$

This is **Bernoulli($p$)**. Each coin flip in §3.3 is one **draw** from it.

- **Mean:** $\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1 - p) = p$.
- **Variance:** a 0/1 variable has $X^2 = X$, so $\mathbb{E}[X^2] = p$ and
  $\mathrm{Var}(X) = p - p^2 = p(1 - p)$.

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

The variance is largest for a fair coin ($1/4$ at $p = 1/2$), where the
outcome is hardest to predict, and falls to 0 as one outcome becomes certain.

Three things built from it:

- **Count the 1s: the Binomial.** The number of successes in $n$
  independent tries, $S = X_1 + \dots + X_n$, has mean $np$ (linearity) and
  variance $np(1 - p)$ (independent variances add). §3.3's "$h$ heads in $n$
  flips" was one Binomial draw.
- **Model a binary label: binary cross-entropy.** A classifier that outputs
  $\hat y = 0.9$ for "spam" models the label as Bernoulli($0.9$). The
  probability it gives the true label $y \in \{0, 1\}$ is $\hat y$ if
  $y = 1$ and $1 - \hat y$ if $y = 0$. Taking $-\log$, both cases in one
  line:

  $$-\log p_\theta(y \mid x) = -\big[\, y \log \hat y + (1 - y) \log(1 - \hat y) \,\big]$$

  This is the **binary cross-entropy** loss: MLE for a Bernoulli label, the
  two-class case of §3.3.
- **Randomize a network: dropout.** Dropout keeps each unit with
  probability $p$ and zeroes it otherwise, so each keep/drop decision is a
  Bernoulli($p$) draw and a layer of $n$ units keeps $np$ on average.
  (Frameworks usually quote the drop probability: PyTorch's
  `nn.Dropout(0.1)` drops with probability 0.1, keep probability
  $p = 0.9$.)

#### Gaussian: the sum of many small effects

The **Gaussian** $\mathcal{N}(\mu, \sigma^2)$ has density
$\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$: the bell
curve, with mean $\mu$ and variance $\sigma^2$. Three reasons it is the
default:

1. **Sums of many small effects are bell-shaped.** Picture §3.3's price gap
   as about 20 small independent contributions: floor $+0.8$k, condition
   $-0.3$k, buyer's mood $+0.5$k, timing $-0.2$k, and so on. A total near
   zero is common, because the contributions only need to roughly cancel and
   many combinations do. A total of $+15$k is rare, because nearly all of
   them would have to point the same way. So totals pile up in the middle
   and thin out at the extremes: a bell. §3.6's central limit theorem makes
   this exact. Measurement noise and regression residuals are usually such
   sums, which is why "assume Gaussian noise" so often fits.
2. **It assumes the least.** Among all distributions with a given mean and
   variance, the Gaussian has the most entropy (proved in §3.7), so it adds
   no structure beyond those two numbers.
3. **Its $-\log$ density is a squared distance:**

   $$-\log\Big(\tfrac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(x-\mu)^2 / 2\sigma^2}\Big)
   = \frac{(x-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi\sigma^2)$$

   That is why Gaussian noise gives least squares (§3.3).

## 3.5 Bias and variance of estimators

**An estimate computed from a random sample is itself random. It can miss
the truth in two ways: it can be off on average (bias), or it can jump
around from sample to sample (variance). Its expected squared error is
exactly bias² + variance.**

Names for the pieces: the **target** $\theta$ is the true, fixed number you
want (the die's mean, a coin's heads probability). The **estimator** is a
formula applied to a sample, and the **estimate** $\hat\theta$ is its output.
§3.3's $\hat p = h/n$ is an estimator, and so is every MLE.

Example: estimate the die's mean ($\theta = 3.5$) by rolling 3 times and
averaging, $\hat\theta = \bar X = \frac{1}{3}(X_1 + X_2 + X_3)$. Different
samples give different estimates:

- rolls $2, 6, 3 \;\rightarrow\; 3.67$
- rolls $1, 2, 4 \;\rightarrow\; 2.33$
- rolls $5, 3, 6 \;\rightarrow\; 4.67$

So $\hat\theta$ is a random variable, with an expectation (the average
estimate over all possible samples) and a variance. Notation: capital
$X_1, \dots, X_n$ and $\bar X$ are the draws before they land (random); lower
case $x_1, \dots, x_n$ and $\bar x$ are one sample's actual numbers, such as
$2, 6, 3$ with $\bar x = 3.67$.

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

- **Bias** is a systematic miss: averaging estimates from thousands of fresh
  samples still would not land on $\theta$. The die average has zero bias:
  by linearity, $\mathbb{E}[\bar X] = \frac{1}{3}(3.5 + 3.5 + 3.5) = 3.5$.
- **Variance** is the sample-to-sample jumpiness, like the spread from 2.33
  to 4.67 above. By §3.1's rules: one roll has variance $2.92$, so the sum
  of three independent rolls has $3 \times 2.92 = 8.76$, and the factor
  $\frac{1}{3}$ comes out squared: $\mathrm{Var}(\bar X) = 8.76/9 \approx 0.97$.

To score an estimator with one number, use the **mean squared error**,
$\mathrm{MSE} = \mathbb{E}[(\hat\theta - \theta)^2]$. Split the miss into
"jump around the average" plus "bias":

$$\hat\theta - \theta =
  \underbrace{\big(\hat\theta - \mathbb{E}[\hat\theta]\big)}_{a\text{: random}}
  + \underbrace{\big(\mathbb{E}[\hat\theta] - \theta\big)}_{b\text{: Bias, fixed}}$$

Then $(a + b)^2 = a^2 + 2ab + b^2$. Take expectations: $\mathbb{E}[a^2]$ is
the variance, $\mathbb{E}[2ab] = 2b\,\mathbb{E}[a] = 0$ because $a$ averages to
zero, and $b^2$ is the squared bias. So

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
cloud's center sits from the bullseye; variance is how scattered the cloud
is. The expected squared miss is exactly $\mathrm{Bias}^2 + \mathrm{Var}$.*

#### Estimating a variance from data (Bessel's correction)

**Computing a sample's variance with divisor $n$ underestimates the true
variance on average. Dividing by $n - 1$ fixes it exactly.** The reason:
deviations are measured from the sample's own mean, which always sits closer
to the sample than the true mean does.

**The natural estimator.** The true variance is
$\sigma^2 = \mathbb{E}[(X - \mu)^2]$. From a sample you have neither the
probabilities nor $\mu$, so replace the expectation by the plain average
over the $n$ draws and $\mu$ by the sample mean $\bar X$:

$$\hat\sigma^2 = \frac{1}{n}\sum_i (X_i - \bar X)^2$$

For the rolls $2, 6, 3$: $\bar x = \frac{11}{3}$, the deviations are
$-\frac{5}{3}, \frac{7}{3}, -\frac{2}{3}$, and

$$\hat\sigma^2 = \frac{1}{3}\left[\left(\tfrac{5}{3}\right)^2 + \left(\tfrac{7}{3}\right)^2 + \left(\tfrac{2}{3}\right)^2\right]
  = \frac{1}{3} \cdot \frac{78}{9} = \frac{26}{9} \approx 2.89$$

**It is biased low.** For the die we can check. The true variance is
$\sigma^2 = \frac{35}{12} \approx 2.92$. Apply the formula to all
$6^3 = 216$ equally likely 3-roll samples and average the results: you get
$\frac{35}{18} \approx 1.94$, a third short.

**Why: the sample mean sits too close to its own sample.** Of all possible
points $c$, the sample mean $\bar X$ is the one that makes the sum of squared
deviations $\sum_i (X_i - c)^2$ smallest: as a function of $c$ it is a
parabola with its bottom at $c = \bar X$. The true mean $\mu$ is some other
point, so measuring from $\bar X$ always gives a smaller sum than measuring
from $\mu$ would.

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

*The sum of squared deviations of one sample from a point $c$. The minimum
sits at $c = \bar x$, where the estimator measures from, so the value at
the true mean $c = \mu$ is higher by exactly $n(\bar x - \mu)^2$. The
vertical scale is zoomed in around the minimum.*

**By exactly how much.** Write $X_i - \mu = (X_i - \bar X) + (\bar X - \mu)$,
square, and sum over $i$. The cross term vanishes because deviations from
their own average sum to zero, which leaves

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

Check on the rolls $2, 6, 3$: from $\bar x$ the squares sum to
$\frac{78}{9} \approx 8.67$; from $\mu = 3.5$ they sum to
$1.5^2 + 2.5^2 + 0.5^2 = 8.75$; and the gap $3 \cdot (\frac{1}{6})^2 \approx 0.08$
is the difference.

On average, the first sum on the right is $n\sigma^2$ (each term averages
$\sigma^2$), and the gap is $n \cdot \mathrm{Var}(\bar X) = n \cdot \sigma^2/n = \sigma^2$,
one draw's worth of variance. So

$$\mathbb{E}\Big[\sum_i (X_i - \bar X)^2\Big] = n\sigma^2 - \sigma^2 = (n-1)\,\sigma^2$$

**The fix.**

- Divide by $n$: the average is $\frac{n-1}{n}\sigma^2$, a bias of
  $-\sigma^2/n$. For 3 die rolls that predicts
  $\frac{2}{3} \cdot \frac{35}{12} = \frac{35}{18} \approx 1.94$, exactly the
  enumeration result.
- Divide by $n - 1$: the average is exactly $\sigma^2$. This is **Bessel's
  correction**.

Another way to see the $n - 1$: the $n$ deviations $X_i - \bar X$ must sum to
zero, so once $n - 1$ of them are known the last is fixed. The sample
supplies only $n - 1$ free deviations. With one draw, the single deviation
is $X_1 - \bar X = 0$: the $1/n$ formula claims zero spread, while Bessel's
$0/0$ is undefined, which is the honest answer.

#### The tradeoff in ML terms

Training a model fits the same template:

- the **sample** is the training set,
- the **estimator** is the whole training procedure,
- the **estimate** is the trained model's prediction at an input $x$,
- **redrawing the sample** means collecting a new training set and
  retraining.

So:

- **Bias**: average the predictions at $x$ over many training sets; bias is
  how far that average is from the truth. It comes from what the model class
  cannot express.
- **Variance**: how much the prediction at $x$ changes between training
  sets. It comes from the fit following the quirks and noise of the one
  sample it saw.

The two classic model types sit at opposite corners of the dartboard
figure:

- A **linear model** is high-bias, low-variance. If the truth curves, a line
  misses it in the same places whichever training set it saw, but refitting
  gives nearly the same line.
- A **deep net or unpruned decision tree** is low-bias, high-variance. It can
  fit nearly any shape, so on average it tracks the truth, but each fit also
  follows its own training set's noise, so two fits disagree.

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

*As model flexibility grows, $\mathrm{Bias}^2$ falls (the model can express
more) and $\mathrm{Var}$ rises (the fit can follow more noise). Their sum is
smallest at an intermediate flexibility (the dot).*

What the standard fixes do:

- **Regularization** stops the fit from following the training set so
  closely: variance drops, some bias is accepted in exchange.
- **More data** lowers variance directly, like $\mathrm{Var}(\bar X) = \sigma^2/n$.
- **Bagging** averages several high-variance models trained on resampled
  data: averaging cancels variance while leaving bias about the same.

For test error the decomposition gains one term:
$\mathrm{Bias}^2 + \mathrm{Var} + \text{irreducible noise}$. The new term is
the noise $\varepsilon$ in the target itself (§3.3), which no model can
predict. The full derivation is in the evaluation module.

## 3.6 Central limit theorem - the $\sqrt{n}$ intuition

**Average $n$ independent draws and two things happen: the average's
typical error shrinks like $\sigma/\sqrt{n}$, and its distribution becomes
approximately Gaussian, whatever distribution the draws came from.** The
second fact is the **central limit theorem (CLT)**.

This matters because almost every number in ML is an average over a sample:
a sample mean (§3.5), a test accuracy (an average of 0/1 scores), a
minibatch gradient (an average of per-example gradients). The standard
deviation of such an average is called its **standard error**, and it is
the error bar on every number computed from a sample.

#### The $\sigma/\sqrt{n}$ error bar

Take $n$ draws $X_1, \dots, X_n$ that are **i.i.d.** (independent and
identically distributed):

- **Identically distributed**: all from the same distribution, with the
  same mean $\mu$ and variance $\sigma^2$. Rolling one die $n$ times
  qualifies.
- **Independent**: no draw affects another. Dealing cards without
  replacement does not qualify, since each card changes what can come next.

Their average $\bar X = \frac{1}{n}\sum_i X_i$ has, by §3.1's rules:

- mean $\mathbb{E}[\bar X] = \mu$ (linearity), and
- variance $\mathrm{Var}(\bar X) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$.

So the standard error is $\sigma/\sqrt{n}$. It goes to zero as $n$ grows
(the **law of large numbers**), but slowly: halving the error bar takes 4×
the data.

#### The Gaussian shape

The CLT says the standardized average approaches a standard Gaussian:

$$\frac{\bar X - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

The draws can come from any distribution with a mean and a variance. Example:
the exponential distribution (density $e^{-x}$, mean 1) is lopsided, highest
at zero with a long right tail. Watch the distribution of $\bar X$ as $n$
grows:

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

*The distribution of $\bar X$ for averages of $n$ exponential draws (curves
scaled to matching heights). The curve narrows around $\mu = 1$ (that is
$\sigma/\sqrt{n}$), and the lopsided shape becomes a symmetric bell (that is
the CLT).*

**Why sums become bell-shaped.** Take the smallest case, two dice. A total
of 7 can happen six ways (1+6, 2+5, ..., 6+1), but a total of 2 only one way
(1+1), and 12 only one way (6+6). Middle totals have many combinations,
extreme totals few, so the flat distribution of one die becomes a triangle
for two. The figure shows the same counting as adding six shifted copies of
one die's distribution, one for each value of the second die. That
shift-and-add is called a **convolution**, and it is how the distribution of
any sum is built from the distributions of its parts.

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

*Top: six copies of one die's flat distribution; the copy for second-die
value $v$ is slid right by $v$, covering totals $v+1$ to $v+6$. Bottom:
adding the rows column by column counts the ways to make each total (out of
36). Total 7 sits under all six copies, totals 2 and 12 under one each. The
flat shape has become a triangle.*

Each extra draw repeats the shift-and-add, with three effects:

- **The starting shape fades.** Each convolution smooths the shape again,
  so features of the original distribution, like the exponential's spike at
  zero, blur away.
- **Mean and variance survive.** Means add and independent variances add,
  whatever the shape.
- **The Gaussian is where it stops.** Convolving two Gaussians gives a
  Gaussian, so once the shape is a bell, more draws leave it a bell.

§3.4's argument (many combinations give a middling total, few give an
extreme one) is the same counting, for small continuous contributions.
`probability_demos.ipynb` tracks it numerically: the skewness of $\bar X$
falls toward 0 and its standard deviation matches $\sigma/\sqrt{n}$.

#### Where ML uses it

- **SGD noise.** A minibatch gradient is a sample mean of per-example
  gradients, so its noise around the full-data gradient shrinks like
  $1/\sqrt{\text{batch size}}$: quadrupling the batch halves the noise.
- **Error bars on test accuracy.** Each test example scores 1 or 0, a
  Bernoulli draw with variance $p(1-p)$ (§3.4), so the accuracy over $n$
  examples has standard error $\sqrt{p(1-p)/n}$. With $n = 1000$ and
  accuracy near 90%, that is $\sqrt{0.9 \cdot 0.1 / 1000} \approx 0.009$, an
  error bar of about $\pm 0.9\%$. A "0.5% improvement" on such a test set is
  within sampling noise.
- **Confidence intervals and A/B tests** use ranges like "$\pm 2$ standard
  errors". The 2 comes from the Gaussian shape: about 95% of a Gaussian lies
  within 2 standard deviations of its mean (95.4%; exactly 95% at 1.96, the
  value §3.8 uses).

## 3.7 Information theory: what the cross-entropy number means

§3.3 showed that cross-entropy is the right loss. This section says what its
value means, and the same few definitions explain the KL penalty in RLHF
(module 7), the floor a label-smoothed loss never crosses, and why the
Gaussian assumes the least (§3.4).

#### Surprise and entropy

**The surprise of an outcome with probability $p$ is $-\log p$.** Certain
outcomes ($p = 1$) have surprise 0, rarer outcomes have more, and two
independent outcomes have probability $pq$ with surprise
$-\log(pq) = -\log p - \log q$: surprises add. Natural logs give **nats**,
base-2 logs give **bits** ($1$ nat $\approx 1.44$ bits). This chapter uses
nats, which is what `torch` losses report.

**Entropy is the average surprise of a distribution:**

$$H(p) = \mathbb{E}[-\log p(X)] = -\sum_x p(x)\log p(x)$$

- Fair die: $H = \log 6 \approx 1.79$ nats ($2.58$ bits).
- Fair coin: $H = \log 2$ nats $= 1$ bit.
- Coin with $p = 0.9$: $H = -(0.9\log 0.9 + 0.1\log 0.1) \approx 0.325$ nats
  ($0.469$ bits). More predictable than the fair coin, so less entropy.

For a coin, entropy against $p$ has the same shape as §3.4's variance
parabola, for the same reason: both measure how unpredictable one draw is.

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
hardest to predict and has the most entropy, exactly one bit.*

#### Cross-entropy and KL divergence

Now two distributions over the same outcomes: $q$, the truth, and $p$, a
model of it.

- **Cross-entropy** is the average surprise when outcomes come from $q$ but
  are scored with $p$. This is §3.3's loss, with $q$ the one-hot label and
  $p$ the model's output.

  $$H(q, p) = -\sum_x q(x)\log p(x)$$

- **KL divergence** is the extra surprise caused by scoring with the wrong
  distribution.

  $$\mathrm{KL}(q \,\|\, p) = \sum_x q(x)\log\frac{q(x)}{p(x)}$$

Splitting $\log\frac{q}{p} = \log q - \log p$ connects the three:

$$H(q, p) = H(q) + \mathrm{KL}(q \,\|\, p)$$

In words: the loss is the labels' own uncertainty plus the extra caused by
the model being wrong. Check with $q = [0.5, 0.5]$, $p = [0.9, 0.1]$:

- $H(q) = \log 2 = 0.693$
- $\mathrm{KL}(q\|p) = 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1} = 0.5(-0.588) + 0.5(1.609) = 0.511$
- $H(q, p) = -(0.5\log 0.9 + 0.5\log 0.1) = 1.204 = 0.693 + 0.511$

#### KL is never negative, and everything that follows from it

**$\mathrm{KL}(q\|p) \ge 0$, and it is 0 only when $p = q$** (Gibbs'
inequality). The proof uses $\log x \le x - 1$, true for every $x > 0$: the
line $x - 1$ touches $\log x$ at $x = 1$ and lies above it everywhere else.
With $x = p/q$:

$$-\mathrm{KL}(q \,\|\, p) = \sum_x q(x)\log\frac{p(x)}{q(x)}
  \;\le\; \sum_x q(x)\Big(\frac{p(x)}{q(x)} - 1\Big)
  = \sum_x p(x) - \sum_x q(x) = 1 - 1 = 0$$

Equality needs $p(x) = q(x)$ everywhere. Three facts used daily follow:

- **The loss has a floor.** Since $H(q, p) = H(q) + \mathrm{KL} \ge H(q)$,
  the cross-entropy can never go below the labels' entropy. With one-hot
  labels that floor is 0. With label smoothing $q = [0.9, 0.05, 0.05]$ it is
  $H(q) = -(0.9\log 0.9 + 2 \cdot 0.05\log 0.05) \approx 0.394$ nats, which is
  why a smoothed training loss levels off above zero even when the model is
  right on every example.
- **The uniform distribution has the most entropy.** For $u$ uniform over
  $K$ outcomes, $\mathrm{KL}(q\|u) = \log K - H(q) \ge 0$, so
  $H(q) \le \log K$. The die's $\log 6$ is the most any six-outcome
  variable can have.
- **KL is not symmetric.** Swap the two distributions in the check above:
  $\mathrm{KL}(p\|q) = 0.9\log\frac{0.9}{0.5} + 0.1\log\frac{0.1}{0.5} = 0.368 \ne 0.511$.
  The direction matters. Minimizing $\mathrm{KL}(q\|p)$ over $p$ charges
  heavily wherever $q$ has mass and $p$ does not, so $p$ spreads out to
  cover all of $q$. Minimizing $\mathrm{KL}(p\|q)$ charges wherever $p$ has
  mass and $q$ does not, so $p$ concentrates on one mode of $q$.

#### Where these appear in ML

- **MLE minimizes a KL.** The average negative log-likelihood over the data
  approximates $H(q, p_\theta) = H(q) + \mathrm{KL}(q\|p_\theta)$, where $q$
  is the data distribution (§3.6: a sample average approximates the
  expectation). $H(q)$ does not depend on $\theta$, so maximizing likelihood
  minimizes $\mathrm{KL}(q\|p_\theta)$: the covering direction, which is why
  an MLE-trained model spreads probability over every mode in the data.
- **RLHF's KL penalty** (module 7, §3.2) is
  $\mathrm{KL}(\pi_\theta\|\pi_{ref})$, the other direction: the fine-tuned
  policy pays wherever it puts mass the reference would not, so it can
  sharpen but not wander.
- **Knowledge distillation** trains a student on
  $H(p_{teacher}, p_{student})$ instead of one-hot labels. The floor is now
  $H(p_{teacher}) > 0$, and the student learns the teacher's uncertainty,
  not just its top choice.

#### Mutual information

**Mutual information measures how much knowing one variable tells you about
another.** It compares the joint distribution with what independence would
give:

$$I(X; Y) = \mathrm{KL}\big(p(x, y) \,\|\, p(x)\,p(y)\big) = \sum_{x, y} p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)}$$

By Gibbs' inequality $I \ge 0$, with $I = 0$ exactly when $X$ and $Y$ are
independent (§3.1). Equivalently, $I(X; Y) = H(X) - H(X \mid Y)$: the
uncertainty in $X$ minus what is left once $Y$ is known. Feature selection
ranks features by it, and the InfoNCE contrastive loss (module 4) is a lower
bound on it.

#### Entropy of a Gaussian, and why it is the maximum

For a density the sum becomes an integral, $H(p) = -\int p(x)\log p(x)\,dx$.
For $\mathcal{N}(\mu, \sigma^2)$, take the expectation of the $-\log$
density from §3.4, using $\mathbb{E}[(X - \mu)^2] = \sigma^2$:

$$H = \mathbb{E}\Big[\frac{(X-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi\sigma^2)\Big]
  = \tfrac{1}{2} + \tfrac{1}{2}\log(2\pi\sigma^2)
  = \tfrac{1}{2}\log(2\pi e\sigma^2)$$

It grows with $\sigma$ and does not depend on $\mu$: shifting a bell does
not make it more or less predictable. For $\sigma = 2$, $H \approx 2.11$
nats.

**Why it is the maximum.** Take any density $p$ with mean $\mu$ and variance
$\sigma^2$, and let $g$ be the Gaussian with the same two numbers. The
cross-entropy $H(p, g)$ only needs $\mathbb{E}_p[(X-\mu)^2] = \sigma^2$, so the
computation above gives $H(p, g) = H(g)$. Then by Gibbs:

$$0 \le \mathrm{KL}(p \,\|\, g) = H(p, g) - H(p) = H(g) - H(p)$$

So $H(p) \le H(g)$: for a given mean and variance, the Gaussian has the most
entropy. Assuming Gaussian noise when you know only the mean and variance
assumes nothing more.

## 3.8 Hypothesis tests: is the difference real?

**A hypothesis test asks: if there were no real difference, how often would
noise alone produce a gap this large?** If rarely, the difference is called
real. Two interview-style examples:

- Model A scores 91.2% and model B 90.5% on the same 1,000 test examples. Is
  A better?
- Checkout page B converts 2.3% of 50,000 users and page A 2.1%. Ship B?

#### The procedure

1. **Null hypothesis $H_0$:** there is no real difference; the gap is
   sampling noise. Everything below assumes $H_0$.
2. **Test statistic:** the gap divided by its standard error,
   $z = \text{gap} / \mathrm{SE}$, the number of standard errors the gap is
   from zero. By the CLT (§3.6), $z$ is about $\mathcal{N}(0, 1)$ under
   $H_0$.
3. **p-value:** the probability under $H_0$ of a gap at least this large in
   either direction, $p = P(|Z| \ge |z|)$.
4. **Decide** against a threshold $\alpha$ fixed in advance, usually
   $0.05$: if $p < \alpha$, reject $H_0$ and call the difference
   **statistically significant**.

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

*The A/B test below lands at $z = 2.16$. The p-value is the shaded area:
the probability that noise alone would push $z$ that far from zero in
either direction.*

#### Two proportions, derived

Both examples compare two rates, $\hat p_A = k_A / n_A$ and
$\hat p_B = k_B / n_B$. Each is a Bernoulli sample mean with variance
$p(1-p)/n$ (§3.4, §3.6), and they come from separate users, so the variance
of their difference is the sum (§3.1):

$$\mathrm{Var}(\hat p_B - \hat p_A) = \frac{p_B(1-p_B)}{n_B} + \frac{p_A(1-p_A)}{n_A}$$

Under $H_0$ both rates equal one $p$, estimated by pooling:
$\hat p = (k_A + k_B)/(n_A + n_B)$. So

$$\mathrm{SE} = \sqrt{\hat p(1-\hat p)\Big(\frac{1}{n_A} + \frac{1}{n_B}\Big)},
  \qquad z = \frac{\hat p_B - \hat p_A}{\mathrm{SE}}$$

**The A/B test.** $n_A = n_B = 50{,}000$, rates $0.021$ and $0.023$, pooled
$\hat p = 0.022$:

$$\mathrm{SE} = \sqrt{0.022 \cdot 0.978 \cdot \tfrac{2}{50{,}000}} \approx 0.000928,
  \qquad z = \frac{0.002}{0.000928} \approx 2.16$$

$P(|Z| \ge 2.16) \approx 0.031 < 0.05$: statistically significant, with the
caveats below.

**The two models.** Treating the accuracies as independent rates:

$$\mathrm{SE} = \sqrt{\tfrac{0.912 \cdot 0.088}{1000} + \tfrac{0.905 \cdot 0.095}{1000}} \approx 0.0129,
  \qquad z = \frac{0.007}{0.0129} \approx 0.54, \qquad p \approx 0.59$$

Nowhere near significant: a 0.7-point gap on 1,000 examples is within noise,
as §3.6's error bar suggested. The computation is also slightly wrong:
both models were scored on the *same* examples, so the accuracies are not
independent. The paired version, **McNemar's test**, uses only the examples
where the models disagree. If A is right and B wrong on $b$ examples, and the
reverse on $c$, then under $H_0$ each disagreement is a fair coin and
$z = (b - c)/\sqrt{b + c}$. A 7-example gap ($b - c = 7$) is significant
only if $\sqrt{b + c} \le 7/1.96$, that is, at most 12 disagreements in
total. With the hundreds of disagreements typical in practice, the gap is
noise.

#### What a p-value is not

**A p-value is $P(\text{data this extreme} \mid H_0)$, not
$P(H_0 \mid \text{data})$**, the probability that there is no real effect.
Turning one into the other is §3.2's problem: it needs a prior on how often
tested changes are real. A team that tests many long-shot ideas will find
that most of its $p < 0.05$ results are false positives, just as most
positive tests in §3.2's rare-disease example were.

#### Errors, power and sample size

Two ways the decision can go wrong:

- **Type I error**: rejecting $H_0$ when it is true, a false positive. Its
  probability is $\alpha$ by construction.
- **Type II error**: missing a real effect of size $\delta$. Its probability
  is $\beta$, and $1 - \beta$ is the test's **power**.

**Power decides the sample size.** To reject at $\alpha = 0.05$, the gap must
exceed $1.96\,\mathrm{SE}$. For a real effect $\delta$ to clear that bar 80%
of the time, it must sit another $0.84\,\mathrm{SE}$ higher (the 80th
percentile of a Gaussian is 0.84 standard deviations above its mean). So
$\delta \ge 2.8\,\mathrm{SE}$. With $\mathrm{SE} \approx \sqrt{2p(1-p)/n}$ for
two equal groups, solve for $n$:

$$n \;\ge\; \frac{2 \cdot 2.8^2 \cdot p(1-p)}{\delta^2} \quad\text{per group}$$

For the checkout test, $p \approx 0.022$ and the lift worth detecting is
$\delta = 0.002$:

$$n \ge \frac{2 \cdot 7.84 \cdot 0.022 \cdot 0.978}{0.002^2} \approx 84{,}000 \text{ per group}$$

The test ran 50,000 per group, so it was underpowered. At that size a real
0.2-point lift sits $0.002 / 0.000928 \approx 2.16$ standard errors from
zero, so the power is $P(Z > 1.96 - 2.16) \approx 58\%$: such a lift would be
missed about 4 times in 10. And the lifts an underpowered test does catch
are biased upward, since only samples that happened to overshoot clear the
bar.

#### Two ways to fool yourself

- **Multiple comparisons.** Test 20 metrics at $\alpha = 0.05$ and, even if
  nothing is real, the chance that at least one comes out significant is
  $1 - 0.95^{20} \approx 0.64$. Pick one primary metric in advance, or
  divide $\alpha$ by the number of tests (Bonferroni).
- **Peeking.** Checking the p-value daily and stopping the first time it
  dips below $0.05$ is many tests disguised as one, with the same inflated
  false-positive rate. Fix the sample size in advance from the power
  calculation, or use a sequential test designed for repeated looks.

A confidence interval says the same thing and also shows the size: the 95%
interval for the gap is $\text{gap} \pm 1.96\,\mathrm{SE}$, and it excludes
0 exactly when the two-sided $p < 0.05$. For the A/B test it is
$0.002 \pm 0.0018$: the lift is somewhere between $0.02$ and $0.38$ points.
Significant, but its size is still almost unknown.
