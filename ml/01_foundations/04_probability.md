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

Notation first: $A \cap B$ is the event "$A$ *and* $B$ both happen", and
$P(A \mid B)$ is read "the probability of $A$ given $B$": the probability of
$A$ once you *know* that $B$ happened.

Knowing $B$ happened rules out every outcome outside $B$. The outcomes inside
$B$ keep their old probabilities, but those probabilities now sum to $P(B)$
rather than to 1 - so divide each by $P(B)$ to make them a proper
distribution again. The part of $A$ still possible is the overlap
$A \cap B$. Put together:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Check it by counting, on a die: $A$ = "roll is even" $= \{2, 4, 6\}$, $B$ =
"roll is at least 4" $= \{4, 5, 6\}$. Given $B$, three equally likely
outcomes remain, and $A$ holds for two of them (4 and 6), so the answer must
be $2/3$. The formula agrees: $P(A \cap B) = P(\{4, 6\}) = 2/6$, $P(B) =
3/6$, ratio $= 2/3$. ✓

There is nothing to prove in the equation above: it is the *definition* of
$P(A \mid B)$. The die check only confirms that the definition gives the
same answer as counting outcomes directly. That is what makes it a safe
starting point for the derivation below.

Multiply the definition through by $P(B)$ to get
$P(A \cap B) = P(A \mid B)\, P(B)$. Now note that $A \cap B$ and $B \cap A$
are the same event ("$A$ and $B$" = "$B$ and $A$"), so the definition also
applies with the roles of $A$ and $B$ swapped:
$P(A \cap B) = P(B \mid A)\, P(A)$. Two expressions for the same quantity -
set them equal:

$$P(A \mid B)\, P(B) = P(B \mid A)\, P(A)$$

Divide both sides by $P(B)$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

That's the whole derivation - two applications of the definition.

Each factor has a name. Read them with $A$ = a hypothesis and $B$ = the
observed evidence (concrete values from the disease example below):

- **Prior** $P(A)$: the probability of the hypothesis *before* seeing the
  evidence. Below: $P(\text{sick}) = 1\%$, the prevalence.
- **Likelihood** $P(B \mid A)$: the probability of seeing this evidence *if
  the hypothesis is true*. Below: $P(+ \mid \text{sick}) = 99\%$, a property
  of the test.
- **Posterior** $P(A \mid B)$: the probability of the hypothesis *after*
  seeing the evidence - the thing you actually want. Below:
  $P(\text{sick} \mid +) \approx 17\%$.
- The denominator $P(B)$: the overall probability of the evidence, from all
  causes. Below: $594/10{,}000$ people test positive.

Why this rule is worth having: $P(B \mid A)$ and $P(A \mid B)$ are different
numbers (here 99% and 17%), and only the first can be measured directly. A
lab estimates $P(+ \mid \text{sick})$ by running the test on people already
known to be sick and counting. Nobody can measure $P(\text{sick} \mid +)$
that way - assembling the group would require already knowing who is sick,
which is the very question. Bayes rule computes the unmeasurable direction
from the measurable one plus the prior.

The classic trap (run it with raw counts, as in `probability_demos.ipynb`).
The setup:

- **Prevalence** 1%: one person in 100 has the disease.
- **Sensitivity** $= P(+ \mid \text{sick}) = 99\%$: how often the test flags
  a sick person.
- **Specificity** $= P(- \mid \text{healthy}) = 95\%$: how often the test
  clears a healthy person. So 5% of healthy people get a false positive.

You test positive - what's the chance you're sick? Count through 10,000
people:

- 100 are sick; 99% of them test positive → **99 true positives**.
- 9,900 are healthy; 5% of them test positive → **495 false positives**.
- Everyone who tested positive: $99 + 495 = 594$. Sick among them:
  $99 / 594 \approx 16.7\%$.

Not 99%. The prior (the rarity of the disease) dominates because the false
positives come from a pool 99 times larger than the sick pool (9,900 vs
100). In ML terms: this is precision under class imbalance - exactly why
accuracy is a useless metric for rare-event detection and why a 99%-accurate
fraud model can be garbage.

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

Bayes rule chains: **today's posterior is tomorrow's prior.** Suppose the
positive patient takes the test again, independently, and it is positive
again. Apply the rule a second time, starting from the first result's
posterior: of the 99 sick positives, 99% test positive again, about 98
people; of the 495 healthy false positives, 5% do, about 25 people. So
$P(\text{sick} \mid \text{two positives}) = 98 / (98 + 25) \approx 80\%$.
The first test moved the probability from 1% to 17%, the second from 17% to
80%: each piece of evidence updates the current belief, and the names
"prior" and "posterior" are relative to where you stand in that chain.

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
594 who tested positive, then to the 123 who tested positive twice - and the
sick share of the survivors climbs from 1% to 17% to 80%. The middle number
wears both names: posterior of the first test, prior of the second.*

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

Start concrete: three classes (cat, dog, bird) and two training photos. A
classifier's output for an input is one probability per class (a softmax over
its scores):

- Photo 1 is a **cat**. The model outputs $[0.7, 0.2, 0.1]$ for (cat, dog, bird).
- Photo 2 is a **dog**. The model outputs $[0.5, 0.2, 0.3]$.

The likelihood of this dataset is the probability the model assigned to what
actually happened: $0.7$ to photo 1 being a cat, times $0.2$ to photo 2 being
a dog:

$$L = 0.7 \cdot 0.2 = 0.14, \qquad -\log L = -\log 0.7 - \log 0.2 = 0.36 + 1.61 = 1.97$$

Each example contributes $-\log(\text{probability on its true class})$, and
nothing else: the wrong-class entries matter only because probabilities sum
to 1, so mass placed on them is mass taken from the true class.

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

*The loss reads only the green bar of each example - the probability on the
true class - and charges $-\log(\text{that bar})$. A confident correct answer
(0.7) costs little; a doubtful one (0.2) costs a lot. Making the loss small
means making the green bars tall, which is making the observed labels
probable.*

The general statement: a classifier with softmax output assigns probabilities
$p_\theta(k \mid x_i)$ to each class $k$. The negative log-likelihood of the data is:

$$-\sum_i \log p_\theta(y_i \mid x_i) = -\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)$$

where $y_{ik}$ is the **one-hot** label: $y_{ik} = 1$ if example $i$'s true
class is $k$, and $0$ otherwise.

To see the equality, work out one example. Say example $i$'s true class is 2,
out of $K = 3$ classes, so $(y_{i1}, y_{i2}, y_{i3}) = (0, 1, 0)$. Its inner
sum is

$$\sum_k y_{ik} \log p_\theta(k \mid x_i) = 0 \cdot \log p_\theta(1 \mid x_i) + 1 \cdot \log p_\theta(2 \mid x_i) + 0 \cdot \log p_\theta(3 \mid x_i) = \log p_\theta(2 \mid x_i)$$

which is exactly $\log p_\theta(y_i \mid x_i)$, that example's term on the
left. The one-hot vector is a selector: the zeros delete the wrong-class
terms, the single 1 keeps the true-class term. So per example the double sum
collapses to the single term, and the two sides are the same sum written two
ways. The right-hand side is the definition of the **cross-entropy loss**. So: **minimizing cross-entropy = maximizing the
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
