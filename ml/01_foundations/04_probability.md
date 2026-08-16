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
choose the $\theta$ that makes the observed data most probable. "Likelihood"
is §3.2's word: the probability of the observed evidence given a hypothesis.
Here the hypothesis is a parameter value $\theta$, and the evidence is the
dataset. The notation $\arg\max_\theta$ means "the value of $\theta$ at
which the expression is largest":

$$\hat\theta = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i \mid x_i)
             = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y_i \mid x_i)$$

The two argmaxes are equal because $\log$ is increasing, so it preserves
which input is largest. The log form is the one used in practice: it turns
the product into a sum, which is numerically stable and differentiable term
by term.

#### The smallest instance: a coin

One parameter, closed-form answer. A coin has unknown heads probability $p$;
you observe $h$ heads in $n$ flips. The log-likelihood of that data is

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

#### Cross-entropy is MLE for a classifier

**Cross-entropy** is the loss classifiers actually train with. For a target
distribution $q$ and a predicted distribution $p$ over $K$ classes it is
defined as

$$H(q, p) = -\sum_k q_k \log p_k$$

In classification the target is the **one-hot** label: $q_k = 1$ for the
true class and $0$ for every other class. So per example the sum keeps a
single term, and the loss is
$-\log(\text{probability the model put on the true class})$.

Run it on a concrete dataset: three classes (cat, dog, bird), two training
photos, and a model that outputs one probability per class (a softmax over
its scores):

- Photo 1 is a **cat**. The model outputs $[0.7, 0.2, 0.1]$ for (cat, dog,
  bird). Loss term: $-\log 0.7 = 0.36$.
- Photo 2 is a **dog**. The model outputs $[0.5, 0.2, 0.3]$. Loss term:
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

*The loss reads only the green bar of each example - the probability on the
true class - and charges $-\log(\text{that bar})$. A confident correct answer
(0.7) costs little; a doubtful one (0.2) costs a lot. Making the loss small
means making the green bars tall, which is making the observed labels
probable.*

Why is this a principled loss rather than a convention? Ask a different
question about the same dataset: what probability did the model give to
*both observed labels together*? It gave $0.7$ to photo 1's label and $0.2$
to photo 2's label, so (examples independent) the probability of the pair is
the product - the likelihood from the principle above:

$$L = 0.7 \cdot 0.2 = 0.14$$

Take $-\log$, and the product becomes a sum:

$$-\log L = -\log 0.7 - \log 0.2 = 0.36 + 1.61 = 1.97$$

These are the cross-entropy's per-example charges, term for term. Nothing
new was computed, and that is the finding: the numbers cross-entropy was
already adding are exactly the $-\log$ of the likelihood's factors. **The
loss you minimize is the negative log of the probability the model assigns
to the observed labels.**

In general: write the one-hot label of example $i$ as $y_{ik}$ ($1$ if the
true class is $k$, else $0$). Per example:

$$-\sum_k y_{ik} \log p_\theta(k \mid x_i) = -\log p_\theta(y_i \mid x_i)$$

(photo 1: $-(1 \cdot \log 0.7 + 0 \cdot \log 0.2 + 0 \cdot \log 0.1) =
-\log 0.7$). Sum over examples:

$$\underbrace{-\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)}_{\text{cross-entropy loss}} \;=\; \underbrace{-\sum_i \log p_\theta(y_i \mid x_i)}_{\text{negative log-likelihood}}$$

So **minimizing cross-entropy = maximizing the likelihood of the observed
labels**. Cross-entropy isn't an arbitrary "good loss"; for one-hot labels
it *equals* the negative log-likelihood, i.e. it is MLE for a categorical
output. (The identity needs one-hot targets: with soft targets such as label
smoothing, $q = [0.9, 0.05, 0.05]$, the cross-entropy formula still applies
but is no longer the log-likelihood of any observed label.)

#### The same pattern gives the regression losses

In regression the target is a number: say, predict an apartment's price from
its size. The prediction $f_\theta(x)$ never matches the observed price $y$
exactly - a 60 m² apartment predicted at 300k might sell for 310k. The 10k
gap, $y - f_\theta(x)$, is everything the input doesn't capture (floor,
condition, the buyer's mood) plus measurement error. Measured after the
fact it is called the **residual** (what remains after the model has
explained what it can); as a modeling assumption before the fact, call it
the **noise** $\varepsilon$:

$$y = f_\theta(x) + \varepsilon$$

Why this matters here: MLE works with $p_\theta(y \mid x)$, the probability
the model gives to the observed target. But a regression model's output is a
single number, $f_\theta(x) = 300$k, and one number is not a distribution:
it does not say how probable a sale at 310k or at 295k would be, so there is
nothing to take the likelihood of. The noise assumption supplies the missing
distribution: assume $\varepsilon$ follows one (say, Gaussian centered at
0), and then $y$ given $x$ is distributed as that noise shifted to sit on
top of the prediction - probable near 300k, less probable further away. Now
$p_\theta(y \mid x)$ exists, and the recipe
from the classifier case runs unchanged: take $-\log p(y \mid x)$ per
example and minimize the sum. The choice of noise distribution decides which
loss comes out:

- Gaussian noise, density $\frac{1}{\sqrt{2\pi\sigma^2}}
  e^{-\varepsilon^2 / 2\sigma^2}$:
  $-\log p(y \mid x) = \frac{(y - f_\theta(x))^2}{2\sigma^2} + \text{const}$,
  a squared error. MLE under Gaussian noise is **MSE**.
- Laplace noise, density $\frac{1}{2b} e^{-|\varepsilon| / b}$:
  $-\log p(y \mid x) = \frac{|y - f_\theta(x)|}{b} + \text{const}$, an
  absolute error. MLE under Laplace noise is **L1 loss**.

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

*Left: two candidate noise distributions on matching scales - the Gaussian
bell, and the Laplace with its sharper peak and heavier tails. Right: their
$-\log$ densities, which are the per-example losses. The Gaussian's is the
parabola $\varepsilon^2/2$ (MSE); the Laplace's is the V shape
$|\varepsilon|$ (L1). The shape of the loss is the noise assumption, made
visible.*

So picking a regression loss *is* picking a noise model. When an interviewer
asks "why this loss?", the answer is: it is the negative log-likelihood
under this output/noise model.

## 3.4 Bernoulli and Gaussian: the two distributions ML leans on

The sections around this one keep reaching for the same two distributions:
the coin's flips and the classifier's labels (§3.3) are Bernoullis; the
noise on a regression target (§3.3) and the limit of averages (§3.6) are
Gaussians. That is not a coincidence. They answer the two commonest modeling
questions:

- What does **one yes/no event** look like? → Bernoulli.
- What does the **sum of many small independent effects** look like? →
  Gaussian.

This section defines each, derives its basic facts, and gives the reasons it
is the right model for its question.

#### Bernoulli: one yes/no event

**Bernoulli($p$)**: a random variable with exactly two values:
$P(X = 1) = p$ and $P(X = 0) = 1 - p$. Each flip of §3.3's coin is one. Its
mean and variance follow from §3.1's definitions:

- Expectation is each value times its probability, summed:
  $\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1 - p) = p$.
- For the variance, use §3.1's identity
  $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2$. A 0/1 variable satisfies
  $X^2 = X$ (because $0^2 = 0$ and $1^2 = 1$), so
  $\mathbb{E}[X^2] = \mathbb{E}[X] = p$, and
  $\mathrm{Var}(X) = p - p^2 = p(1 - p)$.

$p(1 - p)$ is largest at $p = 1/2$ (value $1/4$): the fair coin, where the
outcome is hardest to predict. It shrinks to 0 as $p \to 0$ or $p \to 1$,
where the outcome is nearly certain.

Two things built from it:

- **Count the 1s.** Add $n$ independent Bernoullis,
  $X_1 + \dots + X_n$: the sum is the *number of successes* in $n$ tries,
  and its distribution is called the **Binomial**. §3.1's rules give its
  mean and variance at once: $\mathbb{E} = np$ by linearity,
  $\mathrm{Var} = np(1 - p)$ by additivity over independent draws. (§3.3's
  coin data, $h$ heads in $n$ flips, was one Binomial draw.)
- **Where it appears in ML.** A binary classifier that outputs "spam with
  probability 0.9" is modeling the label as Bernoulli($0.9$); training it
  with binary cross-entropy is MLE under exactly that model - §3.3's
  pattern with two classes. Dropout keeps each unit with probability $p$
  and zeroes it otherwise: each keep/drop decision is one Bernoulli($p$)
  draw, and a layer's mask is a vector of independent ones.

#### Gaussian: the sum of many small effects

**Gaussian** $\mathcal{N}(\mu, \sigma^2)$, density
$\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ - the
bell curve, with its two parameters being exactly its mean $\mu$ and
variance $\sigma^2$. It is not chosen for convenience alone; there are three
reasons it is the default:

1. **Sums of many small effects are bell-shaped (the CLT, §3.6).** Picture
   §3.3's price gap as roughly 20 small independent contributions, summed:

   - floor: $+0.8$k
   - condition: $-0.3$k
   - buyer's mood: $+0.5$k
   - timing: $-0.2$k
   - ...and 16 more of the same size, adding up to that apartment's
     residual $y - f_\theta(x)$.

   Across many apartments, which totals come up often?

   - **Near zero: common.** The nudges only need to roughly cancel, and
     many combinations do.
   - **Around $+15$k: rare.** Nearly all 20 nudges would have to point the
     same way at once.

   So totals pile up near the middle and thin out toward the extremes: a
   bell. §3.6's CLT makes this a theorem - the sum of many small
   independent effects is approximately Gaussian, whatever distribution
   each effect follows. Measurement noise and regression residuals are
   usually such sums, which is why "assume Gaussian noise" so often matches
   reality.
2. **Maximum entropy**: among all distributions with a given mean and
   variance, the Gaussian adds no structure beyond those two constraints -
   so it is the default assumption when mean and variance are all you know.
3. **Its $-\log$ density is a squared distance.** Compute it:

   $$-\log\Big(\tfrac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(x-\mu)^2 / 2\sigma^2}\Big)
   = \frac{(x-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi\sigma^2)$$

   The second term does not depend on $x$; the first is a squared distance
   from the mean. This is the computation §3.3's regression subsection ran:
   assume Gaussian noise, take $-\log$, and the loss that falls out is least
   squares.

## 3.5 Bias and variance of estimators

The setup: there is a true number $\theta$ you want to know, and you cannot
read it off directly - you only get a random sample. An **estimator** is a
formula applied to that sample; its output $\hat\theta$ is your estimate of
$\theta$. The hat marks "computed from data, intended to approximate
$\theta$." §3.3 already produced estimators without using the name: the
coin's $\hat p = h/n$ is a formula applied to the observed flips, and every
MLE is a formula applied to the dataset. This section asks how good such a
formula is.

The concrete instance to keep in mind, continuing §3.1's die (true mean
$\theta = 3.5$): estimate the mean by rolling 3 times and averaging,
$\hat\theta = \bar x = \frac{1}{3}(x_1 + x_2 + x_3)$. Different samples give
different estimates:

- rolls $2, 6, 3 \;\rightarrow\; \hat\theta = 3.67$
- rolls $1, 2, 4 \;\rightarrow\; \hat\theta = 2.33$
- rolls $5, 3, 6 \;\rightarrow\; \hat\theta = 4.67$

The formula is fixed, but its output depends on which random sample was
drawn. So $\hat\theta$ is itself a random variable, and everything from §3.1
applies to it: it has an expectation $\mathbb{E}[\hat\theta]$ (the average
estimate over all possible samples) and a variance. That gives two separate
ways an estimator can be bad:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

- **Bias** compares the *average* estimate to the truth. If it is nonzero,
  the miss is systematic: even averaging the estimates from thousands of
  fresh samples would not land on $\theta$. The die estimator has zero bias:
  by §3.1's linearity, $\mathbb{E}[\bar x] = \frac{1}{3}(3.5 + 3.5 + 3.5) = 3.5 = \theta$.
- **Var** is §3.1's variance applied to the random variable $\hat\theta$:
  how much the estimate jumps around its own average from sample to sample.
  The three estimates above ranged from 2.33 to 4.67; that spread is what
  Var measures. Its exact value comes from applying §3.1's two scaling rules
  to $\bar x = \frac{1}{3}(X_1 + X_2 + X_3)$, one rule at a time:
  - Variances of independent draws add, and one roll has variance $2.92$
    (§3.1). So $\mathrm{Var}(X_1 + X_2 + X_3) = 2.92 + 2.92 + 2.92 = 8.76$.
  - A constant factor comes out squared:
    $\mathrm{Var}\big(\tfrac{1}{3}\,(X_1 + X_2 + X_3)\big) = \tfrac{1}{3^2} \cdot 8.76 = \tfrac{8.76}{9} \approx 0.97$.

The two combine exactly. The quantity to decompose is the expected squared
miss, $\mathbb{E}[(\hat\theta - \theta)^2]$. Inside it, add and subtract
$\mathbb{E}[\hat\theta]$ (net change zero) to split the miss into the two
pieces just defined:

$$\hat\theta - \theta =
  \underbrace{\big(\hat\theta - \mathbb{E}[\hat\theta]\big)}_{\text{jump around the average}}
  + \underbrace{\big(\mathbb{E}[\hat\theta] - \theta\big)}_{\text{Bias, a constant}}$$

To keep the next step readable, name the two pieces:
$a = \hat\theta - \mathbb{E}[\hat\theta]$ (random, changes with the sample)
and $b = \mathbb{E}[\hat\theta] - \theta$ (the Bias, one fixed number).
Square the sum:

$$(\hat\theta - \theta)^2 = (a + b)^2 = a^2 + 2ab + b^2$$

Take the expectation of each term. In the middle term, $2b$ is a constant
factor, and §3.1's linearity lets a constant factor move outside the
expectation: $\mathbb{E}[2ab] = 2b\,\mathbb{E}[a]$. So:

$$\mathbb{E}[(\hat\theta - \theta)^2] = \mathbb{E}[a^2] + 2b\,\mathbb{E}[a] + b^2$$

Now evaluate the three pieces:

- $\mathbb{E}[a^2] = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big] = \mathrm{Var}$, the definition above.
- $\mathbb{E}[a] = \mathbb{E}\big[\hat\theta - \mathbb{E}[\hat\theta]\big] = \mathbb{E}[\hat\theta] - \mathbb{E}[\hat\theta] = 0$,
  because $\mathbb{E}[\hat\theta]$ is a fixed number and the expectation of a
  fixed number is itself. The middle term is therefore $2b \cdot 0 = 0$.
- $b^2 = \mathrm{Bias}^2$, a fixed number, unchanged by the expectation.

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
is. The decomposition above says the expected squared miss is exactly those
two, $\mathrm{Bias}^2 + \mathrm{Var}$.*

#### Why the 1/n sample variance is biased (Bessel's correction)

**The task.** §3.1 defined the variance of one draw through the true mean:
$\sigma^2 = \mathbb{E}[(X - \mu)^2]$. To estimate it from a sample of $n$
draws, replace both unknowns with their sample versions: the expectation
becomes an average over the draws, and $\mu$ (unknown) becomes $\bar x$.
That double substitution produces the natural estimator

$$\hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar x)^2$$

**The issue.** This estimator is biased low. Seeing that needs no algebra,
just the die, whose true variance is $\sigma^2 = \frac{35}{12} \approx 2.92$
(§3.1). Enumerate all $6^3 = 216$ equally likely 3-roll samples, apply the
formula to each, and average the 216 results: the average is
$\frac{35}{18} \approx 1.94$, a third short of $2.92$. So
$\mathbb{E}[\hat\sigma^2] \neq \sigma^2$: by the definition at the top of
§3.5 this estimator has nonzero bias, a systematic miss that no amount of
resampling repairs. The rest of this subsection finds where the shortfall
comes from, and the fix falls out of the derivation.

**The diagnosis.** $\sigma^2$ is defined through deviations from $\mu$, but
the estimator measures deviations from $\bar x$. So first relate the two
kinds of deviation. For a single term, add and subtract $\bar x$ (the same
move as the decomposition above):

$$x_i - \mu = (x_i - \bar x) + (\bar x - \mu)$$

Square it and sum over $i$. The factor $(\bar x - \mu)$ is the same number
for every $i$, so in the cross term it comes out of the sum, and its square
appears $n$ times:

$$\sum_i (x_i - \mu)^2 = \sum_i (x_i - \bar x)^2
  + 2(\bar x - \mu)\sum_i (x_i - \bar x) + n(\bar x - \mu)^2$$

The remaining sum in the middle term is zero by the definition of $\bar x$:
$\sum_i (x_i - \bar x) = \sum_i x_i - n\bar x = n\bar x - n\bar x = 0$
(deviations from their own average cancel exactly). Drop it and move
$n(\bar x - \mu)^2$ to the other side:

$$\sum_i (x_i - \bar x)^2 = \sum_i (x_i - \mu)^2 - n(\bar x - \mu)^2$$

Check on the die rolls $2, 6, 3$ from above ($\mu = 3.5$, $\bar x = \frac{11}{3}$,
so $\bar x - \mu = \frac{1}{6}$): the left side is
$\left(\frac{5}{3}\right)^2 + \left(\frac{7}{3}\right)^2 + \left(\frac{2}{3}\right)^2 = \frac{78}{9} \approx 8.67$;
the right side is $8.75 - 3 \cdot \left(\frac{1}{6}\right)^2 = 8.75 - 0.08 \approx 8.67$.
Both sides agree. And the identity already shows the direction of the bias:
the subtracted term $n(\bar x - \mu)^2$ is never negative, so deviations
measured from $\bar x$ always sum to less than (or at most equal to)
deviations measured from $\mu$.

Take expectations of both sides, one term at a time:

- $\mathbb{E}\big[\sum_i (x_i - \mu)^2\big] = n\sigma^2$: each term is
  $\mathbb{E}[(x_i - \mu)^2] = \sigma^2$ (the definition of variance,
  §3.1), and there are $n$ of them.
- $\mathbb{E}[(\bar x - \mu)^2] = \mathrm{Var}(\bar x)$: since
  $\mathbb{E}[\bar x] = \mu$ (shown at the top of §3.5), this expression is
  the expected squared distance of $\bar x$ from its own mean, which is the
  definition of $\mathrm{Var}(\bar x)$. Its value comes from the same two
  steps as the 3-roll computation earlier in §3.5, now for general $n$,
  applied to $\bar x = \frac{1}{n}(X_1 + \dots + X_n)$:
  - Variances of independent draws add, and each draw has variance
    $\sigma^2$, so $\mathrm{Var}(X_1 + \dots + X_n) = n\sigma^2$. (Die,
    $n = 3$: $3 \cdot 2.92 = 8.76$.)
  - The constant factor $\frac{1}{n}$ comes out squared, so
    $\mathrm{Var}(\bar x) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$.
    (Die: $\frac{8.76}{9} \approx 0.97$.)

So:

$$\mathbb{E}\Big[\sum_i (x_i - \bar x)^2\Big] = n\sigma^2 - \sigma^2 = (n-1)\,\sigma^2$$

**The fix.** The sum of squared deviations averages $(n-1)\sigma^2$. The
last step is to divide by a constant and see what each choice of divisor
does to the expectation (the constant moves outside by linearity):

- **Divide by $n$** (the estimator under study):
  $\mathbb{E}\big[\frac{1}{n}\sum_i (x_i - \bar x)^2\big] = \frac{1}{n}(n-1)\sigma^2 = \frac{n-1}{n}\sigma^2$.
  This is less than $\sigma^2$, so
  $\mathrm{Bias} = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}$:
  a systematic underestimate. For the 3-roll die estimator this predicts an
  average of $\frac{2}{3} \cdot \frac{35}{12} = \frac{35}{18} \approx 1.94$,
  exactly the number the enumeration at the start of this subsection
  produced. The derivation explains the observation.
- **Divide by $n-1$ instead**:
  $\mathbb{E}\big[\frac{1}{n-1}\sum_i (x_i - \bar x)^2\big] = \frac{1}{n-1}(n-1)\sigma^2 = \sigma^2$,
  exactly. Zero bias. Choosing $n-1$ as the divisor is **Bessel's
  correction**.

Where did one whole $\sigma^2$ go? The identity says the sum measured from
$\bar x$ falls short of the sum measured from $\mu$ by $n(\bar x - \mu)^2$,
and in expectation that shortfall is
$n \cdot \mathrm{Var}(\bar x) = n \cdot \frac{\sigma^2}{n} = \sigma^2$:
exactly one draw's worth of variance. The cause: $\bar x$ was computed from
the same sample it is then subtracted from, so it sits closer to these
particular $x_i$ than $\mu$ does, and the measured deviations come out
systematically too small.

#### The tradeoff in ML terms

Everything above was about estimating one number from one sample. The reason
it sits in an ML syllabus: training a model fits the same template. The
dictionary:

- The **random sample** is the training set.
- The **estimator** (the formula applied to the sample) is the whole
  training procedure.
- The **estimate** $\hat\theta$ is the trained model's prediction at some
  input $x$.
- **Redrawing the sample** means collecting a fresh training set and
  retraining.

With that dictionary, §3.5's two definitions read directly:

- **Bias**: train on many independent training sets and average all the
  resulting predictions at $x$; bias is how far that average sits from the
  true relationship. A miss that survives this averaging is systematic: it
  comes from what the model class cannot express.
- **Variance**: how much the prediction at $x$ changes from one training
  set to the next. It comes from the fit following details specific to the
  one sample it saw, including that sample's noise, which a fresh sample
  would not repeat.

The two classic model types sit at opposite corners of the dartboard figure:

- A **linear model** is high-bias/low-variance. If the true relationship
  curves, a straight line misses the curve at the same places in the same
  direction no matter which training set it saw (bias), but refitting on a
  new sample gives nearly the same line (low variance).
- A **deep net or unpruned decision tree** is low-bias/high-variance. It
  can express nearly any shape, so the average over training sets tracks
  the true relationship (low bias), but each individual fit also follows
  its own training set's noise, so two fits on two samples disagree
  (variance).

The standard interventions, stated as which term each one moves:

- **Regularization** limits how closely the fit can follow the particular
  training set: variance drops, and some bias is accepted in exchange.
- **More data** lowers variance directly, the same mechanism as
  $\mathrm{Var}(\bar x) = \sigma^2 / n$ shrinking with $n$.
- **Bagging** trains several high-variance models on resampled data and
  averages their predictions: averaging cancels variance (the $\bar x$
  mechanism again) while leaving bias roughly unchanged.

For a model's test error, the decomposition gains one term:
$\text{Bias}^2 + \text{Variance} + \text{irreducible noise}$. The new term
is the noise $\varepsilon$ in the target itself (§3.3): even a perfect
model of the true relationship cannot predict the noise draw added to the
test label. The full derivation is in the evaluation module.

## 3.6 Central limit theorem - the $\sqrt{n}$ intuition

Why this section: almost every number used to train or evaluate a model is
an average over a sample. $\bar x$ estimating a mean (§3.5), a test-set
accuracy (an average of per-example 0/1 scores), a minibatch gradient (an
average of per-example gradients). This section answers the two questions that apply to all of
them:

- **How far off is the average?** An average computed from a random sample
  is itself a random variable (§3.5), so it has a standard deviation: the
  typical distance between the average you computed and the true mean. This
  section computes it, $\sigma / \sqrt{n}$, which is the error bar on every
  number computed from a sample.
- **What distribution does the average follow?** Approximately Gaussian, no
  matter which distribution the individual draws came from. That is the
  **central limit theorem (CLT)**, and it is the theorem §3.4 leaned on
  when it argued that sums of many small effects are bell-shaped.

Both answers need the setup pinned down first.

A **draw** is one observation of a random variable: one die roll, one coin
flip, one sampled data point. Take $n$ of them, $X_1, \dots, X_n$. They are
**i.i.d.** (independent and identically distributed) when two conditions
hold:

- **Identically distributed**: every draw comes from the same distribution,
  so each one has the same mean $\mu$ and the same variance $\sigma^2$.
  Rolling one die $n$ times qualifies; rolling a die for some draws and
  flipping a coin for others does not.
- **Independent**: no draw affects any other. Die rolls qualify. Dealing
  cards from a deck without replacement does not: each card dealt changes
  what the next draw can be.

Average the $n$ draws: $\bar X_n = \frac1n \sum X_i$. Two facts, both
familiar from §3.5 (each follows from §3.1's rules):

- $\mathbb{E}[\bar X_n] = \mu$ (linearity) - unbiased.
- $\mathrm{Var}(\bar X_n) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$
  (independence → variances add; the $\frac1n$ comes out squared).

Take the square root of that variance: the standard deviation of $\bar X_n$
is $\sigma/\sqrt{n}$. It falls toward zero as $n$ grows - the **law of
large numbers**, with a rate attached. The rate is the practical part: the
$\sqrt{n}$ in the denominator means halving your error bar needs 4× the
data. The CLT adds the
shape: not only does $\bar X_n$ concentrate, its distribution becomes **Gaussian**,
*regardless of the original distribution*:

$$\frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

This equation makes a strong claim: the individual draws $X_i$ can come
from *any* distribution (die rolls, exponentials, anything with a mean and
a variance), and the distribution of their average still turns Gaussian.
Watch it happen. Start from a
deliberately non-Gaussian distribution, the exponential (density $e^{-x}$,
mean $\mu = 1$): highest at zero, long tail to the right, no left tail at
all. The three panels show the distribution of the average $\bar X_n$ as
$n$ grows:

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

*The distribution of $\bar X_n$ for averages of $n$ exponential draws
(curves scaled to matching heights). Both of the section's answers are
visible: the curve narrows around $\mu = 1$ (that is $\sigma/\sqrt{n}$),
and the lopsided shape becomes a symmetric bell (that is the CLT).*

The mechanism behind the change of shape, run on the smallest instance: two
dice. What is the distribution of their sum $S$? For $S$ to equal a total
$t$, the second die must show some value $v$, and the first must then show
$t - v$. Each $v$ has probability $\frac16$, so add up the six cases:

$$P(S = t) = \sum_{v=1}^{6} \tfrac16 \cdot P(\text{first die} = t - v)$$

Read the right side one term at a time, each as a function of the total $t$:

- The $v = 1$ term is $\tfrac16\,P(\text{first die} = t - 1)$. The first
  die only ever shows $1$ through $6$, so this term is nonzero only when
  $t - 1$ is one of those values, i.e. for $t = 2$ through $7$. As a
  function of $t$, it is the first die's distribution slid one step to the
  right and scaled by $\tfrac16$.
- The $v = 2$ term is the same distribution slid two steps right (nonzero
  for $t = 3$ through $8$), the $v = 3$ term three steps, and so on up to
  the $v = 6$ term, which covers $t = 7$ through $12$. No single term spans
  all totals; together the six windows cover every possible total, $2$
  through $12$.

So the formula says: slide six copies of the first die's distribution, one
step apart, and add them up. This slide-and-add operation is called the
**convolution** of the two distributions, and a sum's distribution is
always the convolution of its summands' distributions.

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

*Top: the six slid copies of the first die's flat distribution; the copy
for $v$ is slid right by $v$ steps, so it covers the totals $v+1$ through
$v+6$. Bottom: adding the rows column by column gives the number of ways
each total can happen (out of 36). The total $7$ sits under all six copies;
the totals $2$ and $12$ sit under one each. The flat shape has become a
triangle.*

The diagram shows why adding slid copies smooths the shape: the middle of
the stack collects contributions from every copy, the ends from few. Each
new draw repeats the slide-and-add, and repeating it has three effects:

- **Features of the starting distribution fade.** Every slide-and-add
  averages the shape once more, so anything specific to the original
  distribution (the exponential's spike at zero, in the panels above) gets
  blurred a little further with each draw.
- **The mean and variance survive.** §3.1's rules fix both regardless of
  shape: means add, and variances of independent draws add. Smoothing
  changes the shape, not those two numbers.
- **The process has a resting point: the Gaussian.** Convolving two
  Gaussians gives another Gaussian, so once the shape reaches the bell,
  further draws have nothing left to change.

§3.4's cancellation argument (many combinations of nudges produce a
middling total, few produce an extreme one) is the triangle's counting
argument, applied to small continuous nudges instead of dice.
`probability_demos.ipynb` tracks the process numerically: the skewness of
$\bar X_n$ falls toward 0 and its standard deviation matches
$\sigma/\sqrt{n}$.

Three places this gets used daily in ML:

- **SGD noise.** A minibatch gradient is the average of the per-example
  gradients in the batch: a sample mean. Its standard deviation around the
  full-dataset gradient therefore shrinks like
  $1/\sqrt{\text{batch size}}$: quadrupling the batch halves the gradient
  noise.
- **Error bars on test accuracy.** Each test example scores 1 (correct) or
  0 (wrong): a Bernoulli draw (§3.4) with unknown true accuracy $p$.
  Reported accuracy is the mean of $n$ such draws, so its standard
  deviation is $\sqrt{p(1-p)} / \sqrt{n}$ (Bernoulli variance $p(1-p)$,
  §3.4). With $n = 1000$ test examples and accuracy near 90%:
  $\sqrt{0.9 \cdot 0.1 / 1000} \approx 0.009$, an error bar of about
  $\pm 0.9\%$. A claimed "0.5% improvement" measured on such a test set is
  smaller than the error bar: it could be sampling luck.
- **Confidence intervals and A/B tests** report ranges like "$\pm 2$
  standard errors". The 2 comes from the Gaussian shape the CLT guarantees:
  95% of a Gaussian's probability lies within 2 standard deviations of its
  mean.
