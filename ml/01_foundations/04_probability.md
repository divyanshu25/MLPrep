# Probability & statistics for ML

Three questions come up again and again when you train and evaluate models.
Why is the loss cross-entropy, or MSE, and not something else? When model B
beats model A by half a point, is that real or luck? And when a model is
wrong, is it too simple, or has it latched onto quirks of its training set?
All three have precise answers, and they come from a small set of ideas:
how to describe a random quantity, how to learn from data, and how much to
trust a number computed from data.

## 3.1 Random variables, expectation, variance

A **random variable** is just a number that comes out of a random process.
Roll a die and you get one of 1 through 6, each with probability 1/6. Before
the roll you can't say which, but you can say two useful things about it:
where it lands on average, and how far from that it typically lands.

For the die, the average is 3.5 and the typical distance from it is about
1.7. If you rolled it 100,000 times, the average of your rolls would come out
very close to 3.5, and their typical distance from 3.5 very close to 1.7.

#### Expectation

The average is called the **expectation**: multiply each value by its
probability and add up.

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

The most useful fact about expectations is that they split over sums. Roll
two dice and the total averages $3.5 + 3.5 = 7$. In general,

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

and this is called **linearity**. It holds even when $X$ and $Y$ depend on
each other. Write $P(x, y)$ as shorthand for $P(X = x, Y = y)$, the
probability that $X = x$ *and* $Y = y$, and $P(x)$ for $P(X = x)$. Then,
for $a = b = 1$,

$$\mathbb{E}[X + Y] = \sum_{x,y} (x + y)\,P(x, y) = \sum_x x \sum_y P(x, y) + \sum_y y \sum_x P(x, y) = \sum_x x\,P(x) + \sum_y y\,P(y)$$

The middle step only splits the sum in two, and the last step uses the fact
that adding $P(x, y)$ over every $y$ gives $P(x)$. Nothing in it needed $X$
and $Y$ to be unrelated. Many of the results that follow are one application
of linearity.

Products don't split so easily. $\mathbb{E}[XY]$ equals
$\mathbb{E}[X]\,\mathbb{E}[Y]$ when $X$ and $Y$ are **independent**,
meaning knowing one tells you nothing about the other:
$P(x, y) = P(x)\,P(y)$ for every pair of values. When that holds, the double
sum factors into two single sums:

$$\mathbb{E}[XY] = \sum_{x,y} x\,y\,P(x, y) = \sum_{x,y} x\,y\,P(x)\,P(y) = \Big(\sum_x x\,P(x)\Big)\Big(\sum_y y\,P(y)\Big) = \mathbb{E}[X]\,\mathbb{E}[Y]$$

#### Variance

The spread is measured by the **variance**: the average squared distance
from the true mean $\mu = \mathbb{E}[X]$.

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]$$

"Mean" and "average" are the same operation, but you can apply it to two
different things. The
*true mean* averages over every possible outcome, weighted by probability:
for the die it's 3.5, one fixed number. The *sample mean* averages the
draws you actually got: rolls 2, 6, 3 give 3.67, and the next three rolls
will give something else. The $\mathbb{E}[\,\cdot\,]$ notation always means
the first kind.

There's a shortcut for computing variance. Expand the square and use
linearity: $\mathbb{E}[X^2 - 2\mu X + \mu^2] = \mathbb{E}[X^2] - 2\mu \cdot \mu
+ \mu^2$, so $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2$. For the die,
$\mathbb{E}[X^2] = (1+4+9+16+25+36)/6 = 91/6$, and the variance is
$91/6 - 3.5^2 \approx 2.92$.

Because it averages *squared* distances, the variance is in squared units.
Its square root is the **standard deviation**, $\sigma$, which is back in the
units of the rolls and is the "typical distance from the mean" from the
start of this section: for the die, $\sigma = \sqrt{2.92} \approx 1.71$.

Two rules tell you how variance behaves when you transform things. If you
multiply $X$ by $a$, the mean becomes $a\mu$ (by linearity), so every
distance from the mean gets multiplied by $a$ and every squared distance by
$a^2$: $\mathrm{Var}(aX) = a^2\,\mathrm{Var}(X)$.
And if you add two *independent* variables, their variances add.

To see why, write $\mu_X$ and $\mu_Y$ for the two means. By linearity,
$X + Y$ has mean $\mu_X + \mu_Y$, so its distance from its mean is
$(X - \mu_X) + (Y - \mu_Y)$. The variance is the average of that distance
squared. Squaring a sum of two pieces, $(a + b)^2 = a^2 + b^2 + 2ab$, and
averaging term by term gives

$$\mathrm{Var}(X + Y) = \underbrace{\mathbb{E}[(X - \mu_X)^2]}_{\mathrm{Var}(X)} + \underbrace{\mathbb{E}[(Y - \mu_Y)^2]}_{\mathrm{Var}(Y)} + 2\,\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

The last average, without the 2, is the **covariance** of $X$ and $Y$. It is
positive when $X$ tends to be above its mean at the same times $Y$ is above
its mean, and negative when one tends to be above while the other is below.
If $X$ and $Y$ are independent, so are $X - \mu_X$ and $Y - \mu_Y$, and the
product rule for independent variables, $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$, splits the covariance into
$\mathbb{E}[X - \mu_X] \cdot \mathbb{E}[Y - \mu_Y]$. Each of those averages
is zero by linearity ($\mathbb{E}[X] - \mu_X = 0$), so the covariance is
$0 \cdot 0 = 0$. That
leaves

$$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$

Two dice have variance $2.92 + 2.92 = 5.83$.

The derivation only used independence to make the covariance zero, so
$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ holds whenever the
covariance is zero. Variables with zero covariance are called
**uncorrelated**. Independent variables are always uncorrelated, but
uncorrelated variables are not always independent.

The rule that independent variances add,
$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$, is the reasoning
behind weight initialization schemes like Xavier and He, which choose the weight variance
so that each layer's outputs have the same variance as its inputs.

## 3.2 Bernoulli and Gaussian: the two distributions ML leans on

Two distributions show up far more than any others. The Bernoulli describes
a single yes/no event: a coin flip, whether an email is spam, whether a
neuron is kept by dropout. The Gaussian describes a quantity built from many
small effects: measurement noise, how far a prediction misses, an average over
many examples.

#### Bernoulli: one yes/no event

A **Bernoulli($p$)** variable is 1 with probability $p$ and 0 otherwise.
Flip a coin with $p = 0.3$ many times and about
30% of the flips come out 1.

The mean is $p$, since $1 \cdot p + 0 \cdot (1 - p) = p$. The variance is
$p(1 - p)$, here $0.3 \times 0.7 = 0.21$. (A 0/1 variable equals its own
square, so $\mathbb{E}[X^2] = p$, and the shortcut $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2$ from §3.1 gives
$p - p^2$.)

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

A fair coin is as unpredictable as a coin can be, so
its variance is largest. A coin that almost always lands heads barely varies
at all.

The Bernoulli is the building block for a few things you'll meet constantly.
Add up $n$ independent flips and you're counting successes, which gives the
**Binomial** distribution, with mean $np$ by linearity and variance
$np(1 - p)$ because independent variances add ($n$ copies of $p(1-p)$). A classifier that says "spam
with probability 0.9" is modeling the label as Bernoulli(0.9). And **dropout**, a training method
that sets each neuron's output to zero at random, keeps each neuron with
probability $p$, so a layer of $n$ neurons keeps about $np$ of them. (PyTorch's
dropout layer takes the drop probability instead, so a setting of 0.1 keeps
each neuron with $p = 0.9$.)

#### Gaussian: the sum of many small effects

The **Gaussian** $\mathcal{N}(\mu, \sigma^2)$ is the bell curve, centered at
the mean $\mu$, with its spread set by the standard deviation $\sigma$ (the
second number in the notation is the variance, $\sigma^2$).

Think about how far a house-price prediction misses the actual price. The
miss is the sum of lots of small things the model didn't account for: the
floor, the condition, the buyer's mood, the week it sold. Simulate that as 20
small independent pushes, each anywhere from −1k to +1k, for 100,000 houses,
and count how many total misses land near each value (within 1k of it):

| total miss | −8k | −6k | −4k | −2k | 0 | +2k | +4k | +6k | +8k |
|---|---|---|---|---|---|---|---|---|---|
| houses | 309 | 2,324 | 9,546 | 22,638 | 30,192 | 22,586 | 9,784 | 2,324 | 267 |

Each push on its own is flat: any value from −1k to 1k is equally likely. But
the totals pile up in the middle and thin out toward the edges. That's
because a total near zero only needs the pushes to roughly cancel, and there
are a huge number of ways for that to happen, while a total of +9k needs
nearly all 20 pushes to go the same way. The result is a bell.

The bell's formula is

$$\frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(x-\mu)^2 / 2\sigma^2}$$

A price can be any number, not one of a fixed list, so the chance of
hitting one exact value is zero. The formula gives a **density** instead: the
probability of landing in a narrow interval of width $w$ around $x$ is about
the curve's height at $x$ times $w$, and sums over values become areas under
the curve (integrals).

The part that gives the curve its shape is $e^{-(x-\mu)^2 / 2\sigma^2}$. At the mean
it equals 1, it's symmetric on either side, and it falls off quickly as you
move away. With $\sigma = 10$, a value 10 away from the mean gets a factor of
$e^{-0.5} = 0.61$, and a value 30 away gets $e^{-4.5} = 0.011$. The constant
in front scales the curve so its total area is 1: the area under
$e^{-(x-\mu)^2 / 2\sigma^2}$ is exactly $\sqrt{2\pi\sigma^2}$ (a standard
calculus result called the Gaussian integral), so dividing by it leaves an
area of 1.

So why is the Gaussian everyone's default? Partly because of the simulation
above: real noise is usually the sum of many small effects, so a bell often
fits. And partly because its $-\log$ is a squared distance. The log of a product is the sum of the logs,
and $-\log e^{-z} = z$, so

$$-\log\Big(\tfrac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(x-\mu)^2 / 2\sigma^2}\Big)
= \frac{(x-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi\sigma^2)$$

That's the first hint of why least squares is so common.

## 3.3 Bayes rule, from counting

Bayes rule is about turning a probability around. A lab can measure how
often a test comes back positive for people known to be sick. But a patient
with a positive result wants the reverse: given that I tested positive, how
likely am I to be sick? Those two numbers can be wildly different, and mixing
them up is one of the most common mistakes in reasoning about data.

#### The rare-disease test

Say a disease affects 1 person in 100, and there's a test for it. If you're
sick, the test comes back positive 99% of the time. If you're healthy, it
still comes back positive 5% of the time, a false alarm. Notice these are two
separate numbers, one for each group: a test can be excellent at catching the
sick and only decent at clearing the healthy.

Now you test positive. What's the chance you're actually sick? Most people
guess something close to 99%. Count through 10,000 people instead:

- 100 are sick, and 99% of them test positive: 99 people.
- 9,900 are healthy, and 5% of them test positive: 495 people.
- So 594 people test positive, and only 99 of them are sick: $99/594 \approx 17\%$.

About 17%. The test catches nearly every sick person, but there are so many
more healthy people (9,900 against 100) that even a 5% false-alarm rate
produces five false alarms for every real case.

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

*Everyone who tests positive, drawn to scale: 99 sick, 495 healthy.*

The three numbers we started with have standard names, and it helps to
notice that one of them describes the people while the other two describe
the test:

| Name | Value | Describes |
|---|---|---|
| **Prevalence** | 1% | the people: how common the disease is among them |
| **Sensitivity** | 99% | the test: the share of sick people it flags |
| **Specificity** | 95% | the test: the share of healthy people it clears (so 5% false alarms) |

If you've worked on fraud or anomaly detection, you've seen this exact
problem under a different name: precision under class imbalance. (Precision
is the fraction of flagged cases that are real, which here is exactly the
fraction of positives who are sick.) It's why
accuracy is a useless metric for rare events, and why a 99%-accurate fraud
model can still flag mostly innocent transactions. (`probability_demos.ipynb`
runs the counts.)

#### The rule

The counting above already *is* Bayes rule; the formula just adds notation.
$P(\text{sick} \mid +)$ is read "the probability of sick given positive", and
it means: among the people who test positive, what fraction are sick. That's
exactly what we computed, sick-and-positive divided by all-positive.

Now write those two counts using the numbers we started from. The
sick-and-positive share is 99% of the 1%:
$P(+ \mid \text{sick})\,P(\text{sick}) = 0.99 \times 0.01 = 0.0099$. The
all-positive share counts both ways to test positive:

$$P(+) = P(+ \mid \text{sick})\,P(\text{sick}) + P(+ \mid \text{healthy})\,P(\text{healthy}) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0594$$

Divide one by the other:

$$P(\text{sick} \mid +) = \frac{P(+ \mid \text{sick})\; P(\text{sick})}{P(+)} = \frac{0.0099}{0.0594} \approx 16.7\%$$

Replace "sick" with any hypothesis $A$ and "positive" with any evidence $B$,
and you have the general rule:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

The pieces have names you'll hear a lot. $P(A)$ is the **prior**, what you
believed before the evidence; here it's the prevalence, 1%. $P(B \mid A)$ is
the **likelihood**, how often the evidence shows up when the hypothesis is
true; here it's the sensitivity, 99%. $P(A \mid B)$ is the **posterior**,
what you believe after the evidence, the number you actually wanted: 17%.
And $P(B)$ is the **evidence**, how often you'd see a positive at all: 5.94%.

Why bother with a formula when counting works? Because the test's numbers
stay fixed wherever you use it, but the answer depends on who's being tested.
Use the same test on people who already have symptoms, where maybe 10% are
sick:

$$P(\text{sick} \mid +) = \frac{0.99 \times 0.10}{0.99 \times 0.10 + 0.05 \times 0.90} \approx 69\%$$

Same test, same positive result. In the general population it means 17%;
among people with symptoms it means 69%. Bayes rule is how you carry the
test's fixed properties over to whatever group you're actually in.

#### Today's posterior is tomorrow's prior

Suppose our patient takes the test a second time and it's positive again.
After the first test we no longer believe 1%; we believe 17%, and that becomes
the new prior. Write $+_1$ and $+_2$ for the two positive results, and
assume the second test's errors are independent of the first's, given
whether the patient is sick. Then the second test still catches a sick
person 99% of the time and alarms on a healthy one 5% of the time, whatever
the first test said. (That assumption fails for a retest on the same blood
sample, which tends to repeat the same mistake.) Bayes rule is the same as
before, with every probability now also conditioned on $+_1$:

$$P(\text{sick} \mid +_1, +_2) = \frac{P(+_2 \mid \text{sick})\, P(\text{sick} \mid +_1)}{P(+_2 \mid \text{sick})\, P(\text{sick} \mid +_1) + P(+_2 \mid \text{healthy})\, P(\text{healthy} \mid +_1)}$$

The prior is now $P(\text{sick} \mid +_1) = 1/6$ (that's $99/594$), so
$P(\text{healthy} \mid +_1) = 5/6$, and the likelihoods are 0.99 and 0.05:

$$P(\text{sick} \mid +_1, +_2) = \frac{0.99 \times \tfrac{1}{6}}{0.99 \times \tfrac{1}{6} + 0.05 \times \tfrac{5}{6}} = \frac{0.165}{0.165 + 0.042} \approx 80\%$$

In counts: of the 99 sick people who tested positive, 99% (about 98) test
positive again; of the 495 healthy ones, 5% (about 25) do. That's 98 sick
out of 123, about 80%.

The first positive moved us from 1% to 17%. The second, starting from 17%,
moved us to 80%. Each piece of evidence updates whatever you currently
believe, and "prior" and "posterior" only describe where you are in that
chain.

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

*$P(\text{sick})$ at each stage, the green part of each bar. Each update
keeps only the people consistent with the new evidence: 10,000 shrink to the
594 who tested positive, then to the 123 who tested positive twice.*

## 3.4 Maximum likelihood: where the standard losses come from

One idea ties the common losses together. A model, whatever its
architecture, is making a claim about how probable each possible outcome is.
Training it means adjusting it until the things that actually happened look
as probable as possible. And every standard loss turns out to be the same
quantity in disguise: $-\log$ of the probability the model gave to what
actually happened. For a classifier, that's cross-entropy. For a regression
model, if you think of its output as the center of a bell of likely values,
it works out to squared error.

#### The principle, on a coin

The idea is called **maximum likelihood estimation** (MLE): choose the
**parameters** (the adjustable numbers in the model, here a coin's heads
probability) under which the data you saw is most probable.

Take a coin that landed heads 7 times in 10 flips. Each flip is a
Bernoulli($p$) draw, and the flips are independent, so their probabilities
multiply: the probability of that exact sequence is $p^7 (1-p)^3$. We don't know $p$, so try a few values:

| heads probability $p$ | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|
| probability of the 10 flips | 0.00098 | 0.0018 | 0.0022 | 0.0017 |

$p = 0.7$ makes the data more than twice as probable as $p = 0.5$, and no
value beats it. This holds in general: for $h$ heads in $n$ flips, the best
$p$ is always $h/n$, the frequency you observed. (Taking the log turns
$p^h(1-p)^{n-h}$ into $h \log p + (n-h)\log(1-p)$, which peaks at the same
$p$ because log only increases. Its derivative is $h/p - (n-h)/(1-p)$.
Setting that to zero gives $h(1-p) = (n-h)\,p$, so $h = np$ and $p = h/n$.)

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

*The log-likelihood of every candidate $p$ after 7 heads in 10 flips. It has
one peak, at $p = 0.7$.*

The probability of the data, viewed as a function of the parameters, is the
**likelihood**. It's the same word as in Bayes rule, and the same idea: the
probability of the evidence under a hypothesis, where now the hypothesis is a
choice of parameters, written $\theta$ (for a neural network, all of its
weights). Take a model that predicts a label $y_i$ from an input $x_i$, say a
photo and its class. With independent examples, the likelihood is a product
of one probability per example, $p_\theta(y_i \mid x_i)$. In practice we maximize its log instead,
because a product of thousands of probabilities rounds to zero on a computer.
The log of a product is the sum of the logs, and since log only increases,
the best $\theta$ doesn't change:

$$\hat\theta = \arg\max_\theta \prod_i p_\theta(y_i \mid x_i) = \arg\max_\theta \sum_i \log p_\theta(y_i \mid x_i)$$

($\arg\max_\theta$ just means "the $\theta$ that makes this largest".)

#### Cross-entropy is maximum likelihood for a classifier

Classifiers are trained with a loss called **cross-entropy**: for each
example it is $-\log$ of the probability the model put on the true class.
Take two photos and three classes
(cat, dog, bird):

| | true class | model output | probability on the truth | loss |
|---|---|---|---|---|
| photo 1 | cat | $[0.7, 0.2, 0.1]$ | 0.7 | $-\log 0.7 = 0.36$ |
| photo 2 | dog | $[0.5, 0.2, 0.3]$ | 0.2 | $-\log 0.2 = 1.61$ |

The total loss is 1.97. Now ask a different question: how probable did the
model think the true labels were? It gave them 0.7 and 0.2, so the
likelihood of the data is $0.7 \times 0.2 = 0.14$, and
$-\log 0.14 = -\log 0.7 - \log 0.2 = 0.36 + 1.61 = 1.97$. That's the same
number the loss gave, because the log of a product is the sum of the logs. So minimizing cross-entropy and maximizing the
likelihood are literally the same thing.

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

*The loss reads only the green bar of each example: the probability on the
true class.*

The $-\log$ charge is tiny when the model is confident and right, and grows
without limit as the probability on the true class heads toward zero:

| probability the model put on the true class | 0.9 | 0.5 | 0.01 |
|---|---|---|---|
| loss $= -\log(\text{that probability})$ | 0.11 | 0.69 | 4.61 |

So the total is dominated by the examples with a small probability on the
true class. The derivative of $-\log p$ with respect to $p$ is $-1/p$, so
those are also the examples with the largest gradients.

More generally, cross-entropy compares a target distribution $q$ with the
model's distribution $p$: $H(q, p) = -\sum_k q_k \log p_k$. With a one-hot
target, where $q_k$ is 1 for the true class and 0 elsewhere, only the true
class's term survives and you get exactly the negative log-likelihood above.
With soft targets, such as **label smoothing** (replacing the one-hot 1 with
0.9 and spreading the other 0.1 over the remaining classes),
$q = [0.9, 0.05, 0.05]$, it's still a
perfectly good loss, just no longer a likelihood.

With only two classes the same idea gives **binary cross-entropy**. If the
model outputs $\hat y = 0.9$ for "spam", it's saying the label is
Bernoulli(0.9): probability $\hat y$ for $y = 1$ and $1 - \hat y$ for
$y = 0$. Taking $-\log$ and writing both cases in one line (when $y = 1$ the
second term vanishes, and when $y = 0$ the first does):

$$-\log p_\theta(y \mid x) = -\big[\, y \log \hat y + (1 - y) \log(1 - \hat y) \,\big]$$

#### The regression losses: MSE and L1

MSE is what maximum likelihood gives when you assume a model's errors follow
a bell curve: small errors are common, large ones rare. L1 is what it gives
when large errors are more common than a bell allows.

A regression model outputs one number, say 300k for a house, but maximum
likelihood needs a probability for the price the house actually sold for.
So read the prediction as "about 300k": the actual price is the prediction
plus noise from the bell of §3.2, centered at 0 with standard deviation
$\sigma$. With $\sigma = 10$k, a sale that misses by $d$ thousand (the
**error**, or **residual**) scores the bell's height there, $e^{-d^2/200}$:
1 for a perfect prediction, 0.61 for a 10k miss, 0.011 for a 30k miss. (The
bell's constant in front is the same for every model, so it's dropped.)

Three houses sold for 310k, 290k and 305k. The houses are independent, so a
model's likelihood is the product of its scores:

| | predictions | errors | scores | likelihood | sum of squared errors |
|---|---|---|---|---|---|
| model A | 300, 300, 300 | 10, −10, 5 | 0.61, 0.61, 0.88 | 0.32 | 225 |
| model B | 310, 290, 300 | 0, 0, 5 | 1, 1, 0.88 | 0.88 | 25 |

B wins, and it also has the smaller sum of squared errors. That always
happens: multiplying powers of $e$ adds the exponents, so

$$-\log(\text{likelihood}) = \frac{\text{sum of squared errors}}{200}$$

(Check: $e^{-225/200} = 0.32$ and $e^{-25/200} = 0.88$.) The most likely
model is the one with the smallest squared error, and dividing by the number
of houses gives MSE.

The catch is that one $\sigma = 10$k assumes every house misses by about the
same number of dollars, which fails when a 10k cabin and a 200M estate are in
the same dataset. The usual fix is to predict log(price). The error then
becomes $\log(\text{actual}/\text{predicted})$, so one $\sigma$ means the
same *percentage* miss on every house. (Another fix is to have the model
output its own $\sigma$ for each house.)

L1 comes from **Laplace** noise, whose score for an error $d$ is
$e^{-|d|/b}$, with $b$ setting the spread. The same steps give
$-\log(\text{likelihood}) = (\text{sum of absolute errors})/b$.

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
  <polyline points="20.0,188.8 23.7,188.4 27.3,187.8 31.0,187.2 34.7,186.3 38.3,185.2 42.0,183.9 45.7,182.3 49.3,180.4 53.0,178.0 56.7,175.3 60.3,172.2 64.0,168.5 67.7,164.4 71.3,159.8 75.0,154.8 78.7,149.3 82.3,143.4 86.0,137.2 89.7,130.7 93.3,124.2 97.0,117.6 100.7,111.2 104.3,105.1 108.0,99.4 111.7,94.2 115.3,89.8 119.0,86.3 122.7,83.6 126.3,82.0 130.0,81.5 133.7,82.0 137.3,83.6 141.0,86.3 144.7,89.8 148.3,94.2 152.0,99.4 155.7,105.1 159.3,111.2 163.0,117.6 166.7,124.2 170.3,130.7 174.0,137.2 177.7,143.4 181.3,149.3 185.0,154.8 188.7,159.8 192.3,164.4 196.0,168.5 199.7,172.2 203.3,175.3 207.0,178.0 210.7,180.4 214.3,182.3 218.0,183.9 221.7,185.2 225.3,186.3 229.0,187.2 232.7,187.8 236.3,188.4 240.0,188.8" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <polyline points="20.0,183.2 23.7,182.5 27.3,181.7 31.0,180.9 34.7,179.9 38.3,178.8 42.0,177.7 45.7,176.4 49.3,174.9 53.0,173.3 56.7,171.6 60.3,169.7 64.0,167.5 67.7,165.2 71.3,162.5 75.0,159.7 78.7,156.5 82.3,152.9 86.0,149.0 89.7,144.7 93.3,140.0 97.0,134.7 100.7,128.9 104.3,122.5 108.0,115.4 111.7,107.5 115.3,98.8 119.0,89.2 122.7,78.7 126.3,66.9 130.0,54.0 133.7,66.9 137.3,78.7 141.0,89.2 144.7,98.8 148.3,107.5 152.0,115.4 155.7,122.5 159.3,128.9 163.0,134.7 166.7,140.0 170.3,144.7 174.0,149.0 177.7,152.9 181.3,156.5 185.0,159.7 188.7,162.5 192.3,165.2 196.0,167.5 199.7,169.7 203.3,171.6 207.0,173.3 210.7,174.9 214.3,176.4 218.0,177.7 221.7,178.8 225.3,179.9 229.0,180.9 232.7,181.7 236.3,182.5 240.0,183.2" fill="none" stroke="#10b981" stroke-width="2.5"/>
  <line x1="24" y1="44" x2="44" y2="44" stroke="#6366f1" stroke-width="2.5"/>
  <text x="48" y="48" fill="#6366f1" font-size="10.5">Gaussian</text>
  <line x1="24" y1="62" x2="44" y2="62" stroke="#10b981" stroke-width="2.5"/>
  <text x="48" y="66" fill="#10b981" font-size="10.5">Laplace</text>
  <polyline points="280.0,43.3 283.7,52.9 287.3,62.2 291.0,71.2 294.7,79.8 298.3,88.1 302.0,96.1 305.7,103.8 309.3,111.1 313.0,118.1 316.7,124.8 320.3,131.2 324.0,137.2 327.7,142.9 331.3,148.3 335.0,153.3 338.7,158.1 342.3,162.5 346.0,166.5 349.7,170.3 353.3,173.7 357.0,176.8 360.7,179.6 364.3,182.0 368.0,184.1 371.7,185.9 375.3,187.4 379.0,188.5 382.7,189.3 386.3,189.8 390.0,190.0 393.7,189.8 397.3,189.3 401.0,188.5 404.7,187.4 408.3,185.9 412.0,184.1 415.7,182.0 419.3,179.6 423.0,176.8 426.7,173.7 430.3,170.3 434.0,166.5 437.7,162.5 441.3,158.1 445.0,153.3 448.7,148.3 452.3,142.9 456.0,137.2 459.7,131.2 463.3,124.8 467.0,118.1 470.7,111.1 474.3,103.8 478.0,96.1 481.7,88.1 485.3,79.8 489.0,71.2 492.7,62.2 496.3,52.9 500.0,43.3" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <polyline points="280,92.2 390,190 500,92.2" fill="none" stroke="#10b981" stroke-width="2.5"/>
  <text x="350" y="54" text-anchor="middle" fill="#6366f1" font-size="10.5">&#949;&#178;/2: MSE</text>
  <text x="437" y="54" text-anchor="middle" fill="#10b981" font-size="10.5">|&#949;|: L1</text>
</svg>
</div>

*Left: the two noise shapes. The Laplace has a sharper peak and fatter tails
than the Gaussian. Right: their $-\log$, the loss for one example: a
parabola for MSE, a V for L1.*

The two differ on outliers. Four houses sold for 300k, 305k, 310k and 900k,
the last one a typo, and you predict one price $c$ for all four. MSE picks
the mean, 453.75k, above every real sale, because squaring makes the typo's
error dominate. L1 picks the median, any $c$ from 305k to 310k:

| prediction $c$ | 300k | 305k | 307.5k | 310k | 320k | 453.75k |
|---|---|---|---|---|---|---|
| sum of absolute errors | 615 | 605 | 605 | 605 | 625 | 892.5 |

Moving $c$ up by 1k adds 1 for each house below $c$ and subtracts 1 for each
house above it. So the loss falls while more houses are above $c$, and stops
falling once half are on each side. The typo counts as one house above $c$,
however far above it is.

## 3.5 Entropy and KL: what a loss value means

We now know every loss is $-\log$ of a probability. But what does a loss of
1.97 actually mean? And why does a label-smoothed loss never reach zero, or
an RLHF objective (reinforcement learning from human feedback, the method used
to fine-tune chat models) include a "KL penalty"? These all come from a few
definitions in information theory.

#### Surprise and entropy

Start with a single outcome that had probability $p$. Call $-\log p$ its
**surprise**. A certain outcome has surprise 0; a rare one has a lot; and
because $-\log(pq) = -\log p - \log q$, the surprises of independent events
add. Those cross-entropy charges from §3.4 (0.11 for a probability of 0.9,
4.61 for 0.01) were exactly this. With natural logs the unit is called a
**nat**, with base 2 a **bit** (one nat is about 1.44 bits); PyTorch losses
are in nats.

The **entropy** of a distribution is its average surprise, which is a
measure of how unpredictable a single draw is:

$$H(p) = -\sum_x p(x)\log p(x)$$

Every face of a fair die has probability $1/6$ and so surprise $\log 6$,
which makes the average, the entropy, $\log 6 \approx 1.79$ nats. A fair coin
has $\log 2 \approx 0.693$ nats. Measured in bits, with base-2 logs, each
side has surprise $\log_2 2 = 1$, so the entropy is exactly 1 bit. A coin that lands heads 90%
of the time has $0.9 \times (-\log 0.9) + 0.1 \times (-\log 0.1) = 0.095 +
0.230 = 0.325$ nats: it's more predictable, so there's less surprise on
average.

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

The curve has a similar shape to the Bernoulli variance from §3.2 (zero at
both ends, highest at $1/2$), and for the same reason: both are measuring how hard one flip is to predict.

#### Cross-entropy and KL divergence

Now suppose there are two distributions over the same outcomes: the truth
$q$, and a model's guess $p$. Say the truth is a fair coin,
$q = [0.5, 0.5]$, but the model believes it lands heads 90% of the time,
$p = [0.9, 0.1]$.

If outcomes come from $q$ but you compute each one's surprise as
$-\log p(x)$, your average surprise is the **cross-entropy**,
$H(q, p) = -\sum_x q(x)\log p(x)$. That's the loss from §3.4, with $q$ the
label and $p$ the model's prediction. Part of it is unavoidable: computing
surprise with $q$ itself costs $H(q)$. The excess over $H(q)$ is the **KL
divergence**,
$\mathrm{KL}(q\|p) = \sum_x q(x)\log\frac{q(x)}{p(x)}$.

For the coin, the cross-entropy is
$0.5 \times (-\log 0.9) + 0.5 \times (-\log 0.1) = 0.053 + 1.151 = 1.204$.
The truth's own entropy is $H(q) = \log 2 = 0.693$, and the KL is
$0.5 \log\frac{0.5}{0.9} + 0.5 \log\frac{0.5}{0.1} = -0.294 + 0.805 = 0.511$.
And $1.204$ is exactly $0.693 + 0.511$. This holds for any $q$ and $p$: write
each surprise as $-\log p(x) = -\log q(x) + \log\frac{q(x)}{p(x)}$, multiply
by $q(x)$ and add up over $x$. The first part sums to $H(q)$ and the second
to the KL:

$$H(q, p) = \underbrace{-\sum_x q(x)\log q(x)}_{H(q)} + \underbrace{\sum_x q(x)\log\frac{q(x)}{p(x)}}_{\mathrm{KL}(q\|p)}$$

which reads: the loss is the labels' own unpredictability, plus the extra
cost of the model being wrong.

#### KL is never negative

Using $p$ in place of $q$ can only raise the average surprise:
$\mathrm{KL}(q\|p) \ge 0$, with equality only when $p = q$. (This is called
Gibbs' inequality.) The proof uses the fact that $\log x \le x - 1$ for every
$x > 0$ (the line $x - 1$ is the tangent to $\log x$ at $x = 1$, and the log
curve lies below its tangents), applied with $x = p/q$:

$$-\mathrm{KL}(q \,\|\, p) = \sum_x q(x)\log\frac{p(x)}{q(x)}
  \;\le\; \sum_x q(x)\Big(\frac{p(x)}{q(x)} - 1\Big)
  \le 1 - 1 = 0$$

(The last step is $\le$ rather than $=$ because the sum of $p(x)$ runs only
over outcomes where $q(x) > 0$, so it is at most 1.)

Two consequences follow. The loss has a floor: cross-entropy can never drop below the entropy of the
labels. With one-hot labels that floor is 0, but with label smoothing,
$q = [0.9, 0.05, 0.05]$, it's
$0.9 \times (-\log 0.9) + 2 \times 0.05 \times (-\log 0.05) = 0.095 + 0.300
\approx 0.394$ nats, which is why a smoothed
training loss levels off above zero even when the model gets every example
right. And the KL from any $p$ to the uniform distribution, which gives
each of $K$ outcomes probability $1/K$, is
$\sum_x p(x)\log\frac{p(x)}{1/K} = \sum_x p(x)\log p(x) + \log K = \log K -
H(p)$. That is $\ge 0$, so no distribution over $K$ outcomes can have more entropy than the uniform one,
$\log K$; the die's $\log 6$ is the maximum for six outcomes.

KL is also not symmetric. Swap $p$ and $q$ in the coin example and you get
$0.9 \log\frac{0.9}{0.5} + 0.1 \log\frac{0.1}{0.5} = 0.529 - 0.161 = 0.368$
instead of 0.511, and the direction matters in practice.
$\mathrm{KL}(q\|p)$ contains the term $q(x)\log\frac{q(x)}{p(x)}$, which grows
without limit when $p(x) \to 0$ at an outcome the truth produces, so the $p$
that minimizes it puts probability on every outcome $q$ produces.
$\mathrm{KL}(p\|q)$ weights each term by $p(x)$, so an outcome with
$p(x) = 0$ adds nothing; the $p$ that minimizes it can put all its
probability on one region where $q$ is high and ignore the rest.

#### Where these show up in ML

Maximum likelihood minimizes a KL. Let $q$ be the true distribution the
data comes from. The average of $-\log p_\theta(x)$ over a large dataset
drawn from $q$ approaches $\sum_x q(x)\,(-\log p_\theta(x))$, which is the
cross-entropy $H(q, p_\theta) = H(q) + \mathrm{KL}(q\|p_\theta)$. The first
term doesn't depend on the model at all, so training only ever reduces the
second. That's the covering direction, which is why a model trained by
maximum likelihood ends up assigning probability to everything that appears
in its data.

RLHF uses the other direction. A language model defines a distribution over
outputs, called its **policy**, $\pi_\theta$. RLHF fine-tunes it with a
penalty $\mathrm{KL}(\pi_\theta\|\pi_{ref})$, where $\pi_{ref}$ is the model
before fine-tuning (module 7, §3.2). The penalty grows when $\pi_\theta$ puts
probability on outputs where $\pi_{ref}$ is small. That keeps $\pi_\theta$
close to $\pi_{ref}$ while still allowing it to concentrate on fewer
outputs.

Knowledge distillation is another example. Instead of training a small
student model on one-hot labels, you train it on the teacher's full output
distribution. The loss floor becomes the teacher's entropy rather than zero,
and the student's targets include the teacher's probabilities on the wrong
classes, not just its top answer.

#### Mutual information

A related question: how much does knowing one variable tell you about
another? That's **mutual information**. It's the KL between how often pairs
of values actually occur together, $p(x, y)$, and how often they would if
the two were unrelated, $p(x)\,p(y)$:

$$I(X; Y) = \sum_{x, y} p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)}$$

Because it's a KL, it is never negative, and it's zero exactly when
$p(x, y) = p(x)\,p(y)$ everywhere, which is independence. There's a second
way to write it. The **conditional entropy** $H(X \mid Y)$ is the entropy of
$X$ computed separately for each value of $Y$ and then averaged over $Y$:
$H(X \mid Y) = -\sum_{x,y} p(x, y)\log p(x \mid y)$. Since
$\frac{p(x, y)}{p(y)} = p(x \mid y)$, the log above splits into
$\log p(x \mid y) - \log p(x)$, and summing each part gives

$$I(X; Y) = H(X) - H(X \mid Y)$$

which reads: the uncertainty in $X$, minus what's left once you know $Y$. You can rank
features by how much mutual information they share with the label. The
contrastive loss of module 4, which trains embeddings by matching pairs such
as an image and its caption, is built so that lowering the loss raises a
lower bound on the mutual information between the two.

#### Entropy of a Gaussian, and why it's the maximum

For a bell with standard deviation $\sigma$, the $-\log$ density from §3.2 is
$\frac{(x-\mu)^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)$. Averaging it
over the bell (an integral, since the values are continuous), the
$(x - \mu)^2$ averages to $\sigma^2$ by the definition of variance, so the
first term becomes $\frac{1}{2}$:

$$H = \tfrac{1}{2} + \tfrac{1}{2}\log(2\pi\sigma^2) = \tfrac{1}{2}\log(2\pi e\sigma^2)$$

It grows with $\sigma$ and doesn't care where the bell is centered, which
makes sense: sliding a bell left or right doesn't make it any more or less
predictable. For $\sigma = 2$ it's about 2.11 nats.

This gives one more reason the Gaussian is the default choice for noise:
among all distributions with a given mean and variance, it has the highest
entropy.
Take any distribution $p$ with some mean and variance, and the Gaussian $g$
with the same two numbers. Scoring $p$'s outcomes with $g$ costs exactly
$H(g)$ on average, because $-\log g$ only involves $(x - \mu)^2$, which
averages to $\sigma^2$ for both. So the cross-entropy $H(p, g)$ equals
$H(g)$. Cross-entropy is entropy plus KL, $H(p, g) = H(p) + \mathrm{KL}(p\|g)$,
so this turns into
$\mathrm{KL}(p\|g) = H(g) - H(p)$. Gibbs' inequality says this is
$\ge 0$. No distribution with that mean and
variance is more unpredictable than the Gaussian, so assuming Gaussian noise
when that's all you know adds no structure beyond those two numbers.

## 3.6 Averages: error bars and the central limit theorem

Almost every number you compute from data is an average. A test accuracy is
the average of a 0 or 1 per example; a minibatch gradient (computed on a small
random subset of the training set) is the average of per-example gradients. So it's worth knowing exactly how much an average can
be trusted. Two things happen as you average more draws: the typical error
shrinks like $\sigma/\sqrt{n}$, and the distribution of the average turns into
a bell, whatever the individual draws looked like, as long as their variance
is finite.

#### The $\sigma/\sqrt{n}$ error bar

Take $n$ draws that are independent and all come from the same distribution,
with mean $\mu$ and variance $\sigma^2$. Their average $\bar X$ has mean $\mu$,
by linearity. Its variance is $\sigma^2/n$: the sum of the $n$ draws has
variance $n\sigma^2$, because independent variances add, and dividing the
sum by $n$ divides the variance by $n^2$ (the $\mathrm{Var}(aX) =
a^2\,\mathrm{Var}(X)$ rule with $a = 1/n$). So the typical
error of the average, its standard deviation, is $\sigma/\sqrt{n}$. This is
called the **standard error**, and it's the error bar on anything you compute
by averaging.

For example, the **exponential distribution**, with density $e^{-x}$ for
$x \ge 0$, describes waiting times: most values near zero and a long tail to
the right. Its mean and standard deviation are both 1. (Integrating
$x e^{-x}$ and $x^2 e^{-x}$ over $x \ge 0$ gives $\mathbb{E}[X] = 1$ and
$\mathbb{E}[X^2] = 2$, so the variance is $2 - 1^2 = 1$.) Average 30 independent
draws from it and the average has a standard deviation of only
$1/\sqrt{30} \approx 0.18$.

The error does shrink as $n$ grows, but slowly. Because of the square root,
halving your error bar takes four times as much data.

#### The bell shape

The exponential distribution is very lopsided, but averages of its draws are
not:

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

*The distribution of the average of $n$ exponential draws, with the curves
scaled to matching heights. It narrows around the mean (that's
$\sigma/\sqrt{n}$) and becomes a symmetric bell (that's the central limit
theorem).*

The **central limit theorem** (CLT) says this happens for any distribution
with a finite variance. If you measure
how far the average lands from $\mu$ in units of its standard error, that
number follows the standard bell, centered at 0 with standard deviation 1, more and more
closely as $n$ grows:

$$\frac{\bar X - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

The house-price table in §3.2 was this effect. Each push was uniform on
−1k to +1k, with variance $1/3$, so the sum of 20 had standard deviation
$\sqrt{20/3} \approx 2.6$k, and its histogram was already a bell.

Why should sums turn into bells? Think about two dice. A total of 7 can
happen six ways (1+6, 2+5, and so on up to 6+1), but a total of 2 or 12 only
one way each. Middle totals have lots of combinations and extreme totals have
few, so one die's flat distribution becomes a triangle for two. The figure
below builds that triangle by adding six shifted copies of one die's
distribution, one for each value of the second die. That shift-and-add has a
name, **convolution**.

<div style="margin:1.25rem 0">
<svg viewBox="-32 0 552 250" role="img" aria-label="Six copies of a die's flat distribution, each slid one step further right, stack into a triangle: the total 7 is covered by all six copies while the totals 2 and 12 are covered by one each" style="max-width:560px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <line x1="240" y1="18" x2="240" y2="112" stroke="#f59e0b" stroke-dasharray="4 4" stroke-opacity="0.7"/>
  <line x1="240" y1="149" x2="240" y2="218" stroke="#f59e0b" stroke-dasharray="4 4" stroke-opacity="0.7"/>
  <g fill="currentColor" fill-opacity="0.3">
    <rect x="22" y="24" width="36" height="9" rx="2"/>
    <rect x="62" y="24" width="36" height="9" rx="2"/>
    <rect x="102" y="24" width="36" height="9" rx="2"/>
    <rect x="142" y="24" width="36" height="9" rx="2"/>
    <rect x="182" y="24" width="36" height="9" rx="2"/>
    <rect x="222" y="24" width="36" height="9" rx="2"/>
    <rect x="62" y="39" width="36" height="9" rx="2"/>
    <rect x="102" y="39" width="36" height="9" rx="2"/>
    <rect x="142" y="39" width="36" height="9" rx="2"/>
    <rect x="182" y="39" width="36" height="9" rx="2"/>
    <rect x="222" y="39" width="36" height="9" rx="2"/>
    <rect x="262" y="39" width="36" height="9" rx="2"/>
    <rect x="102" y="54" width="36" height="9" rx="2"/>
    <rect x="142" y="54" width="36" height="9" rx="2"/>
    <rect x="182" y="54" width="36" height="9" rx="2"/>
    <rect x="222" y="54" width="36" height="9" rx="2"/>
    <rect x="262" y="54" width="36" height="9" rx="2"/>
    <rect x="302" y="54" width="36" height="9" rx="2"/>
    <rect x="142" y="69" width="36" height="9" rx="2"/>
    <rect x="182" y="69" width="36" height="9" rx="2"/>
    <rect x="222" y="69" width="36" height="9" rx="2"/>
    <rect x="262" y="69" width="36" height="9" rx="2"/>
    <rect x="302" y="69" width="36" height="9" rx="2"/>
    <rect x="342" y="69" width="36" height="9" rx="2"/>
    <rect x="182" y="84" width="36" height="9" rx="2"/>
    <rect x="222" y="84" width="36" height="9" rx="2"/>
    <rect x="262" y="84" width="36" height="9" rx="2"/>
    <rect x="302" y="84" width="36" height="9" rx="2"/>
    <rect x="342" y="84" width="36" height="9" rx="2"/>
    <rect x="382" y="84" width="36" height="9" rx="2"/>
    <rect x="222" y="99" width="36" height="9" rx="2"/>
    <rect x="262" y="99" width="36" height="9" rx="2"/>
    <rect x="302" y="99" width="36" height="9" rx="2"/>
    <rect x="342" y="99" width="36" height="9" rx="2"/>
    <rect x="382" y="99" width="36" height="9" rx="2"/>
    <rect x="422" y="99" width="36" height="9" rx="2"/>
  </g>
  <g fill="currentColor" fill-opacity="0.65" font-size="10">
    <text x="18" y="32" text-anchor="end">v = 1</text>
    <text x="58" y="47" text-anchor="end">v = 2</text>
    <text x="98" y="62" text-anchor="end">v = 3</text>
    <text x="138" y="77" text-anchor="end">v = 4</text>
    <text x="178" y="92" text-anchor="end">v = 5</text>
    <text x="218" y="107" text-anchor="end">v = 6</text>
  </g>
  <text x="240" y="126" text-anchor="middle" fill="currentColor" fill-opacity="0.7" font-size="10.5">add the cells in each column</text>
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

*Top: six copies of one die's flat distribution, one for each value $v$ of
the second die, shifted one step apart. Bottom: adding the cells in each
column counts the ways to make each total $t$, out of 36.*

Every extra draw repeats the shift-and-add. Each value of the result is a sum
over many values of the input, so each repeat averages away a bit more of the
original shape, like the exponential's spike at zero, while the
mean and variance just keep adding. And once the shape is a bell it stays a
bell, because the sum of independent Gaussians is again a Gaussian. (`probability_demos.ipynb`
tracks this numerically.)

#### Where ML uses it

A minibatch gradient is an average
of per-example gradients, so its noise shrinks like $1/\sqrt{\text{batch
size}}$: quadrupling the batch only halves the noise. Test accuracy is an
average of Bernoulli draws with variance $p(1-p)$, so on $n$ examples its
error bar is $\sigma/\sqrt{n} = \sqrt{p(1-p)/n}$. With 1,000 test examples at 90% accuracy,
that's $\sqrt{0.9 \times 0.1 / 1000} \approx 0.0095$, about ±1 point, which
is why a half-point improvement on that test set means very little. And an
error bar of ±2 standard errors is called a 95% **confidence interval**,
because about 95% of a bell lies within 2 standard deviations of its center
(95.4%, or exactly 95% at 1.96). If you repeated the
experiment many times, intervals built this way would contain the true value
about 95% of the time.

## 3.7 Bias and variance: two ways an estimate misses

Anything you compute from a random sample is itself random: draw a different
sample and you'd get a different number. §3.6 measured how much an average
jumps around. But there's a second, separate way for an estimate to be wrong,
and it matters just as much: it can be off on average.

Some vocabulary first. The **target** $\theta$ is the true number you want to
know, like the die's mean. An **estimator** is the formula you apply to a
sample, and its output is the **estimate** $\hat\theta$. The coin's
$\hat p = h/n$ from §3.4 is an estimator, and so is anything fit by maximum
likelihood.

One estimator for the die's mean is to roll three times and average. One
sample might give rolls 2, 6, 3 and an estimate of 3.67; the next might give
1, 2, 4 and 2.33; the next 5, 3, 6 and 4.67. Each individual estimate is off,
sometimes by more than a full point. But averaged over many samples, the
estimates land right on 3.5, and they scatter around it with variance about
0.97, which is $2.92/3$ from §3.6. That gives us the
two kinds of error:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

**Bias** is a systematic miss, the kind that wouldn't go away even if you
averaged estimates from thousands of fresh samples. **Variance** is how much
the estimate jumps around from sample to sample. Our three-roll average has
zero bias and a variance of about 0.97.

If you score an estimator by its mean squared error,
$\mathbb{E}[(\hat\theta - \theta)^2]$ (the same formula as the MSE loss, now
averaged over samples), the two kinds of error simply add. Split the miss
into how far this estimate is from the estimator's own average, plus how far
that average is from the truth:

$$\hat\theta - \theta = \big(\hat\theta - \mathbb{E}[\hat\theta]\big) + \big(\mathbb{E}[\hat\theta] - \theta\big)$$

The second piece is the bias, a fixed number. Square both sides with
$(a + b)^2 = a^2 + b^2 + 2ab$ and average:

$$\mathbb{E}[(\hat\theta - \theta)^2] = \underbrace{\mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]}_{\mathrm{Var}} + \mathrm{Bias}^2 + 2\,\mathrm{Bias} \cdot \mathbb{E}\big[\hat\theta - \mathbb{E}[\hat\theta]\big]$$

The last average is zero by linearity, since
$\mathbb{E}[\hat\theta] - \mathbb{E}[\hat\theta] = 0$. What's left is

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

Some estimators *are* biased, including one you've probably used. To
estimate a variance from a sample, the obvious thing is to average the
squared distances from the sample mean, dividing by $n$. (NumPy's variance
function divides by $n$ by default.)

Check it on dice. There are 216 equally likely samples of three rolls.
Compute the divide-by-$n$ variance of each and average them: you get 1.94.
The true variance is 2.92, so the estimate is a third short on average.
Divide by $n - 1$ instead and the average is exactly 2.92. Why?

The variance is defined as the average squared distance from the *true*
mean. But the true mean isn't available from the sample, so you measure from
the sample's own mean $\bar X$ instead. And $\bar X$ is special: it's the
point that makes the sum of squared distances to that particular sample as
small as possible. (The derivative of $\sum_i (x_i - c)^2$ with respect to
$c$ is $-2\sum_i (x_i - c)$, which is zero when $c$ is the average of the
$x_i$.)

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

*The sum of squared distances from the rolls 2, 6, 3 to a point $c$. It's
smallest at the sample mean $\bar x$, and higher at the true mean $\mu$ by
exactly $n(\bar x - \mu)^2$.*

So measuring from $\bar X$ always undercounts the spread a little. How much?
Write $X_i - \mu = (X_i - \bar X) + (\bar X - \mu)$ and sum the squares. The
deviations from the sample mean always add up to zero, so the cross term
$2(\bar X - \mu)\sum_i (X_i - \bar X)$ vanishes, which leaves

$$\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$$

Now average both sides over samples. Each $(X_i - \mu)^2$ averages to
$\sigma^2$ by the definition of variance, so the first sum on the right
averages $n\sigma^2$. $\bar X$ has mean $\mu$, so $(\bar X - \mu)^2$ averages
to $\mathrm{Var}(\bar X) = \sigma^2/n$ (§3.6), and the subtracted term
averages $n \cdot \sigma^2/n = \sigma^2$, exactly one draw's worth of
variance. So the sum of squared deviations
averages $(n-1)\sigma^2$, and dividing by $n - 1$ instead of $n$ makes the
estimate exactly right on average. That's **Bessel's correction**.

The fact that the deviations from the sample mean add up to zero also gives
another way to see
the $n - 1$: once you know $n - 1$ of them, the last one is fixed, so a
sample of $n$ only has $n - 1$ free deviations. Take the extreme case of a
single draw. Its one deviation is zero. Dividing by $n$ would claim the
distribution has no spread at all; dividing by $n - 1$ gives $0/0$,
undefined, which is the correct answer to "how spread out is this,
based on one example?"

#### The tradeoff in ML terms

Training a model is also an estimator. The sample is your training set, the formula is the whole
training procedure, and the estimate is the trained model's prediction at
some input $x$. "Drawing a different sample" means collecting a different
training set and retraining.

Seen this way, bias is how far the model's predictions would be from the
truth even if you averaged over many training sets. It comes from what the
model simply can't express. Variance is how much the predictions change from
one training set to the next. It comes from the model fitting the quirks and
noise of its particular training set.

A linear model is high-bias and low-variance. If the true relationship
curves, a straight line will miss it in the same places every time, but
retrain it on new data and you get nearly the same line. A deep net or a
decision tree grown without a depth limit is the opposite: it can fit almost any shape, but each
fit also follows its own training set's noise, so two fits on two samples
disagree.

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

*As model flexibility grows, $\mathrm{Bias}^2$ falls and $\mathrm{Var}$ rises.
Their sum is smallest somewhere in between (the dot).*

The standard fixes each move one of the two. **Regularization**
adds a penalty on large weights to the loss, which reduces how closely the
fit follows the training set, trading a little bias for less
variance. More data lowers variance directly, the same way averaging more
draws shrinks $\sigma^2/n$. Bagging trains several high-variance models on
resampled data and averages them, which cancels much of the variance while
leaving the bias about where it was.

For a model's test error there's one more term:
$\mathrm{Bias}^2 + \mathrm{Var} + \text{irreducible noise}$. That last term is
the noise in the labels themselves (§3.4), which no model can predict. The
evaluation module derives it in full.

## 3.8 Hypothesis tests: is the difference real?

Model A scores 91.2% and model B 90.5% on the same 1,000 test examples. Is A
better? Checkout page B was shown to 50,000 users and converted 2.3% of
them; page A, shown to another 50,000, converted 2.1%. Should you ship B? A hypothesis test answers questions like these by
asking a slightly backward one: if there were no real difference, how often
would noise alone produce a gap this big? If the answer is "rarely", you call
the difference real.

#### The procedure

You start by assuming there's no real difference. That assumption is called
the **null hypothesis**, $H_0$. Then you measure the gap you observed in units
of its standard error (SE), $z = \text{gap} / \text{SE}$. By the central limit
theorem, if $H_0$ is true, $z$ follows the standard bell. The **p-value** is
the probability, under $H_0$, of a gap at least as large as yours in either
direction. If it's below a threshold you chose in advance, usually 0.05, you
call the difference **statistically significant**.

For the checkout pages, $H_0$ says both pages share one true rate, so you
estimate it by **pooling**: treating all 100,000 users as one group, which
gives a conversion rate of 0.022. The standard error of the gap is then
$\sqrt{0.022 \times 0.978 \times 2/50{,}000} \approx 0.000928$, so $z = 0.002 / 0.000928 \approx 2.16$. The area of the standard bell
beyond ±2.16 is $p \approx 0.031$: that's the chance of noise producing a gap
at least that large in either direction.

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

*The checkout test lands at $z = 2.16$. The p-value is the shaded area: how
often noise alone would push $z$ that far from zero in either direction.*

#### Two proportions, derived

Where did that standard error come from? Each conversion rate is an average
of 0/1 outcomes, so it's a Bernoulli average with variance $p(1-p)/n$
(§3.2, §3.6). The two pages were shown to different users, so the two rates
are independent and the variance of their difference is the sum of their
variances: subtracting $\hat p_A$ is adding $-1$ times it, and
$\mathrm{Var}(-\hat p_A) = (-1)^2\,\mathrm{Var}(\hat p_A) = \mathrm{Var}(\hat p_A)$
(§3.1). Under $H_0$ both pages have the same true rate, which we estimate by
pooling everyone together into $\hat p$. Adding $\hat p(1-\hat p)/n_A$ and
$\hat p(1-\hat p)/n_B$ and taking the square root gives

$$\text{SE} = \sqrt{\hat p(1-\hat p)\Big(\frac{1}{n_A} + \frac{1}{n_B}\Big)},
  \qquad z = \frac{\hat p_B - \hat p_A}{\text{SE}}$$

For the checkout test that's $z = 2.16$ and $p = 0.031$: significant at the
0.05 level.

A confidence interval (§3.6) tells you the same thing as a test and more.
The 95% interval for the gap is the gap ± 1.96 SE. It excludes zero exactly
when the gap is more than 1.96 SE from zero, which is the same as
$|z| > 1.96$, which is the same as $p < 0.05$. For the checkout test it's
$0.002 \pm 1.96 \times 0.000928 = 0.002 \pm 0.0018$, a 95%
interval of 0.02 to 0.38 points. Significant, yes, but the size of the lift
is barely pinned down: anywhere from negligible to almost double the
observed 0.2.

The two models are a different story:

$$\text{SE} = \sqrt{\tfrac{0.912 \times 0.088}{1000} + \tfrac{0.905 \times 0.095}{1000}} \approx 0.0129,
  \qquad z = \frac{0.007}{0.0129} \approx 0.54, \qquad p \approx 0.59$$

Here each model's own accuracy goes into the formula. Pooling is a choice
made under $H_0$; when the two rates are close, both give nearly the same SE.
The result is nowhere near significant: a 0.7-point gap on 1,000 examples is
well within noise. The calculation is also a bit off, since both models were scored on
the *same* examples and so aren't independent. The right tool there is
**McNemar's test**, which only looks at the examples where the two models
disagree. If A is right and B wrong on $b$ of them, and the reverse on $c$,
then under $H_0$ each disagreement is a fair coin flip. With
$m = b + c$ disagreements, $b$ is a Binomial count with mean $m/2$ and
variance $m/4$ (§3.2). The gap $b - c = 2b - m$ then has mean 0 and variance
$4 \cdot m/4 = m$, so in standard errors it is
$z = (b - c)/\sqrt{b + c}$. A 7-example gap would only be significant if the
models disagreed on about 10 or fewer examples in total (with $b + c = 11$,
$z = 7/\sqrt{11} \approx 2.1$). With 88 and 95 errors, two models usually
disagree on dozens of examples, and the gap is noise.

#### What a p-value is not

A p-value is the probability of seeing data this extreme *if* there's no
effect. It is not the probability that there's no effect. Getting from one
to the other is exactly the Bayes problem from §3.3: you need a prior on how
often the ideas you test are real. A team that tests lots of long shots will
find that most of its "significant" results are false alarms, for the same
reason most positive tests in the rare-disease example came from healthy
people.

#### Errors, power and sample size

A test can go wrong in two ways. It can call a difference real when it
isn't, a **false positive** (statisticians call it a type I error); the
threshold sets that rate, 5% for $p < 0.05$. Or it can miss a real effect, a
type II error. The chance of catching a real effect is the test's **power**.

Power is what decides how big a test needs to be. To come out significant,
the gap has to clear 1.96 standard errors. For a real effect of size $\delta$
to clear that bar 80% of the time, it needs to sit another 0.84 standard
errors higher, since 80% of a bell lies below 0.84 standard deviations above
its center. So you need $\delta \ge 2.8$ SE. With two groups of $n$ users,
the two-proportion formula
$\text{SE} = \sqrt{p(1-p)(1/n_A + 1/n_B)}$ becomes
$\text{SE} \approx \sqrt{2p(1-p)/n}$. Squaring
$\delta \ge 2.8\sqrt{2p(1-p)/n}$ and solving for $n$ gives

$$n \;\ge\; \frac{2 \cdot 2.8^2 \cdot p(1-p)}{\delta^2} \quad\text{per group}$$

For the checkout test, with $p \approx 0.022$ and a 0.2-point lift worth
detecting, that's $2 \times 7.84 \times 0.022 \times 0.978 / 0.002^2
\approx 84{,}000$ users per group. The test only ran 50,000, so
it was underpowered. At that size a real 0.2-point lift sits just 2.16
standard errors from zero, so the measured gap is centered at 2.16 and has to
land above 1.96, which is 0.2 standard errors below its center. About 58% of
a bell lies above −0.2 standard deviations, so the power is about 58%: the
test would miss a real lift like that about 4 times in 10. And the lifts that an
underpowered test does catch tend to be overestimates, because only the
samples that happened to overshoot make it past the bar.

#### Multiple comparisons and peeking

Two habits inflate the false-alarm rate. The first is testing many things at once. Test 20 metrics when nothing real is going on, and each one has a 95%
chance of correctly coming out not significant. All 20 do with probability
$0.95^{20}$, so the chance that at least one comes out significant is
$1 - 0.95^{20} \approx 0.64$ (if the tests are independent; correlated
metrics give a somewhat lower figure). Either pick one primary metric before you
start, or divide the threshold by the number of tests (the Bonferroni
correction).

The second is peeking: checking the p-value every day and stopping the
moment it dips below 0.05. That's really many tests disguised as one, with
the same inflated false-alarm rate. Fix the sample size in advance from a
power calculation, or use a test designed for repeated looks.
