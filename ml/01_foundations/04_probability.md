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

A **Bernoulli($p$)** variable is 1 with probability $p$ and 0 with
probability $1 - p$. Flip a coin with $p = 0.3$ many times and about 30% of
the flips come out 1.

Its mean is $p$, since $1 \cdot p + 0 \cdot (1 - p) = p$. Its variance is
$p(1 - p)$: a 0/1 variable equals its own square, so $\mathbb{E}[X^2] = p$,
and $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2 = p - p^2$. For $p = 0.3$
that's $0.21$.

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

A fair coin has the largest variance because it is the hardest to predict.
A coin that almost always lands heads barely varies.

The Bernoulli shows up in three common places:

- **Counting.** Add up $n$ independent Bernoulli($p$) flips and you get
  the number of successes, which follows the **Binomial** distribution. Its
  mean is $np$ by linearity, and its variance is $np(1 - p)$ because
  independent variances add.
- **Classifiers.** A spam classifier that outputs 0.9 is saying the label
  (1 for spam, 0 for not spam) is Bernoulli(0.9).
- **Dropout.** This training method sets each neuron's output to zero at
  random. Keeping each neuron with probability $p$ keeps
  about $np$ of a layer's $n$ neurons. (PyTorch's dropout setting is the
  drop probability, so a setting of 0.1 keeps each neuron with $p = 0.9$.)

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

Now take two distributions over the same outcomes: the truth $q$ and a
model's guess $p$. The coin is fair, $q = [0.5, 0.5]$, but the model thinks
it lands heads 90% of the time, $p = [0.9, 0.1]$.

Flip the real coin many times, so half the flips are heads and half are
tails. Score each flip twice: with the model's probability, $-\log p(x)$,
and with the true one, $-\log q(x)$.

| | heads | tails | average over flips |
|---|---|---|---|
| surprise using the model, $-\log p(x)$ | 0.105 | 2.303 | 1.204 |
| surprise using the truth, $-\log q(x)$ | 0.693 | 0.693 | 0.693 |
| extra, $\log\frac{q(x)}{p(x)}$ | −0.588 | 1.609 | 0.511 |

The model's average, 1.204, is the **cross-entropy**,
$H(q, p) = -\sum_x q(x)\log p(x)$. It's the loss from §3.4. The truth's
average, 0.693, is the entropy $H(q)$: the part that comes from the coin's
own randomness, which no model can avoid. The difference, 0.511, is the
**KL divergence**, the extra surprise caused by the model's wrong
probabilities. It is the average of the bottom row:

$$\mathrm{KL}(q\|p) = \sum_x q(x)\log\frac{q(x)}{p(x)}$$

So the loss splits into two parts, the labels' own unpredictability and the
model's extra cost:

$$H(q, p) = H(q) + \mathrm{KL}(q\|p)$$

#### KL is never negative

In the coin table, the extra surprise was negative on heads, yet the average
was positive. That always holds: $\mathrm{KL}(q\|p) \ge 0$, and it is 0 only
when $p = q$ (**Gibbs' inequality**). Giving one outcome more probability
means taking it from another, and the cost there always outweighs the gain.

The proof uses $\log x \le x - 1$ for every $x > 0$ (the line $x - 1$
touches the log curve at $x = 1$ and stays above it), with
$x = p(x)/q(x)$:

$$-\mathrm{KL}(q\|p) = \sum_x q(x)\log\frac{p(x)}{q(x)} \;\le\; \sum_x q(x)\Big(\frac{p(x)}{q(x)} - 1\Big) = \sum_x p(x) - \sum_x q(x) \;\le\; 1 - 1 = 0$$

(The sums run over outcomes with $q(x) > 0$, so $\sum_x p(x) \le 1$.)

This has two consequences.

First, a loss can't go below the entropy of its labels. Cross-entropy is
entropy plus KL, and the KL is never negative, so the loss is at least the
labels' entropy, and equals it only when the model's probabilities match
the labels exactly. A one-hot label like $[1, 0, 0]$ is certain, so its
entropy is 0 and a perfect model can reach a loss of 0. A label-smoothed
label $[0.9, 0.05, 0.05]$ has entropy

$$0.9 \times (-\log 0.9) + 0.05 \times (-\log 0.05) + 0.05 \times (-\log 0.05) = 0.095 + 0.150 + 0.150 \approx 0.394$$

So even a model that outputs exactly $[0.9, 0.05, 0.05]$ has a loss of
0.394, and the training loss levels off there instead of reaching 0.

Second, among distributions over $K$ outcomes, the uniform one ($1/K$ each)
has the most entropy, $\log K$. To see it, take the KL from any $p$ to the
uniform distribution:

$$\sum_x p(x)\log\frac{p(x)}{1/K} = \sum_x p(x)\log p(x) + \log K = \log K - H(p)$$

This KL is never negative, so $H(p) \le \log K$. For six outcomes, no
distribution has more than $\log 6 \approx 1.79$ nats, the fair die's
entropy.

#### KL is not symmetric

Swapping $p$ and $q$ in the coin example gives
$0.9 \log\frac{0.9}{0.5} + 0.1 \log\frac{0.1}{0.5} = 0.368$ instead of
0.511. If the model can match $q$ exactly, both directions are minimized by
$p = q$. When it can't, which is the usual case, they prefer different
models, because each charges for a different mistake:

- $\mathrm{KL}(q\|p) = \sum_x q(x)\log\frac{q(x)}{p(x)}$ multiplies each
  outcome's term by the truth's probability $q(x)$. So every outcome the
  truth produces counts, and the model is charged when it gives one of them
  a small probability. A model of the fair coin that says "always heads",
  $p = [1, 0]$, gets an infinite $\mathrm{KL}(q\|p)$: the truth produces
  tails half the time, and the model gives tails probability 0.
- $\mathrm{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ multiplies each
  outcome's term by the model's probability $p(x)$ instead. An outcome the
  model gives probability 0 drops out, and the model is charged only for
  giving probability to outcomes the truth makes unlikely. The same "always heads" model gets
  $\mathrm{KL}(p\|q) = 1 \times \log\frac{1}{0.5} = 0.693$, a finite number:
  the model gives tails probability 0, so the tails term drops out. This direction
  lets a model ignore some of the outcomes the truth produces.

#### Where these show up in ML

- **Maximum likelihood minimizes a KL.** The training loss is the average of
  $-\log p_\theta(x)$ over the data. On a large dataset that average is the
  cross-entropy $H(q, p_\theta) = H(q) + \mathrm{KL}(q\|p_\theta)$, where
  $q$ is the true distribution of the data. $H(q)$ depends only on the data,
  so lowering the loss means lowering $\mathrm{KL}(q\|p_\theta)$, which
  moves the model's probabilities toward the data's.
- **RLHF keeps a fine-tuned model close to the original.** A language
  model's distribution over outputs is called its **policy**, $\pi_\theta$.
  RLHF fine-tunes it and adds the penalty $\mathrm{KL}(\pi_\theta\|\pi_{ref})$,
  where $\pi_{ref}$ is the model before fine-tuning (module 7, §3.2). The
  penalty grows when $\pi_\theta$ gives high probability to outputs that
  $\pi_{ref}$ rarely produces, so the fine-tuned model stays near the
  original.

  Why is the order the opposite of maximum likelihood's? A KL is an average
  over samples from its first distribution. Maximum likelihood has samples
  of the data, so the data comes first. RLHF has samples of the model's own
  outputs, because it generates them during training, so the model comes
  first.
- **Distillation trains on soft targets.** A small student model is trained
  to match a large teacher's output probabilities instead of one-hot
  labels. For a photo of a cat, the one-hot label over (cat, dog, car) is
  $[1, 0, 0]$, which only says "cat". The teacher might output
  $[0.7, 0.25, 0.05]$, which also says that dog is a plausible mistake and
  car is not, so each example carries more information. The loss is the
  cross-entropy against these targets, so its floor is their entropy, about
  0.75 here, rather than 0: a student that matches the teacher exactly
  stops at 0.75.

#### Mutual information

Mutual information measures how much learning one thing tells you about
another.

Roll a fair die. Before you learn anything, there are 6 equally likely
rolls. Each has probability $1/6$ and surprise $\log_2 6$, so your
uncertainty about the roll is $\log_2 6 = 2.58$ bits. In general, $k$ equally
likely options carry $\log_2 k$ bits. Now someone tells you something:

| what you learn | rolls still possible | uncertainty left | uncertainty removed |
|---|---|---|---|
| the exact roll | 1 | $\log_2 1 = 0$ bits | 2.58 bits |
| whether it's even | 3 | $\log_2 3 = 1.58$ bits | 1 bit |
| an unrelated coin flip | 6 | $\log_2 6 = 2.58$ bits | 0 bits |

The uncertainty removed is the **mutual information**. Writing $X$ for the
roll and $Y$ for what you learn,

$$I(X; Y) = H(X) - H(X \mid Y)$$

$H(X) = 2.58$ bits is the uncertainty about the roll before, and
$H(X \mid Y)$, the **conditional entropy**, is the uncertainty about the roll
after learning $Y$: the "uncertainty left" column.

Mutual information is never negative, since learning something can't leave
you less sure on average, and it is 0 exactly when the two are independent,
like the coin flip.

In ML it ranks input features by how much they tell you about the label.
Contrastive learning (module 4), which trains a model to match pairs such as
a photo and its caption, raises the mutual information between the two.

## 3.6 Averages: error bars and the central limit theorem

Almost every number you compute from data is an average. Test accuracy is
the average of a 0 or 1 per example. A minibatch gradient, computed on a
small random subset of the training set, is the average of per-example
gradients. Measure the same thing on different data and the average comes
out a little different, so you need to know how different. Two facts answer
that:

- The typical error shrinks as you average more examples: with $n$
  examples it is $\sigma/\sqrt{n}$, where $\sigma$ is the standard deviation
  of a single example.
- The possible values of the average follow a bell curve, whatever a single
  example looks like.

#### The $\sigma/\sqrt{n}$ error bar

Averaging cancels noise. Some draws land high and some low, and the highs
and lows partly cancel. The more draws, the more they cancel.

To see exactly how much, take $n$ independent draws $X_1, \dots, X_n$ from
one distribution with mean $\mu$ and variance $\sigma^2$. Their average is
$\bar X = \frac{1}{n}(X_1 + \dots + X_n)$.

- **Its mean is $\mu$.** Expectations split over sums (linearity, §3.1),
  so $\mathbb{E}[\bar X] = \frac{1}{n}(\mu + \dots + \mu) = \frac{1}{n} \cdot n\mu = \mu$.
  On average, the average lands on the true mean.
- **Its variance is $\sigma^2/n$.** Independent variances add, so the sum
  has variance $n\sigma^2$. Multiplying by $\frac{1}{n}$ multiplies the
  variance by $\frac{1}{n^2}$ (the rule $\mathrm{Var}(aX) = a^2\,\mathrm{Var}(X)$),
  so $\mathrm{Var}(\bar X) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$.

So the typical error of the average, its standard deviation, is
$\sigma/\sqrt{n}$. This is the **standard error**, the error bar on anything
computed by averaging. Because of the square root, halving the error bar
takes four times as much data.

For example, one die roll has standard deviation 1.71 (§3.1): a single
roll typically lands about 1.7 away from 3.5. The average of 4 rolls has
standard error $1.71/\sqrt{4} \approx 0.85$, and the average of 100 rolls
$1.71/\sqrt{100} \approx 0.17$. So the average of 100 rolls usually lands
within a few tenths of 3.5.

#### The bell shape

Roll one die and every result from 1 to 6 is equally likely: a flat
distribution. Roll two and add them, and the totals are no longer equally
likely. A total of 7 can happen six ways (1+6, 2+5, and so on up to 6+1),
but 2 and 12 only one way each, so the flat shape becomes a triangle:

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

*Each row is one die's flat distribution, shifted by the value $v$ of the
second die. Adding the cells in each column counts the ways to make each
total $t$, out of 36. This shift-and-add is called **convolution**.*

Add a third die and the triangle's corners round off. Keep adding and the
shape becomes a bell. The reason is the same each time: middle totals can be
made in many ways, and extreme totals in very few.

This happens even when a single draw is lopsided. Waiting times often follow
the **exponential distribution**: mostly short waits, occasionally long ones,
with an average wait of 1 (its density is $e^{-x}$ for $x \ge 0$). One wait
is lopsided, but the average of 30 waits is a narrow, symmetric bell:

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
$\sigma/\sqrt{n}$) and becomes a symmetric bell.*

This is the **central limit theorem** (CLT). For any distribution with a
finite variance, the average of $n$ independent draws approaches a bell
centered at $\mu$ with standard deviation $\sigma/\sqrt{n}$. Equivalently,
the distance of the average from $\mu$, measured in standard errors,
approaches the standard bell, with mean 0 and standard deviation 1:

$$\frac{\bar X - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

The house-price table in §3.2 was this effect: each push was flat, but the
sum of 20 pushes was already a bell. (`probability_demos.ipynb` tracks this
numerically.)

#### Where ML uses it

- **Minibatch noise.** A minibatch gradient is the average of per-example
  gradients over a batch, so its noise is $\sigma/\sqrt{\text{batch size}}$.
  Going from a batch of 32 to a batch of 128, four times larger, only halves
  the noise.
- **Error bars on test accuracy.** Each test example scores 1 if the model
  is right and 0 if it's wrong, a Bernoulli draw with $p$ equal to the
  accuracy. Its standard deviation is $\sqrt{p(1-p)}$, which is 0.3 at 90%
  accuracy. Averaged over 1,000 examples, the error bar is
  $0.3/\sqrt{1000} \approx 0.0095$, about ±1 point. So if a new model scores
  half a point higher on that test set, the difference is smaller than the
  noise.
- **Confidence intervals.** Take the model from the previous bullet: it
  scored 90% on 1,000 test examples. Testing it on a different 1,000
  examples would give a slightly different score, typically about 1 point
  higher or lower; that is the standard error computed above, 0.95 points.
  So how far could the model's true accuracy, its score on all possible
  examples, be from 90%? The measured accuracy follows a bell around the
  true accuracy, and 95% of a bell lies within 1.96 standard deviations of
  its center. So 95% of the time, the measurement lands within 1.96 standard
  errors of the truth, which is $1.96 \times 0.95 \approx 1.9$ points.
  Turning that around gives a range for the truth: 90% ± 1.9 points, or
  88.1% to 91.9%. That range is a 95% **confidence interval**.

Where does 1.96 come from? It is just where the bell's area reaches 95%: the
area within ±1.96 standard deviations of the center is 0.95. The area under
a bell has no simple formula, so these numbers are computed numerically.
Other common widths:

| within ± this many standard deviations | 1 | 1.96 | 2.58 | 3 |
|---|---|---|---|---|
| share of the bell | 68% | 95% | 99% | 99.7% |

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 280" role="img" aria-label="The standard bell curve with nested shaded bands: 68 percent of the area lies within 1 standard deviation of the center, 95 percent within 1.96, and 99.7 percent within 3" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <polygon points="71.4,180 71.4,178.5 72.5,178.4 73.6,178.3 74.7,178.2 75.8,178.1 76.9,178.0 78.0,177.9 79.1,177.8 80.1,177.7 81.2,177.6 82.3,177.5 83.4,177.3 84.5,177.2 85.6,177.1 86.7,176.9 87.8,176.8 88.9,176.6 90.0,176.5 91.0,176.3 92.1,176.1 93.2,175.9 94.3,175.7 95.4,175.5 96.5,175.3 97.6,175.1 98.7,174.9 99.8,174.7 100.8,174.4 101.9,174.2 103.0,173.9 104.1,173.6 105.2,173.4 106.3,173.1 107.4,172.8 108.5,172.5 109.6,172.1 110.7,171.8 111.7,171.5 112.8,171.1 113.9,170.8 115.0,170.4 116.1,170.0 117.2,169.6 118.3,169.2 119.4,168.7 120.5,168.3 121.5,167.8 122.6,167.4 123.7,166.9 124.8,166.4 125.9,165.9 127.0,165.3 128.1,164.8 129.2,164.2 130.3,163.6 131.4,163.1 132.4,162.4 133.5,161.8 134.6,161.2 135.7,160.5 136.8,159.8 136.8,180" fill="#f59e0b" fill-opacity="0.45"/>
  <polygon points="383.2,180 383.2,159.8 384.3,160.5 385.4,161.2 386.5,161.8 387.6,162.4 388.6,163.1 389.7,163.6 390.8,164.2 391.9,164.8 393.0,165.3 394.1,165.9 395.2,166.4 396.3,166.9 397.4,167.4 398.5,167.8 399.5,168.3 400.6,168.7 401.7,169.2 402.8,169.6 403.9,170.0 405.0,170.4 406.1,170.8 407.2,171.1 408.3,171.5 409.3,171.8 410.4,172.1 411.5,172.5 412.6,172.8 413.7,173.1 414.8,173.4 415.9,173.6 417.0,173.9 418.1,174.2 419.2,174.4 420.2,174.7 421.3,174.9 422.4,175.1 423.5,175.3 424.6,175.5 425.7,175.7 426.8,175.9 427.9,176.1 429.0,176.3 430.0,176.5 431.1,176.6 432.2,176.8 433.3,176.9 434.4,177.1 435.5,177.2 436.6,177.3 437.7,177.5 438.8,177.6 439.9,177.7 440.9,177.8 442.0,177.9 443.1,178.0 444.2,178.1 445.3,178.2 446.4,178.3 447.5,178.4 448.6,178.5 448.6,180" fill="#f59e0b" fill-opacity="0.45"/>
  <polygon points="136.8,180 136.8,159.8 137.8,159.2 138.8,158.5 139.8,157.9 140.8,157.2 141.8,156.5 142.8,155.8 143.8,155.0 144.8,154.3 145.9,153.5 146.9,152.8 147.9,152.0 148.9,151.2 149.9,150.3 150.9,149.5 151.9,148.6 152.9,147.8 153.9,146.9 154.9,146.0 155.9,145.1 156.9,144.1 157.9,143.2 158.9,142.2 159.9,141.2 160.9,140.2 161.9,139.2 162.9,138.2 164.0,137.2 165.0,136.1 166.0,135.0 167.0,134.0 168.0,132.9 169.0,131.8 170.0,130.6 171.0,129.5 172.0,128.3 173.0,127.2 174.0,126.0 175.0,124.8 176.0,123.6 177.0,122.4 178.0,121.2 179.0,120.0 180.0,118.7 181.1,117.5 182.1,116.2 183.1,114.9 184.1,113.6 185.1,112.4 186.1,111.1 187.1,109.8 188.1,108.5 189.1,107.1 190.1,105.8 191.1,104.5 192.1,103.2 193.1,101.9 194.1,100.5 195.1,99.2 196.1,97.9 197.1,96.5 197.1,180" fill="#10b981" fill-opacity="0.35"/>
  <polygon points="322.9,180 322.9,96.5 323.9,97.9 324.9,99.2 325.9,100.5 326.9,101.9 327.9,103.2 328.9,104.5 329.9,105.8 330.9,107.1 331.9,108.5 332.9,109.8 333.9,111.1 334.9,112.4 335.9,113.6 336.9,114.9 337.9,116.2 338.9,117.5 340.0,118.7 341.0,120.0 342.0,121.2 343.0,122.4 344.0,123.6 345.0,124.8 346.0,126.0 347.0,127.2 348.0,128.3 349.0,129.5 350.0,130.6 351.0,131.8 352.0,132.9 353.0,134.0 354.0,135.0 355.0,136.1 356.0,137.2 357.1,138.2 358.1,139.2 359.1,140.2 360.1,141.2 361.1,142.2 362.1,143.2 363.1,144.1 364.1,145.1 365.1,146.0 366.1,146.9 367.1,147.8 368.1,148.6 369.1,149.5 370.1,150.3 371.1,151.2 372.1,152.0 373.1,152.8 374.1,153.5 375.2,154.3 376.2,155.0 377.2,155.8 378.2,156.5 379.2,157.2 380.2,157.9 381.2,158.5 382.2,159.2 383.2,159.8 383.2,180" fill="#10b981" fill-opacity="0.35"/>
  <polygon points="197.1,180 197.1,96.5 199.2,93.7 201.3,91.0 203.4,88.2 205.5,85.5 207.6,82.7 209.7,80.1 211.8,77.4 213.9,74.8 216.0,72.3 218.1,69.8 220.2,67.4 222.3,65.0 224.4,62.8 226.5,60.6 228.6,58.5 230.7,56.6 232.8,54.7 234.9,52.9 237.0,51.3 239.0,49.8 241.1,48.4 243.2,47.2 245.3,46.1 247.4,45.1 249.5,44.3 251.6,43.6 253.7,43.1 255.8,42.7 257.9,42.4 260.0,42.4 262.1,42.4 264.2,42.7 266.3,43.1 268.4,43.6 270.5,44.3 272.6,45.1 274.7,46.1 276.8,47.2 278.9,48.4 281.0,49.8 283.0,51.3 285.1,52.9 287.2,54.7 289.3,56.6 291.4,58.5 293.5,60.6 295.6,62.8 297.7,65.0 299.8,67.4 301.9,69.8 304.0,72.3 306.1,74.8 308.2,77.4 310.3,80.1 312.4,82.7 314.5,85.5 316.6,88.2 318.7,91.0 320.8,93.7 322.9,96.5 322.9,180" fill="#6366f1" fill-opacity="0.35"/>
  <polyline points="33.7,179.8 36.9,179.7 40.0,179.7 43.1,179.6 46.3,179.6 49.4,179.5 52.6,179.4 55.7,179.3 58.9,179.2 62.0,179.0 65.1,178.9 68.3,178.7 71.4,178.5 74.6,178.2 77.7,177.9 80.9,177.6 84.0,177.3 87.1,176.9 90.3,176.4 93.4,175.9 96.6,175.3 99.7,174.7 102.9,174.0 106.0,173.2 109.1,172.3 112.3,171.3 115.4,170.2 118.6,169.0 121.7,167.8 124.9,166.4 128.0,164.8 131.1,163.2 134.3,161.4 137.4,159.4 140.6,157.4 143.7,155.1 146.9,152.8 150.0,150.2 153.1,147.6 156.3,144.7 159.4,141.7 162.6,138.6 165.7,135.3 168.9,131.9 172.0,128.3 175.1,124.7 178.3,120.9 181.4,117.0 184.6,113.0 187.7,109.0 190.9,104.8 194.0,100.7 197.1,96.5 200.3,92.3 203.4,88.2 206.6,84.1 209.7,80.1 212.9,76.1 216.0,72.3 219.1,68.6 222.3,65.0 225.4,61.7 228.6,58.5 231.7,55.6 234.9,52.9 238.0,50.5 241.1,48.4 244.3,46.6 247.4,45.1 250.6,43.9 253.7,43.1 256.9,42.5 260.0,42.4 263.1,42.5 266.3,43.1 269.4,43.9 272.6,45.1 275.7,46.6 278.9,48.4 282.0,50.5 285.1,52.9 288.3,55.6 291.4,58.5 294.6,61.7 297.7,65.0 300.9,68.6 304.0,72.3 307.1,76.1 310.3,80.1 313.4,84.1 316.6,88.2 319.7,92.3 322.9,96.5 326.0,100.7 329.1,104.8 332.3,109.0 335.4,113.0 338.6,117.0 341.7,120.9 344.9,124.7 348.0,128.3 351.1,131.9 354.3,135.3 357.4,138.6 360.6,141.7 363.7,144.7 366.9,147.6 370.0,150.2 373.1,152.8 376.3,155.1 379.4,157.4 382.6,159.4 385.7,161.4 388.9,163.2 392.0,164.8 395.1,166.4 398.3,167.8 401.4,169.0 404.6,170.2 407.7,171.3 410.9,172.3 414.0,173.2 417.1,174.0 420.3,174.7 423.4,175.3 426.6,175.9 429.7,176.4 432.9,176.9 436.0,177.3 439.1,177.6 442.3,177.9 445.4,178.2 448.6,178.5 451.7,178.7 454.9,178.9 458.0,179.0 461.1,179.2 464.3,179.3 467.4,179.4 470.6,179.5 473.7,179.6 476.9,179.6 480.0,179.7 483.1,179.7 486.3,179.8" fill="none" stroke="#6366f1" stroke-width="2"/>
  <line x1="33.7" y1="180" x2="486.3" y2="180" stroke="currentColor" stroke-opacity="0.5"/>
  <g fill="currentColor" fill-opacity="0.7" text-anchor="middle" font-size="10.5">
    <line x1="71.4" y1="180" x2="71.4" y2="184" stroke="currentColor"/>
    <text x="71.4" y="196">−3</text>
    <line x1="136.8" y1="180" x2="136.8" y2="184" stroke="currentColor"/>
    <text x="136.8" y="196">−1.96</text>
    <line x1="197.1" y1="180" x2="197.1" y2="184" stroke="currentColor"/>
    <text x="197.1" y="196">−1</text>
    <line x1="260.0" y1="180" x2="260.0" y2="184" stroke="currentColor"/>
    <text x="260.0" y="196">0</text>
    <line x1="322.9" y1="180" x2="322.9" y2="184" stroke="currentColor"/>
    <text x="322.9" y="196">1</text>
    <line x1="383.2" y1="180" x2="383.2" y2="184" stroke="currentColor"/>
    <text x="383.2" y="196">1.96</text>
    <line x1="448.6" y1="180" x2="448.6" y2="184" stroke="currentColor"/>
    <text x="448.6" y="196">3</text>
  </g>
  <text x="260" y="212" text-anchor="middle" fill="currentColor" fill-opacity="0.6" font-size="10">standard deviations from the center</text>
  <line x1="197.1" y1="232" x2="322.9" y2="232" stroke="#6366f1" stroke-width="2"/>
  <line x1="197.1" y1="228" x2="197.1" y2="236" stroke="#6366f1" stroke-width="2"/>
  <line x1="322.9" y1="228" x2="322.9" y2="236" stroke="#6366f1" stroke-width="2"/>
  <text x="330.9" y="236" fill="#6366f1" font-weight="600">68%</text>
  <line x1="136.8" y1="250" x2="383.2" y2="250" stroke="#10b981" stroke-width="2"/>
  <line x1="136.8" y1="246" x2="136.8" y2="254" stroke="#10b981" stroke-width="2"/>
  <line x1="383.2" y1="246" x2="383.2" y2="254" stroke="#10b981" stroke-width="2"/>
  <text x="391.2" y="254" fill="#10b981" font-weight="600">95%</text>
  <line x1="71.4" y1="268" x2="448.6" y2="268" stroke="#f59e0b" stroke-width="2"/>
  <line x1="71.4" y1="264" x2="71.4" y2="272" stroke="#f59e0b" stroke-width="2"/>
  <line x1="448.6" y1="264" x2="448.6" y2="272" stroke="#f59e0b" stroke-width="2"/>
  <text x="456.6" y="272" fill="#f59e0b" font-weight="600">99.7%</text>
</svg>
</div>

*The standard bell. The blue band, within ±1 standard deviation of the
center, holds 68% of the area. Adding the green bands, out to ±1.96, gives
95%. Adding the thin amber tails, out to ±3, gives 99.7%.*

## 3.7 Bias and variance: two ways an estimate misses

A bathroom scale can be wrong in two different ways. It can be **biased**:
it always reads 2 kg heavy. Or it can be **noisy**: step on it twice and you
get two different numbers. Any number computed from data can be wrong in the
same two ways, and the noise part is called **variance**.

Here are four scales, each weighing a 70 kg person four times:

| scale | readings | average | bias | variance | total error (MSE) |
|---|---|---|---|---|---|
| good | 69, 71, 70, 70 | 70 | 0 | 0.5 | 0.5 |
| steady but off | 72, 72, 72, 72 | 72 | 2 | 0 | 4 |
| right on average, jumpy | 64, 76, 66, 74 | 70 | 0 | 26 | 26 |
| off and jumpy | 70, 74, 71, 73 | 72 | 2 | 2.5 | 6.5 |

- **Bias** is how far the average reading is from the truth: $72 - 70 = 2$
  for the two off scales.
- **Variance** is how much the readings scatter around their own average:
  the average squared distance from it. For the jumpy scale, the readings
  are 6, 6, 4 and 4 away from 70, so the variance is
  $(36 + 36 + 16 + 16)/4 = 26$.
- **Total error**, the **mean squared error** (MSE), is the average squared
  distance from the *truth*.

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

*The four scales as darts. Bias is how far the cloud's center sits from the
bullseye; variance is how scattered it is.*

Look at the last column: total error is always bias squared plus variance.
For the off-and-jumpy scale, $2^2 + 2.5 = 6.5$. In symbols, with $\theta$ the
truth and $\hat\theta$ an estimate:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big] \qquad
  \mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$$

The two add because each reading's miss is the bias plus that reading's
distance from the scale's average. Squaring gives
$\text{bias}^2 + \text{distance}^2 + 2 \times \text{bias} \times \text{distance}$,
and the last term averages to zero because the distances average to zero.

The table also shows that unbiased is not the same as best. The steady scale
is always 2 kg off, yet its total error, 4, is far smaller than the jumpy
scale's 26, which is right on average. A little bias can be worth it if it
removes a lot of variance.

#### The tradeoff in ML terms

A trained model's prediction is an estimate too. Training on one dataset is
one reading; a different training set would give a slightly different
model.

- **A too-simple model is the steady-but-off scale.** A straight line fit to
  a curved relationship misses in the same places whatever the training set:
  high bias, low variance.
- **A too-flexible model is the jumpy scale.** A deep net or a decision tree
  with no depth limit follows every training example, so each training set's
  noise moves it: low bias, high variance.
- **The best model is in between.** **Regularization** gets there by adding
  a penalty on large weights to the loss, which pulls a flexible model toward
  a simpler one: a little more bias for a lot less variance.

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

Two other fixes target variance. **More data** lowers it directly, the same
way averaging more draws shrinks $\sigma^2/n$ (§3.6). **Bagging** trains
several flexible models on resampled data and averages them, like averaging
several readings of the jumpy scale. And a model's test error has one more
term, $\mathrm{Bias}^2 + \mathrm{Var} + \text{irreducible noise}$: the noise
in the labels themselves (§3.4), which no model can predict.

#### Bessel's correction

Variance is the average squared distance from the true mean $\mu$. With
only a sample $x_1, \dots, x_n$ you don't know $\mu$, so the obvious
estimate measures from the sample mean $\bar x$ instead:

$$\hat\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar x)^2$$

(NumPy's variance function does this by default.) On average it comes out
too small. Roll a die three times: over all 216 equally likely sets of
rolls, $\hat\sigma^2$ averages 1.94, while the die's true variance is 2.92
(§3.1).

The reason is that $\bar x$ is computed from the same rolls, so it sits in
their middle, and the rolls are closer to it than to $\mu$. For the rolls
2, 6, 3, the squared distances add up to 8.67 from $\bar x \approx 3.67$ but
8.75 from $\mu = 3.5$. The exact relation is

$$\underbrace{\sum_i (x_i - \mu)^2}_{\text{from the true mean}}
= \underbrace{\sum_i (x_i - \bar x)^2}_{\text{from the sample mean}}
+ \underbrace{n(\bar x - \mu)^2}_{\text{the gap}}$$

To get it, write each $x_i - \mu$ as $(x_i - \bar x) + (\bar x - \mu)$ and
square. The cross terms add up to $2(\bar x - \mu)$ times
$\sum_i (x_i - \bar x)$, and that sum is zero because distances from the
average sum to zero. For the rolls 2, 6, 3, the gap is 0.08, and
$8.67 + 0.08 = 8.75$.

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
the gap $n(\bar x - \mu)^2$.*

Now average each term over all possible samples:

- **From the true mean:** each $(x_i - \mu)^2$ averages $\sigma^2$, by the
  definition of variance, so the sum averages $n\sigma^2$.
- **The gap:** $(\bar x - \mu)^2$ averages $\mathrm{Var}(\bar x) = \sigma^2/n$
  (§3.6), so $n(\bar x - \mu)^2$ averages $\sigma^2$.
- **From the sample mean:** whatever is left, $n\sigma^2 - \sigma^2 = (n - 1)\sigma^2$.

So dividing the sum from the sample mean by $n$ gives an average of
$\frac{n-1}{n}\sigma^2$, which is $\frac{2}{3} \times 2.92 = 1.94$ for three
rolls. Dividing by $n - 1$ instead,

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar x)^2,$$

averages exactly $\sigma^2$. That's **Bessel's correction**.

## 3.8 Hypothesis tests: is the difference real?

You train a new model and test it against the old one on the same 1,000
examples. The new one scores 86.2%, the old one 83.8%. Is the new model
better, or did it get lucky on this test set?

Throw away the examples where the two models agree. If both got an example
right, or both got it wrong, it says nothing about which one is better. That
leaves the examples where they disagree. Say there are 100: the new model
wins 62 of them and the old one 38. Those 24 extra wins are the whole
accuracy gap, $862 - 838 = 24$ examples.

#### The test

Suppose the two models were equally good. Then each disagreement would be a
coin flip: heads the new model wins, tails the old one does. So the question
becomes: flip 100 fair coins, and how often do you get a split as lopsided
as 62 to 38?

Simulate it. Flip 100 coins and count heads: 57. Again: 44. Again: 50. Do
it 10,000 times and plot the counts:

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 270" role="img" aria-label="Histogram of how many of 100 disagreements the new model wins, over 10,000 simulated runs where both models are equally good: a bell centered at 50 with a standard deviation of 5; the 206 runs at 62 or more, or 38 or fewer, are colored red" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <text x="260.0" y="20" text-anchor="middle" fill="#6366f1" font-weight="600">10,000 runs with equally good models</text>
  <rect x="35.0" y="210.0" width="10.0" height="0.0" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="46.0" y="210.0" width="10.0" height="0.0" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="57.0" y="209.8" width="10.0" height="0.2" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="68.0" y="209.4" width="10.0" height="0.6" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="79.0" y="209.4" width="10.0" height="0.6" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="90.0" y="208.1" width="10.0" height="1.9" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="101.0" y="207.7" width="10.0" height="2.3" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="112.0" y="206.1" width="10.0" height="3.9" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="123.0" y="199.7" width="10.0" height="10.3" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="134.0" y="195.2" width="10.0" height="14.8" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="145.0" y="192.5" width="10.0" height="17.5" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="156.0" y="180.0" width="10.0" height="30.0" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="167.0" y="165.0" width="10.0" height="45.0" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="178.0" y="151.5" width="10.0" height="58.5" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="189.0" y="133.6" width="10.0" height="76.4" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="200.0" y="118.4" width="10.0" height="91.6" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="211.0" y="102.4" width="10.0" height="107.6" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="222.0" y="79.8" width="10.0" height="130.2" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="233.0" y="68.5" width="10.0" height="141.5" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="244.0" y="52.9" width="10.0" height="157.1" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="255.0" y="50.0" width="10.0" height="160.0" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="266.0" y="61.5" width="10.0" height="148.5" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="277.0" y="64.6" width="10.0" height="145.4" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="288.0" y="83.3" width="10.0" height="126.7" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="299.0" y="100.5" width="10.0" height="109.5" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="310.0" y="110.2" width="10.0" height="99.8" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="321.0" y="132.2" width="10.0" height="77.8" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="332.0" y="151.0" width="10.0" height="59.0" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="343.0" y="158.4" width="10.0" height="51.6" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="354.0" y="182.1" width="10.0" height="27.9" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="365.0" y="190.1" width="10.0" height="19.9" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="376.0" y="197.5" width="10.0" height="12.5" fill="#6366f1" fill-opacity="0.45"/>
  <rect x="387.0" y="201.8" width="10.0" height="8.2" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="398.0" y="204.7" width="10.0" height="5.3" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="409.0" y="206.5" width="10.0" height="3.5" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="420.0" y="208.8" width="10.0" height="1.2" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="431.0" y="209.0" width="10.0" height="1.0" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="442.0" y="209.2" width="10.0" height="0.8" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="453.0" y="209.6" width="10.0" height="0.4" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="464.0" y="210.0" width="10.0" height="0.0" fill="#f43f5e" fill-opacity="0.75"/>
  <rect x="475.0" y="210.0" width="10.0" height="0.0" fill="#f43f5e" fill-opacity="0.75"/>
  <line x1="34" y1="210" x2="486" y2="210" stroke="currentColor" opacity="0.5"/>
  <line x1="40.0" y1="210" x2="40.0" y2="214" stroke="currentColor" opacity="0.5"/>
  <text x="40.0" y="227" text-anchor="middle" fill="currentColor" fill-opacity="0.7">30</text>
  <line x1="128.0" y1="210" x2="128.0" y2="214" stroke="currentColor" opacity="0.5"/>
  <text x="128.0" y="227" text-anchor="middle" fill="#f43f5e">38</text>
  <line x1="260.0" y1="210" x2="260.0" y2="214" stroke="currentColor" opacity="0.5"/>
  <text x="260.0" y="227" text-anchor="middle" fill="currentColor" fill-opacity="0.7">50</text>
  <line x1="392.0" y1="210" x2="392.0" y2="214" stroke="currentColor" opacity="0.5"/>
  <text x="392.0" y="227" text-anchor="middle" fill="#f43f5e">62</text>
  <line x1="480.0" y1="210" x2="480.0" y2="214" stroke="currentColor" opacity="0.5"/>
  <text x="480.0" y="227" text-anchor="middle" fill="currentColor" fill-opacity="0.7">70</text>
  <line x1="133.5" y1="210" x2="133.5" y2="70" stroke="#f43f5e" stroke-dasharray="4 4"/>
  <line x1="386.5" y1="210" x2="386.5" y2="70" stroke="#f43f5e" stroke-dasharray="4 4"/>
  <text x="392.5" y="80" fill="#f43f5e" font-size="11">real result: 62</text>
  <text x="392.5" y="95" fill="#f43f5e" font-size="11">206 runs (2.1%)</text>
  <text x="392.5" y="109" fill="#f43f5e" font-size="11">this far from 50</text>
  <line x1="205.0" y1="115.7" x2="315.0" y2="115.7" stroke="#10b981" stroke-width="1.5"/>
  <line x1="205.0" y1="111.7" x2="205.0" y2="119.7" stroke="#10b981" stroke-width="1.5"/>
  <line x1="315.0" y1="111.7" x2="315.0" y2="119.7" stroke="#10b981" stroke-width="1.5"/>
  <text x="260.0" y="108.7" text-anchor="middle" fill="#10b981" font-size="11" font-weight="600">±5</text>
  <text x="260.0" y="250" text-anchor="middle" fill="currentColor" fill-opacity="0.8">disagreements won by the new model (out of 100)</text>
</svg>
</div>

*Heads in 100 fair flips, over 10,000 runs. The red runs are at least as
far from 50 as the real result of 62: 62 or more, or 38 or fewer.*

206 of the 10,000 runs, 2.1%, are that lopsided. So if the models were
equally good, a result like 62 to 38 would come up about 2 times in 100.
The usual cutoff is 5 times in 100. This is below it, so the usual
conclusion is that the new model is really better.

That's the whole test. The rest is names for its pieces:

- **Null hypothesis** ($H_0$): "the models are equally good", the
  assumption the simulation runs on.
- **p-value**: the fraction of simulated runs at least as lopsided as the
  real result. Here 0.021.
- **Statistically significant**: by convention, $p < 0.05$. This result is.
- **Standard error** (SE): the standard deviation of the pile, 5 wins: the
  typical distance from 50 that chance alone produces. (In §3.6 terms: each
  flip has $\sigma = \frac12$, so the fraction of wins has SE
  $\frac12/\sqrt{100} = 0.05$, which is 5 wins out of 100.)
- $z$: the distance from the middle in standard errors,
  $(62 - 50)/5 = 2.4$. The count is a sum of 100 flips, so by the central
  limit theorem the pile is close to a bell (§3.6). 95% of a bell lies
  within 1.96 SE of its center, so $p < 0.05$ is close to $|z| > 1.96$.
  (Close, not exact, because wins are whole numbers.)

**Where the 5 comes from.** The simulation measured the standard error, 5
wins, by brute force. You can also compute it in three steps:

- One flip is a Bernoulli($\frac12$) variable, so its variance is
  $p(1-p) = \frac12 \times \frac12 = \frac14$ (§3.2).
- The 100 flips are independent, so their variances add (§3.1): the number
  of heads has variance $100 \times \frac14 = 25$.
- The standard deviation is the square root of the variance: $\sqrt{25} = 5$.

With $m$ disagreements instead of 100, the same steps give variance $m/4$
and standard deviation $\sqrt m / 2$.

**$z$ from the lead.** The new model leads 62 to 38,
by 24 wins, and the lead splits in half around the middle: 12 above 50 for
the new model, 12 below for the old one. So with a lead of $L$ wins over $m$
disagreements,

$$z = \frac{L/2}{\sqrt m / 2} = \frac{L}{\sqrt m}$$

The $/2$ on top and bottom cancel. Here $24/\sqrt{100} = 2.4$, the same as
before.
Writing $b$ for the new model's wins and $c$ for the old one's, the lead is
$b - c$ and $m = b + c$, so $z = (b - c)/\sqrt{b + c}$. This recipe, keeping
only the disagreements and treating each one as a coin flip, is called
**McNemar's test**.

#### Power: how many examples you need

So far the models were assumed equal, and the question was how surprising
62 wins is. Now suppose the new model really is better: it wins 60% of
disagreements, not 50%. How often does the test come out significant?

To keep the arithmetic simple, use the bell shortcut from here on:
significant means $z > 1.96$, which is at least
$50 + 1.96 \times 5 = 59.8$, so 60 wins or more.

Simulate 100 disagreements where the new model wins each one with
probability 0.6, and count its wins. The first six runs gave 60, 62, 65, 56,
57 and 60. The runs with 56 and 57 come back "not significant", even though
the new model really is better.

Over 10,000 runs, 5,489 reach 60: about 55%. That fraction is the test's
**power**: the chance the test comes out significant when the improvement is
real. It's low because the bar sits at the center of the better model's
pile, so only about half its runs clear it (a bit over half, because runs
landing exactly on 60 count).

The fix is more disagreements. The same simulation with more of them:

| disagreements | significant at | better model expects | runs that clear the bar |
|---|---|---|---|
| 100 | 60 wins | 60 | 55% |
| 200 | 114 wins | 120 | 83% |
| 400 | 220 wins | 240 | 98% |

**The algebra.** Call the number of disagreements $m$ and the new model's
true win rate $q$ (here 0.6).

- An equal model expects $m/2$ wins, give or take $\sqrt m / 2$. So the bar is
  $\frac m2 + 1.96 \cdot \frac{\sqrt m}{2}$. For $m = 200$:
  $100 + 1.96 \times 7.07 \approx 114$.
- The better model expects $qm$ wins, give or take about the same
  $\sqrt m / 2$. (Exactly, it's $\sqrt{m\,q(1-q)}$, which is $0.49\sqrt m$ for
  $q = 0.6$: close enough.)

A common target is 80% power: the test comes out significant 4 times in 5
when the improvement is real. How many disagreements does that take? Picture
both bells. Each one's standard deviation is the standard error, written SD
below:

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 235" role="img" aria-label="Two bells: the equal model centered at m over 2 and the better model centered at q m. The bar sits 1.96 standard deviations above the equal model's center and 0.84 standard deviations below the better model's center; 80 percent of the better model's bell lies above the bar" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <polygon points="286.8,170 286.8,92.7 288.1,91.0 289.4,89.3 290.6,87.6 291.9,86.0 293.2,84.4 294.5,82.8 295.7,81.2 297.0,79.7 298.3,78.2 299.5,76.8 300.8,75.4 302.1,74.0 303.3,72.7 304.6,71.5 305.9,70.3 307.2,69.1 308.4,68.0 309.7,67.0 311.0,66.1 312.2,65.2 313.5,64.3 314.8,63.6 316.0,62.9 317.3,62.3 318.6,61.7 319.8,61.2 321.1,60.8 322.4,60.5 323.7,60.3 324.9,60.1 326.2,60.0 327.5,60.0 328.7,60.1 330.0,60.2 331.3,60.4 332.5,60.7 333.8,61.1 335.1,61.5 336.4,62.0 337.6,62.6 338.9,63.3 340.2,64.0 341.4,64.8 342.7,65.7 344.0,66.6 345.2,67.6 346.5,68.7 347.8,69.8 349.1,71.0 350.3,72.2 351.6,73.5 352.9,74.8 354.1,76.2 355.4,77.6 356.7,79.1 357.9,80.6 359.2,82.1 360.5,83.7 361.8,85.3 363.0,87.0 364.3,88.6 365.6,90.3 366.8,92.0 368.1,93.7 369.4,95.5 370.6,97.2 371.9,99.0 373.2,100.7 374.4,102.5 375.7,104.3 377.0,106.0 378.3,107.8 379.5,109.6 380.8,111.3 382.1,113.1 383.3,114.8 384.6,116.5 385.9,118.2 387.1,119.9 388.4,121.5 389.7,123.1 391.0,124.8 392.2,126.3 393.5,127.9 394.8,129.4 396.0,130.9 397.3,132.4 398.6,133.9 399.8,135.3 401.1,136.7 402.4,138.0 403.7,139.3 404.9,140.6 406.2,141.8 407.5,143.1 408.7,144.2 410.0,145.4 411.3,146.5 412.5,147.6 413.8,148.6 415.1,149.6 416.4,150.6 417.6,151.5 418.9,152.5 420.2,153.3 421.4,154.2 422.7,155.0 424.0,155.8 425.2,156.5 426.5,157.2 427.8,157.9 429.0,158.6 430.3,159.2 431.6,159.8 432.9,160.4 434.1,160.9 435.4,161.5 436.7,162.0 437.9,162.4 439.2,162.9 440.5,163.3 441.7,163.7 443.0,164.1 444.3,164.5 445.6,164.8 446.8,165.2 448.1,165.5 449.4,165.8 450.6,166.0 451.9,166.3 453.2,166.6 454.4,166.8 455.7,167.0 457.0,167.2 458.3,167.4 459.5,167.6 460.8,167.8 462.1,167.9 463.3,168.1 464.6,168.2 465.9,168.3 467.1,168.5 468.4,168.6 469.7,168.7 471.0,168.8 472.2,168.9 473.5,169.0 474.8,169.0 476.0,169.1 477.3,169.2 478.6,169.3 479.8,169.3 481.1,169.4 482.4,169.4 483.7,169.5 484.9,169.5 486.2,169.6 487.5,169.6 488.7,169.6 490.0,169.7 490.0,170" fill="#10b981" fill-opacity="0.3"/>
  <polyline points="30.0,169.7 32.9,169.6 35.8,169.5 38.6,169.4 41.5,169.3 44.4,169.1 47.2,168.9 50.1,168.7 53.0,168.5 55.9,168.2 58.8,167.8 61.6,167.4 64.5,167.0 67.4,166.4 70.3,165.8 73.1,165.2 76.0,164.4 78.9,163.5 81.7,162.5 84.6,161.4 87.5,160.2 90.4,158.9 93.2,157.4 96.1,155.7 99.0,153.9 101.9,151.9 104.8,149.8 107.6,147.4 110.5,144.9 113.4,142.3 116.2,139.4 119.1,136.4 122.0,133.2 124.9,129.9 127.8,126.4 130.6,122.7 133.5,119.0 136.4,115.2 139.2,111.3 142.1,107.3 145.0,103.3 147.9,99.3 150.8,95.3 153.6,91.4 156.5,87.6 159.4,83.9 162.2,80.4 165.1,77.0 168.0,73.9 170.9,71.0 173.8,68.5 176.6,66.2 179.5,64.2 182.4,62.6 185.2,61.4 188.1,60.5 191.0,60.1 193.9,60.0 196.8,60.4 199.6,61.1 202.5,62.2 205.4,63.7 208.2,65.5 211.1,67.7 214.0,70.1 216.9,72.9 219.8,76.0 222.6,79.2 225.5,82.7 228.4,86.3 231.3,90.1 234.1,94.0 237.0,98.0 239.9,101.9 242.7,105.9 245.6,109.9 248.5,113.9 251.4,117.7 254.2,121.5 257.1,125.2 260.0,128.7 262.9,132.1 265.8,135.4 268.6,138.4 271.5,141.3 274.4,144.1 277.2,146.6 280.1,149.0 283.0,151.2 285.9,153.2 288.8,155.1 291.6,156.8 294.5,158.4 297.4,159.8 300.2,161.0 303.1,162.2 306.0,163.2 308.9,164.1 311.8,164.9 314.6,165.6 317.5,166.3 320.4,166.8 323.2,167.3 326.1,167.7 329.0,168.1 331.9,168.4 334.8,168.6 337.6,168.8 340.5,169.0 343.4,169.2 346.2,169.3 349.1,169.5 352.0,169.6 354.9,169.6 357.7,169.7 360.6,169.8 363.5,169.8 366.4,169.8 369.2,169.9 372.1,169.9 375.0,169.9 377.9,169.9 380.8,169.9 383.6,170.0 386.5,170.0 389.4,170.0 392.3,170.0 395.1,170.0 398.0,170.0 400.9,170.0 403.8,170.0 406.6,170.0 409.5,170.0 412.4,170.0 415.2,170.0 418.1,170.0 421.0,170.0 423.9,170.0 426.7,170.0 429.6,170.0 432.5,170.0 435.4,170.0 438.2,170.0 441.1,170.0 444.0,170.0 446.9,170.0 449.8,170.0 452.6,170.0 455.5,170.0 458.4,170.0 461.2,170.0 464.1,170.0 467.0,170.0 469.9,170.0 472.7,170.0 475.6,170.0 478.5,170.0 481.4,170.0 484.2,170.0 487.1,170.0 490.0,170.0" fill="none" stroke="#6366f1" stroke-width="2"/>
  <polyline points="30.0,170.0 32.9,170.0 35.8,170.0 38.6,170.0 41.5,170.0 44.4,170.0 47.2,170.0 50.1,170.0 53.0,170.0 55.9,170.0 58.8,170.0 61.6,170.0 64.5,170.0 67.4,170.0 70.3,170.0 73.1,170.0 76.0,170.0 78.9,170.0 81.7,170.0 84.6,170.0 87.5,170.0 90.4,170.0 93.2,170.0 96.1,170.0 99.0,170.0 101.9,170.0 104.8,170.0 107.6,170.0 110.5,170.0 113.4,170.0 116.2,170.0 119.1,170.0 122.0,170.0 124.9,170.0 127.8,170.0 130.6,170.0 133.5,170.0 136.4,170.0 139.2,169.9 142.1,169.9 145.0,169.9 147.9,169.9 150.8,169.9 153.6,169.8 156.5,169.8 159.4,169.8 162.2,169.7 165.1,169.6 168.0,169.6 170.9,169.5 173.8,169.3 176.6,169.2 179.5,169.0 182.4,168.8 185.2,168.6 188.1,168.4 191.0,168.1 193.9,167.7 196.8,167.3 199.6,166.8 202.5,166.3 205.4,165.6 208.2,164.9 211.1,164.1 214.0,163.2 216.9,162.2 219.8,161.0 222.6,159.8 225.5,158.4 228.4,156.8 231.3,155.1 234.1,153.2 237.0,151.2 239.9,149.0 242.7,146.6 245.6,144.1 248.5,141.3 251.4,138.4 254.2,135.4 257.1,132.1 260.0,128.7 262.9,125.2 265.8,121.5 268.6,117.7 271.5,113.9 274.4,109.9 277.2,105.9 280.1,101.9 283.0,98.0 285.9,94.0 288.8,90.1 291.6,86.3 294.5,82.7 297.4,79.2 300.2,76.0 303.1,72.9 306.0,70.1 308.9,67.7 311.8,65.5 314.6,63.7 317.5,62.2 320.4,61.1 323.2,60.4 326.1,60.0 329.0,60.1 331.9,60.5 334.8,61.4 337.6,62.6 340.5,64.2 343.4,66.2 346.2,68.5 349.1,71.0 352.0,73.9 354.9,77.0 357.7,80.4 360.6,83.9 363.5,87.6 366.4,91.4 369.2,95.3 372.1,99.3 375.0,103.3 377.9,107.3 380.8,111.3 383.6,115.2 386.5,119.0 389.4,122.7 392.3,126.4 395.1,129.9 398.0,133.2 400.9,136.4 403.8,139.4 406.6,142.3 409.5,144.9 412.4,147.4 415.2,149.8 418.1,151.9 421.0,153.9 423.9,155.7 426.7,157.4 429.6,158.9 432.5,160.2 435.4,161.4 438.2,162.5 441.1,163.5 444.0,164.4 446.9,165.2 449.8,165.8 452.6,166.4 455.5,167.0 458.4,167.4 461.2,167.8 464.1,168.2 467.0,168.5 469.9,168.7 472.7,168.9 475.6,169.1 478.5,169.3 481.4,169.4 484.2,169.5 487.1,169.6 490.0,169.7" fill="none" stroke="#10b981" stroke-width="2"/>
  <line x1="30" y1="170" x2="490" y2="170" stroke="currentColor" opacity="0.5"/>
  <line x1="286.8" y1="170" x2="286.8" y2="42" stroke="#f43f5e" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="286.8" y="36" text-anchor="middle" fill="#f43f5e" font-weight="600">bar</text>
  <text x="192.9" y="54" text-anchor="middle" fill="#6366f1">equal model</text>
  <text x="355.1" y="54" text-anchor="middle" fill="#10b981">better model</text>
  <text x="379.8" y="140" text-anchor="middle" fill="#10b981" font-weight="600">80%</text>
  <line x1="192.9" y1="170" x2="192.9" y2="175" stroke="#6366f1"/>
  <text x="192.9" y="188" text-anchor="middle" fill="#6366f1">m/2</text>
  <line x1="327.1" y1="170" x2="327.1" y2="175" stroke="#10b981"/>
  <text x="327.1" y="188" text-anchor="middle" fill="#10b981">qm</text>
  <line x1="192.9" y1="210" x2="286.8" y2="210" stroke="currentColor" opacity="0.7"/>
  <line x1="192.9" y1="206" x2="192.9" y2="214" stroke="currentColor" opacity="0.7"/>
  <line x1="286.8" y1="206" x2="286.8" y2="214" stroke="currentColor" opacity="0.7"/>
  <text x="239.9" y="227" text-anchor="middle" fill="currentColor">1.96 SD</text>
  <line x1="286.8" y1="210" x2="327.1" y2="210" stroke="currentColor" opacity="0.7"/>
  <line x1="286.8" y1="206" x2="286.8" y2="214" stroke="currentColor" opacity="0.7"/>
  <line x1="327.1" y1="206" x2="327.1" y2="214" stroke="currentColor" opacity="0.7"/>
  <text x="307.0" y="227" text-anchor="middle" fill="currentColor">0.84 SD</text>
  <line x1="286.8" y1="170" x2="286.8" y2="210" stroke="#f43f5e" stroke-dasharray="2 3" opacity="0.6"/>
</svg>
</div>

*The bar is 1.96 SD above the equal model's center. The better model clears
it whenever it lands to the right of the bar (shaded). For that to happen
80% of the time, the bar must be 0.84 SD left of the better model's center:
80% of a bell lies to the right of that point.*

Where 0.84 comes from: 68% of a bell lies within 1 SD of the center (§3.6),
so 16% is below $-1$ SD and 84% is above it. To leave exactly 80% above, the
point moves slightly toward the center, to $-0.84$ SD (a value read from a
bell table, like 1.96).

Reading the distances off the picture, from $m/2$ to $qm$ is $1.96 + 0.84$
standard deviations. So you need

$$qm \;\ge\; \frac m2 + (1.96 + 0.84)\,\frac{\sqrt m}{2}$$

Solve for $m$ step by step:

$$\begin{aligned}
\big(q - \tfrac12\big)\,m &\ge 1.4\sqrt m
&&\text{subtract } \tfrac m2\text{; } 2.8/2 = 1.4 \\
\big(q - \tfrac12\big)\sqrt m &\ge 1.4
&&\text{divide by } \sqrt m \\
m &\ge \Big(\frac{1.4}{q - \frac12}\Big)^2
&&\text{divide by } q - \tfrac12\text{, then square}
\end{aligned}$$

For $q = 0.6$ that's $m \ge (1.4 / 0.1)^2 = 196$. The simulation gave 83%
at 200 disagreements, just above 80%, since 200 is a bit more than 196 and
the better model's spread is slightly under $\sqrt m/2$.

Why more examples help: double the disagreements and the better model's
expected wins above the middle double ($60 - 50 = 10$ becomes
$120 - 100 = 20$), but the standard error grows only by $\sqrt 2$ (5 becomes
7). So the gap stands out more and more from chance.

Why small improvements are expensive: the edge (how far the new model's
win rate is above 50%) is squared in the formula. Halving the edge needs
four times the disagreements, and a tenth of the edge needs 100 times.

| new model wins (of disagreements) | disagreements for 80% power |
|---|---|
| 60% | 196 |
| 55% | 784 |
| 51% | 19,600 |

Disagreements are only part of the test set. In the opening example the
models disagreed on 100 of 1,000 examples, 10%. At that rate, 784
disagreements take about 7,800 test examples.

#### Common traps

**Reading p as the chance the models are equal.** $p = 0.021$ is the chance
of a result this lopsided *if* the models are equal. It is not the chance
that they are equal. It's the same mix-up as the rare-disease test (§3.3).
Say you try 200 changes and 20 of them really help. At 80% power, 16 of the
20 pass. Of the 180 useless ones, 5% pass: 9. So 9 of the 25 changes that
pass, 36%, are useless, not 5%.

**Testing many things.** Try 20 useless changes. A useless change behaves
like the equal-models simulation, so each one passes 5% of the time. If the
tests are independent, the chance that at least one passes is
$1 - 0.95^{20} \approx 64\%$, and the one that passed is a fluke. Either
pick the one comparison before you look, or require
$p < 0.05/20 = 0.0025$ (the **Bonferroni correction**): then the chance that
any of the 20 passes is at most $20 \times 0.0025 = 5\%$.

**Peeking.** Re-run the test after every few hundred examples and stop as
soon as $p < 0.05$, and you're running many tests and keeping the best
one.
Decide the test size in advance.
