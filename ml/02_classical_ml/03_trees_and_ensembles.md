# Trees and ensembles: decision trees, bagging, random forests, boosting

Companion notebook: `trees_and_ensembles_scratch.ipynb`.

---

## 7. Decision trees

### 7.1 What a tree is, and how mixed a group is

A decision tree is a flowchart of yes/no questions about the features. To
classify an email, start at the top, answer each question, and follow the
branch until you reach a **leaf**. The leaf predicts the most common label
among the training emails that ended there.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 250" role="img" aria-label="A decision tree for spam. The root asks whether the email contains the word free. If no, a leaf predicts not spam (3 spam, 60 not). If yes, the next question asks whether it has more than 3 links. If yes, a leaf predicts spam (40 spam, 2 not); if no, a leaf predicts not spam (5 spam, 15 not). The path for an email with free and 5 links is highlighted and ends at spam." style="max-width:540px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <line x1="260" y1="58" x2="150" y2="112" stroke="#f59e0b" stroke-opacity="1" stroke-width="2.4"/>
  <text x="191" y="85" text-anchor="middle" fill="currentColor" fill-opacity="0.8">yes</text>
  <line x1="260" y1="58" x2="390" y2="112" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.3"/>
  <text x="339" y="85" text-anchor="middle" fill="currentColor" fill-opacity="0.8">no</text>
  <line x1="150" y1="148" x2="75" y2="192" stroke="#f59e0b" stroke-opacity="1" stroke-width="2.4"/>
  <text x="98" y="170" text-anchor="middle" fill="currentColor" fill-opacity="0.8">yes</text>
  <line x1="150" y1="148" x2="245" y2="192" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.3"/>
  <text x="212" y="170" text-anchor="middle" fill="currentColor" fill-opacity="0.8">no</text>
  <rect x="175.0" y="22.0" width="170" height="36" rx="7" fill="#64748b" fill-opacity="0.14" stroke="#64748b" stroke-width="1.4"/>
  <text x="260" y="44" text-anchor="middle" fill="currentColor">contains “free”?</text>
  <rect x="65.0" y="112.0" width="170" height="36" rx="7" fill="#64748b" fill-opacity="0.14" stroke="#64748b" stroke-width="1.4"/>
  <text x="150" y="134" text-anchor="middle" fill="currentColor">more than 3 links?</text>
  <rect x="315.0" y="108.0" width="150" height="44" rx="7" fill="#6366f1" fill-opacity="0.14" stroke="#6366f1" stroke-width="1.4"/>
  <text x="390" y="126" text-anchor="middle" fill="currentColor" font-weight="600">leaf: not spam</text>
  <text x="390" y="142" text-anchor="middle" fill="currentColor">3 spam · 60 not</text>
  <rect x="10.0" y="190.0" width="130" height="44" rx="7" fill="#f43f5e" fill-opacity="0.14" stroke="#f43f5e" stroke-width="1.4"/>
  <text x="75" y="208" text-anchor="middle" fill="currentColor" font-weight="600">leaf: spam</text>
  <text x="75" y="224" text-anchor="middle" fill="currentColor">40 spam · 2 not</text>
  <rect x="175.0" y="190.0" width="140" height="44" rx="7" fill="#6366f1" fill-opacity="0.14" stroke="#6366f1" stroke-width="1.4"/>
  <text x="245" y="208" text-anchor="middle" fill="currentColor" font-weight="600">leaf: not spam</text>
  <text x="245" y="224" text-anchor="middle" fill="currentColor">5 spam · 15 not</text>
  <text x="330" y="205" fill="#b45309">orange path: an email with</text>
  <text x="330" y="220" fill="#b45309">“free” and 5 links → spam</text>
</svg>
</div>

*Each leaf shows how many training emails ended there. The leaf predicts the majority label.*

**Training a tree means choosing the questions.** A good question splits a
mixed group into groups that are each mostly one label. "Contains 'free'?"
is useful if spam mostly says it and normal email mostly doesn't. To compare
questions, the tree needs a number for how mixed a group is. The usual one
is **entropy** (Probability §3.5.1):

$$H = -\sum_c p_c \log_2 p_c$$

where $p_c$ is the fraction of the group in class $c$. Read it as how
uncertain you are about the label of a random email from the group,
measured in bits:

| group | fractions | entropy $H$ | meaning |
|---|---|---|---|
| 5 spam, 5 not | 0.5, 0.5 | 1 bit | a coin flip, as mixed as possible |
| 9 spam, 1 not | 0.9, 0.1 | 0.47 bits | mostly one label |
| 10 spam, 0 not | 1, 0 | 0 | pure, no uncertainty |

<div style="margin:1.25rem 0">
<svg viewBox="0 0 420 230" role="img" aria-label="Entropy and Gini impurity of a two-class group, plotted against the fraction p in one class. Entropy is 0 at p = 0, rises to 1 bit at p = 0.5, and falls back to 0 at p = 1; it is 0.47 at p = 0.9. Gini impurity has the same shape with a peak of 0.5." style="max-width:440px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <line x1="50" y1="190" x2="350" y2="190" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="50" y1="190" x2="50" y2="28.0" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="50" y="205" text-anchor="middle" fill="currentColor">0</text>
  <text x="200.0" y="205" text-anchor="middle" fill="currentColor">0.5</text>
  <text x="350" y="205" text-anchor="middle" fill="currentColor">1</text>
  <text x="44" y="119.0" text-anchor="end" fill="currentColor">0.5</text>
  <text x="44" y="44" text-anchor="end" fill="currentColor">1</text>
  <text x="350" y="220" text-anchor="end" fill="currentColor">p = fraction of the group that is spam</text>
  <polyline points="50.0,190.0 53.0,177.9 56.0,168.8 59.0,160.8 62.0,153.7 65.0,147.0 68.0,140.9 71.0,135.1 74.0,129.7 77.0,124.5 80.0,119.7 83.0,115.0 86.0,110.6 89.0,106.4 92.0,102.4 95.0,98.5 98.0,94.9 101.0,91.3 104.0,88.0 107.0,84.8 110.0,81.7 113.0,78.8 116.0,76.0 119.0,73.3 122.0,70.7 125.0,68.3 128.0,66.0 131.0,63.8 134.0,61.7 137.0,59.7 140.0,57.8 143.0,56.0 146.0,54.3 149.0,52.8 152.0,51.3 155.0,49.9 158.0,48.6 161.0,47.4 164.0,46.3 167.0,45.3 170.0,44.4 173.0,43.5 176.0,42.8 179.0,42.1 182.0,41.6 185.0,41.1 188.0,40.7 191.0,40.4 194.0,40.2 197.0,40.0 200.0,40.0 203.0,40.0 206.0,40.2 209.0,40.4 212.0,40.7 215.0,41.1 218.0,41.6 221.0,42.1 224.0,42.8 227.0,43.5 230.0,44.4 233.0,45.3 236.0,46.3 239.0,47.4 242.0,48.6 245.0,49.9 248.0,51.3 251.0,52.8 254.0,54.3 257.0,56.0 260.0,57.8 263.0,59.7 266.0,61.7 269.0,63.8 272.0,66.0 275.0,68.3 278.0,70.7 281.0,73.3 284.0,76.0 287.0,78.8 290.0,81.7 293.0,84.8 296.0,88.0 299.0,91.3 302.0,94.9 305.0,98.5 308.0,102.4 311.0,106.4 314.0,110.6 317.0,115.0 320.0,119.7 323.0,124.5 326.0,129.7 329.0,135.1 332.0,140.9 335.0,147.0 338.0,153.7 341.0,160.8 344.0,168.8 347.0,177.9 350.0,190.0" fill="none" stroke="#6366f1" stroke-width="2.2"/>
  <polyline points="50.0,190.0 53.0,187.0 56.0,184.1 59.0,181.3 62.0,178.5 65.0,175.8 68.0,173.1 71.0,170.5 74.0,167.9 77.0,165.4 80.0,163.0 83.0,160.6 86.0,158.3 89.0,156.1 92.0,153.9 95.0,151.8 98.0,149.7 101.0,147.7 104.0,145.7 107.0,143.8 110.0,142.0 113.0,140.2 116.0,138.5 119.0,136.9 122.0,135.3 125.0,133.8 128.0,132.3 131.0,130.9 134.0,129.5 137.0,128.2 140.0,127.0 143.0,125.8 146.0,124.7 149.0,123.7 152.0,122.7 155.0,121.8 158.0,120.9 161.0,120.1 164.0,119.3 167.0,118.6 170.0,118.0 173.0,117.4 176.0,116.9 179.0,116.5 182.0,116.1 185.0,115.7 188.0,115.5 191.0,115.3 194.0,115.1 197.0,115.0 200.0,115.0 203.0,115.0 206.0,115.1 209.0,115.3 212.0,115.5 215.0,115.8 218.0,116.1 221.0,116.5 224.0,116.9 227.0,117.4 230.0,118.0 233.0,118.6 236.0,119.3 239.0,120.1 242.0,120.9 245.0,121.8 248.0,122.7 251.0,123.7 254.0,124.7 257.0,125.8 260.0,127.0 263.0,128.2 266.0,129.5 269.0,130.9 272.0,132.3 275.0,133.8 278.0,135.3 281.0,136.9 284.0,138.5 287.0,140.2 290.0,142.0 293.0,143.8 296.0,145.7 299.0,147.7 302.0,149.7 305.0,151.8 308.0,153.9 311.0,156.1 314.0,158.3 317.0,160.6 320.0,163.0 323.0,165.4 326.0,167.9 329.0,170.5 332.0,173.1 335.0,175.8 338.0,178.5 341.0,181.3 344.0,184.1 347.0,187.0 350.0,190.0" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="6 3"/>
  <circle cx="200.0" cy="40.0" r="4" fill="#6366f1"/>
  <text x="200.0" y="31.0" text-anchor="middle" fill="currentColor">5/5: 1 bit</text>
  <circle cx="320.0" cy="119.7" r="4" fill="#6366f1"/>
  <text x="328.0" y="110.7" text-anchor="start" fill="currentColor">9/1: 0.47</text>
  <circle cx="350.0" cy="190.0" r="4" fill="#6366f1"/>
  <text x="358.0" y="194.0" text-anchor="start" fill="currentColor">pure: 0</text>
  <line x1="248.0" y1="16.0" x2="272.0" y2="16.0" stroke="#6366f1" stroke-width="2.2"/><text x="278.0" y="20.0" fill="currentColor">entropy</text>
  <line x1="326.0" y1="16.0" x2="350.0" y2="16.0" stroke="#10b981" stroke-width="2" stroke-dasharray="6 3"/><text x="356.0" y="20.0" fill="currentColor">Gini</text>
</svg>
</div>

*Both measures are 0 for a pure group and largest for a 50/50 group.*

**Gini impurity** is the common alternative. Draw two emails at random
from the group (with replacement). Gini is the chance that they have
**different labels**:

- both are spam with chance $p_{\text{spam}}^2$, both are not spam with
  chance $p_{\text{not}}^2$;
- so they differ with chance

$$G = 1 - \sum_c p_c^2$$

For the 9 spam, 1 not group: $G = 1 - (0.9^2 + 0.1^2) = 1 - 0.82 = 0.18$.
Checking directly: spam then not is $0.9 \times 0.1$, not then spam is
$0.1 \times 0.9$, total 0.18.

| group | entropy | Gini |
|---|---|---|
| 5 spam, 5 not | 1 | 0.5 |
| 9 spam, 1 not | 0.47 | 0.18 |
| 10 spam, 0 not | 0 | 0 |

The two measures have the same shape (figure above), so they almost always
pick the same split. Many libraries, including scikit-learn, use Gini by
default because it needs only squares, no logarithm.

### 7.2 Choosing a split: information gain

A split is good if the groups it creates are less mixed than the group it
started from. The **information gain** measures that: the parent's entropy
minus the children's entropy, each child weighted by its share of the
examples:

$$IG = H(\text{parent}) - \tfrac{n_L}{n}\,H(\text{left}) - \tfrac{n_R}{n}\,H(\text{right})$$

**Example.** 10 emails, 5 spam and 5 not, so $H(\text{parent}) = 1$. Two
candidate questions:

| question | yes group | no group | $H$ of each group | weighted $H$ | gain |
|---|---|---|---|---|---|
| contains "free"? | 4 spam, 1 not | 1 spam, 4 not | 0.722 | 0.722 | $1 - 0.722 = 0.278$ |
| sent on a weekday? | 3 spam, 2 not | 2 spam, 3 not | 0.971 | 0.971 | $1 - 0.971 = 0.029$ |

"Contains 'free'" leaves much purer groups, so it wins. A question that
leaves both groups 50/50 has gain 0.

### 7.3 Building the tree

Start with all training examples at the root and repeat:

1. **Try every question.** For each feature, list every possible threshold.
   For a numeric feature, the candidates are the midpoints between its
   sorted values: if the emails have 0, 2, 5 and 9 links, try "links > 1",
   "links > 3.5" and "links > 7".
2. **Keep the question with the highest gain** and split the examples into
   its two groups.
3. **Repeat on each group.**
4. **Stop** when a group is pure, has too few examples, or the tree reaches
   its maximum depth. That group becomes a leaf.

**Greedy: best split now, no looking ahead.** Searching all possible trees
instead is hopeless, since their number grows exponentially (finding the
best tree is NP-hard). Greedy is fast and usually works well, but it can
miss a split that only pays off one level later.

**Where greedy struggles: XOR.** Label a point blue when $x_1$ and $x_2$
have the same sign, red otherwise.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 470 250" role="img" aria-label="XOR data: blue points in the top-right and bottom-left quadrants, red points in the top-left and bottom-right quadrants. A dashed vertical split at x1 = 0 leaves each side half blue and half red, so its gain is 0. Splitting on x1 and then on x2 separates the four quadrants perfectly." style="max-width:500px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <rect x="25.0" y="15.0" width="210" height="210" fill="none" stroke="currentColor" stroke-opacity="0.25"/>
  <line x1="130" y1="8.0" x2="130" y2="232.0" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 4"/>
  <line x1="25.0" y1="120" x2="235.0" y2="120" stroke="currentColor" stroke-opacity="0.35"/>
  <text x="239.0" y="124" fill="currentColor">x₁</text>
  <text x="136" y="27.0" fill="currentColor">x₂</text>
  <circle cx="171" cy="91" r="5" fill="#6366f1"/>
  <circle cx="195" cy="97" r="5" fill="#6366f1"/>
  <circle cx="187" cy="76" r="5" fill="#6366f1"/>
  <circle cx="152" cy="65" r="5" fill="#6366f1"/>
  <circle cx="150" cy="71" r="5" fill="#6366f1"/>
  <circle cx="153" cy="96" r="5" fill="#6366f1"/>
  <rect x="76" y="37" width="10" height="10" fill="#f43f5e"/>
  <rect x="98" y="81" width="10" height="10" fill="#f43f5e"/>
  <rect x="61" y="28" width="10" height="10" fill="#f43f5e"/>
  <rect x="65" y="68" width="10" height="10" fill="#f43f5e"/>
  <rect x="36" y="94" width="10" height="10" fill="#f43f5e"/>
  <rect x="44" y="76" width="10" height="10" fill="#f43f5e"/>
  <circle cx="102" cy="146" r="5" fill="#6366f1"/>
  <circle cx="90" cy="197" r="5" fill="#6366f1"/>
  <circle cx="99" cy="180" r="5" fill="#6366f1"/>
  <circle cx="66" cy="165" r="5" fill="#6366f1"/>
  <circle cx="72" cy="142" r="5" fill="#6366f1"/>
  <circle cx="108" cy="153" r="5" fill="#6366f1"/>
  <rect x="193" y="164" width="10" height="10" fill="#f43f5e"/>
  <rect x="166" y="176" width="10" height="10" fill="#f43f5e"/>
  <rect x="176" y="155" width="10" height="10" fill="#f43f5e"/>
  <rect x="201" y="184" width="10" height="10" fill="#f43f5e"/>
  <rect x="160" y="175" width="10" height="10" fill="#f43f5e"/>
  <rect x="181" y="197" width="10" height="10" fill="#f43f5e"/>
  <text x="275.0" y="60" fill="#b45309">first split x₁ > 0 (dashed):</text>
  <text x="275.0" y="78" fill="currentColor">left side: 6 blue, 6 red</text>
  <text x="275.0" y="96" fill="currentColor">right side: 6 blue, 6 red</text>
  <text x="275.0" y="114" fill="currentColor">gain 0, same for x₂</text>
  <text x="275.0" y="132" fill="currentColor"></text>
  <text x="275.0" y="150" fill="#10b981">but x₁ then x₂ together</text>
  <text x="275.0" y="168" fill="#10b981">separate all four corners</text>
</svg>
</div>

*Any single first split leaves both sides half blue, half red, so its gain is 0.*

Splitting on $x_1 > 0$ leaves each side half blue, half red: gain 0. The
same holds for $x_2$. So greedy sees no useful first split, although $x_1$
followed by $x_2$ separates the data perfectly. In practice, noise usually
gives one of these splits a small gain, and the tree then finds the second
one; the notebook's tree reaches 95% test accuracy on noisy XOR data at
depth 4 (§7.4).

### 7.4 Why trees overfit

A deep enough tree can give every training example its own leaf and reach
100% training accuracy on any data, noise included. In
`trees_and_ensembles_scratch.ipynb` (XOR data with 5% flipped labels):

| max depth | train accuracy | test accuracy |
|---|---|---|
| 2 | 64% | 61% |
| 4 | 95% | 95% |
| 8 | 98% | 92% |
| 20 | 100% | 91% |

Depth 2 is too simple (high bias). Past depth 4, training accuracy keeps
rising while test accuracy falls: the extra splits fit the noise. Two
reasons:

- **Deep splits are chosen from a handful of examples.** A leaf with 3
  examples predicts from 3 noisy labels.
- **Trees are unstable.** Changing one training point can change a split
  near the top, which changes every split below it. So the tree has high
  variance.

Limits on growth help: `max_depth`, `min_samples_leaf`, or pruning (grow
the full tree, then cut back branches that don't improve validation
accuracy). The stronger fix is to average many trees (§8).

### 7.5 Regression trees

For a numeric target, such as a house price, the tree works the same way
with two changes:

- **A leaf predicts the average** of its training examples' values. (The
  average is the single number with the smallest squared error to a group.)
- **Splits are scored by how much they reduce the squared error**, the sum
  of squared differences between each value and its group's average,
  instead of by entropy.

**Example.** Four houses (prices in thousands):

| size (m²) | 80 | 100 | 180 | 200 |
|---|---|---|---|---|
| price | 200 | 220 | 400 | 420 |

Before any split, the average is 310 and the squared error is
$110^2 + 90^2 + 90^2 + 110^2 = 40{,}400$. Compare candidate splits:

| split | groups | averages | squared error after |
|---|---|---|---|
| size > 140 | {200, 220} and {400, 420} | 210 and 410 | $100 + 100 + 100 + 100 = 400$ |
| size > 90 | {200} and {220, 400, 420} | 200 and 347 | 24,267 |

"Size > 140" cuts the error from 40,400 to 400, so it wins: each side is
tightly bunched around its own average. A 190 m² house lands on the right
and is predicted at 410.

**A regression tree predicts a step function.** Every house in a leaf gets
the same price, so the prediction jumps between leaves instead of changing
smoothly. It also **cannot extrapolate**: a 400 m² house still lands in the
"size > 140" leaf and is predicted at 410, the average of the training
houses there. A linear model would predict a higher price.

---

## 8. Ensembles

### 8.1 Bagging: averaging removes variance

**The problem: one deep tree is unstable.** A model's **variance** is how
much its predictions change when it is trained on a different sample of
data (Probability §3.7). A deep tree has high variance (§7.4): change a few
training emails and the top split can change, and with it every prediction
below.

**The fix: average many models.** Averaging cancels errors that point in
different directions. In 1906, Francis Galton collected 787 guesses of an
ox's weight at a country fair. Individual guesses were far off, but their
median was 1,207 lb against a true 1,198 lb, within 1%. That works only
because the errors were **independent**: some guessed high, some low. If
all 787 people had copied one person, the average would be exactly as wrong
as that person.

The same holds for models. Average $B$ models, each with variance
$\sigma^2$, whose errors have correlation $\rho$ (how much they move
together):

$$\operatorname{Var}(\text{average}) = \rho\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

The second term shrinks to 0 as you add models. The first, $\rho\sigma^2$,
never shrinks: it is a floor. With $\sigma^2 = 1$ and $B = 100$ models:

| correlation $\rho$ | 0 | 0.1 | 0.5 | 1 |
|---|---|---|---|---|
| variance of the average | 0.01 | 0.109 | 0.505 | 1 |

Independent models cut the variance 100-fold; perfectly correlated ones
don't help at all. So **the limit is how correlated the models are**, not
how many there are.

**Bagging** (bootstrap aggregating) makes many different models from one
dataset:

1. **Resample.** Draw $n$ examples from the $n$ training examples **with
   replacement**, a **bootstrap sample**. Some examples appear twice, some
   not at all.
2. **Train** one deep tree on each sample.
3. **Combine:** average their predictions (regression) or take a majority
   vote (classification).

<div style="margin:1.25rem 0">
<svg viewBox="0 0 560 250" role="img" aria-label="Bagging. One training set of five examples, A to E, is resampled with replacement into three bootstrap samples: A A C D E, B C C D E, and A B B D E. One deep tree is trained on each sample. A new example is sent to all three trees, which vote spam, spam and not spam, so the bagged prediction is spam." style="max-width:580px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <defs><marker id="bg-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity="0.55"/></marker></defs>
  <rect x="10" y="100" width="100" height="50" rx="7" fill="#64748b" fill-opacity="0.13" stroke="#64748b" stroke-width="1.3"/>
  <text x="60.0" y="122" text-anchor="middle" fill="currentColor" font-weight="600">training set</text>
  <text x="60.0" y="136" text-anchor="middle" fill="currentColor">A B C D E</text>
  <line x1="112" y1="125" x2="148" y2="45" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="150" y="20" width="120" height="50" rx="7" fill="#f59e0b" fill-opacity="0.13" stroke="#f59e0b" stroke-width="1.3"/>
  <text x="210.0" y="42" text-anchor="middle" fill="currentColor" font-weight="600">sample 1</text>
  <text x="210.0" y="56" text-anchor="middle" fill="currentColor">A A C D E</text>
  <text x="210" y="82" text-anchor="middle" fill="currentColor" fill-opacity="0.7">left out: B</text>
  <line x1="272" y1="45" x2="308" y2="45" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="310" y="20" width="90" height="50" rx="7" fill="#10b981" fill-opacity="0.13" stroke="#10b981" stroke-width="1.3"/>
  <text x="355.0" y="42" text-anchor="middle" fill="currentColor" font-weight="600">tree 1</text>
  <text x="355.0" y="56" text-anchor="middle" fill="currentColor">votes spam</text>
  <line x1="402" y1="45" x2="438" y2="125" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <line x1="112" y1="125" x2="148" y2="125" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="150" y="100" width="120" height="50" rx="7" fill="#f59e0b" fill-opacity="0.13" stroke="#f59e0b" stroke-width="1.3"/>
  <text x="210.0" y="122" text-anchor="middle" fill="currentColor" font-weight="600">sample 2</text>
  <text x="210.0" y="136" text-anchor="middle" fill="currentColor">B C C D E</text>
  <text x="210" y="162" text-anchor="middle" fill="currentColor" fill-opacity="0.7">left out: A</text>
  <line x1="272" y1="125" x2="308" y2="125" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="310" y="100" width="90" height="50" rx="7" fill="#10b981" fill-opacity="0.13" stroke="#10b981" stroke-width="1.3"/>
  <text x="355.0" y="122" text-anchor="middle" fill="currentColor" font-weight="600">tree 2</text>
  <text x="355.0" y="136" text-anchor="middle" fill="currentColor">votes spam</text>
  <line x1="402" y1="125" x2="438" y2="125" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <line x1="112" y1="125" x2="148" y2="205" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="150" y="180" width="120" height="50" rx="7" fill="#f59e0b" fill-opacity="0.13" stroke="#f59e0b" stroke-width="1.3"/>
  <text x="210.0" y="202" text-anchor="middle" fill="currentColor" font-weight="600">sample 3</text>
  <text x="210.0" y="216" text-anchor="middle" fill="currentColor">A B B D E</text>
  <text x="210" y="242" text-anchor="middle" fill="currentColor" fill-opacity="0.7">left out: C</text>
  <line x1="272" y1="205" x2="308" y2="205" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="310" y="180" width="90" height="50" rx="7" fill="#10b981" fill-opacity="0.13" stroke="#10b981" stroke-width="1.3"/>
  <text x="355.0" y="202" text-anchor="middle" fill="currentColor" font-weight="600">tree 3</text>
  <text x="355.0" y="216" text-anchor="middle" fill="currentColor">votes not spam</text>
  <line x1="402" y1="205" x2="438" y2="125" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.3" marker-end="url(#bg-ar)"/>
  <rect x="440" y="100" width="110" height="50" rx="7" fill="#f43f5e" fill-opacity="0.13" stroke="#f43f5e" stroke-width="1.3"/>
  <text x="495.0" y="122" text-anchor="middle" fill="currentColor" font-weight="600">majority vote</text>
  <text x="495.0" y="136" text-anchor="middle" fill="currentColor">2 of 3: spam</text>
  <text x="210" y="12" text-anchor="middle" fill="currentColor" fill-opacity="0.75">draw 5 with replacement</text>
</svg>
</div>

*Each tree sees a slightly different dataset, so the trees make different mistakes, and the vote cancels some of them.*

**Out-of-bag error: free validation.** A given example is missed by one
draw with chance $1 - 1/n$, so by all $n$ draws with chance
$(1 - 1/n)^n \approx 1/e \approx 0.37$ (0.366 for $n = 100$). So each tree
never sees about 37% of the data. Score each example using only the trees
that didn't train on it, and you get the **out-of-bag error**, a validation
score without setting aside a holdout set.

**When bagging helps.** Averaging lowers variance, not bias: the average of
many trees that are all too simple is still too simple. So bagging helps
models with low bias and high variance, such as deep trees, and does almost
nothing for stable models such as linear regression, whose bootstrap
versions are nearly identical.

### 8.2 Random forests: lowering the correlation

Bagged trees are still similar: if one feature is strongest, every tree
splits on it first, so $\rho$ stays high. A **random forest** considers only
a random subset of about $\sqrt d$ features at each split. The strong feature
is often unavailable, trees split on different features, and $\rho$ drops.
Each tree gets slightly worse ($\sigma^2$ rises), but the floor $\rho\sigma^2$
falls. In `trees_and_ensembles_scratch.ipynb`, going from all 6 features
(plain bagging) to 2 per split cuts $\rho$ from 0.167 to 0.055, raises
$\sigma^2$ from 0.061 to 0.101, and lowers the floor from 0.0102 to 0.0056.

Too few features per split hurts: with only 1, many splits have no useful
feature to choose from and accuracy is worst. The subset size is a
hyperparameter to tune.

### 8.3 Boosting: gradient descent with trees

**The idea: fix the remaining mistakes, one small tree at a time.** Bagging
trains deep trees independently and averages them. Boosting trains **small**
trees **one after another**. Each new tree looks only at what the model so
far still gets wrong, and corrects part of it. A single small tree is too
simple to be accurate (high bias), but the sum of many is a strong model.
So boosting reduces bias, where bagging reduces variance.

**Example.** Three houses: 80, 120 and 200 m², sold for 200, 300 and 400
(thousands). Each tree is a **stump**, a tree with one split, so it can give
only two answers. Each round adds half of the new tree ($\eta = 0.5$, the
step size).

| round | what the new tree learns from the residuals | predictions | residuals (true − predicted) | squared error |
|---|---|---|---|---|
| 0 | start at the average | 300, 300, 300 | $-100$, 0, $+100$ | 20,000 |
| 1 | size < 100: $-100$; otherwise $+50$ | 250, 325, 325 | $-50$, $-25$, $+75$ | 8,750 |
| 2 | size < 160: $-37.5$; otherwise $+75$ | 231, 306, 363 | $-31$, $-6$, $+38$ | 2,422 |
| 3 | size < 160: $-18.75$; otherwise $+37.5$ | 222, 297, 381 | $-22$, $+3$, $+19$ | 840 |

Each tree is fitted to the current **residuals**, the gaps between the true
prices and the predictions. Adding half of it closes part of every gap.
After 6 rounds the predictions are 204, 301 and 395, and the squared error
is 38.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 250" role="img" aria-label="Predictions for three houses over six boosting rounds. All three start at 300, the average. Each round adds half of a one-split tree fitted to the remaining errors, and the predictions move toward the true prices 200, 300 and 400, reaching 204, 301 and 395 after six rounds." style="max-width:540px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <line x1="60" y1="210.0" x2="390" y2="210.0" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="60" y1="210.0" x2="60" y2="30.0" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="60" y="225.0" text-anchor="middle" fill="currentColor">0</text>
  <text x="115" y="225.0" text-anchor="middle" fill="currentColor">1</text>
  <text x="170" y="225.0" text-anchor="middle" fill="currentColor">2</text>
  <text x="225" y="225.0" text-anchor="middle" fill="currentColor">3</text>
  <text x="280" y="225.0" text-anchor="middle" fill="currentColor">4</text>
  <text x="335" y="225.0" text-anchor="middle" fill="currentColor">5</text>
  <text x="390" y="225.0" text-anchor="middle" fill="currentColor">6</text>
  <text x="390" y="240.0" text-anchor="end" fill="currentColor">boosting round (number of trees added)</text>
  <text x="54" y="199" text-anchor="end" fill="currentColor">200</text>
  <text x="54" y="124" text-anchor="end" fill="currentColor">300</text>
  <text x="54" y="49" text-anchor="end" fill="currentColor">400</text>
  <line x1="60" y1="195" x2="390" y2="195" stroke="#6366f1" stroke-opacity="0.45" stroke-dasharray="4 4"/>
  <polyline points="60,120.0 115,157.5 170,171.6 225,178.6 280,186.8 335,189.3 390,192.1" fill="none" stroke="#6366f1" stroke-width="2"/>
  <circle cx="60" cy="120.0" r="3.2" fill="#6366f1"/>
  <circle cx="115" cy="157.5" r="3.2" fill="#6366f1"/>
  <circle cx="170" cy="171.6" r="3.2" fill="#6366f1"/>
  <circle cx="225" cy="178.6" r="3.2" fill="#6366f1"/>
  <circle cx="280" cy="186.8" r="3.2" fill="#6366f1"/>
  <circle cx="335" cy="189.3" r="3.2" fill="#6366f1"/>
  <circle cx="390" cy="192.1" r="3.2" fill="#6366f1"/>
  <text x="400" y="199" fill="#6366f1">80 m², sold 200</text>
  <line x1="60" y1="120" x2="390" y2="120" stroke="#10b981" stroke-opacity="0.45" stroke-dasharray="4 4"/>
  <polyline points="60,120.0 115,101.2 170,115.3 225,122.3 280,118.2 335,120.7 390,119.3" fill="none" stroke="#10b981" stroke-width="2"/>
  <circle cx="60" cy="120.0" r="3.2" fill="#10b981"/>
  <circle cx="115" cy="101.2" r="3.2" fill="#10b981"/>
  <circle cx="170" cy="115.3" r="3.2" fill="#10b981"/>
  <circle cx="225" cy="122.3" r="3.2" fill="#10b981"/>
  <circle cx="280" cy="118.2" r="3.2" fill="#10b981"/>
  <circle cx="335" cy="120.7" r="3.2" fill="#10b981"/>
  <circle cx="390" cy="119.3" r="3.2" fill="#10b981"/>
  <text x="400" y="124" fill="#10b981">120 m², sold 300</text>
  <line x1="60" y1="45" x2="390" y2="45" stroke="#f43f5e" stroke-opacity="0.45" stroke-dasharray="4 4"/>
  <polyline points="60,120.0 115,101.2 170,73.1 225,59.1 280,55.0 335,50.0 390,48.6" fill="none" stroke="#f43f5e" stroke-width="2"/>
  <circle cx="60" cy="120.0" r="3.2" fill="#f43f5e"/>
  <circle cx="115" cy="101.2" r="3.2" fill="#f43f5e"/>
  <circle cx="170" cy="73.1" r="3.2" fill="#f43f5e"/>
  <circle cx="225" cy="59.1" r="3.2" fill="#f43f5e"/>
  <circle cx="280" cy="55.0" r="3.2" fill="#f43f5e"/>
  <circle cx="335" cy="50.0" r="3.2" fill="#f43f5e"/>
  <circle cx="390" cy="48.6" r="3.2" fill="#f43f5e"/>
  <text x="400" y="49" fill="#f43f5e">200 m², sold 400</text>
  <text x="66" y="34.0" fill="currentColor" fill-opacity="0.75">prediction (dashed: true price)</text>
</svg>
</div>

*Each round adds half of a one-split tree fitted to the remaining errors, and every prediction moves toward its true price.*

**Why it is called gradient boosting.** For the squared loss
$\tfrac12(y - F)^2$, where $F$ is the current prediction, the derivative
with respect to $F$ is $-(y - F)$. So the residual $y - F$ is exactly the
**negative gradient** of the loss. "Fit a tree to the residuals and add a
small step of it" is gradient descent, where each step is a tree instead of
a change to a weight vector.

For other losses, fit each tree to the negative gradient, called the
**pseudo-residual**:

$$r_i = -\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}, \qquad F \leftarrow F + \eta\, h \quad (h = \text{a tree fitted to the } r_i)$$

For classification with log-loss, $r_i = y_i - p_i$, the same $y - p$ as
logistic regression ([§2.2](01_linear_models.md#2.2)). For a spam email
($y = 1$) that the model currently gives $p = 0.2$, the next tree is pushed
toward $+0.8$ for it.

**Step size and when to stop.**

- A smaller $\eta$ (typically 0.01 to 0.3) takes smaller steps, so it needs
  more trees, but usually gives lower test error.
- Each tree adds capacity, so too many rounds fit the noise. Stop when the
  error on a validation set stops improving (**early stopping**).

| | bagging / random forest | boosting |
|---|---|---|
| trees | deep, trained independently (in parallel) | small, trained one after another |
| each tree fits | a bootstrap sample of the data | the current residuals |
| reduces | variance | bias |
| more trees | never overfits, only averages more | eventually overfits; use early stopping |

### 8.4 XGBoost

XGBoost is gradient boosting with three additions:

- **Second-order steps.** Plain gradient boosting uses only the slope of
  the loss. XGBoost also uses its curvature, like Newton's method. With
  $g_i$ and $h_i$ the first and second derivatives of the loss for example
  $i$, the best value for a leaf is
  $w^* = -\sum g_i / (\sum h_i + \lambda)$, summing over the examples in the
  leaf. For squared loss every $h_i = 1$, so this is the leaf's average
  residual, shrunk toward 0 by $\lambda$. The same sums give each candidate
  split an exact score, so splits are chosen on the real objective.
- **Regularization:** a cost $\gamma$ per leaf and an L2 penalty $\lambda$ on
  leaf values, plus shrinkage and row and column subsampling.
- **Speed:** histogram-based split finding, a learned default direction for
  missing values, and parallel split search. (LightGBM and CatBoost are
  similar libraries.)

Gradient-boosted trees are still the best default for tabular data: they
handle mixed feature types, different scales, missing values and sharp
interactions without feature engineering.
