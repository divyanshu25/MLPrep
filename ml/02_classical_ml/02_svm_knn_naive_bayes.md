# Other classifiers: SVMs, kNN, naive Bayes

Notation: $n$ examples, $d$ features, $X \in \mathbb{R}^{n \times d}$ with one
example per row, labels $y \in \mathbb{R}^n$, weights $w \in \mathbb{R}^d$.

---

## 4. Support vector machines

### 4.1 The widest street

Two classes of points can often be separated by many different lines. Which
one should you pick? A line that passes very close to some training points
is risky: a new point of the same class that lands slightly to the other
side gets misclassified. A safer line keeps as far as possible from both
classes.

A support vector machine (SVM) picks the line with the widest empty
**street** around it, with no training points inside the street. The edges
of the street are called the **gutters**, and the width of the street is the
**margin**.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 380 300" role="img" aria-label="Two classes of points separated by a line. Two parallel dashed lines, the gutters, mark the edges of an empty street around it. Four points lie exactly on the gutters and are ringed: these are the support vectors. An arrow across the street marks its width, 2 divided by the length of w." style="max-width:420px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <polygon points="20,68 20,20 68,20 260,212 260,260 212,260" fill="#f59e0b" fill-opacity="0.12"/>
  <rect x="20" y="20" width="240" height="240" fill="none" stroke="currentColor" stroke-opacity="0.25"/>
  <line x1="20" y1="20" x2="260" y2="260" stroke="currentColor" stroke-width="1.8"/>
  <line x1="68" y1="20" x2="260" y2="212" stroke="currentColor" stroke-opacity="0.6" stroke-dasharray="5 4"/>
  <line x1="20" y1="68" x2="212" y2="260" stroke="currentColor" stroke-opacity="0.6" stroke-dasharray="5 4"/>
  <defs><marker id="sv-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b"/></marker></defs>
  <line x1="142" y1="186" x2="186" y2="142" stroke="#f59e0b" stroke-width="1.8" marker-start="url(#sv-ar)" marker-end="url(#sv-ar)"/>
  <text x="272" y="110" fill="#b45309">orange arrow:</text><text x="272" y="124" fill="#b45309">street width 2/‖w‖</text>
  <circle cx="212" cy="92" r="5" fill="#6366f1"/>
  <circle cx="176" cy="56" r="5" fill="#6366f1"/>
  <circle cx="236" cy="128" r="5" fill="#6366f1"/>
  <circle cx="224" cy="49" r="5" fill="#6366f1"/>
  <circle cx="241" cy="80" r="5" fill="#6366f1"/>
  <circle cx="140" cy="92" r="5" fill="#6366f1"/>
  <circle cx="200" cy="152" r="5" fill="#6366f1"/>
  <rect x="63" y="183" width="10" height="10" fill="#f43f5e"/>
  <rect x="51" y="123" width="10" height="10" fill="#f43f5e"/>
  <rect x="99" y="219" width="10" height="10" fill="#f43f5e"/>
  <rect x="111" y="178" width="10" height="10" fill="#f43f5e"/>
  <rect x="39" y="219" width="10" height="10" fill="#f43f5e"/>
  <rect x="87" y="135" width="10" height="10" fill="#f43f5e"/>
  <rect x="147" y="195" width="10" height="10" fill="#f43f5e"/>
  <circle cx="140" cy="92" r="10" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <circle cx="200" cy="152" r="10" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <circle cx="92" cy="140" r="10" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <circle cx="152" cy="200" r="10" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <g fill="currentColor"><text x="266" y="216">f(x) = +1</text><text x="266" y="264">f(x) = 0</text><text x="212" y="278" text-anchor="middle">f(x) = −1</text></g>
  <text x="250" y="34" fill="#6366f1" text-anchor="end" font-size="12">class +1</text>
  <text x="30" y="250" fill="#f43f5e" font-size="12">class −1</text>
  <text x="272" y="60" fill="currentColor">ringed points:</text><text x="272" y="74" fill="currentColor">support vectors</text>
</svg>
</div>

*Blue circles are class $+1$, red squares class $-1$. The solid line is the boundary $f(x) = 0$; the dashed lines are the gutters.*

**Turning the street into a formula.** Label the classes $y = +1$ and
$y = -1$. The model computes a score $f(x) = w^\top x + b$ and predicts the
class from its sign. So the line is where $f(x) = 0$. Put the gutters where
$f(x) = +1$ and $f(x) = -1$.

Rule 1: every training point stays out of the street, on its own side.
A $+1$ point needs $f(x_i) \ge 1$, a $-1$ point needs $f(x_i) \le -1$.
Multiply by the label and both become one line: $y_i f(x_i) \ge 1$.

Rule 2: make the street as wide as possible. So how wide is it? Let's just
compute it. Take $w = (3, 4)$ and $b = 0$, so $f(x) = 3x_1 + 4x_2$. Start
on the line at $x = (0, 0)$, where $f = 0$, and walk straight across it,
in the direction of $w$. Scaled to length 1, that direction is
$(0.6,\ 0.8)$, so after walking a distance $t$ you are at $(0.6t,\ 0.8t)$
and the score is

$$f = 3(0.6t) + 4(0.8t) = 5t$$

The score rises by 5 per unit walked, and 5 is exactly $\|w\|$. It reaches
$+1$, the gutter, at $t = 1/5$. The other gutter is $1/5$ away on the other
side. So the street is $2/5$ wide, and in general $2/\|w\|$.

That is all $\|w\|$ means here: how fast the score climbs as you cross the
line. A steep score hits $\pm 1$ quickly, which makes a narrow street. A
gentle score makes a wide one. So the SVM makes $\|w\|$ as small as Rule 1
allows:

$$\min_{w, b}\ \tfrac12\|w\|^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \ge 1 \text{ for all } i$$

(Minimizing $\tfrac12\|w\|^2$ gives the same $w$ as minimizing $\|w\|$; the
square just makes the derivative simpler.)

**Support vectors.** The points lying exactly on the gutters (ringed in the
figure) are the **support vectors**. They alone decide the street. Move or
delete any other point, without letting it into the street, and the answer
stays the same. Logistic regression is different: every training point
pulls on its weights a little.

### 4.2 Soft margin: when the classes overlap

Real data usually has no empty street: a few points sit inside it or on the
wrong side. So the SVM allows that, but charges each such point by how far
it is past its own gutter. The charge is the **hinge loss**:

$$\text{hinge}(x_i) = \max\big(0,\ 1 - y_i f(x_i)\big)$$

The quantity $y_i f(x_i)$ measures how far a point is on its correct side:
1 means on its gutter, 0 means on the dividing line, negative means on the
wrong side.

| $y\,f(x)$ | 2 | 1 | 0.5 | 0 | $-1$ |
|---|---|---|---|---|---|
| where the point is | beyond its gutter | on its gutter | inside the street | on the line | wrong side |
| hinge loss | 0 | 0 | 0.5 | 1 | 2 |
| logistic loss $\log(1 + e^{-yf})$ | 0.13 | 0.31 | 0.47 | 0.69 | 1.31 |

Points beyond their gutter cost exactly 0 and have no effect on the answer.
Logistic loss is never exactly 0, so every point still counts a little.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 265" role="img" aria-label="Hinge loss and logistic loss plotted against y times f of x. The hinge loss is a straight line falling to zero at y f equals 1 and staying exactly zero after it. The logistic loss is a smooth curve that keeps decreasing but never reaches zero." style="max-width:440px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <rect x="50" y="43.5" width="120" height="181.5" fill="#f43f5e" fill-opacity="0.06"/>
  <rect x="170" y="43.5" width="60" height="181.5" fill="#f59e0b" fill-opacity="0.1"/>
  <text x="110.0" y="37.5" fill="#f43f5e" text-anchor="middle">wrong side</text>
  <text x="200.0" y="37.5" fill="#b45309" text-anchor="middle">in the street</text>
  <text x="290.0" y="37.5" fill="currentColor" fill-opacity="0.7" text-anchor="middle">beyond its gutter</text>
  <line x1="50" y1="225" x2="350" y2="225" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="170" y1="225" x2="170" y2="43.5" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="50" y="241" fill="currentColor" text-anchor="middle">−2</text>
  <text x="110" y="241" fill="currentColor" text-anchor="middle">−1</text>
  <text x="170" y="241" fill="currentColor" text-anchor="middle">0</text>
  <text x="230" y="241" fill="currentColor" text-anchor="middle">1</text>
  <text x="290" y="241" fill="currentColor" text-anchor="middle">2</text>
  <text x="350" y="241" fill="currentColor" text-anchor="middle">3</text>
  <text x="164" y="174" fill="currentColor" text-anchor="end">1</text>
  <text x="164" y="119" fill="currentColor" text-anchor="end">2</text>
  <text x="164" y="64" fill="currentColor" text-anchor="end">3</text>
  <text x="350" y="257" fill="currentColor" text-anchor="end">y·f(x): how far on the correct side</text>
  <polyline points="50,60 230,225 350,225" fill="none" stroke="#6366f1" stroke-width="2.2"/>
  <polyline points="50.0,108.0 53.0,110.4 56.0,112.8 59.0,115.2 62.0,117.6 65.0,119.9 68.0,122.3 71.0,124.6 74.0,126.9 77.0,129.2 80.0,131.4 83.0,133.7 86.0,135.9 89.0,138.1 92.0,140.2 95.0,142.4 98.0,144.5 101.0,146.6 104.0,148.7 107.0,150.7 110.0,152.8 113.0,154.8 116.0,156.7 119.0,158.7 122.0,160.6 125.0,162.5 128.0,164.3 131.0,166.1 134.0,167.9 137.0,169.7 140.0,171.4 143.0,173.1 146.0,174.8 149.0,176.4 152.0,178.0 155.0,179.6 158.0,181.1 161.0,182.6 164.0,184.1 167.0,185.5 170.0,186.9 173.0,188.2 176.0,189.6 179.0,190.8 182.0,192.1 185.0,193.3 188.0,194.5 191.0,195.7 194.0,196.8 197.0,197.9 200.0,198.9 203.0,199.9 206.0,200.9 209.0,201.9 212.0,202.8 215.0,203.7 218.0,204.6 221.0,205.4 224.0,206.2 227.0,207.0 230.0,207.8 233.0,208.5 236.0,209.2 239.0,209.9 242.0,210.5 245.0,211.1 248.0,211.7 251.0,212.3 254.0,212.9 257.0,213.4 260.0,213.9 263.0,214.4 266.0,214.9 269.0,215.3 272.0,215.8 275.0,216.2 278.0,216.6 281.0,217.0 284.0,217.3 287.0,217.7 290.0,218.0 293.0,218.3 296.0,218.6 299.0,218.9 302.0,219.2 305.0,219.5 308.0,219.7 311.0,220.0 314.0,220.2 317.0,220.4 320.0,220.7 323.0,220.9 326.0,221.1 329.0,221.2 332.0,221.4 335.0,221.6 338.0,221.8 341.0,221.9 344.0,222.1 347.0,222.2 350.0,222.3" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="6 3"/>
  <line x1="245.0" y1="82.0" x2="269.0" y2="82.0" stroke="#6366f1" stroke-width="2.2"/><text x="275.0" y="86.0" fill="currentColor">hinge: exactly 0</text>
  <line x1="245.0" y1="103.99999999999999" x2="269.0" y2="103.99999999999999" stroke="#10b981" stroke-width="2" stroke-dasharray="6 3"/><text x="275.0" y="107.99999999999999" fill="currentColor">logistic: never 0</text>
</svg>
</div>

*Hinge loss (solid) is exactly 0 once a point is beyond its gutter,
$y f(x) \ge 1$. Logistic loss (dashed) keeps shrinking but never reaches 0.*

The full objective balances a wide street against the total charge:

$$\min_{w, b}\ \tfrac12\|w\|^2 + C\sum_i \max\big(0,\ 1 - y_i f(x_i)\big)$$

**$C$ sets the price of each violation.** Large $C$ makes violations
expensive: the street gets narrow so that almost every point is outside it,
which fits the training data closely, including its noise (low bias, high
variance). Small $C$ makes violations cheap: the street stays wide and lets
some points in (high bias, low variance). This is the same structure as
regularized logistic regression, a loss plus an L2 penalty, with $C$ acting
as $1/\lambda$.

### 4.3 The kernel trick: curved boundaries without creating new features

The kernel trick solves a simple problem: what if the data cannot be
separated by a straight line?

An SVM is naturally good at drawing straight decision boundaries, but real
data often needs a curved one. One solution is to transform the data into a
new feature space where a straight line can separate it. The problem:
creating all those new features can be extremely expensive. The kernel
trick gives the benefits of the new features without ever calculating them.

#### The problem: a straight line is not enough

Take these points on a number line:

- blue points: $-1$, 0, 1;
- red points: $-3$, $-2$, 2, 3.

Can one threshold separate blue from red? No. The blue points are in the
middle and the red points are on both sides.

**The idea: give every point an additional feature.** Instead of
representing a point by its position $x$ alone, represent it by two
numbers:

$$x \longrightarrow (x,\ x^2)$$

| original $x$ | new feature $x^2$ | class |
|---|---|---|
| $-3$ | 9 | red |
| $-2$ | 4 | red |
| $-1$ | 1 | blue |
| 0 | 0 | blue |
| 1 | 1 | blue |
| 2 | 4 | red |
| 3 | 9 | red |

Now the classes are easy to separate: draw the horizontal line $x^2 = 2.5$.
Below it is blue, above it is red.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 380 290" role="img" aria-label="Bottom: seven points on a number line at x from minus 3 to 3. The three middle points, at minus 1, 0 and 1, are one class; the outer points, at minus 3, minus 2, 2 and 3, are the other. No single cut on the line separates them. Top: the same points plotted at height x squared lie on a parabola; the middle class sits low and the outer class high, and the horizontal line x squared equals 2.5 separates them." style="max-width:420px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <line x1="40" y1="200" x2="340" y2="200" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="190" y1="200" x2="190" y2="20" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="196" y="28" fill="currentColor">x²</text>
  <line x1="45" y1="157.5" x2="335" y2="157.5" stroke="#10b981" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="335" y="151.5" fill="#10b981" text-anchor="end">x² = 2.5 separates them</text>
  <path d="M 46.0,25.9 L 50.5,36.6 L 55.0,47.0 L 59.5,57.0 L 64.0,66.7 L 68.5,76.1 L 73.0,85.1 L 77.5,93.8 L 82.0,102.1 L 86.5,110.1 L 91.0,117.7 L 95.5,125.0 L 100.0,132.0 L 104.5,138.6 L 109.0,144.9 L 113.5,150.9 L 118.0,156.5 L 122.5,161.8 L 127.0,166.7 L 131.5,171.3 L 136.0,175.5 L 140.5,179.4 L 145.0,183.0 L 149.5,186.2 L 154.0,189.1 L 158.5,191.7 L 163.0,193.9 L 167.5,195.8 L 172.0,197.3 L 176.5,198.5 L 181.0,199.3 L 185.5,199.8 L 190.0,200.0 L 194.5,199.8 L 199.0,199.3 L 203.5,198.5 L 208.0,197.3 L 212.5,195.8 L 217.0,193.9 L 221.5,191.7 L 226.0,189.1 L 230.5,186.2 L 235.0,183.0 L 239.5,179.4 L 244.0,175.5 L 248.5,171.3 L 253.0,166.7 L 257.5,161.8 L 262.0,156.5 L 266.5,150.9 L 271.0,144.9 L 275.5,138.6 L 280.0,132.0 L 284.5,125.0 L 289.0,117.7 L 293.5,110.1 L 298.0,102.1 L 302.5,93.8 L 307.0,85.1 L 311.5,76.1 L 316.0,66.7 L 320.5,57.0 L 325.0,47.0 L 329.5,36.6 L 334.0,25.9" fill="none" stroke="currentColor" stroke-opacity="0.25"/>
  <rect x="50" y="42" width="10" height="10" fill="#f43f5e"/>
  <rect x="95" y="127" width="10" height="10" fill="#f43f5e"/>
  <circle cx="145" cy="183" r="5" fill="#6366f1"/>
  <circle cx="190" cy="200" r="5" fill="#6366f1"/>
  <circle cx="235" cy="183" r="5" fill="#6366f1"/>
  <rect x="275" y="127" width="10" height="10" fill="#f43f5e"/>
  <rect x="320" y="42" width="10" height="10" fill="#f43f5e"/>
  <line x1="40" y1="240" x2="340" y2="240" stroke="currentColor" stroke-opacity="0.5"/>
  <rect x="50" y="235" width="10" height="10" fill="#f43f5e"/>
  <text x="55" y="257" fill="currentColor" text-anchor="middle">−3</text>
  <rect x="95" y="235" width="10" height="10" fill="#f43f5e"/>
  <text x="100" y="257" fill="currentColor" text-anchor="middle">−2</text>
  <circle cx="145" cy="240" r="5" fill="#6366f1"/>
  <text x="145" y="257" fill="currentColor" text-anchor="middle">−1</text>
  <circle cx="190" cy="240" r="5" fill="#6366f1"/>
  <text x="190" y="257" fill="currentColor" text-anchor="middle">0</text>
  <circle cx="235" cy="240" r="5" fill="#6366f1"/>
  <text x="235" y="257" fill="currentColor" text-anchor="middle">1</text>
  <rect x="275" y="235" width="10" height="10" fill="#f43f5e"/>
  <text x="280" y="257" fill="currentColor" text-anchor="middle">2</text>
  <rect x="320" y="235" width="10" height="10" fill="#f43f5e"/>
  <text x="325" y="257" fill="currentColor" text-anchor="middle">3</text>
  <text x="40" y="282" fill="currentColor" fill-opacity="0.8">bottom: one feature x, no single cut works</text>
  <text x="40" y="30" fill="currentColor" fill-opacity="0.8">top: add the feature x²</text>
</svg>
</div>

*Bottom: with one feature $x$, the blue points sit between the red ones. Top: plotted at height $x^2$, blue is low and red is high.*

In the original space, this one line corresponds to two boundaries,
$x = -\sqrt{2.5}$ and $x = +\sqrt{2.5}$ (about $\pm 1.58$). So a straight
line in the new feature space becomes a curved (here, two-part) boundary in
the original space. Data that a straight line cannot separate can become
separable after mapping it into more dimensions. This is called a
**feature mapping**, written $\phi(x)$.

**But there is a problem.** With 100 original features instead of one, we
might create features such as $x_1^2,\ x_2^2,\ x_1 x_2,\ x_1 x_3, \ldots$
All products of up to 10 features number about 47 trillion. Building and
storing all of them is impractical. We need a shortcut.

#### The big idea: the SVM doesn't need the new features

Here is the observation that makes the kernel trick possible: an SVM does
not need to look at individual features directly. It only needs to
**compare one data point with another**, and it compares two points with a
dot product:

$$x \cdot z = x_1 z_1 + x_2 z_2 + \cdots + x_m z_m$$

With a feature mapping $\phi$, the SVM would need $\phi(x) \cdot \phi(z)$,
which normally means three steps:

1. transform $x$ into its new features;
2. transform $z$ into its new features;
3. take their dot product.

What if we could get that same number directly from the original points?
That is exactly what a **kernel** does:

$$\boxed{K(x, z) = \phi(x) \cdot \phi(z)}$$

A kernel is a shortcut formula for the dot product in the new feature
space, computed without constructing the new features.

| without the kernel trick | with the kernel trick |
|---|---|
| original points $x, z$ | original points $x, z$ |
| build all new features $\phi(x), \phi(z)$ | compute $K(x, z)$ directly |
| take their dot product | same result, no new features built |

That is the whole kernel trick in one idea. But why can an SVM get away
with using only dot products?

#### Why an SVM only needs dot products

The SVM decision function is $f(x) = w \cdot x + b$: $w$ sets the direction
of the boundary, $b$ its position, and the sign of $f(x)$ is the predicted
class. Normally we think of the SVM as learning $w$. But there is another
way to see how $w$ is built.

**Step 1: the weight vector is a weighted sum of training points.** The
SVM solves (§4.1)

$$\min_{w, b}\ \tfrac12\|w\|^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \ge 1$$

Introduce a Lagrange multiplier $\alpha_i \ge 0$ for every training point.
The Lagrangian is

$$L = \tfrac12\|w\|^2 - \sum_i \alpha_i \big[\,y_i(w^\top x_i + b) - 1\,\big]$$

At the optimum, its derivative with respect to $w$ is zero:

$$\frac{\partial L}{\partial w} = w - \sum_i \alpha_i\, y_i\, x_i = 0 \quad\Longrightarrow\quad \boxed{w = \sum_i \alpha_i\, y_i\, x_i}$$

The weight vector is a weighted sum of training points. Points that don't
affect the boundary end with $\alpha_i = 0$; the points with $\alpha_i > 0$
are the support vectors.

**Step 2: prediction can be written with dot products only.** Substitute
that $w$ into $f(x) = w \cdot x + b$:

$$f(x) = \Big(\sum_i \alpha_i y_i x_i\Big) \cdot x + b = \sum_i \alpha_i\, y_i\, (x_i \cdot x) + b$$

The SVM never needs $w$ itself. It only needs the training points, their
labels, their weights $\alpha_i$, and the dot product between each training
point and the new point.

**Example.** Three training points: red at $-1$, blue at 1 and 3. The
widest street runs from $-1$ to 1, so $f(x) = x$ ($w = 1$, $b = 0$).

<div style="margin:1.25rem 0">
<svg viewBox="0 0 560 150" role="img" aria-label="Three training points on a number line: a red point at minus 1 and blue points at 1 and 3. The street runs from minus 1 to 1 with the boundary at 0. The points at minus 1 and 1 are ringed support vectors with alpha 0.5 each; the point at 3 has alpha 0." style="max-width:580px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <rect x="150" y="40" width="180" height="50" fill="#f59e0b" fill-opacity="0.12"/>
  <line x1="240" y1="34" x2="240" y2="96" stroke="currentColor" stroke-width="1.8"/>
  <line x1="150" y1="34" x2="150" y2="96" stroke="currentColor" stroke-opacity="0.6" stroke-dasharray="5 4"/>
  <line x1="330" y1="34" x2="330" y2="96" stroke="currentColor" stroke-opacity="0.6" stroke-dasharray="5 4"/>
  <line x1="60" y1="65" x2="564.0" y2="65" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="240" y="26" text-anchor="middle" fill="currentColor">f = 0</text>
  <text x="150" y="26" text-anchor="middle" fill="currentColor">f = −1</text>
  <text x="330" y="26" text-anchor="middle" fill="currentColor">f = +1</text>
  <rect x="144" y="59" width="12" height="12" fill="#f43f5e"/>
  <circle cx="330" cy="65" r="6" fill="#6366f1"/>
  <circle cx="510" cy="65" r="6" fill="#6366f1"/>
  <circle cx="150" cy="65" r="11" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <circle cx="330" cy="65" r="11" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="150" y="115" text-anchor="middle" fill="currentColor">x = −1</text>
  <text x="150" y="132" text-anchor="middle" fill="#b45309" font-weight="600">α = 0.5</text>
  <text x="330" y="115" text-anchor="middle" fill="currentColor">x = 1</text>
  <text x="330" y="132" text-anchor="middle" fill="#b45309" font-weight="600">α = 0.5</text>
  <text x="510" y="115" text-anchor="middle" fill="currentColor">x = 3</text>
  <text x="510" y="132" text-anchor="middle" fill="#b45309" font-weight="600">α = 0</text>
</svg>
</div>

*The two points on the street's edges are the support vectors; the point at 3 has $\alpha = 0$.*

| point | label $y_i$ | weight $\alpha_i$ |
|---|---|---|
| $-1$ | $-1$ | 0.5 |
| 1 | $+1$ | 0.5 |
| 3 | $+1$ | 0 |

Check step 1: $w = 0.5\,(-1)(-1) + 0.5\,(+1)(1) = 1$. Check step 2 on a new
point $x = 2$:

$$f(2) = 0.5\,(-1)(-1 \times 2) + 0.5\,(+1)(1 \times 2) + 0 = 1 + 1 = 2$$

Positive, so blue, the same as $w \times 2 = 2$.

**Step 3: training also needs only dot products.** During training, the
SVM checks whether each point is correctly classified and outside its
margin: $y_j f(x_j) \ge 1$. By step 2,

$$f(x_j) = \sum_i \alpha_i\, y_i\, (x_i \cdot x_j) + b$$

So training depends only on dot products between training points. The SVM
can train by keeping track of the weights $\alpha_i$ instead of building
$w$, and it only needs the pairwise dot products $x_i \cdot x_j$.

**Step 4: replace the dot product with a kernel.** To make the SVM work in
a new feature space, we would normally replace each point $x$ with
$\phi(x)$. But since training and prediction only need dot products, we
can simply replace every $x_i \cdot x_j$ with $K(x_i, x_j)$, where
$K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$. The SVM behaves as if it were
working with all the new features, although they were never created.

#### A concrete example: the long way vs the shortcut

Two points $a = (1, 2)$ and $b = (3, 1)$. We want features containing all
products of two original features:

$$\phi(x) = \big(x_1^2,\ \sqrt2\, x_1 x_2,\ x_2^2\big)$$

(The $\sqrt2$ makes the dot product of the new features equal the square of
the original dot product.)

**Method 1: the long way.** Transform both points:
$\phi(a) = (1,\ 2\sqrt2,\ 4)$ and $\phi(b) = (9,\ 3\sqrt2,\ 1)$. Then

$$\phi(a) \cdot \phi(b) = (1)(9) + (2\sqrt2)(3\sqrt2) + (4)(1) = 9 + 12 + 4 = 25$$

**Method 2: the kernel shortcut.** Take the original dot product,
$a \cdot b = (1)(3) + (2)(1) = 5$, and square it:
$K(a, b) = (a \cdot b)^2 = 5^2 = 25$.

| | how | result |
|---|---|---|
| long way | build the features, then dot product | 25 |
| kernel trick | square the original dot product | 25 |

They match because expanding the kernel gives exactly the long way:

$$(a \cdot b)^2 = (a_1 b_1 + a_2 b_2)^2 = a_1^2 b_1^2 + 2 a_1 a_2 b_1 b_2 + a_2^2 b_2^2$$

The idea becomes much more powerful with many original features:
$K(a, b) = (a \cdot b + 1)^{10}$ is a single calculation, yet it equals a
dot product in a feature space containing all polynomial combinations up to
degree 10.

#### The kernels you actually use

In practice you don't invent a kernel. You choose one from a small
collection, and the SVM library handles the calculations.

**1. Linear kernel**, $K(x, z) = x \cdot z$. The ordinary dot product: no
new features, no curved boundaries. Use it when a straight boundary is
enough, when the data is very high-dimensional (such as text), or when you
want a simpler, faster model.

**2. Polynomial kernel**, $K(x, z) = (x \cdot z + c)^d$. It implicitly adds
polynomial features: with $d = 2$, squares of features and products of
pairs of features; higher $d$ captures more complex interactions. $d$ sets
the degree and $c$ the weight of the lower-degree terms. Use it when you
expect a curved boundary built from feature interactions.

**3. RBF (Gaussian) kernel**, $K(x, z) = e^{-\gamma\|x - z\|^2}$. One of
the most used nonlinear kernels. It is not limited to one polynomial
degree; it measures how close two points are. Think of it as a similarity
score:

- identical points: similarity 1;
- nearby points: similarity close to 1;
- distant points: similarity close to 0.

$\gamma$ controls how quickly the similarity drops with distance. $\gamma$
multiplies the squared distance, so a small $\gamma$ keeps the exponent
small and the similarity near 1 even for distant points:

| similarity $e^{-\gamma d^2}$ at distance $d$ | $d = 0.5$ | $d = 1$ | $d = 2$ |
|---|---|---|---|
| $\gamma = 0.1$ | 0.98 | 0.90 | 0.67 |
| $\gamma = 1$ | 0.78 | 0.37 | 0.02 |
| $\gamma = 10$ | 0.08 | 0.00005 | about 0 |

A training point's vote is weighted by this similarity, so with
$\gamma = 0.1$ a point still has two thirds of its full say at distance 2.
With $\gamma = 10$ it has almost none beyond distance 0.5. Roughly, a point
influences a region of radius $1/\sqrt\gamma$: 3.2 for $\gamma = 0.1$, 0.32
for $\gamma = 10$.

| | small $\gamma$ | large $\gamma$ |
|---|---|---|
| region each point influences | large | small |
| decision boundary | smooth | bends around single training points |
| risk | underfitting | overfitting |

An RBF SVM predicts mostly from nearby training points, somewhat like kNN
(§5). Because it uses distances, scale the features first.

#### The cost: kernels are not free

The kernel trick avoids building trillions of features, but the SVM still
computes the kernel between pairs of training points: about $n^2$ pairs for
$n$ examples.

| training examples | pairwise comparisons |
|---|---|
| 100 | 10,000 |
| 1,000 | 1 million |
| 10,000 | 100 million |
| 1,000,000 | 1 trillion |

Kernel SVM training takes roughly $O(n^2)$ to $O(n^3)$ time, which makes it
hard to scale to millions of examples. Neural networks became popular partly
because they learn useful features directly and scale much better, with
minibatch training on GPUs.

#### The whole idea in one list

1. **Start with data a straight line cannot separate.** The original space
   needs a curved boundary.
2. **Imagine mapping the data into more dimensions,** where a flat boundary
   can separate the classes.
3. **Notice that the SVM only needs dot products,** in both training and
   prediction.
4. **Replace those dot products with a kernel:** $x \cdot z \to K(x, z)$.
5. **Train the SVM without building the new features.** You get the effect
   of the larger space without paying to construct it.

The one equation to remember:

$$\boxed{K(x, z) = \phi(x) \cdot \phi(z)}$$

A kernel computes the dot product of two transformed points directly from
the original points.

**Quick check.**

1. What problem does the kernel trick solve?
2. Why can an SVM use a kernel instead of building new features?
3. What happens when $\gamma$ becomes very large in an RBF kernel?

<details><summary>Answers</summary>

1. It lets an SVM learn curved boundaries without explicitly creating new
   features.
2. Because training and prediction only need dot products between points.
3. Each point influences a smaller region, so the boundary can bend around
   single points and overfit.

</details>

If you remember one thing: the kernel trick doesn't make the data simpler.
It makes the math of working in a higher-dimensional feature space cheap,
by never constructing that space.

---

## 5. kNN and the curse of dimensionality

### 5.1 k-nearest neighbors

To classify a new point, find the $k$ training points closest to it and take
their majority label (for regression, average their values). There is no
training step: the model is the training data.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 260" role="img" aria-label="Blue circles and red squares scattered in two dimensions, with a new point marked by a star. A dashed circle around the star encloses its 5 nearest training points: 3 blue and 2 red, so the vote predicts blue." style="max-width:440px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <circle cx="185" cy="150" r="63" fill="#f59e0b" fill-opacity="0.08" stroke="#f59e0b" stroke-dasharray="5 4" stroke-width="1.5"/>
  <line x1="185" y1="150" x2="160" y2="140" stroke="#f59e0b" stroke-opacity="0.7"/>
  <line x1="185" y1="150" x2="200" y2="175" stroke="#f59e0b" stroke-opacity="0.7"/>
  <line x1="185" y1="150" x2="175" y2="105" stroke="#f59e0b" stroke-opacity="0.7"/>
  <line x1="185" y1="150" x2="220" y2="120" stroke="#f59e0b" stroke-opacity="0.7"/>
  <line x1="185" y1="150" x2="190" y2="205" stroke="#f59e0b" stroke-opacity="0.7"/>
  <circle cx="120" cy="80" r="6" fill="#6366f1"/>
  <circle cx="150" cy="60" r="6" fill="#6366f1"/>
  <circle cx="175" cy="105" r="6" fill="#6366f1"/>
  <circle cx="95" cy="120" r="6" fill="#6366f1"/>
  <circle cx="160" cy="140" r="6" fill="#6366f1"/>
  <circle cx="205" cy="75" r="6" fill="#6366f1"/>
  <circle cx="105" cy="185" r="6" fill="#6366f1"/>
  <circle cx="220" cy="120" r="6" fill="#6366f1"/>
  <rect x="244" y="174" width="12" height="12" fill="#f43f5e"/>
  <rect x="274" y="144" width="12" height="12" fill="#f43f5e"/>
  <rect x="294" y="194" width="12" height="12" fill="#f43f5e"/>
  <rect x="224" y="209" width="12" height="12" fill="#f43f5e"/>
  <rect x="314" y="114" width="12" height="12" fill="#f43f5e"/>
  <rect x="264" y="224" width="12" height="12" fill="#f43f5e"/>
  <rect x="194" y="169" width="12" height="12" fill="#f43f5e"/>
  <rect x="184" y="199" width="12" height="12" fill="#f43f5e"/>
  <rect x="294" y="84" width="12" height="12" fill="#f43f5e"/>
  <polygon points="185.0,141.0 187.4,146.8 193.6,147.2 188.8,151.2 190.3,157.3 185.0,154.0 179.7,157.3 181.2,151.2 176.4,147.2 182.6,146.8" fill="currentColor"/>
  <text x="20" y="24" fill="currentColor">★ new point. Its 5 nearest neighbors (inside the circle):</text>
  <text x="20" y="40" fill="currentColor">3 blue, 2 red, so k = 5 predicts blue.</text>
</svg>
</div>

*With $k = 1$ the prediction would copy the single closest point (blue).
With $k = 5$ it is a vote of the five points in the circle.*

The choice of $k$ sets the bias-variance tradeoff. With $k = 1$, each
prediction copies a single neighbor, so one mislabeled point changes the
answer near it: a jagged boundary that fits the noise (high variance). With
$k = n$, every prediction is the overall majority class (high bias). A
middle value, chosen on a validation set, works best.

kNN relies on one assumption: nearby points have similar labels. Two
practical points follow from "nearby":

- **Scale the features.** Distance adds up differences across features, so
  a feature with large numbers dominates. With house size in square feet
  (differences of hundreds) and number of bedrooms (differences of 1 or 2),
  the distance is almost entirely size. Standardize each feature first.
- **Prediction is slow.** Each prediction compares the new point with all
  $n$ training points, $O(nd)$ work, unless you build an index such as a
  KD-tree or use approximate nearest-neighbor search.

### 5.2 The curse of dimensionality

In many dimensions, the nearest neighbors are not near.

Spread 1,000 points uniformly in the cube $[0,1]^d$. How big a box around a
new point do you need to catch its 10 nearest neighbors, 1% of the data? A
box whose side covers a fraction $s$ of each feature's range holds $s^d$ of
the data, so $s^d = 0.01$ and $s = 0.01^{1/d}$:

| dimensions $d$ | 2 | 10 | 100 |
|---|---|---|---|
| side of the box | 0.1 | 0.63 | 0.955 |

In 2 dimensions the neighbors are within 10% of the range in each feature. In
100 dimensions the box must cover 95.5% of every feature's range: the
"neighbors" differ from the point in almost every feature.

**Second problem: every point is about equally far away.** Take 1,000
random points and one new point:

- in 2 dimensions, the nearest point is 100 times closer than the farthest;
- in 1,000 dimensions, the nearest point is only about 10% closer than the
  farthest.

So in high dimensions the "nearest" neighbor is barely nearer than any
other point, and kNN's vote is close to a vote of random points.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 190" role="img" aria-label="For 1,000 random points, each bar spans from 0 to the distance of the farthest point, and a marker shows the nearest point as a fraction of it: 1 percent in 2 dimensions, 22 percent in 10, 72 percent in 100, 90 percent in 1,000." style="max-width:440px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <text x="20" y="18" fill="currentColor">distance to the nearest point, as a share of the farthest</text>
  <text x="90" y="52" fill="currentColor" text-anchor="end">d = 2</text>
  <rect x="100" y="40" width="260" height="16" fill="currentColor" fill-opacity="0.08"/>
  <rect x="100" y="40" width="3" height="16" fill="#6366f1" fill-opacity="0.75"/>
  <text x="108" y="52" fill="currentColor">1%</text>
  <text x="90" y="88" fill="currentColor" text-anchor="end">d = 10</text>
  <rect x="100" y="76" width="260" height="16" fill="currentColor" fill-opacity="0.08"/>
  <rect x="100" y="76" width="57" height="16" fill="#6366f1" fill-opacity="0.75"/>
  <text x="162" y="88" fill="currentColor">22%</text>
  <text x="90" y="124" fill="currentColor" text-anchor="end">d = 100</text>
  <rect x="100" y="112" width="260" height="16" fill="currentColor" fill-opacity="0.08"/>
  <rect x="100" y="112" width="187" height="16" fill="#6366f1" fill-opacity="0.75"/>
  <text x="292" y="124" fill="currentColor">72%</text>
  <text x="90" y="160" fill="currentColor" text-anchor="end">d = 1,000</text>
  <rect x="100" y="148" width="260" height="16" fill="currentColor" fill-opacity="0.08"/>
  <rect x="100" y="148" width="234" height="16" fill="#6366f1" fill-opacity="0.75"/>
  <text x="329" y="160" fill="#fff" text-anchor="end">90%</text>
  <text x="100" y="186" fill="currentColor" fill-opacity="0.7">0</text><text x="360" y="186" fill="currentColor" fill-opacity="0.7" text-anchor="end">farthest</text>
</svg>
</div>

*1,000 random points in the cube $[0,1]^d$, distances from one more random
point. In 2 dimensions the nearest point is very close; in 1,000 dimensions
it is almost as far as the farthest.*

 Keeping neighborhoods small would need a number of points
that grows exponentially with $d$.

**What helps.**

- **Real data is not spread evenly.** The numbers above are for random
  points filling the whole cube. Real data sits near a much smaller surface.
  A 64×64 face photo has 4,096 pixel features, but faces differ in only a
  few ways (pose, lighting, expression, identity), so face photos fill a
  tiny part of that space. Neighbors on that surface can still be close, so
  kNN often works better on real data than the cube suggests.
- **Reduce the dimensions first.** PCA ([§10](04_clustering_and_dim_reduction.md#10))
  keeps the few directions along which the data varies most, for example
  4,096 pixels down to 50 numbers. Then run kNN on those 50.
- **Drop useless features.** Every irrelevant feature adds noise to every
  distance. Shoe size in a house-price model makes houses look different
  for no reason.
- **Learn the distance.** Train a neural network to map each input to an
  embedding where points with the same label end up close together (for
  example with a contrastive loss, Probability §3.4.4). Then use kNN on the
  embeddings. This is how face recognition, image search and retrieval
  systems find neighbors.

---

## 6. Naive Bayes

### 6.1 The idea

A new email contains the words "free" and "meeting". Is it spam?

Naive Bayes asks two questions about each class, spam and not spam:

1. **How common is the class?** If half of all emails are spam,
   $p(\text{spam}) = 0.5$.
2. **If the email were in this class, how likely are its words?** Spam
   often contains "free" and rarely "meeting". Normal email is the
   opposite.

It multiplies the two answers for each class, and the class with the
larger product wins. This is Bayes rule (Probability §3.3):

$$p(\text{class} \mid \text{email}) \propto \underbrace{p(\text{class})}_{\text{question 1}} \times \underbrace{p(\text{email} \mid \text{class})}_{\text{question 2}}$$

Question 2 is the hard part. Naive Bayes makes it easy by assuming
the words appear **independently** of each other within a class, so the
probability of the email is a product of one number per word:

$$p(\text{email} \mid \text{class}) = \prod_{j} p(\text{word}_j \mid \text{class})$$

Each number is a count from the training data, such as "60% of spam emails
contain 'free'":

| | $p(\text{class})$ | $p(\text{"free"} \mid \text{class})$ | $p(\text{"meeting"} \mid \text{class})$ | product |
|---|---|---|---|---|
| spam | 0.5 | 0.6 | 0.01 | 0.003 |
| not spam | 0.5 | 0.05 | 0.3 | 0.0075 |

"Not spam" has the larger product, so that is the prediction.
(Normalizing, $p(\text{spam}) = 0.003/(0.003 + 0.0075) = 0.29$.)

**Why the assumption is needed.** Without it, you would need the probability
of every combination of words. With a 10,000-word vocabulary there are
$2^{10000}$ combinations, and almost every email in the training set is a
combination seen only once. With the assumption, you need one count per
word per class.

In practice, add the logs of the probabilities instead of multiplying them:
a product of hundreds of small numbers underflows to 0 in floating point.

### 6.2 Laplace smoothing

Suppose "meeting" never appears in the 200 spam training emails. Then
$p(\text{"meeting"} \mid \text{spam}) = 0/200 = 0$, and every email
containing "meeting" gets a spam score of exactly 0, even "FREE PRIZE, claim
your free money, meeting". One unseen word overrides all the others.

The fix is to pretend you saw a few extra emails: add a pseudo-count
$\alpha$ (usually 1) to every count. For a word that is either present or
absent, add one imaginary spam email that has it and one that doesn't:

$$p(\text{"meeting"} \mid \text{spam}) = \frac{0 + 1}{200 + 2} \approx 0.005$$

Small, but no longer 0, so the other words still count.

### 6.3 Why it works when the assumption is wrong

Words are not independent: "free" and "prize" often appear together. Naive
Bayes then counts the same evidence twice, which pushes its probabilities
toward 0 and 1. So its probabilities are overconfident. But classifying only
needs the right class to have the larger score, and double-counting often
leaves the ranking unchanged. So naive Bayes is a decent classifier with
poorly calibrated probabilities.

### 6.4 Generative vs discriminative

There are two ways to build a spam classifier.

- **Generative (naive Bayes): learn what each class looks like.** It learns
  how spam emails are written (which words they contain, how often) and
  how normal emails are written: $p(\text{email} \mid \text{class})$ for each class. To classify a
  new email, it asks which class was more likely to produce it, using Bayes
  rule. Because it describes each class completely, it could even generate
  new fake emails of either class.
- **Discriminative (logistic regression): learn only the difference.** It
  never describes what an email looks like. It learns one weight per word
  that pushes the score toward spam or not spam, which is $p(\text{class} \mid \text{email})$
  directly ([§2](01_linear_models.md#2)).

| | naive Bayes (generative) | logistic regression (discriminative) |
|---|---|---|
| models | $p(\text{email} \mid \text{class})$, then Bayes rule | $p(\text{class} \mid \text{email})$ directly |
| training | counting, one pass over the data | gradient descent |
| small datasets | better: simple counts are estimated well from few examples | worse: weights are noisy |
| large datasets | worse: limited by the wrong independence assumption | better: no independence assumption |
| probabilities | overconfident (§6.3) | usually well calibrated |

A classic result (Ng and Jordan, 2002) makes the data tradeoff precise:
naive Bayes reaches its best accuracy with far fewer examples, but logistic
regression's best accuracy is higher. So with little data, naive Bayes
often wins; with plenty, logistic regression does.
