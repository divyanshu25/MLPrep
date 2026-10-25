# Calculus for ML

## 2.1 The derivative is local sensitivity

The derivative answers one question: **if the input changes a little, how
much does the output change?**

$$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad \text{for tiny } h$$

Take $f(x) = x^2$ at $x = 3$: $f(3) = 9$ and $f(3.001) = 9.006001$. The output
moved $0.006001$ for an input change of $0.001$, a ratio of about 6, which is
$2x$ at $x = 3$. So the derivative is a rate: output change per unit of input
change, for small changes.

Another way to see it: zoom in far enough on a smooth curve and it looks like
a straight line. $f'(x)$ is that line's slope:

$$f(x + h) \approx f(x) + f'(x)\,h$$

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 230" role="img" aria-label="The parabola f(x) = x squared with its tangent line at a point: an input step h moves the output by about f prime of x times h" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="30" y1="200" x2="260" y2="200"/>
    <line x1="50" y1="215" x2="50" y2="25"/>
  </g>
  <path d="M 41 199.6 Q 137.7 208.2 234.5 31.9" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <line x1="95" y1="200" x2="225" y2="84.4" stroke="#f59e0b" stroke-width="2"/>
  <line x1="140" y1="160" x2="140" y2="200" stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="4 4"/>
  <line x1="140" y1="160" x2="176" y2="160" stroke="currentColor" stroke-opacity="0.5" stroke-dasharray="4 4"/>
  <line x1="176" y1="160" x2="176" y2="128" stroke="#f59e0b" stroke-dasharray="4 4"/>
  <circle cx="140" cy="160" r="4" fill="#6366f1"/>
  <text x="140" y="214" text-anchor="middle" fill="currentColor" fill-opacity="0.7">x</text>
  <text x="158" y="175" text-anchor="middle" fill="currentColor" fill-opacity="0.8">h</text>
  <text x="181" y="146" fill="#f59e0b">f&#8242;(x)&#183;h</text>
  <text x="216" y="40" text-anchor="end" fill="#6366f1" font-weight="600">f(x) = x&#178;</text>
  <text x="196" y="126" fill="#f59e0b">slope f&#8242;(x) = 2x</text>
</svg>
</div>

*The tangent is what the curve looks like up close. Step $h$ to the right and
the output rises by about $f'(x)\,h$. The small gap between tangent and curve
shrinks like $h^2$ as you zoom in.*

**Why this matters for training.** The derivative of the loss with respect to
a weight says: increase this weight by a small $\epsilon$, and the loss changes
by about (derivative) $\times\ \epsilon$. Gradient descent uses exactly this. It
moves every weight in the direction that lowers the loss, by an amount
proportional to that derivative.

**Checking derivatives in code: use the centered difference.** The formula
above is the **forward difference**. The **centered difference**

$$\frac{f(x+h) - f(x-h)}{2h}$$

is far more accurate for the same $h$. Its error shrinks like $h^2$ (written
$O(h^2)$), while the forward difference's error shrinks like $h$ ($O(h)$). On
$f(x) = x^3$ at $x = 1$, where the true derivative is 3:

| $h$ | forward | error | centered | error |
|---|---|---|---|---|
| $0.01$ | $3.0301$ | $0.0301$ | $3.0001$ | $0.0001$ |
| $0.001$ | $3.003001$ | $0.0030$ | $3.000001$ | $0.000001$ |

Shrinking $h$ tenfold makes the forward error 10× smaller and the centered
error 100× smaller.

*Why.* Taylor's theorem writes $f$ near $x$ as a polynomial in the step:

$$f(x \pm h) = f(x) \pm f'(x)\,h + \tfrac{f''(x)}{2} h^2 \pm \tfrac{f'''(x)}{6} h^3 + \cdots$$

The forward difference keeps the $h^2$ term, which becomes an error of
$\tfrac{f''(x)}{2} h$ after dividing by $h$. In $f(x+h) - f(x-h)$, the even
powers ($f(x)$, the $h^2$ term, ...) have the same sign on both sides and
cancel, so the leading error is $\tfrac{f'''(x)}{6} h^2$. For $x^3$ at $x = 1$
with $h = 0.01$, these predict errors of $\tfrac{6}{2}(0.01) = 0.03$ and
$\tfrac{6}{6}(0.01)^2 = 0.0001$, matching the table. The centered difference
is the standard gradient checker, used in `gradients_from_scratch.ipynb`.

## 2.2 The chain rule: differentiating a pipeline

A network computes its loss through a pipeline: weights → layer 1 → layer 2
→ ... → loss. Training needs $\partial L / \partial w$ for every weight, so it
needs derivatives through a pipeline. The chain rule gives them:
**rates multiply along a pipeline.**

#### The rule

Take two steps: $z = g(x)$, then $y = f(z)$. Change $x$ by a small $\epsilon$.
Then $z$ changes by $g'(x)\,\epsilon$, and that change in $z$ changes $y$ by
$f'(z)$ times as much, $f'(z)\,g'(x)\,\epsilon$. So

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}$$

#### Run it with numbers

$z = 3x$, $y = z^2$, at $x = 2$ (so $z = 6$):

$$\frac{dz}{dx} = 3, \qquad \frac{dy}{dz} = 2z = 12, \qquad
\frac{dy}{dx} = 12 \cdot 3 = 36$$

Check without the chain rule: $y = (3x)^2 = 9x^2$, so $y' = 18x = 36$ at
$x = 2$.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 170" role="img" aria-label="Pipeline x to z to y: forward pass computes values left to right, backward pass multiplies local rates of change right to left giving dy/dx = 12 times 3 = 36" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="p7-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
    <marker id="p7-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e"/></marker>
  </defs>
  <text x="117" y="50" text-anchor="middle" fill="currentColor" fill-opacity="0.8">z = 3x</text>
  <text x="222" y="50" text-anchor="middle" fill="currentColor" fill-opacity="0.8">y = z&#178;</text>
  <rect x="30" y="60" width="70" height="34" rx="7" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <rect x="135" y="60" width="70" height="34" rx="7" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <rect x="240" y="60" width="70" height="34" rx="7" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="65" y="82" text-anchor="middle" fill="currentColor">x = 2</text>
  <text x="170" y="82" text-anchor="middle" fill="currentColor">z = 6</text>
  <text x="275" y="82" text-anchor="middle" fill="currentColor">y = 36</text>
  <line x1="100" y1="77" x2="128" y2="77" stroke="currentColor" stroke-opacity="0.6" marker-end="url(#p7-f)"/>
  <line x1="205" y1="77" x2="233" y2="77" stroke="currentColor" stroke-opacity="0.6" marker-end="url(#p7-f)"/>
  <line x1="240" y1="112" x2="212" y2="112" stroke="#f43f5e" marker-end="url(#p7-b)"/>
  <line x1="135" y1="112" x2="107" y2="112" stroke="#f43f5e" marker-end="url(#p7-b)"/>
  <text x="226" y="132" text-anchor="middle" fill="#f43f5e">dy/dz = 2z = 12</text>
  <text x="120" y="132" text-anchor="middle" fill="#f43f5e">dz/dx = 3</text>
  <text x="170" y="158" text-anchor="middle" fill="#f43f5e" font-weight="600">dy/dx = 12 &#183; 3 = 36</text>
</svg>
</div>

*Forward (top): compute values left to right. Backward (red): multiply local
rates right to left.*

#### What dy/dx = 36 means

It is a rate for small steps, not the change from a step of 1. From $x = 2$ to
$x = 3$, $y$ goes from 36 to 81, a change of 45.

The exact change from a step of size $h$ is ($\Delta y$, read "delta y", is the
value after the step minus the value before):

$$\Delta y = 9(2+h)^2 - 9 \cdot 2^2 = 36h + 9h^2$$

The $36h$ is what the derivative predicts. The $9h^2$ is what it misses, and
that part shrinks much faster:

- $h = 1$: predicted 36, actual 45, missed 9.
- $h = 0.1$: predicted 3.6, actual 3.69, missed 0.09.
- $h = 0.01$: predicted 0.36, actual 0.3609, missed 0.0009.

So $\Delta y / h = 36 + 9h$, which approaches 36 as $h$ shrinks. That limit is
what $dy/dx = 36$ means. It is also why learning rates are small: the gradient
predicts the loss well only near the current weights.

#### Backprop: the chain rule over a whole graph

**Backprop finds $\partial L / \partial v$ for every variable $v$ in one
backward walk: start at the loss and multiply by one local derivative per
step.**

1. **Forward pass:** compute and store every intermediate value.
2. **Backward pass:** start at the loss with $\partial L / \partial L = 1$ and
   walk the graph in reverse. If $v$ feeds into $u$, then
   $\partial L / \partial v = (\partial L / \partial u) \cdot (\partial u / \partial v)$.
   The first factor is already known, because the walk reached $u$ first;
   libraries call it the **upstream gradient**. The second factor is the
   **local derivative** of the operation that computed $u$, and it is the only
   thing each operation has to provide.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 500 210" role="img" aria-label="Computation graph of L equals (w x plus b) squared: boxes hold the forward value and, in red, the derivative of the loss with respect to that variable; amber edge labels are local derivatives" style="max-width:520px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="bp-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
    <marker id="bp-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.5">
    <rect x="24" y="18" width="96" height="44" rx="7"/>
    <rect x="24" y="83" width="72" height="30" rx="7"/>
    <rect x="24" y="148" width="96" height="44" rx="7"/>
    <rect x="220" y="76" width="112" height="44" rx="7"/>
    <rect x="400" y="76" width="92" height="44" rx="7"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="72" y="36">w = 2</text>
    <text x="60" y="102">x = 3</text>
    <text x="72" y="166">b = 1</text>
    <text x="276" y="94">z = wx + b = 7</text>
    <text x="446" y="94">L = z&#178; = 49</text>
  </g>
  <g fill="#f43f5e" text-anchor="middle" font-size="10.5">
    <text x="72" y="54">&#8706;L/&#8706;w = 42</text>
    <text x="72" y="184">&#8706;L/&#8706;b = 14</text>
    <text x="276" y="112">&#8706;L/&#8706;z = 14</text>
    <text x="446" y="112">&#8706;L/&#8706;L = 1</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.6">
    <line x1="120" y1="40" x2="218" y2="90" marker-end="url(#bp-f)"/>
    <line x1="96" y1="98" x2="218" y2="98" marker-end="url(#bp-f)"/>
    <line x1="120" y1="170" x2="218" y2="106" marker-end="url(#bp-f)"/>
    <line x1="332" y1="98" x2="398" y2="98" marker-end="url(#bp-f)"/>
  </g>
  <g font-size="10" fill="#f59e0b" text-anchor="middle">
    <text x="166" y="52">&#8706;z/&#8706;w = 3</text>
    <text x="166" y="152">&#8706;z/&#8706;b = 1</text>
    <text x="365" y="66">&#8706;L/&#8706;z = 2z = 14</text>
  </g>
  <line x1="410" y1="198" x2="110" y2="198" stroke="#f43f5e" marker-end="url(#bp-b)"/>
  <text x="260" y="190" text-anchor="middle" fill="#f43f5e" font-size="10">backward walk</text>
</svg>
</div>

*$L = (wx + b)^2$ at $w = 2$, $x = 3$, $b = 1$. Red inside a box:
$\partial L / \partial(\text{that variable})$. Each red number is the next red
number times the amber local derivative between them: $42 = 14 \cdot 3$,
$14 = 14 \cdot 1$. $x$ is data, not a parameter, so the walk skips it.*

#### A variable used in two places

If a variable feeds two later steps, it affects the loss along two paths. Each
path contributes one chain-rule product, and the contributions add:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial u}\,\frac{\partial u}{\partial w} + \frac{\partial L}{\partial v}\,\frac{\partial v}{\partial w}$$

Check it on $L = w^2 + 3w$, which uses $w$ twice ($u = w^2$, $v = 3w$,
$L = u + v$). The path through $u$ contributes $1 \cdot 2w$, the path through
$v$ contributes $1 \cdot 3$, total $2w + 3$, which is the derivative of $L$
taken directly.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 460 240" role="img" aria-label="Computation graph of L equals w squared plus 3 w at w equals 2: w feeds two paths that both reach L, and the two backward contributions add at w" style="max-width:480px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="tu-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.5">
    <rect x="24" y="98" width="120" height="44" rx="7"/>
    <rect x="196" y="24" width="100" height="44" rx="7"/>
    <rect x="196" y="172" width="100" height="44" rx="7"/>
    <rect x="346" y="98" width="112" height="44" rx="7"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="84" y="116">w = 2</text>
    <text x="246" y="42">u = w&#178; = 4</text>
    <text x="246" y="190">v = 3w = 6</text>
    <text x="402" y="116">L = u + v = 10</text>
  </g>
  <g fill="#f43f5e" text-anchor="middle" font-size="10.5">
    <text x="84" y="134">&#8706;L/&#8706;w = 4 + 3 = 7</text>
    <text x="246" y="60">&#8706;L/&#8706;u = 1</text>
    <text x="246" y="208">&#8706;L/&#8706;v = 1</text>
    <text x="402" y="134">&#8706;L/&#8706;L = 1</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.6">
    <line x1="144" y1="106" x2="194" y2="52" marker-end="url(#tu-f)"/>
    <line x1="144" y1="136" x2="194" y2="188" marker-end="url(#tu-f)"/>
    <line x1="296" y1="52" x2="344" y2="104" marker-end="url(#tu-f)"/>
    <line x1="296" y1="190" x2="344" y2="136" marker-end="url(#tu-f)"/>
  </g>
  <g font-size="10" fill="#f59e0b" text-anchor="end">
    <text x="164" y="74">&#8706;u/&#8706;w = 4</text>
    <text x="164" y="174">&#8706;v/&#8706;w = 3</text>
  </g>
</svg>
</div>

*$L = u + v$ with $u = w^2$ and $v = 3w$, at $w = 2$. One contribution per
path arrives back at $w$: $1 \cdot 4$ through $u$ and $1 \cdot 3$ through $v$.
They add: $\partial L / \partial w = 7$, matching $2w + 3$ at $w = 2$.*

So backprop needs only two rules: **multiply along a path, add across
paths.** Every layer type (attention, LayerNorm, convolution) only has to
supply its local derivatives.

#### Why walk the graph backward?

Training needs the derivative of one number, the loss, with respect to every
weight. A backward walk gives all of them at once; a forward walk gives one.

- A **forward walk** picks one weight $w_1$ and pushes its effect left to
  right, computing $\partial v / \partial w_1$ for every variable $v$. At the
  end it has $\partial L / \partial w_1$. That is one weight per walk, so $n$
  weights take $n$ walks.
- A **backward walk** starts at the loss and computes $\partial L / \partial v$
  for every variable $v$ (the recipe above). One walk gives the derivative for
  all $n$ weights.

Each walk costs about one pass over the graph. For a billion weights, that is
a billion forward walks or one backward walk. This is why backprop goes
backward, and why training large networks is feasible at all.

## 2.3 Gradients of vector/matrix expressions

A loss takes all $n$ weights and returns one number, $L(w_1, \dots, w_n)$. (The
notation $f: \mathbb{R}^n \to \mathbb{R}$ means $n$ numbers in, one number
out.) The **gradient** lists its sensitivity to each input:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]$$

Each entry is a **partial derivative**: hold every other input fixed, change
$x_i$ alone, and measure the rate as in §2.1. $\nabla$ is read "nabla" or
"grad".

Example: $f(x_1, x_2) = x_1^2 + 3x_2$ at $(2, 5)$ has
$\partial f / \partial x_1 = 2x_1 = 4$ and $\partial f / \partial x_2 = 3$, so
$\nabla f = [4, 3]$. The backward walk in §2.2 computed a gradient too: with
the data $x$ held fixed, $L = (wx + b)^2$ is a function of $w$ and $b$, and its
gradient is $[42, 14]$.

#### The gradient points straight uphill

**The gradient points in the direction where $f$ increases fastest, and its
length is that fastest rate.** Gradient descent steps along $-\nabla f$,
straight downhill.

Why: start with one concrete step on the example above, where
$\nabla f = [4, 3]$. That gradient says $f$ rises about 4 per unit of $x_1$
and 3 per unit of $x_2$. Move $x_1$ by $0.1$ and $x_2$ by $0.2$ at the same
time. The $x_1$ move adds about $4 \times 0.1 = 0.4$, the $x_2$ move adds
about $3 \times 0.2 = 0.6$, and for small steps the two effects simply add:
$\Delta f \approx 1.0$. (Exactly: $f$ goes from 19 to 20.01.)

Each sensitivity times its input's change, summed, is a dot product. So for
any small step $\delta$:

$$\Delta f \approx \nabla f \cdot \delta \qquad \text{here } [4, 3] \cdot [0.1, 0.2] = 1.0$$

To compare directions fairly, give every step the same small length
$\epsilon$: write the step as $\delta = \epsilon\, u$, where $u$ is a unit
vector ($\|u\| = 1$) that only says which way to go. Then
$\Delta f \approx \epsilon\, (\nabla f \cdot u)$, so the rate at which $f$
rises in direction $u$ is $\nabla f \cdot u$.

Which $u$ makes this largest? By the dot-product identity (§1.2 of the
linear algebra notes), $\nabla f \cdot u = \|\nabla f\| \cos\theta$, where
$\theta$ is the
angle between $u$ and $\nabla f$. This is largest at $\theta = 0$, when $u$
points along $\nabla f$, and the rate there is $\|\nabla f\|$.

With $\nabla f = [4, 3]$: stepping along $[1, 0]$ raises $f$ at rate 4, along
$[0, 1]$ at rate 3, and along $[4, 3]/5$ at rate
$4 \cdot \tfrac{4}{5} + 3 \cdot \tfrac{3}{5} = 5 = \|\nabla f\|$, faster than
either axis.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 400 270" role="img" aria-label="Unit circle of candidate step directions for gradient [4,3]: rate 4 along the x axis, 3 along the y axis, 0 perpendicular to the gradient, and 5 along the gradient direction; level lines of f run perpendicular to the gradient" style="max-width:420px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="gu-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
    <marker id="gu-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.15">
    <line x1="108" y1="101" x2="180" y2="197"/>
    <line x1="136" y1="80" x2="208" y2="176"/>
    <line x1="164" y1="59" x2="236" y2="155"/>
    <line x1="192" y1="38" x2="264" y2="134"/>
  </g>
  <circle cx="140" cy="152" r="95" fill="none" stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="5 5"/>
  <circle cx="140" cy="152" r="3.5" fill="currentColor"/>
  <g stroke="currentColor" stroke-opacity="0.75">
    <line x1="140" y1="152" x2="233" y2="152" marker-end="url(#gu-a)"/>
    <line x1="140" y1="152" x2="140" y2="59" marker-end="url(#gu-a)"/>
    <line x1="140" y1="152" x2="85" y2="78" marker-end="url(#gu-a)"/>
  </g>
  <line x1="140" y1="152" x2="216" y2="95" stroke="#10b981" stroke-width="2.5" marker-end="url(#gu-g)"/>
  <g fill="currentColor" fill-opacity="0.8" font-size="11">
    <text x="243" y="156" text-anchor="start">rate 4</text>
    <text x="140" y="47" text-anchor="middle">rate 3</text>
    <text x="76" y="64" text-anchor="middle">rate 0</text>
  </g>
  <text x="222" y="112" text-anchor="start" fill="#10b981" font-weight="600" font-size="11">&#8711;f direction: rate 5</text>
  <text x="282" y="74" text-anchor="start" fill="currentColor" fill-opacity="0.5" font-size="10">level lines of f</text>
</svg>
</div>

*Candidate steps are the unit directions (dashed circle); here
$\nabla f = [4, 3]$. The rate along $u$ is $\nabla f \cdot u$: 4 along
$[1, 0]$, 3 along $[0, 1]$, $5 = \|\nabla f\|$ along $\nabla f / \|\nabla f\|$,
and 0 perpendicular to $\nabla f$. That is why the level lines of $f$ (faint)
run perpendicular to the gradient.*

#### Three gradient identities cover most of ML

Most ML losses are built from three expressions, and each one's gradient is
the 1-D rule you already know, written with vectors:

| Expression | Gradient | Same rule in 1-D |
|---|---|---|
| $f(x) = a^\top x$ | $\nabla f = a$ | $ax \to a$ |
| $f(x) = \|x\|^2 = x^\top x$ | $\nabla f = 2x$ | $x^2 \to 2x$ |
| $f(x) = x^\top A x$ | $\nabla f = (A + A^\top)x$ | $ax^2 \to 2ax$ |

- **$a^\top x = a_1 x_1 + \dots + a_n x_n$, a weighted sum** (a linear model's
  prediction). Input $x_i$ is multiplied by $a_i$, so changing $x_i$ changes
  $f$ at rate $a_i$, and $\nabla f = a$. For $a = [2, 5]$:
  $f = 2x_1 + 5x_2$ and $\nabla f = [2, 5]$.
- **$\|x\|^2 = x_1^2 + \dots + x_n^2$, the squared length** (every squared
  error and every L2 penalty). Each $x_i^2$ has derivative $2x_i$, so
  $\nabla f = 2x$. At $x = [3, 1]$, $\nabla f = [6, 2]$.
- **$x^\top A x = \sum_{i,j} A_{ij}\, x_i x_j$, a quadratic form** (curvature
  terms such as $w^\top H w$ with the Hessian). Each $x_k$ shows up twice:
  in row $k$ of $A$ (terms $A_{kj} x_k x_j$) and in column $k$ (terms
  $A_{ik} x_i x_k$). Differentiating both gives $(Ax)_k + (A^\top x)_k$, so
  $\nabla f = (A + A^\top)x$. When $A$ is symmetric, as a Hessian or
  covariance matrix is, this is $2Ax$, exactly like $ax^2 \to 2ax$.
  Check with $A = [[1, 2], [3, 4]]$: written out,
  $x^\top A x = x_1^2 + 5 x_1 x_2 + 4 x_2^2$, whose partials are
  $[2 x_1 + 5 x_2,\; 5 x_1 + 8 x_2]$, which is $(A + A^\top)x$ with
  $A + A^\top = [[2, 5], [5, 8]]$.

**Shortcut:** pretend every vector is a single number, use the 1-D rule, then
arrange transposes so the shapes match.

#### Worked example: the linear regression gradient

**The idea:** a weight's gradient adds up, over all examples, (how wrong the
example is) × (that weight's feature value in the example). Examples that are
badly wrong and have a large value of the feature push the weight hardest. In
matrix form, for the squared-error loss $L(w) = \|Xw - y\|^2$:

$$\nabla_w L = 2 X^\top (Xw - y)$$

Here $X$ is the $n \times d$ data matrix (one row per example, one column per
feature), $y$ the $n$ targets, and $w$ the $d$ weights. $Xw - y$ is the vector
of errors, called the **residuals** $r$.

**On numbers.** $X = [[1, 2], [3, 4]]$, $y = [1, 0]$, $w = [1, 1]$. The
predictions are $Xw = [3, 7]$, so $r = [2, 7]$.

- Weight 1's feature column is $[1, 3]$: gradient $2(1 \cdot 2 + 3 \cdot 7) = 46$.
- Weight 2's feature column is $[2, 4]$: gradient $2(2 \cdot 2 + 4 \cdot 7) = 64$.

So $\nabla_w L = [46, 64]$, matching the finite-difference gradient of $L$.
Example 2, with error 7, contributes most of both.

**Why.** Write the loss as a sum over examples, $L = \sum_i r_i^2$, with
$r_i = \sum_j X_{ij} w_j - y_i$. Weight $w_k$ appears in every residual, so it
reaches the loss through $n$ paths ($w_k \to r_i \to L$), and the add rule
(§2.2) sums one chain-rule product per path:

$$\frac{\partial L}{\partial w_k} = \sum_i \underbrace{2 r_i}_{\partial L / \partial r_i} \cdot \underbrace{X_{ik}}_{\partial r_i / \partial w_k}$$

$\partial L / \partial r_i = 2 r_i$ because $r_i^2$ is the only term containing
$r_i$, and $\partial r_i / \partial w_k = X_{ik}$ because that is the
coefficient of $w_k$ in $r_i$. The sum $\sum_i X_{ik}\, r_i$ is column $k$ of
$X$ dotted with $r$, and doing that for every column at once is $X^\top r$.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 480 250" role="img" aria-label="The data matrix X with row i highlighted in green and column k highlighted in indigo: a prediction dots row i with w, a weight's derivative dots column k with r" style="max-width:500px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="xg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
    <marker id="xg-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
  </defs>
  <g fill="currentColor" fill-opacity="0.55" text-anchor="middle" font-size="10.5">
    <text x="92" y="42">feature 1</text>
    <text x="136" y="24">feature k</text>
    <text x="180" y="42">feature d</text>
  </g>
  <g fill="currentColor" fill-opacity="0.55" text-anchor="end" font-size="10.5">
    <text x="58" y="78">ex. 1</text>
    <text x="58" y="114">ex. i</text>
    <text x="58" y="186">ex. n</text>
  </g>
  <path d="M 72 52 L 66 52 L 66 196 L 72 196" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
  <path d="M 200 52 L 206 52 L 206 196 L 200 196" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
  <rect x="114" y="52" width="44" height="144" fill="#6366f1" fill-opacity="0.12" stroke="#6366f1" stroke-opacity="0.4"/>
  <rect x="70" y="88" width="132" height="36" fill="#10b981" fill-opacity="0.12" stroke="#10b981" stroke-opacity="0.4"/>
  <g fill="currentColor" fill-opacity="0.85" text-anchor="middle" font-size="11">
    <text x="136" y="78">X&#8321;&#8342;</text>
    <text x="92" y="114">X&#7522;&#8321;</text>
    <text x="136" y="114">X&#7522;&#8342;</text>
    <text x="136" y="152">&#8942;</text>
    <text x="136" y="186">X&#8345;&#8342;</text>
  </g>
  <line x1="210" y1="106" x2="234" y2="106" stroke="#10b981" stroke-opacity="0.8" marker-end="url(#xg-a)"/>
  <text x="240" y="102" fill="#10b981" font-weight="600" font-size="11">row i &#183; w = prediction for example i</text>
  <text x="240" y="118" fill="currentColor" fill-opacity="0.65" font-size="10.5">one example, all its features</text>
  <line x1="136" y1="200" x2="136" y2="218" stroke="#6366f1" stroke-opacity="0.8" marker-end="url(#xg-b)"/>
  <text x="136" y="234" text-anchor="middle" fill="#6366f1" font-weight="600" font-size="11">column k &#183; r = &#8706;L/&#8706;w&#8342;</text>
  <text x="136" y="248" text-anchor="middle" fill="currentColor" fill-opacity="0.65" font-size="10.5">one feature, across all examples</text>
</svg>
</div>

*The two directions through the same matrix: a prediction dots a **row** of
$X$ (one example) with $w$; a weight's gradient dots a **column** of $X$
(one feature) with $r$.*

That switch from rows to columns is why a transpose appears in backward
formulas: the forward pass uses the rows of $X$, the backward pass uses its
columns, and $X^\top$ turns columns into rows.

**Shape check:** $X^\top$ is $d \times n$ and $r$ is $n \times 1$, so the
gradient is $d \times 1$, the shape of $w$. **The gradient of a scalar loss
always has the same shape as the thing you differentiate with respect to**,
the most useful debugging rule in matrix calculus. Setting $\nabla_w L = 0$
gives the normal equations $X^\top X w = X^\top y$, the closed-form
least-squares solution.

## 2.4 Jacobians: the full sensitivity table

A layer takes a vector in and gives a vector out
($f: \mathbb{R}^n \to \mathbb{R}^m$). Each input can affect each output, so one
number per input is no longer enough. The **Jacobian** $J$ is the $m \times n$
table of all of them: $J_{ij} = \partial f_i / \partial x_j$, how much output
$i$ changes per unit change of input $j$.

#### The Jacobian is the local linear map

**Near a point, a layer acts like multiplication by its Jacobian:**

$$f(x + \delta) \approx f(x) + J\delta$$

This is §2.1's tangent line $f(x+h) \approx f(x) + f'(x)\,h$ for vectors: the
Jacobian is the slope of a vector function.

Why: change the inputs by a small vector $\delta$. Output $i$ gets a
contribution $(\partial f_i / \partial x_j)\,\delta_j$ from each input $j$, and
the contributions add:

$$\Delta f_i \approx \sum_j \frac{\partial f_i}{\partial x_j}\, \delta_j
= (\text{row } i \text{ of } J) \cdot \delta$$

Doing this for every output at once is $J\delta$.

Check: $f(x_1, x_2) = [x_1^2,\; x_1 x_2]$ at $(2, 3)$ has
$J = [[2x_1, 0], [x_2, x_1]] = [[4, 0], [3, 2]]$. For the step
$\delta = [0.01, 0]$, the exact value is $f(2.01, 3) = [4.0401,\; 6.03]$, and
the prediction is $f(2, 3) + J\delta = [4 + 0.04,\; 6 + 0.03] = [4.04,\; 6.03]$.

#### Chain rule for vectors: Jacobians multiply

**The Jacobian of a pipeline is the product of its steps' Jacobians.** For
$z = g(x)$, then $y = f(z)$:

$$J_{f \circ g}(x) = J_f(g(x))\, J_g(x)$$

Here $f \circ g$ names the whole pipeline, "first $g$, then $f$". Why: a small
input change $\delta$ becomes $J_g \delta$ after the first step and
$J_f J_g \delta$ after the second, and applying two matrices in a row is the
same as multiplying them (§1.4 of the linear algebra notes). Each Jacobian is
taken where its step actually operates: $J_g$ at $x$, and $J_f$ at
$z = g(x)$.

#### One layer back in one product: the VJP

**Backprop moves the loss gradient back through a layer by multiplying it by
the layer's Jacobian, without ever building the Jacobian.**

Suppose the loss's sensitivities to the layer's outputs are known, as the row
vector
$s = [\partial L / \partial \text{out}_1, \dots, \partial L / \partial \text{out}_m]$.
Input $i$ reaches the loss through every output:

$$\text{in}_i \to \text{out}_1 \to L, \qquad \text{in}_i \to \text{out}_2 \to L, \qquad \dots, \qquad \text{in}_i \to \text{out}_m \to L$$

so the add rule gives one chain-rule product per path, summed:

$$\frac{\partial L}{\partial \text{in}_i} = \sum_j \frac{\partial L}{\partial \text{out}_j}\,\frac{\partial \text{out}_j}{\partial \text{in}_i} = (s\,J)_i$$

So one layer back is $s \mapsto sJ$, a **vector–Jacobian product** (VJP).
Frameworks compute $sJ$ directly and never store $J$. For a layer of width
4096, $J$ would have $4096 \times 4096 \approx 17$ million entries per
example.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 480 230" role="img" aria-label="One layer drawn as a fully connected graph with two inputs and two outputs: amber edge labels are the Jacobian entries, red numbers at the outputs are the known sensitivities, and each input's sensitivity is the sum over its edges" style="max-width:500px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="vj-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
    <marker id="vj-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.5">
    <rect x="24" y="28" width="150" height="44" rx="7"/>
    <rect x="24" y="140" width="150" height="44" rx="7"/>
    <rect x="330" y="28" width="126" height="44" rx="7"/>
    <rect x="330" y="140" width="126" height="44" rx="7"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="99" y="46">in1</text>
    <text x="99" y="158">in2</text>
    <text x="393" y="46">out1</text>
    <text x="393" y="158">out2</text>
  </g>
  <g fill="#f43f5e" text-anchor="middle" font-size="10.5">
    <text x="99" y="64">&#8706;L/&#8706;in1 = 1&#183;3 + 2&#183;4 = 11</text>
    <text x="99" y="176">&#8706;L/&#8706;in2 = 1&#183;2 + 2&#183;5 = 12</text>
    <text x="393" y="64">&#8706;L/&#8706;out1 = 1</text>
    <text x="393" y="176">&#8706;L/&#8706;out2 = 2</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.6">
    <line x1="174" y1="44" x2="328" y2="44" marker-end="url(#vj-f)"/>
    <line x1="174" y1="58" x2="328" y2="154" marker-end="url(#vj-f)"/>
    <line x1="174" y1="154" x2="328" y2="58" marker-end="url(#vj-f)"/>
    <line x1="174" y1="168" x2="328" y2="168" marker-end="url(#vj-f)"/>
  </g>
  <g font-size="10" fill="#f59e0b" text-anchor="middle">
    <text x="251" y="36">3</text>
    <text x="224" y="80">4</text>
    <text x="224" y="140">2</text>
    <text x="251" y="162">5</text>
  </g>
  <line x1="440" y1="214" x2="60" y2="214" stroke="#f43f5e" marker-end="url(#vj-b)"/>
  <text x="250" y="206" text-anchor="middle" fill="#f43f5e" font-size="10">backward walk</text>
</svg>
</div>

*One layer as a graph: every input touches every output. Amber edge labels
are the local derivatives $\partial \text{out}_j / \partial \text{in}_i$,
the entries of $J$ (here $J = [[3, 2], [4, 5]]$). Red at the outputs: the
known row vector $s = [1, 2]$. Each input's red number is the sum over its
edges, $1 \cdot 3 + 2 \cdot 4 = 11$ for in1, and computing both at once is
$sJ = [11, 12]$, the VJP.*

#### Three Jacobians worth knowing cold

- **ReLU** passes the gradient where the input was positive and zeroes it
  elsewhere. $\text{ReLU}(x)_i = \max(0, x_i)$ depends only on $x_i$, so $J$ is
  diagonal, with 1 where $x_i > 0$ and 0 where $x_i < 0$ (written
  $\mathbb{1}[x_i > 0]$: 1 when the condition holds, 0 otherwise). With
  $x = [0.5, -1.2]$ and $s = [7, 9]$, the backward step gives $[7, 0]$.
- **A linear layer** $f(x) = Wx$ has Jacobian $W$ itself, because output $j$ is
  $\sum_i W_{ji}\, x_i$. The backward step is $s \mapsto sW$, or $W^\top s$ when
  $s$ is written as a column.
- **Softmax**, $p_i = e^{z_i} / \sum_j e^{z_j}$, has a dense Jacobian: every
  output shares the same denominator, so every output depends on every input.
  The entries are $J_{ij} = p_i(\delta_{ij} - p_j)$, where $\delta_{ij}$ is 1 if
  $i = j$ and 0 otherwise (derived in exercise 9). In practice softmax is
  followed by the cross-entropy loss, and frameworks differentiate the two
  together. The backward step then collapses to $p - y$, which avoids both the
  dense Jacobian and the numerically risky $1/p_i$ factor.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 190" role="img" aria-label="Softmax followed by cross-entropy on three classes: forward boxes compute probabilities and the loss; the step-by-step backward path through 1 over p and the dense Jacobian is skipped, replaced by the single fused step p minus y" style="max-width:540px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="sm-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
    <marker id="sm-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e"/></marker>
    <marker id="sm-d" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity="0.35"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.5">
    <rect x="24" y="34" width="130" height="44" rx="7"/>
    <rect x="192" y="34" width="140" height="44" rx="7"/>
    <rect x="370" y="34" width="126" height="44" rx="7"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="89" y="52">logits</text>
    <text x="89" y="70">z = [2, 0, 1]</text>
    <text x="262" y="52">softmax</text>
    <text x="262" y="70">p &#8776; [.67, .09, .24]</text>
    <text x="433" y="52">cross-entropy</text>
    <text x="433" y="70">L = &#8722;log p1 &#8776; .41</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.6">
    <line x1="154" y1="56" x2="190" y2="56" marker-end="url(#sm-f)"/>
    <line x1="332" y1="56" x2="368" y2="56" marker-end="url(#sm-f)"/>
  </g>
  <line x1="470" y1="108" x2="60" y2="108" stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="5 5" marker-end="url(#sm-d)"/>
  <text x="265" y="100" text-anchor="middle" fill="currentColor" fill-opacity="0.5" font-size="10">step-by-step (skipped): multiply by 1/p1, then by the dense 3&#215;3 J</text>
  <line x1="470" y1="152" x2="60" y2="152" stroke="#f43f5e" stroke-width="2" marker-end="url(#sm-b)"/>
  <text x="265" y="144" text-anchor="middle" fill="#f43f5e" font-size="10.5" font-weight="600">fused: &#8706;L/&#8706;z = p &#8722; y &#8776; [&#8722;.33, .09, .24]</text>
</svg>
</div>

*Three classes, logits $z = [2, 0, 1]$, true class 1 so $y = [1, 0, 0]$.
Forward: $p \approx [.67, .09, .24]$, $L = -\log p_1 \approx 0.41$. The
step-by-step backward walk would multiply by $1/p_1$ and then by the dense
$3 \times 3$ Jacobian; the fused step computes
$\partial L / \partial z = p - y \approx [-.33, .09, .24]$ directly, with one
subtraction.*

#### Vanishing and exploding gradients

**The backward signal is multiplied by one Jacobian per layer, so over many
layers it can shrink or grow exponentially.** If each layer scales it by about
0.9, fifty layers scale it by $0.9^{50} \approx 0.005$; at about 1.1, by
$1.1^{50} \approx 117$. Precisely, this happens when the Jacobians' singular
values (§1.6 of the linear algebra notes) are consistently below or above 1.

Residual connections help. A residual layer computes $h + F(h)$, so its
Jacobian is $I + \partial F / \partial h$. The product over layers then always
contains a path through the identity terms, so some gradient reaches every
layer unchanged.
