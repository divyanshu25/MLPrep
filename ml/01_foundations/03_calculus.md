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

- A **forward walk** (forward-mode autodiff) picks one weight $w_1$ and
  pushes its effect left to right, computing $\partial v / \partial w_1$ for every variable $v$. At the
  end it has $\partial L / \partial w_1$. That is one weight per walk, so $n$
  weights take $n$ walks.
- A **backward walk** (reverse-mode autodiff, which is backprop) starts at
  the loss and computes $\partial L / \partial v$ for every variable $v$ (the
  recipe above). One walk gives the derivative for all $n$ weights.

Each walk costs a small multiple of one forward pass (a backward walk is
about 2 to 3 times the forward pass's arithmetic). For a billion weights,
that is a billion forward walks or one backward walk. The price of the
backward walk is memory: it needs every intermediate value from the forward
pass, so those values are kept until the backward walk uses them.

The general rule: forward mode costs one walk per input, reverse mode one
walk per output. A loss has one output and billions of inputs, so training
uses reverse mode. A function with few inputs and many outputs is the
opposite case, and there forward mode is cheaper.

## 2.3 Gradients of vector/matrix expressions

A loss depends on many weights at once; call them $x_1, \dots, x_n$. The
**gradient** is the list of how fast the loss changes when you change each
weight on its own:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]$$

Each entry is a **partial derivative**: the ordinary derivative of §2.1,
taken with only $x_i$ changing. $\nabla$ is read "grad".

Example: $f(x_1, x_2) = x_1^2 + 3x_2$ at $(2, 5)$, where $f = 19$.

- Change $x_1$ alone: $x_1^2$ changes at rate $2x_1 = 4$.
- Change $x_2$ alone: $3x_2$ changes at rate 3.

So $\nabla f = [4, 3]$. (The backward walk in §2.2 also produced a gradient:
$[42, 14]$ for the two weights $w$ and $b$.)

#### The gradient points straight uphill

**The gradient points in the direction where $f$ increases fastest, and its
length is that fastest rate.** That is why gradient descent steps along
$-\nabla f$: it is the steepest way down.

**Any small step.** With $\nabla f = [4, 3]$, $f$ rises about 4 per unit of
$x_1$ and 3 per unit of $x_2$. Move $x_1$ by $0.1$ and $x_2$ by $0.2$ together:
the first move adds about $0.4$, the second about $0.6$, and for small steps
the effects add, so $f$ rises about $1.0$ (exactly: from 19 to 20.01). Each
rate times its input's change, summed, is a dot product:

$$\Delta f \approx \nabla f \cdot \text{step} \qquad \text{here } [4, 3] \cdot [0.1, 0.2] = 1.0$$

**The best direction.** To compare directions fairly, make every step the
same small length $\epsilon$: step $= \epsilon\, u$, where $u$ is a unit vector
($\|u\| = 1$) that only says which way. Then $f$ rises at rate
$\nabla f \cdot u$. By the dot-product identity (§1.2 of the linear algebra
notes), $\nabla f \cdot u = \|\nabla f\| \cos\theta$, where $\theta$ is the
angle between $u$ and $\nabla f$. That is largest at $\theta = 0$, when $u$
points along $\nabla f$, and the rate there is $\|\nabla f\|$.

With $\nabla f = [4, 3]$: along $[1, 0]$ the rate is 4, along $[0, 1]$ it is
3, and along $[4, 3]/5$ it is $4 \cdot \tfrac{4}{5} + 3 \cdot \tfrac{3}{5} = 5$,
faster than either axis.

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

Most losses are built from three expressions. Each one's gradient is a 1-D
rule you already know, written with vectors:

| Expression | Gradient | 1-D version |
|---|---|---|
| $a^\top x$ | $a$ | $ax \to a$ |
| $\|x\|^2$ | $2x$ | $x^2 \to 2x$ |
| $x^\top A x$, $A$ symmetric | $2Ax$ | $ax^2 \to 2ax$ |

- **$a^\top x$, a weighted sum** (a linear model's prediction). For
  $f = 2x_1 + 5x_2$, raising $x_1$ by 1 adds 2 and raising $x_2$ by 1 adds 5,
  so $\nabla f = [2, 5] = a$, the same at every point.
- **$\|x\|^2$, a squared length** (every squared error and L2 penalty). Each
  $x_i^2$ changes at rate $2x_i$, so $\nabla f = 2x$. At $x = [3, 1]$,
  $\nabla f = [6, 2]$: bigger entries change $f$ faster.
- **$x^\top A x$, a quadratic form** (curvature terms such as $w^\top H w$ with
  the Hessian). For $A = [[1, 2], [3, 4]]$,
  $x^\top A x = x_1^2 + 5 x_1 x_2 + 4 x_2^2$, with partials
  $[2x_1 + 5x_2,\; 5x_1 + 8x_2] = (A + A^\top)x$. For a symmetric $A$, such
  as a Hessian or covariance matrix, that is $2Ax$.

**Shortcut:** treat every vector as a single number, use the 1-D rule, then
arrange transposes so the shapes match.

#### The linear regression gradient

$$L(w) = \|Xw - y\|^2 \qquad \nabla_w L = 2 X^\top r, \quad r = Xw - y$$

$X$ has one row per example and one column per feature, and $r$ is the
vector of errors (the **residuals**).

**Per weight:** $\partial L / \partial w_k = 2 \sum_i r_i X_{ik}$, each
example's error times its value of feature $k$, summed. This is the chain
rule on each $r_i^2$: $r_i$ changes at rate $X_{ik}$ when $w_k$ changes, and
$r_i^2$ at rate $2 r_i$ when $r_i$ changes.

**Why $X^\top$:** feature $k$'s values are column $k$ of $X$, so the sum is
column $k$ dotted with $r$. Every column at once is $X^\top r$.

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
  <text x="136" y="234" text-anchor="middle" fill="#6366f1" font-weight="600" font-size="11">2 (column k &#183; r) = &#8706;L/&#8706;w&#8342;</text>
  <text x="136" y="248" text-anchor="middle" fill="currentColor" fill-opacity="0.65" font-size="10.5">one feature, across all examples</text>
</svg>
</div>

*A prediction dots a **row** of $X$ with $w$; a weight's gradient dots a
**column** of $X$ with $r$. That switch from rows to columns is why
transposes appear in backward formulas.*

**Shape check:** the gradient has the same shape as $w$, one entry per
weight. This holds for every gradient and is the most useful debugging rule
in matrix calculus.

**Setting it to zero** gives the normal equations $X^\top X w = X^\top y$,
solved by $w = (X^\top X)^{-1} X^\top y$ when the columns of $X$ are linearly
independent.

- **Linearly independent columns:** no column can be built from the
  other columns (for example, no column is a copy of another or the sum of
  two others).
- **Why it is needed:** say the data has height recorded twice, as feature
  1 and feature 2, and the best prediction is $2 \times$ height. Then
  $w = [2, 0]$, $[0, 2]$, $[1, 1]$ and $[5, -3]$ all predict exactly
  $2 \times$ height, so the data cannot pick one of them. With no single
  best $w$, the formula breaks: $X^\top X$ has no inverse.

## 2.4 Jacobians: the full sensitivity table

A layer has several inputs and several outputs, so one rate per input is no
longer enough. The **Jacobian** $J$ is the table of every output's rate with
respect to every input, one row per output and one column per input:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

Example: $f(x_1, x_2) = [x_1^2,\; x_1 x_2]$ at $(2, 3)$:

$$J = \begin{bmatrix} 2x_1 & 0 \\ x_2 & x_1 \end{bmatrix} = \begin{bmatrix} 4 & 0 \\ 3 & 2 \end{bmatrix}$$

Row $i$ is the gradient (§2.3) of output $i$. The 0 is there because output
1, $x_1^2$, does not depend on $x_2$.

#### The Jacobian is the local linear map

$J$ turns a small input change $\delta$ into the output change:

$$f(x + \delta) \approx f(x) + J\delta$$

Output $i$ changes by its gradient dotted with $\delta$ (§2.3), and that
gradient is row $i$ of $J$. This is §2.1's tangent line with $J$ as the
slope.

Check: from $(2, 3)$ with $\delta = [0.01, 0]$,

$$f(x) + J\delta = \begin{bmatrix} 4 \\ 6 \end{bmatrix} + \begin{bmatrix} 4 & 0 \\ 3 & 2 \end{bmatrix} \begin{bmatrix} 0.01 \\ 0 \end{bmatrix} = \begin{bmatrix} 4.04 \\ 6.03 \end{bmatrix}, \qquad \text{exact: } f(2.01, 3) = \begin{bmatrix} 4.0401 \\ 6.03 \end{bmatrix}$$

#### Chain rule for vectors: Jacobians multiply

For $z = g(x)$ followed by $y = f(z)$:

$$J_{f \circ g}(x) = J_f(z)\, J_g(x)$$

A small change $\delta$ becomes $J_g \delta$ after the first layer and
$J_f J_g \delta$ after the second. The later layer goes on the left, and
each Jacobian is taken at its own layer's input. This is §2.2's "rates
multiply", with matrices.

#### One layer back: the VJP

Backprop arrives at a layer knowing how the loss responds to each output:
the row vector $s$ with $s_j = \partial L / \partial \text{out}_j$. An input
affects the loss through every output it feeds, so add one term per output:

$$\frac{\partial L}{\partial \text{in}_i} = \sum_j s_j\, J_{ji} = (sJ)_i$$

Example: $J = [[3, 2], [4, 5]]$ and $s = [1, 2]$ give
$sJ = [1 \cdot 3 + 2 \cdot 4,\; 1 \cdot 2 + 2 \cdot 5] = [11, 12]$.

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

*The same example as a graph. Amber edge labels are the entries of $J$; red
at the outputs is $s = [1, 2]$. Each input's red number is the sum over its
edges: $1 \cdot 3 + 2 \cdot 4 = 11$ for in1.*

$sJ$ is a **vector–Jacobian product** (VJP). Frameworks compute it directly
and never build $J$: for a layer of width 4096, $J$ would hold 17 million
entries per example.

The mirror image, $J\delta$, is a **Jacobian–vector product** (JVP), used by
forward mode. One VJP gives one output's sensitivity to every input; one JVP
gives every output's response to one input direction. This is §2.2's rule of
one walk per output versus one walk per input.

#### Three Jacobians worth knowing cold

- **ReLU:** $J$ is diagonal, 1 where $x_i > 0$ and 0 elsewhere (frameworks
  use 0 at exactly 0). The gradient passes where the input was positive and
  is blocked elsewhere: $x = [0.5, -1.2]$ and $s = [7, 9]$ give $[7, 0]$.
- **Linear layer** $f(x) = Wx$: $J = W$, so the backward step is $sW$, or
  $W^\top s$ with $s$ as a column.
- **Softmax** $p_i = e^{z_i} / \sum_j e^{z_j}$: every output shares the
  denominator, so $J$ has no zeros: $J_{ij} = p_i(\delta_{ij} - p_j)$, where
  $\delta_{ij}$ is 1 if $i = j$ and 0 otherwise (exercise 9). Followed by
  cross-entropy, $L = -\log p_c = -z_c + \log \sum_j e^{z_j}$ for true class
  $c$, the combined gradient is just $\partial L / \partial z = p - y$, with
  $y$ the one-hot target. Frameworks use this fused step: it skips the dense
  $J$ and the $1/p_c$ factor, which is infinite when $p_c$ rounds to 0.

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

*Three classes, logits $z = [2, 0, 1]$, true class 1, so $y = [1, 0, 0]$.
Forward: $p \approx [.67, .09, .24]$ and $L = -\log p_1 \approx 0.41$.
Backward: the fused step gives $p - y \approx [-.33, .09, .24]$ in one
subtraction, instead of multiplying by $1/p_1$ and then the dense
$3 \times 3$ Jacobian.*

#### Computing softmax without overflow: log-sum-exp

Softmax of $z = [1000, 999]$ needs $e^{1000}$, but 64-bit floats top out
near $e^{709}$ (32-bit near $e^{88}$), so it overflows and gives NaN. The
fix: subtracting the same $c$ from every logit leaves softmax unchanged,
because $e^{-c}$ cancels top and bottom:

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Pick $c = \max_j z_j$, so the largest exponent is $e^0 = 1$. Here
$z - c = [0, -1]$ and $p \approx [0.731, 0.269]$.

For the loss, compute $\log p$ directly instead of taking the log of $p$: a
logit 745 below the max (103 in 32-bit) gives $p = 0$, and $\log 0 = -\infty$.
With $m = \max_j z_j$:

$$\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \qquad \log p_i = z_i - \log \sum_j e^{z_j}$$

This is **log-sum-exp**, and every cross-entropy implementation uses it. For
$z = [1000, 999]$ it gives $1000 + \log(1 + e^{-1}) \approx 1000.313$.

#### Vanishing and exploding gradients

On the way back the gradient is multiplied by one Jacobian per layer, so it
can shrink or grow exponentially with depth: $0.9^{50} \approx 0.005$ and
$1.1^{50} \approx 117$. This happens when the Jacobians' singular values
(§1.6 of the linear algebra notes) stay below or above 1.

Residual connections help. A residual layer computes $h + F(h)$, with
Jacobian $I + \partial F / \partial h$. Multiplying these over many layers
gives one term that is just $I$: a direct path from the top gradient to every
layer. The other terms can still grow large, which is why deep residual
networks scale each $F$ down at initialization.
