# Calculus for ML

## 2.1 The derivative is local sensitivity

Forget limits-as-ritual. The derivative answers one concrete question:
**if I change the input by a small amount, how much does the output change?**

$$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad \text{for tiny } h$$

Take $f(x) = x^2$ at $x = 3$: $f(3) = 9$, $f(3.001) = 9.006001$. Output moved by
$0.006001$ for an input change of $0.001$ → sensitivity $\approx 6.001 \approx 6 = 2x$.
The derivative is the **exchange rate between input changes and output changes**.
Equivalently: zoom in far enough and every smooth function is a straight line;
$f'$ is that line's slope: $f(x + h) \approx f(x) + f'(x) h$.

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

*The tangent is what the curve looks like up close. Step $h$ to the right and the
output rises by about $f'(x)\,h$; the tiny gap between tangent and curve is the
$O(h^2)$ error that vanishes as you zoom in.*

This sensitivity reading of the derivative is *the* mental model for training: the
gradient of the loss w.r.t. a weight says "if I increase this weight by $\epsilon$,
the loss changes by (gradient) $\times \epsilon$." Gradient descent moves every
weight in whichever direction lowers the loss, proportional to its sensitivity.

**Numerics aside: two ways to estimate a derivative in code.** The formula at the
top of this section is the **forward** difference $\frac{f(x+h) - f(x)}{h}$. There is a
better one, the **centered** difference $\frac{f(x+h) - f(x-h)}{2h}$, and here is
exactly why it is better.

Taylor's theorem writes $f$ near $x$ as a polynomial in the step $h$:

$$f(x+h) = f(x) + f'(x)\,h + \tfrac{f''(x)}{2} h^2 + \tfrac{f'''(x)}{6} h^3 + \cdots$$
$$f(x-h) = f(x) - f'(x)\,h + \tfrac{f''(x)}{2} h^2 - \tfrac{f'''(x)}{6} h^3 + \cdots$$

Note the sign pattern in the second line: replacing $h$ with $-h$ flips the sign of
every odd power of $h$ and leaves every even power alone.

*Forward difference*: subtract $f(x)$ from the first line and divide by $h$:

$$\frac{f(x+h) - f(x)}{h} = f'(x) + \underbrace{\tfrac{f''(x)}{2} h + \cdots}_{\text{error}}$$

The leading error term is proportional to $h^1$. Shrink $h$ tenfold and the error
shrinks tenfold. That is what "$O(h)$" means.

*Centered difference*: subtract the second line from the first. Every even-power term
($f(x)$, the $h^2$ term, the $h^4$ term, ...) appears with the **same** sign in both
lines, so subtraction cancels it exactly. Every odd-power term appears with opposite
signs, so subtraction doubles it:

$$f(x+h) - f(x-h) = 2 f'(x)\,h + \tfrac{f'''(x)}{3} h^3 + \cdots$$

Divide by $2h$:

$$\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \underbrace{\tfrac{f'''(x)}{6} h^2 + \cdots}_{\text{error}}$$

The $h^1$ error term is gone - it was killed by the cancellation - and the leading
error is now proportional to $h^2$. Shrink $h$ tenfold and the error shrinks a
**hundred**fold. That is "$O(h^2)$".

Check it on $f(x) = x^3$ at $x = 1$, where the true derivative is $3$. With $h = 0.01$:

- forward: $\frac{1.01^3 - 1}{0.01} = 3.0301$, error $0.0301$
  (matches the prediction $\tfrac{f''}{2} h = \tfrac{6}{2}(0.01) = 0.03$)
- centered: $\frac{1.01^3 - 0.99^3}{0.02} = 3.0001$, error $0.0001$
  (matches $\tfrac{f'''}{6} h^2 = \tfrac{6}{6}(0.01)^2 = 0.0001$)

Shrink to $h = 0.001$: the forward error becomes $0.0030$ (10× smaller), the centered
error becomes $0.000001$ (100× smaller). The centered difference is the standard
gradient checker, implemented in `gradients_from_scratch.ipynb`.

## 2.2 The chain rule: differentiating a pipeline

A neural network computes its loss through a pipeline of steps: weights →
layer 1 → layer 2 → ... → loss. Training needs $\partial L / \partial w$ for
every weight $w$, which means differentiating *through* a pipeline. The chain
rule is the rule for that, and it is the only rule backprop uses. This section
builds it on the smallest possible pipeline, then scales it up.

#### The rule

The smallest pipeline has two steps: $z = g(x)$, then $y = f(z)$. Change $x$ by
$\epsilon$. By §2.1, $z$ changes by $g'(x)\,\epsilon$. That change in $z$ then
changes $y$ by $f'(z)$ times it: $f'(z)\,g'(x)\,\epsilon$. So the rate of $y$
per unit change of $x$ is the product of the two step rates:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}$$

**Rates of change multiply through a pipeline.** That's the whole theorem.

#### Run it with numbers

$z = 3x$, $y = z^2$, at $x = 2$ (so $z = 6$):

$$\frac{dz}{dx} = 3, \qquad \frac{dy}{dz} = 2z = 12, \qquad
\frac{dy}{dx} = 12 \cdot 3 = 36$$

Check without the chain rule: $y = (3x)^2 = 9x^2$, so $y' = 18x = 36$ at
$x = 2$. Same answer. ✓

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

#### What dy/dx = 36 means (and what it does not)

Not "step $x$ by 1 and $y$ moves by 36": from $x = 2$ to $x = 3$, $y$ goes from
36 to 81, a change of 45. Write the exact change for a step $h$, calling it
$\Delta y$ ($\Delta$ is read "delta"; $\Delta y$ = the value of $y$ after the
step minus the value before):

$$\Delta y = 9(2+h)^2 - 9 \cdot 2^2 = 36h + 9h^2$$

The derivative accounts for the $36h$ part; the $9h^2$ part is what the linear
approximation of §2.1 drops. At $h = 1$ the dropped part is big: $36 + 9 = 45$.
Shrink the step and it dies faster than the kept part: $h = 0.1$ gives
$\Delta y = 3.69$ vs the predicted $3.6$; $h = 0.01$ gives $0.3609$ vs $0.36$.
So $\Delta y / h \to 36$ as $h \to 0$, and *that* limit is the claim
$\frac{dy}{dx} = 36$. Gradient descent lives on this: the gradient is a
per-unit rate that is only trustworthy for small steps, which is why learning
rates are small.

#### Backprop: the chain rule walked over a whole graph

A network is the same pipeline with more steps and more variables, and
**backprop** is the chain rule applied to it in an organized way:

1. **Forward pass**: compute and cache every intermediate value.
2. **Backward pass**: compute the number $\partial L / \partial v$ for every
   variable $v$ in the graph, walking in reverse order. The walk starts at the
   output, where $\partial L / \partial L = 1$. Each step back is one
   application of the chain rule: if $v$ feeds into $u$, then
   $\partial L / \partial v = (\partial L / \partial u) \cdot (\partial u / \partial v)$.
   The first factor is already known, because $u$ sits later in the graph and
   the walk visited it first (deep learning libraries call it the *upstream
   gradient*). The second factor $\partial u / \partial v$ is the node's
   **local derivative**, the only thing each operation must know how to compute.

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

The pipeline picture misses one case: a variable that feeds into *two* later
steps. Then the chain rule produces one term per use, and the terms add:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial u}\,\frac{\partial u}{\partial w} + \frac{\partial L}{\partial v}\,\frac{\partial v}{\partial w}$$

Check it on $L = w^2 + 3w$, which uses the weight $w$ twice ($u = w^2$ and
$v = 3w$, with $L = u + v$): the path through $u$ contributes $1 \cdot 2w$,
the path through $v$ contributes $1 \cdot 3$, total $2w + 3$. Differentiating
$L$ directly gives the same $2w + 3$.

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

That is all the math in backprop. Every exotic layer - attention, layernorm,
convolutions - just needs its local derivative; the chain rule does the rest.

#### Why walk the graph backward?

The chain rule can be walked in either direction, and the two directions
answer different questions.

- **Forward walk**: pick *one* weight, say $w_1$, and push its rate left to
  right through the graph: compute $\partial v / \partial w_1$ for every
  variable $v$, until $\partial L / \partial w_1$ comes out at the end. One
  walk answers "how does everything respond to $w_1$?" But training needs
  $\partial L / \partial w_i$ for *all* $n$ weights, and this walk delivered
  it for one weight. Getting all of them costs $n$ walks.
- **Backward walk**: start at the loss and walk right to left, computing
  $\partial L / \partial v$ for every variable $v$ (the recipe above). One
  walk answers "how does $L$ respond to everything?" That is exactly the
  question training asks - and the answer for all $n$ weights comes out of
  the *single* walk.

Each walk costs roughly one pass over the graph. So for a model with a billion
weights: a billion forward walks, or one backward walk. That asymmetry - many
inputs, one scalar loss - is why backprop goes backward, and why training
billion-parameter networks is feasible at all.

## 2.3 Gradients of vector/matrix expressions

A model's loss takes all $n$ weights in at once and returns one number:
$L(w_1, \dots, w_n)$. The notation for this shape is
$f : \mathbb{R}^n \to \mathbb{R}$, read "$n$ real numbers in, one real number
out." The backprop graph in §2.2 becomes one of these once the data $x$ is
held fixed: $L = (wx + b)^2$ is then a function of the two weights,
$L(w, b) : \mathbb{R}^2 \to \mathbb{R}$, and the backward walk produced its
gradient, $[42, 14]$.

With $n$ inputs there are $n$ separate questions of the §2.1 kind: hold every
input fixed except $x_i$, change $x_i$ by $\epsilon$, and the output changes by
$(\partial f / \partial x_i)\,\epsilon$. That per-input sensitivity is the
**partial derivative** $\partial f / \partial x_i$. The **gradient** collects
all $n$ of them into a vector:

$$\nabla f = [\partial f / \partial x_1, \dots, \partial f / \partial x_n]$$

The symbol $\nabla$ is read "nabla" (or just "grad"); $\nabla f$ is the
gradient vector of $f$, and it appears in every formula below.

Tiny example: $f(x_1, x_2) = x_1^2 + 3 x_2$ at $(x_1, x_2) = (2, 5)$:
$\partial f / \partial x_1 = 2 x_1 = 4$, $\partial f / \partial x_2 = 3$, so
$\nabla f = [4, 3]$. Compare this with the two-use figure in §2.2: same
formula, but there both terms depended on the *same* variable $w$, so the 4
and the 3 added into one number. Here $x_1$ and $x_2$ are different inputs,
so each keeps its own slot in the vector.

#### The gradient points straight uphill

Stand at a point and take a small step of length $\epsilon$ in some direction.
A direction is a unit vector $u$ (so $\|u\| = 1$); the step changes input $i$
by $\epsilon\, u_i$. Each input contributes its per-input change times its
sensitivity, and the contributions add (§2.2, a variable used in many places),
giving the change in $f$ ($\Delta f$, as in §2.2's $\Delta y$):

$$\Delta f \approx \sum_i \frac{\partial f}{\partial x_i}\, \epsilon\, u_i = \epsilon\, (\nabla f \cdot u)$$

So the direction that raises $f$ fastest is the $u$ that maximizes
$\nabla f \cdot u$. From the linear algebra notes,
$\nabla f \cdot u = \|\nabla f\|\, \|u\| \cos\theta = \|\nabla f\| \cos\theta$,
which is biggest at $\theta = 0$: $u$ pointing along $\nabla f$ itself.

Check it on $\nabla f = [4, 3]$: stepping along the axis $[1, 0]$ raises $f$
at rate $4$; along $[0, 1]$, rate $3$; along
$u = [4, 3] / 5$, rate $4 \cdot \tfrac{4}{5} + 3 \cdot \tfrac{3}{5} = 5 =
\|\nabla f\|$. The blend beats both axes. So the gradient is the locally
steepest uphill direction, and gradient *descent* steps along $-\nabla f$,
straight downhill.

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
$\nabla f = [4, 3]$. By the $\Delta f$ display, the rate along $u$ is
$\nabla f \cdot u$: 4 along $[1, 0]$, 3 along $[0, 1]$,
$5 = \|\nabla f\|$ along $\nabla f / \|\nabla f\|$, and 0 perpendicular to
$\nabla f$ - which is why the level lines of $f$ (faint) run perpendicular to
the gradient.*

#### Three gradient identities cover most of ML

ML losses are built from three recurring expression shapes:

- the **weighted sum** $a^\top x = a_1 x_1 + \dots + a_n x_n$ - every linear
  model's prediction has this form;
- the **quadratic form** $x^\top A x = \sum_{i,j} A_{ij}\, x_i x_j$ - the
  shape of curvature terms like $w^\top H w$ with the Hessian;
- the **squared norm** $\|x\|^2 = x_1^2 + \dots + x_n^2$ - every squared
  error and every L2 regularizer.

Their gradients come from writing out components and taking the partial
derivatives one at a time:

- $f = a_1 x_1 + \dots + a_n x_n$: each $x_i$ appears once, with coefficient
  $a_i$, so $\partial f / \partial x_i = a_i$. Stack them: $\nabla f = a$.
- $f = x_1^2 + \dots + x_n^2$: only the $x_i^2$ term involves $x_i$, so
  $\partial f / \partial x_i = 2 x_i$. Stack them: $\nabla f = 2x$.
- $f = \sum_{i,j} A_{ij}\, x_i x_j$: the variable $x_k$ appears in the terms
  with $i = k$ (a row's worth) and the terms with $j = k$ (a column's worth),
  giving $\partial f / \partial x_k = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i =
  (Ax)_k + (A^\top x)_k$. Stack them: $\nabla f = (A + A^\top) x$.

Numeric check of the third, $A = [[1, 2], [3, 4]]$: written out,
$x^\top A x = x_1^2 + 5 x_1 x_2 + 4 x_2^2$, whose partials are
$[2 x_1 + 5 x_2,\; 5 x_1 + 8 x_2]$ - exactly $(A + A^\top) x$ with
$A + A^\top = [[2, 5], [5, 8]]$. ✓

The summary to memorize:

| Expression | Gradient | Same rule in 1-D |
|---|---|---|
| $f(x) = a^\top x$ | $\nabla f = a$ | $ax \to a$ |
| $f(x) = x^\top A x$ | $\nabla f = (A + A^\top)x$ | $ax^2 \to 2ax$ |
| $f(x) = \|x\|^2 = x^\top x$ | $\nabla f = 2x$ | $x^2 \to 2x$ |

The last column is the memory aid: pretend every vector is a single number,
differentiate with high-school rules, and the matrix version is that answer
with transposes arranged so the shapes work out. ($ax \to a$ means: the 1-D
version of $a^\top x$ is $ax$, whose derivative is $a$.)

#### Worked example: the linear regression gradient

Setup: $X$ is the $n \times d$ data matrix ($n$ examples, $d$ features), $y$
the $n$ targets, $w$ the $d$ weights. Example $i$'s prediction is row $i$ of
$X$ dotted with $w$, its error is the **residual**
$r_i = \sum_j X_{ij} w_j - y_i$, and the loss adds the squared errors:

$$L(w) = \sum_i r_i^2 = \|Xw - y\|^2$$

Differentiate with respect to one weight $w_k$. Every example's prediction
uses $w_k$, so $w_k$ appears in every residual - by §2.2's add rule, the
contributions add across examples, and each contribution is one chain rule
step, (sensitivity of $L$ to $r_i$) times (sensitivity of $r_i$ to $w_k$):

$$\frac{\partial L}{\partial w_k} = \sum_i \frac{\partial L}{\partial r_i}\,\frac{\partial r_i}{\partial w_k} = \sum_i 2 r_i \cdot X_{ik}$$

The two factors: $\partial L / \partial r_i = 2 r_i$ because $r_i^2$ is the
only term of the sum containing $r_i$; and $\partial r_i / \partial w_k =
X_{ik}$ because in $r_i = \sum_j X_{ij} w_j - y_i$ the coefficient of $w_k$
is $X_{ik}$.

Now read $\sum_i X_{ik}\, r_i$ as a matrix product: it dots **column** $k$ of
$X$ with $r$. Forward, each *row* of $X$ acts on $w$ to make a prediction;
backward, each weight collects from its *column*, one entry per example.
Dotting every column of $X$ against a vector is exactly the product
$X^\top r$. Stack the $d$ components:

$$\nabla_w L = 2 X^\top r = 2 X^\top (Xw - y)$$

That column-versus-row switch is the whole reason a transpose appears in
backward formulas.

Tiny check: $X = [[1, 2], [3, 4]]$, $y = [1, 0]$, $w = [1, 1]$. Residuals
$r = [1 + 2 - 1,\; 3 + 4 - 0] = [2, 7]$, and
$2 X^\top r = 2\,[1 \cdot 2 + 3 \cdot 7,\; 2 \cdot 2 + 4 \cdot 7] = [46, 64]$,
matching the finite-difference gradient of $L$.

Shape check: $X^\top$ is $d \times n$, $r$ is $n \times 1$, result
$d \times 1$ = the shape of $w$. ✓ **The gradient of a scalar loss always has
the same shape as the thing you're differentiating with respect to** - the
single most useful debugging rule in matrix calculus. Setting
$\nabla_w L = 0$ gives the normal equations $X^\top X w = X^\top y$, i.e. the
closed-form least-squares solution.

## 2.4 Jacobians: the full sensitivity table

When $f : \mathbb{R}^n \to \mathbb{R}^m$ (vector in, vector out - i.e., a layer), one
number can't describe sensitivity. The **Jacobian** is the $m \times n$ table of all of
them: $J_{ij} = \partial f_i / \partial x_j$ - "how much does output $i$ change when I
change input $j$?" To first order, $f(x + \delta) \approx f(x) + J\delta$: **the
Jacobian is the local linear approximation of the layer** - the matrix that the
function locally *is*. (Full circle to the linear algebra notes: nonlinear functions are locally
linear maps.)

Chain rule for vectors: Jacobians **multiply**: $J_{f \circ g}(x) = J_f(g(x))\, J_g(x)$.
Backprop propagates the row vector $\partial L / \partial \text{output}$ through each
layer by multiplying by that layer's Jacobian (a *vector–Jacobian product* - the
Jacobian itself is usually never materialized; for a hidden layer of width 4096 it
would be 4096², per example). Examples worth knowing cold:

- Elementwise ReLU: Jacobian is diagonal, entries $\mathbb{1}[x_i > 0]$ - backprop
  just masks the upstream gradient.
- Linear layer $f(x) = Wx$: Jacobian w.r.t. $x$ is exactly $W$; the backward
  step is $g \mapsto gW$ with $g$ the upstream row vector (written with
  columns, $g \mapsto W^\top g$ - the same statement transposed).
- Softmax: $J_{ij} = p_i(\delta_{ij} - p_j)$ - dense, which is why softmax is always
  fused with cross-entropy (see exercises: the fused gradient collapses to $p - y$).

Vanishing/exploding gradients, restated precisely: the backward signal is a product of
many Jacobians; if their singular values are systematically $< 1$ or $> 1$, the product
shrinks or blows up exponentially with depth. Residual connections fix this by making
each layer's Jacobian $\approx I + \text{small}$, so products stay near $I$.

