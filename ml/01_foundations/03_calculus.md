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

## 2.2 The chain rule is the ONLY thing backprop needs

Compose two functions: $z = g(x)$, then $y = f(z)$. Change $x$ by $\epsilon$; then
$z$ changes by $g'(x)\,\epsilon$; that change in $z$ in turn changes $y$ by
$f'(z) \cdot g'(x)\, \epsilon$. So:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}$$

**Rates of change multiply through a pipeline.** That's the whole theorem.

Run it once with numbers: $z = 3x$, $y = z^2$, at $x = 2$ (so $z = 6$):

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
rates right to left. That right-to-left sweep, applied to a whole graph, is
backprop.*

Backprop is *nothing but* this, applied to a graph and organized efficiently:

1. **Forward pass**: compute and cache every intermediate value.
2. **Backward pass**: walk the graph in reverse, and at each node multiply the
   incoming "rate of change of the loss w.r.t. my output" (the upstream gradient) by
   the node's **local derivative**, then pass it on. Where paths merge (a variable used
   twice), **add** the contributions - to first order, changes arriving through
   separate paths add.

There is no other math in backprop. Every exotic layer - attention, layernorm,
convolutions - just needs its local derivative; the chain rule does the rest.
Why backward and not forward? With many parameters ($n$ inputs, 1 scalar loss),
reverse mode computes all $n$ sensitivities in one backward sweep at roughly the cost
of one forward pass; forward mode would need $n$ sweeps. That asymmetry is the entire
reason deep learning is computationally feasible.

## 2.3 Gradients of vector/matrix expressions

When $f : \mathbb{R}^n \to \mathbb{R}$ (a loss!), there are $n$ inputs you could
change, so there are $n$ sensitivities. The **gradient** just collects them into
a vector: $\nabla f = [\partial f / \partial x_1, \dots, \partial f / \partial x_n]$.

Key fact: $\nabla f$ points in the direction of **steepest ascent**, and the
argument is two lines. Step a tiny unit distance in some direction $u$; to first
order (§2.1), $f$ changes by about $\nabla f \cdot u$. Which $u$ makes that change
biggest? $\nabla f \cdot u = \|\nabla f\| \cos\theta$, maximized at $\theta = 0$:
when $u$ points along $\nabla f$ itself (the dot product as agreement score,
again). So the gradient is the locally-uphill direction, and gradient *descent*
steps along $-\nabla f$, straight downhill.

The three identities that cover 90% of ML derivations (all provable by writing out
components - do it once by hand):

| Expression | Gradient | Sanity check (1-D) |
|---|---|---|
| $f(x) = a^\top x$ | $\nabla f = a$ | $ax \to a$ |
| $f(x) = x^\top A x$ | $\nabla f = (A + A^\top)x$ | $ax^2 \to 2ax$ |
| $f(x) = \|x\|^2 = x^\top x$ | $\nabla f = 2x$ | $x^2 \to 2x$ |

The 1-D sanity check column is the trick: matrix calculus identities are the scalar
rules with transposes arranged so shapes work out. Worked example - linear regression,
$L(w) = \|Xw - y\|^2$. Let $r = Xw - y$ (the residuals). Chain rule in two hops:
the sensitivity of $L$ to $r$ is $2r$, and $r$ depends on $w$ through the linear
map $X$. Why does a transpose appear when we combine them? Going *forward*, a
change $\delta$ in $w$ becomes the change $X\delta$ in $r$. Going *backward*,
sensitivity has to ride the same connections in reverse - each entry of $w$
collects contributions from every residual it touched - and summing over the
opposite index of $X$ is exactly what $X^\top$ does. So:

$$\nabla_w L = X^\top \cdot 2r = 2 X^\top (Xw - y)$$

Shape check: $X^\top$ is $d \times n$, $r$ is $n \times 1$, result $d \times 1$ = shape
of $w$. ✓ **The gradient of a scalar loss always has the same shape as the thing you're
differentiating with respect to** - the single most useful debugging rule in matrix
calculus. Setting $\nabla_w L = 0$ gives the normal equations
$X^\top X w = X^\top y$, i.e. the closed-form least-squares solution.

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
- Linear layer $f(x) = Wx$: Jacobian w.r.t. $x$ is exactly $W$; backprop multiplies
  the upstream gradient by $W^\top$.
- Softmax: $J_{ij} = p_i(\delta_{ij} - p_j)$ - dense, which is why softmax is always
  fused with cross-entropy (see exercises: the fused gradient collapses to $p - y$).

Vanishing/exploding gradients, restated precisely: the backward signal is a product of
many Jacobians; if their singular values are systematically $< 1$ or $> 1$, the product
shrinks or blows up exponentially with depth. Residual connections fix this by making
each layer's Jacobian $\approx I + \text{small}$, so products stay near $I$.

