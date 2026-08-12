# Foundations: Linear Algebra, Calculus, Probability - from first principles

The goal of this note: you should be able to *re-derive* everything here on a whiteboard,
not just recite it. Every ML system - a linear model, a transformer, a diffusion model -
is built out of exactly three ingredients:

1. **Linear algebra**: how we *represent* data and transform it (matrices).
2. **Calculus**: how we *improve* the transformation (gradients).
3. **Probability**: how we *justify* the loss function we're improving (likelihood).

If you deeply understand these three, "backprop through a transformer" stops being magic
and becomes bookkeeping.

---

## Part 1: Linear algebra for ML

### 1.1 Vectors are data points

Forget arrows in space for a second. In ML, a vector is just **a list of measurements
about one thing**. A house might be $x = [1400, 3, 1985]$ - square feet, bedrooms,
year built. A word in a transformer is a vector of 768 learned numbers. That's it.

The "arrow" picture is still useful for one reason: it gives us geometry. Once data
points are vectors, we can ask geometric questions: *how long is it? how similar are
two points? what direction does it point?* Those questions turn out to be the entire
substrate of ML.

### 1.2 Dot product = similarity = projection

Take two tiny vectors: $a = [1, 2]$, $b = [3, 1]$.

$$a \cdot b = 1\cdot 3 + 2 \cdot 1 = 5$$

Mechanically: multiply matching entries, add them up. But *why* is this number
meaningful? Two views, both essential:

**View 1: similarity.** The identity (provable from the law of cosines):

$$a \cdot b = \|a\| \, \|b\| \cos\theta$$

where $\theta$ is the angle between them. Hold the lengths fixed and read off the
three regimes: vectors pointing the same way have $\cos\theta = 1$, so the dot
product is as large and positive as it can be; perpendicular vectors have
$\cos\theta = 0$, so the dot product is exactly zero, no agreement at all;
opposite vectors make it negative. So the dot product is an (unnormalized)
**agreement score**. Divide out the lengths and you get **cosine similarity** -
the thing every embedding search engine computes. Attention scores in a
transformer are literally $q \cdot k$: "how much does this query agree with this
key?"

**View 2: projection.** $\frac{a \cdot b}{\|b\|}$ is the length of the shadow $a$ casts
onto the direction of $b$. So a dot product asks: *how much of $a$ lives along $b$'s
direction?* Why not the other way? Both are true: $a \cdot b = b \cdot a$, so the same
number is also "$b$'s shadow on $a$" - just normalized differently
($a \cdot b / \|a\|$ instead of $a \cdot b / \|b\|$). Which reading you pick depends on
which vector you're treating as the *direction* and which as the *data*.
A linear model's prediction $w \cdot x$ is exactly this: project the data
point onto the learned direction $w$; points far along $w$ get big scores.
**One learned direction = one feature detector.** That single sentence explains
what every neuron in a network's linear layer is doing.

Check with numbers: $\|a\| = \sqrt{5} \approx 2.24$, $\|b\| = \sqrt{10} \approx 3.16$,
so $\cos\theta = 5 / (2.24 \cdot 3.16) \approx 0.707$, i.e. $\theta = 45°$. Draw the two
arrows - it's visibly 45°. The formula isn't abstract; it just works.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 320 240" role="img" aria-label="Vectors a=[1,2] and b=[3,1] drawn from the origin with a 45 degree angle between them and the shadow a casts on b" style="max-width:340px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="arr-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="arr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.12">
    <line x1="110" y1="40" x2="110" y2="200"/><line x1="180" y1="40" x2="180" y2="200"/><line x1="250" y1="40" x2="250" y2="200"/>
    <line x1="20" y1="130" x2="300" y2="130"/><line x1="20" y1="60" x2="300" y2="60"/>
  </g>
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="20" y1="200" x2="300" y2="200"/>
    <line x1="40" y1="220" x2="40" y2="40"/>
  </g>
  <g fill="currentColor" fill-opacity="0.55">
    <text x="110" y="214" text-anchor="middle">1</text>
    <text x="180" y="214" text-anchor="middle">2</text>
    <text x="250" y="214" text-anchor="middle">3</text>
    <text x="30" y="134" text-anchor="end">1</text>
    <text x="30" y="64" text-anchor="end">2</text>
  </g>
  <path d="M 85.5 184.8 A 48 48 0 0 0 61.5 157.1" fill="none" stroke="currentColor" stroke-opacity="0.7"/>
  <text x="90" y="163" fill="currentColor">45&#176;</text>
  <line x1="40" y1="200" x2="145" y2="165" stroke="#f59e0b" stroke-width="5" stroke-opacity="0.85"/>
  <line x1="110" y1="60" x2="145" y2="165" stroke="currentColor" stroke-opacity="0.45" stroke-dasharray="4 4"/>
  <path d="M 152.6 162.5 L 150.1 154.9 L 142.5 157.4" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <text x="150" y="186" fill="#f59e0b">shadow &#8776; 1.58</text>
  <line x1="40" y1="200" x2="106" y2="68" stroke="#6366f1" stroke-width="2.5" marker-end="url(#arr-a)"/>
  <line x1="40" y1="200" x2="244" y2="132" stroke="#10b981" stroke-width="2.5" marker-end="url(#arr-b)"/>
  <text x="112" y="52" fill="#6366f1" font-weight="600">a = [1, 2]</text>
  <text x="252" y="126" fill="#10b981" font-weight="600">b = [3, 1]</text>
</svg>
</div>

*Both views in one picture: the arc is the 45° angle (View 1), and the amber
segment is the shadow $a$ casts on $b$'s direction (View 2), with length
$\frac{a \cdot b}{\|b\|} = 5/\sqrt{10} \approx 1.58$.*

### 1.3 Matrix multiply, view 1: a batch of dot products

Take $W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $x = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$.

$$Wx = \begin{bmatrix} 1\cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4\cdot 6\end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

Row 1 of $W$ dotted with $x$, then row 2 of $W$ dotted with $x$. So $Wx$ is just
**many similarity scores computed at once**: each row of $W$ is a feature detector,
and $Wx$ asks "how much does $x$ activate each detector?" A linear layer with 768
inputs and 3072 outputs is 3072 dot products. Stack a batch of inputs as rows of $X$
and $XW^\top$ does *batch × detectors* dot products in one shot. This is why GPUs +
matmuls = deep learning: the whole field runs on "many dot products at once."

### 1.4 Matrix multiply, view 2: a linear map (columns tell you everything)

Same numbers, different lens. Forget the rows for a moment and ask a simpler
question: what does $W$ do to the two easiest possible inputs, the unit axis
vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$?

$$W e_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix} \qquad W e_2 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}2\\4\end{bmatrix} \tag{1.4a}$$

Do the arithmetic and you'll see the pattern: a one-hot input zeroes out every
column except one. **Multiplying $W$ by $e_1$ returns column 1 of $W$;
multiplying by $e_2$ returns column 2.** That's step one.

Step two: any input is just a recipe built from those two ingredients.

$$x = \begin{bmatrix}5\\6\end{bmatrix} = 5\,e_1 + 6\,e_2 \tag{1.4b}$$

Read it as "5 parts $e_1$, 6 parts $e_2$."

Step three: matrix multiplication is **linear**, meaning $W$ processes a recipe
ingredient by ingredient: $W(5\,e_1 + 6\,e_2) = 5\,(W e_1) + 6\,(W e_2)$. We already
know what $W$ does to each ingredient (it returns a column), so substitute:

$$Wx = 5 \begin{bmatrix}1\\3\end{bmatrix} + 6\begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}17\\39\end{bmatrix} \tag{1.4c}$$

Same $[17, 39]$ as the row picture, but now it says something different:
**$Wx$ is a blend of $W$'s columns, and the entries of $x$ are the blend weights.**

Two consequences worth internalizing:

1. **The columns tell you everything.** A matrix is a function: vector in,
   vector out. "$W$ *sends* $e_1$ to $[1,3]$" is just function talk for
   $We_1 = [1,3]$, the way $f(x) = x^2$ sends $3$ to $9$. Once you know where $W$
   sends $e_1$ and $e_2$, linearity pins down where it sends *every* vector. A
   matrix is nothing more than a lookup table of "where do the axes land."
2. **Geometry.** $W$ warps all of space - rotating, stretching, shearing - and the
   columns are literally the warped images of the unit axes: the square spanned by
   $e_1, e_2$ gets mapped to the parallelogram spanned by $[1,3]$ and $[2,4]$.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 240" role="img" aria-label="The unit square spanned by e1 and e2 is mapped by W to the parallelogram spanned by the columns [1,3] and [2,4]" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="p2-c1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="p2-c2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.45">
    <line x1="45" y1="210" x2="210" y2="210"/>
    <line x1="60" y1="225" x2="60" y2="20"/>
  </g>
  <path d="M 60 210 L 86 210 L 86 184 L 60 184 Z" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.55" stroke-dasharray="3 3"/>
  <path d="M 60 210 L 86 132 L 138 28 L 112 106 Z" fill="#6366f1" fill-opacity="0.07" stroke="#6366f1" stroke-opacity="0.35"/>
  <line x1="60" y1="210" x2="84" y2="139" stroke="#6366f1" stroke-width="2.5" marker-end="url(#p2-c1)"/>
  <line x1="60" y1="210" x2="109" y2="112" stroke="#10b981" stroke-width="2.5" marker-end="url(#p2-c2)"/>
  <text x="90" y="224" fill="currentColor" fill-opacity="0.7">e&#8321;</text>
  <text x="42" y="186" fill="currentColor" fill-opacity="0.7">e&#8322;</text>
  <text x="78" y="128" text-anchor="end" fill="#6366f1" font-weight="600">We&#8321; = [1,3]</text>
  <text x="120" y="100" fill="#10b981" font-weight="600">We&#8322; = [2,4]</text>
</svg>
</div>

*The dashed square is the "before" picture: its sides are $e_1$ and $e_2$. Apply
$W$ and each corner moves to its output: $e_1$ ends up at column 1 $= [1,3]$,
$e_2$ at column 2 $= [2,4]$, and the square comes out as the shaded
parallelogram. The squashed, near-flat shape is not a drawing mistake - it is
what this $W$ really does to space, because its two columns point in nearly the
same direction.*

Why hold both views? View 1 (rows = feature detectors) tells you what a single
layer *computes*. View 2 is the one that explains what happens when layers *stack*.

Remember: a matrix is a function. So $A(Bx)$ reads inside-out like any nested
function call: first $B$ warps space, then $A$ warps the result - it's $f(g(x))$
with matrices. And the matrix product $AB$ is *defined* to be the single matrix
that does that two-step warp in one go: $(AB)\,x = A(Bx)$. Matrix multiplication
is not an arbitrary rule about rows and columns; **it is function composition,
written in numbers.** Two facts now come for free:

- **Associativity.** $(AB)C = A(BC)$, because both sides mean the same thing:
  "apply $C$, then $B$, then $A$." Grouping the steps differently doesn't change
  the steps. No index gymnastics required.
- **The collapse.** Stack two linear layers with no nonlinearity between them:
  $W_2(W_1 x) = (W_2 W_1)\,x$. But $W_2 W_1$ is just *one* matrix - so the
  two-layer stack computes nothing a single linear layer couldn't. Depth without
  nonlinearities adds zero expressive power; the nonlinearity between layers is
  what blocks this collapse (a classic interview question).

### 1.5 Eigenvectors and eigenvalues: the directions a matrix doesn't rotate

A matrix warps space: a typical vector comes out rotated *and* stretched. Natural
question: are there special directions the matrix treats more simply - directions
it stretches but does *not* rotate? Think of the grain in a piece of wood: the
material has built-in directions it prefers, and along them it behaves simply.
A vector $v \ne 0$ with

$$A v = \lambda v \tag{1.5a}$$

is such a direction, called an **eigenvector**: multiply it by $A$ and the
result points in the *same direction*, merely scaled by the number $\lambda$ (its **eigenvalue**).
Along an eigenvector, the whole matrix acts like multiplication by one number.

Make it concrete with $A = \begin{bmatrix} 2 & 1 \\ 1 & 2\end{bmatrix}$:

- Try $v = [1, 1]$: $Av = [2 + 1,\ 1 + 2] = [3, 3] = 3v$. Same direction, three
  times longer: eigenvalue $\lambda = 3$.
- Try $v = [1, -1]$: $Av = [2 - 1,\ 1 - 2] = [1, -1] = v$. Completely untouched:
  eigenvalue $\lambda = 1$.
- Try a non-eigenvector, $u = [1, 0]$: $Au = [2, 1]$, which points somewhere
  new. Most vectors are like $u$; eigenvectors are the exceptions.

So this $A$ has a grain: stretch 3× along the diagonal, do nothing along the
anti-diagonal. Notice the two eigen-directions are orthogonal. That's not luck:
**a symmetric matrix always has a full set of orthogonal eigenvectors with real
eigenvalues** (the spectral theorem).

Why eigenvectors are a *payoff*, not just a curiosity. In ordinary coordinates
$A$ mixes its inputs: writing $v = [x, y]$,

$$Av = \begin{bmatrix} 2x + y \\ x + 2y \end{bmatrix} \tag{1.5b}$$

so each output entry needs *both* input entries - the coordinates are coupled,
and it's hard to see what repeated applications of $A$ will do. Now reuse the
recipe argument from §1.4 (equations 1.4b and 1.4c), but with the eigenvectors
as the ingredients instead of $e_1, e_2$. Any vector is some amount of
$v_1 = [1,1]$ plus some amount of $v_2 = [1,-1]$:

$$x = c_1 v_1 + c_2 v_2 \tag{1.5c}$$

and linearity lets $A$ process each ingredient separately - but eigen-ingredients
are the ones $A$ merely scales:

$$Ax = c_1 (A v_1) + c_2 (A v_2) = 3 c_1 v_1 + 1 \cdot c_2 v_2 \tag{1.5d}$$

Read that off: in eigen-coordinates the map is just $(c_1, c_2) \to (3c_1, c_2)$.
Multiply the first coordinate by 3, leave the second alone, done - no mixing
between coordinates at all. *That* is what "$A$ becomes a diagonal matrix in the
eigenbasis" means: same machine, better axes. Along those axes the map is
nothing but an independent stretch $\lambda_i$ per axis, which is why questions
like "what does $A^{100}$ do?" become trivial ($c_1$ gets $3^{100}$, $c_2$ stays
put).

Notice the *shape* of what we just did, because the next section reuses it.
Computing $Ax$ was a three-step procedure:

1. **Find the amounts** $c_1, c_2$: re-describe $x$ in the eigenbasis - the
   split in equation (1.5c).
2. **Scale each amount** by its eigenvalue: $c_1 \to 3c_1$, $c_2 \to c_2$ - the
   middle step of equation (1.5d).
3. **Add the scaled ingredients back up**: the right-hand side of (1.5d),
   $3c_1 v_1 + c_2 v_2$, is exactly that - scaled amounts times their
   eigenvectors, summed back into an ordinary vector.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 220" role="img" aria-label="For A = [[2,1],[1,2]] the eigenvector [1,1] is stretched by 3 without rotating, [1,-1] is unchanged, while a generic vector is rotated" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="p3-a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1" fill-opacity="0.5"/></marker>
    <marker id="p3-v1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="p3-v2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
    <marker id="p3-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.35">
    <line x1="25" y1="150" x2="260" y2="150"/>
    <line x1="70" y1="205" x2="70" y2="25"/>
  </g>
  <line x1="70" y1="150" x2="154" y2="66" stroke="#6366f1" stroke-width="3" stroke-opacity="0.5" marker-end="url(#p3-a1)"/>
  <line x1="70" y1="150" x2="96" y2="124" stroke="#6366f1" stroke-width="2.5" marker-end="url(#p3-v1)"/>
  <line x1="70" y1="150" x2="96" y2="176" stroke="#10b981" stroke-width="2.5" marker-end="url(#p3-v2)"/>
  <line x1="70" y1="150" x2="94" y2="150" stroke="#9ca3af" stroke-width="2" stroke-dasharray="4 4" marker-end="url(#p3-g)"/>
  <line x1="70" y1="150" x2="125" y2="123" stroke="#9ca3af" stroke-width="2" stroke-dasharray="4 4" marker-end="url(#p3-g)"/>
  <text x="94" y="112" text-anchor="end" fill="#6366f1">v = [1,1]</text>
  <text x="162" y="58" fill="#6366f1">Av = [3,3] = 3v &#160;(&#955; = 3)</text>
  <text x="104" y="192" fill="#10b981">Av = v &#160;(&#955; = 1)</text>
  <text x="98" y="166" fill="#9ca3af">u</text>
  <text x="132" y="118" fill="#9ca3af">Au (rotated)</text>
</svg>
</div>

*A generic vector $u$ comes out rotated. The eigenvectors don't: $[1,1]$ triples
in place, $[1,-1]$ is untouched. Eigen-directions are the axes along which $A$ is
just a number.*

Why care in ML:

- **PCA.** Center your data (subtract each feature's mean) and build the
  **covariance matrix** $C$: entry $C_{ij}$ measures how features $i$ and $j$
  move together across the dataset - the average of
  $(\text{feature } i\text{'s deviation}) \times (\text{feature } j\text{'s deviation})$,
  with each feature's variance on the diagonal. It's symmetric for a trivial
  reason: $C_{ij} = C_{ji}$ because the two deviations multiply in either order.
  Symmetric means the spectral theorem applies: $C$ has orthogonal eigenvectors,
  and the ones with the biggest eigenvalues are the directions along which the
  data varies most. PCA is literally "rotate into the eigenbasis, keep only the
  axes with big $\lambda$."
- **Repeated application.** Work in the eigenbasis and apply $A$ over and over:
  each component gets multiplied by its own $\lambda_i$ at every step, so after
  $k$ steps a component has grown (or shrunk) by $\lambda_i^k$. Whatever has
  $|\lambda| > 1$ explodes exponentially and whatever has $|\lambda| < 1$ dies -
  so $A^k v$ ends up pointing along the largest-$|\lambda|$ eigenvector. Where
  this appears in ML: an RNN applies the *same* update rule at every timestep,
  so backprop through $T$ timesteps multiplies the gradient by roughly the same
  matrix $T$ times - literally the $A^k$ story, which is why RNN gradients
  explode or vanish over long sequences. (The per-step matrix is that update's
  *Jacobian* - properly defined in §2.4; for now, "the matrix the update locally
  acts like.")
- **Power iteration and PageRank.** The blow-up above is also an *algorithm*.
  Want the top eigenvector of some matrix $A$? Start from a random vector and
  repeat: multiply by $A$, rescale back to unit length. Every multiply grows the
  component along the biggest-$|\lambda|$ eigenvector faster than all the
  others, so after enough rounds those others are negligible and the vector has
  swung onto the top eigenvector; the rescaling only keeps the numbers from
  overflowing. That loop is **power iteration**. **PageRank** is this idea
  pointed at the web: model a random surfer, where entry $(j, i)$ of the matrix
  is the chance of hopping from page $i$ to page $j$, and define a page's
  importance self-consistently as "the importance flowing into it" - in symbols,
  an importance vector $p$ with $Ap = p$, i.e. an eigenvector. Google's original
  ranking computed $p$ by exactly this keep-multiplying loop.
- **Optimization.** In 1-D, the second derivative is curvature: how fast the
  slope changes as you move. With $n$ weights, curvature depends on which
  direction you move in, and the **Hessian** is the matrix that stores all of
  it: $H_{ij} = \partial^2 L / \partial w_i \partial w_j$, read as "wiggle
  $w_j$, and watch how the slope with respect to $w_i$ changes." Each entry
  involves exactly two weights because a second derivative is two wiggles, one
  per differentiation - but $i$ and $j$ range over *all* $n$ weights, so $H$ is
  the full $n \times n$ table of every pair (the diagonal $H_{ii}$ is the plain
  1-D curvature along weight $i$ alone). At a minimum the
  slope is zero, so curvature is the whole local story - the loss surface is a
  bowl whose shape $H$ describes. $H$ is symmetric (wiggling $w_i$ then $w_j$
  is the same as the reverse), so the spectral theorem applies yet again: it
  has orthogonal eigen-directions, and each eigenvalue is the curvature along
  its direction - big $\lambda$ is a steep wall, small $\lambda$ a nearly flat
  floor. The ratio $\lambda_{max}/\lambda_{min}$ (the **condition number**)
  says how elongated the bowl is, and elongated bowls make gradient descent
  miserable: the gradient points across the narrow valley rather than along it,
  so the iterates zig-zag.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 480 205" role="img" aria-label="Contour plots of two loss bowls: a round bowl where gradient descent goes straight to the minimum, and an elongated bowl where it zig-zags across the narrow valley" style="max-width:500px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="hz-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
    <marker id="hz-z" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e"/></marker>
    <marker id="hz-e" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.25">
    <circle cx="110" cy="105" r="20"/><circle cx="110" cy="105" r="40"/><circle cx="110" cy="105" r="60"/>
    <ellipse cx="350" cy="105" rx="36" ry="9"/><ellipse cx="350" cy="105" rx="73" ry="18"/><ellipse cx="350" cy="105" rx="110" ry="27"/>
  </g>
  <circle cx="110" cy="105" r="2.5" fill="currentColor" fill-opacity="0.7"/>
  <circle cx="350" cy="105" r="2.5" fill="currentColor" fill-opacity="0.7"/>
  <line x1="152" y1="63" x2="117" y2="98" stroke="#10b981" stroke-width="2" marker-end="url(#hz-g)"/>
  <circle cx="143" cy="72" r="2" fill="#10b981"/><circle cx="133" cy="82" r="2" fill="#10b981"/><circle cx="124" cy="91" r="2" fill="#10b981"/>
  <line x1="350" y1="105" x2="350" y2="74" stroke="#6366f1" stroke-width="1.5" marker-end="url(#hz-e)"/>
  <line x1="350" y1="105" x2="276" y2="105" stroke="#6366f1" stroke-width="1.5" marker-end="url(#hz-e)"/>
  <text x="350" y="62" text-anchor="middle" fill="#6366f1">big &#955;: steep</text>
  <text x="290" y="145" text-anchor="middle" fill="#6366f1">small &#955;: flat</text>
  <polyline points="438,92 400,118 372,95 356,112 348,101" fill="none" stroke="#f43f5e" stroke-width="2" marker-end="url(#hz-z)"/>
  <circle cx="438" cy="92" r="2" fill="#f43f5e"/><circle cx="400" cy="118" r="2" fill="#f43f5e"/><circle cx="372" cy="95" r="2" fill="#f43f5e"/><circle cx="356" cy="112" r="2" fill="#f43f5e"/>
  <text x="110" y="195" text-anchor="middle" fill="currentColor" fill-opacity="0.7">condition number &#8776; 1</text>
  <text x="350" y="195" text-anchor="middle" fill="currentColor" fill-opacity="0.7">condition number &#8811; 1</text>
</svg>
</div>

*Contour lines of two loss bowls, seen from above; the dot is the minimum. In
the round bowl the gradient (always perpendicular to the contour you stand on)
points straight at the minimum. In the elongated bowl it points across the
narrow valley almost everywhere, so gradient descent bounces wall to wall while
creeping along the flat direction. This picture is why momentum and Adam exist.*

### 1.6 SVD: every matrix is rotate → stretch → rotate

This section builds to the SVD in three parts.

- **Part A** takes what §1.5 showed - write the input vector as a combination
  of *the matrix's own eigenvectors* (amounts $c_1, c_2$), and multiplying by
  $A$ collapses to multiplying each amount by a number,
  $(c_1, c_2) \to (3c_1, c_2)$ - and turns it into a formula:
  $A = Q \Lambda Q^{-1}$, the **eigendecomposition**.
- **Part B** fixes that formula's weakness: it only exists for square
  (and, for orthogonal axes, symmetric) matrices. The fix is $A = U \Sigma V^\top$, the
  **SVD**, which exists for every matrix.
- **Part C** is the payoff for ML: cut the SVD short to compress a matrix down
  to its few biggest patterns - **low-rank approximation**, the idea inside
  PCA, recommenders, and LoRA.

#### Part A: the eigendecomposition - §1.5 as a formula

Take the §1.5 matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ and a
test vector $x = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$.

§1.5 computed $Ax$ in three steps:

1. **Split $x$ into eigenvectors.** $[3,1] = 2\,[1,1] + 1\,[1,-1]$. The
   amounts: 2 and 1.
2. **Multiply each amount by its eigenvalue.** $2 \to 6$ (times
   $\lambda_1 = 3$), and $1 \to 1$ (times $\lambda_2 = 1$).
3. **Rebuild the vector.** $6\,[1,1] + 1\,[1,-1] = [7,5]$.

Direct check: $Ax = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 7 \\ 5 \end{bmatrix}$.
Same answer.

Each of the three steps is a matrix multiplication. Let
$Q = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$: the eigenvectors,
standing side by side as columns. Then:

- **Step 3 is "multiply by $Q$".** Rule (1.4c) says $Qc = c_1 v_1 + c_2 v_2$:
  give $Q$ the two amounts, and it returns the vector those amounts build.
  Check: $Q \begin{bmatrix} 6 \\ 1 \end{bmatrix} = \begin{bmatrix} 7 \\ 5 \end{bmatrix}$. ✓
- **Step 1 is "multiply by $Q^{-1}$".** Splitting is the reverse of
  rebuilding. Here $Q^{-1} = \tfrac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$.
  Check: $Q^{-1} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$. ✓
- **Step 2 is "multiply by $\Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$".**
  A diagonal matrix multiplies coordinate $i$ by $\lambda_i$, nothing else.
  Check: $\Lambda \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 6 \\ 1 \end{bmatrix}$. ✓

<div style="margin:1.25rem 0">
<svg viewBox="0 0 520 178" role="img" aria-label="Four frames following x=[3,1] through Q inverse, Lambda, and Q: eigen-directions map onto the axes, the stretch happens along the axes, then everything maps back" style="max-width:540px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <defs>
    <marker id="pa-v" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="pa-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.3">
    <line x1="12" y1="120" x2="105" y2="120"/><line x1="18" y1="62" x2="18" y2="158"/>
    <line x1="145" y1="120" x2="238" y2="120"/><line x1="151" y1="62" x2="151" y2="158"/>
    <line x1="278" y1="120" x2="371" y2="120"/><line x1="284" y1="62" x2="284" y2="158"/>
    <line x1="411" y1="120" x2="504" y2="120"/><line x1="417" y1="62" x2="417" y2="158"/>
  </g>
  <g stroke="#10b981" stroke-opacity="0.65" stroke-dasharray="4 3">
    <line x1="18" y1="120" x2="56" y2="82"/><line x1="18" y1="120" x2="56" y2="158"/>
    <line x1="151" y1="120" x2="206" y2="120"/><line x1="151" y1="120" x2="151" y2="65"/>
    <line x1="284" y1="120" x2="339" y2="120"/><line x1="284" y1="120" x2="284" y2="65"/>
    <line x1="417" y1="120" x2="455" y2="82"/><line x1="417" y1="120" x2="455" y2="158"/>
  </g>
  <line x1="18" y1="120" x2="51" y2="109" stroke="#6366f1" stroke-width="2.2" marker-end="url(#pa-v)"/>
  <text x="54" y="103" fill="#6366f1" font-weight="600">x = [3,1]</text>
  <line x1="151" y1="120" x2="173" y2="109" stroke="#6366f1" stroke-width="2.2" marker-end="url(#pa-v)"/>
  <text x="177" y="104" fill="#6366f1" font-weight="600">(2, 1)</text>
  <line x1="284" y1="120" x2="306" y2="109" stroke="#6366f1" stroke-width="2" stroke-opacity="0.3"/>
  <line x1="284" y1="120" x2="350" y2="109" stroke="#6366f1" stroke-width="2.2" marker-end="url(#pa-v)"/>
  <text x="342" y="99" fill="#6366f1" font-weight="600">(6, 1)</text>
  <line x1="417" y1="120" x2="494" y2="65" stroke="#6366f1" stroke-width="2.2" marker-end="url(#pa-v)"/>
  <text x="500" y="57" text-anchor="end" fill="#6366f1" font-weight="600">Ax = [7,5]</text>
  <line x1="109" y1="120" x2="140" y2="120" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#pa-f)"/>
  <text x="124" y="110" text-anchor="middle" fill="currentColor">Q&#8315;&#185;</text>
  <line x1="242" y1="120" x2="273" y2="120" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#pa-f)"/>
  <text x="257" y="110" text-anchor="middle" fill="currentColor">&#923;</text>
  <line x1="375" y1="120" x2="406" y2="120" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#pa-f)"/>
  <text x="390" y="110" text-anchor="middle" fill="currentColor">Q</text>
  <g fill="currentColor" fill-opacity="0.6">
    <text x="58" y="172" text-anchor="middle">1. start</text>
    <text x="191" y="172" text-anchor="middle">2. after Q&#8315;&#185;</text>
    <text x="324" y="172" text-anchor="middle">3. after &#923;</text>
    <text x="457" y="172" text-anchor="middle">4. after Q</text>
  </g>
</svg>
</div>

*The same $x = [3,1]$ walking through equation (1.6a). The dashed green lines
are the eigen-directions. $Q^{-1}$ sends them onto the coordinate axes, and in
that frame $x$'s coordinates are its amounts, $(2,1)$. $\Lambda$ stretches
along the axes: $(2,1) \to (6,1)$, with the old arrow left faint. $Q$ sends
the axes back to the eigen-directions, and the result lands on $Ax = [7,5]$.
Frames 2 and 3 are where the matrix is simple; $Q^{-1}$ only moves the vector
into that frame, and $Q$ moves the result back out.*

So the whole procedure is: multiply by $Q^{-1}$, then $\Lambda$, then $Q$.
Written as one product (rightmost acts first):

$$A = Q \Lambda Q^{-1} \tag{1.6a}$$

This formula is the **eigendecomposition** of $A$. Multiply the three matrices
out and you get $\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ back. It
checks.

(When $A$ is symmetric, the eigenvectors are orthogonal, $Q$ is a pure
rotation, and $Q^{-1} = Q^\top$.)

Two limitations. It needs $A$ **square**: "does the output point the same
direction as the input?" can't even be asked when inputs and outputs live in
spaces of different sizes. And it only comes with clean orthogonal axes when
$A$ is **symmetric**. A real data matrix ($n$ houses × $d$ features) is
neither.

#### Part B: the SVD - eigendecomposition for a rectangular matrix

Part A ended with a problem: formula (1.6a) needs a square matrix, and a real
data matrix is $n \times d$. What is the corresponding decomposition for a
rectangular matrix?

Start with what has to change. In (1.6a), $Q^{-1}$ changed the input's
coordinates and $Q$ changed them back: one matrix and its inverse, usable only
because inputs and outputs lived in the same space. A rectangular $A$ takes
$d$-dimensional inputs to $n$-dimensional outputs. Now the matrix acting on the
input side must be $d \times d$ and the one on the output side must be
$n \times n$ - they cannot be the same matrix, or each other's inverse. So
allow two unrelated rotations, one per side. That is the only change. The
proposed decomposition:

$$A = U \Sigma V^\top \tag{1.6b}$$

- **$V^\top$ rotates the input space.** It turns a set of perpendicular
  input directions $v_1, v_2, \dots$ onto the coordinate axes - the same job
  $Q^{-1}$ did in Part A's four-frame picture.
- **$\Sigma$ does the stretches.** Axis $i$ is scaled by $\sigma_i \ge 0$:
  the **singular values**, conventionally numbered biggest first. (This step also moves from
  $d$ input dimensions to $n$ output dimensions, padding with zeros.)
- **$U$ rotates the output space.** It turns the coordinate axes onto the
  output directions $u_1, u_2, \dots$

$A$ itself is usually none of these three operations. The claim is a
factorization, the way $12 = 2 \cdot 2 \cdot 3$: 12 is not a prime, but it is
a product of primes; $A$ is not a rotation or a stretch, but it is a product of
two rotations and one stretch.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 190" role="img" aria-label="A maps the unit circle with directions v1 and v2 to an ellipse with radii sigma1 u1 and sigma2 u2" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="p4-1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="p4-2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/></marker>
    <marker id="p4-m" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <circle cx="75" cy="120" r="45" fill="#6366f1" fill-opacity="0.05" stroke="currentColor" stroke-opacity="0.55"/>
  <line x1="75" y1="120" x2="109" y2="101" stroke="#6366f1" stroke-width="2.5" marker-end="url(#p4-1)"/>
  <line x1="75" y1="120" x2="56" y2="86" stroke="#10b981" stroke-width="2.5" marker-end="url(#p4-2)"/>
  <text x="115" y="95" fill="#6366f1">v&#8321;</text>
  <text x="40" y="78" fill="#10b981">v&#8322;</text>
  <text x="156" y="106" text-anchor="middle" fill="currentColor" font-weight="600">A = U&#931;V&#7488;</text>
  <line x1="138" y1="120" x2="170" y2="120" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#p4-m)"/>
  <ellipse cx="245" cy="120" rx="58" ry="26" transform="rotate(-20 245 120)" fill="#6366f1" fill-opacity="0.05" stroke="currentColor" stroke-opacity="0.55"/>
  <line x1="245" y1="120" x2="294" y2="102" stroke="#6366f1" stroke-width="2.5" marker-end="url(#p4-1)"/>
  <line x1="245" y1="120" x2="238" y2="101" stroke="#10b981" stroke-width="2.5" marker-end="url(#p4-2)"/>
  <text x="298" y="92" fill="#6366f1">&#963;&#8321;u&#8321;</text>
  <text x="228" y="88" text-anchor="end" fill="#10b981">&#963;&#8322;u&#8322;</text>
</svg>
</div>

*Multiply every point of the unit circle by $A$ and the result is always an
ellipse - and you can see why: a rotation keeps the circle a circle, the
stretch pulls it into an ellipse with radii $\sigma_1, \sigma_2$, and the
last rotation just turns the ellipse into place. The input axes $v_1, v_2$
land on the ellipse's axes $\sigma_1 u_1, \sigma_2 u_2$.*

So far (1.6b) is a proposal. Now find the three factors - which will also show
the decomposition exists for every matrix. The obstacle is that $A$ is
rectangular, so Part A's machinery doesn't apply to it. But it applies to a
matrix built *from* $A$. Consider $A^\top A$:

- **It is square.** The shapes multiply as
  $(d \times n)(n \times d) = d \times d$.
- **It is symmetric.** Transpose it:
  $(A^\top A)^\top = A^\top (A^\top)^\top = A^\top A$. The same matrix
  comes back.

Square and symmetric is exactly what eigendecomposition handles best. Now
substitute the proposal (1.6b) into $A^\top A$, using $U^\top U = I$ (a
rotation followed by its undo changes nothing):

$$A^\top A = (U \Sigma V^\top)^\top (U \Sigma V^\top)
           = V \Sigma^\top U^\top U \Sigma V^\top = V \Sigma^2 V^\top \tag{1.6c}$$

Compare the right side with (1.6a), $Q \Lambda Q^{-1}$, piece by piece:
$Q = V$, $\Lambda = \Sigma^2$, and $Q^{-1} = V^\top$ (for a rotation the
inverse is the transpose). So (1.6c) is an eigendecomposition - of a square
symmetric matrix we can actually compute. That turns the proposal into a
recipe:

1. Form $A^\top A$ and eigendecompose it.
2. Its eigenvectors are the $v_i$. Its eigenvalues are the $\sigma_i^2$; take
   square roots to get the $\sigma_i$.
3. Recover each output direction from $A$ itself: $u_i = A v_i / \sigma_i$.

Run the recipe on the §1.5 matrix, so every symbol gets a number:

1. $A^\top A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$
   (here $A^\top = A$ because $A$ is symmetric).
2. Eigendecompose $\begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$. Multiply
   it by $[1,1]$: the result is $[9, 9] = 9 \cdot [1,1]$. Multiply it by
   $[1,-1]$: the result is $[1,-1] = 1 \cdot [1,-1]$. So its eigenvectors are
   $[1,1]$ and $[1,-1]$, with eigenvalues $9$ and $1$. Normalize the
   eigenvectors to unit length: $v_1 = [1,1]/\sqrt{2}$,
   $v_2 = [1,-1]/\sqrt{2}$. Square-root the eigenvalues:
   $\sigma_1 = \sqrt{9} = 3$, $\sigma_2 = \sqrt{1} = 1$.
3. Output directions:
   $u_1 = A v_1 / \sigma_1 = \left([3,3]/\sqrt{2}\right) / 3 = [1,1]/\sqrt{2} = v_1$,
   and $u_2 = A v_2 / \sigma_2 = \left([1,-1]/\sqrt{2}\right) / 1 = v_2$.

For this symmetric $A$, the recipe returns $u_i = v_i$: each direction lands
on itself. For a non-symmetric matrix, step 3 is exactly where $U$ and $V$
come apart.

Each step works for any matrix, so every matrix has an SVD. **Every linear map
is rotate → stretch → rotate. No exceptions.**

What the factors mean geometrically: $v_1$ is the input direction $A$
stretches the most, $\sigma_1$ is that biggest stretch, and $u_1$ is the
direction it lands on. $v_2$ answers the same question among the directions
perpendicular to $v_1$, and so on. The SVD is a ranking of input directions by
how much $A$ stretches them.

Two quick checks:

- The symmetric $\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ from §1.5
  stretches $[1,1]$ by 3 and $[1,-1]$ by 1 - so those are its $v$'s, they land
  on themselves ($U = V$), and its SVD is just its eigendecomposition.
- The shear $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ looks nothing
  like rotations and stretches, yet it factors too: $\sigma_1 \approx 1.62$,
  $\sigma_2 \approx 0.62$. Slant a deck of cards and you have secretly
  rotated, stretched, and rotated it.

#### Part C: compressing a matrix - why ML cares

A ratings matrix with $n = 10{,}000$ users and $d = 500$ movies holds
$n \times d = 5{,}000{,}000$ numbers. The question of Part C: can far fewer
numbers stand in for almost the same matrix?

**The recipe: keep the top $k$ of the SVD.** Run Part B's recipe, then keep
only the first $k$ columns of $U$, the first $k$ singular values, and the
first $k$ columns of $V$. That stores $k$ columns of length $n$, plus $k$
columns of length $d$, plus $k$ numbers: $k(n + d + 1)$ in total. For the
ratings matrix with $k = 20$: $20 \times (10{,}000 + 500 + 1) = 210{,}020$
numbers in place of $5{,}000{,}000$, about 4%. To use the matrix, multiply the
three shrunken factors back together; the result approximates $A$.

Two questions decide whether this is a good idea. How much does the
approximation lose? And could some other $k$-sized summary lose less?

**What the approximation loses.** Multiplying $U \Sigma V^\top$ out splits
it into one summand per singular value (the algebra is at the end of this
part):

$$A = \sum_{i} \sigma_i \, u_i v_i^\top \tag{1.6d}$$

Look at summand $i$: it is $\sigma_i \, u_i v_i^\top$, built from $u_i$,
$v_i$, and $\sigma_i$ and nothing else - the other triples don't appear in it.
And $u_i v_i^\top$ is a simple object, a column times a row. For example,
$u = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and
$v^\top = \begin{bmatrix} 3 & 1 \end{bmatrix}$:

$$u v^\top = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 3 & 1 \end{bmatrix}
= \begin{bmatrix} 1 \cdot 3 & 1 \cdot 1 \\ 2 \cdot 3 & 2 \cdot 1 \end{bmatrix}
= \begin{bmatrix} 3 & 1 \\ 6 & 2 \end{bmatrix}$$

The same row at different strengths: one pattern per summand. Nothing about
$u$ or $v$ was special here - *any* column times *any* row produces this
one-pattern shape. (A one-pattern matrix is called **rank-1**, a sum of $k$ of
them **rank-$k$** - when ML says "low-rank", it means exactly this.) Since $u_i$ and $v_i$ are unit
vectors, every entry of $u_i v_i^\top$ has size at most 1, so summand $i$'s
magnitude is set by its $\sigma_i$ alone.

Now number the triples so that $\sigma_1 \ge \sigma_2 \ge \dots$ (the
numbering is free to choose; this is the standard convention). Keeping the
first $k$ columns of $U$, $\Sigma$, and $V$ and multiplying gives exactly the
first $k$ summands:

$$A \approx \sum_{i=1}^{k} \sigma_i \, u_i v_i^\top \tag{1.6e}$$

So the recipe throws away precisely the tail
$\sigma_{k+1} u_{k+1} v_{k+1}^\top + \dots$, and every tail summand is
small because its $\sigma$ is small. The approximation loses little whenever
the $\sigma_i$ decay fast, and for real-world matrices they usually do.

Check (1.6d) on the $2 \times 2$ example, with the numbers from Part B's
recipe run ($v_1 = [1,1]/\sqrt{2}$, $v_2 = [1,-1]/\sqrt{2}$, $u_i = v_i$,
$\sigma_1 = 3$, $\sigma_2 = 1$):

$$3 \cdot \tfrac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
+ 1 \cdot \tfrac{1}{2} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
= \begin{bmatrix} 1.5 & 1.5 \\ 1.5 & 1.5 \end{bmatrix}
+ \begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{bmatrix}
= \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

The summands add back to $A$. It checks.

**Could a different summary do better?** No. The **Eckart–Young theorem**
says (1.6e) is the best approximation of $A$ achievable with $k$ patterns:
truncating the SVD is optimal, not merely convenient.

**Where ML uses exactly this:**

- **PCA** is the SVD of the centered data matrix.
- **Classic recommenders**: approximate the user × item ratings matrix as
  (1.6e) with small $k$; each $v_i$ is a taste archetype over items, each
  $u_i$ says how much every user has of that taste. Concretely:

<div style="margin:1.25rem 0">
<svg viewBox="0 0 560 190" role="img" aria-label="A 4 by 4 ratings matrix approximated as an action taste column times row plus a romance taste column times row" style="max-width:580px;width:100%;height:auto;font-size:11px;font-family:inherit">
<rect x="70" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="82" y="61" text-anchor="middle" fill="currentColor">5</text><rect x="94" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="106" y="61" text-anchor="middle" fill="currentColor">4</text><rect x="118" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="130" y="61" text-anchor="middle" fill="currentColor">0</text><rect x="142" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="154" y="61" text-anchor="middle" fill="currentColor">0</text><rect x="70" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="82" y="85" text-anchor="middle" fill="currentColor">0</text><rect x="94" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="106" y="85" text-anchor="middle" fill="currentColor">0</text><rect x="118" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="130" y="85" text-anchor="middle" fill="currentColor">5</text><rect x="142" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="154" y="85" text-anchor="middle" fill="currentColor">4</text><rect x="70" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="82" y="109" text-anchor="middle" fill="currentColor">3</text><rect x="94" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.22" stroke="currentColor" stroke-opacity="0.25"/><text x="106" y="109" text-anchor="middle" fill="currentColor">2.4</text><rect x="118" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="130" y="109" text-anchor="middle" fill="currentColor">4</text><rect x="142" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.29" stroke="currentColor" stroke-opacity="0.25"/><text x="154" y="109" text-anchor="middle" fill="currentColor">3.2</text><rect x="70" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.25"/><text x="82" y="133" text-anchor="middle" fill="currentColor">2</text><rect x="94" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="106" y="133" text-anchor="middle" fill="currentColor">1.6</text><rect x="118" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="130" y="133" text-anchor="middle" fill="currentColor">0</text><rect x="142" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="154" y="133" text-anchor="middle" fill="currentColor">0</text>
<text x="64" y="61" text-anchor="end" fill="currentColor" fill-opacity="0.7">Ana</text>
<text x="64" y="85" text-anchor="end" fill="currentColor" fill-opacity="0.7">Ben</text>
<text x="64" y="109" text-anchor="end" fill="currentColor" fill-opacity="0.7">Cara</text>
<text x="64" y="133" text-anchor="end" fill="currentColor" fill-opacity="0.7">Dan</text>
<text x="82" y="38" fill="currentColor" fill-opacity="0.7" transform="rotate(-40 82 38)">Die Hard</text>
<text x="106" y="38" fill="currentColor" fill-opacity="0.7" transform="rotate(-40 106 38)">Mad Max</text>
<text x="130" y="38" fill="currentColor" fill-opacity="0.7" transform="rotate(-40 130 38)">Titanic</text>
<text x="154" y="38" fill="currentColor" fill-opacity="0.7" transform="rotate(-40 154 38)">Notebook</text>
<text x="118" y="162" text-anchor="middle" fill="currentColor" fill-opacity="0.6">ratings (users &#215; movies)</text>
<text x="196" y="99" text-anchor="middle" fill="currentColor" font-size="15">&#8776;</text>
<rect x="225" y="45" width="24" height="24" fill="#6366f1" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="237" y="61" text-anchor="middle" fill="currentColor">1</text><rect x="225" y="69" width="24" height="24" fill="#6366f1" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="237" y="85" text-anchor="middle" fill="currentColor">0</text><rect x="225" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="237" y="109" text-anchor="middle" fill="currentColor">.6</text><rect x="225" y="117" width="24" height="24" fill="#6366f1" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.25"/><text x="237" y="133" text-anchor="middle" fill="currentColor">.4</text>
<rect x="270" y="81" width="24" height="24" fill="#6366f1" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="282" y="97" text-anchor="middle" fill="currentColor">5</text><rect x="294" y="81" width="24" height="24" fill="#6366f1" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="306" y="97" text-anchor="middle" fill="currentColor">4</text><rect x="318" y="81" width="24" height="24" fill="#6366f1" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="330" y="97" text-anchor="middle" fill="currentColor">0</text><rect x="342" y="81" width="24" height="24" fill="#6366f1" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="354" y="97" text-anchor="middle" fill="currentColor">0</text>
<text x="259" y="98" text-anchor="middle" fill="currentColor">&#215;</text>
<text x="237" y="38" text-anchor="middle" fill="#6366f1">u&#8321;</text>
<text x="318" y="74" text-anchor="middle" fill="#6366f1">v&#8321;&#7488;</text>
<text x="295" y="162" text-anchor="middle" fill="#6366f1" font-weight="600">action taste</text>
<text x="386" y="99" text-anchor="middle" fill="currentColor" font-size="15">+</text>
<rect x="400" y="45" width="24" height="24" fill="#f43f5e" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="412" y="61" text-anchor="middle" fill="currentColor">0</text><rect x="400" y="69" width="24" height="24" fill="#f43f5e" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="412" y="85" text-anchor="middle" fill="currentColor">1</text><rect x="400" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="412" y="109" text-anchor="middle" fill="currentColor">.8</text><rect x="400" y="117" width="24" height="24" fill="#f43f5e" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="412" y="133" text-anchor="middle" fill="currentColor">0</text>
<rect x="445" y="81" width="24" height="24" fill="#f43f5e" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="457" y="97" text-anchor="middle" fill="currentColor">0</text><rect x="469" y="81" width="24" height="24" fill="#f43f5e" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="481" y="97" text-anchor="middle" fill="currentColor">0</text><rect x="493" y="81" width="24" height="24" fill="#f43f5e" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="505" y="97" text-anchor="middle" fill="currentColor">5</text><rect x="517" y="81" width="24" height="24" fill="#f43f5e" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="529" y="97" text-anchor="middle" fill="currentColor">4</text>
<text x="434" y="98" text-anchor="middle" fill="currentColor">&#215;</text>
<text x="412" y="38" text-anchor="middle" fill="#f43f5e">u&#8322;</text>
<text x="493" y="74" text-anchor="middle" fill="#f43f5e">v&#8322;&#7488;</text>
<text x="470" y="162" text-anchor="middle" fill="#f43f5e" font-weight="600">romance taste</text>
</svg>
</div>

  *A $4 \times 4$ ratings matrix built from just two patterns. The row
  $v_1^\top = [5, 4, 0, 0]$ is the action archetype: the ratings a pure
  action fan would give the four movies. The column $u_1$ holds each user's
  amount of that taste. Same for romance. Read off Cara: she is $0.6$ action
  and $0.8$ romance, so her predicted row is
  $0.6 \cdot [5,4,0,0] + 0.8 \cdot [0,0,5,4] = [3, 2.4, 4, 3.2]$ - exactly
  her row in the matrix. (At this toy size there is no saving - two pieces cost
  $2 \times (4+4+1) = 18$ numbers for 16 ratings. The saving is at scale: the
  same two-pattern structure on a $10{,}000 \times 500$ matrix stores
  5 million ratings in $2 \times 10{,}501 = 21{,}002$ numbers.)*
- **Model compression**: replace a trained weight matrix with its truncated
  SVD.
- **LoRA**: assume the *fine-tuning update* $\Delta W$ is low-rank and learn
  it directly as $\Delta W = BA$ with a tiny inner dimension - a few
  patterns' worth of change instead of a whole new matrix.

**Why the product splits into the sum (the promised algebra).** Multiply any
$x$ through the three steps of (1.6b). Step 1 measures the amounts: entry $i$
of $V^\top x$ is $v_i \cdot x$. Step 2 scales amount $i$ by $\sigma_i$.
Step 3 points each scaled amount along its output direction and adds:

$$Ax = \sum_i \sigma_i \, (v_i \cdot x) \, u_i$$

Since $(v_i \cdot x)\, u_i = u_i (v_i^\top x) = (u_i v_i^\top)\, x$, term
$i$ is the summand $\sigma_i u_i v_i^\top$ applied to $x$. Both sides of
(1.6d) produce the same output for every input, so they are the same matrix.

### 1.7 Norms: what "big" means

A norm is a formalized notion of length. The ones that matter:

- **L2**: $\|x\|_2 = \sqrt{\sum_i x_i^2}$. Ordinary Euclidean length. Squared-L2 shows
  up everywhere because it's smooth and its gradient is linear:
  $\nabla \|x\|_2^2 = 2x$. Weight decay penalizes $\|w\|_2^2$, shrinking all weights
  smoothly toward 0.
- **L1**: $\|x\|_1 = \sum_i |x_i|$. Its unit ball is a diamond whose corners sit
  *on the axes* - and a point on an axis is a point where the other coordinates
  are exactly zero. Constrained optima tend to land on those corners, which is
  why L1 regularization produces **sparsity** (lasso). L2's ball is round: no
  corners, so it shrinks weights smoothly but rarely zeros one out.
- **L∞**: $\max_i |x_i|$ - worst-case coordinate; the norm of adversarial-example fame.
- **Frobenius** (matrices): $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$ - L2 with the matrix
  flattened.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 170" role="img" aria-label="Unit balls of the L1 norm (a diamond), L2 norm (a circle), and L-infinity norm (a square)" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <g stroke="currentColor" stroke-opacity="0.25">
    <line x1="15" y1="95" x2="105" y2="95"/><line x1="60" y1="50" x2="60" y2="140"/>
    <line x1="125" y1="95" x2="215" y2="95"/><line x1="170" y1="50" x2="170" y2="140"/>
    <line x1="235" y1="95" x2="325" y2="95"/><line x1="280" y1="50" x2="280" y2="140"/>
  </g>
  <polygon points="60,57 98,95 60,133 22,95" fill="#6366f1" fill-opacity="0.07" stroke="#6366f1" stroke-width="2"/>
  <circle cx="60" cy="57" r="3" fill="#6366f1"/><circle cx="98" cy="95" r="3" fill="#6366f1"/><circle cx="60" cy="133" r="3" fill="#6366f1"/><circle cx="22" cy="95" r="3" fill="#6366f1"/>
  <circle cx="170" cy="95" r="38" fill="#10b981" fill-opacity="0.07" stroke="#10b981" stroke-width="2"/>
  <rect x="242" y="57" width="76" height="76" fill="#f59e0b" fill-opacity="0.07" stroke="#f59e0b" stroke-width="2"/>
  <text x="60" y="158" text-anchor="middle" fill="currentColor" font-weight="600">L1</text>
  <text x="170" y="158" text-anchor="middle" fill="currentColor" font-weight="600">L2</text>
  <text x="280" y="158" text-anchor="middle" fill="currentColor" font-weight="600">L&#8734;</text>
</svg>
</div>

*Unit balls: every point with $\|x\| = 1$. L1's corners (dots) sit exactly on the
axes - constrained optima land there, zeroing the other coordinate. That corner
picture is the geometric reason "L1 selects" while round L2 only shrinks.*

Norms also define distance ($\|a - b\|$) and normalization ($x / \|x\|$, as in
LayerNorm's core idea and cosine similarity). Interview-relevant contrast in one line:
*L2 shrinks, L1 selects.*

---

## Part 2: Calculus for ML

### 2.1 The derivative is local sensitivity

Forget limits-as-ritual. The derivative answers one concrete question:
**if I wiggle the input a tiny bit, how much does the output wiggle?**

$$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad \text{for tiny } h$$

Take $f(x) = x^2$ at $x = 3$: $f(3) = 9$, $f(3.001) = 9.006001$. Output moved by
$0.006001$ for an input wiggle of $0.001$ → sensitivity $\approx 6.001 \approx 6 = 2x$.
The derivative is the **exchange rate between input wiggles and output wiggles**.
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

This wiggle view is *the* mental model for training: the gradient of the loss w.r.t. a
weight says "if I nudge this weight up by $\epsilon$, the loss changes by (gradient)
$\times \epsilon$." Gradient descent just nudges every weight in whichever direction
lowers the loss, proportional to its sensitivity.

Numerics aside: in code, prefer the **centered** difference
$\frac{f(x+h) - f(x-h)}{2h}$ - its error is $O(h^2)$ instead of $O(h)$ (the even-order
error terms cancel; Taylor-expand both sides to see it). This is the standard gradient
checker, implemented in `gradients_from_scratch.ipynb`.

### 2.2 The chain rule is the ONLY thing backprop needs

Compose two functions: $z = g(x)$, then $y = f(z)$. Wiggle $x$ by $\epsilon$; $z$
wiggles by $g'(x)\,\epsilon$; that wiggle in $z$ makes $y$ wiggle by
$f'(z) \cdot g'(x)\, \epsilon$. So:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx}$$

**Wiggle rates multiply through a pipeline.** That's the whole theorem. Concrete:
$y = (3x)^2$ at $x = 2$: inner rate 3, outer rate $2z = 12$, total $36$. Check
directly: $y = 9x^2$, $y' = 18x = 36$. ✓

<div style="margin:1.25rem 0">
<svg viewBox="0 0 340 170" role="img" aria-label="Pipeline x to z to y: forward pass computes values left to right, backward pass multiplies local wiggle rates right to left giving dy/dx = 12 times 3 = 36" style="max-width:360px;width:100%;height:auto;font-size:12px;font-family:inherit">
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
   incoming "wiggle rate of the loss w.r.t. my output" (the upstream gradient) by the
   node's **local derivative**, then pass it on. Where paths merge (a variable used
   twice), **add** the contributions - because independent wiggle paths superpose.

There is no other math in backprop. Every exotic layer - attention, layernorm,
convolutions - just needs its local derivative; the chain rule does the rest.
Why backward and not forward? With many parameters ($n$ inputs, 1 scalar loss),
reverse mode computes all $n$ sensitivities in one backward sweep at roughly the cost
of one forward pass; forward mode would need $n$ sweeps. That asymmetry is the entire
reason deep learning is computationally feasible.

### 2.3 Gradients of vector/matrix expressions

When $f : \mathbb{R}^n \to \mathbb{R}$ (a loss!), there are $n$ inputs you could
wiggle, so there are $n$ sensitivities. The **gradient** just collects them into
a vector: $\nabla f = [\partial f / \partial x_1, \dots, \partial f / \partial x_n]$.

Key fact: $\nabla f$ points in the direction of **steepest ascent**, and the
argument is two lines. Step a tiny unit distance in some direction $u$; by the
wiggle view, $f$ changes by about $\nabla f \cdot u$. Which $u$ makes that change
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
wiggle $\delta$ in $w$ becomes the wiggle $X\delta$ in $r$. Going *backward*,
sensitivity has to ride the same connections in reverse - each entry of $w$
collects contributions from every residual it touched - and summing over the
opposite index of $X$ is exactly what $X^\top$ does. So:

$$\nabla_w L = X^\top \cdot 2r = 2 X^\top (Xw - y)$$

Shape check: $X^\top$ is $d \times n$, $r$ is $n \times 1$, result $d \times 1$ = shape
of $w$. ✓ **The gradient of a scalar loss always has the same shape as the thing you're
differentiating with respect to** - the single most useful debugging rule in matrix
calculus. Setting $\nabla_w L = 0$ gives the normal equations
$X^\top X w = X^\top y$, i.e. the closed-form least-squares solution.

### 2.4 Jacobians: the full wiggle table

When $f : \mathbb{R}^n \to \mathbb{R}^m$ (vector in, vector out - i.e., a layer), one
number can't describe sensitivity. The **Jacobian** is the $m \times n$ table of all of
them: $J_{ij} = \partial f_i / \partial x_j$ - "how much does output $i$ wiggle when I
wiggle input $j$?" To first order, $f(x + \delta) \approx f(x) + J\delta$: **the
Jacobian is the local linear approximation of the layer** - the matrix that the
function locally *is*. (Full circle to Part 1: nonlinear functions are locally linear
maps.)

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

---

## Part 3: Probability & statistics

### 3.1 Random variables, expectation, variance - from first principles

A **random variable** is just a number attached to a random outcome: die roll $X \in
\{1..6\}$, each with probability $1/6$. The **expectation** is the probability-weighted
average - the long-run mean if you repeat forever:

$$\mathbb{E}[X] = \sum_x x \, P(X = x) = \tfrac{1}{6}(1+2+3+4+5+6) = 3.5$$

Two properties you use constantly: **linearity**,
$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$, which holds *even when $X, Y$
are dependent* (it's just rearranging a weighted sum) - this is what makes so many ML
proofs one-liners. And $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ only under
independence.

**Variance** is the expected squared distance from the mean - the natural "spread":

$$\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mu^2$$

(The identity: expand $(X-\mu)^2 = X^2 - 2\mu X + \mu^2$, apply linearity, note
$\mathbb{E}[X] = \mu$.) Die: $\mathbb{E}[X^2] = 91/6 \approx 15.17$, so
$\mathrm{Var} = 15.17 - 12.25 \approx 2.92$. Scaling: $\mathrm{Var}(aX) = a^2
\mathrm{Var}(X)$ (squared units!), and for **independent** variables variances add:
$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$. That last fact is the seed
of the CLT, of $1/\sqrt{n}$ error bars, and of every variance-scaling weight-init
scheme (Xavier/He: pick weight variance so activations keep unit variance layer to
layer).

### 3.2 Bayes rule - derived, not memorized

Conditional probability is a *definition*: restrict attention to the world where $B$
happened, and renormalize:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Write it twice: $P(A \cap B) = P(A \mid B) P(B)$ and also $P(A \cap B) = P(B \mid A)
P(A)$ - same quantity, two factorizations. Set them equal, divide by $P(B)$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

That's the whole derivation - two lines from the definition. The point of Bayes rule is
that it converts **likelihood** ("probability of the evidence given the hypothesis",
often easy to model) into the **posterior** ("probability of the hypothesis given the
evidence", the thing you actually want), weighted by the **prior** $P(A)$.

The classic trap (run it with raw counts, as in `probability_demos.ipynb`): a disease has
prevalence 1%, the test has 99% sensitivity and 95% specificity. You test positive -
what's the chance you're sick? Take 10,000 people: 100 sick → 99 true positives;
9,900 healthy → 5% false positive rate → 495 false positives. Positive tests: $99 +
495 = 594$; sick among them: $99 / 594 \approx 16.7\%$. Not 99%. The prior (rarity of
the disease) dominates because false positives come from a *huge* pool. In ML terms:
this is precision under class imbalance - exactly why accuracy is a useless metric for
rare-event detection and why a 99%-accurate fraud model can be garbage.

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

*The two bars are everyone who tests positive, drawn to scale. The sick pool is
tiny (100 people), so even a 5% false-positive rate on the 9,900 healthy people
buries the 99 true positives.*

### 3.3 MLE, and why minimizing cross-entropy IS maximum likelihood

**Maximum likelihood estimation**: given data and a model family $p_\theta$, choose
the $\theta$ that makes the observed data most probable:

$$\hat\theta = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i \mid x_i)
             = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y_i \mid x_i)$$

(Log is monotonic so the argmax is unchanged; it turns the product into a sum -
numerically stable and differentiable term by term.)

Now the key step. A classifier with softmax output assigns probabilities
$p_\theta(k \mid x_i)$ to each class. The negative log-likelihood of the data is:

$$-\sum_i \log p_\theta(y_i \mid x_i) = -\sum_i \sum_k y_{ik} \log p_\theta(k \mid x_i)$$

where $y_{ik}$ is the one-hot label (the inner sum has one surviving term - the
correct class). The right-hand side is *literally the definition of the cross-entropy
loss*. So: **minimizing cross-entropy = maximizing the log-probability the model
assigns to the true labels.** Cross-entropy isn't an arbitrary "good loss"; it's MLE
for a categorical output. Same story elsewhere: MSE is MLE under a Gaussian noise
model ($-\log$ of a Gaussian density is a squared error plus constants), and L1 loss
is MLE under Laplace noise. When an interviewer asks "why this loss?", the deep answer
is always "it's the negative log-likelihood under this noise/output model."

Simplest concrete MLE (verified numerically in `probability_demos.ipynb`): coin with
unknown $p$, observe $h$ heads in $n$ flips. $\log L(p) = h \log p + (n - h)\log(1-p)$.
Differentiate, set to zero: $h/p = (n-h)/(1-p) \Rightarrow \hat p = h/n$. The empirical
frequency - MLE formalizes the obvious.

### 3.4 Bernoulli and Gaussian - where they come from

**Bernoulli($p$)**: the atom of randomness - one binary event. $P(X=1) = p$,
$\mathbb{E}[X] = p$, $\mathrm{Var} = p(1-p)$ (maximal at $p = 1/2$: a fair coin is the
most unpredictable coin). Sum $n$ independent Bernoullis → **Binomial**. Every
binary-classification output and every dropout mask is a Bernoulli.

**Gaussian** $\mathcal{N}(\mu, \sigma^2)$, density $\frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. It is not chosen for convenience alone; it *emerges*:

1. **CLT** (next section): sums of many small independent effects are Gaussian -
   which is why measurement noise so often is.
2. **Maximum entropy**: among all distributions with a given mean and variance, the
   Gaussian assumes the least extra structure - the honest default.
3. Its $-\log$ density is quadratic - the bridge from "Gaussian noise" to
   "least-squares loss" (§3.3).

### 3.5 Bias and variance of estimators

An estimator $\hat\theta$ is a formula applied to a random sample, so it's a random
variable itself. Two failure modes:

$$\mathrm{Bias} = \mathbb{E}[\hat\theta] - \theta \qquad
  \mathrm{Var} = \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big]$$

Bias: wrong *on average*, a systematic miss. Variance: right on average but
*jumpy* from sample to sample. They combine exactly. The derivation is one move:
inside $\mathbb{E}[(\hat\theta - \theta)^2]$, add and subtract
$\mathbb{E}[\hat\theta]$, expand the square, and the cross term dies because it
contains the factor $\mathbb{E}\big[\hat\theta - \mathbb{E}[\hat\theta]\big] = 0$:

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

Canonical example: sample variance with $\frac{1}{n}$ is biased low - the sample mean
$\bar x$ chases the data, so deviations from $\bar x$ are systematically smaller than
deviations from the true $\mu$; the exact correction factor is $\frac{n-1}{n}$, fixed
by **Bessel's correction** $\frac{1}{n-1}$. The ML version of the tradeoff: a linear
model is high-bias/low-variance (systematically misses curvature, but stable across
datasets); a deep net or unpruned tree is low-bias/high-variance (can fit anything,
including this particular sample's noise). Regularization, more data, and ensembling
(bagging averages away variance) are all moves on this dial. Test error decomposes as
$\text{Bias}^2 + \text{Variance} + \text{irreducible noise}$.

### 3.6 Central limit theorem - the $\sqrt{n}$ intuition

Average $n$ i.i.d. draws with mean $\mu$, variance $\sigma^2$:
$\bar X_n = \frac1n \sum X_i$. Two facts, both from §3.1:

- $\mathbb{E}[\bar X_n] = \mu$ (linearity) - unbiased.
- $\mathrm{Var}(\bar X_n) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$
  (independence → variances add; the $\frac1n$ comes out squared).

So the spread of the mean shrinks like $\sigma/\sqrt{n}$ - the **law of large
numbers** with a rate. To halve your error bar you need 4× the data. The CLT adds the
shape: not only does $\bar X_n$ concentrate, its distribution becomes **Gaussian**,
*regardless of the original distribution*:

$$\frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)$$

Intuition for the shape: summing convolves distributions, and convolution repeatedly
smooths - idiosyncratic features (skew, bumps) wash out, and the Gaussian is the fixed
point of this smoothing (only skew/kurtosis die slowly; mean and variance are
preserved by design of the normalization). Start with something as lopsided as an
exponential distribution: means of 100 draws are already visibly symmetric and bell-
shaped (see `probability_demos.ipynb`, which tracks the skewness → 0 and std →
$\sigma/\sqrt{n}$ numerically).

Why care in ML: minibatch gradients are sample means of per-example gradients (SGD
noise is $\propto 1/\sqrt{\text{batch size}}$); test-set accuracy is a sample mean, so
its error bar is $\sqrt{p(1-p)/n}$ - with $n = 1000$ test examples, accuracy has a
±1.5% standard error, i.e. a "0.5% improvement" is pure noise; A/B tests and
confidence intervals all lean on CLT normality.

---

## Common interview questions

**Q1. Why does the dot product measure similarity?**
Because $a \cdot b = \|a\|\|b\|\cos\theta$: for fixed lengths it's maximal when the
vectors align, zero when orthogonal, negative when opposed. Normalized, it's cosine
similarity; unnormalized, it's the projection of one vector on the other scaled by
its length - "how much of $a$ points along $b$."

**Q2. What happens if you stack linear layers with no nonlinearity?**
$W_2(W_1 x) = (W_2 W_1) x$ - the composition of linear maps is a single linear map, so
depth adds no expressive power. Nonlinearities are what break this collapse.

**Q3. What do eigenvectors mean, in one sentence, and one ML place they appear?**
Directions a matrix scales without rotating ($Av = \lambda v$); the top eigenvectors of
the data covariance matrix are the PCA axes (directions of maximal variance).

**Q4. What's the difference between eigendecomposition and SVD?**
Eigendecomposition needs a square matrix (and orthogonal eigenvectors only if
symmetric); SVD ($U\Sigma V^\top$) exists for *any* rectangular matrix and reads as
rotate–stretch–rotate. They connect via $A^\top A = V \Sigma^2 V^\top$. Truncated SVD
gives the best low-rank approximation (basis of PCA, recommenders, LoRA intuition).

**Q5. Why does L1 regularization give sparsity but L2 doesn't?**
The L1 ball is a diamond with corners on the coordinate axes; the constrained optimum
tends to land on a corner where some coordinates are exactly 0. Equivalently, L1's
gradient is a constant-magnitude push toward 0 that can win at $w=0$, while L2's push
$\propto w$ fades near 0, shrinking weights without zeroing them.

**Q6. Explain backprop in two sentences.**
Forward pass computes and caches all intermediates; backward pass walks the graph in
reverse applying the chain rule - multiply the upstream gradient by each node's local
derivative, summing where paths merge. It's reverse-mode autodiff, chosen because one
backward sweep gets the gradient w.r.t. *all* parameters at roughly the cost of one
forward pass.

**Q7. What is a Jacobian and what's its role in backprop?**
The $m \times n$ matrix of all partials $\partial f_i / \partial x_j$ - the local
linear approximation of a vector-valued function. Backprop propagates gradients via
vector–Jacobian products; the full Jacobian is rarely materialized.

**Q8. Why do gradients vanish or explode in deep/recurrent nets?**
The backward signal is a product of per-layer Jacobians; if their singular values are
consistently below/above 1, the product decays/grows exponentially with depth.
Fixes: residual connections (Jacobian ≈ $I$), careful init, normalization layers,
gradient clipping, gating (LSTM).

**Q9. What is the gradient of $\|Xw - y\|^2$, and what do you get by setting it to 0?**
$\nabla_w = 2X^\top(Xw - y)$; setting it to zero gives the normal equations
$X^\top X w = X^\top y$, the closed-form least-squares solution.

**Q10. Derive Bayes rule.**
From the definition $P(A|B) = P(A\cap B)/P(B)$: the joint factors both ways,
$P(A|B)P(B) = P(B|A)P(A)$; divide by $P(B)$ to get
$P(A|B) = P(B|A)P(A)/P(B)$.

**Q11. Why is minimizing cross-entropy the same as maximum likelihood?**
For a probabilistic classifier, the data's negative log-likelihood is
$-\sum_i \log p_\theta(y_i|x_i)$, which with one-hot labels is exactly the
cross-entropy loss. Minimizing one is minimizing the other. Similarly MSE = MLE under
Gaussian noise, L1 = MLE under Laplace noise.

**Q12. A test is 99% accurate for a disease with 1% prevalence; you test positive.
Roughly what's the chance you're sick?**
Around 17%, not 99% (with 95% specificity): per 10,000 people, 99 true positives vs
495 false positives, so $99/594 \approx 0.167$. The false positives swamp the rare
true positives - same reason accuracy misleads on imbalanced classes.

**Q13. Why divide by $n-1$ in the sample variance?**
Deviations are measured from the sample mean, which itself chases the data, so
$\frac1n\sum(x_i - \bar x)^2$ underestimates $\sigma^2$ by exactly the factor
$\frac{n-1}{n}$; dividing by $n-1$ makes the estimator unbiased (Bessel's correction).

**Q14. Why does averaging reduce noise, and at what rate?**
For i.i.d. terms, variances add while the $1/n$ prefactor comes out squared:
$\mathrm{Var}(\bar X) = \sigma^2/n$, so the standard error shrinks as $1/\sqrt{n}$ -
4× the data for half the error. CLT adds that the distribution of the mean is
approximately Gaussian regardless of the source distribution.

**Q15. Your model improved test accuracy from 91.0% to 91.5% on 1,000 examples. Real?**
Standard error of accuracy is $\sqrt{p(1-p)/n} \approx \sqrt{0.09 \cdot 0.91/1000}
\approx 0.9$–$1.5\%$ per model - the 0.5% gap is within noise. You'd need a bigger
test set, paired comparison (McNemar), or multiple seeds to claim a real improvement.

## Exit checklist

- [ ] Cold, on paper: derive the projection of b onto a and explain dot product as similarity - 3 min
- [ ] State SVD as rotate → stretch → rotate and give the geometric meaning of eigenvectors/eigenvalues - 3 min
- [ ] Write the chain rule for a two-stage composition and give the Jacobian shape for f: R^n → R^m - 3 min
- [ ] Derive Bayes rule from the definition of conditional probability, then apply it to a base-rate problem - 4 min
- [ ] Show that minimizing cross-entropy is maximum likelihood for a Bernoulli model - 5 min
- [ ] Define bias and variance of an estimator and explain why the 1/n variance estimator is biased - 4 min
- [ ] Use standard error to decide whether a 0.5% accuracy gap between two models on n=1000 is real - 4 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
