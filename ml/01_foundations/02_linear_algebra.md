# Linear algebra for ML

## 1.1 Vectors are data points

In ML, a vector is **a list of measurements
about one thing**. A house might be $x = [1400, 3, 1985]$ - square feet, bedrooms,
year built. A word in a transformer is a vector of 768 learned numbers.

The "arrow" picture is still useful for one reason: it gives us geometry. Once data
points are vectors, we can ask geometric questions: *how long is it? how similar are
two points? what direction does it point?* Length, similarity and direction are the
operations the rest of this chapter builds on.

## 1.2 Dot product = similarity = projection

Take two tiny vectors: $a = [1, 2]$, $b = [3, 1]$.

$$a \cdot b = 1\cdot 3 + 2 \cdot 1 = 5$$

Mechanically: multiply matching entries, add them up. But *why* is this number
meaningful? Two views, both essential:

**View 1: similarity.** The identity (provable from the law of cosines):

$$a \cdot b = \|a\| \, \|b\| \cos\theta$$

where $\theta$ is the angle between them. Hold the two lengths constant;
then the dot product depends only on $\cos\theta$, and there are three
cases:

- **Same direction** ($\theta = 0$): $\cos\theta = 1$, so the dot product
  is as large and positive as the two lengths allow.
- **Perpendicular** ($\theta = 90°$): $\cos\theta = 0$, so the dot product
  is exactly zero. No agreement at all.
- **Opposite directions** ($\theta = 180°$): $\cos\theta = -1$, so the dot
  product is as negative as possible.

So the dot product is an (unnormalized) **agreement score**. Divide out the
lengths and you get **cosine similarity** -
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
arrows - it's visibly 45°.

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

## 1.3 Matrix multiply, view 1: a batch of dot products

Take $W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $x = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$.

$$Wx = \begin{bmatrix} 1\cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4\cdot 6\end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

Row 1 of $W$ dotted with $x$, then row 2 of $W$ dotted with $x$. So $Wx$ is just
**many similarity scores computed at once**: each row of $W$ is a feature detector,
and $Wx$ asks "how much does $x$ activate each detector?" A linear layer with 768
inputs and 3072 outputs is 3072 dot products. Stack a batch of inputs as rows of $X$
and $XW^\top$ computes *batch × detectors* dot products in a single matrix
multiplication. This is why deep learning runs on GPUs: a GPU's specialty is
computing many dot products at once, and that is nearly all a network does.

## 1.4 Matrix multiply, view 2: a linear map (columns tell you everything)

Take the same $W$ and $x$, and ask a different
question: what does $W$ do to the two simplest possible inputs, the unit axis
vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$?

$$W e_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix} \qquad W e_2 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}2\\4\end{bmatrix} \tag{1.4a}$$

Step one: a one-hot input zeroes out every column except one, so
**multiplying $W$ by $e_1$ returns column 1 of $W$, and multiplying by
$e_2$ returns column 2.**

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
   vector out. Once you know $We_1$ and $We_2$, linearity pins down the
   output for *every* input. A matrix is nothing more than a lookup table of
   "where do the axes land."
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

*The unit square (dashed) after $W$: $e_1$ lands on column 1 $= [1,3]$,
$e_2$ on column 2 $= [2,4]$. The near-flat shape is not a mistake - $W$'s
columns really do point in nearly the same direction.*

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
  the steps.
- **The collapse.** Stack two linear layers with no nonlinearity between them:
  $W_2(W_1 x) = (W_2 W_1)\,x$. But $W_2 W_1$ is just *one* matrix - so the
  two-layer stack computes nothing a single linear layer couldn't. Depth without
  nonlinearities adds zero expressive power; the nonlinearity between layers is
  what blocks this collapse (a classic interview question).

## 1.5 Eigenvectors and eigenvalues: the directions a matrix doesn't rotate

A matrix warps space: a typical vector comes out rotated *and* stretched. Natural
question: are there special directions the matrix treats more simply - directions
it stretches but does *not* rotate? A vector $v \ne 0$ with

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

So this $A$ has exactly two such directions: stretch 3× along the diagonal, do
nothing along the anti-diagonal. The two eigen-directions are orthogonal, and
that is a general fact:
**a symmetric matrix always has a full set of orthogonal eigenvectors with real
eigenvalues** (the spectral theorem).

**Why eigenvectors are useful.** In ordinary coordinates $A$ mixes its
inputs. For $x = [x_1, x_2]$,

$$Ax = \begin{bmatrix} 2x_1 + x_2 \\ x_1 + 2x_2 \end{bmatrix} \tag{1.5b}$$

Every output entry depends on both input entries, so it is hard to see what
applying $A$ ten or a hundred times would do.

Describe $x$ a different way: as an amount of each eigenvector,
$v_1 = [1, 1]$ (eigenvalue 3) and $v_2 = [1, -1]$ (eigenvalue 1).

$$x = c_1 v_1 + c_2 v_2 \tag{1.5c}$$

For example, $x = [3, 1] = 2\,[1, 1] + 1\,[1, -1]$, so $c_1 = 2$ and $c_2 = 1$.

$A$ is linear (§1.4), so it can be applied to each piece separately. It
multiplies $v_1$ by 3 and leaves $v_2$ unchanged:

$$Ax = c_1 (A v_1) + c_2 (A v_2) = 3 c_1 v_1 + c_2 v_2 \tag{1.5d}$$

For $x = [3, 1]$: $Ax = 6\,[1, 1] + 1\,[1, -1] = [7, 5]$. Equation (1.5b) gives
the same result: $[2 \cdot 3 + 1,\ 3 + 2 \cdot 1] = [7, 5]$.

The amounts $(c_1, c_2)$ are called the **eigen-coordinates** of $x$: its
coordinates when the eigenvectors are used as the axes (the **eigenbasis**).
In eigen-coordinates, $A$ only does $(c_1, c_2) \to (3c_1, c_2)$: multiply the
first coordinate by 3 and leave the second unchanged. Neither coordinate
affects the other. As a matrix, that map is
$\begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$: numbers only on the diagonal
(top-left to bottom-right), zeros everywhere else. A matrix like this is
called **diagonal**, and it multiplies each coordinate by its own number
without mixing them. So "$A$ is diagonal in the eigenbasis" means: measured
along the eigenvectors, $A$ only stretches each coordinate.

Repeated application is now simple. Each application multiplies $c_1$ by 3
again, so $A^{100} x = 3^{100} c_1 v_1 + c_2 v_2$.

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

- **PCA.** Picture a cloud of data points shaped like a tilted, stretched
  ellipse. PCA finds the long axis of that ellipse (the direction in which the
  data spreads out most), then the next-longest axis, and so on. These axes are
  the eigenvectors of the covariance matrix.
  - The **covariance matrix** $C$ records how pairs of features vary together.
    Center the data first (subtract each feature's mean). Then $C_{ij}$ is the
    average of (feature $i$'s deviation) $\times$ (feature $j$'s deviation),
    and the diagonal entry $C_{ii}$ is feature $i$'s variance.
  - $C$ is symmetric, $C_{ij} = C_{ji}$, because the two deviations give the
    same product in either order. So the spectral theorem applies and $C$ has
    orthogonal eigenvectors.
  - Each eigenvalue is the variance of the data along its eigenvector. PCA
    keeps the eigenvectors with the largest eigenvalues and drops the rest.
- **Repeated application.** Apply the same matrix many times and the
  eigenvector with the largest $|\lambda|$ takes over. Each application
  multiplies eigen-coordinate $c_i$ by $\lambda_i$, so after $k$ applications
  it has been multiplied by $\lambda_i^k$. With the $A$ and $x = [3, 1]$ from
  above, ten applications multiply $c_1$ by $3^{10} = 59{,}049$ and leave $c_2$
  at 1: $A^{10} x = [118099,\ 118097]$, almost exactly along $v_1 = [1, 1]$.
  - In general, components with $|\lambda| > 1$ grow exponentially and
    components with $|\lambda| < 1$ shrink toward zero.
  - In ML: an RNN uses the same weight matrix $W$ at every timestep, so
    backprop through $T$ timesteps multiplies the gradient by $W^\top$ (and a
    tanh slope that varies by step) $T$ times. That is the $A^k$ pattern, and
    it is why RNN gradients explode or vanish over long sequences.
- **Power iteration.** The same takeover gives an algorithm for the top
  eigenvector (the one with the largest $|\lambda|$). Start from a random
  vector and repeat two steps: multiply by $A$, then rescale to length 1. Each
  multiplication makes the top component larger relative to all the others, so
  the vector converges to the top eigenvector. The rescaling only keeps the
  numbers from overflowing.
- **PageRank** is power iteration on the web's link graph. The idea: a page is
  important if a reader who clicks links at random for a long time often ends
  up on it.
  - Build $A$ from the links. If page $i$ has 3 outgoing links, a reader on
    page $i$ clicks each one with probability $\frac13$. Entry $A_{ji}$ is the
    probability that a reader on page $i$ goes to page $j$ next.
  - Let $p_i$ be the fraction of readers on page $i$. After everyone clicks
    once, the fraction on page $j$ is $\sum_i A_{ji}\, p_i$, the readers
    arriving from every page that links to $j$. That sum is entry $j$ of $Ap$,
    so one round of clicking turns $p$ into $Ap$.
  - A page's importance is its share of readers once this settles: the $p$
    that one more round does not change, $Ap = p$. So $p$ is an eigenvector of
    $A$ with eigenvalue 1.
  - Google computed $p$ by power iteration: spread the readers evenly, then
    multiply by $A$ until $p$ stops changing.
- **Optimization.** In 1-D, the second derivative is curvature: how fast the
  slope changes as you move. A loss over $n$ weights has no single
  curvature - it depends on which direction you move in - and the
  **Hessian** is the matrix that stores all of it:
  - **The entries.** $H_{ij} = \partial^2 L / \partial w_i \partial w_j$,
    read as "change $w_j$ slightly, and watch how the slope with respect
    to $w_i$ changes." Two indices because a second derivative
    differentiates twice, once per weight; $i$ and $j$ range over all $n$
    weights, so $H$ is the full $n \times n$ table of every pair. The
    diagonal entry $H_{ii}$ is the plain 1-D curvature along weight $i$
    alone.
  - **Why it matters at a minimum.** The slope there is zero, so
    curvature is all that is left to describe the surface: near a
    minimum, the loss is a bowl, and $H$ is the bowl's shape.
  - **The spectral theorem applies yet again.** $H$ is symmetric
    (differentiating with respect to $w_i$ then $w_j$ gives the same
    value as the reverse order), so it has orthogonal eigen-directions,
    and each eigenvalue is the curvature along its own direction. Big
    $\lambda$: a steep wall. Small $\lambda$: a nearly flat floor.
  - **The condition number.** The ratio $\lambda_{max}/\lambda_{min}$
    measures how stretched the bowl is. A ratio of 1 is a round bowl; a
    ratio of 100 means one direction is 100 times steeper than another.
    A stretched bowl slows gradient descent badly. The gradient points
    mostly across the narrow valley, toward the steep walls, instead of
    along the valley toward the minimum. So each step overshoots to the
    opposite wall and makes little progress along the valley, and the path
    bounces from side to side (the figures below).

<div style="margin:1.25rem 0">
<svg viewBox="0 0 480 245" role="img" aria-label="A 3D loss bowl over two weights, sliced at three heights colored green, amber, and rose, next to the same bowl seen from above where the slices become contour rings" style="max-width:500px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="hz3-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity="0.6"/></marker>
  </defs>
  <line x1="40" y1="190" x2="40" y2="70" stroke="currentColor" stroke-opacity="0.5" marker-end="url(#hz3-a)"/>
  <text x="34" y="62" fill="currentColor" fill-opacity="0.7">loss</text>
  <path d="M 65 90 Q 150 290 235 90" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.5"/>
  <g fill="none" stroke-opacity="0.8">
    <ellipse cx="150" cy="90" rx="85" ry="24" stroke="#f43f5e"/>
    <ellipse cx="150" cy="125" rx="68" ry="19" stroke="#f59e0b"/>
    <ellipse cx="150" cy="160" rx="46" ry="13" stroke="#10b981"/>
  </g>
  <circle cx="150" cy="190" r="2.5" fill="currentColor" fill-opacity="0.8"/>
  <line x1="255" y1="128" x2="300" y2="128" stroke="currentColor" stroke-opacity="0.6" marker-end="url(#hz3-a)"/>
  <text x="277" y="114" text-anchor="middle" fill="currentColor" fill-opacity="0.7" font-size="10.5">from above</text>
  <g fill="none" stroke-opacity="0.8">
    <circle cx="390" cy="128" r="48" stroke="#f43f5e"/>
    <circle cx="390" cy="128" r="36" stroke="#f59e0b"/>
    <circle cx="390" cy="128" r="23" stroke="#10b981"/>
  </g>
  <circle cx="390" cy="128" r="2.5" fill="currentColor" fill-opacity="0.8"/>
  <g fill="currentColor" fill-opacity="0.65" text-anchor="middle" font-size="10.5">
    <text x="150" y="232">the loss bowl over two weights</text>
    <text x="390" y="232">the same bowl seen from above</text>
  </g>
</svg>
</div>

*A loss surface over two weights, drawn in 3-D: a bowl with its minimum at
the bottom dot. Slice the bowl at three loss heights (emerald lowest, then
amber, then rose) and look straight down: each slice becomes a ring, and
that ring picture is the contour plot used below.*

<div style="margin:1.25rem 0">
<svg viewBox="0 0 480 205" role="img" aria-label="Contour plots of two loss bowls with rings colored by loss height: a round bowl where gradient descent goes straight to the minimum, and an elongated bowl where it zig-zags across the narrow valley" style="max-width:500px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="hz-t" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="hz-e" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" fill-opacity="0.6"/></marker>
  </defs>
  <g fill="none" stroke-opacity="0.6">
    <circle cx="110" cy="105" r="20" stroke="#10b981"/>
    <circle cx="110" cy="105" r="40" stroke="#f59e0b"/>
    <circle cx="110" cy="105" r="60" stroke="#f43f5e"/>
    <ellipse cx="350" cy="105" rx="36" ry="9" stroke="#10b981"/>
    <ellipse cx="350" cy="105" rx="73" ry="18" stroke="#f59e0b"/>
    <ellipse cx="350" cy="105" rx="110" ry="27" stroke="#f43f5e"/>
  </g>
  <circle cx="110" cy="105" r="2.5" fill="currentColor" fill-opacity="0.7"/>
  <circle cx="350" cy="105" r="2.5" fill="currentColor" fill-opacity="0.7"/>
  <line x1="152" y1="63" x2="117" y2="98" stroke="#6366f1" stroke-width="2" marker-end="url(#hz-t)"/>
  <circle cx="143" cy="72" r="2" fill="#6366f1"/><circle cx="133" cy="82" r="2" fill="#6366f1"/><circle cx="124" cy="91" r="2" fill="#6366f1"/>
  <line x1="350" y1="105" x2="350" y2="74" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.5" marker-end="url(#hz-e)"/>
  <line x1="350" y1="105" x2="276" y2="105" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.5" marker-end="url(#hz-e)"/>
  <text x="350" y="62" text-anchor="middle" fill="currentColor" fill-opacity="0.75">big &#955;: steep</text>
  <text x="290" y="145" text-anchor="middle" fill="currentColor" fill-opacity="0.75">small &#955;: flat</text>
  <polyline points="438,92 400,118 372,95 356,112 348,101" fill="none" stroke="#6366f1" stroke-width="2" marker-end="url(#hz-t)"/>
  <circle cx="438" cy="92" r="2" fill="#6366f1"/><circle cx="400" cy="118" r="2" fill="#6366f1"/><circle cx="372" cy="95" r="2" fill="#6366f1"/><circle cx="356" cy="112" r="2" fill="#6366f1"/>
  <text x="110" y="195" text-anchor="middle" fill="currentColor" fill-opacity="0.7">condition number &#8776; 1</text>
  <text x="350" y="195" text-anchor="middle" fill="currentColor" fill-opacity="0.7">condition number &#8811; 1</text>
  <g font-size="10.5">
    <text x="20" y="28" fill="currentColor" fill-opacity="0.7">loss height:</text>
    <text x="85" y="28" fill="#10b981">low</text>
    <text x="112" y="28" fill="#f59e0b">mid</text>
    <text x="140" y="28" fill="#f43f5e">high</text>
  </g>
</svg>
</div>

*Loss contours from above; the dot is the minimum, and ring color is loss
height (emerald just above the minimum, then amber, then rose). The indigo
path is gradient descent. Each step moves perpendicular to the contour ring
at the current point. In the round bowl that direction points at the
minimum. In the stretched bowl it points across the valley, so each step
crosses to the opposite wall and the path bounces from side to side. This
picture is why momentum and Adam exist.*

## 1.6 SVD: every matrix is rotate → stretch → rotate

#### Part A: the eigendecomposition - §1.5 as a formula

§1.5 computed $Ax$ for $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ and
$x = [3, 1]$ in three steps: split $x$ into eigenvectors (amounts 2 and 1),
multiply each amount by its eigenvalue (6 and 1), and rebuild:
$6\,[1,1] + 1\,[1,-1] = [7,5]$. Each step is a multiplication by a matrix.
Put the eigenvectors side by side as the columns of
$Q = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$.

- **Step 1: multiply by $Q^{-1}$** (into eigen-coordinates). The amounts
  $c = [c_1, c_2]$ satisfy $Qc = c_1 v_1 + c_2 v_2 = x$ (the blend rule,
  1.4c), so $c = Q^{-1} x$. Here
  $Q^{-1} = \tfrac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, and
  $Q^{-1}[3, 1] = [2, 1]$.
- **Step 2: multiply by $\Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$**
  (the stretch). This is the diagonal matrix from §1.5: it multiplies
  coordinate $i$ by $\lambda_i$ and mixes nothing: $\Lambda [2, 1] = [6, 1]$.
- **Step 3: multiply by $Q$** (back to ordinary coordinates). Each amount
  multiplies its own eigenvector and the results add:
  $Q [6, 1] = 6\,[1,1] + 1\,[1,-1] = [7, 5] = Ax$.

<div style="margin:1.25rem 0">
<svg viewBox="0 0 560 205" role="img" aria-label="Four frames following x=[3,1]. Q inverse and Q leave the arrow in place and only change whether it is measured along the gray axes or the green eigen-directions; Lambda is the only step that moves the arrow, tripling its part along [1,1]" style="max-width:580px;width:100%;height:auto;font-size:11px;font-family:inherit">
  <defs>
    <marker id="pa-v" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1"/></marker>
    <marker id="pa-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1" fill-opacity="0.3"/></marker>
    <marker id="pa-f" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.4">
    <line x1="7" y1="135" x2="115" y2="135" />
    <line x1="15" y1="160" x2="15" y2="47" />
  </g>
  <line x1="15" y1="135" x2="106" y2="44" stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3" stroke-opacity="0.7"/>
  <line x1="15" y1="135" x2="38.4" y2="158.4" stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3" stroke-opacity="0.7"/>
  <g stroke="currentColor" stroke-opacity="0.15">
    <line x1="147" y1="135" x2="255" y2="135" />
    <line x1="155" y1="160" x2="155" y2="47" />
  </g>
  <line x1="155" y1="135" x2="246" y2="44" stroke="#10b981" stroke-width="1.5" stroke-opacity="0.9"/>
  <line x1="155" y1="135" x2="178.4" y2="158.4" stroke="#10b981" stroke-width="1.5" stroke-opacity="0.9"/>
  <g stroke="currentColor" stroke-opacity="0.15">
    <line x1="287" y1="135" x2="395" y2="135" />
    <line x1="295" y1="160" x2="295" y2="47" />
  </g>
  <line x1="295" y1="135" x2="386" y2="44" stroke="#10b981" stroke-width="1.5" stroke-opacity="0.9"/>
  <line x1="295" y1="135" x2="318.4" y2="158.4" stroke="#10b981" stroke-width="1.5" stroke-opacity="0.9"/>
  <g stroke="currentColor" stroke-opacity="0.4">
    <line x1="427" y1="135" x2="535" y2="135" />
    <line x1="435" y1="160" x2="435" y2="47" />
  </g>
  <line x1="435" y1="135" x2="526" y2="44" stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3" stroke-opacity="0.7"/>
  <line x1="435" y1="135" x2="458.4" y2="158.4" stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3" stroke-opacity="0.7"/>
  <line x1="155" y1="135" x2="181" y2="109" stroke="#10b981" stroke-width="4" stroke-opacity="0.45" stroke-linecap="round"/>
  <line x1="181" y1="109" x2="194" y2="122" stroke="#10b981" stroke-width="4" stroke-opacity="0.45" stroke-linecap="round"/>
  <line x1="295" y1="135" x2="373" y2="57" stroke="#10b981" stroke-width="4" stroke-opacity="0.45" stroke-linecap="round"/>
  <line x1="373" y1="57" x2="386" y2="70" stroke="#10b981" stroke-width="4" stroke-opacity="0.45" stroke-linecap="round"/>
  <line x1="295" y1="135" x2="334" y2="122" stroke="#6366f1" stroke-opacity="0.3" stroke-width="2" marker-end="url(#pa-g)"/>
  <line x1="15" y1="135" x2="54" y2="122" stroke="#6366f1" stroke-width="2.5" marker-end="url(#pa-v)"/>
  <line x1="155" y1="135" x2="194" y2="122" stroke="#6366f1" stroke-width="2.5" marker-end="url(#pa-v)"/>
  <line x1="295" y1="135" x2="386" y2="70" stroke="#6366f1" stroke-width="2.5" marker-end="url(#pa-v)"/>
  <line x1="435" y1="135" x2="526" y2="70" stroke="#6366f1" stroke-width="2.5" marker-end="url(#pa-v)"/>
  <g fill="#6366f1" font-weight="600">
    <text x="60" y="126">[3, 1]</text>
    <text x="200" y="126">(2, 1)</text>
    <text x="392" y="78">(6, 1)</text>
    <text x="514" y="96">[7, 5]</text>
  </g>
  <line x1="112" y1="118" x2="145" y2="118" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#pa-f)"/>
  <text x="128" y="109" text-anchor="middle" fill="currentColor">Q&#8315;&#185;</text>
  <line x1="252" y1="118" x2="285" y2="118" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#pa-f)"/>
  <text x="268" y="109" text-anchor="middle" fill="currentColor">&#923;</text>
  <line x1="392" y1="118" x2="425" y2="118" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#pa-f)"/>
  <text x="408" y="109" text-anchor="middle" fill="currentColor">Q</text>
  <g stroke="#10b981" stroke-width="1.5">
    <line x1="165.9" y1="119.9" x2="170.1" y2="124.1"/>
    <line x1="178.9" y1="106.9" x2="183.1" y2="111.1"/>
    <line x1="191.9" y1="93.9" x2="196.1" y2="98.1"/>
    <line x1="204.9" y1="80.9" x2="209.1" y2="85.1"/>
    <line x1="217.9" y1="67.9" x2="222.1" y2="72.1"/>
    <line x1="230.9" y1="54.9" x2="235.1" y2="59.1"/>
    <line x1="243.9" y1="41.9" x2="248.1" y2="46.1"/>
    <line x1="165.9" y1="150.1" x2="170.1" y2="145.9"/>
    <line x1="305.9" y1="119.9" x2="310.1" y2="124.1"/>
    <line x1="318.9" y1="106.9" x2="323.1" y2="111.1"/>
    <line x1="331.9" y1="93.9" x2="336.1" y2="98.1"/>
    <line x1="344.9" y1="80.9" x2="349.1" y2="85.1"/>
    <line x1="357.9" y1="67.9" x2="362.1" y2="72.1"/>
    <line x1="370.9" y1="54.9" x2="375.1" y2="59.1"/>
    <line x1="383.9" y1="41.9" x2="388.1" y2="46.1"/>
    <line x1="305.9" y1="150.1" x2="310.1" y2="145.9"/>
  </g>
  <g fill="#10b981" font-size="10px" font-weight="600" text-anchor="end">
    <text x="176" y="104">2</text>
    <text x="368" y="52">6</text>
  </g>
  <g fill="#10b981" font-size="10px">
    <text x="74" y="70" text-anchor="end">v&#8321; = [1, 1]</text>
    <text x="42" y="162">v&#8322; = [1, &#8722;1]</text>
  </g>
  <g fill="currentColor" fill-opacity="0.6" text-anchor="middle">
    <text x="60" y="182">1. start</text>
    <text x="200" y="182">2. after Q&#8315;&#185;</text>
    <text x="340" y="182">3. after &#923;</text>
    <text x="480" y="182">4. after Q</text>
  </g>
  <g font-size="10px" text-anchor="middle">
    <text x="60" y="197" fill="currentColor" fill-opacity="0.6">normal coordinates</text>
    <text x="200" y="197" fill="#10b981">eigen-coordinates</text>
    <text x="340" y="197" fill="#10b981">eigen-coordinates</text>
    <text x="480" y="197" fill="currentColor" fill-opacity="0.6">normal coordinates</text>
  </g>
</svg>
</div>

*$x = [3,1]$ through the three steps. $Q^{-1}$ does not move the arrow: it
measures the same arrow along the green eigen-directions instead of the gray
axes. Each tick on a green line is one step of $v_1$ or $v_2$. Getting to
the tip takes 2 steps along $v_1$ and 1 step along $v_2$ (the green path), so
the eigen-coordinates are $(2,1)$. $\Lambda$ is
the only step that moves the arrow: it triples the part along $v_1$, giving
$(6,1)$, which is 6 steps along $v_1$ and 1 along $v_2$. $Q$ again moves nothing; it measures the new arrow along the
gray axes, which gives $[7,5] = Ax$.*

Written as one product (the rightmost factor acts first):

$$A = Q \Lambda Q^{-1} \tag{1.6a}$$

This is the **eigendecomposition** of $A$: switch to eigen-coordinates
($Q^{-1}$), stretch each coordinate ($\Lambda$), switch back ($Q$). Multiplying the three matrices out gives
exactly $\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.

When $A$ is symmetric, its eigenvectors are orthogonal. Scaled to unit
length, they make $Q$ an orthogonal matrix (a rotation, possibly combined
with a reflection), and then $Q^{-1} = Q^\top$. The unscaled $Q$ above has
columns of length $\sqrt{2}$, which is why its inverse has the factor
$\tfrac{1}{2}$.

The eigendecomposition has two limits.

- **It needs a square matrix.** $Av = \lambda v$ compares the output with the
  input, so both must have the same number of entries. For example,
  $B = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$ ($3 \times 2$)
  turns $[1, 1]$ into $[2, 1, 1]$. A 3-entry output cannot be a multiple of
  a 2-entry input, so $B$ has no eigenvectors.
- **Its axes are perpendicular only when $A$ is symmetric.** The
  non-symmetric $\begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}$ has
  eigenvectors $[1, 0]$ (eigenvalue 2) and $[1, -1]$ (eigenvalue 1), which
  are 45° apart instead of 90°. Its eigendecomposition exists, but the axes
  are skewed and $Q^{-1} \ne Q^\top$. Some square matrices have too few
  eigenvectors for any eigendecomposition: the shear
  $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ has only one
  eigen-direction, $[1, 0]$ (worked out at the end of Part B).

A data matrix ($n$ rows × $d$ features) is usually rectangular, so it has no
eigendecomposition. The SVD, next in Part B, works for every matrix, and its
axes are always perpendicular.

#### Part B: the SVD - the same idea for any matrix

The running example has 3 rows and 2 columns, so it takes 2-number inputs to
3-number outputs:

$$B = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$$

$Bv = \lambda v$ would set a 3-entry vector equal to a 2-entry one, so $B$ has
no eigenvectors. The fix: inputs and outputs live in different spaces, so use
a separate set of perpendicular directions for each.

$$A = U \Sigma V^\top \tag{1.6b}$$

- **$V^\top$: into input coordinates.** Its rows are perpendicular unit
  vectors $v_1, v_2, \dots$ in the input space, so entry $i$ of $V^\top x$ is
  $v_i \cdot x$, the amount of $x$ along $v_i$ (§1.2). This is the job
  $Q^{-1}$ did.
- **$\Sigma$: the stretch.** It multiplies amount $i$ by $\sigma_i \ge 0$.
  These are the **singular values**, numbered largest first. (Σ also changes
  the length from $d$ to $n$, padding with zeros.)
- **$U$: rebuild the output.** Its columns are perpendicular unit vectors
  $u_1, u_2, \dots$ in the output space. Amounts $(a_1, a_2, \dots)$ become
  $a_1 u_1 + a_2 u_2 + \dots$. This is the job $Q$ did.

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

*The unit circle always maps to an ellipse: rotations keep circles round,
the stretch makes the ellipse (radii $\sigma_1, \sigma_2$), the last
rotation turns it into place. The inputs $v_1, v_2$ land on
$\sigma_1 u_1, \sigma_2 u_2$.*

**Finding $V$ and $\Sigma$.** Part A's method needs a square, symmetric
matrix. $A^\top A$ is both: it is $d \times d$, and
$(A^\top A)^\top = A^\top A$. Substitute (1.6b) and use $U^\top U = I$:

$$A^\top A = V \Sigma^\top U^\top U \Sigma V^\top = V \Sigma^2 V^\top \tag{1.6c}$$

This has the form $Q \Lambda Q^{-1}$ of (1.6a), with $Q = V$ and
$\Lambda = \Sigma^2$. So the $v_i$ are the eigenvectors of $A^\top A$, and the
$\sigma_i$ are the square roots of its eigenvalues. The spectral theorem
(§1.5) makes the $v_i$ perpendicular.

**Finding $U$.** The SVD says $A$ sends each input direction $v_i$ to the
output direction $u_i$, stretched by $\sigma_i$ (the figure above):

$$A v_i = \sigma_i u_i$$

So to find $u_i$, send $v_i$ through $A$ and shrink the result to length 1.
That result has length exactly $\sigma_i$, because
$\|A v_i\|^2 = v_i^\top A^\top A\, v_i = \sigma_i^2$. Dividing by it:

$$u_i = \frac{A v_i}{\sigma_i}$$

Step 3 of the recipe below does this with numbers. (For $\sigma_i = 0$,
$A v_i = 0$ has no direction; fill in any perpendicular unit vectors.) These
$u_i$ come out perpendicular automatically. For $i \ne j$, use
$(Ax) \cdot y = x \cdot (A^\top y)$ and then $A^\top A\, v_j = \sigma_j^2 v_j$:

$$u_i \cdot u_j = \frac{(A v_i) \cdot (A v_j)}{\sigma_i \sigma_j}
= \frac{v_i \cdot (A^\top A\, v_j)}{\sigma_i \sigma_j}
= \frac{\sigma_j^2\, (v_i \cdot v_j)}{\sigma_i \sigma_j} = 0$$

**The recipe on $B$.**

1. **Form $B^\top B$:**

   $$B^\top B = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix}
   \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}
   = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

   which is the §1.5 matrix.

2. **Eigendecompose it.** §1.5 found $[1,1]$ with eigenvalue 3 and $[1,-1]$
   with eigenvalue 1. Normalize, and take square roots:

   $$v_1 = \frac{[1,1]}{\sqrt{2}}, \quad v_2 = \frac{[1,-1]}{\sqrt{2}}
   \qquad \sigma_1 = \sqrt{3}, \quad \sigma_2 = 1$$

3. **Output directions:**

   $$u_1 = \frac{B v_1}{\sigma_1} = \frac{[2, 1, 1]}{\sqrt{6}}
   \qquad u_2 = \frac{B v_2}{\sigma_2} = \frac{[0, -1, 1]}{\sqrt{2}}$$

   These have 3 entries: they live in the output space. Check:
   $u_1 \cdot u_2 = (0 - 1 + 1)/\sqrt{12} = 0$.

The $u_i$ are the columns of $U$, the $\sigma_i$ sit on $\Sigma$'s diagonal,
and the $v_i$ are the rows of $V^\top$:

$$B = U \Sigma V^\top =
\begin{bmatrix} \tfrac{2}{\sqrt{6}} & 0 \\ \tfrac{1}{\sqrt{6}} & -\tfrac{1}{\sqrt{2}} \\ \tfrac{1}{\sqrt{6}} & \tfrac{1}{\sqrt{2}} \end{bmatrix}
\begin{bmatrix} \sqrt{3} & 0 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} \tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\ \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \end{bmatrix}$$

Multiplying out gives $B$ exactly. The factors send each $v_i$ where $B$
does, and the $v_i$ span the input space, so by §1.4 they agree with $B$ on
every input. (This is the *compact* form, which keeps only the columns with
nonzero $\sigma$. The *full* form pads $U$ to $3 \times 3$ and $\Sigma$ to
$3 \times 2$; the product is the same.)

None of these steps needed anything special about $B$, so **every matrix has
an SVD: every linear map is rotate → stretch → rotate.** In words: $v_1$ is
the input direction $A$ stretches most, $\sigma_1$ is that stretch, and $u_1$
is where it lands. $v_2$ is the same among directions perpendicular to $v_1$,
and so on.

Two checks:

- For a symmetric matrix with nonnegative eigenvalues, such as the §1.5 $A$,
  the SVD is the eigendecomposition with unit eigenvectors:
  $u_i = v_i = [1, \pm 1]/\sqrt{2}$, $\sigma_1 = 3$, $\sigma_2 = 1$.
- The shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ has no
  eigendecomposition. $S[x, y] = [x + y,\ y] = \lambda [x, y]$ forces
  $\lambda = 1$ and then $y = 0$, so its only eigen-direction is $[1, 0]$,
  one short of the two needed. Its SVD exists anyway:
  $\sigma_1 \approx 1.62$, $\sigma_2 \approx 0.62$.

#### Part C: compressing a matrix - why ML cares

A ratings matrix with $n = 10{,}000$ users and $d = 500$ movies holds
$5{,}000{,}000$ numbers. Can far fewer numbers stand in for almost the same
matrix?

**Keep the top $k$ of the SVD.** Keep the first $k$ columns of $U$ and $V$
and the first $k$ singular values: $k(n + d + 1)$ numbers. With $k = 20$ that
is $20 \times 10{,}501 = 210{,}020$, about 4% of the original. Multiplying the
three smaller factors gives an approximation of $A$.

**What it loses.** Multiplying $U \Sigma V^\top$ out gives one term per
singular value:

$$A = \sum_{i} \sigma_i \, u_i v_i^\top \tag{1.6d}$$

Each $u_i v_i^\top$ is a column times a row. For example:

$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 3 & 1 \end{bmatrix}
= \begin{bmatrix} 3 & 1 \\ 6 & 2 \end{bmatrix}$$

Every row is a multiple of the same row $[3, 1]$: one pattern. A matrix like
this is **rank-1**, and a sum of $k$ of them is **rank-$k$** ("low-rank" in ML
means this). Because $u_i$ and $v_i$ are unit vectors, the entries of
$u_i v_i^\top$ are at most 1 in size, so the size of term $i$ is set by
$\sigma_i$.

Keeping the top $k$ factors keeps exactly the first $k$ terms:

$$A \approx \sum_{i=1}^{k} \sigma_i \, u_i v_i^\top \tag{1.6e}$$

and drops the terms with the smallest $\sigma$. The loss is small when the
$\sigma_i$ fall off quickly, which they usually do for real data.

Check (1.6d) on the §1.5 $A$ ($u_i = v_i$, $\sigma_1 = 3$, $\sigma_2 = 1$):

$$3 \cdot \tfrac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
+ 1 \cdot \tfrac{1}{2} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
= \begin{bmatrix} 1.5 & 1.5 \\ 1.5 & 1.5 \end{bmatrix}
+ \begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{bmatrix}
= \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

**Can another summary do better?** No. The **Eckart–Young theorem** says
(1.6e) is the best approximation of $A$ that uses $k$ patterns.

**Where ML uses this:**

- **PCA** is the SVD of the centered data matrix.
- **Classic recommenders** approximate the user × movie ratings matrix with
  (1.6e) for a small $k$. Each $v_i$ is a taste pattern over movies, and each
  $u_i$ says how much of that taste every user has. Concretely:

<div style="margin:1.25rem 0">
<svg viewBox="0 0 465 205" role="img" aria-label="The 5 by 3 ratings matrix equals 8.8 times u1 v1 transpose plus 6.0 times u2 v2 transpose, using the true SVD numbers; pattern 1 is overall appetite, pattern 2 contrasts romance against action" style="max-width:485px;width:100%;height:auto;font-size:11px;font-family:inherit">
<rect x="40" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="60" text-anchor="middle" fill="currentColor" font-size="9px">5</text><rect x="64" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="60" text-anchor="middle" fill="currentColor" font-size="9px">4</text><rect x="88" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="60" text-anchor="middle" fill="currentColor" font-size="9px">0</text><rect x="40" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="84" text-anchor="middle" fill="currentColor" font-size="9px">0</text><rect x="64" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="84" text-anchor="middle" fill="currentColor" font-size="9px">0</text><rect x="88" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="84" text-anchor="middle" fill="currentColor" font-size="9px">5</text><rect x="40" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="108" text-anchor="middle" fill="currentColor" font-size="9px">3</text><rect x="64" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.22" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="108" text-anchor="middle" fill="currentColor" font-size="9px">2.4</text><rect x="88" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="108" text-anchor="middle" fill="currentColor" font-size="9px">4</text><rect x="40" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="132" text-anchor="middle" fill="currentColor" font-size="9px">2</text><rect x="64" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="132" text-anchor="middle" fill="currentColor" font-size="9px">1.6</text><rect x="88" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="132" text-anchor="middle" fill="currentColor" font-size="9px">0</text><rect x="40" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.09" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="156" text-anchor="middle" fill="currentColor" font-size="9px">1</text><rect x="64" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.07" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="156" text-anchor="middle" fill="currentColor" font-size="9px">.80</text><rect x="88" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="156" text-anchor="middle" fill="currentColor" font-size="9px">3</text>
<text x="34" y="61" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">A</text><text x="34" y="85" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">B</text><text x="34" y="109" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">C</text><text x="34" y="133" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">D</text><text x="34" y="157" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">E</text>
<text x="52" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="76" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="100" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="76" y="192" text-anchor="middle" fill="currentColor" fill-opacity="0.6">ratings</text>
<text x="121" y="110" text-anchor="middle" fill="currentColor" font-size="14">&#8776;</text>
<text x="141" y="109" text-anchor="middle" fill="#6366f1" font-size="10px">8.8 &#183;</text>
<rect x="158" y="45" width="24" height="24" fill="#6366f1" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="60" text-anchor="middle" fill="currentColor" font-size="9px">.60</text><rect x="158" y="69" width="24" height="24" fill="#6366f1" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="84" text-anchor="middle" fill="currentColor" font-size="9px">.33</text><rect x="158" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.28" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.62</text><rect x="158" y="117" width="24" height="24" fill="#6366f1" fill-opacity="0.11" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="132" text-anchor="middle" fill="currentColor" font-size="9px">.24</text><rect x="158" y="141" width="24" height="24" fill="#6366f1" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="156" text-anchor="middle" fill="currentColor" font-size="9px">.31</text>
<text x="170" y="38" text-anchor="middle" fill="#6366f1">u&#8321;</text>
<text x="196" y="109" text-anchor="middle" fill="currentColor">&#215;</text>
<rect x="208" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.29" stroke="currentColor" stroke-opacity="0.25"/><text x="220" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.64</text><rect x="232" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.23" stroke="currentColor" stroke-opacity="0.25"/><text x="244" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.51</text><rect x="256" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.26" stroke="currentColor" stroke-opacity="0.25"/><text x="268" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.57</text>
<text x="220" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="244" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="268" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="244" y="75" text-anchor="middle" fill="#6366f1">v&#8321;&#7488;</text>
<text x="204" y="192" text-anchor="middle" fill="#6366f1" font-weight="600" font-size="10px">pattern 1: overall appetite</text>
<text x="288" y="110" text-anchor="middle" fill="currentColor" font-size="14">+</text>
<text x="308" y="109" text-anchor="middle" fill="#f43f5e" font-size="10px">6.0 &#183;</text>
<rect x="326" y="45" width="24" height="24" fill="#f43f5e" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="60" text-anchor="middle" fill="currentColor" font-size="9px">-.61</text><rect x="326" y="69" width="24" height="24" fill="#f43f5e" fill-opacity="0.31" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="84" text-anchor="middle" fill="currentColor" font-size="9px">.68</text><rect x="326" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.18</text><rect x="326" y="117" width="24" height="24" fill="#f43f5e" fill-opacity="0.11" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="132" text-anchor="middle" fill="currentColor" font-size="9px">-.24</text><rect x="326" y="141" width="24" height="24" fill="#f43f5e" fill-opacity="0.13" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="156" text-anchor="middle" fill="currentColor" font-size="9px">.28</text>
<text x="338" y="38" text-anchor="middle" fill="#f43f5e">u&#8322;</text>
<text x="364" y="109" text-anchor="middle" fill="currentColor">&#215;</text>
<rect x="376" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.25"/><text x="388" y="108" text-anchor="middle" fill="currentColor" font-size="9px">-.45</text><rect x="400" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.16" stroke="currentColor" stroke-opacity="0.25"/><text x="412" y="108" text-anchor="middle" fill="currentColor" font-size="9px">-.36</text><rect x="424" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.37" stroke="currentColor" stroke-opacity="0.25"/><text x="436" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.82</text>
<text x="388" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="412" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="436" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="412" y="75" text-anchor="middle" fill="#f43f5e">v&#8322;&#7488;</text>
<text x="380" y="192" text-anchor="middle" fill="#f43f5e" font-weight="600" font-size="10px">pattern 2: romance &#8722; action</text>
</svg>
</div>

  *The 5 × 3 ratings matrix split into its two SVD terms. Term 1 (weight
  8.8): $v_1^\top = [.64, .51, .57]$ is positive and about equal for every
  movie, so a user's $u_1$ amount measures overall enthusiasm. ("Overall
  appetite" is our label; the SVD only outputs numbers.) Term 2 (weight
  6.0): $v_2^\top = [-.45, -.36, .82]$ contrasts Titanic with the action
  films: romance fans score positive (Ben $.68$), action fans negative (Ana
  $-.61$). Cara's row: $8.8(.62)\,v_1^\top + 6.0(.18)\,v_2^\top =
  [3, 2.4, 4]$, her exact ratings.*

<div style="margin:1.25rem 0">
<svg viewBox="0 0 500 210" role="img" aria-label="The 5 by 3 ratings matrix factored as U times Sigma times V transpose with real numbers, then truncated to the top singular value to give the k equals 1 prediction" style="max-width:520px;width:100%;height:auto;font-size:11px;font-family:inherit">
<defs><marker id="lc-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>
<rect x="40" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="60" text-anchor="middle" fill="currentColor" font-size="10px">5</text><rect x="64" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="60" text-anchor="middle" fill="currentColor" font-size="10px">4</text><rect x="88" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="60" text-anchor="middle" fill="currentColor" font-size="10px">0</text><rect x="40" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="84" text-anchor="middle" fill="currentColor" font-size="10px">0</text><rect x="64" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="84" text-anchor="middle" fill="currentColor" font-size="10px">0</text><rect x="88" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="84" text-anchor="middle" fill="currentColor" font-size="10px">5</text><rect x="40" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="108" text-anchor="middle" fill="currentColor" font-size="10px">3</text><rect x="64" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.22" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="108" text-anchor="middle" fill="currentColor" font-size="10px">2.4</text><rect x="88" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.36" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="108" text-anchor="middle" fill="currentColor" font-size="10px">4</text><rect x="40" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="132" text-anchor="middle" fill="currentColor" font-size="10px">2</text><rect x="64" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="132" text-anchor="middle" fill="currentColor" font-size="10px">1.6</text><rect x="88" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="132" text-anchor="middle" fill="currentColor" font-size="10px">0</text><rect x="40" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.09" stroke="currentColor" stroke-opacity="0.25"/><text x="52" y="156" text-anchor="middle" fill="currentColor" font-size="10px">1</text><rect x="64" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.07" stroke="currentColor" stroke-opacity="0.25"/><text x="76" y="156" text-anchor="middle" fill="currentColor" font-size="10px">.80</text><rect x="88" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="100" y="156" text-anchor="middle" fill="currentColor" font-size="10px">3</text>
<text x="34" y="61" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">A</text><text x="34" y="85" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">B</text><text x="34" y="109" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">C</text><text x="34" y="133" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">D</text><text x="34" y="157" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">E</text>
<text x="52" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="76" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="100" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="76" y="192" text-anchor="middle" fill="currentColor" fill-opacity="0.6">ratings</text>
<text x="122" y="110" text-anchor="middle" fill="currentColor" font-size="14">&#8776;</text>
<rect x="134" y="45" width="24" height="24" fill="#6366f1" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="146" y="60" text-anchor="middle" fill="currentColor" font-size="9px">.60</text><rect x="158" y="45" width="24" height="24" fill="#6366f1" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="60" text-anchor="middle" fill="currentColor" font-size="9px">-.61</text><rect x="134" y="69" width="24" height="24" fill="#6366f1" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.25"/><text x="146" y="84" text-anchor="middle" fill="currentColor" font-size="9px">.33</text><rect x="158" y="69" width="24" height="24" fill="#6366f1" fill-opacity="0.31" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="84" text-anchor="middle" fill="currentColor" font-size="9px">.68</text><rect x="134" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.28" stroke="currentColor" stroke-opacity="0.25"/><text x="146" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.62</text><rect x="158" y="93" width="24" height="24" fill="#6366f1" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.18</text><rect x="134" y="117" width="24" height="24" fill="#6366f1" fill-opacity="0.11" stroke="currentColor" stroke-opacity="0.25"/><text x="146" y="132" text-anchor="middle" fill="currentColor" font-size="9px">.24</text><rect x="158" y="117" width="24" height="24" fill="#6366f1" fill-opacity="0.11" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="132" text-anchor="middle" fill="currentColor" font-size="9px">-.24</text><rect x="134" y="141" width="24" height="24" fill="#6366f1" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="146" y="156" text-anchor="middle" fill="currentColor" font-size="9px">.31</text><rect x="158" y="141" width="24" height="24" fill="#6366f1" fill-opacity="0.13" stroke="currentColor" stroke-opacity="0.25"/><text x="170" y="156" text-anchor="middle" fill="currentColor" font-size="9px">.28</text>
<text x="158" y="38" text-anchor="middle" fill="#6366f1">U</text>
<rect x="196" y="81" width="24" height="24" fill="#f59e0b" fill-opacity="0.45" stroke="currentColor" stroke-opacity="0.25"/><text x="208" y="96" text-anchor="middle" fill="currentColor" font-size="9px">8.8</text><rect x="220" y="81" width="24" height="24" fill="#f59e0b" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="232" y="96" text-anchor="middle" fill="currentColor" font-size="9px">0</text><rect x="196" y="105" width="24" height="24" fill="#f59e0b" fill-opacity="0.0" stroke="currentColor" stroke-opacity="0.25"/><text x="208" y="120" text-anchor="middle" fill="currentColor" font-size="9px">0</text><rect x="220" y="105" width="24" height="24" fill="#f59e0b" fill-opacity="0.31" stroke="currentColor" stroke-opacity="0.25"/><text x="232" y="120" text-anchor="middle" fill="currentColor" font-size="9px">6.0</text>
<text x="220" y="74" text-anchor="middle" fill="#f59e0b">&#931;</text>
<rect x="258" y="81" width="24" height="24" fill="#10b981" fill-opacity="0.29" stroke="currentColor" stroke-opacity="0.25"/><text x="270" y="96" text-anchor="middle" fill="currentColor" font-size="9px">.64</text><rect x="282" y="81" width="24" height="24" fill="#10b981" fill-opacity="0.23" stroke="currentColor" stroke-opacity="0.25"/><text x="294" y="96" text-anchor="middle" fill="currentColor" font-size="9px">.51</text><rect x="306" y="81" width="24" height="24" fill="#10b981" fill-opacity="0.26" stroke="currentColor" stroke-opacity="0.25"/><text x="318" y="96" text-anchor="middle" fill="currentColor" font-size="9px">.57</text><rect x="258" y="105" width="24" height="24" fill="#10b981" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.25"/><text x="270" y="120" text-anchor="middle" fill="currentColor" font-size="9px">-.45</text><rect x="282" y="105" width="24" height="24" fill="#10b981" fill-opacity="0.16" stroke="currentColor" stroke-opacity="0.25"/><text x="294" y="120" text-anchor="middle" fill="currentColor" font-size="9px">-.36</text><rect x="306" y="105" width="24" height="24" fill="#10b981" fill-opacity="0.37" stroke="currentColor" stroke-opacity="0.25"/><text x="318" y="120" text-anchor="middle" fill="currentColor" font-size="9px">.82</text>
<text x="294" y="63" text-anchor="middle" fill="#10b981">V&#7488;</text>
<text x="270" y="77" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="294" y="77" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="318" y="77" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="158" y="192" text-anchor="middle" fill="currentColor" fill-opacity="0.6">5&#215;2</text>
<text x="220" y="192" text-anchor="middle" fill="currentColor" fill-opacity="0.6">2&#215;2</text>
<text x="294" y="192" text-anchor="middle" fill="currentColor" fill-opacity="0.6">2&#215;3</text>
<line x1="342" y1="105" x2="368" y2="105" stroke="currentColor" stroke-opacity="0.7" marker-end="url(#lc-a)"/>
<text x="356" y="95" text-anchor="middle" fill="currentColor" font-size="10px">keep &#963;&#8321;</text>
<rect x="388" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.31" stroke="currentColor" stroke-opacity="0.25"/><text x="400" y="60" text-anchor="middle" fill="currentColor" font-size="9px">3.4</text><rect x="412" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.24" stroke="currentColor" stroke-opacity="0.25"/><text x="424" y="60" text-anchor="middle" fill="currentColor" font-size="9px">2.7</text><rect x="436" y="45" width="24" height="24" fill="currentColor" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="448" y="60" text-anchor="middle" fill="currentColor" font-size="9px">3</text><rect x="388" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.16" stroke="currentColor" stroke-opacity="0.25"/><text x="400" y="84" text-anchor="middle" fill="currentColor" font-size="9px">1.8</text><rect x="412" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="424" y="84" text-anchor="middle" fill="currentColor" font-size="9px">1.5</text><rect x="436" y="69" width="24" height="24" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="448" y="84" text-anchor="middle" fill="currentColor" font-size="9px">1.6</text><rect x="388" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.32" stroke="currentColor" stroke-opacity="0.25"/><text x="400" y="108" text-anchor="middle" fill="currentColor" font-size="9px">3.5</text><rect x="412" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.25" stroke="currentColor" stroke-opacity="0.25"/><text x="424" y="108" text-anchor="middle" fill="currentColor" font-size="9px">2.8</text><rect x="436" y="93" width="24" height="24" fill="currentColor" fill-opacity="0.28" stroke="currentColor" stroke-opacity="0.25"/><text x="448" y="108" text-anchor="middle" fill="currentColor" font-size="9px">3.1</text><rect x="388" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.25"/><text x="400" y="132" text-anchor="middle" fill="currentColor" font-size="9px">1.3</text><rect x="412" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.25"/><text x="424" y="132" text-anchor="middle" fill="currentColor" font-size="9px">1.1</text><rect x="436" y="117" width="24" height="24" fill="currentColor" fill-opacity="0.11" stroke="currentColor" stroke-opacity="0.25"/><text x="448" y="132" text-anchor="middle" fill="currentColor" font-size="9px">1.2</text><rect x="388" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.16" stroke="currentColor" stroke-opacity="0.25"/><text x="400" y="156" text-anchor="middle" fill="currentColor" font-size="9px">1.8</text><rect x="412" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.13" stroke="currentColor" stroke-opacity="0.25"/><text x="424" y="156" text-anchor="middle" fill="currentColor" font-size="9px">1.4</text><rect x="436" y="141" width="24" height="24" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.25"/><text x="448" y="156" text-anchor="middle" fill="currentColor" font-size="9px">1.6</text>
<text x="382" y="61" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">A</text><text x="382" y="85" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">B</text><text x="382" y="109" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">C</text><text x="382" y="133" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">D</text><text x="382" y="157" text-anchor="end" fill="currentColor" fill-opacity="0.45" font-size="9px">E</text>
<text x="400" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="424" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="448" y="38" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="424" y="192" text-anchor="middle" fill="currentColor" fill-opacity="0.6">k = 1 prediction</text>
</svg>
</div>

  *The same two terms as matrices: $U$ is $5 \times 2$ (one column per
  pattern, one row per user), $\Sigma$ holds $8.8$ and $6.0$ ($\sigma_3 = 0$,
  so the compact form drops it), and $V^\top$ is $2 \times 3$ (one row per
  pattern, one column per movie). Keeping only $\sigma_1$ gives the blurry
  $k = 1$ prediction: Ben, a pure romance fan, gets 1.8 for Die Hard. $k = 1$
  works only when $\sigma_1$ dominates; here $\sigma_2$ is close behind, and
  $k = 2$ is exact.*

- **Model compression**: replace a trained weight matrix with its truncated
  SVD.
- **LoRA**: assume the *fine-tuning update* $\Delta W$ is low-rank and learn
  it directly as $\Delta W = BA$ with a small inner dimension, a few patterns'
  worth of change instead of a whole new matrix.

## 1.7 Norms: what "big" means

Three common ML questions each reduce to measuring the size of one vector:

- How far apart are two embeddings $a$ and $b$? Their difference $a - b$
  is a single vector, and the distance between them is its size:
  $\|a - b\|$.
- How large are a model's weights? (the quantity weight decay controls)
- Is this gradient exploding? (the quantity gradient clipping checks)

Each needs a single number for "how big is this vector." A **norm** is that
number, and there is more than one useful way to define it.

The notation: $\|x\|$ (double bars) is the size of the vector $x$, with a
subscript naming which definition of size is in use. Measure the same
vector $x = [3, -4, 0]$ three ways:

- **L2, straight-line length:**
  $\|x\|_2 = \sqrt{3^2 + (-4)^2 + 0^2} = \sqrt{25} = 5$.
- **L1, total amount:** $\|x\|_1 = |3| + |{-4}| + |0| = 7$. No diagonal
  shortcuts: every coordinate counts in full.
- **L∞, worst coordinate:** $\|x\|_\infty = \max(3, 4, 0) = 4$. Only the
  biggest entry matters.

Same vector; sizes 5, 7, and 4. There is no single "right" size. Each norm is
the right tool somewhere in ML:

**L2 is the norm of weight decay.** The squared version
$\|w\|_2^2 = \sum_i w_i^2$ appears everywhere because it is smooth and its
gradient is plain: $\nabla \|w\|_2^2 = 2w$ (the 1-D check:
$\frac{d}{dw} w^2 = 2w$). Add it to the loss and each weight feels a pull
toward zero *proportional to its own size*: big weights are pulled hard, small
weights barely at all. So everything shrinks smoothly, and almost nothing is
pushed to exactly zero - the pull weakens as the weight gets small.

**L1 is the norm of sparsity.** The derivative of $|w|$ is $+1$ or $-1$: a
pull toward zero of *constant strength*, no matter how small the weight
already is. A weight whose contribution to the loss is worth less than that
constant pull gets driven to exactly $0$ and held there. Why held: at
$w = 0$, moving in *either* direction raises the penalty at that same
constant rate, so unless the loss gradient at $0$ is larger than that rate,
$0$ is where the weight stays. That is the lasso,
and it is why "L1 selects features." The same conclusion, geometrically.
The set of all vectors of size at most 1 under a given norm
($\|x\| \le 1$) is called that norm's **unit ball**;
the name comes from L2, whose version really is round, but each norm has
its own shape - the figure below shows L1's is a diamond and L∞'s is a
square. A constrained optimum lands on the unit ball's surface, and L1's
diamond has corners exactly on the axes: points where the other
coordinates are zero.

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

*Every point with $\|x\| = 1$ under each norm. L1's corners (dots) sit on
the axes; constrained optima land on corners, zeroing the other coordinate.
L2's ball is round - no corners, nothing gets zeroed.*

**L∞ is the norm of adversarial examples.** "Change every pixel by at most
$\epsilon$" describes an L∞ ball of radius $\epsilon$ around the image:
the constraint caps the worst single coordinate and says nothing about the
total.

**Frobenius is L2 for matrices:** flatten the matrix and take ordinary L2,
$\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$.

Two things every norm gives you for free: a **distance** between vectors,
$\|a - b\|$, and a **normalization**, $x / \|x\|$, which rescales $x$ to
length 1 - the core move inside cosine similarity (§1.2), and, after
subtracting the mean and up to a constant factor, inside LayerNorm. The
one-line contrast to keep for interviews: *L2 shrinks, L1 selects.*
