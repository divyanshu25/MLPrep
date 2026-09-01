# Linear algebra for ML

## 1.1 Vectors are data points

Forget arrows in space for a second. In ML, a vector is just **a list of measurements
about one thing**. A house might be $x = [1400, 3, 1985]$ - square feet, bedrooms,
year built. A word in a transformer is a vector of 768 learned numbers. That's it.

The "arrow" picture is still useful for one reason: it gives us geometry. Once data
points are vectors, we can ask geometric questions: *how long is it? how similar are
two points? what direction does it point?* Those questions turn out to be the entire
substrate of ML.

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
  the steps. No index gymnastics required.
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
nothing along the anti-diagonal. Notice the two eigen-directions are orthogonal. That's not luck:
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

Read that off. The pair of amounts $(c_1, c_2)$ is $x$ described in the
eigenbasis - its **eigen-coordinates**; the two words name the same numbers,
and this file uses them interchangeably. In eigen-coordinates the map is
just $(c_1, c_2) \to (3c_1, c_2)$.
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
  $|\lambda| > 1$ grows exponentially and whatever has $|\lambda| < 1$ shrinks
  toward zero -
  so $A^k v$ ends up pointing along the largest-$|\lambda|$ eigenvector. Where
  this appears in ML: an RNN applies the *same* update rule at every timestep,
  so backprop through $T$ timesteps multiplies the gradient by roughly the same
  matrix $T$ times - literally the $A^k$ story, which is why RNN gradients
  explode or vanish over long sequences.
- **Power iteration and PageRank.** That exponential growth is also an
  *algorithm*.
  Want the top eigenvector of some matrix $A$? Start from a random vector and
  repeat: multiply by $A$, rescale back to unit length. Every multiply grows the
  component along the biggest-$|\lambda|$ eigenvector faster than all the
  others, so after enough rounds those others are negligible and the vector has
  landed on the top eigenvector; the rescaling only keeps the numbers from
  overflowing. That loop is **power iteration**. **PageRank** is this idea
  applied to the web:
  - Build a matrix $A$ from the link structure. A reader on page $i$ is
    assumed to click one of its outgoing links at random (3 links, each
    with probability $\frac13$), so entry $(j, i)$ of $A$ is the
    probability that a reader on page $i$ goes to page $j$ next.
  - Let $p_i$ be the fraction of all readers currently on page $i$. After
    everyone clicks once, the new fraction on page $j$ is
    $\sum_i A_{ji}\, p_i$: readers arrive at $j$ from every page that
    links to it. That sum is entry $j$ of $Ap$. So one round of clicking
    turns the vector $p$ into the vector $Ap$.
  - Define a page's importance as its share of readers once this process
    settles: a $p$ that one more round of clicking does not change,
    $Ap = p$. That equation says $p$ is an eigenvector of $A$ with
    eigenvalue $1$.
  - Google's original ranking computed $p$ by power iteration: start from
    any spread of readers and keep multiplying by $A$ until the vector
    stops changing.
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
    measures how elongated the bowl is, and an elongated bowl slows
    gradient descent badly: the gradient points across the narrow valley
    rather than along it, so the iterates zig-zag (the figures below).

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
path is gradient descent, and the gradient is perpendicular to the contour
underfoot: in the round bowl that points at the minimum, in the elongated
bowl it points across the valley, so descent zig-zags. This picture is why
momentum and Adam exist.*

## 1.6 SVD: every matrix is rotate → stretch → rotate

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

- **Step 1 is "multiply by $Q^{-1}$".** Splitting $x$ means finding the
  amounts $c_1, c_2$ with $c_1 v_1 + c_2 v_2 = x$. Stack the two unknown
  amounts into a vector,
  $c = \begin{bmatrix} c_1 \\ c_2 \end{bmatrix}$ (for this run,
  $c = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$). §1.4's blend rule (1.4c)
  says $Qc = c_1 v_1 + c_2 v_2$, so the equation to solve is $Qc = x$:
  multiply both sides by $Q^{-1}$ to get $c = Q^{-1}x$.
  Here $Q^{-1} = \tfrac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$.
  Check: $Q^{-1} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$,
  the amounts step 1 found. ✓
- **Step 2 is "multiply by $\Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$".**
  A diagonal matrix multiplies coordinate $i$ by $\lambda_i$, nothing else.
  Check: $\Lambda \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 6 \\ 1 \end{bmatrix}$. ✓
- **Step 3 is "multiply by $Q$".** Step 2 handed over the *scaled* amounts
  $\Lambda c = \begin{bmatrix} 6 \\ 1 \end{bmatrix}$. Multiplying by $Q$
  turns amounts back into an ordinary vector, by the same blend rule as
  step 1: each amount multiplies its own column of $Q$ (its eigenvector),
  and the results add.
  Check: $Q \begin{bmatrix} 6 \\ 1 \end{bmatrix} = 6 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + 1 \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 7 \\ 5 \end{bmatrix}$:
  exactly the rebuild in numbered step 3, and the final answer $Ax$. ✓

The intuition for the three factors: they split $A$'s job into *translate,
act, translate back*.

- $Q^{-1}$ is the translator **into** eigen-coordinates: give it an ordinary
  vector, and it returns the amounts - how much of each eigenvector the
  vector contains.
- $\Lambda$ does the actual work, and in this coordinate system the work is
  trivial: stretch each coordinate by its own $\lambda_i$, with no mixing
  between coordinates.
- $Q$ is the translator **back**: give it amounts, and it rebuilds the
  ordinary vector.

So $A = Q \Lambda Q^{-1}$, read right to left, says: change to the basis
where $A$ is diagonal, apply the stretch, change back.

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

*$x = [3,1]$ through (1.6a): $Q^{-1}$ turns the eigen-directions (dashed)
onto the axes, so the coordinates become the amounts $(2,1)$; $\Lambda$
stretches to $(6,1)$; $Q$ maps back, landing on $Ax = [7,5]$.*

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

**The problem.** Formula (1.6a) needs a square matrix, and a real data
matrix is rectangular: $n$ rows × $d$ columns. The running example for this
part:

$$B = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$$

Three rows, two columns: $B$ takes 2-number inputs to 3-number outputs, so
"does the output point the same direction as the input?" cannot even be
asked - $B$ has no eigenvectors at all. What plays the role of
eigendecomposition here?

**What has to change.** In (1.6a), $Q^{-1}$ translated the input into
amounts and $Q$ translated back: one matrix and its inverse, possible only
because inputs and outputs lived in the same space. For a rectangular
matrix the two translators handle different spaces:

- the input-side translator must be $d \times d$ (it processes $d$-number
  inputs),
- the output-side translator must be $n \times n$ (it produces $n$-number
  outputs),

so they cannot be one matrix and its inverse. The fix is to allow two
*unrelated* rotations, one per side, and that is the only change.

**The proposal:**

$$A = U \Sigma V^\top \tag{1.6b}$$

- **$V^\top$ produces the input's coordinates in the $v$ basis.** For
  $x = c_1 v_1 + c_2 v_2$,
  $V^\top x = \begin{bmatrix} c_1 \\ c_2 \end{bmatrix}$: the amounts of $x$
  along a set of perpendicular unit directions $v_1, v_2, \dots$ - the same
  job $Q^{-1}$ did in Part A (there the amounts were eigen-coordinates; here
  the $v_i$ stand where the eigenvectors stood). Why it works: row $i$ of
  $V^\top$ is $v_i$, so entry $i$ of $V^\top x$ is $v_i \cdot x$, and the
  dot product with a unit vector reads off the amount along it (§1.2).
- **$\Sigma$ does the stretches.** Axis $i$ is scaled by $\sigma_i \ge 0$:
  the **singular values**, conventionally numbered biggest first. (This step also moves from
  $d$ input dimensions to $n$ output dimensions, padding with zeros.)
- **$U$ rebuilds an output vector from the scaled amounts.** Each amount
  multiplies its own output direction and the results add:
  $U$ applied to amounts $(a_1, a_2, \dots)$ gives
  $a_1 u_1 + a_2 u_2 + \dots$ - the same rebuild job $Q$ did in Part A,
  now with the output directions $u_i$ as the columns.

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

*The unit circle always maps to an ellipse: rotations keep circles round,
the stretch makes the ellipse (radii $\sigma_1, \sigma_2$), the last
rotation turns it into place. The inputs $v_1, v_2$ land on
$\sigma_1 u_1, \sigma_2 u_2$.*

**Finding the factors.** So far (1.6b) is a proposal naming three unknowns.
The tool for finding them is Part A's machinery, but that needs a square,
symmetric matrix, and $A$ is neither. So manufacture one out of $A$:
multiplying $A$ by its own transpose is the way to get a square matrix from
a rectangular one, and there are exactly two ways to do it, $A^\top A$
($d \times d$) and $A A^\top$ ($n \times n$). Take the first. It has both
properties the machinery needs:

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
   square roots to get the $\sigma_i$. Because $A^\top A$ is symmetric, the
   spectral theorem from §1.5 guarantees the $v_i$ are orthogonal to each
   other - exactly what "$V$ is a rotation" requires.
3. Recover each output direction from $A$ itself: $u_i = A v_i / \sigma_i$.
   The $u_i$ inherit orthogonality from the $v_i$: for $i \ne j$,
   $u_i \cdot u_j = \frac{(A v_i) \cdot (A v_j)}{\sigma_i \sigma_j}
   = \frac{v_i \cdot (A^\top A\, v_j)}{\sigma_i \sigma_j}
   = \frac{\sigma_j^2 \, (v_i \cdot v_j)}{\sigma_i \sigma_j} = 0$,
   so $U$ is a rotation too.

**The recipe, run on numbers - twice.** First on the §1.5 matrix, so every
symbol gets a number on familiar ground:

1. $A^\top A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$
   (here $A^\top = A$ because $A$ is symmetric).
2. Eigendecompose $\begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$. Multiply
   it by $[1,1]$: the result is $[9, 9] = 9 \cdot [1,1]$. Multiply it by
   $[1,-1]$: the result is $[1,-1] = 1 \cdot [1,-1]$. So its eigenvectors are
   $[1,1]$ and $[1,-1]$, with eigenvalues $9$ and $1$. Orthogonal, as
   promised: $[1,1] \cdot [1,-1] = 1 - 1 = 0$. Normalize the
   eigenvectors to unit length: $v_1 = [1,1]/\sqrt{2}$,
   $v_2 = [1,-1]/\sqrt{2}$. Square-root the eigenvalues:
   $\sigma_1 = \sqrt{9} = 3$, $\sigma_2 = \sqrt{1} = 1$.
3. Output directions:
   $u_1 = A v_1 / \sigma_1 = \left([3,3]/\sqrt{2}\right) / 3 = [1,1]/\sqrt{2} = v_1$,
   and $u_2 = A v_2 / \sigma_2 = \left([1,-1]/\sqrt{2}\right) / 1 = v_2$.

For this symmetric $A$, the recipe returns $u_i = v_i$: each direction lands
on itself. For a non-symmetric matrix, step 3 is exactly where $U$ and $V$
come apart - so run the recipe once more, this time on the rectangular
running example from the top of Part B:

$$B = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$$

$B$ has no eigenvectors, as noted there. The recipe does not care:

1. $B^\top B$ is $2 \times 2$; entry $(i, j)$ is column $i$ dotted with
   column $j$. Column 1 is $[1, 0, 1]$ and column 2 is $[1, 1, 0]$, so
   $B^\top B = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ - exactly the
   §1.5 matrix again.
2. So we already know its eigendecomposition: $v_1 = [1,1]/\sqrt{2}$ with
   eigenvalue 3, $v_2 = [1,-1]/\sqrt{2}$ with eigenvalue 1. Therefore
   $\sigma_1 = \sqrt{3}$ and $\sigma_2 = 1$. The $v_i$ have 2 entries each:
   they live in the input space.
3. $u_1 = B v_1 / \sigma_1 = [2, 1, 1]/\sqrt{6}$ and
   $u_2 = B v_2 / \sigma_2 = [0, -1, 1]/\sqrt{2}$. These have 3 entries
   each: they live in the output space. Asking whether $u_1$ equals $v_1$ is
   not even meaningful now - they are different lengths. Orthogonality still
   holds where it should: $u_1 \cdot u_2 = (0 - 1 + 1)/\sqrt{12} = 0$. ✓

Assemble the results into the three matrices: the $u_i$ stand side by side as
the columns of $U$, the $\sigma_i$ sit on $\Sigma$'s diagonal, and the
$v_i^\top$ stack as the rows of $V^\top$:

$$B = U \Sigma V^\top =
\begin{bmatrix} \tfrac{2}{\sqrt{6}} & 0 \\ \tfrac{1}{\sqrt{6}} & -\tfrac{1}{\sqrt{2}} \\ \tfrac{1}{\sqrt{6}} & \tfrac{1}{\sqrt{2}} \end{bmatrix}
\begin{bmatrix} \sqrt{3} & 0 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} \tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\ \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \end{bmatrix}$$

Multiply the three out and $B = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$
comes back exactly. (This is the *compact* form: $U$ keeps only the columns
with nonzero $\sigma$, so the shapes are $(3 \times 2)(2 \times 2)(2 \times 2)$.
The "full" convention pads $U$ to $3 \times 3$ with a third orthogonal column
and $\Sigma$ to $3 \times 2$ with a zero row - the between-dimensions padding
mentioned earlier; the product is the same either way.)

Why do these factors really rebuild $B$? By construction, $B v_1 = \sigma_1
u_1$ and $B v_2 = \sigma_2 u_2$: the factorization does the right thing to
$v_1$ and $v_2$. And $v_1, v_2$ span the whole input space, so by the recipe
argument of §1.4, knowing what happens to them pins down what happens to
*every* input.

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
  rotated, stretched, and rotated it. (The shear is also proof that "square"
  is not enough for eigendecomposition: it is square, but every eigenvector it
  has lies along $[1, 0]$ - one direction, when diagonalizing 2-D space needs
  two. No eigendecomposition exists for it. Its SVD does.)

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
<svg viewBox="0 0 465 205" role="img" aria-label="The 5 by 3 ratings matrix equals 8.8 times u1 v1 transpose plus 6.0 times u2 v2 transpose, using the true SVD numbers; archetype 1 is overall appetite, archetype 2 contrasts romance against action" style="max-width:485px;width:100%;height:auto;font-size:11px;font-family:inherit">
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
<text x="204" y="192" text-anchor="middle" fill="#6366f1" font-weight="600" font-size="10px">archetype 1: overall appetite</text>
<text x="288" y="110" text-anchor="middle" fill="currentColor" font-size="14">+</text>
<text x="308" y="109" text-anchor="middle" fill="#f43f5e" font-size="10px">6.0 &#183;</text>
<rect x="326" y="45" width="24" height="24" fill="#f43f5e" fill-opacity="0.27" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="60" text-anchor="middle" fill="currentColor" font-size="9px">-.61</text><rect x="326" y="69" width="24" height="24" fill="#f43f5e" fill-opacity="0.31" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="84" text-anchor="middle" fill="currentColor" font-size="9px">.68</text><rect x="326" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.18</text><rect x="326" y="117" width="24" height="24" fill="#f43f5e" fill-opacity="0.11" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="132" text-anchor="middle" fill="currentColor" font-size="9px">-.24</text><rect x="326" y="141" width="24" height="24" fill="#f43f5e" fill-opacity="0.13" stroke="currentColor" stroke-opacity="0.25"/><text x="338" y="156" text-anchor="middle" fill="currentColor" font-size="9px">.28</text>
<text x="338" y="38" text-anchor="middle" fill="#f43f5e">u&#8322;</text>
<text x="364" y="109" text-anchor="middle" fill="currentColor">&#215;</text>
<rect x="376" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.25"/><text x="388" y="108" text-anchor="middle" fill="currentColor" font-size="9px">-.45</text><rect x="400" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.16" stroke="currentColor" stroke-opacity="0.25"/><text x="412" y="108" text-anchor="middle" fill="currentColor" font-size="9px">-.36</text><rect x="424" y="93" width="24" height="24" fill="#f43f5e" fill-opacity="0.37" stroke="currentColor" stroke-opacity="0.25"/><text x="436" y="108" text-anchor="middle" fill="currentColor" font-size="9px">.82</text>
<text x="388" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">DH</text><text x="412" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">MM</text><text x="436" y="89" text-anchor="middle" fill="currentColor" fill-opacity="0.45" font-size="9px">Ti</text>
<text x="412" y="75" text-anchor="middle" fill="#f43f5e">v&#8322;&#7488;</text>
<text x="380" y="192" text-anchor="middle" fill="#f43f5e" font-weight="600" font-size="10px">archetype 2: romance &#8722; action</text>
</svg>
</div>

  *The 5 × 3 ratings matrix split into its two SVD pieces. Piece 1 (weight 8.8): $v_1^\top = [.64, .51, .57]$ has all-positive,
  roughly equal entries - it scores every movie about the same - so a user's
  $u_1$ amount can only mean overall enthusiasm; that reading of the numbers
  is the label "overall appetite" (the SVD outputs vectors, naming them is on
  us). Piece 2
  (weight 6.0): $v_2^\top = [-.45, -.36, .82]$ contrasts Titanic against the
  action films - romance fans score positive (Ben $.68$), action fans negative
  (Ana $-.61$). Cara's row: $8.8(.62)\,v_1^\top + 6.0(.18)\,v_2^\top =
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

  *The same two pieces assembled into matrices: $U$ is $5 \times 2$ (one
  column per archetype, rows in the same user order), $\Sigma$ holds the importances
  $8.8$ and $6.0$ ($\sigma_3 = 0$, so the compact form drops it), and
  $V^\top$ is $2 \times 3$ (one row per archetype, columns marked by movie).
  Keeping only $\sigma_1$ gives the blurry $k = 1$ prediction - Ben, a pure
  romance fan, gets 1.8 for Die Hard. $k = 1$ works only when $\sigma_1$
  dominates; here $\sigma_2$ is close behind, and $k = 2$ is exact.*

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

## 1.7 Norms: what "big" means

Three ML questions are secretly the same question. How far apart are two
embeddings? How large are a model's weights (the thing weight decay controls)?
Is this gradient exploding (the thing gradient clipping controls)? Each needs
one number for "how big is this vector." A **norm** is that number - and there
is more than one useful way to define it.

Measure the same vector $x = [3, -4, 0]$ three ways:

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
and it is why "L1 selects features." The same conclusion, geometrically: the
constrained optimum lands on the unit ball's surface, and L1's ball has
corners exactly on the axes - points where the other coordinates are zero.

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
length 1 - the core move inside cosine similarity (§1.2) and LayerNorm. The
one-line contrast to keep for interviews: *L2 shrinks, L1 selects.*
