# Deep Learning: CNNs, RNNs, Transformers - from first principles

The theme of this whole file: **every architecture is a prior about the data,
baked into the weights' structure.** CNNs bake in "nearby pixels matter and
statistics are the same everywhere." RNNs bake in "process a sequence one step
at a time with the same rule." Transformers bake in almost nothing - "let every
token look at every other token and learn what matters" - and pay for that
generality with $O(n^2)$ compute and a hunger for data.

Companion code (all runnable, CPU-only):

- `attention_scratch.ipynb` - attention in numpy, verified against PyTorch
- `mini_gpt.ipynb` - a complete GPT trained on an embedded corpus in ~5 seconds
- `cnn_scratch.ipynb` - conv forward/backward from scratch, gradient-checked

---

## 1. Convolutional Neural Networks

### 1.1 Deriving convolution from two priors

Start from a fully-connected layer on an image and ask what's wrong with it.
A $5\times 5$ grayscale image is a 25-dim vector; a dense layer to 25 outputs
has a $25 \times 25$ weight matrix: every output pixel gets its own private
weight for every input pixel. Now impose two beliefs about images:

1. **Locality.** An output pixel should only depend on input pixels *near* it.
   So zero out every weight connecting distant pixels. Each output now sees
   only a small window, say $3\times 3$: the weight matrix becomes sparse and
   banded.
2. **Translation equivariance.** A cat detector should work the same in the
   top-left and bottom-right corner. So force every window to use the *same*
   9 weights. This is **weight sharing**: the sparse banded matrix now has
   only 9 distinct values, repeated along the diagonal band.

A dense layer, plus locality, plus weight sharing, **is** a convolution.
Nothing else. The $25\times25 = 625$ free parameters collapsed to $9$ (+1
bias), and the layer is equivariant by construction: shift the input, the
output shifts identically. (Strictly, deep learning "convolution" is
cross-correlation - no kernel flip - but nobody cares because the kernel is
learned; a flipped optimum is just as reachable.)

### 1.2 A 3×3 kernel over a 5×5 input, by hand

Input $X$ (a ramp, 0..24) and kernel $K$ (all ones - a "sum the patch"
detector, easy to check mentally):

$$
X = \begin{pmatrix}
0 & 1 & 2 & 3 & 4\\
5 & 6 & 7 & 8 & 9\\
10 & 11 & 12 & 13 & 14\\
15 & 16 & 17 & 18 & 19\\
20 & 21 & 22 & 23 & 24
\end{pmatrix},
\qquad
K = \begin{pmatrix} 1&1&1\\ 1&1&1\\ 1&1&1 \end{pmatrix}
$$

Slide $K$ over every position where it fully fits (stride 1, no padding).
The top-left output: overlay $K$ on the top-left $3\times3$ patch of $X$,
multiply elementwise, sum:

$$
\underbrace{\begin{pmatrix}0&1&2\\5&6&7\\10&11&12\end{pmatrix}}_{\text{patch}}
\odot K \;\to\; 0+1+2+5+6+7+10+11+12 = 54
$$

Slide one step right: the patch is columns 1–3, sum $= 1+2+3+6+7+8+11+12+13 = 63$
(each of the 9 entries grew by 1, so the sum grew by 9 - sanity check).
Doing all 9 positions:

$$
Y = X * K = \begin{pmatrix} 54 & 63 & 72\\ 99 & 108 & 117\\ 144 & 153 & 162 \end{pmatrix}
\qquad \text{shape } 3\times 3 = (5-3+1)\times(5-3+1)
$$

Now a kernel that actually *detects* something. Take an image with a bright
vertical bar in the middle column and a vertical-edge kernel:

$$
X = \begin{pmatrix}
0&0&1&0&0\\ 0&0&1&0&0\\ 0&0&1&0&0\\ 0&0&1&0&0\\ 0&0&1&0&0
\end{pmatrix},
\qquad
K = \begin{pmatrix} 1&0&-1\\ 1&0&-1\\ 1&0&-1 \end{pmatrix}
$$

$K$ computes (left column of patch) − (right column of patch): it fires when
brightness changes left-to-right. Output:

$$
Y = \begin{pmatrix} -3 & 0 & 3\\ -3 & 0 & 3\\ -3 & 0 & 3 \end{pmatrix}
$$

Read it: $-3$ where the bar enters the patch from the right (dark→bright
going right), $+3$ where it exits (bright→dark), $0$ where the bar sits dead
center and cancels. Nine numbers, and you have an edge detector that works at
every position in the image. That is the entire pitch for CNNs.

### 1.3 The arithmetic: shapes, parameters, FLOPs, receptive field

**Output shape.** With input size $H$, kernel $k$, padding $p$, stride $s$:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

Intuition: after padding you have $H+2p$ pixels; the kernel's left edge can
start anywhere in the first $H+2p-k+1$ positions; stride keeps every $s$-th.
Common settings: "same" conv is $k=3, p=1, s=1$ (output size = input size);
downsampling conv is $s=2$ (halves the resolution).

**Parameters.** A conv layer with $C_{in}$ input channels, $C_{out}$ output
channels, kernel $k\times k$:

$$
\#\text{params} = C_{out}\,(C_{in}\, k^2 + 1)
$$

Each of the $C_{out}$ filters is a $C_{in}\times k \times k$ block (it looks
at *all* input channels in its window) plus one bias. **Note what's absent:**
$H$ and $W$. Parameter count is independent of image size - that's weight
sharing paying off.

**FLOPs.** Each output element is a dot product of length $C_{in}k^2$, i.e.
$C_{in}k^2$ multiply-adds $= 2C_{in}k^2$ FLOPs. Multiply by the number of
output elements:

$$
\text{FLOPs} = 2\, C_{in}\, k^2 \cdot C_{out} \cdot H_{out} W_{out}
$$

**Worked example** (a VGG-style layer): $C_{in}=64$, $C_{out}=128$, $k=3$,
input $56\times56$, same padding.

- Params: $128 \cdot (64\cdot 9 + 1) = 128 \cdot 577 = 73{,}856$
- FLOPs: $2\cdot 64 \cdot 9 \cdot 128 \cdot 56 \cdot 56 = 2\cdot 231{,}211{,}008 \approx 4.6\times 10^8$

Note the asymmetry: 74K parameters but 462 MFLOPs. Conv layers are
compute-heavy and parameter-light; fully-connected layers are the opposite.
(In old VGG, the FC layers held ~90% of the parameters and ~1% of the FLOPs.)

**Receptive field** = how many input pixels influence one output activation.
For stride-1 convs stacked, each $k\times k$ layer grows it by $k-1$:

$$
r_L = 1 + \sum_{l=1}^{L} (k_l - 1)\prod_{i<l} s_i
$$

(The $\prod s_i$ term: after a stride-2 layer, one step in the output equals
two steps in the input, so later layers grow the receptive field twice as
fast.) Worked example: three stacked $3\times3$ stride-1 convs give
$r = 1 + 2 + 2 + 2 = 7$ - the same view as one $7\times7$ conv, but with
$3\cdot(3^2 C^2) = 27C^2$ weights instead of $49C^2$, and two extra
nonlinearities in between. This is VGG's whole thesis.

### 1.4 Pooling

Max-pool $2\times2$ stride 2: take the max of each non-overlapping
$2\times2$ window; halves $H$ and $W$, keeps channels. Zero parameters. It
buys (a) downsampling, so later layers see a coarser, wider view cheaply, and
(b) small local translation *invariance* - the max output is the same
whichever of the 4 positions fired. Backward pass: the gradient flows entirely to the
argmax position, zero to the others (max is locally the identity on the
winner). Average pooling spreads the gradient uniformly. Modern nets often
replace pooling with stride-2 convs (learned downsampling) and end with
**global average pooling** - average each channel's whole map to one number  - 
which kills the giant FC layers of the AlexNet era.

### 1.5 Why CNNs beat MLPs on images: count the parameters

Concrete example: $224\times224\times3$ input (=150,528 numbers).

- **MLP**: one hidden layer of just 1000 units already costs
  $150{,}528 \times 1000 \approx 1.5\times 10^8$ parameters - 150M for one
  mediocre layer, and it has *no idea* that pixel $(10,10)$ is near
  $(10,11)$: permute the pixels and the MLP learns just as well (i.e. it
  can't exploit spatial structure; it must rediscover it from data, pixel by
  pixel, position by position).
- **CNN**: a $3\times3$ conv from 3 to 64 channels costs
  $64\cdot(3\cdot 9+1) = 1{,}792$ parameters - about **100,000× fewer**  - 
  and translation equivariance means a feature learned anywhere is available
  everywhere, so every training image teaches every image location.

Fewer parameters isn't the point per se; the point is the *right inductive
bias*: the CNN's hypothesis space is a tiny, well-chosen slice of the MLP's,
so it needs far less data to find a good function.

### 1.6 Classic architectures, one paragraph each

**LeNet-5 (1998).** The template: [conv → pool] × 2 → FC → FC → output, tanh
activations, trained with backprop on digits (MNIST-scale). It established
the pattern every CNN still follows - alternate feature extraction and
downsampling, then classify - but at ~60K parameters it was limited by 1998
compute and data.

**AlexNet (2012).** LeNet scaled up ~1000× and made trainable: ReLU instead
of tanh/sigmoid (no saturation → gradients survive depth, ~6× faster
training), dropout to fight overfitting in its huge FC layers, data
augmentation, and - crucially - GPU training on ImageNet's 1.2M images.
Dropped ImageNet top-5 error from ~26% to ~16% and single-handedly restarted
deep learning.

**VGG (2014).** The simplification: use *only* $3\times3$ convs and
$2\times2$ pools, stacked deep (16–19 layers), doubling channels after each
pool. Its insight is §1.3's arithmetic: two/three stacked 3×3 convs match a
5×5/7×7 receptive field with fewer parameters and more nonlinearity. Proved
that depth with uniform small kernels is what matters - but at 138M
parameters (mostly in FC layers) it also showed the cost.

**ResNet (2015).** The fix for depth itself. Plain 30+ layer nets showed
*higher training error* than shallow ones - not overfitting, an
**optimization** failure: a deep net can't even learn the identity map
easily, because that requires many stacked nonlinear layers to jointly
approximate identity. ResNet reformulates each block to compute
$y = x + F(x)$: the block learns a *residual* $F$, and "do nothing" is now
$F=0$ - i.e. just push weights to zero, which is trivially reachable (and
where weight decay already pushes). For gradients:
$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$, so the
backward signal always has a clean identity path - an unattenuated gradient
highway through 100+ layers. Result: 152-layer nets that train *better* than
22-layer ones. Every serious architecture since - including the transformer  - 
is residual.

---

## 2. Recurrent Neural Networks

### 2.1 The recurrence, unrolled

An RNN processes a sequence $x_1, \dots, x_T$ by maintaining a state vector
and applying **the same function with the same weights at every step**:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b), \qquad y_t = W_{hy} h_t
$$

Unroll it in time and you're looking at a $T$-layer deep MLP whose layers all
share one weight matrix, with an input injected at each layer. That's the
RNN's prior: "the rule for absorbing the next element doesn't depend on the
position" - weight sharing across *time*, exactly analogous to the CNN's
weight sharing across *space*. Fixed parameter count for any sequence length;
$O(T)$ sequential steps that cannot be parallelized (each $h_t$ needs
$h_{t-1}$ - this, as much as gradients, is why transformers won).

### 2.2 Backprop through time (BPTT)

Training = unroll for $T$ steps, sum the per-step losses
$L = \sum_t L_t$, and backprop through the unrolled graph. Because $W_{hh}$
is used at every step, its gradient is a **sum over all timesteps**, and each
term contains a chain of Jacobians carrying the error from step $t$ back to
step $k$:

$$
\frac{\partial L_t}{\partial h_k}
= \frac{\partial L_t}{\partial h_t}
\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}
$$

### 2.3 Why gradients vanish/explode: the $W^\top$ product

Differentiate one step of $h_i = \tanh(a_i),\; a_i = W_{hh}h_{i-1} + \dots$:

$$
\frac{\partial h_i}{\partial h_{i-1}} = \operatorname{diag}\!\big(1 - h_i^2\big)\, W_{hh}
$$

So the $t-k$ step chain is a product of $t-k$ copies of
(diagonal damping matrix) × (the *same* matrix $W_{hh}$). Ignore the diagonal
for a second and think about $W^{t-k}$ acting on a vector. Write $W$'s
largest singular value as $\sigma_{\max}$ (for a symmetric/diagonalizable
view, think eigenvalues / spectral radius $\rho$). Then:

$$
\left\| \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} \right\|
\;\le\; \big(\gamma\, \sigma_{\max}(W_{hh})\big)^{t-k},
\qquad \gamma = \max |\tanh'| = 1
$$

- If $\sigma_{\max} < 1$ (in practice $\gamma\sigma_{\max}<1$, and $\gamma$
  is well below 1 whenever units saturate, since $\tanh'(a)=1-\tanh^2(a)\to 0$):
  the gradient norm decays **geometrically** in the time gap. A dependency 50
  steps back with per-step factor 0.9 arrives scaled by $0.9^{50} \approx 0.005$
  - the network is effectively blind beyond a short horizon. This is
  **vanishing gradients**: not zero gradients, but long-range terms drowned
  out by short-range ones in the sum.
- If $\sigma_{\max} > 1$: components of the gradient along the top singular
  direction grow geometrically - **exploding gradients**, visible as loss
  spikes/NaNs. The standard fix is **gradient clipping** (rescale $g$ to norm
  $\theta$ when $\|g\| > \theta$), which is crude but works because explosion
  is about magnitude, not direction.

The knife-edge: a *scalar* intuition is $w^{50}$, which is $\approx 0$ for
$w=0.9$ and $\approx 117$ for $w=1.1$. Only $w \approx 1$ transmits - and a
generic learned $W$ has no reason to sit exactly there. Vanishing is the
harder problem: clipping can't fix it because the information is *gone*
(dominated), not oversized. Fixing vanishing requires changing the
architecture so that the Jacobian is near identity **by construction**. Enter
gates.

### 2.4 LSTM and GRU: learned write / keep / read, and the gradient highway

The LSTM adds a **cell state** $c_t$ - a separate memory vector updated
*additively*, controlled by three gates (each a sigmoid of the current input
and hidden state, so each is a learned, data-dependent value in $(0,1)$ per
dimension):

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) &&\text{forget gate: \textbf{keep}?}\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) &&\text{input gate: \textbf{write}?}\\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) &&\text{output gate: \textbf{read}?}\\
\tilde c_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) &&\text{candidate content}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t &&\text{memory update}\\
h_t &= o_t \odot \tanh(c_t) &&\text{exposed state}
\end{aligned}
$$

Read the update line as a tiny RAM controller: per dimension, $f_t$ decides
how much of the old memory to **keep**, $i_t$ how much new content to
**write**, and $o_t$ how much of the memory to **read** out into the working
state $h_t$ that the rest of the network sees.

**Why this fixes vanishing gradients:** look at the Jacobian of the cell
state path:

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) \;(+\,\text{gate-dependent terms})
$$

No repeated multiplication by a weight matrix, no $\tanh$ squashing on the
main path - just an elementwise multiply by the forget gate. Where the
network sets $f_t \approx 1$ ("keep this"), the factor is $\approx 1$ and
gradients flow back through many steps essentially unchanged: the cell state
is a **gradient highway**, the same trick as ResNet's skip connection
($c_t = c_{t-1} + \text{stuff}$ vs $y = x + F(x)$), discovered years
earlier. And unlike a fixed skip, the gate is *learned and data-dependent*:
the model can hold one dimension for 500 steps and flush another every step.
(Practical detail: initialize $b_f > 0$, e.g. 1, so gates start open.)

**GRU** is the budget LSTM: merge cell and hidden state, merge forget/input
gates into one update gate $z_t$ (write exactly as much as you forget):
$h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t$, plus a reset gate for
the candidate. Two gates instead of three, ~25% fewer parameters, usually
comparable quality. The highway argument is the same:
$\partial h_t / \partial h_{t-1} \supset \operatorname{diag}(1 - z_t)$.

What LSTMs *don't* fix: the $O(T)$ sequential bottleneck, and the fact that
all history is crushed into one fixed-size vector. Both problems point
straight at attention.

---

## 3. Attention and Transformers (the centerpiece)

### 3.1 Deriving attention from first principles

Problem: token $i$ needs information from other tokens, but *which* others
depends on content ("it" needs its referent, wherever it is). A fixed-weight
average won't do; we want a **weighted average of other tokens'
information, where the weights are computed from the data**.

Build it up in four steps:

1. **A weighted average of values.** For each position $i$, output
   $o_i = \sum_j w_{ij} v_j$ where $v_j$ is a vector of "information token
   $j$ offers" and $\sum_j w_{ij} = 1,\ w_{ij}\ge 0$. Averages are smooth,
   differentiable, and don't grow with sequence length. All of attention is
   choosing the $w_{ij}$.
2. **Weights from similarity.** Give each token a **query** $q_i$ ("what I'm
   looking for") and a **key** $k_j$ ("what I contain"). Let relevance be the
   dot product $s_{ij} = q_i^\top k_j$ - the simplest differentiable
   similarity, and it's just a matmul, which hardware loves.
3. **Normalize with softmax:** $w_{ij} = \operatorname{softmax}_j(s_{ij})$,
   turning arbitrary scores into a positive, sum-to-1 distribution, with a
   temperature-like sharpness that the model controls via score magnitude.
4. **Make $q, k, v$ learned linear functions of the token representations:**
   $q_i = W_Q x_i$, $k_j = W_K x_j$, $v_j = W_V x_j$. The *same* $x_j$ plays
   different roles through different projections - what I seek, what I
   advertise, what I hand over are three different subspaces of me.

Stack all positions into matrices $Q, K, V \in \mathbb{R}^{T\times d}$ and
you get the famous one-liner:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

$QK^\top$ is a $T\times T$ matrix of *all pairwise* query-key similarities;
softmax is applied per row (each query normalizes over all keys); the result
times $V$ takes, per row, a convex combination of value vectors. It is a
**soft, differentiable dictionary lookup**: instead of retrieving the single
best match, retrieve everything, weighted by match quality.

**Self**-attention = $Q, K, V$ all come from the same sequence.
**Cross**-attention (e.g. translation decoder) = $Q$ from one sequence,
$K, V$ from another. Same formula.

### 3.2 Why divide by $\sqrt{d_k}$: the variance of a dot product

Let $q, k \in \mathbb{R}^{d_k}$ have i.i.d. entries with mean 0, variance 1
(roughly true at initialization). Then

$$
s = q^\top k = \sum_{i=1}^{d_k} q_i k_i
$$

Each term: $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ and
$\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$.
The $d_k$ terms are uncorrelated, and variances of uncorrelated sums add:

$$
\mathbb{E}[s] = 0, \qquad \operatorname{Var}(s) = d_k
$$

So at $d_k = 64$ the raw scores have standard deviation
$\sqrt{64} = 8$: typical logit gaps of $\pm 16$ or more. Softmax of logits
that spread out is essentially an argmax - one weight near 1, the rest near
0. Two bad things follow: (1) the softmax Jacobian
$\frac{\partial w}{\partial s} = \operatorname{diag}(w) - ww^\top \to 0$ as
$w$ approaches one-hot, so **gradients vanish** through the attention
weights, right at initialization when you need them most; (2) the model
starts with hard, arbitrary attention patterns it must unlearn. Dividing by
$\sqrt{d_k}$ restores $\operatorname{Var}(s/\sqrt{d_k}) = 1$ regardless of
$d_k$, keeping softmax in its soft, trainable regime.
`attention_scratch.ipynb` measures this: at $d_k=64$, $T=16$, mean row entropy
is ~0.46 nats unscaled vs ~2.37 nats scaled (max $\ln 16 \approx 2.77$).

### 3.3 Full worked example: attention on 3 tokens, $d=2$

Take three tokens with (post-projection) vectors, and for arithmetic's sake
let $Q = K = V = X$ (i.e. identity projections):

$$
x_1 = \begin{pmatrix}1\\0\end{pmatrix},\quad
x_2 = \begin{pmatrix}0\\1\end{pmatrix},\quad
x_3 = \begin{pmatrix}1\\1\end{pmatrix}
$$

**Step 1 - scores** $S = XX^\top$ (all pairwise dot products):

$$
S = \begin{pmatrix}
1 & 0 & 1\\
0 & 1 & 1\\
1 & 1 & 2
\end{pmatrix}
\qquad \text{e.g. } s_{31} = x_3^\top x_1 = 1\cdot1 + 1\cdot0 = 1
$$

**Step 2 - scale** by $1/\sqrt{d_k} = 1/\sqrt{2} \approx 0.7071$:

$$
S' = \begin{pmatrix}
0.7071 & 0 & 0.7071\\
0 & 0.7071 & 0.7071\\
0.7071 & 0.7071 & 1.4142
\end{pmatrix}
$$

**Step 3 - softmax each row.** Row 1: $e^{0.7071} = 2.028$, $e^0 = 1$, so
exponentials are $(2.028,\, 1,\, 2.028)$, sum $= 5.056$:

$$
w_1 = (0.401,\ 0.198,\ 0.401)
$$

Row 2 by symmetry: $w_2 = (0.198,\ 0.401,\ 0.401)$.
Row 3: $e^{0.7071}=2.028$ (twice), $e^{1.4142} = 4.113$, sum $= 8.170$:

$$
w_3 = (0.248,\ 0.248,\ 0.503)
$$

Every row sums to 1. Note token 3 ($=(1,1)$, similar to *both* others)
attends broadly; tokens 1 and 2 mostly split attention between themselves and
token 3, largely ignoring their orthogonal partner. Similarity in, weights
out.

**Step 4 - outputs** $o_i = \sum_j w_{ij} v_j$:

$$
\begin{aligned}
o_1 &= 0.401\,x_1 + 0.198\,x_2 + 0.401\,x_3 = (0.802,\ 0.599)\\
o_2 &= (0.599,\ 0.802)\\
o_3 &= 0.248\,x_1 + 0.248\,x_2 + 0.503\,x_3 = (0.752,\ 0.752)
\end{aligned}
$$

Each output is a blend of value vectors, pulled toward the tokens it found
similar. **Causal version** (each token sees only itself + the past): row 1
becomes $(1, 0, 0)$ so $o_1 = x_1$ exactly; row 2 renormalizes over
positions 1–2: exponentials $(1, 2.028)$, weights $(0.330, 0.670)$,
$o_2 = (0.330,\ 0.670)$; row 3 is unchanged since it already sees everything.

### 3.4 Multi-head attention: multiple subspace views

One head produces *one* attention pattern per token - one convex combination.
But a token may need several relations at once (subject→verb, pronoun→
referent, token→previous-token). **Multi-head attention** runs $h$ attentions
in parallel in *lower-dimensional subspaces*: split $d_{\text{model}}$ into
$h$ heads of $d_{\text{head}} = d_{\text{model}}/h$, give each head its own
$W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ (projecting to $d_{\text{head}}$), attend
independently, concatenate the $h$ outputs, and mix with $W_O$:

$$
\operatorname{MHA}(x) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

Because each head measures similarity in its *own* learned subspace, head 1
can implement "attend to the previous token" while head 2 implements "attend
to tokens with matching syntax." Total compute and parameter count are
roughly the same as one full-width head (you split the dimensions, not
multiply them); what you buy is $h$ *independent* attention distributions per
position instead of one. Empirically, trained heads specialize into
recognizable roles (positional heads, syntactic heads, induction heads).

### 3.5 Positional encodings: attention is permutation-blind

Look back at the formula: $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$
uses only *pairwise dot products of content*. Shuffle the input tokens and
the outputs shuffle identically - self-attention is **permutation
equivariant**. It literally cannot distinguish "dog bites man" from "man
bites dog": position is not in the computation anywhere. (Contrast: RNNs get
order from the recurrence, CNNs from the kernel's spatial layout.)

Fix: inject position into the token vectors before attention sees them.

- **Learned absolute** (GPT-2, our `mini_gpt.ipynb`): a trainable table
  $P \in \mathbb{R}^{T_{\max}\times d}$; input is $x_t = \text{tokemb}(t) + P[t]$.
  Simple, works, can't extrapolate past $T_{\max}$.
- **Sinusoidal** (original transformer):
  $P[t, 2i] = \sin(t/10000^{2i/d})$, $P[t, 2i+1] = \cos(t/10000^{2i/d})$ - 
  each dimension pair is a rotation at a different frequency (a "clock" from
  fast to slow), and $P[t+\Delta]$ is a *linear* function of $P[t]$, so
  relative offsets are easy to represent. No parameters.
- **Relative / rotary (RoPE, modern LLMs):** encode position *inside the
  attention score* so what matters is $t_i - t_j$, not absolute $t$: RoPE
  rotates $q$ and $k$ by position-dependent angles so that
  $q_i^\top k_j$ depends on the offset $i - j$. Generalizes better to long
  contexts. (Derived properly in §7.1.)

### 3.6 The full transformer block

One block (pre-LN, the modern standard - used in GPT-2 onward):

$$
\begin{aligned}
x &\leftarrow x + \operatorname{MHA}(\operatorname{LN}(x)) && \text{"communicate": tokens exchange info}\\
x &\leftarrow x + \operatorname{MLP}(\operatorname{LN}(x)) && \text{"compute": per-token processing}
\end{aligned}
$$

where $\operatorname{MLP}(x) = W_2\,\phi(W_1 x)$ with $W_1: d \to 4d$,
$W_2: 4d \to d$, $\phi$ = GELU. Notes on each piece:

- **Residual stream.** The additions mean each block *edits* a running vector
  rather than replacing it; the identity path gives gradients an unbroken
  highway from the loss to layer 1 (§1.6, §2.4 - same trick, third
  appearance).
- **Pre-LN vs post-LN.** Original transformer used post-LN:
  $x \leftarrow \operatorname{LN}(x + \text{sublayer}(x))$ - the LN sits *on*
  the residual path and rescales it, which destabilizes early training deep
  stacks (needed lr warmup to survive). Pre-LN keeps the residual path
  completely clean (only additions touch it) and normalizes just the input to
  each sublayer; it trains stably at essentially any depth. (More in §6.4.)
- **LayerNorm**, not BatchNorm: normalize each token's vector across its $d$
  features, $\operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$.
  No batch statistics → works with variable lengths, batch size 1, and
  identically at train/inference.
- **The MLP is ~2/3 of the parameters** per block
  ($8d^2$ vs $4d^2$ for attention's $W_{Q,K,V,O}$) and is applied to each
  position independently - attention is the only place tokens interact.

A GPT is: token embedding + positional embedding → $N$ such blocks → final LN
→ linear to vocab logits. That's the entire architecture of every modern LLM,
modulo details (RoPE, RMSNorm, SwiGLU, grouped-query attention).

### 3.7 Causal masking

For autoregressive language modeling - predict token $t{+}1$ from tokens
$\le t$ - position $i$ must not read positions $j > i$, or the model can
trivially cheat during training (the "answer" is in the input). Implement by
setting future scores to $-\infty$ *before* the softmax:

$$
s_{ij} \leftarrow \begin{cases} s_{ij} & j \le i \\ -\infty & j > i \end{cases}
\qquad\Rightarrow\qquad w_{ij} = 0 \text{ for } j > i
$$

(Must be pre-softmax: zeroing weights *after* softmax would break the
sum-to-1 normalization over the allowed positions.) The mask is a fixed
lower-triangular matrix. The payoff: with the mask, position $t$'s output
depends only on tokens $\le t$, so **one forward pass over a $T$-token
sequence yields $T$ valid training examples at once** - next-token prediction
losses at every position, in parallel. This is the parallelism RNNs never
had, and the reason transformers train so efficiently.

### 3.8 Complexity, and the KV-cache

**Training / full forward:** the score matrix $QK^\top$ is $T \times T$ and
costs $O(T^2 d)$ to build and $O(T^2 d)$ to apply to $V$; attention memory is
$O(T^2)$ per head (unless you use FlashAttention, which tiles the computation
to avoid materializing the matrix - same FLOPs, less memory traffic). The
MLP costs $O(T d^2)$. So per layer: $O(T^2 d + T d^2)$ - the quadratic term
dominates once $T > d$, which is why long context is expensive.

**Naive generation** would re-run the full forward on the whole prefix for
each new token: step $t$ costs $O(t^2 d)$, so $n$ tokens cost $O(n^3 d)$.
But notice what's redundant: at step $t$, the keys and values of tokens
$1..t{-}1$ are **identical to last step** (causality: earlier tokens can't
see the new one, so their representations don't change). So cache them.

**KV-cache:** store every layer's $K$ and $V$ for all generated positions.
Each new token then requires only: compute its own $q, k, v$ ($O(d^2)$ per
layer), append $k, v$ to the cache, and attend its single query against $t$
cached keys - $O(t \cdot d)$ per layer. Per-token cost is $O(t d + d^2)$
instead of $O(t^2 d)$: **linear in context length per token**, total
$O(n^2 d)$ for the sequence - same as one training forward pass, as it
should be. The price is memory:
$2 \cdot n_{\text{layers}} \cdot T \cdot d$ values per sequence
(e.g. a 7B model, 32 layers, $d = 4096$, $T = 4096$, fp16:
$2\cdot 32\cdot 4096\cdot 4096\cdot 2\,\text{B} \approx 2$ GB *per
sequence*), which is why inference at scale is a memory problem and why
tricks like grouped-query attention (share K/V across query heads, §7.2) and
quantized KV-caches exist. Note the cache helps only generation: in training
there is nothing to reuse - all positions are computed in one parallel pass
(see exercises).

---

## 4. Embeddings

### 4.1 One-hot → embedding = a row lookup

The dumbest encoding of token $i$ from a vocab of $V$: a one-hot vector
$e_i \in \{0,1\}^V$. It's huge, and every pair of tokens is equidistant  - 
"cat" is as far from "kitten" as from "the". Now feed a one-hot into a linear
layer $E \in \mathbb{R}^{V \times d}$:

$$
e_i^\top E = E[i,:] \qquad \text{(the } i\text{-th row)}
$$

Multiplying by a one-hot just *selects a row*. So an **embedding layer is
literally a linear layer specialized for one-hot inputs, implemented as a
table lookup** (`nn.Embedding` is `nn.Linear` without the matmul). The rows
are learned like any other weights: gradients flow only into the rows of the
tokens present in the batch. Trained on a real task, rows of tokens used in
similar contexts receive similar gradient pressure and end up close in the
$d$-dim space - similarity structure emerges from the objective, never
hand-coded.

### 4.2 word2vec in one breath

word2vec (2013) showed you don't need a "real" task: **predicting
co-occurrence is enough**. Skip-gram: for each word, predict the words within
a small window around it, via a dot product between center-word and
context-word vectors fed through softmax (in practice, negative sampling:
logistic regression separating true context pairs from randomly sampled fake
ones - cheap, no $V$-way softmax). Words appearing in similar contexts get
similar vectors (the distributional hypothesis, made differentiable). The
famous surprise was linear structure:
$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$ - directions in the space encode relations. Modern LLM token embeddings are
the same object, trained end-to-end with everything else; the difference is
that transformer layers then build *contextual* representations on top
("bank" gets different vectors in "river bank" vs "bank loan"), which static
word2vec vectors can't do.

---

## 5. Language modeling

### 5.1 Next-token prediction is just classification

A language model defines $p(x_1, \dots, x_n)$ by the chain rule:

$$
p(x_1, \dots, x_n) = \prod_{t=1}^{n} p(x_t \mid x_{<t})
$$

Each factor is a $V$-way **classification problem**: given the prefix, output
a softmax over the vocabulary; train with cross-entropy against the actual
next token. There is no new theory beyond a classifier; the notable fact is
that this one objective, at scale, forces the model to learn syntax, facts,
and reasoning, because they all reduce prediction error. As noted in §3.7,
causal masking lets one forward pass supply all $n$ classification losses in
parallel. Useful sanity number: an untrained model gives uniform
probabilities, so initial loss $\approx \ln V$ (our `mini_gpt.ipynb` prints
$\ln 36 = 3.58$ and indeed starts at ~3.76 before dropping).

### 5.2 Tokenization: BPE in five sentences

Byte-Pair Encoding starts with a base vocabulary of bytes/characters, so
*any* string is representable and there is no "unknown token." It counts the
most frequent adjacent pair of tokens in a training corpus and merges that
pair into a new single token, records the merge rule, and repeats - thousands
of times - until reaching the target vocabulary size (e.g. 50k). Frequent
strings ("the", " ing", "tion", common whole words) thereby become single
tokens, while rare words decompose into a few subword chunks - a learned
compression scheme balancing sequence length against vocabulary size. To
tokenize new text, apply the recorded merges in the order they were learned.
Consequences worth knowing: token boundaries are why models historically
struggled with character-level tasks (counting letters, reversing strings),
why numbers tokenize inconsistently, and why the same text costs different
token counts in different models.

### 5.3 Perplexity, derived from cross-entropy

The per-token average cross-entropy (in nats) of a model on a corpus:

$$
H = -\frac{1}{N} \sum_{t=1}^{N} \ln p(x_t \mid x_{<t}) \qquad \text{(this is exactly the training loss)}
$$

Perplexity is its exponential:

$$
\operatorname{PPL} = e^{H}
$$

Why exponentiate? Interpretation. If the model were uniformly torn between
$k$ equally likely choices at each step, each correct token would get
probability $1/k$, so $H = \ln k$ and $\operatorname{PPL} = k$. So perplexity
reads as the model's **effective branching factor**: "on average, as confused
as if choosing uniformly among PPL options." Random guessing over vocab $V$:
$\operatorname{PPL} = V$. Perfect prediction: $\operatorname{PPL} = 1$. Good
LLMs on English text sit in the low single digits per token. Caveats: it's
$2^H$ if you measured $H$ in bits ($\log_2$); comparisons are only valid with
the **same tokenizer** (per-token perplexity changes if the tokens change  - 
compare bits-per-byte across tokenizers).

### 5.4 Sampling: temperature, top-k, top-p

At generation time you have logits $z \in \mathbb{R}^V$ and must pick a
token. All standard methods reshape the softmax:

- **Temperature** $\tau$: sample from
  $p_i = \operatorname{softmax}(z/\tau)_i = \dfrac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$.
  Effect on a probability *ratio*: 
  $\frac{p_i}{p_j} = e^{(z_i - z_j)/\tau} = \left(\frac{p_i^{(1)}}{p_j^{(1)}}\right)^{1/\tau}$  - 
  temperature exponentiates odds. $\tau \to 0$: argmax (greedy).
  $\tau = 1$: the model's distribution. $\tau > 1$: flattened, more random.
  Concrete: logits $(2, 1, 0)$ give $p \approx (0.665, 0.245, 0.090)$ at
  $\tau=1$, $\approx (0.867, 0.117, 0.016)$ at $\tau=0.5$, and
  $\to (1/3, 1/3, 1/3)$ as $\tau \to \infty$.
- **Top-k**: keep only the $k$ highest-probability tokens, renormalize,
  sample. Kills the long tail of individually-unlikely-but-collectively-
  probable junk tokens (with $V = 50$k, the tail below rank 50 can carry
  substantial total mass; one bad sample can derail a generation).
- **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative
  probability $\ge p$ (e.g. 0.9), renormalize, sample. Adaptive version of
  top-k: when the model is confident the nucleus is 1–2 tokens; when
  uncertain it may be hundreds. Fixed $k$ can't do both.
- Greedy/beam search maximize likelihood but produce degenerate repetitive
  text for open-ended generation; sampling with $\tau \in [0.7, 1.0]$ +
  top-p is the usual default.

---

## 6. Modern topics interviewers ask about

### 6.1 Transfer learning and fine-tuning

Pretrain a big model on a huge generic task (next-token prediction on the
web; ImageNet classification), then adapt to your task: the early/middle
representations (edges→textures→parts; syntax→semantics) are broadly
reusable, so your small dataset only needs to teach the *task*, not the
*modality*. Options, in increasing cost: (a) zero/few-shot prompting - no
weight updates at all; (b) linear probe - freeze everything, train a new
head; (c) full fine-tuning - update all weights with a small learning rate
(~10–100× smaller than pretraining, to avoid destroying pretrained features  - 
"catastrophic forgetting"); (d) parameter-efficient fine-tuning (LoRA,
below). Rule of thumb: less data → freeze more.

### 6.2 LoRA in one paragraph

Full fine-tuning of a weight matrix $W \in \mathbb{R}^{d \times k}$ learns an
update $\Delta W$ with $dk$ parameters - for every adapted matrix in a 7B
model, and per task. **LoRA** (Low-Rank Adaptation) constrains the update to
rank $r \ll \min(d,k)$:

$$
W' = W + \Delta W = W + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}
$$

Any rank-$r$ matrix factors this way, so you parameterize $\Delta W$ by $B$
and $A$ directly: $r(d + k)$ trainable parameters instead of $dk$ - at
$d = k = 4096$, $r = 8$: 65K vs 16.8M, ~250× fewer. Freeze $W$, train only
$A, B$ (init $A$ random, $B = 0$ so training starts exactly at the
pretrained model), typically on the attention projections. The bet, verified
empirically: the *change* needed to adapt a pretrained model lives in a
low-dimensional subspace even though $W$ itself is high-rank. Bonuses: at
inference, merge $W + BA$ once - **zero added latency**; adapters are a few
MB, so you can keep one $W$ and swap tiny per-task adapters; optimizer state
(the real memory hog of full fine-tuning: Adam keeps 2 extra copies per
trainable param) shrinks proportionally.

### 6.3 RLHF sketch

Three stages. **(1) SFT:** supervised fine-tune the pretrained LM on
human-written demonstrations of good responses - teaches format and
instruction-following. **(2) Reward model:** collect human *comparisons*
(which of two responses is better - far easier for humans than writing ideal
answers or giving absolute scores), train a model $r_\phi(x, y)$ to predict
them via the Bradley–Terry loss
$-\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$. **(3) RL:** optimize the LM
(the "policy") to maximize reward, minus a KL penalty keeping it near the
SFT model:
$\max_\theta\; \mathbb{E}[\,r_\phi(x, y)\,] - \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{SFT}})$,
classically with PPO. The KL term prevents **reward hacking** - drifting to
degenerate text that games the imperfect learned reward. Worth knowing:
**DPO** collapses stages 2–3 into one supervised loss directly on preference
pairs (the KL-regularized RL problem has a closed form that lets the policy
itself act as an implicit reward model), trading some flexibility for far
simpler training.

### 6.4 Why LayerNorm + residual ordering matters (pre-LN vs post-LN)

Post-LN (original transformer): $x \leftarrow \operatorname{LN}(x + f(x))$.
The LN sits **on the residual path**, so a gradient flowing from the loss to
the embeddings passes through one LN per sublayer - dozens of multiplicative
rescalings - and there is no clean identity path left; empirically gradients
near the input are tiny at initialization and deep post-LN stacks diverge
without a careful learning-rate warmup. Pre-LN (GPT-2+):
$x \leftarrow x + f(\operatorname{LN}(x))$. The residual path is touched
**only by additions**; since $\frac{\partial}{\partial x}(x + f(\operatorname{LN}(x))) = I + (\text{stuff})$,
every layer's gradient contains an unattenuated identity term, exactly the
ResNet argument. Consequence: stable training at large depth, warmup optional.
Cost: the residual stream's magnitude grows with depth (each block adds to
it), which is why there's one final LN after the last block. This tiny
reordering is a classic interview question because it tests whether you
actually think about gradient paths rather than memorizing diagrams.

### 6.5 Mixed precision in one paragraph

Do the bulk math in 16-bit (fp16 or bf16) - half the memory and 2–8× the
throughput on tensor cores - while keeping a **master copy of weights and
the optimizer state in fp32**, because tiny gradient updates
($\text{lr} \times g \sim 10^{-7}$) underflow when added to weights in
16-bit. fp16 has a narrow exponent range, so small gradients flush to zero;
the fix is **loss scaling**: multiply the loss by $S$ (e.g. $2^{14}$) before
backward - every gradient scales by $S$ into representable range - then
divide by $S$ before the optimizer step, with $S$ adjusted dynamically on
overflow. bf16 keeps fp32's 8 exponent bits (same range, less mantissa
precision), which mostly eliminates the need for loss scaling and is why
modern training defaults to bf16. Accumulations (softmax sums, LayerNorm
statistics, matmul partial sums) still happen in fp32 internally.

---

## 7. Modern attention variants

Everything in §3 was the 2017 transformer. Modern LLMs run on a handful of
upgrades that interviews now treat as core material. (The systems side of
the LLM stack - scaling laws, distributed training, quantization, inference
engines - gets its own deep dive in module 7; here we cover what belongs
with the architecture.)

### 7.1 RoPE: rotary position embeddings, derived

Recall §3.5: attention is permutation-blind, and additive absolute
embeddings ($x_t + P[t]$) fix that crudely - position gets mixed into the
content and washed through every layer. RoPE's move: don't add position to
the embeddings at all. Inject it **multiplicatively into $q$ and $k$, as a
rotation**, so position enters exactly one place - the attention score  - 
and enters it *relatively*.

Take $d = 2$ (one "frequency"; real models apply this to each pair of
dimensions). Rotate the query at position $m$ by angle $m\theta$ and the
key at position $n$ by angle $n\theta$:

$$
R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix},
\qquad q_m = R(m\theta)\,q, \qquad k_n = R(n\theta)\,k
$$

Now compute the attention score between them:

$$
q_m^\top k_n = q^\top R(m\theta)^\top R(n\theta)\,k = q^\top R\big((n-m)\theta\big)\,k
$$

The middle step is the whole trick, so do the $2\times 2$ math instead of
waving at it. $R(a)^\top R(b)$ is

$$
\begin{pmatrix}\cos a & \sin a\\ -\sin a & \cos a\end{pmatrix}
\begin{pmatrix}\cos b & -\sin b\\ \sin b & \cos b\end{pmatrix}
=
\begin{pmatrix}\cos a\cos b + \sin a \sin b & -\cos a\sin b + \sin a\cos b\\
-\sin a\cos b + \cos a\sin b & \sin a\sin b + \cos a\cos b\end{pmatrix}
= R(b - a)
$$

by the angle-addition identities ($\cos a\cos b + \sin a\sin b = \cos(b-a)$,
$\sin a\cos b - \cos a\sin b = \sin(a - b)$). Rotations compose by adding
angles and transposing reverses one, so the absolute positions $m$ and $n$
enter the score **only through the gap $n - m$** - relative position, exact,
with nothing learned.

Tiny numbers. Let $\theta = 30°$ and $q = k = (1, 0)$.

- Position $m=1$: $q_1 = (\cos 30°, \sin 30°) = (0.866,\ 0.5)$.
  Position $n=3$: $k_3 = (\cos 90°, \sin 90°) = (0,\ 1)$.
  Score $= 0.866\cdot 0 + 0.5\cdot 1 = 0.5 = \cos 60° = \cos((3-1)\cdot 30°)$.
- Shift both positions by 4 (now 5 and 7): $q_5 = (\cos 150°, \sin 150°) =
  (-0.866,\ 0.5)$, $k_7 = (\cos 210°, \sin 210°) = (-0.866,\ -0.5)$.
  Score $= 0.75 - 0.25 = 0.5$. Same gap, same score, different absolute
  positions.

Full-dimensional version: split $d$ into $d/2$ pairs; pair $i$ rotates at
frequency $\theta_i = 10000^{-2i/d}$ - the same geometric ladder of fast
and slow "clocks" as the sinusoidal encoding of §3.5, but applied as
rotations to $q, k$ inside the score instead of additions to $x$. $V$ is
left unrotated: position affects *who attends to whom*, never *what gets
handed over*.

Why it beats learned absolute embeddings (the interview answer):

1. **Relative structure is built in.** "Attend to the token 2 back" is one
   pattern, shared across all positions; with a learned absolute table it
   must be re-expressed as a content match between $P[i]$ and $P[i-2]$ for
   every $i$ separately.
2. **Extrapolation.** There is no table to run off the end of: scores at
   positions beyond training length are perfectly well defined. (Quality at
   unseen *gaps* still degrades - attention has never seen $\cos(\Delta\theta_i)$
   patterns for huge $\Delta$ - hence frequency-rescaling tricks like
   NTK/YaRN for context extension; module 7.)
3. **Position lives where it's used** - in every layer's scores - instead
   of being injected once at the bottom and hopefully preserved.

Trap: "RoPE makes attention depend only on distance" is wrong. The score
depends on the gap **and** on the content $q, k$ (the $q^\top R(\cdot) k$
form); only the *positional* dependence collapses to the gap.

### 7.2 MQA / GQA: shrinking the KV-cache

§3.8 ended with the conclusion that inference is a memory problem: ~2 GB of
KV-cache per sequence for a 7B model at $T = 4096$. Look at where that
number comes from: the cache stores $K$ and $V$ **per head**, so

$$
\text{cache bytes} = 2 \cdot n_{\text{layers}} \cdot T \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot (\text{bytes/val})
$$

Standard multi-head attention has $n_{\text{kv}} = h$: every query head
gets its own $K, V$. **Multi-query attention (MQA)** keeps all $h$ query
heads but shares a *single* $K, V$ across them: $n_{\text{kv}} = 1$, cache
shrinks $h\times$. **Grouped-query attention (GQA)** interpolates: $g$
groups of query heads, one $K, V$ per group, cache shrinks $(h/g)\times$.

Numbers: 32 layers, $h = 32$ heads of $d_{\text{head}} = 128$, $T = 8192$,
fp16.

- MHA: $2 \cdot 32 \cdot 8192 \cdot 32 \cdot 128 \cdot 2\,\text{B} = 4\ \text{GiB}$
  per sequence.
- GQA, $g = 8$: $4 \cdot \tfrac{8}{32} = 1\ \text{GiB}$ per sequence.
- MQA: $4 \cdot \tfrac{1}{32} = 128\ \text{MiB}$ per sequence.

At batch 16 that's 64 GiB (MHA - doesn't fit on an 80 GB card next to the
~14 GB of fp16 weights) vs 16 GiB (GQA - fits with room to spare). Same
model quality target, 4× the serving batch, and each decoded token also
*streams* 4× less cache from memory - a direct latency win, since decoding
is bandwidth-bound (§3.8, Q19).

Why quality mostly survives: the $h$ query heads keep their $h$ distinct
learned subspaces, so you still get $h$ independent attention
*distributions* per token (the thing multi-head actually buys, §3.4); what
you give up is diversity in what tokens *advertise* and *hand over*. In
practice MQA costs a little quality; GQA at $g = 8$ is near-indistinguishable
from MHA - which is why Llama-2-70B onward, Mistral, and essentially all
modern LLMs ship GQA.

### 7.3 FlashAttention, at intuition level

The $O(T^2)$ *memory* in §3.8 comes from materializing the score matrix
$QK^\top$. FlashAttention computes **exactly the same attention** without
ever writing that matrix to GPU main memory (HBM): tile $Q, K, V$ into
blocks small enough to live in on-chip SRAM, and accumulate the output
block-by-block using **online softmax** - softmax over a stream, keeping a
running max and running normalizer, and rescaling the partial output
whenever the running max updates. Each output row is built incrementally
from key/value tiles; extra memory is $O(T)$ (the running statistics), not
$O(T^2)$, and the result is bit-for-bit the same computation, *not* an
approximation.

Why is it also *faster*, given the FLOPs are the same (slightly more,
actually - the backward pass recomputes tiles instead of storing them)?
Because attention on a GPU is **IO-bound, not FLOP-bound**: the dominant
cost is moving the $T\times T$ matrix back and forth between HBM and SRAM,
not the arithmetic on it. Tiling reads each element of $Q, K, V$ from HBM
$O(1)$ times and never round-trips the score matrix at all. The general
lesson - count memory traffic, not FLOPs - is the same one behind the
KV-cache bandwidth story (§3.8). Trap: FlashAttention reduces memory and
wall-clock, **not** the $O(T^2 d)$ FLOP complexity. Full IO-complexity
analysis in module 7.

### 7.4 Mixture of Experts: decoupling parameters from FLOPs

In a dense transformer every parameter touches every token: params and
per-token FLOPs are locked together ($\approx 2$ FLOPs per param per
token). But a model's capacity to *store* (facts, patterns, languages)
scales with parameters somewhat independently of the compute it spends per
token. **MoE breaks the lock** - conditional computation: have many
parameters, use few per token.

Concretely, replace each block's MLP (which is ~2/3 of the parameters,
§3.6) with $E$ expert MLPs plus a tiny **router**. Per token $x$:

$$
p = \operatorname{softmax}(W_r\, x) \in \mathbb{R}^E, \qquad
y = \sum_{i \in \operatorname{top-}k(p)} \tilde p_i \, E_i(x)
$$

where $\tilde p$ renormalizes the top-$k$ gate values to sum to 1. Only $k$
experts execute, so the token pays $k$ experts' FLOPs while the model owns
$E$ experts' parameters.

Worked example, $E = 8$, $k = 2$, $d = 4096$: one expert MLP has
$8d^2 \approx 134$M params (§3.6). Dense block MLP: 134M params, ~268M
FLOPs/token. MoE block: $8 \times 134\text{M} \approx 1.07$B params, but
only $2 \times 268\text{M} \approx 537$M FLOPs/token - **8× the MLP
parameters at 2× the MLP compute**. Whole-model version of the same
arithmetic (attention and embeddings stay dense and shared): Mixtral
"8×7B" has ~47B total parameters with ~13B *active* per token - note it is
not $8\times 7 = 56$B, because only the MLPs are replicated.

**The load-balancing problem.** Left alone, routing collapses: an expert
that starts slightly better attracts more tokens, hence more gradient,
hence gets better - rich-get-richer until a few experts do everything
(dead experts waste parameters; hot experts blow past their capacity).
Since "fraction of tokens routed to $i$" is a hard, non-differentiable
count, add an auxiliary loss (Switch-style): with $f_i$ = fraction of the
batch's tokens dispatched to expert $i$ and $\bar p_i$ = mean router
probability for expert $i$,

$$
L_{\text{aux}} = \alpha \, E \sum_{i=1}^{E} f_i\, \bar p_i
$$

which is minimized when both distributions are uniform ($=1$ at
$f_i = \bar p_i = 1/E$); the differentiable $\bar p_i$ factor carries the
gradient that nudges the router toward balance. Plus capacity factors: each
expert takes at most $c \cdot T/E$ tokens per batch, overflow tokens get
dropped (skip the MLP via the residual stream).

**Cost.** Experts are sharded across devices (*expert parallelism*), so
every MoE layer does an all-to-all shuffle - each device sends its tokens
to whichever devices hold their chosen experts and receives them back.
That's communication-heavy and latency-sensitive, and all $E$ experts'
weights must sit in memory even though only $k$ run. **Why frontier models
are MoE anyway:** at a fixed training and inference FLOP budget, more
parameters means lower loss (the scaling-law arithmetic - module 7), and
MoE is the only known way to buy parameters without buying FLOPs.
Mixtral, DeepSeek-V3, and reportedly GPT-4-class models all take this
trade.

Trap: experts are routed *per token, per layer*, and do **not** cleanly
specialize into human-legible topics ("the math expert") - observed
specialization is mostly syntactic/positional.

---

## 8. Generative models beyond autoregression

A generative model learns $p(x)$ well enough to *sample* from it.
Autoregression (§5) makes the density tractable by chain rule and pays
with sequential sampling. There are three other classic ways to make
generation tractable, each a different trade - and "compare VAEs, GANs,
and diffusion" is a staple interview question.

### 8.1 Autoencoder → VAE, with the ELBO derived in full

A plain **autoencoder** - encoder $z = f(x)$, decoder $\hat x = g(z)$,
minimize reconstruction error through a bottleneck - learns compression,
but it is *not* generative: nothing makes the latent space fill any known
distribution, so decoding a random $z$ gives garbage. To sample, you need
a latent space with a distribution you can sample from.

**The VAE** makes the bottleneck probabilistic. Model:

$$
p_\theta(x) = \int p(z)\, p_\theta(x \mid z)\, dz,
\qquad p(z) = \mathcal{N}(0, I),
$$

with a neural decoder $p_\theta(x\mid z)$. Maximum likelihood needs
$\log p_\theta(x)$, and that integral is intractable (high-dimensional $z$
through a nonlinear decoder); the posterior
$p_\theta(z \mid x) = p(z)p_\theta(x\mid z)/p_\theta(x)$ is intractable for
the same reason. So introduce an **approximate posterior**
$q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x))$ - the encoder - and derive what we *can* optimize. Start from the KL gap
between the approximate and true posteriors and just expand definitions:

$$
\begin{aligned}
\operatorname{KL}\!\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)
&= \mathbb{E}_{q}\big[\log q_\phi(z\mid x) - \log p_\theta(z\mid x)\big]\\
&= \mathbb{E}_{q}\Big[\log q_\phi(z\mid x) - \log p_\theta(x, z) + \log p_\theta(x)\Big]
&& \text{since } p(z\mid x) = \tfrac{p(x,z)}{p(x)}\\
&= \log p_\theta(x) - \underbrace{\mathbb{E}_{q}\big[\log p_\theta(x, z) - \log q_\phi(z\mid x)\big]}_{\displaystyle \operatorname{ELBO}(\theta, \phi; x)}
\end{aligned}
$$

($\log p_\theta(x)$ exits the expectation because it doesn't depend on
$z$.) Rearranged:

$$
\log p_\theta(x) = \operatorname{ELBO} + \operatorname{KL}\!\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big) \;\ge\; \operatorname{ELBO}
$$

since KL $\ge 0$. This is an *identity*, not an approximation: the gap
between the true log-likelihood and the ELBO **is exactly** the posterior
mismatch. Maximizing the ELBO over $\phi$ tightens the bound (drives $q$
toward the true posterior); over $\theta$ it pushes up the likelihood
itself. Split $p(x,z) = p(x\mid z)p(z)$ to get the computable form:

$$
\operatorname{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}}
\;-\; \underbrace{\operatorname{KL}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularization}}
$$

The KL between two Gaussians is closed-form; the reconstruction term is
estimated by sampling $z \sim q_\phi$.

**Reparameterization trick.** One problem left: the gradient w.r.t. $\phi$
goes through the *sampling distribution* - you can't backprop through
"draw $z \sim q_\phi$", because the sample is not written as a
differentiable function of $\phi$. Rewrite the sample so the randomness
comes from a fixed source:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

Now $z$ is a deterministic, differentiable function of $\phi$ given
$\varepsilon$, and gradients flow through $\mu$ and $\sigma$ by ordinary
backprop. (The alternative - the score-function/REINFORCE estimator  - 
needs no reparameterization but has far higher variance; reparameterization
is why VAEs train easily.)

**What the KL term does:** pins every per-example posterior near
$\mathcal{N}(0, I)$, so the latents of the whole dataset tile a smooth,
known region - that's what makes sampling ($z \sim \mathcal{N}(0,I)$, then
decode) and interpolation work. Failure mode to name: **posterior
collapse** - with a decoder powerful enough to model $x$ unconditionally
(e.g. an autoregressive decoder), the cheapest optimum is $q \to p(z)$,
KL $\to 0$, latent ignored. And the classic VAE artifact: **blurry
samples**, because a Gaussian likelihood = squared-error reconstruction,
and the decoder averages over everything the latent doesn't pin down.

### 8.2 GANs: the minimax game, and what the generator really minimizes

GANs skip likelihood entirely. A **generator** $G$ maps noise
$z \sim \mathcal{N}(0,I)$ to samples; a **discriminator** $D$ classifies
real vs generated; they play

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]
+ \mathbb{E}_{z}[\log(1 - D(G(z)))]
$$

**Derive the optimal discriminator.** Fix $G$, giving samples with density
$p_g$. Then
$V = \int \big[p_{\text{data}}(x)\log D(x) + p_g(x)\log(1 - D(x))\big]dx$,
which can be maximized pointwise in $D(x)$: with $a = p_{\text{data}}(x)$,
$b = p_g(x)$,

$$
\frac{d}{dD}\big[a \log D + b\log(1 - D)\big] = \frac{a}{D} - \frac{b}{1 - D} = 0
\;\Rightarrow\; D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

(Sanity check: if $p_g = p_{\text{data}}$, $D^* = \tfrac12$ everywhere  - 
the discriminator is reduced to coin-flipping.) Substitute $D^*$ back into
$V$ and massage the two expectations:

$$
C(G) = \mathbb{E}_{p_{\text{data}}}\!\left[\log\tfrac{p_{\text{data}}}{p_{\text{data}} + p_g}\right]
+ \mathbb{E}_{p_g}\!\left[\log\tfrac{p_g}{p_{\text{data}} + p_g}\right]
= -\log 4 + 2\,\operatorname{JSD}\!\big(p_{\text{data}} \,\|\, p_g\big)
$$

So **against an optimal discriminator, the generator is minimizing the
Jensen–Shannon divergence** to the data distribution - a genuine
distribution-matching objective hiding inside the game, minimized (at
$-\log 4$) iff $p_g = p_{\text{data}}$.

Why training is famously unstable: it's a saddle-point problem, not a
minimization - simultaneous SGD on two coupled objectives has no descent
guarantee and can orbit; there is no trustworthy loss curve ($D$'s loss
falling could mean $D$ improved *or* $G$ degraded); when $D$ gets too
good, $\log(1 - D(G(z)))$ saturates and $G$'s gradient vanishes (hence the
non-saturating $-\log D(G(z))$ trick); and **mode collapse** - $G$ maps
many $z$'s to the few outputs currently fooling $D$; the JS objective
wants coverage, but the alternating game can hop between modes instead of
covering them, and nothing *per-sample* penalizes a missing mode. The
payoff when it works: one forward pass per sample and sharp outputs - no
explicit likelihood means no averaging-induced blur.

### 8.3 Diffusion: generation as learned denoising

**Forward process - fixed, nothing learned.** A Markov chain that
gradually destroys the data with Gaussian noise over $T$ steps
($T \sim 1000$), with a variance schedule $\beta_t$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\; x_{t-1},\; \beta_t I\big)
$$

Composing Gaussians gives a closed-form jump to any noise level: with
$\bar\alpha_t = \prod_{s\le t}(1 - \beta_s)$,

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\; \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, I)
$$

By $t = T$, $\bar\alpha_T \approx 0$: pure standard normal, all structure
gone. Generation = learning to run this film backwards, starting from
noise you *can* sample.

**The objective.** Learn a reverse chain $p_\theta(x_{t-1}\mid x_t)$ as
small Gaussian steps (for small $\beta_t$ the true reversal is
near-Gaussian). This is a latent-variable model - in fact a **hierarchical
VAE whose encoder is fixed** to the noising process, which is exactly why
it dodges the VAE's posterior pathologies: there's nothing to learn on the
inference side. Writing the ELBO for it and simplifying (with a
convenient reweighting of terms), essentially everything reduces to a
**noise-prediction regression**:

$$
L_{\text{simple}} = \mathbb{E}_{x_0,\, t,\, \varepsilon}
\Big\|\, \varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\; t\big) \Big\|^2
$$

In words: noise a real example to a random level $t$, ask the network
which noise was added, penalize with MSE. Intuition for why that's enough:
predicting the noise = predicting the direction back toward the data;
sampling runs $t = T \to 1$, each step subtracting a bit of predicted
noise (plus fresh noise for stochasticity), walking noise into data. The
one network is shared across all $t$ - at high noise it fixes global
structure, at low noise it fixes fine detail.

**Why diffusion beat GANs:** (1) the loss is stable supervised
regression - trains with ordinary tools, scales, and the loss curve means
something, versus a saddle-point game; (2) **mode coverage** - it's
(a bound on) likelihood training over the whole distribution, so no mode
collapse; (3) capacity scales gracefully (one denoiser, many noise
levels). The price: sampling takes tens-to-hundreds of network
evaluations vs a GAN's one - hence fast samplers and distillation. Text
runs on autoregression, images/audio/video on diffusion (usually *latent*
diffusion - run in a VAE's latent space; the two compose).

**Classifier-free guidance**, one paragraph: train the conditional
denoiser $\varepsilon_\theta(x_t, t, c)$ with the condition randomly
dropped (some fraction of steps use $c = \varnothing$), so one network is
both conditional and unconditional. At sampling, extrapolate past the
condition:
$\hat\varepsilon = \varepsilon(x_t, \varnothing) + w\,\big(\varepsilon(x_t, c) - \varepsilon(x_t, \varnothing)\big)$
with guidance scale $w > 1$ - amplify the direction "what the condition
adds." Higher $w$ = tighter prompt adherence and sharper, less diverse
samples; this is the guidance knob behind every text-to-image system.

### 8.4 The comparison table

| | Autoregressive | VAE | GAN | Diffusion |
|---|---|---|---|---|
| **Sample quality** | excellent (discrete data) | blurry (likelihood averaging) | sharp when it works | state of the art (continuous data) |
| **Sampling speed** | slow - sequential, one token per pass | fast - one decoder pass | fast - one generator pass | slow - many denoiser passes |
| **Likelihood** | exact, tractable | lower bound (ELBO) | none (implicit density) | bound (ELBO-style); good in practice |
| **Training stability** | very stable (cross-entropy) | stable (watch posterior collapse) | unstable (adversarial saddle point) | very stable (regression) |
| **Mode coverage** | full (likelihood forces it) | full but smoothed | prone to mode collapse | full |

Interview one-liner: *VAE = optimize a likelihood bound through a latent
bottleneck - fast, stable, blurry; GAN = adversarial game, implicitly
minimizing JS - fast, sharp, unstable, drops modes; diffusion = learn to
reverse fixed noising via noise-prediction regression - stable AND sharp
AND mode-covering, but slow to sample; autoregressive = exact likelihood,
sequential sampling, owns text.*

---

## 9. Self-supervised and contrastive learning

### 9.1 Pretext tasks: labels for free

Supervised learning is bottlenecked on labels; the data itself is not.
**Self-supervision**: hide part of the data and predict it from the rest  - 
the *structure* of the data is the label, so every unlabeled example is a
training example. Next-token prediction (§5) is exactly this. Early vision
pretexts (predict rotation, solve jigsaws, colorize) worked only so-so  - 
the representations solved the pretext, not your task. Two families won:
**contrastive learning** and **masked prediction**.

### 9.2 InfoNCE, derived: contrastive learning is just cross-entropy

Setup: an anchor $x$, a **positive** $x^+$ that "means the same thing"
(another augmentation of the same image; the caption of the image), and
$N - 1$ **negatives** (other examples). Embed everything, score by
similarity $s(u, v)$ (cosine or dot product). The InfoNCE loss:

$$
L = -\log \frac{e^{\,s(z, z^+)/\tau}}{\sum_{i=1}^{N} e^{\,s(z, z_i)/\tau}}
$$

Now *recognize* it: this is exactly softmax cross-entropy (module 3) for
an $N$-way classification problem - "**which of these $N$ candidates is my
true partner?**" - with similarities as logits and the positive as the
correct class. No new math. The twist is that the "classifier weights" are
not learned parameters; they are the embeddings of the candidates
themselves. Gradient behavior falls out of softmax-CE too: the positive is
pulled toward the anchor, each negative is pushed away *in proportion to
its softmax probability* - so **hard negatives** (lookalike impostors with
high similarity) dominate the gradient, which is where the learning
signal lives.

**Temperature $\tau$** is the same object as sampling temperature (§5.4):
it scales the logits. Small $\tau$ sharpens the distribution and
concentrates the push onto the very hardest negatives (powerful, but
brutal to false negatives - two genuinely-similar images treated as a
negative pair); large $\tau$ spreads the push uniformly. Typical values
0.05–0.2, sometimes learned (CLIP learns it).

Why contrastive learning wants **large batches**: negatives are usually
"everyone else in the batch," and more negatives = a harder, finer-grained
$N$-way problem = a tighter bound on the mutual information being
maximized and better-separated features. SimCLR used batches up to 4096;
MoCo faked a huge $N$ with a queue of past embeddings and a slowly-updated
momentum encoder to keep them consistent.

### 9.3 SimCLR in one paragraph

Take a batch of $N$ images; make two random augmentations of each (crop +
resize, color jitter, blur) - each pair is a positive, the other $2(N-1)$
views in the batch are its negatives; encode, pass through a small MLP
**projection head**, apply InfoNCE both ways. What stuck from the paper:
the *augmentation choice is the supervision* - composing crop with strong
color jitter is essential, because the augmentations define exactly which
variations the representation must ignore; the projection head matters
(contrast on the head's output, use the features *before* it - the head
discards information the loss wants gone); and a linear probe on the
frozen features approaches supervised accuracy - the classic evidence
that labels are not where representations come from.

### 9.4 CLIP: contrastive across modalities = zero-shot classifiers for free

Take $N$ (image, caption) pairs. An image encoder and a text encoder map
into one shared space; compute the $N \times N$ matrix of cosine
similarities. **Symmetric InfoNCE**: each image must classify its own
caption among the $N$ captions (cross-entropy along the row), each caption
must classify its image (along the column); average the two losses. Train
on ~400M web pairs. The matched diagonal is pulled together; the $N^2 - N$
mismatched pairs are the negatives - no labels anywhere.

Why **zero-shot classification** falls out: to classify an image among $K$
classes, write each class as a prompt ("a photo of a dog", …), embed them
to get $t_1, \dots, t_K$, and predict
$\arg\max_k \cos(v_{\text{image}}, t_k)$. The view that makes it click: a
linear classifier computes logits $Wv$ where row $k$ of $W$ is class
$k$'s weight vector - CLIP's text encoder is a machine that **generates
classifier weight vectors from class names**. Any label set, no training
run. That's also why prompt wording ("a photo of…", prompt ensembling)
behaves like feature engineering: you are hand-tuning the weight matrix
through the text encoder.

### 9.5 BERT vs GPT: masked LM vs causal LM (the classic question)

Both are transformers trained self-supervised on text; the difference is
the pretext task, and it decides what the model is for.

- **BERT (encoder, masked LM):** mask ~15% of tokens, predict them using
  **both left and right context** - attention is fully bidirectional. Each
  position's representation summarizes the whole sentence, which is ideal
  for *understanding*: classification, NER, retrieval. But it doesn't
  define a sampleable joint distribution over text - it models
  $p(\text{masked} \mid \text{rest})$, not $\prod_t p(x_t \mid x_{<t})$  - 
  so there is no natural way to generate.
- **GPT (decoder, causal LM):** predict every next token from left context
  only (§3.7, §5.1). Generation is native; every token position is a
  training signal (vs 15% for BERT); and at scale, understanding tasks get
  solved *through* generation (prompting, in-context learning).

The interview answer for "when to use which": if the job is to **produce a
representation of a given, complete text** - classify it, tag it, embed it
for search - bidirectional context is strictly more information, so use an
encoder (or encoder-style model); if the job is to **produce text**, you
need the causal factorization, so use a decoder. Modern practice defaults
to decoder-only at frontier scale (one model, everything by prompting),
but small fine-tuned encoders still win on latency and cost for
classification and retrieval. Sentence embeddings in one line: pool an
encoder's token vectors and fine-tune with a contrastive loss (InfoNCE
yet again - SBERT/E5-style) so that cosine similarity means semantic
similarity; that vector is what sits in every RAG vector database.

---

## Common interview questions

**Q1. Why do CNNs use so many fewer parameters than MLPs on images?**
Weight sharing + locality: a conv layer's parameter count
$C_{out}(C_{in}k^2 + 1)$ is independent of image size, while a dense layer
scales with (#pixels)². Concretely, $224^2\times3$ input → 1000 hidden units
is 150M dense parameters vs ~1.8K for a 3→64 channel 3×3 conv. The deeper
answer: the CNN's structure hard-codes translation equivariance, which the
MLP would have to burn data learning, position by position.

**Q2. What's the output size of a conv layer, and how many parameters?**
$H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$; params
$= C_{out}(C_{in}k^2 + 1)$. E.g. $32\times32$, $k=5$, $p=0$, $s=1$,
$C_{in}=3$, $C_{out}=8$: output $28\times28\times8$, params
$8(3\cdot25+1) = 608$.

**Q3. Why did ResNet's skip connections let networks go from ~20 to 150+ layers?**
Plain deep nets underfit - *training* error rose with depth, an optimization
failure: stacked nonlinear layers can't easily represent identity.
$y = x + F(x)$ makes identity the default ($F=0$) and gives backprop an
identity term $\partial y/\partial x = I + \partial F/\partial x$ - an
unattenuated gradient path through arbitrary depth.

**Q4. Derive why RNN gradients vanish or explode.**
$\partial h_t/\partial h_{t-1} = \operatorname{diag}(\tanh')\,W_{hh}$; over a
gap of $n$ steps you multiply $n$ such Jacobians, so the norm behaves like
$(\gamma\,\sigma_{\max}(W))^n$ - geometric decay if the effective spectral
factor is below 1 (vanish), geometric growth if above (explode). Only
$\approx 1$ transmits, and nothing pins a learned $W$ there.

**Q5. How exactly does the LSTM fix vanishing gradients?**
The cell state updates additively: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$,
so $\partial c_t/\partial c_{t-1} \approx \operatorname{diag}(f_t)$ - no
repeated $W$ multiplication, no squashing on the main path. Where the learned
forget gate $\approx 1$, gradients pass essentially unchanged: a
data-dependent gradient highway (same principle as ResNet). It does *not* fix
exploding gradients (still clip) or the sequential bottleneck.

**Q6. Write down attention and explain each part.**
$\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$ - for each query (what a
token seeks), dot with all keys (what tokens advertise) to get relevance
scores; scale; softmax into a distribution; take that convex combination of
values (what tokens hand over). A soft, differentiable dictionary lookup,
with $Q, K, V$ as learned projections of the same token states.

**Q7. Why divide by $\sqrt{d_k}$?**
$q^\top k = \sum_i q_i k_i$ with unit-variance uncorrelated entries has mean
0 and variance $d_k$ (variances of uncorrelated terms add). At $d_k=64$,
std 8: softmax saturates to near one-hot, its Jacobian
$\operatorname{diag}(w) - ww^\top \to 0$, gradients vanish at init. Dividing
by $\sqrt{d_k}$ restores unit variance for any $d_k$.

**Q8. Why does attention need positional encodings when RNNs and CNNs don't?**
Attention computes only content-based pairwise dot products - it is
permutation equivariant, so "dog bites man" ≡ "man bites dog" without
position info. RNNs get order from sequential processing, CNNs from kernel
layout. Fix: add learned/sinusoidal absolute embeddings to the inputs, or
encode relative position in the scores (RoPE, relative bias).

**Q9. What do multiple heads buy over one big head?**
$h$ independent attention *distributions* per token, each measuring
similarity in its own learned subspace ($d/h$ dims) - one head can track
previous-token position while another tracks coreference. Compute matches a
single full-width head (dimensions are split, not multiplied); a single head
gives only one convex combination per position.

**Q10. Explain the KV-cache. Why doesn't it help training?**
By causality, generating token $t$ leaves the keys/values of tokens $<t$
unchanged, so cache them per layer; each new token computes only its own
$q,k,v$ and attends one query against $t$ cached keys: $O(td)$ per layer per
token (linear) instead of re-running the $O(t^2 d)$ prefix. Training already
computes all positions in one parallel masked pass - every K/V is used
immediately and exactly once per step, and weights change every step, so
there's nothing valid to reuse.

**Q11. What is the complexity of self-attention, and what dominates when?**
Scores + weighted sum: $O(T^2 d)$ compute, $O(T^2)$ attention-matrix memory
per head (FlashAttention removes the memory term, not the FLOPs); the MLP
adds $O(T d^2)$. Quadratic term dominates once $T \gtrsim d$ - hence the
cottage industry of long-context approximations.

**Q12. Pre-LN vs post-LN - why does it matter?**
Post-LN puts LayerNorm on the residual path: every gradient to the bottom
passes through dozens of rescalings, no identity path survives, deep stacks
need lr warmup and still can be unstable. Pre-LN
($x + f(\operatorname{LN}(x))$) keeps the residual stream addition-only, so
$I$ appears in every layer's Jacobian - stable at depth. Interviewers use
this to check you reason about gradient paths.

**Q13. What is perplexity and what does a value of, say, 20 mean?**
$\operatorname{PPL} = e^{H}$ with $H$ the mean per-token cross-entropy in
nats (i.e. exponentiated validation loss). PPL 20 ⇒ on average the model is
as uncertain as a uniform choice among 20 tokens (effective branching
factor). Only comparable under the same tokenizer; use bits-per-byte across
tokenizers.

**Q14. How do temperature, top-k, and top-p differ?**
Temperature rescales logits before softmax ($p_i \propto e^{z_i/\tau}$),
exponentiating odds ratios by $1/\tau$: it reshapes the whole distribution.
Top-k truncates to the $k$ most likely then renormalizes (fixed-size
nucleus); top-p truncates to the smallest set with cumulative mass $\ge p$
(adaptive size: narrow when the model is confident, wide when not). They
compose: typically temperature then top-p.

**Q15. What is BPE and why not just use characters or words?**
Iteratively merge the most frequent adjacent token pair, starting from bytes,
until the vocab budget is hit; tokenizing applies merges in learned order.
Characters ⇒ very long sequences (quadratic attention pain) and every word
must be re-composed; word-level ⇒ huge vocab, OOV failures, no morphology
sharing. BPE interpolates: common words are single tokens, rare words split
into meaningful subwords, and byte fallback means nothing is ever OOV.

**Q16. Explain LoRA and why it's so cheap.**
Freeze pretrained $W$; learn $\Delta W = \frac{\alpha}{r}BA$ with
$B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r\times k}$, $r \ll d$  - 
$r(d+k)$ params instead of $dk$ (~250× fewer at $d{=}k{=}4096, r{=}8$).
Works because the adaptation delta is empirically low-rank even when $W$
isn't. $B{=}0$ init starts at the pretrained model; merging $W + BA$ after
training adds zero inference latency; optimizer memory shrinks with the
trainable-param count.

**Q17. Sketch RLHF.**
(1) SFT on human demonstrations; (2) train a reward model on human pairwise
preferences with the Bradley–Terry loss
$-\log\sigma(r(x,y_w) - r(x,y_l))$; (3) RL (e.g. PPO) maximizing learned
reward minus $\beta\,\mathrm{KL}(\pi\|\pi_{\text{SFT}})$ - the KL stops the
policy from hacking the imperfect reward. DPO folds (2)+(3) into one
supervised loss on preference pairs.

**Q18. Why LayerNorm in transformers rather than BatchNorm?**
LN normalizes across the feature dimension per token: no dependence on batch
statistics, so it handles variable sequence lengths, tiny/streaming batches,
and behaves identically in training and inference. BN's cross-example
statistics are noisy and ill-defined over padded, variable-length token
batches.

**Q19. Why is generation with a transformer still slow if training is so parallel?**
Training parallelizes across the $T$ positions of known text (teacher
forcing under a causal mask). Generation is inherently sequential - token
$t{+}1$'s input includes sampled token $t$ - so it's $n$ dependent forward
passes; the KV-cache makes each pass $O(td)$ per layer but can't remove the
sequential dependency. Each step is also memory-bandwidth-bound (stream all
weights to produce one token), hence batching/speculative decoding.

**Q20. A conv net trains fine but test images shifted by a few pixels break it. Isn't it translation invariant?**
Conv layers are translation *equivariant* (output shifts with input), not
invariant; invariance only accumulates from pooling/downsampling, and
striding/padding/boundary effects break exact equivariance (aliasing). Global
average pooling at the end and shift augmentation are the practical fixes  - 
and this distinction (equivariance vs invariance) is precisely what the
question is probing.

**Q21. Why has RoPE replaced learned absolute positional embeddings?**
Learned absolute: a lookup table added to the input - position is content
by the time attention sees it, every offset pattern must be learned per
position, and nothing exists past $T_{\max}$. RoPE rotates $q$ and $k$ by
position-proportional angles so
$q_m^\top k_n = q^\top R((n-m)\theta)k$ - the score's positional
dependence is *exactly* the gap $n - m$, by rotation algebra
($R(a)^\top R(b) = R(b-a)$), not by learning. Relative structure is shared
across all positions, and any position is defined, so it extrapolates far
better (with frequency rescaling for extreme lengths).

**Q22. Why grouped-query attention?**
KV-cache size (and per-token cache bandwidth) scales with the number of
KV heads: $2\,n_{\text{layers}} T\, n_{\text{kv}} d_{\text{head}}$ values.
GQA shares one K,V among each group of query heads: $h/g$ times less cache
(32 heads, $g{=}8$: 4×), so bigger serving batches and faster
bandwidth-bound decoding. Quality survives because the $h$ query subspaces - hence $h$ attention distributions per token - are kept; only K/V
diversity is reduced. MQA ($g{=}1$) is the extreme and costs a little
quality; GQA is the modern default.

**Q23. An "8×7B" MoE - how many parameters run per token, and why bother?**
Per token, per MoE layer, the router picks top-$k$ (say 2) of 8 expert
MLPs: you own 8 experts' parameters but spend ~2 experts' FLOPs
(Mixtral: ~47B total - not 56B, attention is shared - ~13B active). Point:
loss scales with parameters, cost scales with FLOPs; MoE is conditional
computation that buys parameters without buying per-token compute. Prices:
all params in memory, all-to-all routing communication, and a
load-balancing aux loss ($\alpha E\sum_i f_i \bar p_i$) to stop
rich-get-richer router collapse.

**Q24. VAE vs GAN vs diffusion, one line each.**
VAE: maximize a likelihood lower bound (ELBO) through a stochastic latent
bottleneck - stable, fast sampling, blurry (reconstruction averaging).
GAN: generator vs discriminator minimax, implicitly minimizing JS
divergence - fast, sharp, unstable, mode-collapses. Diffusion: fixed
Gaussian noising chain, learn the reverse as noise-prediction regression  - 
stable *and* sharp *and* mode-covering, but sampling needs many passes.
(Autoregressive: exact likelihood, sequential sampling, owns text.)

**Q25. Why does contrastive learning need large batches (or memory banks)?**
InfoNCE is an $N$-way cross-entropy: classify the positive among $N{-}1$
negatives, usually the rest of the batch. Small $N$ = a coarse, easy
discrimination task with weak gradients; large $N$ = more (and more
likely *hard*) negatives, a finer-grained task, and a tighter mutual-
information bound - hard negatives carry most of the gradient because the
push on each negative scales with its softmax probability. Hence SimCLR's
4096 batches, or MoCo's momentum-encoder queue to fake large $N$ cheaply.

**Q26. BERT or GPT - which do you use when?**
BERT: masked-LM encoder, bidirectional attention - each token's vector
sees the whole input, best for representing *given* text (classification,
NER, retrieval/embeddings), but it defines no sampleable joint, so it
can't generate. GPT: causal-LM decoder - models
$\prod_t p(x_t\mid x_{<t})$, so generation is native and every position is
a training target. Rule: representation of complete text → encoder;
producing text → decoder; at frontier scale decoder-only does both via
prompting, but small encoders still win latency/cost for classification
and search.

## Exit checklist

- [ ] Cold, on paper: conv-layer arithmetic - output shape, parameter count, FLOPs for given kernel/stride/padding - 4 min
- [ ] Derive scaled dot-product attention from first principles, including why the 1/√d_k - 5 min
- [ ] Sketch the full pre-LN transformer block with tensor shapes at every step, plus the causal mask - 5 min
- [ ] KV-cache: what is stored, per-token decode cost, and why it turns O(n²) regeneration into O(n) - 3 min
- [ ] Show why RNN gradients vanish/explode (the repeated W^T product) and how LSTM gates make a gradient highway - 4 min
- [ ] Derive perplexity from cross-entropy, then explain temperature, top-k, and top-p sampling - 3 min
- [ ] State LoRA in one paragraph and RoPE's core idea (rotation angle encodes relative position) - 4 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
