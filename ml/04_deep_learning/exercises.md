# Deep Learning Exercises

Fifteen exercises, easy → hard. Do them on paper first; several have runnable
numpy checks in the solutions. Notation follows `notes.md`.

---

## Exercise 1 (easy) - Shapes and parameters of a conv stack

Input: $32 \times 32 \times 3$ image. Network:

1. conv $3\times3$, 16 filters, stride 1, padding 1
2. maxpool $2\times2$, stride 2
3. conv $3\times3$, 32 filters, stride 1, padding 1
4. maxpool $2\times2$, stride 2
5. flatten, fully-connected to 10 classes

Give the output shape after every layer, the parameter count of every layer,
and the FLOPs (multiply-adds × 2) of layer 1.

<details><summary>Solution</summary>

Using $H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$:

| layer | output shape | params |
|---|---|---|
| conv1 (3→16, k3, p1) | $32\times32\times16$ | $16(3\cdot9+1) = 448$ |
| pool | $16\times16\times16$ | 0 |
| conv2 (16→32, k3, p1) | $16\times16\times32$ | $32(16\cdot9+1) = 4{,}640$ |
| pool | $8\times8\times32$ | 0 |
| FC ($2048 \to 10$) | $10$ | $2048\cdot10 + 10 = 20{,}490$ |

Total: $25{,}578$. Note the FC layer holds ~80% of the parameters despite
doing the least interesting work - the classic CNN pattern.

FLOPs of conv1: each output element is a dot product of length
$C_{in}k^2 = 27$, so
$2 \cdot 27 \cdot 16 \cdot 32 \cdot 32 = 884{,}736 \approx 0.9$ MFLOPs.
</details>

---

## Exercise 2 (easy) - Convolution by hand

Compute the full valid (no padding, stride 1) cross-correlation of

$$
X = \begin{pmatrix} 1 & 2 & 0 & 1\\ 0 & 1 & 2 & 3\\ 1 & 0 & 0 & 2\\ 2 & 1 & 1 & 0 \end{pmatrix}
\quad\text{with}\quad
K = \begin{pmatrix} 1 & 0\\ 0 & -1 \end{pmatrix}
$$

What pattern does this kernel respond to?

<details><summary>Solution</summary>

Output is $(4-2+1)\times(4-2+1) = 3\times3$. Each entry is
$X[i,j] - X[i{+}1, j{+}1]$ (top-left minus bottom-right of each $2\times2$
patch):

$$
Y = \begin{pmatrix}
1-1 & 2-2 & 0-3\\
0-0 & 1-0 & 2-2\\
1-1 & 0-1 & 0-0
\end{pmatrix}
= \begin{pmatrix} 0 & 0 & -3\\ 0 & 1 & 0\\ 0 & -1 & 0 \end{pmatrix}
$$

It's a diagonal gradient detector: fires (positive or negative) where
intensity changes along the ↘ diagonal, zero on diagonally-constant patches.
</details>

---

## Exercise 3 (easy) - Receptive field

(a) What is the receptive field of a stack of $L$ conv layers, each
$3\times3$ stride 1? Give the general formula and evaluate at $L = 1, 2, 3$.

(b) Now compute the receptive field of: conv $3\times3$ s1 → maxpool
$2\times2$ s2 → conv $3\times3$ s1.

(c) Why does VGG prefer three $3\times3$ convs over one $7\times7$? Two
reasons.

<details><summary>Solution</summary>

General recurrence: $r_l = r_{l-1} + (k_l - 1)\prod_{i<l} s_i$, with $r_0=1$.

(a) All strides 1: $r_L = 1 + 2L$. So $L=1: 3$, $L=2: 5$, $L=3: 7$ - each
$3\times3$ layer adds 2.

(b) Step by step (track the running stride product $J$):
- conv1 ($k=3$, $J=1$): $r = 1 + 2 = 3$; now $J = 1$
- pool ($k=2$, $s=2$): $r = 3 + (2-1)\cdot 1 = 4$; now $J = 2$
- conv2 ($k=3$): $r = 4 + (3-1)\cdot 2 = \boxed{8}$

Downsampling makes later layers grow the receptive field faster - that's
its main computational purpose.

(c) (1) Fewer parameters: $3 \cdot 9C^2 = 27C^2 < 49C^2$; (2) two extra
nonlinearities in between, making the composed function more expressive for
the same view. (Bonus: less compute per output for the same receptive field.)
</details>

---

## Exercise 4 (easy-medium) - Attention on 2 tokens, by hand

Two tokens with $d = 2$; use identity projections ($Q = K = V = X$):

$$
x_1 = (1, 0), \qquad x_2 = (1, 1)
$$

(a) Compute full (bidirectional) scaled dot-product attention: scores,
scaled scores, attention weights, outputs. (b) Repeat with a causal mask.
(c) Check every attention row sums to 1.

<details><summary>Solution</summary>

(a) Scores $S = XX^\top$: $s_{11} = 1$, $s_{12} = x_1^\top x_2 = 1$,
$s_{21} = 1$, $s_{22} = 2$.

Scale by $1/\sqrt{2} = 0.7071$:

$$
S' = \begin{pmatrix} 0.7071 & 0.7071\\ 0.7071 & 1.4142 \end{pmatrix}
$$

Softmax rows. Row 1: equal logits → $w_1 = (0.5,\ 0.5)$. Row 2:
$e^{0.7071} = 2.028$, $e^{1.4142} = 4.113$, sum $6.141$ →
$w_2 = (0.330,\ 0.670)$.

Outputs:
$o_1 = 0.5(1,0) + 0.5(1,1) = (1,\ 0.5)$;
$o_2 = 0.330(1,0) + 0.670(1,1) = (1,\ 0.670)$.

(b) Causal: token 1 sees only itself → $w_1 = (1, 0)$, $o_1 = x_1 = (1,0)$.
Token 2 sees both, mask changes nothing → same as row 2 above:
$o_2 = (1,\ 0.670)$.

(c) $0.5+0.5 = 1$; $0.330+0.670 = 1$; $1+0=1$. ✓
</details>

---

## Exercise 5 (medium) - Masking and softmax invariances

(a) Prove that softmax is invariant to adding a constant $c$ to all logits:
$\operatorname{softmax}(z + c\mathbf{1}) = \operatorname{softmax}(z)$. Why do
implementations subtract $\max(z)$?

(b) Why must the causal mask be applied to the *scores* (as $-\infty$)
before the softmax, rather than zeroing the attention *weights* after it?

<details><summary>Solution</summary>

(a) $\dfrac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \dfrac{e^c e^{z_i}}{e^c \sum_j e^{z_j}} = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$.
Subtracting $\max(z)$ uses this invariance for numerical stability: the
largest exponent becomes $e^0 = 1$, so $\exp$ never overflows (and the
denominator is $\ge 1$, so no division blow-up).

(b) Two failures if you zero after softmax: (1) the surviving weights no
longer sum to 1 - the output is no longer a convex combination and its scale
depends on how much mass the mask deleted; (2) worse, the *allowed*
positions' weights were computed while competing against forbidden
positions, so the forbidden scores still influenced the result (information
leaks from the future through the normalization). Setting scores to
$-\infty$ pre-softmax gives $e^{-\infty} = 0$ *and* renormalizes over
allowed positions only. ($-\infty$ rather than a large negative number for
exactness; in fp16 code you'll see $-10^4$-ish for overflow safety.)
</details>

---

## Exercise 6 (medium) - The $\sqrt{d_k}$ derivation

Let $q, k \in \mathbb{R}^{d}$ have i.i.d. entries, mean 0, variance 1,
independent of each other.

(a) Derive $\mathbb{E}[q^\top k]$ and $\operatorname{Var}(q^\top k)$.
(b) At $d = 512$, what is the standard deviation of an unscaled score? What
does that do to the softmax?
(c) What scaling makes the variance 1 for any $d$?

<details><summary>Solution</summary>

(a) $q^\top k = \sum_{i=1}^d q_i k_i$. Per term:
$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ (independence);
$\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1\cdot1 = 1$.
Terms are uncorrelated, and variances of uncorrelated sums add:
$\mathbb{E}[q^\top k] = 0$, $\operatorname{Var}(q^\top k) = d$.

(b) $\sqrt{512} \approx 22.6$. Logit gaps of tens of units: softmax puts
essentially all mass on the argmax. The softmax Jacobian
$\operatorname{diag}(w) - ww^\top$ vanishes as $w \to$ one-hot, so gradients
through the attention weights die at initialization, and the model starts
with hard random attention patterns.

(c) $\operatorname{Var}(q^\top k / \sqrt{d}) = d / (\sqrt{d})^2 = 1$. Hence
$1/\sqrt{d_k}$, independent of width.
</details>

---

## Exercise 7 (medium) - Parameter count of a transformer block

For model width $d$ (ignore all biases and LayerNorm):

(a) How many parameters in the attention sublayer ($W_Q, W_K, W_V, W_O$)?
(b) In the MLP with expansion factor 4?
(c) What fraction of block parameters is the MLP? Verify at $d = 768$
(GPT-2 small) that one block has ~7.1M parameters.

<details><summary>Solution</summary>

(a) Four $d\times d$ matrices: $4d^2$. (Multi-head doesn't change this  - 
heads split the same matrices.)
(b) $W_1: d\times 4d$ and $W_2: 4d\times d$: $8d^2$.
(c) MLP fraction $= 8d^2 / 12d^2 = 2/3$. At $d = 768$:
$12 \cdot 768^2 = 12 \cdot 589{,}824 = 7{,}077{,}888 \approx 7.1$M. ✓
GPT-2 small = 12 such blocks ≈ 85M, plus embeddings
($50257 \times 768 \approx 38.6$M, weight-tied with the output head) ≈ 124M
total - matches the advertised size.
</details>

---

## Exercise 8 (medium-hard) - Gradient of attention w.r.t. $V$

Let $A = \operatorname{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{T\times T}$
(treat it as a constant here) and $O = AV$ with $V \in \mathbb{R}^{T\times d_v}$.
Given upstream gradient $G = \partial L/\partial O \in \mathbb{R}^{T\times d_v}$,
derive $\partial L/\partial V$. Then interpret the result: which queries
contribute to the gradient of value $j$, and with what weight?

<details><summary>Solution</summary>

Elementwise: $O_{iq} = \sum_j A_{ij} V_{jq}$, so
$\dfrac{\partial O_{iq}}{\partial V_{jq'}} = A_{ij}\,\delta_{qq'}$. Chain
rule:

$$
\frac{\partial L}{\partial V_{jq}} = \sum_i G_{iq} A_{ij}
\qquad\Longleftrightarrow\qquad
\boxed{\dfrac{\partial L}{\partial V} = A^\top G}
$$

(The general rule for $O = AV$: $dV = A^\top dO$, $dA = dO\, V^\top$.)

Interpretation: value $j$'s gradient is a weighted sum of the output
gradients of **all queries that attended to it**, each weighted by exactly
the attention weight $A_{ij}$ it received. A value nobody attends to
(column $j$ of $A$ near 0) gets almost no gradient - attention gates
learning signal in the backward pass exactly as it gates information in the
forward pass. Quick numeric check with Exercise 4's numbers:
$A = \binom{0.5\ \ 0.5}{0.33\ 0.67}$, $G = \binom{1\ \ 0}{0\ \ 1}$ gives
$\partial L/\partial V = A^\top G = \binom{0.5\ \ 0.33}{0.5\ \ 0.67}$.
</details>

---

## Exercise 9 (medium-hard) - KV-cache

(a) Explain precisely why the KV-cache is valid during autoregressive
generation (what property of the model guarantees cached $K, V$ never go
stale?).

(b) Why does the KV-cache **not** help during training?

(c) Compute the KV-cache memory for: 12 layers, $d = 768$, context
$T = 1024$, fp16, batch size 8.

<details><summary>Solution</summary>

(a) Causal masking ⇒ position $j$'s representation at every layer depends
only on tokens $\le j$. Appending token $t{+}1$ therefore changes *nothing*
about positions $1..t$ - their $K, V$ at every layer are bit-identical to the
previous step, so caching is exact, not an approximation. (Also required:
the weights aren't changing between steps.)

(b) Training computes all $T$ positions **in one parallel forward pass**
under the mask - each $K, V$ is produced and consumed once within the pass,
so there is nothing to reuse within a step. Across steps, the weights just
changed (that's the point of training), so any cached $K, V$ would be stale.
The cache exploits *repeated forward passes with frozen weights and a
growing prefix* - a situation that only exists at inference.

(c) Elements $= 2\,(\text{K and V}) \times 12 \times 1024 \times 768
= 18{,}874{,}368$ per sequence; at 2 bytes: $\approx 37.7$ MB per sequence;
$\times 8 = \boxed{\approx 302\text{ MB}}$. Note it scales linearly in every
factor - batch, context, depth, width - which is why serving long contexts
is memory-bound.
</details>

---

## Exercise 10 (hard) - Vanishing gradients, exactly

Consider a *linear* RNN $h_t = W h_{t-1}$ with symmetric
$W = U \Lambda U^\top$ (eigenvalues $\lambda_1, \dots, \lambda_d$).

(a) Write $\partial h_T / \partial h_0$ in terms of the eigendecomposition.
(b) Decompose a gradient vector $g$ into eigencoordinates and state exactly
which components vanish or explode as $T$ grows.
(c) The nonlinear RNN has $\partial h_t/\partial h_{t-1} = D_t W$ with
$D_t = \operatorname{diag}(\tanh'(a_t))$, $\|D_t\| \le 1$. What one-sided
guarantee survives, and why is the LSTM's $\partial c_t/\partial c_{t-1} \approx \operatorname{diag}(f_t)$
fundamentally different from *any* choice of $W$?

<details><summary>Solution</summary>

(a) $\partial h_T/\partial h_0 = W^T = U \Lambda^T U^\top$ (power of the
matrix, since each step's Jacobian is exactly $W$).

(b) Write $g = \sum_i c_i u_i$ in the eigenbasis. Backpropagated:
$W^{T\top} g = \sum_i c_i \lambda_i^T u_i$ - each component scales by
$\lambda_i^T$. Components with $|\lambda_i| < 1$ decay geometrically
(vanish), $|\lambda_i| > 1$ grow geometrically (explode), $|\lambda_i| = 1$
are preserved. E.g. $\lambda = 0.9$, $T = 60$: $0.9^{60} \approx 0.0018$;
$\lambda = 1.1$: $1.1^{60} \approx 304$. The gradient is a *sum* over gaps,
so distant-dependency terms are drowned out by recent ones - the model can't
learn long-range structure even though the gradient isn't literally zero.

(c) Since $\|D_t\| \le 1$, we get only
$\|\prod D_t W\| \le \sigma_{\max}(W)^{T}$ - an upper bound. So you can
*guarantee* vanishing (if $\sigma_{\max} < 1$) but never guarantee
preservation: keeping gradients alive requires the product to sit near norm
1 along the needed directions for the whole trajectory, which nothing
enforces, and saturation ($D_t \to 0$) actively fights it. The LSTM's
$\operatorname{diag}(f_t)$ path differs in kind, not degree: it is (i)
element-wise, not a rotating matrix product, (ii) *data-dependent* - the
network chooses $f_t \approx 1$ per dimension per step, rather than relying
on a fixed spectrum, and (iii) additive in the forward pass, so no repeated
squashing nonlinearity sits on the highway. A fixed $W$ with unit
eigenvalues (e.g. identity/orthogonal RNNs) preserves *everything*
indiscriminately - it can't forget; gates give selective, learned retention.
</details>

---

## Exercise 11 (hard, code) - Implement relative position bias

Content-only attention can't prefer "the token 1 step back." T5-style fix:
add a learned scalar bias $b_{i-j}$ to each score before softmax:

$$
A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + B\right),
\qquad B_{ij} = b_{i-j}
$$

Write numpy code that builds $B \in \mathbb{R}^{T\times T}$ from a learned
table `bias` of shape $(2T{-}1,)$ (offsets $-(T{-}1) \dots T{-}1$), applies
it inside causal attention, and demonstrate that a suitable table makes
every token attend ~entirely to its previous token, *regardless of content*.
Why can't absolute position embeddings do this as directly?

<details><summary>Solution</summary>

```python
import numpy as np
np.random.seed(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

T, d = 6, 8

def rel_bias_matrix(bias, T):
    # bias: (2T-1,) learned scalars for offsets -(T-1)..(T-1)
    # B[i, j] = bias[(i - j) + (T - 1)]
    i = np.arange(T)[:, None]              # (T, 1)
    j = np.arange(T)[None, :]              # (1, T)
    return bias[(i - j) + (T - 1)]         # (T, T) via fancy indexing

def causal_attention_with_bias(Q, K, V, bias):
    S = (Q @ K.T) / np.sqrt(Q.shape[-1])   # (T, T) content scores
    S = S + rel_bias_matrix(bias, len(Q))  # + position scores
    S = np.where(np.triu(np.ones_like(S, dtype=bool), 1), -np.inf, S)
    A = softmax(S, axis=-1)
    return A @ V, A

Q = np.random.randn(T, d); K = np.random.randn(T, d); V = np.random.randn(T, d)

# A table that screams "offset +1" (previous token): big bias at i - j = 1.
bias = np.zeros(2 * T - 1)
bias[(T - 1) + 1] = 10.0                   # offset +1 slot

_, A = causal_attention_with_bias(Q, K, V, bias)
print(np.round(A, 3))
# Rows 1..T-1 put ~all weight on column i-1, independent of Q,K content.
assert all(A[i, i - 1] > 0.98 for i in range(1, T))
```

The bias enters the *logits*, so a large $b_{+1}$ dominates the
content scores $QK^\top/\sqrt{d_k}$ (which are $O(1)$ after scaling) and
softmax concentrates on offset 1 for every row - a purely positional head.

Why absolute embeddings can't do it as directly: with absolute positions,
"attend to the previous token" must be expressed as a *content* match
between the encodings of position $i$ and position $i{-}1$ - the model has
to learn a $W_Q, W_K$ pair such that $\langle W_Q p_i, W_K p_j\rangle$ is
large iff $j = i - 1$ for **every** $i$ simultaneously (and it won't
generalize past trained positions). Relative bias parameterizes the offset
*once*, shared across all $i$, and generalizes to any position by
construction. This is the conceptual core of T5 bias, ALiBi, and RoPE.
</details>

---

## Exercise 12 (hard) - Generation FLOPs, with and without the cache

For one transformer layer with width $d$ and current context length $t$,
count the attention compute in two parts: (i) the projections $q, k, v, o$,
and (ii) the scores plus the weighted sum. Per token generated:

(a) Without a KV-cache (re-run the whole prefix each step), what is the cost
of generating token $t{+}1$, and the total for $n$ tokens?
(b) With a KV-cache, same questions.
(c) Both totals still contain an $n^2$ term. So what did the cache actually
buy? And why, despite (b), is generation throughput usually limited by
memory bandwidth rather than FLOPs?

<details><summary>Solution</summary>

Per layer, full forward on $t$ tokens: projections
$\approx 8td^2$ FLOPs ($4$ matrices, $2td^2$ each), attention
$\approx 4t^2 d$ (scores $2t^2d$ + weighted sum $2t^2d$), MLP
$\approx 16td^2$. Call it $O(td^2 + t^2 d)$.

(a) No cache: step $t$ re-processes all $t$ tokens:
$O(t d^2 + t^2 d)$ per step. Summing $t = 1..n$:
$O(n^2 d^2 + n^3 d)$ - **cubic** in sequence length.

(b) With cache: step $t$ processes one token - projections/MLP $O(d^2)$,
plus one query against $t$ cached keys: $O(t d)$. Total over $n$ tokens:
$O(n d^2 + n^2 d)$.

(c) The cache removed a factor of $n$: total generation cost dropped from
$O(n^3 d + n^2d^2)$ to $O(n^2 d + nd^2)$ - the same order as a *single*
training forward pass over $n$ tokens, which is the information-theoretic
floor (you must at least "look at" each pair once under causal attention).
The residual $n^2$ is the price of exact attention itself, not of
generation. Bandwidth: each cached-generation step does only $O(d^2)$ work
per layer but must *stream the entire layer's weights* (and the growing KV
cache) from memory to compute one token's activations - arithmetic
intensity (FLOPs per byte moved) is tiny at batch 1, so the GPU idles on
memory. Hence batching many sequences (reuse each streamed weight tile
across the batch) and speculative decoding (amortize a big model's memory
pass over several drafted tokens) are throughput wins even though they add
FLOPs.
</details>

---

## Exercise 13 (easy-medium) - RoPE by hand in $d = 2$

Let $\theta = 30°$ and take raw (pre-rotation) vectors $q = k = (1, 0)$.
RoPE places a vector at position $m$ by rotating it by $m\theta$:
$q_m = R(m\theta)q$ with
$R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}$.

(a) Compute $q_1$ (query at position 1) and $k_3$ (key at position 3), and
their dot product.
(b) Now shift both tokens 4 positions later: compute $q_5$ and $k_7$ and
their dot product. Compare.
(c) Prove the pattern in general: show
$q_m^\top k_n = q^\top R\big((n - m)\theta\big) k$ using only the
$2\times 2$ rotation algebra, and state precisely what does and does not
depend on absolute position.

<details><summary>Solution</summary>

(a) $q_1 = (\cos 30°, \sin 30°) = (0.866,\ 0.5)$;
$k_3 = (\cos 90°, \sin 90°) = (0,\ 1)$.
Dot product: $0.866\cdot 0 + 0.5 \cdot 1 = 0.5$.
(Check: $\cos((3-1)\cdot 30°) = \cos 60° = 0.5$.)

(b) $q_5 = (\cos 150°, \sin 150°) = (-0.866,\ 0.5)$;
$k_7 = (\cos 210°, \sin 210°) = (-0.866,\ -0.5)$.
Dot product: $(-0.866)(-0.866) + (0.5)(-0.5) = 0.75 - 0.25 = 0.5$.
Same score: the gap is still 2, and the score moved not at all even though
every absolute position changed.

(c) $q_m^\top k_n = (R(m\theta)q)^\top R(n\theta)k
= q^\top R(m\theta)^\top R(n\theta) k$. Multiply out:

$$
R(a)^\top R(b) =
\begin{pmatrix}\cos a & \sin a\\ -\sin a & \cos a\end{pmatrix}
\begin{pmatrix}\cos b & -\sin b\\ \sin b & \cos b\end{pmatrix}
=
\begin{pmatrix}\cos(b-a) & -\sin(b-a)\\ \sin(b-a) & \cos(b-a)\end{pmatrix}
= R(b-a)
$$

using $\cos a\cos b + \sin a\sin b = \cos(b - a)$ and
$\sin a\cos b - \cos a\sin b = \sin(a - b)$. So
$q_m^\top k_n = q^\top R((n-m)\theta)k$: the **positional** dependence is
only through the gap $n - m$; the score still depends fully on the
*content* $q$ and $k$. (With our $q = k = (1,0)$ the score is exactly
$\cos((n-m)\theta)$ - content chosen to make the position part visible.)
</details>

---

## Exercise 14 (easy-medium) - KV-cache: MHA vs GQA

A decoder has $n_{\text{layers}} = 32$, $h = 32$ query heads of
$d_{\text{head}} = 128$ (so $d = 4096$), context $T = 8192$, cache in fp16
(2 bytes).

(a) Write the general formula for KV-cache bytes per sequence with
$n_{\text{kv}}$ key/value heads, and evaluate it for standard MHA
($n_{\text{kv}} = h$).
(b) Same for GQA with $g = 8$ KV groups, and for MQA. What's the general
shrink factor?
(c) The GPU has 80 GB, and the model's fp16 weights take ~14 GB. Roughly
what batch of full-context sequences fits under MHA vs GQA($g{=}8$)?
Why does GQA also speed up each decoding step, not just enlarge batches?

<details><summary>Solution</summary>

(a) $\text{bytes} = 2 \cdot n_{\text{layers}} \cdot T \cdot n_{\text{kv}}
\cdot d_{\text{head}} \cdot 2$ (the leading 2 = K and V, the trailing 2 =
fp16). MHA: $2 \cdot 32 \cdot 8192 \cdot 32 \cdot 128 \cdot 2 =
4{,}294{,}967{,}296$ B $= 4$ GiB per sequence.

(b) Cache scales linearly in $n_{\text{kv}}$, so the shrink factor is
$h / g$. GQA $g = 8$: $4 \cdot \frac{8}{32} = 1$ GiB per sequence.
MQA ($g = 1$): $4/32 = 128$ MiB per sequence.

(c) Budget for cache $\approx 80 - 14 \approx 66$ GB. MHA: $\lfloor 66/4
\rfloor \approx 16$ sequences. GQA: $\approx 66$ sequences - ~4× the
serving throughput from the same hardware. Speed: decoding is
memory-bandwidth-bound (Exercise 12c); every generated token must stream
the cache through the GPU, and GQA streams $4\times$ fewer cache bytes per
step - so per-token latency drops too. Quality is roughly preserved
because all $h$ query heads (hence $h$ attention distributions per token)
remain; only the K/V subspace diversity shrinks (notes §7.2).
</details>

---

## Exercise 15 (medium-hard) - The ELBO from the KL-gap identity

Let $p_\theta(x, z) = p(z)\,p_\theta(x\mid z)$ be a latent-variable model
and $q_\phi(z \mid x)$ any distribution with the right support.

(a) Starting from $\operatorname{KL}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x))$
and using only $p(z\mid x) = p(x,z)/p(x)$, derive the identity
$\log p_\theta(x) = \operatorname{ELBO} + \operatorname{KL}(q_\phi \| p_\theta(z\mid x))$
and read off the ELBO.
(b) Rewrite the ELBO as reconstruction minus a KL to the *prior*.
(c) Why does maximizing the ELBO over $\phi$ with $\theta$ fixed make the
bound tighter? What is the gap exactly?
(d) Why can't you backprop $\nabla_\phi$ through $z \sim q_\phi$ directly,
and how does reparameterization fix it?

<details><summary>Solution</summary>

(a)

$$
\begin{aligned}
\operatorname{KL}(q \,\|\, p(z\mid x))
&= \mathbb{E}_{q}[\log q(z\mid x) - \log p(z\mid x)]\\
&= \mathbb{E}_{q}[\log q(z\mid x) - \log p(x,z)] + \log p(x)
\end{aligned}
$$

(using $\log p(z\mid x) = \log p(x,z) - \log p(x)$; the $\log p(x)$ term
leaves the expectation because it has no $z$ in it). Rearranged:

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{q}[\log p_\theta(x,z) - \log q_\phi(z\mid x)]}_{\text{ELBO}}
+ \operatorname{KL}(q_\phi \,\|\, p_\theta(z\mid x)) \ \ge\ \operatorname{ELBO},
$$

since KL $\ge 0$. This is an identity - no approximation was made.

(b) Split $\log p(x,z) = \log p(x\mid z) + \log p(z)$:

$$
\operatorname{ELBO} = \mathbb{E}_{q}[\log p_\theta(x \mid z)]
- \operatorname{KL}(q_\phi(z\mid x)\,\|\,p(z)),
$$

reconstruction minus prior-regularization; both terms computable (the KL
between Gaussians is closed-form, reconstruction by sampling
$z \sim q_\phi$).

(c) With $\theta$ fixed, $\log p_\theta(x)$ is a constant in $\phi$, so by
the identity, raising the ELBO in $\phi$ can only *lower* the other term:
the gap between $\log p(x)$ and the ELBO is exactly
$\operatorname{KL}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x))$ - the
approximate posterior's error. ELBO maximization in $\phi$ *is* posterior
approximation.

(d) The expectation is over a distribution that itself depends on $\phi$:
in $\nabla_\phi \mathbb{E}_{z\sim q_\phi}[f(z)]$, "sample, then
differentiate" is invalid because the sampling step is not written as a
differentiable function of $\phi$. Reparameterize:
$z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$ with
$\varepsilon \sim \mathcal{N}(0,I)$ makes the expectation over a *fixed*
distribution, $\mathbb{E}_{\varepsilon}[f(\mu_\phi + \sigma_\phi \odot \varepsilon)]$,
and $z$ a deterministic differentiable map - ordinary backprop through
$\mu, \sigma$ gives a low-variance gradient (unlike the
score-function/REINFORCE alternative, which needs no reparameterization
but is high-variance).
</details>

---

## Exercise 16 (medium) - The diffusion closed form, and reading a schedule

(a) Starting from $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$,
prove by induction that $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$
with $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$. Show the variance step
explicitly.

(b) For the linear schedule $\beta_t$ from $10^{-4}$ to $0.02$ over
$T = 1000$ steps, estimate $\bar\alpha_T$, and $\bar\alpha_t$ at $t = 100$
and $t = 500$. At which of these can a viewer still recognize the image?

(c) Why does training never need to simulate the chain step by step?

<details><summary>Solution</summary>

(a) Assume $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\varepsilon'$.
Then $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon''$ has
mean coefficient $\sqrt{\alpha_t\bar\alpha_{t-1}} = \sqrt{\bar\alpha_t}$ and
noise variance $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = \alpha_t - \bar\alpha_t + 1 - \alpha_t = 1 - \bar\alpha_t$,
since independent Gaussian noises add in variance. A Gaussian with that
mean and variance is $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$.

(b) $\log\bar\alpha_T = \sum_t \log(1-\beta_t) \approx -\sum_t \beta_t$
for small $\beta_t$. The linear schedule's sum is
$1000 \cdot (10^{-4} + 0.02)/2 = 10.05$, so
$\bar\alpha_T \approx e^{-10.05} \approx 4 \times 10^{-5}$ (exact product:
$4.0 \times 10^{-5}$). At $t = 100$ the sum of the first 100 betas is
about $0.11$, so $\bar\alpha_{100} \approx 0.90$: signal coefficient
$\sqrt{0.90} = 0.95$, noise coefficient $0.32$; clearly recognizable. At
$t = 500$, $\bar\alpha_{500} \approx 0.079$: signal $0.28$, noise $0.96$;
barely, if at all. Most of the perceptual destruction happens in the
middle of the schedule, which is why later schedules (cosine) spread it
out.

(c) The closed form gives $x_t$ from $x_0$ in one draw of $\varepsilon$,
for any $t$. Training samples $(x_0, t, \varepsilon)$, forms $x_t$
directly, and regresses $\varepsilon_\theta(x_t, t)$ onto $\varepsilon$.
The chain is only ever run at sampling time.
</details>

---

## Exercise 17 (easy-medium) - Mixture-of-experts arithmetic

Mixtral 8x7B: 32 layers, $d = 4096$, feed-forward width $14336$, SwiGLU
experts (three weight matrices each), $E = 8$ experts per layer, top-$2$
routing.

(a) Parameters per expert, and in all experts across the model.

(b) Expert parameters touched per token. Why is the model described as
"47B total, 13B active"?

(c) A colleague says an 8-expert top-2 MoE should be 4 times cheaper to
serve than a dense model of the same total size. Give the two reasons
that is wrong for decode.

(d) The load-balancing loss is $E\sum_i f_i P_i$. Evaluate it for uniform
routing and for total collapse, and explain why it has a gradient at all
given that the $f_i$ are counts.

<details><summary>Solution</summary>

(a) One expert: $3 \cdot 4096 \cdot 14336 \approx 1.76 \times 10^8$, about
176M. All experts: $32 \cdot 8 \cdot 176\text{M} \approx 45$B. With
attention and embeddings the total is about 47B.

(b) Per token, 2 experts per layer: $32 \cdot 2 \cdot 176\text{M} \approx 11.3$B,
plus the shared attention parameters, about 13B active. Every token's
forward pass uses roughly 13B of the 47B, which is the compute of a 13B
dense model.

(c) First, module 7's decode arithmetic: decode is memory-bandwidth-bound,
and every step must read the weights it uses. With batched requests
routing to different experts, most or all 47B of weights get read per
step, so bandwidth cost tracks total size, not active size. Second, all
47B must be resident in accelerator memory regardless, so the model needs
the memory footprint of a 47B model. MoE's FLOP saving shows up in
training and in compute-bound prefill, much less in decode.

(d) Uniform: $f_i = P_i = 1/E$, loss $= E \cdot E \cdot (1/E)^2 = 1$.
Collapse onto one expert: $f = P = 1$ there, $0$ elsewhere, loss $= E$.
The $f_i$ carry no gradient, but each $P_i$ is a mean of softmax outputs
and does; $\partial \mathcal{L}/\partial P_i = E f_i$ is largest for the
most-used expert, so the router is pushed to lower its probability on
whichever experts are currently overloaded.
</details>
