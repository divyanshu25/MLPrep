# The LLM Stack - scaling, distributed training, post-training, inference, evals

This module goes one level below module 4. There we derived the transformer;
here we derive everything around it: how big to make it, how to fit training
on real hardware, how to turn a next-token predictor into an assistant, how to
serve it fast, and how to know whether any of it worked. Every formula is
derived and every estimate worked with real numbers, because these are exactly
the questions frontier labs ask - "walk me through the memory of training a
70B model" is not trivia, it's a fluency check.

Companion scripts: `memory_math.ipynb` (all the memory arithmetic, runnable),
`speculative_decoding_demo.ipynb` (proves the sampling algorithm is exact),
`lora_scratch.ipynb` (LoRA by hand, matching full fine-tuning).

---

## 1. Scaling laws

### 1.1 The empirical fact: loss is a power law

Train a family of transformers of different sizes on different amounts of
data, plot final cross-entropy loss against parameters $N$ or tokens $D$ on
log-log axes, and you get straight lines over many orders of magnitude. The
functional form that fits (Hoffmann et al. 2022, "Chinchilla"):

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Read this term by term, because each has a meaning:

- $E$ - the **irreducible loss**: the entropy of language itself. Even a
  perfect model cannot predict genuinely uncertain next tokens. No amount of
  scale removes this term.
- $A/N^\alpha$ - the **approximation error** from finite capacity: a model
  with $N$ parameters can only represent so much. Fitted $\alpha \approx 0.34$.
- $B/D^\beta$ - the **estimation error** from finite data: with $D$ tokens
  you can only estimate so much. Fitted $\beta \approx 0.28$.

Why power laws and not exponentials? An honest answer: it's an empirical
regularity first, with theory catching up (one story: natural data has a
long-tailed spectrum of "features" by frequency, and each order of magnitude
of scale lets you learn the next slice of the tail; learning a power-law tail
of skills yields a power-law loss). What matters for practice is that the fit
is stable enough to **extrapolate**: measure the line with cheap runs, predict
the loss of a run 1000× more expensive. That is the whole game.

### 1.2 Deriving the compute cost: $C \approx 6ND$

Before we can optimize anything we need the cost of training in FLOPs. Take a
dense transformer with $N$ parameters and push one token through it.

**Forward pass ≈ $2N$ FLOPs per token.** Almost all parameters live in weight
matrices, and a weight matrix $W \in \mathbb{R}^{d_{in} \times d_{out}}$
applied to one token's vector costs $d_{in} \cdot d_{out}$ multiplies plus
$d_{in} \cdot d_{out}$ adds $= 2\,d_{in} d_{out}$ FLOPs - i.e. **2 FLOPs per
parameter**, per token. Summing over all matrices: $2N$ FLOPs. (Attention's
$QK^\top$ and $AV$ products add about $4sd$ FLOPs per token per layer, against
about $24d^2$ for the layer's parameters, so the ratio is about $s/(6d)$, or
$s/(12d)$ with a causal mask; the number of layers cancels. For $s \ll 6d$
this is a small correction we ignore; for typical training contexts it is a
few percent to ~10%.)

**Backward pass ≈ $4N$ FLOPs per token.** For each matmul $Y = XW$ the
backward needs two matmuls: $\partial X = \partial Y\, W^\top$ (to keep the
chain rule going) and $\partial W = X^\top \partial Y$ (to get the weight
gradient). Each costs the same as the forward matmul, so backward is 2×
forward: $4N$.

Total: $6N$ FLOPs per token, so for $D$ training tokens

$$\boxed{C \approx 6ND}$$

Sanity check on a real run: Chinchilla, $N = 70$B, $D = 1.4$T tokens:
$C = 6 \times 7\times10^{10} \times 1.4\times10^{12} \approx 5.9 \times
10^{23}$ FLOPs - matches the paper's reported budget.

### 1.3 The Chinchilla question: given $C$, how do you split it?

You have a fixed budget $C$. You can train a big model on little data or a
small model on lots of data - $C = 6ND$ is a hyperbola of choices. Which point
minimizes loss? Substitute the constraint $D = C/6N$ into the loss:

$$L(N) = E + A N^{-\alpha} + B \left(\frac{6N}{C}\right)^{\beta}$$

Now it's a 1-D optimization. The first scale term *falls* in $N$ (bigger model
= less approximation error), the second *rises* in $N$ (bigger model = fewer
tokens affordable = more estimation error). Differentiate and set to zero:

$$\frac{dL}{dN} = -\alpha A N^{-\alpha - 1} + \beta B \, 6^\beta C^{-\beta} N^{\beta - 1} = 0$$

$$\Rightarrow\; N^{\alpha + \beta} = \frac{\alpha A}{\beta B\, 6^\beta}\, C^{\beta}
\;\Rightarrow\; N_{opt} \propto C^{\beta/(\alpha+\beta)}, \qquad D_{opt} = \frac{C}{6N_{opt}} \propto C^{\alpha/(\alpha+\beta)}$$

With the fitted $\alpha \approx 0.34$, $\beta \approx 0.28$: both exponents
are $\approx 0.5$. That is the Chinchilla headline:

> **Scale parameters and tokens together, each like $\sqrt{C}$.** If you get
> 100× more compute, make the model ~10× bigger and train on ~10× more tokens.

The pre-Chinchilla mistake (GPT-3 era, from the earlier Kaplan fit) was
scaling $N$ much faster than $D$ - GPT-3 was 175B params on only 300B tokens,
i.e. dramatically **undertrained**: a smaller model on more tokens would have
been strictly better at the same cost.

### 1.4 The 20-tokens-per-parameter rule, and a worked example

At the Chinchilla optimum the fitted constants give a simple ratio:
$D_{opt}/N_{opt} \approx 20$ tokens per parameter (Chinchilla itself: 1.4T /
70B = 20). Treat 20 as a rule of thumb at roughly this compute scale: the
fitted exponents make the ratio drift slowly with $C$, and a 2024 replication
(Besiroglu et al.) found errors in the original fit. Kaplan's earlier, more
parameter-heavy answer differed mainly because it counted only non-embedding
parameters and did not match the learning-rate schedule to each run's
length. Combine with $C = 6ND$:

$$C = 6 N (20N) = 120 N^2 \;\Rightarrow\; N_{opt} = \sqrt{C/120}, \quad D_{opt} = 20 N_{opt}$$

**Worked interview example.** *"You have $10^{23}$ FLOPs. What model do you
train?"*

$$N = \sqrt{10^{23}/120} = \sqrt{8.3\times10^{20}} \approx 2.9\times10^{10} \approx 29\text{B params}$$
$$D = 20N \approx 5.8\times10^{11} \approx 580\text{B tokens}$$

Check: $6 \times 2.9\times10^{10} \times 5.8\times10^{11} \approx 1.0 \times
10^{23}$. ✓ So: a ~30B model on ~600B tokens. Being able to do this in 30
seconds on a whiteboard is the point of this section.

**The modern caveat (always mention it).** Chinchilla-optimal minimizes
*training* loss per training FLOP. But a model is trained once and served
forever - inference cost scales with $N$, not $D$. If inference dominates your
lifetime compute, you should **overtrain a smaller model**: Llama-3-8B saw
~15T tokens ($\approx$ 1900 tokens/param, ~100× "over" Chinchilla), eating a
worse training-compute trade to buy a cheaper, faster model at serving time.
The right ratio depends on the (training + lifetime inference) total cost, not
training alone.

### 1.5 How labs actually use scaling laws

1. **Fit small, extrapolate big.** Run a grid of small models (e.g. 10M–1B)
   across data sizes; fit $E, A, B, \alpha, \beta$. The fit at <1% of the
   final budget predicts the final loss. GPT-4's report states its final loss
   was predicted from runs with ~10,000× less compute.
2. **Tune hyperparameters at small scale, transfer.** LR, batch, init don't
   transfer naively - width-aware parameterizations (µP) make the optimal LR
   approximately scale-invariant so you can sweep at 40M and trust it at 40B.
3. **De-risk architecture changes.** A new attention variant must shift the
   whole scaling *curve*, not win at one size. Many "improvements" help at
   small scale and vanish at large scale; the scaling-law comparison is the
   filter.
4. **Data strategy.** If you're data-constrained (the internet is finite),
   the laws quantify how much repeating data hurts (roughly: up to ~4 epochs
   is nearly as good as fresh tokens; beyond that returns decay fast), and
   justify data-quality filtering - better data shifts $B$ down.

---

## 2. Distributed training

### 2.1 The memory arithmetic of training - derive 16 bytes/param

Everything in this section follows from one small table. Standard recipe:
**mixed precision + Adam**. Per parameter we must store:

| what | precision | bytes | why it exists |
|---|---|---|---|
| weights (working copy) | fp16/bf16 | 2 | fast matmuls on tensor cores |
| gradients | fp16/bf16 | 2 | produced by backward at matmul precision |
| master weights | fp32 | 4 | fp16 update $w \mathrel{-}= \eta g$ gets lost: fp16 values near 1 are spaced $2^{-10}$ apart, so with round-to-nearest any update below $2^{-11}$ vanishes, and $\eta g$ is often far smaller; accumulate in fp32 and round to fp16 for the matmuls |
| Adam first moment $m$ | fp32 | 4 | Adam's running mean of gradients |
| Adam second moment $v$ | fp32 | 4 | Adam's running mean of squared gradients |

$$\text{Total} = 2 + 2 + 4 + 4 + 4 = \boxed{16 \text{ bytes/param}}$$

The "12" you'll see in the ZeRO paper is the **optimizer state**: master
weights + $m$ + $v$ $= 12$ bytes. Variants push 16 to 20: fp32 gradient
accumulation across microbatches adds 4 (→ 20 bytes/param); pure-fp32
training is $4+4+4+4 = 16$ too but with 4-byte grads and no fp16 copy.
So the interview-safe answer: **~16–20 bytes per parameter, before
activations**.

Notice the surprising part: the model you're training (2 bytes/param) is the
*smallest* item in memory. Training state is ~8× the model.

### 2.2 The canonical question: "How much memory to train a 70B model?"

Walk it through exactly like this:

1. **States**: $16 \times 70\times10^9 = 1.12\times10^{12}$ bytes = **1.12 TB**
   (weights 140 GB, grads 140 GB, optimizer 840 GB).
2. **Activations** (before any tricks): for a transformer layer the
   intermediate tensors scale like $s \cdot b \cdot h$ per layer. A good
   estimate (Megatron paper, per layer, fp16, standard attention):
   $\text{bytes} \approx s b h (34 + 5as/h)$ where $s$=seq len, $b$=batch,
   $h$=hidden, $a$=heads. For 70B ($h=8192, a=64, L=80$), $s=4096$, $b=1$
   per GPU: $\approx 4096 \cdot 8192 \cdot (34 + 5\cdot64\cdot4096/8192)
   \approx 6.5$ GB *per layer* → ~520 GB for 80 layers. Nobody pays this  - 
   gradient checkpointing (keep only each layer's input and recompute the
   rest during backward; §2.7 derives it) reduces it to roughly the layer inputs,
   $2sbh L \approx 5.4$ GB, plus recompute.
3. **Conclusion**: ~1.12 TB of state alone. One H100 has 80 GB. So you need
   at least $1120/80 = 14$ GPUs *just to hold training state*, ignoring
   activations, buffers, and fragmentation - and that's only possible if you
   **shard** the state (ZeRO), because plain data parallelism replicates all
   1.12 TB on every GPU. This is why the rest of this section exists.

Quick reference: 7B model → $16 \times 7\times10^9 = 112$ GB of state. Does
not fit on one 80 GB GPU. *Training even a 7B model naively doesn't fit on
the best single GPU* - a fact interviewers love because people who've never
done the arithmetic assume it fits.

### 2.3 Data parallelism (DP): replicate the model, split the data

The baseline. $G$ GPUs each hold a **full copy** of everything; each step,
each GPU takes a different shard of the batch, runs forward+backward locally,
then all GPUs **average their gradients** before updating.

Why must gradients sync? Because the loss is a sum over the batch, and the
gradient of a sum is the sum of gradients:
$\nabla L = \frac{1}{B}\sum_i \nabla \ell_i = \frac{1}{G}\sum_g \left(\frac{1}{B/G}\sum_{i \in g} \nabla \ell_i\right)$.
Averaging local gradients makes $G$ GPUs mathematically identical to one GPU
with a $G\times$ bigger batch. If you skip the sync, replicas drift apart and
you're training $G$ diverging models.

The averaging primitive is **all-reduce**. The bandwidth-optimal ring
algorithm: arrange $G$ GPUs in a ring, split the gradient into $G$ chunks,
do $G-1$ steps of reduce-scatter (each GPU ends owning the full sum of one
chunk) then $G-1$ steps of all-gather (circulate the reduced chunks). Total
traffic per GPU: $2\frac{G-1}{G}\cdot(\text{bytes}) \approx 2\times$ the
gradient size, *independent of $G$*. For 70B in fp16 that's ~280 GB moved per
step per GPU - overlapped with backward (sync layer $\ell$'s grads while
computing layer $\ell-1$'s) so it hides behind compute.

DP scales the batch, not the model: memory per GPU is unchanged. Fix that
next.

### 2.4 ZeRO: stop replicating what you can shard

Observation: in DP, all $G$ GPUs store identical optimizer state, identical
gradients, identical weights. Redundant. ZeRO (Zero Redundancy Optimizer)
shards each in turn across the DP group. Per-GPU memory with $\Psi$ params:

| stage | shards | per-GPU bytes | 70B, $G=64$ |
|---|---|---|---|
| DP baseline | nothing | $16\Psi$ | 1120 GB ✗ |
| ZeRO-1 | optimizer state | $2\Psi + 2\Psi + 12\Psi/G$ | $280 + 13.1 = 293$ GB ✗ |
| ZeRO-2 | + gradients | $2\Psi + (2{+}12)\Psi/G$ | $140 + 15.3 = 155$ GB ✗ |
| ZeRO-3 | + weights | $16\Psi/G$ | $17.5$ GB ✓ |

Derivations of each row:

- **Stage 1**: each GPU keeps full fp16 weights and grads (so forward/backward
  are unchanged), but only $1/G$ of the 12-byte optimizer state. After the
  grad all-reduce, each GPU updates only its shard of the master weights, then
  an all-gather distributes the updated fp16 weights. Communication volume:
  same as DP.
- **Stage 2**: gradients are also sharded - replace all-reduce with
  **reduce-scatter** (each GPU receives only the summed gradient shard it
  owns, then frees the rest). Still same total communication as DP.
- **Stage 3**: weights sharded too. Now no GPU holds the full model, so
  layers' weights are **all-gathered on the fly** just before use in
  forward and backward, then freed. Cost: an extra all-gather of the weights
  in backward → total comm ≈ 1.5× DP. You trade 50% more communication for
  $G\times$ less memory.

So: 70B on 64 GPUs *requires* stage 3 (or model parallelism); stages 1–2
aren't enough because 140–280 GB of unsharded weights+grads still exceed one
GPU. On the other hand a 7B model on 8 GPUs is fine at stage 2
($14 + 98/8 \approx 26$ GB): prefer the lowest stage that fits, since higher
stages cost communication.

**FSDP** (PyTorch Fully Sharded Data Parallel) is ZeRO-3 as a native PyTorch
implementation: parameters are flattened, sharded, all-gathered per layer
("unit") on demand, freed after use, gradients reduce-scattered. Same math,
different engineering.

### 2.5 Tensor parallelism (TP): split individual matmuls

ZeRO shards *storage* but every GPU still computes full layers. When a single
layer's compute or activation is too big, split the matmul itself
(Megatron-LM). Two splits compose beautifully in the MLP block
$Z = \sigma(XW_1)W_2$:

**Column split** $W_1 = [W_1^{(1)}\; W_1^{(2)}]$ (columns divided over 2
GPUs):

$$XW_1 = [\,XW_1^{(1)} \;\; XW_1^{(2)}\,]$$

Each GPU computes its half of the hidden vector **with no communication**  - 
and can apply the elementwise nonlinearity locally, since $\sigma$ acts
per-coordinate.

**Row split** $W_2 = \begin{bmatrix} W_2^{(1)} \\ W_2^{(2)} \end{bmatrix}$
(rows divided the same way):

$$YW_2 = Y^{(1)}W_2^{(1)} + Y^{(2)}W_2^{(2)}$$

Each GPU multiplies the hidden half it already has by its row block, producing
a **partial sum** of the full output. One **all-reduce** adds them. So an
entire MLP block = column-split → local nonlinearity → row-split → *one*
all-reduce in forward (and one in backward, for the same reason mirrored).
Attention is analogous: split heads across GPUs (heads are independent - a
natural column split of $W_{Q,K,V}$), row-split the output projection
$W_O$, one all-reduce.

Per transformer layer: 2 all-reduces forward, 2 backward, each of size
$s \cdot b \cdot h$ - every layer, every step, **not** overlappable with much.
That's why TP lives *inside a node* (NVLink, ~900 GB/s) and rarely crosses
nodes (InfiniBand, ~50 GB/s/GPU): TP degree is typically ≤ 8.

### 2.6 Pipeline parallelism (PP): split by layers

Put layers 1–20 on GPU0, 21–40 on GPU1, … The problem: GPU1 idles while GPU0
computes, and vice versa - with naive scheduling, $p$ stages are each busy
$1/p$ of the time.

Fix: split the batch into $m$ **microbatches** and stream them through. In a
steady state all stages work on different microbatches simultaneously; only
the fill and drain of the pipe are idle. Count idle slots: the pipeline takes
$m + p - 1$ time units to push $m$ microbatches through $p$ stages, of which
$m$ are useful, so

$$\text{bubble fraction} = \frac{p-1}{m + p - 1}$$

$p=8$, $m=32$: bubble $= 7/39 \approx 18\%$ wasted. Push $m$ up (more
microbatches) to amortize - but each in-flight microbatch holds activations,
so schedules like 1F1B interleave forwards and backwards to cap the number of
live activations at ~$p$ microbatches instead of $m$.

**Composing all three (3D parallelism)**: TP inside a node (degree ≤ 8), PP
across nodes (cheap point-to-point activations between stages), DP/ZeRO
across the remaining axis to scale the batch. E.g. Llama-3 405B: TP=8, PP=16,
DP=whatever fills the cluster.

### 2.7 Activation memory and gradient checkpointing - the $\sqrt{L}$ trick

Backward needs the forward's intermediate values ($\partial W = X^\top
\partial Y$ needs $X$). Naively you store every layer's activations:
memory $\propto L$ layers. For long contexts this dwarfs the weights (§2.2).

**Gradient checkpointing**: store only some activations ("checkpoints"),
recompute the rest during backward. Derive the classic trade: split $L$
layers into $k$ segments and keep only the $k$ segment-boundary activations.
During backward, to process a segment you re-run its forward from its
checkpoint, temporarily materializing $L/k$ layers of activations. Peak
memory:

$$M(k) \propto \underbrace{k}_{\text{checkpoints}} + \underbrace{L/k}_{\text{recompute window}}, \qquad \frac{dM}{dk} = 1 - \frac{L}{k^2} = 0 \Rightarrow k = \sqrt{L}$$

giving $M \propto 2\sqrt{L}$ - memory drops from $O(L)$ to $O(\sqrt{L})$  - 
at the cost of exactly **one extra forward pass** (every layer is recomputed
once). Since forward is $1/3$ of the $6N$ total (§1.2), checkpointing costs
~33% more FLOPs - nearly always worth it. (Common practice is even simpler:
checkpoint *every* layer boundary; memory $\propto L$ but with a tiny
constant - one $sbh$ tensor per layer instead of ~34.)

### 2.8 The linear scaling rule: LR vs batch size

Scale-out data parallelism grows the batch $B \to kB$. Should the LR change?
Compare $k$ small-batch SGD steps against 1 big-batch step. Small batches,
LR $\eta$, assuming gradients change slowly over these steps
($\nabla L(w_t) \approx \nabla L(w_0)$):

$$w_{k} \approx w_0 - \eta \sum_{j=1}^{k} g_j(w_0)$$

One step with batch $kB$ (which averages those same $k$ mini-gradients) and
LR $\hat\eta$:

$$w_1 = w_0 - \hat\eta \cdot \frac{1}{k}\sum_{j=1}^{k} g_j(w_0)$$

These match when $\hat\eta = k\eta$: **scale LR linearly with batch size**
(Goyal et al., who trained ImageNet at batch 8192 this way). Caveats that
show you actually understand it: (i) the "gradients change slowly" assumption
breaks early in training → use LR **warmup**; (ii) it breaks past a
problem-dependent **critical batch size** where gradient noise is no longer
the bottleneck - beyond it, bigger batches stop buying faster convergence per
sample; (iii) for Adam the empirical rule is closer to $\sqrt{k}$ scaling,
since Adam normalizes by gradient scale.

---

## 3. Post-training: from token predictor to assistant

A pretrained LM models $P(\text{text})$. It completes documents; it doesn't
answer questions, follow instructions, or decline harmful requests. Post-
training reshapes the *policy* - which continuations it chooses - without
re-teaching the knowledge. Pipeline: **SFT → reward model → RL** (or the DPO
shortcut).

**SFT (supervised fine-tuning)**: continue next-token training on
(instruction, good response) pairs, loss only on response tokens. This sets
the format and persona. Limitation: it can only imitate demonstrations - it
teaches "write like this", never "this answer is *better* than that one",
and demonstration data caps at the demonstrators' quality.

### 3.1 Reward model: Bradley–Terry from pairwise preferences

Humans are bad at absolute scores ("rate this 7.2/10") but reliable at
comparisons ("A is better than B"). So we collect pairs: prompt $x$, two
responses, label which is preferred: $y_w \succ y_l$. To learn from
comparisons we need a probabilistic model of comparisons.

**Bradley–Terry**: posit each response has a latent scalar quality
$r(x, y)$, and preferences are noisy comparisons of qualities:

$$P(y_1 \succ y_2 \mid x) = \frac{e^{r(x,y_1)}}{e^{r(x,y_1)} + e^{r(x,y_2)}} = \sigma\big(r(x,y_1) - r(x,y_2)\big)$$

(divide through by $e^{r(x,y_2)}$ to get the sigmoid form). This is exactly
logistic regression on quality *differences* - the same functional form as
Elo ratings in chess. Note only differences are identified: $r$ and $r + c(x)$
give identical preferences. Keep this fact; DPO will exploit it.

Train a reward model $r_\phi$ (usually the LM with a scalar head) by maximum
likelihood on the preference dataset:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\right]$$

### 3.2 PPO: optimize the reward without destroying the model

Naive objective: $\max_\theta \mathbb{E}_{y \sim \pi_\theta}[r_\phi(x,y)]$.
Two failure modes, and the PPO recipe adds one term for each:

**Failure 1 - reward hacking / distribution collapse.** $r_\phi$ is only a
model of human preference, trained near the SFT distribution. Optimize hard
against it and the policy finds out-of-distribution text the RM scores highly
but humans hate (sycophancy, verbose padding, weird token exploits), and the
policy can collapse to a low-entropy "reward-maxing" mode. Fix: penalize
divergence from the reference (SFT) policy. The actual objective is

$$\max_\theta \; \mathbb{E}_{x,\, y\sim\pi_\theta}\Big[ r_\phi(x,y)\Big] - \beta\, \mathbb{D}_{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{ref}(\cdot|x)\big)$$

The KL term is a leash: you may improve reward only in proportion to how far
you wander from a policy known to produce sensible language. $\beta$ trades
reward gain against drift.

**Failure 2 - destructive policy updates.** RL data is generated by the
current policy; after an update, old samples misrepresent the new policy.
Policy-gradient steps on stale data with a large step size can crater
performance unrecoverably. PPO's fix: for each token, define the probability
ratio $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$ and optimize
the **clipped surrogate**

$$\mathcal{L}^{CLIP} = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat A_t\big)\Big]$$

Read it: $\hat A_t$ is the advantage (how much better this token was than
expected - computed with a learned **value network** as baseline, to cut
variance). If the update tries to change a token's probability by more than
$\epsilon$ (~20%) in the direction the advantage pushes, the objective's
gradient is *zeroed* by the clip - a poor man's trust region, keeping each
update small enough that stale samples remain approximately valid.

So PPO-RLHF needs, in memory simultaneously: policy, reference policy, reward
model, value model - four LM-scale networks, plus on-policy generation every
step. It works but it's an engineering beast. That cost motivates DPO.

### 3.3 DPO: derive it - RLHF without RL

The key insight: the KL-regularized objective in §3.2 has a **closed-form
optimal policy**, and we can use it to eliminate the reward model entirely.

**Step 1 - solve the RLHF objective analytically.** Fix a prompt $x$ and any
reward $r$. We maximize over distributions $\pi(\cdot|x)$:

$$J(\pi) = \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\, \mathbb{D}_{KL}(\pi \,\|\, \pi_{ref}) = \mathbb{E}_{y\sim\pi}\left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

Divide by $-\beta$ and complete the KL: define
$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x)\, e^{r(x,y)/\beta}$ with
$Z(x) = \sum_y \pi_{ref}(y|x) e^{r(x,y)/\beta}$ (a valid distribution by
construction). Then

$$-\frac{J(\pi)}{\beta} = \mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y|x)}{\pi_{ref}(y|x)\, e^{r/\beta}}\right] = \mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y|x)}{\pi^*(y|x)}\right] - \log Z(x) = \mathbb{D}_{KL}(\pi\|\pi^*) - \log Z(x)$$

$Z$ doesn't depend on $\pi$, and KL is minimized (=0) iff $\pi = \pi^*$. So
the optimum of RLHF is, exactly:

$$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{ref}(y|x)\, \exp\!\big(r(x,y)/\beta\big)$$

Intuition: tilt the reference policy exponentially toward reward, with $\beta$
setting the tilt strength.

**Step 2 - invert: express the reward in terms of the policy.** Take logs and
solve for $r$:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Step 3 - substitute into Bradley–Terry.** Preferences only see reward
*differences* (§3.1), and the intractable $\beta \log Z(x)$ is the same for
both responses to the same prompt, so it **cancels**:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

**Step 4 - maximum likelihood, with $\pi_\theta$ playing $\pi^*$.** Every
policy is the optimal policy for *some* reward (its own implicit one,
$\hat r_\theta = \beta\log\frac{\pi_\theta}{\pi_{ref}}$). So fit the policy
directly to the preference data:

$$\boxed{\;\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)\right]\;}$$

The reward model, the sampling loop, the value network, PPO itself - all
gone. What remains is a supervised classification loss on fixed preference
pairs, computable with two forward passes per pair (policy and frozen
reference). The gradient (exercise 6) weighs each pair by how *wrong* the
implicit reward currently is, so confidently-wrong pairs dominate the update.

Tradeoffs vs PPO: DPO is offline (no fresh samples from the improving policy
→ can't discover new good behaviors, only reweight what's in the data), can
over-optimize by driving $\pi(y_l)$ to zero, and its implicit reward isn't
reusable for other tasks. PPO explores on-policy and takes gradient signal
from a live reward model, at ~4× the infrastructure.

### 3.4 GRPO: PPO reshaped for reasoning models

GRPO (DeepSeek) keeps PPO's clipped-ratio update but changes where the
advantage comes from. PPO's value network - a whole LM-sized critic - exists
only to estimate "how good is the average completion from here" as a
baseline. GRPO replaces it with an *empirical* baseline: sample a **group**
of $G$ completions $\{y_1..y_G\}$ for the same prompt, score each, and set

$$\hat A_i = \frac{r_i - \text{mean}(r_1..r_G)}{\text{std}(r_1..r_G)}$$

Each sample's advantage is its reward relative to its siblings. No value
model: no second policy-sized network to store and train, and no critic to
be wrong. Why this fits
reasoning models: (i) with **verifiable rewards** (did the math answer check
out? did tests pass?) the reward is exact but *sparse and binary*, which is
exactly where a learned value function struggles and where a group mean is a
clean baseline; (ii) sampling many attempts per prompt is natural for
reasoning tasks anyway; (iii) the normalization makes the signal "which of
these attempts beat the others", which is the actual quantity of interest.
This is the algorithm behind DeepSeek-R1-style RL on math/code.

### 3.5 Constitutional AI (one solid paragraph - Anthropic asks this)

Constitutional AI replaces most human preference labels with AI feedback
steered by an explicit list of principles (the "constitution"). Two phases.
*Supervised phase*: the model generates responses to (often adversarial)
prompts, then is prompted to **critique** its own response against a
constitutional principle ("identify ways this is harmful…") and **revise**
it; fine-tune on the revisions. *RL phase (RLAIF)*: generate response pairs,
ask an AI to judge which better satisfies the constitution, train a
preference model on those AI labels, then optimize the policy against it  - 
RLHF with the human replaced by a principle-guided AI. Why it matters: it
scales oversight (principles are cheap to apply to millions of comparisons;
human labels aren't), makes the normative target *explicit and auditable*
(you can read the constitution; you can't read a preference dataset's implicit
values), and reduces exposing human labelers to harmful content. Limitation:
quality is bounded by the model's ability to judge its own outputs against
the principles, and principles can conflict.

### 3.6 SFT vs RLHF vs DPO - when each

| | SFT | DPO | RLHF (PPO/GRPO) |
|---|---|---|---|
| data | demonstrations | preference pairs | preferences → RM, or verifiable reward |
| signal | "write this" | "prefer this to that" | scalar reward, on-policy |
| infra | 1 model | 2 models (policy + frozen ref), no sampling | 3–4 models + generation loop |
| ceiling | labeler quality | quality of the pair data | can exceed data via exploration |
| risks | can't say "don't" | overfits pairs, drives $\pi(y_l)\to 0$ | reward hacking, RL instability |

Practical recipe most labs converge on: SFT for format → DPO (or variants)
for cheap broad preference alignment → RL with verifiable rewards for
reasoning; PPO/GRPO where the extra ceiling is worth the infrastructure.

---

## 4. Fine-tuning

### 4.1 LoRA, derived

Full fine-tuning of a 7B model needs the full 16 bytes/param training state:
112 GB (§2.2). The empirical rescue: fine-tuning updates are **intrinsically
low-dimensional** - task adaptation lives in a tiny subspace of the weight
space (Aghajanyan et al. measured that ~200 free parameters, randomly
projected, suffice to fine-tune some tasks). So don't learn a full update
$\Delta W \in \mathbb{R}^{d\times k}$; constrain it to rank $r$:

$$W' = W + \Delta W = W + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)$$

Details that interviews probe:

- **Init**: $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$, so $\Delta W = 0$ at
  start - the adapted model begins exactly at the pretrained model, and
  training moves it smoothly away. (Both zero would kill all gradients to
  $A$: $\partial A \propto B^\top$.)
- **$\alpha/r$** is just a scale so the update magnitude is comparable across
  ranks; it folds into the LR.
- **No inference cost**: after training, *merge* $W \leftarrow W +
  \frac{\alpha}{r}BA$ - the served model has identical architecture and
  latency to the base model. Or keep adapters separate and hot-swap many
  tasks over one frozen base.

**Parameter count, worked for 7B** (Llama-7B: $d = 4096$, 32 layers). Adapt
$W_Q$ and $W_V$ (each $4096 \times 4096$) with $r = 16$:

- per matrix: $A$ is $16 \times 4096$ and $B$ is $4096 \times 16$ →
  $2 \times 65{,}536 = 131{,}072$ params
- per layer: 2 matrices → $262{,}144$;  ×32 layers → $\approx 8.4$M

$$\frac{8.4\text{M}}{6.7\text{B}} \approx 0.12\%$$

Trainable state: $8.4\text{M} \times 16 \approx 134$ MB instead of 107 GB  - 
~800× less optimizer+gradient memory. The frozen base still occupies 13–14 GB
in fp16 (forward passes need it), which QLoRA attacks next.

Why does rank 8–64 suffice when $W$ has rank 4096? Because *pretraining
already built the features*; fine-tuning mostly re-weights and re-routes
existing directions rather than learning new ones. When that's false - a truly
new domain, a new language, new low-level skills - LoRA underperforms full
fine-tuning, and that's the honest answer to "when does LoRA fail".

### 4.2 QLoRA: quantize the frozen part

LoRA's remaining cost is the fp16 frozen base. QLoRA stores it in **NF4**
(NormalFloat-4): 4 bits per weight, with 16 quantization levels placed at the
quantiles of a standard normal - optimal for weights that are approximately
normally distributed, which trained weights are. Blocks of 64 weights share
an absmax scale (plus a second-level quantization of the scales themselves  - 
"double quantization" - bringing the overhead to ~0.127 bits/weight).

Memory for 7B: base $7\times10^9 \times 0.5$ bytes $\approx 3.5$ GB (plus
$7\times10^9 \times 0.127$ bits $\approx 0.11$ GB of scales with double
quantization; ~0.44 GB without it), LoRA params + Adam state ~0.15 GB, activations a few GB → fine-tune
a 7B on a single 16–24 GB consumer GPU; the paper's headline was 65B on one
48 GB card. Gradients never flow *into* the 4-bit weights (they're frozen);
NF4 blocks are dequantized to bf16 on the fly per matmul, so compute precision
is unchanged - quantization error acts like small fixed noise on the base
weights, which the LoRA update can partially absorb.

### 4.3 Fine-tune vs RAG vs prompting

Decision logic, not a menu: **prompting/few-shot** first - if instructions
and examples in context solve it, ship that (zero training, instant
iteration). **RAG** when the failure mode is *missing or stale knowledge*:
retrieval injects facts at inference time, updates instantly, and provides
citations; fine-tuning is a terrible database (lossy, unverifiable, needs
retraining per update). **Fine-tuning** when the failure mode is *behavior*:
format, style, tone, domain jargon, tool-call syntax, latency (shorter
prompts), or distilling a big model's behavior into a small one. They
compose: RAG for facts + a fine-tune that teaches the model to *use*
retrieved context well. The interview trap is treating fine-tuning as a way
to add knowledge - it can, but slowly, expensively, and with hallucination
risk when facts half-stick.

---

## 5. Inference

### 5.1 Why decode is memory-bandwidth-bound: arithmetic intensity

A processor has peak compute $F$ (FLOP/s) and memory bandwidth $W$ (bytes/s).
A kernel with **arithmetic intensity** $I$ = FLOPs per byte moved is
compute-bound if $I > F/W$ (the "ridge point"), else bandwidth-bound. A100:
$F = 312$ TFLOP/s (bf16), $W = 2$ TB/s → ridge $\approx 156$ FLOPs/byte.

**Decode = GEMV.** Generating one token for one sequence multiplies a
*vector* by every weight matrix. Per matrix $d_{in}\times d_{out}$: read
$2\,d_{in}d_{out}$ bytes (fp16), do $2\,d_{in}d_{out}$ FLOPs →

$$I_{GEMV} \approx 1 \text{ FLOP/byte} \lll 156$$

The GPU spends its time streaming weights from HBM while tensor cores idle at
<1% utilization. **Prefill = GEMM**: processing a prompt of $s$ tokens reuses
each weight for all $s$ rows: $I \approx s$ FLOPs/byte - a 512-token prompt
is already compute-bound. This asymmetry - prefill compute-bound, decode
bandwidth-bound - explains almost every serving technique below: batching
(share the weight-read across sequences), quantization (shrink the bytes),
speculative decoding (verify many tokens per weight-read).

**Tokens/sec upper bound from bandwidth.** Each decoded token must stream
the whole model (plus that sequence's KV cache) once:

$$t/s \le \frac{W}{\text{bytes(model)} + \text{bytes(KV)}}$$

7B in fp16 on A100 (ignore KV): $2\times10^{12} / 14\times10^9 \approx$
**143 tok/s** for batch 1 - and real systems hit ~60–80% of this. Note what's
*not* in the formula: FLOPs. Quantize to int4 (3.5 GB) and the same GPU's
ceiling is ~570 tok/s - quantization at batch 1 is a *speed* optimization
before it is a memory one.

### 5.2 KV-cache memory, derived and worked

Causal attention at step $t$ needs $K, V$ for all previous tokens. Recomputing
them is $O(t)$ extra work per step; caching them is the standard trade of
memory for compute. Count the memory: per token, per layer, we store one K
and one V vector, each of dimension $n_{heads} \cdot d_{head} = d_{model}$:

$$\text{KV bytes} = \underbrace{2}_{K \text{ and } V} \times\; n_{layers} \times d_{model} \times \underbrace{2}_{\text{fp16}} \times\; s \times b$$

**The canonical question: 7B (32 layers, $d_{model}=4096$), batch 32, 4k
context.** Per token per layer: $2 \times 4096 \times 2 = 16$ KB. Per token
across 32 layers: $512$ KB. Full:

$$524{,}288 \times 4096 \times 32 = 68.7 \text{ GB}$$

Against 14 GB of weights: the cache is **5× the model**. Total 83 GB - does
not fit an 80 GB A100. This one calculation motivates a whole research area:

- **MQA/GQA**: share K,V across heads. GQA with $g$ KV heads instead of $h$
  cuts the cache by $h/g$ - Llama-2-70B uses $g=8$ vs $h=64$: 8× smaller
  cache for ~no quality loss. (Then $d_{model} \to g \cdot d_{head}$ in the
  formula.)
- **KV quantization** to int8/fp8: another 2×.
- **PagedAttention** (§5.5): stop wasting the memory you do have.

### 5.3 Quantization: the actual arithmetic

Map fp16 values $x$ to $b$-bit integers. **Absmax (symmetric)**: scale
$s = \max|x| / (2^{b-1}-1)$, store $q = \text{round}(x/s)$, dequantize
$\hat x = sq$. Tiny example, int8 on $x = [0.1, -0.2, 0.5, -1.0]$:
$s = 1.0/127 = 0.00787$; $q = [13, -25, 64, -127]$;
$\hat x = [0.1024, -0.1969, 0.5039, -1.0]$ - max error $\le s/2 \approx 0.004$.

**Zero-point (asymmetric)**: for skewed ranges (e.g. post-ReLU activations,
all $\ge 0$), $s = (\max - \min)/(2^b - 1)$, $z = \text{round}(-\min/s)$,
$q = \text{round}(x/s) + z$, $\hat x = s(q - z)$ - spends all $2^b$ levels on
the actual range instead of wasting half on a sign that never occurs.

**What breaks: outlier channels.** In LLMs past ~7B, a few hidden channels
carry activations 20–100× larger than the rest. One outlier in a tensor
blows up $\max|x|$, so $s$ becomes huge and *every other value* rounds to a
handful of levels - the outlier steals all the precision. Fixes: per-channel
or per-group (e.g. 128 weights) scales instead of per-tensor;
LLM.int8's mixed decomposition (compute the ~0.1% outlier columns in fp16,
the rest in int8).

**GPTQ** (intuition): quantization is not per-weight rounding-to-nearest in
isolation - what matters is the layer's *output* error
$\|XW - X\hat W\|^2$. GPTQ quantizes weights one column at a time and, using
second-order (Hessian $\approx X^\top X$) information, **updates the not-yet-
quantized columns to compensate** for the error just introduced. Rounding
errors get absorbed instead of accumulating; this is why 4-bit GPTQ ≈ fp16
quality while naive 4-bit rounding is noticeably worse.

**AWQ** (intuition): not all weights matter equally - the ones multiplying
large activations matter most. AWQ finds salient channels *from activation
statistics*, scales those weights up (and the activations correspondingly
down, keeping the product invariant) before quantizing, so the important
weights get finer effective resolution. No second-order solve; robust and
fast.

### 5.4 Speculative decoding - algorithm, proof, speedup

Decode is bandwidth-bound (§5.1): one weight-stream per token, tensor cores
idle. But *verifying* $k$ proposed tokens costs one forward pass over $k$
positions - a small GEMM, nearly the same wall time as one token. So: let a
small **draft** model propose, and the big **target** model verify $k$ tokens
per weight-stream. The trick is accepting drafts *without changing the output
distribution at all*.

**The algorithm** (target $p$, draft $q$; one step):

1. Draft autoregressively samples $k$ tokens $x_1..x_k$ from $q$.
2. Target runs **one** forward pass on all $k$ positions in parallel, giving
   $p(\cdot \mid \text{prefix}, x_{<i})$ for every $i$.
3. For each $i = 1..k$ in order: accept $x_i$ with probability
   $\min\!\big(1,\; p(x_i)/q(x_i)\big)$.
4. At the first rejection, sample a replacement from the **residual**
   distribution $p'(x) = \frac{\max(0,\, p(x) - q(x))}{\sum_y \max(0,\, p(y) - q(y))}$
   and stop.
5. If all $k$ accepted, sample one bonus token from the target's $(k{+}1)$-th
   distribution (free - we already computed it).

**Proof that the output is exactly $p$** (single position; induction gives
the sequence). The emitted token $X$ is either an accepted draft or a residual
sample. For any token $x$:

$$P(X = x) = \underbrace{q(x)\min\!\Big(1, \frac{p(x)}{q(x)}\Big)}_{\text{drafted } x \text{, accepted}} + \underbrace{P(\text{reject}) \cdot p'(x)}_{\text{rejected, resampled}}$$

First term: $q \min(1, p/q) = \min(q, p)$. For the rejection probability, sum
acceptance over all drafts:

$$P(\text{accept}) = \sum_y q(y)\min\!\Big(1,\tfrac{p(y)}{q(y)}\Big) = \sum_y \min(q(y), p(y))$$

$$P(\text{reject}) = 1 - \sum_y \min(p,q) = \sum_y \big(p(y) - \min(p(y),q(y))\big) = \sum_y \max(0, p(y)-q(y))$$

(using $\sum p = 1$ and $p - \min(p,q) = \max(0, p-q)$). But that sum is
exactly the residual's normalizer, so
$P(\text{reject}) \cdot p'(x) = \max(0, p(x) - q(x))$. Add the two terms:

$$P(X = x) = \min(p(x), q(x)) + \max(0,\, p(x) - q(x)) = p(x) \qquad \blacksquare$$

(the identity $\min(p,q) + (p-q)^+ = p$ holds case-by-case: if $p \le q$ it's
$p + 0$; if $p > q$ it's $q + (p - q)$). No approximation anywhere - the big
model's distribution, exactly, at a fraction of the weight-streams.
`speculative_decoding_demo.ipynb` verifies this empirically to TV distance ~0.

**Speedup.** Note $P(\text{accept}) = \sum_y \min(p, q) = 1 - \text{TV}(p,q)$ - acceptance rate *is* draft–target agreement. If per-position acceptance is
$a$ (i.i.d. approximation), expected tokens emitted per target pass =
$1 + a + a^2 + \dots + a^k = \frac{1 - a^{k+1}}{1 - a}$ (each extra token
requires all previous accepts; +1 because a rejection still emits the
residual token, and full acceptance emits the bonus). With $a = 0.8$,
$k = 4$: $(1 - 0.8^5)/0.2 \approx 3.36$ tokens per big-model pass - minus the
draft's own cost, a ~2–3× wall-clock speedup, output distribution untouched.

### 5.5 Continuous batching and PagedAttention

Batching fixes decode's bandwidth problem - one weight-stream serves $B$
sequences. Two systems problems remain.

**Static batching wastes compute on length variance.** If you batch 32
requests and run them to completion, the batch runs until its *longest*
output finishes; a request that finished at token 50 occupies a slot doing
nothing while a neighbor generates 2000 tokens, and new requests queue.
**Continuous batching** schedules at token granularity: at every decode step,
finished sequences leave, queued sequences join (their prefill can be chunked
in with decodes). Throughput gains of 2–10× purely from not computing on
finished slots.

**Contiguous KV allocation wastes memory.** Pre-vLLM systems allocated each
sequence's KV cache as one contiguous buffer sized to max output length.
Derive the waste: reserve 2048 tokens, generate 300 → 85% of the reservation
is dead (**internal fragmentation**); freed buffers of varying sizes leave
unusable gaps between live ones (**external fragmentation**). Measured waste:
60–80% of KV memory. Batch size is capped by KV memory (§5.2), so wasted KV
memory is wasted throughput.

**PagedAttention** = virtual memory for the KV cache. Chop KV storage into
fixed-size **blocks** (e.g. 16 tokens); a per-sequence **block table** maps
logical positions to physical blocks scattered anywhere in memory; allocate
blocks on demand as the sequence grows. Internal fragmentation shrinks to
<1 block per sequence; external fragmentation vanishes (all blocks are the
same size). Bonus, exactly like OS page sharing: sequences with a common
prefix (system prompt, beam candidates, n-best sampling) share physical
blocks with copy-on-write. Net effect: 2–4× more concurrent sequences in the
same memory → 2–4× throughput.

### 5.6 FlashAttention: exact attention, IO-optimally

Standard attention materializes $S = QK^\top \in \mathbb{R}^{s \times s}$ and
$P = \text{softmax}(S)$ in HBM: $O(s^2)$ memory, and - the real killer  - 
$O(s^2)$ HBM reads/writes for cheap elementwise ops. Attention at long $s$ is
bandwidth-bound not FLOP-bound.

The obstacle to tiling it like a matmul is softmax: it normalizes over a
**whole row**, which you never hold at once. The fix is **online softmax**  - 
softmax computed in one pass with running statistics. Keep, per query row: a
running max $m$, a running normalizer $\ell$, and a running output $o$. On
seeing a new block of scores $s^{(j)}$ with values $v^{(j)}$:

$$m' = \max(m, \max s^{(j)}), \qquad \ell' = \ell\, e^{m - m'} + \sum_i e^{s^{(j)}_i - m'}$$
$$o' = o\, \frac{\ell\, e^{m - m'}}{\ell'} + \frac{\sum_i e^{s^{(j)}_i - m'} v^{(j)}_i}{\ell'}$$

The factors $e^{m - m'}$ *retroactively rescale* everything computed under
the old max - algebraically identical to having known the true max from the
start. So: load a tile of $Q$ and stream tiles of $K, V$ through fast on-chip
SRAM (~20 MB, ~19 TB/s on A100), maintain $(m, \ell, o)$ per row, never write
$S$ or $P$ to HBM. It is **exact** - same output bit-for-bit-ish as naive
attention, this is a memory-movement reorganization, not an approximation.
HBM traffic drops from $\Theta(sd + s^2)$ to $\Theta(s^2 d^2 / M)$ ($M$ = SRAM size,
which is much larger than $d$) - in
practice 5–10× fewer HBM accesses, 2–4× wall-clock, and $O(s)$ memory
(backward recomputes $S$ tile-by-tile from the saved $(m, \ell)$ - gradient
checkpointing applied inside attention).

---

## 6. LLM evaluation

### 6.1 Benchmarks: what they measure and what they don't

**MMLU** (57 subjects, 4-way multiple choice, ~14k questions) measures
breadth of knowledge + light reasoning. Multiple-choice mechanics matter:
score by comparing the model's likelihood of "A/B/C/D" or generate-and-parse;
the two disagree, and formatting/prompt changes move scores by points  - 
always ask "under which harness?". **GSM8K/MATH**: multi-step math, checkable
final answers. **HumanEval/SWE-bench**: code, verified by tests - execution
makes these the most trustworthy class. **Chatbot Arena**: humans vote on
paired anonymous responses → Elo; measures preference, which bundles
helpfulness with style and sycophancy. The meta-point: a benchmark is a
*proxy* with a shelf life - once it's a target, models overfit to its
distribution (Goodhart, §6.5), and saturated benchmarks (many are >90%)
stop discriminating.

### 6.2 Contamination

Pretraining corpora are web-scale; benchmark test sets leak into them, and a
model that memorized the test isn't measuring generalization. Detection:

- **N-gram overlap**: flag test examples sharing long n-grams (e.g. any
  13-gram) with training documents. Cheap, catches verbatim copies, misses
  paraphrases and translations.
- **Canary strings**: benchmarks (BIG-bench pioneered this) embed a unique
  GUID in their files; ask the model to complete the canary - reproduction
  proves the files were trained on. Proves ingestion, not the converse.
- **Behavioral probes**: can the model complete a test question verbatim
  given its prefix? Does accuracy drop on freshly-written items of equal
  difficulty (the strongest evidence - e.g. GSM1k-style rebuilds, where some
  models dropped double digits)? Is performance suspiciously better on the
  benchmark's exact phrasing than on paraphrases?

Ounce of prevention: decontaminate training data against known test sets,
and keep private held-out sets that never touch the internet.

### 6.3 LLM-as-judge

Human evaluation is the gold standard and doesn't scale; using a strong LLM
to grade outputs is now standard (MT-Bench, Arena-Hard, most internal evals).
Known biases, each with a mitigation:

- **Position bias**: judges favor the first (sometimes second) response in a
  pairwise prompt. → Judge both orders; count only consistent verdicts, or
  average.
- **Verbosity bias**: longer answers score higher independent of quality. →
  Length-controlled scoring (regress out length, as in length-controlled
  AlpacaEval), or rubrics that explicitly reward concision.
- **Self-preference**: models rate their own family's style higher. → Use a
  judge from a different family, or a panel of judges.
- **Sycophancy toward confidence / style over substance**: confident fluent
  wrong answers beat hesitant right ones. → Reference-guided judging (give
  the judge a gold answer to compare against), decompose into checkable
  claims.

Always validate the judge itself: measure its agreement with human labels on
a subset before trusting it at scale - treat the judge as a *model of the
metric* with its own eval.

### 6.4 Hallucination

Taxonomy: **factuality** errors (output contradicts world knowledge  - 
invented citations, wrong dates) vs **faithfulness** errors (output
contradicts the provided source/context - a summary asserting what the
document doesn't say). Both subdivide into intrinsic (contradicts) vs
extrinsic (unsupported-but-unverifiable).

Why next-token training *causes* it: the objective rewards
$P(\text{plausible continuation})$, not truth - a fluent made-up citation
has high likelihood under the training distribution of academic prose,
because the training signal never distinguishes "true" from
"looks like text that occurs". The model is trained to always continue,
never to abstain: "I don't know" is rare in web text precisely where
confident prose is common. Post-training can worsen calibration: pretrained
models' token probabilities are surprisingly well calibrated, and RLHF  - 
optimizing for answers *raters prefer*, and raters prefer confidence - trades
that calibration away.

Mitigations, mapped to causes: **RAG** (ground generation in retrieved
sources; shifts the problem to faithfulness, which is easier to check);
**abstention training** (SFT/RL that rewards calibrated "I don't know"  - 
including the elegant trick of asking the model things it can't know and
rewarding refusal); **calibration/verification** (sample multiple times and
check self-consistency - hallucinations vary across samples, knowledge
doesn't; verify claims against sources post-hoc); **decoding** interventions
(lower temperature reduces but doesn't eliminate). None solves it; a model
that never asserts beyond its knowledge would also rarely be useful - it's a
precision/recall trade on assertions.

### 6.5 Why evals are the hard part of the job

Goodhart's law: *when a measure becomes a target, it ceases to be a good
measure*. Every layer of this module optimizes a proxy - the RM is a proxy
for human preference (PPO exploits it, §3.2), benchmarks are proxies for
capability (labs overfit them, §6.1), the judge is a proxy for human
judgment (§6.3), and loss is a proxy for everything (a model can match loss
and differ wildly in behavior). Optimization pressure finds exactly the gap
between proxy and target - the stronger the optimizer, the faster. So the
job is not "run the benchmark"; it's maintaining a measurement that survives
being optimized against: fresh held-out sets, multiple uncorrelated metrics,
human spot-audits of the judge, red-teaming the metric itself before trusting
gains on it. Common lab interview question: "your new RLHF run improved the
reward by 30% - what do you check before believing it?" Answer: KL from
reference, human eval on fresh prompts, length/style drift, per-category
breakdowns - i.e., you check the *proxy gap*.

---

## Common interview questions

**Q1. You have $10^{23}$ FLOPs of compute. What model do you train?**
Chinchilla: $D \approx 20N$ and $C = 6ND = 120N^2$, so
$N = \sqrt{10^{23}/120} \approx 29$B params, $D = 20N \approx 580$B tokens.
Then caveat: if this model will serve heavy traffic, overtrain a smaller
model instead (Llama-3 style) - Chinchilla optimizes training loss per
training FLOP, not lifetime cost.

**Q2. Derive the 6ND rule.**
Forward: each param does one multiply-add per token = 2 FLOPs → $2N$/token.
Backward: two matmuls per layer matmul ($\partial X = \partial Y W^\top$,
$\partial W = X^\top \partial Y$) → $4N$/token. Total $6N$ per token × $D$
tokens.

**Q3. How much memory to train a 70B model with Adam in mixed precision?**
Per param: fp16 weights (2) + fp16 grads (2) + fp32 master (4) + Adam $m$ (4)
+ $v$ (4) = 16 B. $16 \times 70\text{e}9 = 1.12$ TB of state, plus
activations (hundreds of GB at long context without checkpointing). ≥14 H100s
just to hold state, and only with ZeRO-3/FSDP sharding.

**Q4. Why fp32 master weights at all?**
The update $\eta g$ is often smaller than the fp16 ulp near the weight's
magnitude (fp16 has ~11 bits of mantissa; near 1.0 increments below $2^{-11}$
vanish; with typical $\eta g \sim 10^{-6}$ the update rounds to zero).
Accumulate in fp32, round to fp16 for the matmuls.

**Q5. ZeRO stages - what does each shard, and per-GPU memory for 70B on 64
GPUs?**
Stage 1: optimizer state (12B/param) → $4\Psi + 12\Psi/G$ = 293 GB. Stage 2:
+gradients → $2\Psi + 14\Psi/G$ = 155 GB. Stage 3: +weights → $16\Psi/G$ =
17.5 GB. Only stage 3 fits an 80 GB GPU; it costs ~1.5× DP communication
(extra weight all-gather in backward). FSDP ≈ ZeRO-3.

**Q6. Where do the all-reduces land in tensor parallelism?**
Column-split first matmul (no comm, nonlinearity is elementwise-local), row-
split second matmul (each GPU holds a partial sum) → one all-reduce per MLP
block forward, one per attention block (heads = column split, $W_O$ = row
split), mirrored in backward. Frequent + unhideable → TP stays within a
node's NVLink.

**Q7. Pipeline bubble fraction?**
$(p-1)/(m+p-1)$ for $p$ stages, $m$ microbatches: the pipe takes $m+p-1$
slots, $m$ useful. $p=8, m=32$: 18%. Increase $m$ to amortize; 1F1B schedules
cap live activation memory at ~$p$ microbatches.

**Q8. How does gradient checkpointing get $O(\sqrt L)$ memory?**
Store $k$ checkpoints, recompute segments of $L/k$ layers during backward:
peak $\propto k + L/k$, minimized at $k = \sqrt L$ → $2\sqrt L$. Cost: one
extra forward ≈ +33% FLOPs (forward is $2N$ of $6N$).

**Q9. Batch 256 → 2048. What do you do to the LR?**
Linear scaling: 8× (one big-batch step ≈ 8 small steps when gradients are
locally stable), with warmup because that assumption fails early in
training; watch for the critical batch size beyond which returns vanish; for
Adam, $\sqrt{8}$× is often better.

**Q10. Derive DPO from the RLHF objective.**
KL-regularized reward maximization has closed form
$\pi^* = \frac{1}{Z}\pi_{ref} e^{r/\beta}$ (complete the KL). Invert:
$r = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z$. Substitute into
Bradley–Terry $\sigma(r_w - r_l)$: $Z$ cancels →
$\mathcal{L} = -\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} -
\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$. RM + RL collapse into
supervised learning on preferences.

**Q11. Why the KL term in PPO-RLHF? Why the clip?**
KL-to-reference: the RM is only valid near the SFT distribution; unpenalized
optimization finds adversarial text the RM loves and humans hate (reward
hacking) and collapses diversity. Clip: updates use stale samples from
$\pi_{old}$; the min-of-clipped objective zeroes the gradient for tokens whose
ratio has already moved past $1 \pm \epsilon$ in the direction their advantage
favors (moves in the other direction are not clipped), bounding each update (trust-region-lite) so
one bad batch can't crater the policy.

**Q12. GRPO vs PPO?**
GRPO deletes the value network: sample $G$ completions per prompt, advantage
= (reward − group mean)/group std. Fits verifiable-reward reasoning RL:
sparse exact rewards are hard for critics to fit but trivial to baseline
with a group mean; saves an LM-sized model; the relative signal ("which
attempt won") is what you actually want.

**Q13. LoRA: parameter math for adapting $W_Q, W_V$ of a 7B, rank 16?**
$d = 4096$: per matrix $2 \times 4096 \times 16 = 131$k; ×2 matrices ×32
layers ≈ 8.4M params ≈ 0.12% of 7B. Init $B=0$ so training starts at the
pretrained model; merge $W + \frac{\alpha}{r}BA$ after training for zero
inference overhead.

**Q14. Why is single-stream decode memory-bound? Ceiling for 7B fp16 on
2 TB/s?**
Decode is GEMV: 2 FLOPs per 2 bytes of weight read → intensity ~1 FLOP/B vs
ridge ~156 (A100). Every token streams all weights:
$2\text{e}12 / 14\text{e}9 \approx 143$ tok/s max. Batching/speculation raise
arithmetic intensity; quantization shrinks the bytes.

**Q15. KV cache for a 7B at batch 32, 4k context?**
$2_{KV} \times 32_{layers} \times 4096_{d} \times 2_{fp16} = 512$ KB/token;
$\times 4096 \times 32 = 68.7$ GB - 5× the 14 GB weights; doesn't fit an
A100 with the model. Hence GQA (÷8), KV quantization (÷2), PagedAttention
(reclaim the 60–80% fragmentation waste).

**Q16. Why does speculative decoding not change the output distribution?**
Accept draft $x \sim q$ w.p. $\min(1, p(x)/q(x))$, else resample from
$(p-q)^+$ normalized. $P(x) = \min(p,q) + \max(0, p-q) = p(x)$ exactly
(rejection mass equals the residual normalizer). Acceptance rate
$= 1 - \text{TV}(p, q)$; expected tokens/pass $= (1-a^{k+1})/(1-a)$.

**Q17. Is FlashAttention approximate?**
No - exact. It's an IO reorganization: tile $K,V$ through SRAM with online
softmax (running max $m$, normalizer $\ell$, rescale by $e^{m-m'}$), never
materializing the $s \times s$ matrix in HBM. Attention is bandwidth-bound;
cutting HBM traffic ~10× gives 2–4× speed and $O(s)$ memory.

**Q18. What breaks in naive int8 LLM quantization?**
Outlier channels: a few activations 20–100× larger dominate the absmax
scale, crushing everything else's resolution. Fix with per-channel/group
scales, mixed-precision decomposition (LLM.int8), or scale migration onto
weights (AWQ/SmoothQuant-style).

**Q19. How would you detect benchmark contamination?**
N-gram (13-gram) overlap between test set and training corpus; canary-GUID
completion; verbatim-completion probes of test items; accuracy drop on
newly-authored items of matched difficulty (strongest); paraphrase
sensitivity. Prevent: decontamination filters + private held-out sets.

**Q20. Your new model improved the LLM-judge win-rate by 10 points. What
do you check before believing it?**
Judge biases: position (swap orders), verbosity (length-controlled score),
self-preference (different-family judge). Then Goodhart checks: human
agreement on a sample, fresh prompts the judge/model haven't seen, per-
category breakdown, KL/style drift from the base model. A 10-point win-rate
gain with a +40% length increase is a length gain, not a quality gain.

## Exit checklist

- [ ] Cold, on paper: derive C ≈ 6ND, then apply the Chinchilla 20-tokens-per-param rule to a worked budget - 5 min
- [ ] Walk the full training-memory budget for a 70B model in mixed precision with Adam (16 bytes/param + activations) - 4 min
- [ ] Explain ZeRO stages 1-3: what each shards, memory saved, and the communication cost - 4 min
- [ ] Derive DPO from the RLHF objective (KL-constrained reward max → closed-form policy → Bradley-Terry substitution) - 6 min
- [ ] Show via arithmetic intensity why decode is memory-bandwidth-bound while prefill is compute-bound - 4 min
- [ ] Compute KV-cache memory for a given model/batch/context, and state what MQA/GQA save - 4 min
- [ ] Explain speculative decoding: the algorithm, why outputs are exactly the target model's, expected speedup - 4 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
