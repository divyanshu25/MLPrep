# LLM Stack Exercises

Twelve exercises, easy → hard, mirroring questions reported from frontier-lab
interviews. Do the arithmetic on paper: every one of these is meant to be
solvable at a whiteboard. Notation follows `notes.md`; `memory_math.ipynb` checks
several answers numerically.

---

## Exercise 1 (easy): Chinchilla-optimal model for a fixed budget

Your budget is $C = 10^{23}$ FLOPs. Using $C \approx 6ND$ and the
20-tokens-per-parameter rule, find the compute-optimal $N$ and $D$. Then:
GPT-3 was 175B params on 300B tokens; how far off optimal was it for *its*
budget?

<details><summary>Solution</summary>

With $D = 20N$: $C = 6N \cdot 20N = 120N^2$, so

$$N = \sqrt{C/120} = \sqrt{8.3\times10^{20}} \approx 2.9\times10^{10} \approx 29\text{B}, \qquad D = 20N \approx 580\text{B tokens.}$$

Check: $6 \times 2.9\text{e}10 \times 5.8\text{e}11 \approx 1.0\times10^{23}$. ✓

GPT-3: $C = 6 \times 1.75\text{e}11 \times 3\text{e}11 = 3.15\times10^{23}$
FLOPs. Optimal for that budget: $N = \sqrt{3.15\text{e}23/120} \approx 51$B,
$D \approx 1.0$T. So GPT-3 was ~3.4× too many parameters and ~3.4× too few
tokens. A compute-optimal ~50B model trained on ~1T tokens would have
reached lower loss for the same cost. (This is precisely the Chinchilla
paper's argument.)
</details>

---

## Exercise 2 (easy): Derive $C \approx 6ND$

Show where the 6 comes from, term by term. Which parts of a transformer does
the estimate ignore, and when is that a bad approximation?

<details><summary>Solution</summary>

Forward: each weight in a matmul does one multiply and one add per token
→ $2N$ FLOPs/token. Backward: each forward matmul $Y = XW$ needs
$\partial X = \partial Y W^\top$ and $\partial W = X^\top \partial Y$, two
matmuls of the same size → $4N$ FLOPs/token. Total $6N$/token, × $D$ tokens.

Ignored: attention's $QK^\top$ and $AV$ (about $4 s\, d_{model}$ FLOPs per
token per layer, no *parameters* involved, against about $24 d_{model}^2$ for
the layer's weights). The ratio is about $s/(6 d_{model})$, independent of the
number of layers, so the approximation breaks down once $s$ approaches a few
times $d_{model}$, i.e. very long contexts; embedding/unembedding
lookups; and it assumes no recomputation (gradient checkpointing adds ~$2N$
for the extra forward → $8ND$).
</details>

---

## Exercise 3 (easy): Tokens/sec ceiling for a 13B model

A 13B-parameter model in fp16 runs batch-1 decode on one H100
(3.35 TB/s HBM). (a) Upper-bound tokens/sec, ignoring KV cache. (b) Redo
with an int4-quantized model. (c) Why doesn't the H100's 1000 TFLOPs figure
appear anywhere in your answer?

<details><summary>Solution</summary>

(a) Each token must stream all weights once:
$3.35\text{e}12 / (13\text{e}9 \times 2) = 3.35\text{e}12/2.6\text{e}10
\approx 129$ tok/s.

(b) int4 → 6.5 GB: $3.35\text{e}12/6.5\text{e}9 \approx 515$ tok/s. Same
FLOPs, 4× fewer bytes, ~4× the ceiling: at batch 1, quantization is a
throughput optimization.

(c) Because batch-1 decode is a GEMV with arithmetic intensity ~1 FLOP/byte,
while the H100's compute/bandwidth ridge is ~300 FLOPs/byte. Compute is idle;
only bandwidth binds. FLOPs enter only once batch size (or speculative
verification width) pushes intensity toward the ridge.
</details>

---

## Exercise 4 (medium): The full 70B training-memory walk

Walk through, from scratch: memory to train a 70B model, mixed precision +
Adam. Give (a) bytes/param with each component justified, (b) total state,
(c) per-GPU state for plain DP and each ZeRO stage at $G = 64$, (d) which
stages fit an 80 GB GPU, and (e) what ZeRO-3 pays for the fit.

<details><summary>Solution</summary>

(a) fp16 weights 2 + fp16 grads 2 + fp32 master 4 + Adam $m$ 4 + Adam $v$ 4
= **16 B/param**. Master weights exist because an update $\eta g$ smaller than
half the spacing between adjacent fp16 values ($2^{-10}$ near 1) is rounded
away; $m, v$ are Adam's two moments; all fp32 for accumulation accuracy.

(b) $16 \times 70\text{e}9 = 1.12$ TB (weights 140, grads 140, optimizer 840
GB).

(c) Per GPU: DP $= 1120$ GB; ZeRO-1 $= 4\Psi + 12\Psi/64 = 280 + 13.1 = 293$
GB; ZeRO-2 $= 2\Psi + 14\Psi/64 = 140 + 15.3 = 155$ GB; ZeRO-3
$= 16\Psi/64 = 17.5$ GB.

(d) Only ZeRO-3 (or adding tensor/pipeline parallelism). Stages 1–2 still
replicate 140–280 GB of weights(+grads) per GPU.

(e) ~1.5× DP communication volume: weights must be all-gathered layer-by-
layer in forward *and again* in backward (they were freed), on top of the
gradient reduce-scatter. Plus activations (~5–11 GB with checkpointing at
seq 4096) and runtime buffers. Run `memory_math.ipynb` to check the numbers.
</details>

---

## Exercise 5 (medium): KV cache: GQA vs MHA, and the batch-size ceiling

A 70B model: 80 layers, $d_{model} = 8192$, 64 heads ($d_{head} = 128$),
fp16 cache, 4096 context. (a) KV cache per sequence with full MHA. (b) With
GQA using 8 KV heads. (c) On one 80 GB GPU holding int4 weights (35 GB),
what max batch does each allow?

<details><summary>Solution</summary>

(a) MHA per token: $2 \times 80 \times (64 \cdot 128) \times 2 =
2 \times 80 \times 8192 \times 2 = 2.62$ MB. Per 4096-token sequence:
$\approx 10.7$ GB. (That is about ten gigabytes of cache per sequence.)

(b) GQA-8 stores $8 \cdot 128 = 1024$ dims instead of 8192: per token
$2 \times 80 \times 1024 \times 2 = 328$ KB → per sequence $\approx 1.34$ GB, 8× less ($h/g = 64/8$).

(c) Free memory $\approx 80 - 35 - (\text{few GB overhead}) \approx 40$ GB.
MHA: $40/10.7 \approx 3$ sequences. GQA: $40/1.34 \approx 29$ sequences.
Since decode throughput scales ~linearly with batch until compute-bound,
GQA here gives ~10× serving throughput, which is why most modern large models
(Llama-2/3-70B, Mistral, etc.) use GQA.
</details>

---

## Exercise 6 (medium): Derive the DPO gradient and interpret it

Starting from
$\mathcal{L} = -\log\sigma(\beta[\Delta_w - \Delta_l])$ where
$\Delta_y = \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$, derive
$\nabla_\theta \mathcal{L}$ and interpret each factor. What happens to the
gradient when the model already strongly prefers $y_w$? When it strongly
prefers $y_l$?

<details><summary>Solution</summary>

Let $u = \beta(\Delta_w - \Delta_l)$. Using $\frac{d}{du}\log\sigma(u) =
1 - \sigma(u) = \sigma(-u)$:

$$\nabla_\theta \mathcal{L} = -\,\sigma(-u)\,\beta\big[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\big]$$

(the $\pi_{ref}$ terms are constants in $\theta$ and vanish). Read it as
three factors:

1. $\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l)$: push up the
   chosen response's log-prob, push down the rejected one's, like paired
   SFT with a negative example.
2. $\sigma(-u) = 1 - P_\theta(y_w \succ y_l)$ under the implicit reward: the
   **example weight is the model's current error** on this pair. Already
   correct and confident ($u \gg 0$): weight $\to 0$, so the pair adds almost
   no gradient.
   Confidently *wrong* ($u \ll 0$): weight $\to 1$, maximal update. DPO is
   an implicitly hard-example-weighted objective.
3. $\beta$ scales both the gradient and the margin $u$. A high $\beta$ (a
   stronger KL penalty in the underlying RLHF objective) makes a smaller
   log-ratio gap $\Delta_w - \Delta_l$ already give a large $u$, so the
   weight $\sigma(-u)$ falls toward 0 sooner and the policy moves less far
   from $\pi_{ref}$.

Failure mode visible in the gradient: nothing limits *how* $u$ grows. The
model can raise $u$ mostly by driving $\pi_\theta(y_l) \to 0$, which can also
lower $\pi_\theta(y_w)$ and move probability mass to responses outside the
data (why DPO can need early stopping / regularization).
</details>

---

## Exercise 7 (medium): Prove speculative sampling is unbiased, vocab of 2

Vocab $\{a, b\}$, target $p = (0.9, 0.1)$, draft $q = (0.5, 0.5)$. Execute
the accept/reject rule fully by hand (enumerate every path and its
probability) and show the emitted token is distributed exactly as $p$. Also
compute the acceptance rate two ways.

<details><summary>Solution</summary>

Rule: draft $x \sim q$; accept w.p. $\min(1, p(x)/q(x))$; else sample the
normalized residual $(p - q)^+$.

Residual: $(p-q)^+ = (0.4, 0)$, normalized $p' = (1, 0)$: a rejection
always emits $a$.

Paths:
- draft $a$ (prob 0.5): accept w.p. $\min(1, 0.9/0.5) = 1$ → emit $a$.
  Contributes $0.5$ to $a$.
- draft $b$ (prob 0.5): accept w.p. $\min(1, 0.1/0.5) = 0.2$ → emit $b$
  w.p. $0.5 \times 0.2 = 0.1$; reject w.p. $0.5 \times 0.8 = 0.4$ → emit
  $a$ (residual). Contributes $0.1$ to $b$, $0.4$ to $a$.

Totals: $P(a) = 0.5 + 0.4 = 0.9$, $P(b) = 0.1$, exactly $p$. ∎

Acceptance rate: by paths, $0.5 \cdot 1 + 0.5 \cdot 0.2 = 0.6$; by formula,
$\sum \min(p, q) = \min(.9,.5) + \min(.1,.5) = 0.5 + 0.1 = 0.6 =
1 - \text{TV}(p,q)$ (TV $= 0.4$). ✓ Both match.
</details>

---

## Exercise 8 (medium): ZeRO comparison table from memory

Fill in, for $\Psi$ params, $G$ GPUs, mixed-precision Adam: what each ZeRO
stage shards, its per-GPU memory formula, its communication volume relative
to plain DP, and one gotcha per stage. Then explain in one sentence why FSDP
"is" ZeRO-3.

<details><summary>Solution</summary>

| stage | shards | per-GPU bytes | comm vs DP | gotcha |
|---|---|---|---|---|
| DP | - | $16\Psi$ | 1× (all-reduce grads) | replicates everything |
| 1 | optimizer (12B/p) | $4\Psi + 12\Psi/G$ | 1× (reduce-scatter + all-gather = all-reduce) | weights/grads still full |
| 2 | + grads (2B/p) | $2\Psi + 14\Psi/G$ | 1× | grad accumulation across microbatches needs care (partial grads live sharded) |
| 3 | + weights (2B/p) | $16\Psi/G$ | ~1.5× (weights all-gathered in fwd AND bwd) | latency-sensitive: params fetched layer-by-layer; needs prefetch overlap |

FSDP is ZeRO-3 because it does exactly the stage-3 mechanics natively in
PyTorch: flatten + shard parameters, all-gather a unit's weights just before
its forward/backward, free them after, reduce-scatter gradients to their
owner shard.
</details>

---

## Exercise 9 (medium): Gradient checkpointing arithmetic

A 48-layer model stores $M_1$ bytes of activations per layer. (a) Derive the
optimal number of checkpoint segments and the peak activation memory. (b) How
much extra compute does it cost, as a fraction of a $6ND$ training step?
(c) Your teammate proposes checkpointing every layer instead; compare both
memory and compute against the $\sqrt{L}$ scheme.

<details><summary>Solution</summary>

(a) With $k$ segments: keep $k$ boundary activations + recompute one segment
($L/k$ layers) at a time: $M(k) \propto k + L/k$, minimized at
$k = \sqrt{L} = \sqrt{48} \approx 7$ → peak $\approx 2\sqrt{48} \approx 14$
layer-activations $\approx 14 M_1$, vs $48 M_1$ naive, about a 3.5× saving.

(b) Every layer's forward is recomputed once → one extra forward $= 2ND$
extra on top of $6ND$ → **+33%**.

(c) Checkpointing every layer stores one *boundary* tensor per layer
($\approx 2sbh$ bytes: the layer input, a small part of $M_1$, not the
~34× internal tensors): memory $\propto L$ but with a tiny constant, and
compute is the same +1 forward. For transformers this is usually *better*
than the textbook $\sqrt L$ scheme because $M_1$'s internals (the 34×) dwarf
the boundaries. The $\sqrt L$ result is the general answer for a chain of
layers: derive it, then state this caveat.
</details>

---

## Exercise 10 (hard): Where do the all-reduces land? (tensor parallelism)

Two-GPU tensor parallelism for one transformer block: MLP
$Z = \sigma(XW_1)W_2$ and attention with 16 heads. (a) Say exactly how to
split $W_1, W_2, W_{QKV}, W_O$ and why the nonlinearity forces the order.
(b) Count communication ops per layer per forward and backward. (c) Estimate
per-layer comm bytes for $b \cdot s = 8192$ tokens, $h = 8192$, fp16, and
explain why TP degree rarely exceeds 8.

<details><summary>Solution</summary>

(a) $W_1$: **column**-split → each GPU gets $XW_1^{(i)}$, half the hidden
units. $\sigma$ is elementwise, so it applies locally *only* because the
split is by columns (each hidden unit is wholly on one GPU). Splitting
$W_1$ by rows would give partial sums that must be added *before* the
nonlinearity ($\sigma(u+v) \ne \sigma(u) + \sigma(v)$), forcing an extra
all-reduce. $W_2$: **row**-split so it consumes the local halves, producing
partial outputs → one all-reduce. Attention: heads 1–8 on GPU0, 9–16 on GPU1
(column split of $W_{QKV}$; heads are independent), $W_O$ row-split → one
all-reduce.

(b) 2 all-reduces forward (one per sub-block), 2 backward (mirrored, on the
input gradients).

(c) A ring all-reduce over $G$ GPUs moves $2(G-1)/G$ times the tensor size
per GPU, which is $1\times$ for $G = 2$: tensor
$= 8192 \times 8192 \times 2$ B $= 128$ MB → 128 MB per all-reduce, ~0.5 GB
per layer per step (4 ops), × 80 layers ≈ 40 GB per step,
every step, and poorly overlappable because the next op *needs* the result.
Feasible on 900 GB/s NVLink within a node; ruinous over ~50 GB/s inter-node
links. Hence TP ≤ 8 (node size), with PP/DP across nodes.
</details>

---

## Exercise 11 (hard): When does LoRA fail?

For each scenario, predict whether LoRA (r = 8–64 on attention projections)
matches full fine-tuning, and justify via the low-intrinsic-rank argument:
(a) instruction-tuning a pretrained 7B on English Q&A format; (b) adapting an
English-only model to Japanese; (c) teaching a chat model a new proprietary
API's call syntax; (d) continued pretraining on 50B tokens of legal text;
(e) RLHF-style preference tuning. What are the practical knobs when LoRA
underperforms?

<details><summary>Solution</summary>

The principle: LoRA constrains the update to rank $r$ per matrix: enough
when adaptation *re-weights and re-routes existing features*, insufficient
when the task needs *new features* distributed across many directions.

(a) **Matches.** Format/style/persona changes are the standard example of
a low-rank shift.
(b) **Fails / large gap.** A new language needs new token statistics,
morphology, even tokenizer coverage: high-rank, many-layer changes (and
embedding-table changes LoRA on attention doesn't touch).
(c) **Matches**, mostly: syntax imitation is behavioral; add the MLP
matrices and the embeddings if new tokens are involved.
(d) **Fails.** 50B tokens of domain shift is pretraining, not adaptation:
the required update is spread over many directions, and rank 64 cannot
represent it.
(e) **Usually matches**: preference tuning makes small changes to the
policy (DPO with LoRA is common and effective).

Knobs when it underperforms: raise $r$; apply LoRA to *all* linear layers
(MLP included; empirically this matters more than raising $r$); train embeddings
/ layernorms; or concede and full fine-tune (possibly with ZeRO offload).
</details>

---

## Exercise 12 (hard): Design a serving stack, with numbers

You must serve a 70B GQA model (80 layers, $d = 8192$, 64 Q heads, 8 KV
heads) at ~50 concurrent requests, 4k mean context, on 8×H100-80GB (aggregate
HBM 640 GB, per-GPU BW 3.35 TB/s). Choose parallelism, precision, batching,
and two latency optimizations. Justify each with arithmetic.

<details><summary>Solution</summary>

A defensible design (numbers are the point, not the exact choices):

1. **Weights**: fp8/int8 → 70 GB (fp16's 140 GB works too, but leaves less
   room for KV). **TP = 8** across NVLink: ~9 GB weights/GPU, and per-token weight
   streaming parallelizes across 8 × 3.35 TB/s.
2. **KV cache**: GQA-8 per 4k sequence: $2 \times 80 \times (8 \cdot 128)
   \times 2 \times 4096 \approx 1.34$ GB; 50 sequences ≈ 67 GB fp16 (33 GB
   fp8), which fits easily in the ~500 GB left. The spare memory allows
   larger batches, e.g. 100+ concurrent.
3. **Continuous batching + PagedAttention** (vLLM-style): token-level
   scheduling so finished requests don't hold slots; paged KV (16-token
   blocks) reclaims the 60–80% fragmentation waste → the 50-request target
   is sustained rather than peak.
4. **Latency opt 1: speculative decoding**: a ~1B draft (or self-drafting
   heads); at acceptance $a \approx 0.8$, $k = 4$: ~3.4 tokens per 70B pass
   → ~2.5× lower per-token latency at low batch (its benefit shrinks at high
   batch, where the GPU is already closer to compute-bound; say this).
5. **Latency opt 2: chunked prefill** (+ FlashAttention): split long
   prompt prefills into chunks interleaved with decode steps so a 4k-token
   compute-bound prefill doesn't stall everyone's decode; FlashAttention
   keeps long-context attention from becoming an HBM-traffic problem.

Sanity throughput: aggregate BW $8 \times 3.35 = 26.8$ TB/s. Each decode
step streams the 70 GB of weights plus the batch's KV cache (~67 GB at batch
50 with fp16 KV), about 137 GB per step → ~195 steps/s. With 50 sequences per
step that's order $10^4$ tok/s aggregate, and each user gets at most ~195
tok/s, in practice tens of tok/s after prefill and scheduling overheads. All targets are met
with margin.
</details>
