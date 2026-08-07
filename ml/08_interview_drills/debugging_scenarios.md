# Production Debugging Scenarios

12 cards in the format interviewers actually use: you get a symptom and an
open prompt — "how do you debug this?" What's being graded is not the final
answer but the *order* of your checks: cheap and discriminating first, and a
mechanism attached to every hypothesis (a cause you can't explain is a guess,
not a diagnosis).

Protocol: read the symptom, talk through your diagnosis tree out loud for
~5 minutes, then unfold. Score yourself on whether your first three checks
match — the long tail matters less than the triage order.

---

## Card 1 — "Loss is NaN at step 400."

<details><summary>Diagnosis tree</summary>

It ran fine for 399 steps, so shapes and wiring are right — something
*numerical* crossed a threshold. Check in order:

1. **Was there a spike first?** Plot loss/grad-norm just before death. A
   growing grad norm → exploding gradients: lr too high or no clipping.
   Mechanism: per-layer Jacobian scales > 1 compound; one large step enters a
   steep region, the next is larger — geometric blow-up, then inf, then
   inf − inf = NaN. Fix: clip global norm, lower lr, warmup.
2. **Mixed precision?** fp16 overflows at 65504 — one large activation or
   logit saturates, and NaN propagates through everything it touches. Check
   the loss-scaler stats / try bf16 or fp32 to confirm. Mechanism: 5-bit
   exponent range, not precision.
3. **A log/sqrt/div of a bad value.** Hand-written `log(p)` with p → 0,
   `x/std` with std → 0, custom loss without eps. Bisect by checking
   `torch.isfinite` on activations (forward hooks) to find the first layer
   producing inf/NaN.
4. **A poisoned batch.** NaN in the *data* (corrupt row, divide-by-zero
   feature) surfaces exactly when that shard is reached — deterministic step
   number across reruns is the tell. `assert torch.isfinite(x).all()` in the
   loader.
5. Rarer: lr schedule bug at a boundary (e.g. warmup ends, lr jumps), Adam
   eps too small with fp16's coarse grid.

The step-number heuristic: same step every run → data or schedule;
different step every run → numerics riding random batches.
</details>

---

## Card 2 — "Train loss falls nicely; eval loss is flat from step 0."

<details><summary>Diagnosis tree</summary>

Flat-from-the-start is the key clue: this is almost never overfitting —
overfitting shows eval *improving first, then* diverging. Something is broken
in the eval path itself.

1. **Is eval seeing the model's updates?** Evaluating a stale copy, an EMA
   that never updates, or a checkpoint loaded once. Print a weight hash at
   each eval.
2. **Eval data pipeline broken:** labels misaligned (shuffled features vs
   unshuffled labels — the classic), wrong preprocessing/normalization, wrong
   tokenizer. A model can't score aligned-random labels, so loss pins at
   chance. Check: eval loss ≈ ln V / chance level exactly? Then eval targets
   carry no signal.
3. **`model.eval()`/mode bugs** — usually degrades rather than flattens, but
   dropout at eval on a small head can pin metrics. Cheap to check.
4. **Train/eval distribution mismatch:** eval set from a different domain than
   train — loss won't be at exact chance but stuck high. Compare a training
   batch evaluated in eval mode: if *that* loss is low while the eval set's
   isn't, it's the data, not the plumbing.
5. Only after all that: genuine zero generalization (memorizing a tiny train
   set) — confirmable because train-in-eval-mode loss is low AND eval is at
   chance AND the train set is small.

The single most diagnostic number: eval loss vs exact chance level. Exactly
chance = broken labels/plumbing; above chance but stuck = distribution or
capacity.
</details>

---

## Card 3 — "Loss plateaus at exactly ln(vocab_size)."

<details><summary>Diagnosis tree</summary>

ln V is the uniform-prediction loss — the model has learned *nothing*, and
"exactly" means the output is genuinely uniform, not merely bad. Signal is
severed somewhere between input and logits.

1. **Targets decorrelated from inputs:** shuffled labels, off-by-one that
   pairs x with a random y, or targets overwritten. Check one batch by hand —
   literally print x and y and verify y is x shifted by one.
2. **Input path dead:** all-zero/constant inputs (preprocessing bug), or an
   embedding whose output is detached/overwritten. Print input stats and
   embedding-output variance.
3. **Gradient path dead:** a `detach()`/`no_grad` in the forward, a frozen
   module, an optimizer constructed over the wrong parameter list (e.g.
   built before moving/replacing modules). Check `p.grad` is non-None and
   nonzero for the *first* layer after backward.
4. **Attention fully masked:** a causal mask with the wrong comparison can
   mask everything except self, or everything, making context useless — for
   an LM that caps learning near unigram entropy (close to, though not
   exactly, ln V; "exactly" points harder at 1–3).
5. **lr pathologically small / optimizer never stepping** — verify a weight
   actually changes between step 0 and step 100 (`(w0 - w100).abs().max()`).

Fastest single experiment: overfit one batch. If loss won't go below ln V on
32 memorizable examples, it's model/optimizer/targets — data volume and
architecture size are exonerated.
</details>

---

## Card 4 — "Offline AUC improved; online CTR dropped."

<details><summary>Diagnosis tree</summary>

The offline metric and the online objective disagree — the question is which
link in the chain broke. Most likely first:

1. **Training/serving skew:** the feature values at serving differ from
   training (missing features silently defaulting, staler aggregates,
   different preprocessing code path). Log serving-time features and diff
   distributions against training data. Mechanism: model applies learned
   weights to inputs from a different distribution — offline eval never sees
   this because it uses training-pipeline features.
2. **Offline eval leakage:** the AUC gain is fake — a feature that
   post-dates the label (e.g. built from data after the click), or eval rows
   overlapping training. New feature + big offline jump = treat as guilty
   until proven innocent. Check per-feature: does removing it collapse the
   gain? Is it available at request time?
3. **AUC measures the wrong thing:** AUC is ranking over the *whole*
   population; CTR lives at the top of each user's slate. A model can win
   global ranking while losing top-1 precision, or improve on segments that
   never see impressions. Check ranking metrics at serving cutoff (nDCG@k,
   precision@1) and per-segment.
4. **Feedback loop / distribution shift:** the old model chose what got
   shown; the new model's "better" scores redistribute impressions into
   regions where its training data is thin. Also: calibration shift breaking
   a downstream bidder/threshold even with better ranking.
5. **Boring but real:** experiment plumbing — wrong model actually deployed,
   A/B assignment bias, insufficient power. Verify the served model's hash
   and the experiment's sample-ratio first if the drop is small.

Module 5 (leakage, AUC semantics, calibration) and module 6
(training/serving skew) cover the mechanisms.
</details>

---

## Card 5 — "Model quality collapsed after int8 quantization."

<details><summary>Diagnosis tree</summary>

Quantization maps floats to 256 levels per tensor (or per channel/group);
collapse means some tensor's range made that mapping destructive.

1. **Outlier activations:** a few channels carry values 100× the rest
   (ubiquitous in LLMs > ~6B), so per-tensor scales stretch to cover them and
   normal values round to a handful of levels. Check activation ranges per
   channel. Fix: per-channel/group scales, outlier-aware schemes
   (LLM.int8-style mixed precision, SmoothQuant), or leave those layers in
   fp16.
2. **Calibration data mismatch:** static ranges were computed on
   unrepresentative samples, so real inputs clip. Recalibrate on in-domain
   data; compare observed ranges vs calibrated ranges.
3. **Sensitive layers quantized:** embeddings, final LM head / logits,
   LayerNorm, small residual paths. Standard practice keeps them in higher
   precision; check what the toolchain actually converted.
4. **Localize it:** binary-search by dequantizing half the layers at a time,
   or compare per-layer output MSE (float vs int8) to find the first layer
   whose error explodes. Error compounds through depth — earliest bad layer
   is the target.
5. **Accumulator/zero-point bugs** in custom kernels: int8×int8 must
   accumulate in int32; symmetric-vs-asymmetric mismatches shift everything.
   Test one layer against the float reference on random inputs.

Key mental model: quantization error is uniform noise of size range/256 —
harmless at activations' typical scale, fatal when outliers inflate "range".
</details>

---

## Card 6 — "Distributed training diverges at 64 GPUs but is stable at 8."

<details><summary>Diagnosis tree</summary>

Two families: the *math changed* with scale (batch/lr), or the *plumbing*
is wrong (sync bugs). Disambiguate with one question: does the 64-GPU run
match 8-GPU loss for the first few hundred steps and *then* diverge (math),
or is it worse from step 1 (plumbing)?

1. **Effective batch × lr scaling:** 8× more GPUs at fixed per-GPU batch =
   8× global batch; if lr was scaled linearly (or someone scaled it) the run
   can sit past the stability edge. Large batch also shrinks gradient noise,
   which was implicitly regularizing/damping. Fix: longer warmup, sqrt or
   tuned scaling, lower lr. This is the most common cause.
2. **Warmup too short for the new lr:** Adam's second-moment estimates are
   garbage early; big lr + big batch + short warmup diverges exactly like
   this. Check the schedule actually in effect on the 64-GPU run.
3. **Gradient all-reduce wrong:** summing instead of averaging (gradients
   8× too large — equivalently lr silently 8×), or a framework flag
   (`gradient_as_bucket_view`, custom hooks) behaving differently at the new
   world size. Verify: one step on identical data at world size 8 vs 64,
   compare gradient norms — they should match, not scale.
4. **Data sharding bug:** duplicate shards (each of 64 workers seeing the
   same data = correlated gradients), or a shard-boundary bug producing
   corrupt batches only when sharded 64 ways. Check per-worker data
   fingerprints.
5. **Numerics at scale:** ring all-reduce over more ranks changes summation
   order — noticeable in fp16 with loss scaling near the edge; also one bad
   node (faulty GPU producing inf) — check per-rank grad norms for an
   outlier machine.

Golden debugging tool: fixed batch, fixed seed, one optimizer step —
bitwise/tolerance-compare gradients and updates across world sizes before
staring at loss curves.
</details>

---

## Card 7 — "Generation repeats the same phrase forever."

<details><summary>Diagnosis tree</summary>

1. **Decoding strategy, not the model:** greedy/low-temperature decoding on
   LMs has a known self-reinforcement failure — once a phrase appears, its
   continuation becomes even more probable (the context now contains it), and
   argmax loops. Check the sampling config first: temperature 0? beam search
   without length/ngram penalties? Fix: temperature ~0.7–1.0, top-p/top-k,
   repetition or no-repeat-ngram penalty.
2. **Sampling code bug:** temperature applied after argmax, sampling from
   unrenormalized probs, top-k=1 by accident, or taking `argmax` where
   `multinomial` was intended. Print the actual next-token distribution: is
   it genuinely peaked, or is the code ignoring its spread?
3. **KV-cache/position bug:** if positions aren't offset (`pos_emb[0]` fed
   for every new token) or the cache is corrupted, the model sees the same
   effective state each step and emits the same token — a loop of period 1–2
   right from the start is the signature. Test: cached vs uncached generation
   must match token-for-token under greedy decoding.
4. **Model degeneracy:** heavily overfit small-data models or
   over-optimized RLHF models collapse to high-reward phrases. Check
   perplexity on held-out text; if fine, the model's fine and it's decoding.
5. **Prompt-format mismatch:** chat model without its expected template can
   fall into degenerate continuations. Verify the exact token sequence in.

Order matters here: 1–2 are minutes to check; people who jump to "the model
is broken" fail this card.
</details>

---

## Card 8 — "Train loss dropped to near zero in 50 steps. Eval is garbage."

<details><summary>Diagnosis tree</summary>

A language/classification model that "solves" training almost instantly has
found a shortcut, not the task. Ask: what information does the training
forward pass have that the eval pass doesn't?

1. **Target leaked into the input:** the label is (a function of) an input
   feature; for LMs, inputs and targets aligned without the one-position
   shift — the model learns the identity map. Inspect one (x, y) pair by
   hand.
2. **Causal mask missing or wrong direction:** each position can attend to
   its own target one step ahead (a `triu`/`tril` swap, or no mask). Training
   loss collapses; generation — where the future genuinely doesn't exist —
   is garbage. The flagship drill's bug 2. Check: perturb a future token,
   assert past outputs unchanged.
3. **Eval/train split contamination inverted:** train loss is real but the
   task is trivially memorizable (duplicated rows, tiny dataset, model far
   too large) — legitimate memorization, and eval reveals it. Distinguish
   from 1–2 by data-size arithmetic: can this model plausibly memorize this
   set? 50 steps says shortcut, not memorization.
4. **Loss mis-wired:** loss computed on inputs instead of targets, or
   on a constant slice — near-zero loss that means nothing. Verify the loss
   of an untrained model is ln V, and that shuffling *labels only* sends
   training loss back up (it must — if not, targets aren't driving the loss).

Rule of thumb: legitimate learning curves bend; cliffs mean leakage.
Module 5 (leakage) and module 4 (masking).
</details>

---

## Card 9 — "Loss spikes sharply at regular intervals, then recovers."

<details><summary>Diagnosis tree</summary>

Periodicity is the clue — find what shares the period.

1. **Epoch boundary:** period = steps-per-epoch. Causes: dataloader
   re-shuffle changing sample correlation, a last partial batch with
   batch-size-sensitive layers (BatchNorm with batch of 1–2), stateful
   iterators resetting, or curriculum/sorted data restarting from "easy".
   Check `drop_last`, and whether the spike step × batch size ≈ dataset size.
2. **lr schedule events:** warm restarts, decay milestones, cycle
   boundaries. Overlay lr on the loss plot — takes one minute, catches this
   class entirely.
3. **Data ordering within the epoch:** unshuffled data with a hard/different
   segment at a fixed offset (e.g. sorted by length or source — the spike is
   "the hard shard"). Compare per-batch loss against a data fingerprint.
4. **Checkpoint/eval interference:** eval or checkpointing every N steps
   that mutates state — most infamously, forgetting `model.train()` after
   eval, so dropout/BN stay in eval mode for some steps; or an EMA swap
   that isn't swapped back.
5. **Infra with a period:** shard rotation in streaming loaders,
   prefetch-buffer exhaustion — check whether the period matches a data
   service's rotation rather than the math.

Mechanism note for the recovery shape: Adam absorbs a transient gradient
scale change within ~1/(1−β₂) steps, which is why spikes self-heal — the
spike is information about *inputs*, the recovery about the optimizer.
</details>

---

## Card 10 — "Memory usage grows every step until OOM at step ~3000."

<details><summary>Diagnosis tree</summary>

Slow growth = something accumulates per step. In order of frequency:

1. **Keeping the graph alive:** appending `loss` (the tensor, not
   `loss.item()`) to a list, or summing `total_loss += loss` — every stored
   loss retains its entire computation graph. The classic. Grep every place
   a loss/output leaves the step function.
2. **`retain_graph=True`** left in from a debugging session, or a hidden
   second backward path keeping buffers.
3. **Hidden state carried across steps without detach** (RNNs/TBPTT,
   caches): the graph grows one step longer every iteration — memory *and*
   step time grow together. Growing step time is the fingerprint that
   separates this from fragmentation. Fix: `h = h.detach()`.
4. **Hooks/metrics accumulating tensors:** logging frameworks storing GPU
   tensors, activation hooks appending, custom metrics lists. Move to
   `.item()`/`.cpu()` at the boundary.
5. **Fragmentation, not a leak:** allocator can't reuse blocks under
   variable sequence lengths — memory sawtooths upward without a true leak.
   Diagnose with `torch.cuda.memory_summary()`: reserved grows while
   allocated cycles. Fix: bucketed/padded batching, expandable segments.

Tooling order: plot `memory_allocated()` per step (linear growth = leak);
then diff `Tensor` counts via `gc` between steps to find who's accumulating.
</details>

---

## Card 11 — "Eval loss is consistently *lower* than train loss."

<details><summary>Diagnosis tree</summary>

Not always a bug — but verify the boring causes before accepting the benign
ones.

1. **Regularization accounting (benign, most common):** train loss is
   computed *with* dropout/augmentation active and includes the penalty term;
   eval runs the clean model on clean data. Check: recompute train loss in
   eval mode — if the gap closes, case closed.
2. **Timing artifact (benign):** "train loss" is averaged over the epoch
   while eval happens at the epoch's *end*, so eval sees a better model than
   the epoch-average. Compare against end-of-epoch train batches only.
3. **Eval set is easier (bug-adjacent):** non-i.i.d. split — eval drawn from
   a cleaner source, shorter sequences, or duplicated "easy" content. Check
   difficulty stats (length, class mix, source) across the split.
4. **Leakage into eval (bug):** eval examples (near-)duplicated in training
   — the model has memorized them. Deduplicate across the split; the LLM-era
   version is n-gram/fuzzy overlap, not exact match.
5. **Different computation (bug):** eval uses teacher forcing / different
   loss reduction / different sequence packing than training. Verify both
   paths call the same loss code on comparable batches.

The one-line discriminator: same data, same mode, same code → if the gap
persists, it's the *data split*; if it vanishes, it was the *mode*.
</details>

---

## Card 12 — "Offline metrics are great; the deployed model's predictions are visibly wrong. Same model file."

<details><summary>Diagnosis tree</summary>

Same weights, different answers ⇒ the *inputs* or the *surrounding code*
differ. This is training/serving skew, module 6's core pathology.

1. **Feature skew:** serving computes features with different code than
   training (reimplemented in another language, different defaults for
   missing values, staler feature-store snapshots). Log raw serving features
   for N requests, replay them through the offline pipeline, and diff — this
   single check settles most cases.
2. **Preprocessing mismatch:** normalization stats (mean/std) not shipped
   with the model, different tokenizer version/vocab, different image
   resize/crop. A model is (weights + exact preprocessing); check both are
   versioned together.
3. **Time-travel in offline eval:** offline features were computed with
   post-event information (aggregates including the label window), so offline
   metrics were never real. Audit feature timestamps vs request time.
4. **Serialization/runtime drift:** export to ONNX/TensorRT changed op
   semantics, fp16 serving vs fp32 eval, wrong output head selected,
   label-index mapping shuffled between training and serving. Golden-input
   test: fixed input, assert serving output == offline output to tolerance —
   this belongs in CI, and interviewers love hearing that.
5. **Population shift:** the served traffic differs from the offline
   distribution (new market, bot traffic, feedback loops). Monitor input
   distributions (PSI/KL per feature) against training data.

Answer shape that scores well: "replay logged serving inputs offline; if
outputs match, it's the data distribution; if they don't, binary-search the
pipeline stages" — a bisection framing, stated upfront.
</details>
