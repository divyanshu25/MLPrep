# Solutions — buggy_transformer.py

**Do not read this until you've spent the 45 minutes.** Finding bugs is the
skill; reading about bugs is not.

## Expected numbers (seed 1337, 300 steps, CPU)

| | step 0 | step 299 | sample |
|---|---|---|---|
| **Buggy** | 3.64 | **~3.8** (oscillates 3.1–3.9, never sustainably below ln 37 = 3.61) | garbage — a few characters repeated |
| **Fixed** | 3.69 | **~0.37** (under 2.0 by step ~100) | recognizable corpus phrases |

The buggy model is not merely slow — it hovers at the "knows nothing" baseline
and gets *worse* late in training. Each bug below explains its share of that.

---

## The 7 bugs

### Bug 1 — positional embedding indexed by token ids

**Location:** `TinyLM.forward`, line 190: `pos = self.pos_emb(idx)`.

**Symptom:** the model never learns word order — it behaves like a bag-of-chars
model. Generations have locally plausible character frequencies but no
sequential structure. Sneaky detection: it doesn't crash, because
`vocab_size (37) < block_size (64)`, so token ids are valid *indices* into the
positional table. Test: feed the same token at two different positions —
the input representations are identical when they must differ.

**Fix:** `pos = self.pos_emb(torch.arange(T))` — index by *position*, then
broadcast `(T, C)` against `tok (B, T, C)`.

**Concept:** attention is permutation-equivariant; positional embeddings are
the *only* thing telling the model where a token sits. Module 4 notes
(attention section) and `mini_gpt.py`.

### Bug 2 — causal mask uses the wrong triangle

**Location:** `CausalSelfAttention.__init__`, line 124:
`torch.triu(torch.ones(...))`.

**Symptom:** with `triu`, the *kept* entries are `j >= i` — every position
attends to itself and the **future**, and is blind to the past. Training loss
can drop artificially fast in a clean model (position `t` can see token `t+1`,
which is its own target — leakage), but generation is broken: the last position
attends only to itself. Classic signature when this is the *only* bug:
suspiciously low train loss + garbage samples.

**Fix:** `torch.tril(...)` — keep `j <= i`, mask out `j > i`.

**Concept:** autoregressive factorization requires `p(x_t | x_{<t})`; the mask
enforces it. Module 4 notes (the causal-mask derivation) and
`attention_scratch.py`. Related reported variants: mask missing entirely, or
applied *after* softmax (rows then don't sum to 1).

### Bug 3 — softmax over the wrong axis

**Location:** `CausalSelfAttention.forward`, line 142:
`F.softmax(att, dim=-2)`.

**Symptom:** normalizes over *queries* (columns) instead of *keys* (rows), so
each row of attention weights no longer sums to 1 — the output for token `i` is
not a weighted average of values but an arbitrarily scaled mix. Detect with the
one-line sanity check every interviewer loves:
`att.sum(-1)` must be all-ones. Here it isn't.

**Fix:** `F.softmax(att, dim=-1)`.

**Concept:** attention is "a probability distribution over which keys to look
at, per query" — the distribution axis is the key axis. Module 4 notes;
module 3 for softmax itself.

### Bug 4 — missing `optimizer.zero_grad()`

**Location:** training loop in `main()`, lines 246–249 — `loss.backward()`
then `opt.step()` with no `opt.zero_grad()` anywhere.

**Symptom:** `.backward()` *accumulates* into `.grad`, so the effective
gradient at step `t` is the sum of all gradients since step 0. Loss falls a
little early (the sum is momentum-like), then stalls and oscillates upward —
exactly the late-run 3.6 → 3.9 drift you see. Detect by printing
`p.grad.norm()` for one parameter across steps: it grows roughly linearly
instead of hovering.

**Fix:** `opt.zero_grad()` before `loss.backward()` (or after `opt.step()`).

**Concept:** the accumulate-by-default contract of autograd (it exists to
support gradient accumulation over micro-batches — when done *deliberately*).
Module 3 notes, training-loop anatomy; `training_dynamics.py`.

### Bug 5 — missing output projection after concatenating heads

**Location:** `CausalSelfAttention.forward`, line 146: `return y` — the
`self.proj` defined in `__init__` is never applied.

**Symptom:** the subtlest one. Shapes are fine (`(B, T, C)` either way) and the
model still trains — but concatenated heads are stacked side by side with no
channel mixing, so head `h`'s output only ever writes to its own 16-dim slice
of the residual stream. Capacity quietly drops. Detect by auditing parameters:
`proj.weight` exists but its gradient is `None`/zero after `backward()` —
unused-parameter checks catch exactly this in real codebases.

**Fix:** `return self.proj(y)`.

**Concept:** W_o is part of the multi-head design: heads compute in disjoint
subspaces and W_o recombines them. Module 4 notes, multi-head section.

### Bug 6 — scaling by d_model instead of √d_head

**Location:** `CausalSelfAttention.forward`, line 138:
`(q @ k.T) / n_embd` — divides by 64 instead of √16 = 4.

**Symptom:** scores are 16× too small, so after softmax every row is nearly
uniform — attention starts almost contentless, and gradients through the
softmax are tiny (softmax is flattest where inputs are equal). The model *can*
partially compensate by growing W_q/W_k, but it wastes the budget. Detect by
histogramming attention weights early in training: they should show structure
within a few dozen steps; here they stay near 1/T.

**Fix:** `/ math.sqrt(self.head_dim)`.

**Concept:** for unit-variance q, k entries, Var(q·k) = d_head, so dividing by
√d_head restores unit variance — same variance-tracking argument as
Xavier init. Module 4 notes (the √d_k derivation); module 3 for init variance.

### Bug 7 — residual taken from the normalized tensor

**Location:** `Block.forward`, lines 171–172:

```python
h = self.ln1(x)
x = h + self.attn(h)      # residual should be x, not h
```

**Symptom:** the skip connection carries `LN(x)` instead of `x`, so the
identity path — the whole point of residuals — is broken: what flows forward is
a normalized, re-centered copy, and the clean gradient highway through the
plain addition is gone. With 2 layers this degrades quietly; with depth it
would kill training. Detect by comparing per-layer gradient norms
(they shrink layer by layer) or simply by tracing the pre-LN pattern by hand:
pre-LN is `x + f(LN(x))` — normalize *only* on the branch.

**Fix:**

```python
x = x + self.attn(self.ln1(x))
```

**Concept:** residual streams and why pre-LN keeps optimization depth-proof.
Module 3 notes (LayerNorm), module 4 notes (block structure).

---

## Extension — KV-cached generation (worked implementation)

Why: uncached generation re-runs the full forward on all `T` tokens for every
new token — O(T²·d) attention work per step, O(N·T²·d) for N tokens. But the
keys and values of past tokens *never change* (causality!), so cache them and
feed only the newest token: per-step attention cost drops to O(T·d).

The three changes: attention accepts and returns a cache; the model threads
caches through blocks and offsets positional embeddings by `past`; generation
feeds the full prompt once, then one token at a time.

```python
class CausalSelfAttention(nn.Module):
    # __init__ unchanged (fixed version: tril mask, proj, sqrt scaling)

    def forward(self, x, cache=None):
        B, T, C = x.shape                            # T = tokens in THIS call
        qkv = self.qkv(x)                            # (B, T, 3C)
        q, k, v = qkv.split(n_embd, dim=2)           # each (B, T, C)
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2)  # (B,h,T,d)
        k = k.view(B, T, n_head, self.head_dim).transpose(1, 2)  # (B,h,T,d)
        v = v.view(B, T, n_head, self.head_dim).transpose(1, 2)  # (B,h,T,d)
        if cache is not None:
            pk, pv = cache                           # (B,h,P,d) past K, V
            k = torch.cat([pk, k], dim=2)            # (B,h,P+T,d)
            v = torch.cat([pv, v], dim=2)            # (B,h,P+T,d)
        S = k.size(2)                                # total keys = P + T
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,T,S)
        # Query i of this call sits at absolute position S-T+i and may see
        # keys 0..S-T+i. Slice the last T rows of the (S,S) causal mask.
        # (For single-token decoding, T=1: the row is all-keep — no masking
        # needed, the cache only ever contains the past.)
        att = att.masked_fill(self.mask[S - T:S, :S] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                 # (B,h,T,S) rows sum to 1
        y = att @ v                                  # (B,h,T,S)@(B,h,S,d)->(B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)                  # output + updated cache


class Block(nn.Module):
    def forward(self, x, cache=None):
        att_out, new_cache = self.attn(self.ln1(x), cache)
        x = x + att_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyLM(nn.Module):
    def forward(self, idx, targets=None, caches=None, past=0):
        B, T = idx.shape
        tok = self.tok_emb(idx)                            # (B, T, C)
        pos = self.pos_emb(torch.arange(past, past + T))   # (T, C) -- OFFSET!
        x = tok + pos
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if caches is None else caches[i])
            new_caches.append(c)
        x = self.ln_f(x)
        logits = self.head(x)                              # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size), targets.view(B * T))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # Cache demo: no window sliding, so total length must fit.
        assert idx.size(1) + max_new_tokens <= block_size
        caches, past, x = None, 0, idx      # first call: the whole prompt
        for _ in range(max_new_tokens):
            logits, _, caches = self(x, caches=caches, past=past)
            past += x.size(1)
            logits = logits[:, -1, :] / temperature        # (B, V)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)              # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)
            x = nxt                          # subsequent calls: ONE new token
        return idx
```

The two classic mistakes to avoid (both silent-wrong, not crashes):

1. **Forgetting the position offset** — the new token must get
   `pos_emb[past]`, not `pos_emb[0]`. Miss this and cached generation
   diverges from uncached after the first token.
2. **Masking wrong on the incremental step** — with `T=1` the query is the
   newest position and may see *all* cached keys; applying a fresh `(1,1)`
   causal mask is fine, but reusing the `[:T,:T]` slice of the training mask
   against `S` keys crashes or silently truncates. The `[S-T:S, :S]` slice
   above handles prompt-pass and single-token-pass uniformly.

**Acceptance test** (do this in the interview unprompted): greedy-decode 50
tokens with and without cache from the same prompt and `assert torch.equal`.
Also compare final-position logits of an incremental pass against one full
forward with `torch.allclose(atol=1e-5)`. Both checks pass on this
implementation (verified).

Memory footnote they may probe: cache size is
`2 · n_layer · B · T · n_embd` floats (per token: 2·L·C). For a 7B model at
fp16, ~0.5 MB per token — this, not compute, is what limits batch size at
serving time.

---

## The general debugging playbook

The order matters — cheap, high-information checks first. This is what the
interviewer wants to see you *have*, not improvise.

1. **Loss at init ≈ ln(vocab_size).** Uniform prediction gives
   `-ln(1/V) = ln V` (3.61 here). Much higher → init/scaling broken
   (logits too large). Much lower at step 0 → leakage. This one number
   validates the whole forward pass before you read any code.
2. **Overfit one batch.** Train on a single fixed batch; a correct model
   drives loss to ~0 in a few hundred steps. If it can't memorize 32
   examples, the bug is in model/optimizer, not data or capacity. If it can,
   but the full run fails, look at data pipeline or regularization.
3. **Check the loss *trend*, not the value.** Flat at ln V → no signal reaches
   the logits (embedding path, dead mask). Falls then drifts up → optimizer
   state problem (zero_grad, lr, accumulation). Suspiciously fast fall →
   leakage (wrong mask direction, target in input).
4. **Gradient norms.** `[(n, p.grad.norm().item()) for n, p in
   model.named_parameters()]` after one backward. Zeros → dead branch
   (unused proj — bug 5). Growing across steps → missing zero_grad (bug 4).
   Shrinking with depth → broken residuals (bug 7).
5. **Attention-weight sanity.** `att.sum(-1) == 1` (softmax axis);
   `att[0,0]` upper triangle is 0 (causality); early-training rows not
   uniform (scaling). Three asserts, three bugs covered.
6. **Perturbation tests.** Change token at position `j`; outputs at positions
   `< j` must not change (causality, catches bug 2 from the outside). Permute
   the input sequence; output *should* change (catches dead positional
   embeddings, bug 1).
7. **Only then read code line by line**, comparing each formula against the
   spec you can re-derive: `softmax(QKᵀ/√d_h + M)V`, `x + f(LN(x))`,
   `tok_emb(idx) + pos_emb(arange(T))`.

Full derivations behind every check: modules 3 (init, optimizer, LayerNorm)
and 4 (attention, mask, √d_k). The one-batch-overfit and ln V tricks are in
module 3's notes on training dynamics.
