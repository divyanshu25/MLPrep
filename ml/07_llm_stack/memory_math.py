"""
memory_math.py — every memory number in an LLM interview, derived and computed.

Run:  uv run python ml/07_llm_stack/memory_math.py

This is the arithmetic behind the canonical questions:
  "How much memory does it take to train a 70B model? Walk me through it."
  "What's the KV cache for a 7B at batch 32, 4k context?"

Everything is a handful of multiplications. The derivations live in the
comments; the code just refuses to let you make arithmetic mistakes.

-------------------------------------------------------------------------------
TRAINING MEMORY, derived (see notes.md §2.1)
-------------------------------------------------------------------------------
Mixed precision + Adam stores, per parameter:

  fp16 working weights   2 bytes   (fast tensor-core matmuls)
  fp16 gradients         2 bytes   (backward output, matmul precision)
  fp32 master weights    4 bytes   (fp16 update eta*g underflows: often < ulp)
  fp32 Adam m            4 bytes   (running mean of grads)
  fp32 Adam v            4 bytes   (running mean of squared grads)
                        ---------
                        16 bytes/param   ("optimizer state" = the fp32 12)

Plain fp32 + SGD(momentum) would be 4+4+4 = 12; fp32+Adam 16; mixed precision
with fp32 grad accumulation 20. We parameterize this below.

ZeRO shards parts of this across the G data-parallel GPUs:
  stage 0 (plain DP):  16 * N                        on every GPU
  stage 1:             (2+2) * N + 12 * N / G        (shard optimizer state)
  stage 2:             2 * N + (2+12) * N / G        (+ shard gradients)
  stage 3 (FSDP):      16 * N / G                    (+ shard weights)

-------------------------------------------------------------------------------
ACTIVATION MEMORY, derived
-------------------------------------------------------------------------------
Per transformer layer, per token, the saved-for-backward tensors are a few
multiples of the hidden size h. The standard estimate (Megatron-LM paper,
"Reducing Activation Recomputation", eq. 2), fp16, vanilla attention, no
sequence parallelism, per layer:

    bytes ≈ s * b * h * (34 + 5 * a * s / h)

where s=seq len, b=microbatch, h=hidden dim, a=heads. The 34 counts the
h-sized tensors (LN inputs, QKV, MLP intermediates at 4h, etc.); the
5*a*s/h term is the s x s attention matrices (softmax input/output +
dropout mask) — the part FlashAttention deletes.

With full gradient checkpointing we store only each layer's input
(2*s*b*h bytes, fp16) and recompute the rest during backward (+~33% FLOPs).

-------------------------------------------------------------------------------
KV CACHE, derived (see notes.md §5.2)
-------------------------------------------------------------------------------
Per generated/prompt token, per layer, we cache one K and one V vector of
dimension n_kv_heads * d_head:

    bytes = 2 (K and V) * n_layers * n_kv_heads * d_head * bytes_per_el
            * seq_len * batch

For MHA n_kv_heads = n_heads so n_kv_heads*d_head = d_model; GQA divides
this by (n_heads / n_kv_heads).
"""

GiB = 1024**3  # we report GiB ("GB" colloquially); GPUs are marketed in GB=1e9
GB = 1e9


# ---------------------------------------------------------------------------
# model configs (the two workhorses of interview arithmetic)
# ---------------------------------------------------------------------------
LLAMA_7B = dict(name="7B (Llama-2-7B-like)", params=7e9,
                n_layers=32, n_heads=32, n_kv_heads=32, d_model=4096)
LLAMA_70B = dict(name="70B (Llama-2-70B-like)", params=70e9,
                 n_layers=80, n_heads=64, n_kv_heads=8, d_model=8192)
# note: real Llama-2-70B uses GQA with 8 KV heads. For the *training* memory
# question we only need the param count; GQA matters for the KV cache.


# ---------------------------------------------------------------------------
# training-state memory
# ---------------------------------------------------------------------------
def training_state_bytes(params, precision="mixed", optimizer="adam"):
    """Per-component bytes for the full (unsharded) training state.

    Returns dict of component -> bytes. Derivation in the module docstring.
    """
    if precision == "mixed":
        w_bytes, g_bytes = 2, 2          # fp16/bf16 working copies
        master = 4                        # fp32 master weights
    elif precision == "fp32":
        w_bytes, g_bytes = 4, 4
        master = 0                        # weights already fp32
    else:
        raise ValueError(precision)

    if optimizer == "adam":
        opt = master + 4 + 4              # master + m + v (all fp32)
    elif optimizer == "sgd_momentum":
        opt = master + 4                  # master + velocity
    elif optimizer == "sgd":
        opt = master
    else:
        raise ValueError(optimizer)

    return {
        "weights": params * w_bytes,
        "gradients": params * g_bytes,
        "optimizer state": params * opt,
    }


def zero_shard(state, G, stage):
    """Apply ZeRO sharding to the state dict: divide sharded parts by G."""
    sharded = {0: [], 1: ["optimizer state"],
               2: ["optimizer state", "gradients"],
               3: ["optimizer state", "gradients", "weights"]}[stage]
    return {k: (v / G if k in sharded else v) for k, v in state.items()}


# ---------------------------------------------------------------------------
# activation memory
# ---------------------------------------------------------------------------
def activation_bytes_per_layer(seq, batch, d_model, n_heads):
    """Megatron estimate, fp16, vanilla attention, no recomputation."""
    return seq * batch * d_model * (34 + 5 * n_heads * seq / d_model)


def activation_bytes(seq, batch, d_model, n_heads, n_layers,
                     checkpointing=False):
    if checkpointing:
        # store only each layer's input activation (fp16): 2 bytes * s*b*h,
        # plus one layer's worth of recompute workspace at peak
        return (2 * seq * batch * d_model * n_layers
                + activation_bytes_per_layer(seq, batch, d_model, n_heads))
    return n_layers * activation_bytes_per_layer(seq, batch, d_model, n_heads)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
def kv_cache_bytes(seq, batch, n_layers, n_kv_heads, n_heads, d_model,
                   bytes_per_el=2):
    d_head = d_model // n_heads
    per_token_per_layer = 2 * n_kv_heads * d_head * bytes_per_el  # K and V
    return per_token_per_layer * n_layers * seq * batch


# ---------------------------------------------------------------------------
# pretty printing
# ---------------------------------------------------------------------------
def fmt(b):
    return f"{b / GiB:10.1f} GiB"


def table(title, rows, total_label="TOTAL"):
    width = 58
    print(f"\n{title}")
    print("-" * width)
    for k, v in rows.items():
        print(f"  {k:<38}{fmt(v)}")
    print("-" * width)
    print(f"  {total_label:<38}{fmt(sum(rows.values()))}")


# ---------------------------------------------------------------------------
# Case 1: train a 7B model, single GPU attempt (the "does it even fit?" case)
# ---------------------------------------------------------------------------
def case_1():
    m = LLAMA_7B
    seq, batch = 4096, 1
    print("=" * 70)
    print("CASE 1 — train a 7B model, mixed precision + Adam, seq 4096, "
          "microbatch 1")
    print("=" * 70)
    state = training_state_bytes(m["params"])
    # 16 bytes/param: 2 (fp16 w) + 2 (fp16 g) + 12 (fp32 master + m + v)
    table("Training state (unsharded) — 16 bytes/param", state)
    act = activation_bytes(seq, batch, m["d_model"], m["n_heads"],
                           m["n_layers"], checkpointing=False)
    act_ckpt = activation_bytes(seq, batch, m["d_model"], m["n_heads"],
                                m["n_layers"], checkpointing=True)
    table("Activations", {
        "no checkpointing (all layers)": act,
        "with full checkpointing": act_ckpt,
    }, total_label="(alternatives, not summed)")
    need = sum(state.values()) + act_ckpt
    print(f"\n  Verdict: {fmt(sum(state.values()))} of state alone."
          f"  An 80 GB GPU holds {80 * GB / GiB:.1f} GiB.")
    print("  => even a 7B model does NOT train on one 80 GB GPU without")
    print("     sharding (ZeRO-2 across 8 GPUs: "
          f"{fmt(sum(zero_shard(state, 8, 2).values()))} per GPU + acts). "
          f"Total needed ~{need / GiB:.0f} GiB.")


# ---------------------------------------------------------------------------
# Case 2: the canonical 70B walk-through, then ZeRO-3 across 64 GPUs
# ---------------------------------------------------------------------------
def case_2():
    m = LLAMA_70B
    G = 64
    seq, batch = 4096, 1
    print("\n" + "=" * 70)
    print(f"CASE 2 — train a 70B model, ZeRO stages across {G} GPUs")
    print("=" * 70)
    state = training_state_bytes(m["params"])
    table("Training state (unsharded) — 16 bytes/param", state)

    print(f"\nPer-GPU state under ZeRO (G = {G}):")
    print("-" * 70)
    print(f"  {'stage':<10}{'shards':<34}{'per-GPU state':>16}  {'fits 80GB?':>10}")
    shard_desc = {0: "nothing (plain DP)", 1: "optimizer state (12B/p)",
                  2: "+ gradients (2B/p)", 3: "+ weights (2B/p)  [= FSDP]"}
    for stage in range(4):
        per_gpu = sum(zero_shard(state, G, stage).values())
        fits = "yes" if per_gpu < 75 * GiB else "NO"
        print(f"  ZeRO-{stage:<5}{shard_desc[stage]:<34}"
              f"{per_gpu / GiB:>12.1f} GiB  {fits:>10}")

    # activations per GPU (each DP rank runs its own microbatch)
    act_ckpt = activation_bytes(seq, batch, m["d_model"], m["n_heads"],
                                m["n_layers"], checkpointing=True)
    per_gpu3 = sum(zero_shard(state, G, 3).values())
    table(f"\nZeRO-3 per-GPU total (seq {seq}, microbatch {batch}, "
          "full checkpointing)", {
        "sharded state (16B/param / 64)": per_gpu3,
        "activations (checkpointed)": act_ckpt,
    })
    print("  + comms buffers/fragmentation overhead in practice; ZeRO-3 pays")
    print("    ~1.5x DP communication (extra weight all-gather in backward).")


# ---------------------------------------------------------------------------
# Case 3: 7B inference, batch 32, 4k context (the canonical KV question)
# ---------------------------------------------------------------------------
def case_3():
    m = LLAMA_7B
    seq, batch = 4096, 32
    print("\n" + "=" * 70)
    print("CASE 3 — 7B inference, batch 32, context 4096 (KV-cache math)")
    print("=" * 70)
    d_head = m["d_model"] // m["n_heads"]
    per_tok_layer = 2 * m["n_kv_heads"] * d_head * 2
    per_tok = per_tok_layer * m["n_layers"]
    print(f"\n  per token per layer: 2(KV) x {m['n_kv_heads']}x{d_head} x "
          f"2B(fp16) = {per_tok_layer / 1024:.0f} KiB")
    print(f"  per token, {m['n_layers']} layers:"
          f"                    = {per_tok / 1024:.0f} KiB")

    kv = kv_cache_bytes(seq, batch, m["n_layers"], m["n_kv_heads"],
                        m["n_heads"], m["d_model"])
    weights_fp16 = m["params"] * 2
    table(f"Inference memory, batch {batch} @ {seq} ctx", {
        "weights (fp16)": weights_fp16,
        "KV cache (MHA, fp16)": kv,
    })
    print(f"\n  KV cache is {kv / weights_fp16:.1f}x the weights. Total "
          f"{(kv + weights_fp16) / GiB:.0f} GiB vs {80 * GB / GiB:.1f} GiB "
          "on an '80 GB' A100: does NOT fit.")

    # what the standard fixes buy
    kv_gqa = kv_cache_bytes(seq, batch, m["n_layers"], 8,
                            m["n_heads"], m["d_model"])
    kv_gqa_int8 = kv_cache_bytes(seq, batch, m["n_layers"], 8,
                                 m["n_heads"], m["d_model"], bytes_per_el=1)
    table("The standard fixes", {
        "GQA, 8 KV heads (/4)": kv_gqa,
        "GQA + int8 KV (/8)": kv_gqa_int8,
    }, total_label="(alternatives, not summed)")

    # bandwidth-bound decode ceiling (notes.md §5.1)
    bw = 2e12  # A100 HBM, bytes/s
    print("\n  Decode speed ceiling (A100, 2 TB/s): every token streams the "
          "weights once:")
    print(f"    batch 1, fp16:  {bw / weights_fp16:6.0f} tok/s   "
          f"(2e12 / {weights_fp16 / GB:.0f}e9)")
    print(f"    batch 1, int4:  {bw / (m['params'] * 0.5):6.0f} tok/s   "
          "(quantization is a *speed* optimization at batch 1)")


if __name__ == "__main__":
    case_1()
    case_2()
    case_3()
    print("\nDone. Every number above is <5 multiplications — do them on a "
          "whiteboard.")
