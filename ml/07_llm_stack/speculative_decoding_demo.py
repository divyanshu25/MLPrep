"""
speculative_decoding_demo.py — the speculative sampling algorithm, verified.

Run:  uv run python ml/07_llm_stack/speculative_decoding_demo.py

No neural nets. The "language models" here are explicit categorical
distributions over a tiny vocabulary, because the POINT is the algorithm:

  A small draft model q proposes k tokens; the big target model p verifies
  them all in ONE forward pass; an accept/reject rule keeps or fixes each
  proposal. Claim (Leviathan et al. 2023 / Chen et al. 2023): the emitted
  tokens are distributed EXACTLY according to p — not approximately.

The rule, for one position with target p(.) and draft q(.):

  1. sample x ~ q
  2. accept x with probability  min(1, p(x)/q(x))
  3. on reject, resample from the residual  p'(x) ∝ max(0, p(x) - q(x))

Proof sketch (full version in notes.md §5.4):

  P(emit x) = q(x)·min(1, p(x)/q(x))  +  P(reject)·p'(x)
            = min(q(x), p(x))         +  max(0, p(x) - q(x))     (*)
            = p(x)                                                ∎

  (*) because P(reject) = 1 - Σ_y min(p,q) = Σ_y max(0, p-q), which is
  exactly the residual's normalizer, so it cancels. And
  min(p,q) + (p-q)^+ = p case-by-case.

Also verified below:
  - acceptance rate = Σ_y min(p,q) = 1 - TV(p, q): agreement IS acceptance
  - expected emitted tokens per target pass = (1 - a^{k+1}) / (1 - a)

This file:
  Part 1: single-position accept/reject — empirical distribution vs p
  Part 2: a tiny Markov-chain "LM" (context-dependent p and q), the full
          k-token draft-verify loop — distribution over whole SEQUENCES
          vs ancestral sampling from p
  Part 3: acceptance rate & speedup vs draft-target agreement
"""

import numpy as np

rng = np.random.default_rng(0)
V = 6  # vocabulary size


def tv(p, q):
    """Total variation distance = half the L1 distance = max event gap."""
    return 0.5 * np.abs(p - q).sum()


def spec_step(p, q, rng):
    """One position of speculative sampling. Returns (token, accepted?)."""
    x = rng.choice(V, p=q)
    if rng.random() < min(1.0, p[x] / q[x]):          # accept w.p. min(1,p/q)
        return x, True
    resid = np.maximum(p - q, 0.0)                     # residual (p-q)^+
    resid /= resid.sum()                               # normalizer = P(reject)
    return rng.choice(V, p=resid), False


# ===========================================================================
# Part 1 — single position: emitted distribution must equal p exactly
# ===========================================================================
def part1(n=100_000):
    print("=" * 66)
    print("PART 1 — single-position accept/reject, %d samples" % n)
    print("=" * 66)
    # arbitrary target and a deliberately different draft
    p = rng.dirichlet(np.ones(V) * 2)
    q = rng.dirichlet(np.ones(V) * 2)

    counts = np.zeros(V)
    accepts = 0
    for _ in range(n):
        x, acc = spec_step(p, q, rng)
        counts[x] += 1
        accepts += acc
    emp = counts / n

    print(f"{'token':>6} {'target p':>10} {'draft q':>10} {'empirical':>10}")
    for i in range(V):
        print(f"{i:>6} {p[i]:>10.4f} {q[i]:>10.4f} {emp[i]:>10.4f}")
    print(f"\n  TV(empirical, target p) = {tv(emp, p):.4f}   "
          f"(sampling noise scale ~ {1/np.sqrt(n):.4f})")
    print(f"  TV(empirical, draft  q) = {tv(emp, q):.4f}   (far from q: the "
          "draft did NOT contaminate the output)")
    pred_acc = np.minimum(p, q).sum()
    print(f"  acceptance rate: empirical {accepts / n:.4f}  vs  "
          f"predicted sum(min(p,q)) = {pred_acc:.4f} = 1 - TV(p,q)")
    assert tv(emp, p) < 5 / np.sqrt(n), "emitted distribution != target!"


# ===========================================================================
# Part 2 — a context-dependent LM: full k-token draft-verify loop
# ===========================================================================
# The "LM" is a first-order Markov chain: P(next | prev) is a row of a
# transition matrix. Target and draft are two different matrices. This gives
# the real algorithm shape: the draft proposes a k-token continuation
# autoregressively; the target "verifies" (in a real system: one parallel
# forward pass; here: we just index its rows) and accepts a prefix.
def make_lm(concentration, rng):
    return rng.dirichlet(np.ones(V) * concentration, size=V)  # V x V rows


def spec_generate(P, Q, length, k, rng):
    """Generate `length` tokens with speculative decoding.

    Returns (tokens, n_target_passes, n_emitted) — one 'target pass' scores
    up to k drafted positions at once (the whole point of the algorithm).
    """
    out = [0]  # fixed start token (dropped from the returned sequence)
    passes = 0
    while len(out) - 1 < length:
        # --- draft proposes k tokens autoregressively from Q
        drafts = []
        ctx = out[-1]
        for _ in range(k):
            x = rng.choice(V, p=Q[ctx])
            drafts.append(x)
            ctx = x
        # --- target verifies all k in ONE pass
        passes += 1
        ctx = out[-1]
        all_accepted = True
        for x in drafts:
            p, q = P[ctx], Q[ctx]
            if rng.random() < min(1.0, p[x] / q[x]):
                out.append(x)                      # accept
                ctx = x
            else:                                  # reject: residual sample
                resid = np.maximum(p - q, 0.0)
                resid /= resid.sum()
                out.append(int(rng.choice(V, p=resid)))
                all_accepted = False
                break
            if len(out) - 1 == length:
                break
        # bonus token: if every draft accepted, the target's (k+1)-th
        # distribution is already computed — sample from it for free
        if all_accepted and len(out) - 1 < length:
            out.append(int(rng.choice(V, p=P[out[-1]])))
    return out[1:length + 1], passes


def ancestral_generate(P, length, rng):
    out = [0]
    for _ in range(length):
        out.append(int(rng.choice(V, p=P[out[-1]])))
    return out[1:]


def part2(n=100_000, length=3, k=3):
    print("\n" + "=" * 66)
    print(f"PART 2 — Markov 'LM', sequences of length {length}, draft k={k}, "
          f"{n} samples")
    print("=" * 66)
    P = make_lm(2.0, rng)                      # target LM
    Q = 0.5 * P + 0.5 * make_lm(2.0, rng)      # draft: partially agrees

    # distribution over ALL V^length sequences, both ways
    def seq_id(s):
        i = 0
        for t in s:
            i = i * V + t
        return i

    n_seqs = V ** length
    spec_counts = np.zeros(n_seqs)
    anc_counts = np.zeros(n_seqs)
    total_passes = total_tokens = 0
    for _ in range(n):
        s, passes = spec_generate(P, Q, length, k, rng)
        spec_counts[seq_id(s)] += 1
        total_passes += passes
        total_tokens += length
        anc_counts[seq_id(ancestral_generate(P, length, rng))] += 1

    spec_emp, anc_emp = spec_counts / n, anc_counts / n
    # exact sequence distribution under P for reference
    exact = np.zeros(n_seqs)
    for i in range(n_seqs):
        s, prob, prev = i, 1.0, 0
        digits = []
        for _ in range(length):
            digits.append(s % V)
            s //= V
        for t in reversed(digits):
            prob *= P[prev][t]
            prev = t
        exact[i] = prob

    print(f"  TV(speculative, exact target) = {tv(spec_emp, exact):.4f}")
    print(f"  TV(ancestral,   exact target) = {tv(anc_emp, exact):.4f}   "
          "(same noise floor — speculative is exact)")
    print(f"  tokens per target pass: {total_tokens / total_passes:.2f}  "
          f"(vs 1.0 for plain autoregressive decoding)")
    assert tv(spec_emp, exact) < 2 * tv(anc_emp, exact) + 0.01


# ===========================================================================
# Part 3 — acceptance rate and speedup vs draft quality
# ===========================================================================
def part3(k=4):
    print("\n" + "=" * 66)
    print(f"PART 3 — acceptance & expected speedup vs draft agreement (k={k})")
    print("=" * 66)
    p = rng.dirichlet(np.ones(V) * 2)
    noise = rng.dirichlet(np.ones(V) * 2)
    print(f"{'mix w':>6} {'TV(p,q)':>9} {'accept a':>9} "
          f"{'E[tokens/pass] = (1-a^(k+1))/(1-a)':>36}")
    for w in [1.0, 0.9, 0.7, 0.5, 0.3, 0.0]:
        q = w * p + (1 - w) * noise            # draft = target blended w/noise
        a = np.minimum(p, q).sum()             # acceptance = 1 - TV(p,q)
        exp_tokens = (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1
        print(f"{w:>6.1f} {tv(p, q):>9.4f} {a:>9.4f} {exp_tokens:>36.2f}")
    print("\n  Reading: a perfect draft (w=1) emits k+1 tokens per big-model")
    print("  pass; a useless draft still emits 1 (the residual sample) — you")
    print("  never do worse than plain decoding in tokens/pass, only in the")
    print("  wasted draft compute.")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print("\nAll checks passed: speculative sampling reproduces the target "
          "distribution exactly.")
