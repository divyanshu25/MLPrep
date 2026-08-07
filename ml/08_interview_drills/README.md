# Module 8 - Interview Drills

Modules 1–6 taught you to *derive*. This module trains you to *perform*: the
drills here recreate the actual interview formats reported at OpenAI, Anthropic,
and Google DeepMind in 2024–2026, as timed hands-on exercises. Nothing new is
taught here - every concept lives in an earlier module. What's new is the clock.

## Contents

| File | What it is |
|------|-----------|
| [`buggy_transformer.py`](buggy_transformer.py) | The flagship drill: a working transformer LM with 7 planted bugs. Find and fix them, then add KV caching. |
| [`buggy_transformer_solutions.md`](buggy_transformer_solutions.md) | Bug-by-bug solutions, the KV-cache implementation, and the general debugging playbook. **Do not open until done.** |
| [`drills.md`](drills.md) | 10 timed implementation drills with full reference solutions folded away. |
| [`rapid_fire_quiz.md`](rapid_fire_quiz.md) | 40 rapid-fire definition questions, DeepMind style. |
| [`debugging_scenarios.md`](debugging_scenarios.md) | 12 production-debugging scenario cards: symptom → diagnosis tree. |

## The closed-book protocol

This module only works if you treat it as deliberate practice, not reading
material. For every drill:

1. **Read the reference** - the relevant implementation in modules 3–4 (e.g.
   `attention_scratch.py`, `mini_gpt.py`, `micrograd_plus.py`). Understand it
   until you could explain every line.
2. **Close it.** Close the file, close the notes, close the solution.
3. **Re-implement, timed.** Set a real timer for the drill's time box. Write
   code in an empty file. Run it. No peeking - in the real round there is
   nothing to peek at.
4. **Diff against the reference.** Every divergence is either a style choice
   (fine) or a gap (that's your study list for tomorrow). Log the gaps.
5. **Repeat in 2–3 days.** A drill is "done" when you finish inside the time
   box twice, on different days, without looking anything up.

The failure mode to avoid: reading a solution, nodding, and marking the drill
complete. Recognition is not recall, and interviews test recall under pressure.

## Which drill simulates which round

| Drill | Company / round simulated | Time box |
|-------|---------------------------|----------|
| `buggy_transformer.py` | **OpenAI** - the most-reported ML round: ~300-line PyTorch transformer with planted bugs; fix, then extend with KV caching | 45 min + 30 min extension |
| drills.md #1 attention in numpy | OpenAI / Anthropic - "implement attention from scratch" | 25 min |
| drills.md #2 BPE tokenizer | **Anthropic** - build BPE train/encode/decode from scratch | 40 min |
| drills.md #3 sampling (top-k/top-p/temperature) | Anthropic - sampling from logits | 20 min |
| drills.md #4 manual-backward logistic regression | General ML screen - "no autograd allowed" | 25 min |
| drills.md #5 stable softmax + cross-entropy | **DeepMind** - "code must run", numerics focus | 15 min |
| drills.md #6 k-means | Classical-ML screen (all three) | 20 min |
| drills.md #7 AUC from scores | Metrics screen; pairs with module 5 | 15 min |
| drills.md #8 LRU cache | **OpenAI** - the SWE-style staple | 20 min |
| drills.md #9 progressive KV store | **Anthropic** - CodeSignal-style multi-level build | 45 min |
| drills.md #10 batched inference queue | Anthropic / systems-flavored coding; concurrency thinking | 30 min |
| rapid_fire_quiz.md | **DeepMind** - rapid-fire math/definitions quiz | ~2 min per answer, spoken aloud |
| debugging_scenarios.md | OpenAI / Anthropic - "here's a symptom, debug it" discussion | 5 min per card, spoken aloud |

Suggested schedule: one big drill (flagship, BPE, or KV store) plus two small
drills per sitting; the quiz and scenario cards on alternate days, out loud.
Speaking the answer is the drill - knowing it silently doesn't count.

## Self-scoring rubric

Score each drill 1–4 on three axes. Interviewers score you on all three
simultaneously - most candidates only train the first.

**Correctness** - does it run and is it right?
- 4: runs first or second try; edge cases handled; you wrote a sanity test unprompted.
- 3: correct after debugging within the time box.
- 2: core idea right, but a bug survived the time box (wrong axis, off-by-one, missing scale).
- 1: didn't reach running code.

**Speed** - where was the clock when it worked?
- 4: done with time to spare, used the spare time to test or extend.
- 3: done at the buzzer.
- 2: needed +25% over the box.
- 1: needed a second sitting.

**Communication** - what an interviewer *listens for* while you code:
- You state the plan before typing ("three parts: scores, causal mask, weighted sum").
- You annotate shapes as you go and say invariants out loud ("rows now sum to 1").
- When stuck, you narrate hypotheses instead of going silent ("loss is flat at
  ln(V), so logits carry no signal - checking the embedding path first").
- You name tradeoffs unprompted ("dict copy is O(n) per transaction; an undo
  log makes it O(writes) - I'll do the log").
- 4 = all of the above unprompted; 3 = plan + shapes; 2 = narrates only when
  asked; 1 = silent coding.

A drill counts as passed at 3/3/3. The flagship counts as passed when you find
all 7 bugs in 45 minutes *and* your KV-cached generation produces identical
output to the uncached version (greedy decoding, same seed).
