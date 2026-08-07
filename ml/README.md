# ML Interview Prep - From First Principles

A self-contained curriculum for machine learning interviews. Every concept is built
from scratch - derivations done by hand, code written in plain numpy before any
framework, and nothing justified with "it can be shown". If you can't rebuild an
idea from nothing, you don't own it yet; the goal of this material is that you own it.

Every script runs on CPU in seconds-to-minutes with no downloads:

```bash
uv sync                                   # once
uv run python ml/03_neural_networks/micrograd_plus.py   # any script, like this
```

## The modules

| # | Module | What you'll be able to derive from scratch |
|---|--------|--------------------------------------------|
| 1 | [Foundations](01_foundations/notes.md) | Gradients, chain rule, Jacobians; Bayes rule; why minimizing cross-entropy *is* maximum likelihood; CLT and estimator bias/variance |
| 2 | [Classical ML](02_classical_ml/notes.md) | Linear/logistic regression (normal equation, gradient, MLE view); L1/L2 as priors; SVMs and the kernel trick; trees, entropy, bagging/boosting; k-means; PCA from variance maximization |
| 3 | [Neural Networks](03_neural_networks/notes.md) | Backprop node-by-node on real numbers (micrograd style); softmax+CE gradient = p − y; Xavier/Kaiming init derived from variance tracking; SGD/momentum/Adam; BatchNorm/LayerNorm; dropout |
| 4 | [Deep Learning](04_deep_learning/notes.md) | Convolution from locality+weight-sharing; BPTT and why RNN gradients explode; attention derived from "weighted average by similarity", the √d_k scaling, a full GPT trained in the file |
| 5 | [Evaluation](05_evaluation/notes.md) | Every metric from the confusion matrix; ROC/AUC built by hand and its rank-statistic meaning; bias–variance decomposition in full; calibration; leakage and honest cross-validation |
| 6 | [ML Systems](06_ml_systems/notes.md) | The design-interview framework; recsys/fraud/search walkthroughs with real numbers; training/serving skew; drift and monitoring; latency and hardware back-of-envelope math |
| 7 | [The LLM Stack](07_llm_stack/notes.md) | Chinchilla scaling laws with $C=6ND$ derived; the 16-bytes/param training-memory arithmetic and ZeRO/TP/PP; RLHF→DPO derived step by step, GRPO; LoRA/QLoRA math; inference (KV cache, speculative decoding proved exact, FlashAttention, vLLM); LLM evals |
| 8 | [Interview Drills](08_interview_drills/README.md) | Timed practice in the formats labs actually use: a transformer with 7 planted bugs to find in 45 min (+ KV-cache extension), 10 timed coding drills, a rapid-fire definitions quiz, 12 debugging-scenario cards |

Each module directory contains:

- **`notes.md`** - the concepts, derived from first principles, ending with a bank of
  common interview questions and concise answers.
- **`*.py`** - runnable from-scratch implementations (numpy first; sklearn/torch only
  to verify our math against a reference). Read them top to bottom - the comments
  carry the derivations.
- **`exercises.md`** - problems from easy to hard, each with a full solution folded
  under a `<details>` block. Attempt before unfolding.

Companion material already in the repo: [src/roc_curve.ipynb](../src/roc_curve.ipynb)
(ROC/AUC notebook) and [src/gradient_explosion.py](../src/gradient_explosion.py)
(exploding gradients in RNNs - pairs with module 3's vanishing-gradient demos).

## What each company actually tests (2024–2026 loop reports)

Compiled from firsthand interview reports (1point3acres, Blind, Glassdoor, Exponent,
interviewing.io, candidate writeups). Use it to weight your prep, not to skip modules  - 
fundamentals questions appear everywhere as follow-ups.

| Company | Fingerprint | Prep emphasis |
|---|---|---|
| **OpenAI** (RE/MTS) | Speed-coding with a high execution bar; the signature ML round is *debug a buggy transformer, then implement KV caching*; a stats/information-theory round; project deep-dive presentation | Module 8's `buggy_transformer.py` closed-book, module 4, module 7 inference sections, module 1 probability |
| **Anthropic** (MTS/RE) | Progressive multi-level build (the CodeSignal "bank/KV-store" style) rewarding tests-first iteration; concurrency; theory on scaling laws, RL/RLHF, Constitutional AI; a values/safety round that is a genuine gate | Module 8 drills 9–10, module 7 post-training + scaling laws, rehearsed safety-motivation answers |
| **Google DeepMind** (RE/RS) | Rapid-fire math-definitions quiz (eigen-things, SVD, autodiff modes, stable cross-entropy); code must run by end of round; distributed-training and evals-infrastructure design; paper defense for RS | Module 8's `rapid_fire_quiz.md`, modules 1–3 derivations, module 7 distributed training + evals |
| **Google** (broad MLE) | 2–3 DSA rounds plus ML fundamentals and the L5+ gate: end-to-end ML system design | Modules 2 and 5 for fundamentals, module 6 for the design round, separate DSA practice |

LeetCode-style DSA practice lives in its own top-level folder - see [dsa/](../dsa/README.md)
for pattern-first notes, tested solutions, and timed drills (DeepMind/Google still
require classic DSA; OpenAI/Anthropic grade practical speed-coding). One thing this
repo deliberately does not cover: your own project deep-dive / paper defense
(prepare a 10-slide job talk on your past work and rehearse defending it).

## How to study (a 6-week plan)

**Week 1 - Foundations + Classical ML (modules 1–2).**
Do the derivations on paper *before* reading them. The exercises here are the ones
interviewers actually ask ("derive the gradient of logistic regression", "why does
L1 give sparsity?"). Run `linear_models_scratch.py` and make sure you could rewrite
its backward pass from memory.

**Week 2 - Neural Networks (module 3).**
The core week. Work through `micrograd_plus.py` line by line, then close it and
re-implement the `Value` class yourself - this single exercise covers ~half of all
deep-learning interview questions, because backprop *is* the interview question.
Then `mlp_numpy.py` for the vectorized version.

**Week 3 - Deep Learning + Evaluation (modules 4–5).**
Hand-compute attention on the 3-token example in the notes before reading the code.
Train `mini_gpt.py` and watch the samples improve. For evaluation, build the ROC
curve from the 6-example table by hand - being fast and correct at confusion-matrix
arithmetic is table stakes.

**Week 4 - ML Systems (module 6).**
Read the two mock transcripts in `design_walkthroughs.md` out loud, playing the
candidate. Practice the latency and hardware back-of-envelope math until the
estimation constants are automatic.

**Week 5 - The LLM Stack (module 7).**
The frontier-lab week. Work the memory math by hand before running `memory_math.py`
to check yourself - "walk me through training memory for a 70B model" is now a
canonical question. Derive DPO from the RLHF objective on paper once; run the
speculative-decoding demo and make sure you can state *why* it's exact.

**Week 6 - Drills (module 8) + mock interviews.**
Now switch from learning to retrieval practice, which is the part that transfers to
a live interview. Do `buggy_transformer.py` cold with a 45-minute timer. Work through
the timed drills closed-book, diffing against the reference solutions after. Run the
rapid-fire quiz until every answer is 2 sentences and instant. Take the debugging
scenario cards one per day. Then take each module's `exercises.md` cold, under time
pressure. Finish with module 6's `checklists.md` the night before.

## The practice system

The schedule above says what to learn; [practice/](../practice/README.md) says how
to keep it retrievable: a spaced-repetition queue (`practice/review.py`) that
re-schedules every miss at expanding intervals, unlabeled mixed sets because
interview problems arrive without a module name attached, ~160 flashcards distilled
from the modules' question banks, and a mock-interview kit for the final two weeks.
Start feeding the queue from week 1; everything else switches on in week 2.

One rule that changes how you use `exercises.md`: **every exercises file gets a
scheduled cold second pass 1–2 weeks after the first, via the queue.** The moment
you unfold a solution, that item stops being a retrieval test and becomes a worked
example - re-doing it the same week only confirms recognition. Queue it
(`uv run python practice/review.py add ml03/ex4 ml/03_neural_networks/exercises.md`)
and the second pass arrives after enough forgetting that solving it again is
evidence you own it, not that you remember Tuesday.

## The three rules of this material

1. **Derive, don't memorize.** In an interview, a memorized fact collapses under one
   follow-up question; a derivation doesn't. Every formula here comes with its "why".
2. **Tiny numbers first.** Before any general formula, work a 2×2 example with actual
   numbers. If you can't do the tiny case by hand, the general case is an illusion.
3. **Code is the proof.** Every claim in the notes is backed by a script you can run.
   When a derivation and an experiment disagree, the experiment wins - go find the bug.
