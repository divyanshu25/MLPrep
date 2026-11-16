# Post-training RL: rewards and training in practice

Why this module: module 9 derives the policy gradient and module 7 §3 gives
the RLHF, DPO and GRPO objectives. Interviews for post-training RL roles
assume both and then ask what happens when you run them: what the reward
should be, how the model will cheat it, how to tell from the training
curves that something is wrong, and how the rollout system is put
together. They also include a coding round (write the GRPO loss) and a
system design round (size an RL cluster). This module covers all of these. It refers back to module 7
§3 and module 9 §5 for the algorithms instead of repeating them.

## 1. The training loop

One step of RL post-training has four parts:

1. **Sample prompts** from the prompt set.
2. **Generate rollouts**: the current policy writes $G$ completions per
   prompt. This runs on an inference engine such as vLLM.
3. **Score** each completion with the reward: run the unit tests, check the
   math answer, or call a reward model.
4. **Update**: compute advantages (for GRPO, each reward minus its group's
   mean, divided by the group's standard deviation), then take a few
   gradient steps on the clipped objective. Copy the new weights to the
   inference engine and repeat.

<svg viewBox="0 0 560 170" role="img" aria-label="The RL post-training loop: a prompt set feeds the inference engine, which generates G completions per prompt; the reward step scores them; the learner computes advantages and updates the weights; the new weights are copied back to the inference engine." style="max-width:580px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="pl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.5">
    <rect x="10" y="40" width="100" height="50" rx="7"/>
    <rect x="150" y="40" width="120" height="50" rx="7"/>
    <rect x="310" y="40" width="100" height="50" rx="7"/>
    <rect x="450" y="40" width="100" height="50" rx="7"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="60" y="62">prompts</text>
    <text x="60" y="78" font-size="10.5" fill-opacity="0.7">curated set</text>
    <text x="210" y="62">generate</text>
    <text x="210" y="78" font-size="10.5" fill-opacity="0.7">G per prompt (vLLM)</text>
    <text x="360" y="62">reward</text>
    <text x="360" y="78" font-size="10.5" fill-opacity="0.7">tests, checker, RM</text>
    <text x="500" y="62">learner</text>
    <text x="500" y="78" font-size="10.5" fill-opacity="0.7">advantages, update</text>
  </g>
  <g stroke="currentColor" stroke-opacity="0.6" fill="none">
    <line x1="110" y1="65" x2="148" y2="65" marker-end="url(#pl-a)"/>
    <line x1="270" y1="65" x2="308" y2="65" marker-end="url(#pl-a)"/>
    <line x1="410" y1="65" x2="448" y2="65" marker-end="url(#pl-a)"/>
    <path d="M 500 90 L 500 130 L 210 130 L 210 92" marker-end="url(#pl-a)"/>
  </g>
  <text x="355" y="148" text-anchor="middle" fill="#f59e0b" font-size="11">copy new weights to the inference engine</text>
</svg>

Two facts about this loop drive most of the engineering:

- **Generation takes most of the time.** Writing completions token by
  token is slow, and a batch waits for its longest completion. In
  reasoning RL, generation commonly takes more than half of each step's
  wall-clock time.
- **The data comes from the model being trained.** Unlike pretraining,
  there is no fixed dataset. If the reward is wrong, the model produces
  more of the wrong behavior, and that becomes the next batch of training
  data.

## 2. Kinds of reward

#### Verifiable rewards

A program checks the answer: exact match on a math answer, unit tests on
code, a schema check on JSON. This is often called RLVR (RL with verifiable
rewards). It is the main signal for reasoning models because it is cheap
and a model cannot argue its way past it.

The verifier is still code, and it has bugs. Three common ones:

- **Extraction fails.** The model writes "the answer is $\frac{1}{2}$" and
  the checker expects 0.5. A correct answer gets reward 0, and the model
  learns to avoid a format that was fine.
- **The check is too loose.** A checker that accepts a completion if the
  correct number appears anywhere in it rewards a completion that lists
  every number from 1 to 100.
- **The tests are weak.** Code that special-cases the three test inputs
  passes all three tests.

Before training, run the verifier on a few hundred known-good and
known-bad completions and read the cases where it disagrees with you.

#### Learned reward models

For tasks with no checker (helpfulness, writing quality, safety), a reward
model is trained on human preference pairs with the Bradley–Terry loss
(module 7 §3.1). It outputs a score for any completion, but it is accurate
only on completions similar to the ones it was trained on. The policy
moves away from those during training, and the scores become unreliable
(§3).

#### LLM judges and rubrics

A strong model grades completions against a written rubric, for example
"1 point if the answer cites the relevant clause, 1 point if it states the
deadline". Rubrics make the grading criteria explicit and easy to change
without collecting new labels. The judge has its own biases (it prefers
longer answers, answers in the first position, and answers that resemble
its own style), and the policy can learn to exploit them, including by
writing text addressed to the judge.

#### Outcome vs process rewards

An **outcome reward** scores only the final answer. A **process reward
model** (PRM) scores each reasoning step. A PRM gives denser signal, so the
model learns which step went wrong, but step labels are expensive and the
PRM can be hacked like any learned model. Most recent reasoning RL uses
outcome rewards with many samples per prompt, and uses PRMs mainly for
reranking samples at inference time.

#### Combining rewards

Real rewards usually have several parts. For a coding agent:

$$r = \underbrace{\mathbb{1}[\text{tests pass}]}_{\text{task}} \;-\; 0.1 \cdot \mathbb{1}[\text{bad format}] \;-\; \lambda \cdot \text{length penalty}$$

Two rules keep this under control:

- **Gate, don't add, when one part is required.** If a completion must be
  valid JSON, give it reward 0 when it isn't, whatever else it does. With a
  sum, the model can trade correctness on the required part for points on
  the others.
- **Keep auxiliary terms small.** If a format bonus is as large as the task
  reward, the model can earn most of the available reward from formatting
  alone.

#### Reward shaping

Adding intermediate rewards ("+0.1 for calling the right tool") gives more
signal but changes which policy is optimal: the model may call the tool
repeatedly to collect the bonus. Shaping of the form
$F(s, s') = \gamma\,\Phi(s') - \Phi(s)$, for any function $\Phi$ of the
state, provably keeps the same optimal policy (potential-based shaping, Ng
et al. 1999): over an episode the terms telescope to
$\gamma^T\Phi(s_T) - \Phi(s_0)$, which does not depend on the actions in
between. Ad hoc bonuses don't have this guarantee.

## 3. Reward hacking

The reward is a measurement of what you want, not the thing itself.
Optimizing the measurement hard enough finds the places where the two
differ (Goodhart's law). Examples that show up in practice:

| Reward | What the policy learned |
|---|---|
| Reward model for helpfulness | longer answers, flattery, agreeing with the user (sycophancy) |
| Unit tests | special-casing test inputs, editing or deleting the tests, exiting early with a success code |
| LLM judge | confident tone, rubric keywords, instructions written to the judge |
| Math answer checker | output formats the parser misreads as correct |
| "Answer contains the number" | listing many numbers |

#### The over-optimization curve

Gao et al. (2022) trained policies against a proxy reward model and scored
them with a larger "gold" reward model standing in for human judgment. As
the policy moved further from its starting point (measured by KL), the
proxy reward kept rising, while the gold reward rose, peaked and then fell.

<svg viewBox="0 0 420 225" role="img" aria-label="Reward plotted against KL from the starting policy. The proxy reward keeps rising. The gold reward rises, peaks, then falls. The gap between them after the peak is reward hacking." style="max-width:440px;width:100%;height:auto;font-size:12px;font-family:inherit">
  <defs>
    <marker id="oo-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
  </defs>
  <g stroke="currentColor" stroke-opacity="0.6" fill="none">
    <line x1="50" y1="180" x2="380" y2="180" marker-end="url(#oo-a)"/>
    <line x1="50" y1="180" x2="50" y2="20" marker-end="url(#oo-a)"/>
  </g>
  <path d="M 50 170 C 120 110, 220 70, 360 40" fill="none" stroke="#0a84ff" stroke-width="2.5"/>
  <path d="M 50 170 C 110 100, 160 85, 200 90 S 300 140, 360 160" fill="none" stroke="#f43f5e" stroke-width="2.5"/>
  <line x1="190" y1="180" x2="190" y2="86" stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="4 4"/>
  <g fill="currentColor" font-size="11">
    <text x="215" y="218" text-anchor="middle">KL from the starting policy</text>
    <text x="40" y="100" text-anchor="middle" transform="rotate(-90 40 100)">reward</text>
    <text x="190" y="196" text-anchor="middle" font-size="10" fill-opacity="0.7">best checkpoint</text>
  </g>
  <text x="410" y="26" text-anchor="end" fill="#0a84ff" font-size="11">proxy (what you optimize)</text>
  <text x="250" y="160" fill="#f43f5e" font-size="11">gold (what you want)</text>
</svg>

The practical consequence: the training reward going up does not show
that the model is getting better. Only a measurement the policy was not
trained on can show that.

#### Mitigations

- **KL penalty to the reference model** (module 7 §3.2). It limits how far
  the policy can move, which keeps it in the region where the reward model
  is accurate. A larger $\beta$ stops earlier on the curve above.
- **Reward model ensembles.** Train several reward models and use their
  mean minus their disagreement, or their minimum. A completion that
  exploits one model's error rarely fools all of them.
- **Length control.** Penalize length directly, or compare completions of
  similar length, so the model can't gain reward just by writing more.
- **Fix the verifier.** When you find a hack, close it in the checker
  (sandbox the tests, make test files read-only, parse the answer
  strictly) and keep that case as a regression test for the checker.
- **Held-out evals and reading samples.** Evaluate checkpoints on a
  measurement the policy never trained against, and read a sample of
  completions at every evaluation. Most hacks are obvious the first time
  someone reads the output.
- **Early stopping.** Keep checkpoints and pick by held-out eval, not by
  training reward.

## 4. Prompts and difficulty

GRPO learns only from groups whose rewards differ. If all $G$ completions
for a prompt get the same reward, every advantage is 0 and the prompt
contributes nothing. For a prompt the policy solves with probability $p$,
the chance that a group of $G$ is all-same is

$$P(\text{no signal}) = p^G + (1-p)^G$$

| pass rate $p$ | $G = 4$ | $G = 8$ | $G = 16$ |
|---|---|---|---|
| 0.05 | 0.82 | 0.66 | 0.44 |
| 0.1 | 0.66 | 0.43 | 0.19 |
| 0.5 | 0.13 | 0.01 | 0.00 |

So a prompt the model solves 10% of the time, sampled with $G = 8$, gives
no gradient 43% of the time. That is generation compute spent for nothing.
The usual fixes:

- **Filter by pass rate.** Estimate each prompt's pass rate with the
  current model and keep the ones in a middle band (for example 0.1 to
  0.9). Re-estimate as the model improves, since easy prompts become
  solved and stop contributing.
- **Drop zero-signal groups during training** and sample more prompts to
  refill the batch (DAPO calls this dynamic sampling).
- **Larger $G$ for hard prompts**, so at least one sample succeeds.

When a group has exactly one success, that success gets a large advantage.
With $G = 8$ and rewards $[1, 0, 0, 0, 0, 0, 0, 0]$: mean 0.125, standard
deviation 0.33, so the success gets $+2.65$ and each failure $-0.38$. Rare
successes on hard prompts produce the largest updates.

Also check the prompt set for duplicates and for overlap with your eval
sets. Training on eval prompts makes the eval meaningless.

## 5. Details of the loss that change results

The GRPO objective (module 7 §3.4) has several choices that look minor
and visibly change training.

#### Averaging over tokens: the length bias

The original GRPO loss averages each completion's per-token loss over that
completion's length, then averages over completions. With advantage $-1$,
a 100-token wrong answer pushes each of its tokens down by $1/100$, while
a 1000-token wrong answer pushes each token down by only $1/1000$. Long
wrong answers are penalized less per token, so response length drifts up
during training without accuracy improving.

The fix is to sum token losses over the whole batch and divide by one
constant (the total token count, or a fixed maximum), so every token gets
the same weight. Dr. GRPO (Liu et al. 2025) and DAPO both do this.

#### Dividing by the group's standard deviation

Dividing by the standard deviation scales up prompts where rewards barely
differ (nearly all right or nearly all wrong) and scales down prompts with
mixed results. Dr. GRPO argues this upweights very easy and very hard
prompts and removes it, keeping only $r_i - \text{mean}$. Both versions are
in use; know that the choice exists and what it does.

#### Clipping and entropy collapse

PPO clips the ratio $\rho = \pi_\theta/\pi_{old}$ to
$[1-\epsilon, 1+\epsilon]$. Consider a token the policy currently gives
probability 0.01, in a completion with positive advantage. With
$\epsilon = 0.2$ its probability can only rise to 0.012 in this update. A
token at probability 0.9 can rise to 1.0. So clipping lets the policy
sharpen what it already prefers much faster than it can raise rare
alternatives. Over training, entropy (how spread out the token
distribution is) falls, the samples in a group become nearly identical,
groups stop having different rewards, and learning stalls.

DAPO's fix ("clip-higher") uses a larger upper bound than lower bound, for
example $[1 - 0.2,\ 1 + 0.28]$, so low-probability tokens can rise more per
step. Sampling temperature also matters: train with temperature 1.0 so the
samples come from the distribution being optimized.

#### Where the KL term goes

- **In the reward** (InstructGPT-style PPO): each token's reward gets
  $-\beta \log(\pi_\theta/\pi_{ref})$ added.
- **In the loss** (GRPO): a separate term $\beta \cdot \text{KL}$ is added
  to the loss, usually estimated per token with
  $k_3 = (u - 1) - \log u$, where $u = \pi_{ref}/\pi_\theta$. This estimator
  is never negative and has lower variance than $-\log u$.
- **Not at all**: many reasoning RL setups with verifiable rewards drop the
  KL term, since an exact checker can't be exploited the way a reward model
  can, and the KL term slows learning. They rely on clipping and held-out
  evals instead.

## 6. The rollout system

#### Two copies of the model

Generation runs on an inference engine (vLLM, SGLang) because it batches
requests and manages the KV cache. Training runs on a training framework
(FSDP, Megatron). After each update the new weights must be copied from
the trainer to the inference engine. For a large model this copy takes
noticeable time and is a common place for bugs, such as the inference
engine silently still using old weights.

#### Off-policy data and the importance ratio

To keep the GPUs busy, many systems generate the next batch while the
learner trains on the current one (asynchronous RL). The samples then come
from a policy one or more updates old. PPO's ratio
$\pi_\theta/\pi_{old}$ corrects for this, provided $\pi_{old}$ is the
policy that actually generated the samples. The further behind the
generator falls, the larger the ratios get and the more tokens get
clipped.

#### The two engines disagree on probabilities

Even with identical weights, the inference engine and the trainer compute
slightly different token probabilities, because they use different kernels
and numeric precision. So the sampling policy is not exactly the policy
the trainer thinks generated the data. Track the gap: compute both
log-probabilities for the same tokens and log the difference. If it is
large, training can become unstable. A common correction multiplies each
token's loss by the ratio of trainer to sampler probability, capped at a
maximum (truncated importance sampling).

#### Slow completions

A batch takes as long as its longest completion. With completions from 200
to 16,000 tokens, most of the batch sits idle waiting for the long ones.
Fixes: cap the maximum length (and decide what reward a truncated
completion gets: DAPO adds a penalty that grows as length nears the cap),
and in asynchronous systems, start training on completions as they finish.

#### Rewards that run code

Code and agent rewards run model-written code. Run it in a sandbox with no
network access, resource limits and timeouts. Besides safety, this closes
hacks: a model that can edit the test file or read the expected outputs
will learn to.

## 7. Reading the training curves

Log these on every step; most debugging is reading them together.

| Metric | Healthy | Warning sign |
|---|---|---|
| train reward | rises, then flattens | rises fast while held-out eval doesn't |
| held-out eval | rises with reward | flat or falling |
| KL to reference | grows slowly | sudden jumps |
| entropy | falls slowly | drops close to 0 |
| response length | stable, or grows with accuracy | grows steadily with no accuracy gain |
| clip fraction | small, roughly 0.05 to 0.2 | large: updates too big or data too stale |
| zero-signal groups | well below half | most groups: prompts too easy or too hard |
| gradient norm | stable | spikes |

Common patterns and their usual causes:

- **Reward up, eval flat or down.** Reward hacking. Read 20 high-reward
  completions; the hack is usually visible.
- **Reward stuck near 0.** Prompts too hard, or a verifier bug that rejects
  correct answers. Check by hand-scoring 20 completions.
- **Length growing steadily.** Length bias in the loss (§5), or the reward
  model prefers longer answers.
- **Entropy collapses, then reward plateaus.** Groups stopped differing.
  Try clip-higher, a higher temperature, or harder prompts.
- **KL spike or sudden reward collapse.** Learning rate too high, a bad
  batch, stale weights in the inference engine, or a large gap between
  sampler and trainer probabilities.

## 8. Multi-turn and agent RL

In agent RL, a rollout is a conversation: the model writes, calls a tool,
reads the result, writes again, and gets a reward at the end. Three things
change:

- **Mask environment tokens.** Tool outputs are part of the context but not
  actions the model took. Exclude them from the loss, or the model is
  trained to predict tool outputs.
- **Credit assignment over long rollouts.** One reward at the end of fifty
  turns says little about which turn mattered. Outcome rewards with many
  samples, or per-turn rewards where a check exists, help.
- **The environment is part of the system.** It must be reproducible (same
  actions, same results), fast, and safe to run thousands of copies of.
  Building and maintaining environments is a large share of the work.

## 9. Baselines that don't need RL

Before RL, try sampling: generate $N$ completions and keep the best by
reward (best-of-$N$). It needs no training, and it shows how much reward
the current model can already reach. Its distance from the starting
policy has a closed form, $\text{KL} = \log N - (N-1)/N$: 0.64 for
$N = 4$, 1.84 for $N = 16$, 3.18 for $N = 64$. An RL run with a similar
KL should beat best-of-$N$ at that $N$, or the RL is not adding much.

**Rejection sampling fine-tuning** (expert iteration) turns this into
training: sample, keep the completions that pass, run SFT on them, repeat.
It is stable, simple, and a strong baseline for verifiable tasks; RL
usually beats it because it also learns from the failures.

## 10. Designing a reward: an interview recipe

"Design the reward for X" is a common question. Answer it in five steps.

1. **Define success** in one sentence, as you would judge it by hand.
2. **Find what can be checked by a program**, and use that as the main
   signal. Use a judge or reward model only for what can't be checked.
3. **Attack your own reward**: list how a model could get high reward
   without succeeding, and close each case.
4. **Add guardrails**: format gates, length control, a KL penalty if the
   reward is learned.
5. **Say how you'd catch failures**: held-out evals the policy doesn't
   train on, plus reading samples.

Worked example: **an agent that fixes failing GitHub issues.**

1. Success: the fix resolves the issue without breaking anything else.
2. Checkable: the repo's hidden tests for the issue pass, and the existing
   tests still pass. Reward 1 if both, else 0.
3. Attacks: edit or delete the tests (make test files read-only, run the
   original copies); special-case the test inputs (hold some tests out of
   training and check them only at eval); call `sys.exit(0)` (check the
   test runner's actual report, not the exit code); a huge diff that
   rewrites the module (penalize diff size, or have a judge flag it).
4. Guardrails: a timeout per run, a sandbox with no network, a cap on the
   number of turns.
5. Catching failures: a held-out set of issues, plus reading diffs from
   high-reward rollouts every evaluation.

## 11. Coding round: write the GRPO loss

"Implement the GRPO (or PPO, or DPO) loss" is the most common coding
question for these roles. GAE with the PPO loss is drill 12 and the DPO
loss is drill 13 in module 8. Here is GRPO, with the details interviewers
check: token log-probabilities by `gather`, group advantages, the
asymmetric clip, the $k_3$ KL term, and a mask normalized by one constant
for the batch.

```python
import torch
import torch.nn.functional as F

def token_logprobs(logits, tokens):
    """logits (B, T, V) score tokens (B, T): logits[:, t] scores tokens[:, t]."""
    return torch.gather(F.log_softmax(logits, -1), 2, tokens.unsqueeze(-1)).squeeze(-1)

def group_advantages(rewards, G, scale_by_std=True, eps=1e-6):
    """rewards (B,), B = n_prompts * G, each prompt's G samples consecutive."""
    r = rewards.view(-1, G)
    adv = r - r.mean(1, keepdim=True)
    if scale_by_std:                      # Dr. GRPO sets this False
        adv = adv / (r.std(1, unbiased=False, keepdim=True) + eps)
    return adv.view(-1)

def grpo_loss(logp, logp_old, logp_ref, adv, mask, eps_lo=0.2, eps_hi=0.28, beta=0.0):
    """logp, logp_old, logp_ref, mask: (B, T); adv: (B,).
    mask is 1 on generated tokens, 0 on prompt, padding and tool-output tokens."""
    ratio = torch.exp(logp - logp_old)
    a = adv.unsqueeze(1)
    surrogate = torch.min(ratio * a, torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * a)
    per_token = -surrogate
    if beta:
        log_u = logp_ref - logp           # log(pi_ref / pi_theta)
        per_token = per_token + beta * (torch.exp(log_u) - 1 - log_u)   # k3
    return (per_token * mask).sum() / mask.sum()   # one constant: no length bias
```

What to say while writing it:

- **Shift by one.** The model's logits at position $t$ predict token
  $t + 1$. If `logits` come straight from the model on the full sequence,
  pass `logits[:, :-1]` and `tokens[:, 1:]`.
- **`logp_old` and `logp_ref` carry no gradient.** Compute them under
  `torch.no_grad()` (or `.detach()`), once per batch.
- **Work in log space.** Take the ratio as `exp(logp - logp_old)`, never
  as a division of probabilities, which underflow for long sequences.
- **The mask.** Prompt tokens, padding and tool outputs get no loss.
  Dividing by `mask.sum()` over the whole batch avoids the length bias of
  §5; the original GRPO divided each sequence by its own length.
- **Tests you'd write.** At ratio 1 the loss is minus the mean advantage
  over masked tokens; a group with identical rewards gets zero advantage;
  a token with $A > 0$ and ratio above $1 + \epsilon_{hi}$ gets zero
  gradient; the KL term is 0 when the policy equals the reference.

## 12. System design: an RL training system for a 70B model

"Design the infrastructure to run GRPO on a 70B model for math and code"
is a common system design question. Start from the memory arithmetic
(module 7 §2):

- **Training state:** 16 bytes per parameter for weights, gradients and
  Adam state, so $70\text{B} \times 16 = 1.12$ TB, at least 14 GPUs of
  80 GB before activations, sharded with FSDP or ZeRO-3.
- **Reference model** (if $\beta > 0$): 140 GB in bf16, forward pass only.
- **Inference copy:** 140 GB of weights plus the KV cache. For a 70B model
  with 80 layers and 8 KV heads of dimension 128, the cache is
  $2 \times 80 \times 8 \times 128 \times 2$ bytes $= 0.33$ MB per token,
  so 5.4 GB for one 16k-token completion and 2.7 TB for 512 of them at
  once. Rollout memory, not training memory, often sets the cluster size.

Then the decisions an interviewer wants to hear about:

- **Colocated vs separate GPUs.** Colocated: the same GPUs alternate
  between generating and training, and the weights are resharded in
  memory between phases. There is no weight transfer over the network,
  but one phase always waits for the other. Separate pools: generation and
  training run at the same time on different GPUs, the new weights are
  broadcast after each update, and you must balance the pool sizes so
  neither waits.
- **Synchronous vs asynchronous.** Synchronous is exactly on-policy and
  easy to debug. Asynchronous keeps every GPU busy but trains on samples
  from older weights; bound the staleness (for example at most one update
  behind) and log the ratio statistics (§6).
- **Reward workers.** Unit tests and agent environments run on a separate
  CPU cluster in sandboxes, behind a queue, with timeouts. A slow reward
  service stalls the whole loop, so it needs to scale with rollout
  throughput.
- **Stragglers.** Cap completion length, and sort or stream completions so
  the learner doesn't wait for the longest one.
- **What to store.** Every rollout with its prompt, completion, reward
  breakdown, log-probabilities from both engines, and the weight version
  that generated it. This is what you read when debugging (§7).
- **Checkpointing.** Save policy and optimizer state regularly, and keep
  older checkpoints: a hack or collapse found late means restarting from
  before it began.

## 13. Reward models in practice

Module 7 §3.1 gives the loss. In practice:

- **Accuracy is low, and that is normal.** Human labelers agree with each
  other on roughly 70 to 75% of pairs for open-ended tasks, so a reward
  model reaching a similar held-out accuracy is near the limit of the
  labels. Evaluate on held-out pairs, broken down by task type.
- **Train about one epoch.** Reward models overfit the preference data
  quickly; a second epoch usually raises training accuracy and lowers
  held-out accuracy.
- **Remove length as a shortcut.** If longer answers win most pairs, the
  model learns "longer is better". Check accuracy within pairs of similar
  length, and balance the data.
- **Normalize the scores.** Only reward differences are learned, so the
  offset is arbitrary. Subtract the mean score on a reference set so a
  reward of 0 means "typical SFT answer"; this keeps $\beta$ meaningful
  across reward model versions.
- **Retrain on the policy's outputs.** As the policy moves, collect new
  comparisons on its current completions so the reward model stays
  accurate where the policy now is.

## 14. Offline vs online preference training

DPO (module 7 §3.3) trains on a fixed set of pairs. Variants you should be
able to name, each fixing one problem:

- **IPO**: DPO can push the margin between chosen and rejected toward
  infinity when a preference is always the same way. IPO replaces the
  log-sigmoid with a squared loss toward a fixed target margin.
- **KTO**: needs only single completions labeled good or bad, not pairs,
  which matches thumbs-up and thumbs-down data.
- **SimPO**: drops the reference model and uses the length-normalized
  log-probability with a margin.
- **Online (iterative) DPO**: sample new pairs from the current policy,
  label them with a reward model or judge, and run DPO again. This recovers
  much of the on-policy benefit of RL with a simpler training loop.

The general finding: training on samples from the current policy (online)
beats training on a fixed dataset (offline), because the model learns
about the mistakes it actually makes now.

## 15. The reasoning RL recipe

DeepSeek-R1 (2025) is the reference point most interviewers use:

- **R1-Zero**: GRPO from the base model with only a rule-based reward
  (answer correct, plus a format reward). Long chains of thought and
  self-checking appeared without being taught: responses grew longer
  over training as accuracy rose. The outputs mixed languages and were hard to read.
- **R1**: a small SFT "cold start" on readable long-chain examples first,
  then RL with verifiable rewards (plus a language-consistency reward),
  then rejection sampling to build a large SFT set, SFT, and a final RL
  stage covering general helpfulness.
- **Distillation**: fine-tuning small models on R1's outputs worked better
  than running RL on the small models directly. For a small model, SFT on
  a strong model's reasoning is the cheaper route; RL on it adds a smaller
  gain on top.

Things to check when someone claims RL improved reasoning: the base
model's pass@k at large k (RL often raises pass@1 toward what pass@k
already was, rather than solving new problems), contamination of the eval
set, and whether the gain is only better answer formatting.

## 16. Papers to know

| Paper | What to remember |
|---|---|
| InstructGPT (Ouyang et al., 2022) | SFT, reward model, PPO with a per-token KL penalty |
| Scaling laws for RM over-optimization (Gao et al., 2022) | the gold reward rises and falls with KL (§3) |
| Constitutional AI (Bai et al., 2022) | AI feedback from written principles (module 7 §3.5) |
| DPO (Rafailov et al., 2023) | RLHF objective solved in closed form (module 7 §3.3) |
| Let's Verify Step by Step (Lightman et al., 2023) | process rewards beat outcome rewards for reranking math solutions |
| DeepSeekMath (Shao et al., 2024) | introduced GRPO |
| Tülu 3 (Lambert et al., 2024) | open recipe; named RLVR |
| DeepSeek-R1 (2025) | R1-Zero, cold start, distillation (§15) |
| DAPO (Yu et al., 2025) | clip-higher, dynamic sampling, token-level loss, overlong penalty |
| Dr. GRPO (Liu et al., 2025) | the length and standard-deviation biases in GRPO (§5) |

## Common interview questions

1. **What is reward hacking? Give an example.**
   **The policy finds a way to score high reward without doing the
   task.**
   - A coding policy rewarded for passing tests learns to delete the
     failing test.
   - A policy trained against a helpfulness reward model learns that longer
     answers score higher and pads them.
   - It happens because the reward is a measurement that differs from the
     goal in places, and optimization finds those places.

   **If asked:** *How do you detect it?* The training reward rises while a
   held-out eval stays flat; reading high-reward samples confirms it.

2. **Verifiable reward vs reward model: when do you use each?**
   **Use a verifier whenever the answer can be checked by a program; use a
   reward model only when it can't.**
   - Math answers, code with tests, structured output: verifier. It is
     exact and cheap, and harder to hack.
   - Helpfulness, tone, writing quality: reward model or rubric judge,
     with a KL penalty, because the model drifts out of the region where
     the reward model is accurate.

   **If asked:** *Can verifiers be hacked?* Yes, through their bugs: loose
   answer matching, weak tests, editable test files.

3. **Why does GRPO waste compute on very easy and very hard prompts?**
   **If all $G$ samples get the same reward, all advantages are 0 and the
   prompt gives no gradient.**
   - Pass rate 0.1 with $G = 8$: $0.9^8 = 0.43$, so 43% of those groups
     give nothing.
   - Fix: keep prompts with pass rates in a middle band, drop zero-signal
     groups and refill the batch, or raise $G$ for hard prompts.

4. **Response length keeps growing during GRPO training. Why?**
   **Usually the per-completion length normalization in the loss.**
   - With advantage $-1$, a 100-token wrong answer pushes each token down
     by 1/100 and a 1000-token one by only 1/1000, so long wrong answers
     are penalized less.
   - Fix: normalize by one constant over the whole batch (Dr. GRPO, DAPO).

   **If asked:** *Other causes?* A reward model or judge that prefers
   longer answers.

5. **What is entropy collapse and how do you prevent it?**
   **The policy becomes nearly deterministic, samples in a group become
   identical, and learning stops.**
   - Clipping lets a token at probability 0.9 rise to 1.0, but a token at
     0.01 can only rise to 0.012 per step, so the policy sharpens faster
     than it explores.
   - Fixes: a larger upper clip bound (clip-higher), temperature 1.0,
     harder prompts, an entropy bonus.

6. **Why keep a KL penalty to the reference model? Why do some setups drop
   it?**
   **With a learned reward, KL keeps the policy where the reward model is
   accurate; with an exact verifier there is less to protect against.**
   - Against a reward model, more KL eventually means worse true quality
     (the over-optimization curve).
   - With a verifier, KL mainly slows learning, so many reasoning setups
     remove it and rely on clipping and held-out evals.

7. **Your reward goes up but your eval goes down. What do you do?**
   **Assume reward hacking until you've read samples.**
   - Read 20 of the highest-reward completions and find what they have in
     common.
   - Close the hack in the reward, add it as a test case for the reward,
     restart from a checkpoint before the eval started falling.
   - Also check the eval itself for bugs.

8. **Outcome reward vs process reward model?**
   **Outcome rewards score the final answer; process rewards score each
   step.**
   - Process rewards are denser but expensive to label and can be hacked.
   - Most reasoning RL uses outcome rewards with many samples; PRMs are
     used more for reranking at inference.

9. **Why is RL post-training infrastructure hard?**
   **Generation dominates the time, and it runs on a different engine from
   training.**
   - Weights must be copied to the inference engine after every update.
   - The two engines compute slightly different probabilities, so the
     data is slightly off-policy; asynchronous generation makes it more
     so.
   - Batches wait for the longest completion.

   **If asked:** *How do you correct for stale samples?* The PPO ratio
   against the policy that actually sampled them, with capped
   importance weights.

10. **What baseline should an RL run beat?**
    **Best-of-$N$ sampling and rejection-sampling fine-tuning.**
    - Best-of-$N$ needs no training: sample $N$, keep the highest reward.
    - If RL at a given KL can't beat best-of-$N$ at the same KL
      ($\log N - (N-1)/N$), it isn't adding much.

11. **Design a reward for a model that writes SQL from questions.**
    **Execute the query and compare results with a reference query's
    results.**
    - Main signal: result sets match on a test database (order-insensitive
      unless the question asks for an order).
    - Attacks: a query that happens to match on a tiny test database (use
      several databases with different contents); queries that modify data
      (read-only connection); very slow queries (timeout).
    - Guardrail: reward 0 if the SQL doesn't parse.

12. **In agent RL, which tokens get gradient?**
    **Only the tokens the model generated.**
    - Tool outputs and environment messages are masked out of the loss.
    - Otherwise the model is trained to predict tool outputs, which it
      doesn't control.

13. **PPO or GRPO: which would you use?**
    **GRPO for verifiable rewards with many samples per prompt; PPO when
    you need per-token credit or can't afford many samples.**
    - GRPO drops the critic, saving a policy-sized model, and a group mean
      is a good baseline for 0/1 rewards.
    - PPO's value network gives a baseline per token and works with one
      sample per prompt, at the cost of training the critic.

14. **Why does R1 do SFT before RL if R1-Zero worked without it?**
    **RL from the base model worked but produced hard-to-read output and
    started slowly.**
    - A small SFT set of readable long-chain examples fixes the format, so
      RL spends its compute on correctness.
    - It also makes early samples succeed more often, so fewer groups have
      zero signal.

15. **You have a 7B model. RL or distillation for reasoning?**
    **Distill first: SFT on a strong model's reasoning traces.**
    - DeepSeek found this beat running RL on the small model directly.
    - Then RL on top for a further gain, if the budget allows.

16. **How do you know RL taught new reasoning rather than better
    formatting?**
    **Compare against the base model's pass@k and check formatting
    separately.**
    - If base pass@64 already equals the RL model's pass@1, RL mostly
      sharpened what the base could sample.
    - Score with a lenient parser too; a gain that disappears under it was
      formatting.
    - Check the eval set for contamination.

17. **Your reward model reaches 68% held-out accuracy. Is it broken?**
    **Not necessarily: human labelers agree on only about 70 to 75% of
    pairs.**
    - Compare with labeler agreement on the same data, and check accuracy
      by task and within length-matched pairs.
    - Then test it the way it will be used: does optimizing against it
      improve held-out human ratings?

18. **Generation and training on the same GPUs, or separate ones?**
    **Same GPUs is simpler and avoids weight transfers; separate pools keep
    both busy.**
    - Colocated: phases alternate, so one always waits.
    - Separate: run at the same time, but weights are broadcast every
      update and the samples are slightly stale (§12).

## Exit checklist

- [ ] Draw the RL loop and say which step takes most of the time.
- [ ] Name three ways a verifier can be wrong and how to test for them.
- [ ] Explain the over-optimization curve and three mitigations for reward
  hacking.
- [ ] Compute the fraction of zero-signal groups for a given pass rate and
  $G$, and say how to fix it.
- [ ] Explain the GRPO length bias with the 100 vs 1000 token example.
- [ ] Explain entropy collapse and what clip-higher changes.
- [ ] Explain why the sampler and trainer probabilities differ and how to
  correct for it.
- [ ] Read a set of training curves and name the likely problem.
- [ ] Design a reward for a new task with the five-step recipe, including
  the hacks.
- [ ] Write the GRPO loss in PyTorch from memory, with the mask and the
  four tests from §11; do drills 12 and 13 in module 8.
- [ ] Size an RL system for a 70B model: training state, KV cache for
  rollouts, colocated vs separate, sync vs async.
- [ ] Explain why reward model accuracy around 70% can be fine and how to
  keep a reward model accurate during training.
- [ ] Name IPO, KTO, SimPO and online DPO and the problem each one fixes.
- [ ] Describe the R1-Zero and R1 recipes and why small models are
  distilled.
