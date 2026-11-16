# Post-training RL: exercises

Nine exercises: six with numbers, then three in the format of interview
questions (find the hacks, read the curves, design a reward). Solutions are
collapsed under each one.

---

## 1. How much of the batch gives no signal?

A prompt set has 100 prompts: 40 the model solves with probability 0.02, 30
with probability 0.5, and 30 with probability 0.95. GRPO samples $G = 8$
completions per prompt with a 0/1 reward. How many of the 100 groups do you
expect to give zero gradient? What would you change?

<details><summary>Solution</summary>

A group gives no gradient when all 8 rewards are equal, with probability
$p^8 + (1-p)^8$:

- $p = 0.02$: $0.98^8 \approx 0.851$, so $40 \times 0.851 = 34.0$ groups.
- $p = 0.5$: $2 \times 0.5^8 = 0.008$, so $30 \times 0.008 = 0.2$ groups.
- $p = 0.95$: $0.95^8 \approx 0.663$, so $30 \times 0.663 = 19.9$ groups.

About 54 of the 100 groups give nothing. Over half the generation compute
produces no update. Keep the 30 middle prompts and find more like them;
move the 30 easy ones out of the set; for the 40 hard ones, either raise
$G$ so some samples succeed or hold them back until the model improves.
</details>

---

## 2. GRPO advantages for a rare and a common success

Compute GRPO advantages $(r_i - \text{mean})/\text{std}$ for the groups
$[1, 1, 0, 0, 0, 0, 0, 0]$ and $[1, 1, 1, 1, 1, 1, 0, 0]$ (population
standard deviation). Then compute them without dividing by the standard
deviation (Dr. GRPO). What does the division change?

<details><summary>Solution</summary>

Both groups have standard deviation $\sqrt{0.25 \times 0.75} = 0.433$.

- $[1,1,0,0,0,0,0,0]$: mean 0.25. With division: successes $+1.73$,
  failures $-0.58$. Without: $+0.75$ and $-0.25$.
- $[1,1,1,1,1,1,0,0]$: mean 0.75. With division: successes $+0.58$,
  failures $-1.73$. Without: $+0.25$ and $-0.75$.

In each group the advantages sum to 0. Rare outcomes get the large
advantages: in the hard group the two successes, in the easy group the two
failures. Dividing by the standard deviation multiplies every advantage in
these groups by $1/0.433 = 2.3$. A group with mean 0.5 has standard
deviation 0.5 and is multiplied by only 2. So the division gives prompts
with lopsided results (nearly all right or all wrong) more weight than
prompts with mixed results. Dr. GRPO removes it so every prompt is
weighted by its raw reward differences.
</details>

---

## 3. The length bias, by hand

Two wrong answers in a batch both have advantage $-1$: one is 200 tokens,
the other 2,000. (a) With the original GRPO loss (average over each
completion's own tokens, then over completions), how hard is each token
pushed down? (b) With the total-tokens normalization, where every token is
divided by the batch's total token count of 2,200? (c) Which one lets the
policy reduce its loss by writing longer wrong answers?

<details><summary>Solution</summary>

(a) Short answer: each token gets weight $1/200 = 0.005$. Long answer:
$1/2000 = 0.0005$, ten times smaller. The whole long completion gets the
same total penalty as the short one, spread over ten times as many tokens.

(b) Every token gets $1/2200 \approx 0.00045$, in both completions. The
long answer now receives ten times the total penalty of the short one,
in proportion to its length.

(c) The original loss. When an answer is going to be wrong, making it
longer lowers the penalty on each of its tokens, so training raises the
probability of long wrong answers relative to short ones. Mean response
length grows without accuracy improving.
</details>

---

## 4. Which tokens does the clipped objective update?

PPO's per-token objective is $\min(\rho A,\ \text{clip}(\rho, 0.8, 1.2)\,A)$.
For each token, say whether it gets a gradient: (a) $A = +1$,
$\rho = 1.3$; (b) $A = +1$, $\rho = 0.7$; (c) $A = -1$, $\rho = 0.7$;
(d) $A = -1$, $\rho = 1.3$.

<details><summary>Solution</summary>

- (a) $\rho A = 1.3$, clipped term $1.2$. The min is $1.2$, a constant:
  **no gradient**. The token is already up as far as one update allows.
- (b) $\rho A = 0.7$, clipped term $0.8$. The min is $0.7$, the unclipped
  term: **gradient**, which raises the token's probability.
- (c) $\rho A = -0.7$, clipped term $-0.8$. The min is $-0.8$, a constant:
  **no gradient**. The token is already down as far as allowed.
- (d) $\rho A = -1.3$, clipped term $-1.2$. The min is $-1.3$, the
  unclipped term: **gradient**, which lowers the token's probability.

The pattern: clipping stops a token only after it has moved past the bound
in the direction the advantage wants. A token that moved the wrong way,
as in (b) and (d), always gets a gradient to correct it.
</details>

---

## 5. The KL of best-of-N

Best-of-$N$ samples $N$ completions from the policy $\pi$ and keeps the one
with the highest reward. Assume rewards have no ties, and let $F(y)$ be the
fraction of completions with reward below $y$'s. (a) Show that best-of-$N$
picks $y$ with probability $\pi_N(y) \approx N F(y)^{N-1} \pi(y)$. (b) Show
$\text{KL}(\pi_N \,\|\, \pi) = \log N - (N-1)/N$. (c) Evaluate for $N = 8$.

<details><summary>Solution</summary>

(a) $y$ is chosen when it is one of the $N$ samples and the other $N - 1$
all have lower reward. There are $N$ positions it could be drawn in, it is
drawn with probability $\pi(y)$, and each other sample is lower with
probability $F(y)$. So $\pi_N(y) \approx N\, F(y)^{N-1}\, \pi(y)$.

(b) The KL is
$\mathbb{E}_{y \sim \pi_N}\big[\log \tfrac{\pi_N(y)}{\pi(y)}\big] = \log N + (N-1)\,\mathbb{E}_{\pi_N}[\log F(y)]$.
Under $\pi$, $F(y)$ is uniform on $[0, 1]$ (it is a quantile). Best-of-$N$
keeps the maximum of $N$ uniform quantiles, $U = \max(U_1, \dots, U_N)$,
which has density $N u^{N-1}$. Then
$\mathbb{E}[\log U] = \int_0^1 N u^{N-1} \log u \, du = -1/N$. So
$\text{KL} = \log N - (N-1)/N$.

(c) $\log 8 - 7/8 = 2.079 - 0.875 = 1.20$ nats. The KL grows only like
$\log N$: going from $N = 8$ to $N = 64$ adds about 2 nats.
</details>

---

## 6. Two KL estimators

The KL from $\pi_\theta$ to $\pi_{ref}$ is estimated from tokens sampled
from $\pi_\theta$, using $u = \pi_{ref}(x)/\pi_\theta(x)$. Two estimators are
$k_1 = -\log u$ and $k_3 = (u - 1) - \log u$. (a) Show both have expected
value equal to the KL. (b) Show $k_3 \geq 0$ for every sample, while $k_1$
can be negative. (c) Evaluate both at $u = 0.5$ and $u = 2$.

<details><summary>Solution</summary>

(a) $\mathbb{E}_{\pi_\theta}[-\log u] = \mathbb{E}_{\pi_\theta}\big[\log \tfrac{\pi_\theta}{\pi_{ref}}\big] = \text{KL}$
by definition. For $k_3$, the extra term has expectation
$\mathbb{E}_{\pi_\theta}[u - 1] = \sum_x \pi_\theta(x) \tfrac{\pi_{ref}(x)}{\pi_\theta(x)} - 1 = \sum_x \pi_{ref}(x) - 1 = 0$.

(b) $\log u \leq u - 1$ for all $u > 0$ (the line $u - 1$ is tangent to
$\log u$ at $u = 1$, and $\log$ is concave). So $(u - 1) - \log u \geq 0$.
$k_1 = -\log u$ is negative whenever $u > 1$, that is, for every token the
reference model likes more than the policy does.

(c) $u = 0.5$: $k_1 = 0.693$, $k_3 = -0.5 + 0.693 = 0.193$. $u = 2$:
$k_1 = -0.693$, $k_3 = 1 - 0.693 = 0.307$. A single $k_1$ sample can say
the KL is negative, which no KL can be; $k_3$ never does, and its values
vary less from token to token.
</details>

---

## 7. Find the hacks

Here are two reward functions. For each, list every way a policy could get
reward 1 without solving the task, and fix it.

```python
def math_reward(completion, answer):
    return 1.0 if str(answer) in completion else 0.0

def code_reward(repo_dir):
    result = subprocess.run(["pytest"], cwd=repo_dir)
    return 1.0 if result.returncode == 0 else 0.0
```

<details><summary>Solution</summary>

`math_reward`:

- List many numbers ("the answer is 1, 2, 3, ..., 1000"); one will match.
- Substring matches: for answer 12, a completion ending "= 112" or
  "120" contains "12".
- Fix: require one final answer in a fixed format (for example the last
  `\boxed{}`), parse it, and compare it numerically to the reference.
  Give 0 for a completion with more than one final answer.

`code_reward`:

- Edit the tests to assert nothing, or add a `conftest.py` that skips
  every test: pytest exits with code 0 when all tests are skipped.
- Make the process exit early with code 0, for example `os._exit(0)` in
  code that the tests import.
- Special-case the inputs the tests use.
- Hang forever, or use the network, since there is no timeout or sandbox.
- Fix: copy the original test files in after the model's edits, run in a
  sandbox without network and with a timeout, read pytest's report
  (number of tests passed, none skipped) instead of the exit code, and
  score on hidden tests the model never sees.
</details>

---

## 8. Read the curves

Diagnose each run and say what you would check first.

(a) Helpfulness reward model, PPO. Over 500 steps the train reward rises
from 0.3 to 0.8. The win rate against the starting model, judged by humans
on held-out prompts, goes from 50% to 47%. Mean response length goes from
400 to 2,500 tokens. Entropy is steady.

(b) Math with a verifier, GRPO. Reward rises from 0.2 to 0.6 by step 300,
then stays flat. Entropy falls from 0.8 to 0.05. The fraction of groups
with zero advantage rises from 0.2 to 0.85.

<details><summary>Solution</summary>

(a) Reward hacking through length. The reward model scores longer answers
higher, and the policy has learned to pad: proxy reward up, human
preference down. Check first: read high-reward completions, and plot
reward model score against length on a fixed set of completions. Fixes: a
length penalty or length-matched comparisons, a larger KL coefficient,
retraining the reward model with length-balanced pairs, and picking the
checkpoint by human eval, not reward.

(b) Entropy collapse. The policy became nearly deterministic, so the 8
samples for a prompt are almost identical and usually all right or all
wrong; 85% of groups give no gradient, so reward stops moving. Check
first: samples within a group for one prompt, and the token-probability
distribution. Fixes: clip-higher (larger upper clip bound), sampling
temperature 1.0, dropping zero-signal groups and refilling the batch,
and harder prompts.
</details>

---

## 9. Design a reward

Design the RL reward for a model that summarizes long documents. Use the
five steps from notes §10.

<details><summary>Solution</summary>

1. **Success:** the summary covers the document's main points, says
   nothing the document doesn't support, and fits the length limit.
2. **Checkable by a program:** the length limit (gate: reward 0 if over),
   and format. The rest needs a judge: a rubric judge that checks the
   summary against the document for (i) each claim supported by the
   document, (ii) coverage of a list of key points extracted beforehand.
   Score = fraction of supported claims times fraction of key points
   covered.
3. **Attacks:** copying sentences verbatim (cap overlap with the source,
   or accept it if extractive summaries are fine); vague text with no
   checkable claims, which has nothing to be unsupported (the coverage
   term catches it); text aimed at the judge ("this summary is accurate")
   (strip it, or instruct the judge to ignore it); exploiting the judge's
   preference for its own style (rotate between judges).
4. **Guardrails:** the length gate, a KL penalty since the reward is
   learned, and an ensemble of two judges.
5. **Catching failures:** human-rated summaries on held-out documents,
   checked every evaluation, plus reading the highest-reward summaries.
</details>
