# Reinforcement learning: exercises

Nine exercises, easy to hard. Do them on paper first; several are verbatim
whiteboard questions. Solutions are collapsed under each one.

---

## 1. Solve a chain by hand (warm-up)

Three states in a line, $s_1 \to s_2 \to s_3 \to$ terminal, one action,
reward $0$ on the first two moves and $r = 4$ on the last, $\gamma = 0.5$.
Write the Bellman equation for each state and solve for $V(s_1)$, $V(s_2)$,
$V(s_3)$. Then say what changes if $\gamma = 1$.

<details><summary>Solution</summary>

Work backward from the terminal state, whose value is 0:

- $V(s_3) = 4 + \gamma \cdot 0 = 4$
- $V(s_2) = 0 + \gamma V(s_3) = 0.5 \cdot 4 = 2$
- $V(s_1) = 0 + \gamma V(s_2) = 0.5 \cdot 2 = 1$

Each step back discounts the terminal reward once more:
$V(s_1) = \gamma^2 \cdot 4$. With $\gamma = 1$ there is no discount and all
three states have value 4: the reward is certain, only its timing differs,
and an undiscounted agent does not care about timing.
</details>

---

## 2. A loop, solved as a linear system

The chain from the notes: from $s_1$, stay with probability $0.5$ (reward
0) or move to $s_2$ with probability $0.5$ (reward 0); from $s_2$, move to
the terminal state with reward 1. Take $\gamma = 0.8$ this time. Solve for
$V(s_1)$. Then compute the expected number of steps spent in $s_1$ and
check the value against it.

<details><summary>Solution</summary>

$V(s_2) = 1$. For $s_1$:
$V(s_1) = 0.5[\gamma V(s_1)] + 0.5[\gamma \cdot 1] = 0.4 V(s_1) + 0.4$, so
$V(s_1) = 0.4/0.6 = 2/3$.

Check by counting steps. Let $k$ be the number of steps taken in $s_1$
before the agent is in $s_2$, counting the step that leaves: $k = 1$ with
probability $0.5$, $k = 2$ with probability $0.25$, and in general
$P(k) = 0.5^k$. The reward of 1 is earned on the step *after* entering
$s_2$, so it is the $(k+1)$-th reward, and the return's weights start at
$\gamma^0$ for the first reward, so it is discounted by $\gamma^k$:

$$V(s_1) = \sum_{k \ge 1} 0.5^k \gamma^k = \sum_{k \ge 1} 0.4^k = \frac{0.4}{1 - 0.4} = \frac{2}{3}$$

The same number, from the geometric series instead of the linear system.
</details>

---

## 3. Two sweeps of value iteration

A gridworld with two cells, $L$ and $R$, and two actions, *left* and
*right*. Moving right from $R$ or left from $L$ bumps into a wall and
returns the agent to the same cell with reward $-1$. Moving between the
cells costs $0$. Reaching $R$ from $L$ earns $+2$. $\gamma = 0.9$.
Starting from $V_0 = 0$, run two sweeps of value iteration by hand and
give the greedy policy after the second sweep.

<details><summary>Solution</summary>

Sweep 1, using $V_0 = 0$ so only immediate rewards count:

- $V_1(L) = \max(\text{left}: -1,\ \text{right}: +2) = 2$
- $V_1(R) = \max(\text{left}: 0,\ \text{right}: -1) = 0$

Sweep 2, using $V_1$:

- $V_2(L) = \max(-1 + 0.9 \cdot 2,\ 2 + 0.9 \cdot 0) = \max(0.8, 2) = 2$
- $V_2(R) = \max(0 + 0.9 \cdot 2,\ -1 + 0.9 \cdot 0) = \max(1.8, -1) = 1.8$

Greedy policy: right in $L$, left in $R$. The agent will oscillate,
collecting $+2$ every other step, which is the best available. Further
sweeps raise both values toward the fixed point ($V(L) = 2 + 0.9 V(R)$,
$V(R) = 0.9 V(L)$, giving $V(L) = 2/(1 - 0.81) \approx 10.5$,
$V(R) \approx 9.5$) without changing the policy.
</details>

---

## 4. The TD target is the Bellman equation, sampled

Show that the expected TD target,
$\mathbb{E}[r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s]$, equals the
right side of the Bellman expectation equation. Then explain in one
sentence why TD(0) with the *current* estimate $V$ in place of $V^\pi$ is
biased, and why that bias disappears at convergence.

<details><summary>Solution</summary>

Condition on $s_t = s$ and average over the action and the next state:
$\mathbb{E}[r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s] = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)[R(s, a) + \gamma V^\pi(s')]$,
which is the Bellman expectation equation's right side, equal to
$V^\pi(s)$. So the TD target is an unbiased one-sample estimate of
$V^\pi(s)$ *when the true $V^\pi$ is plugged in*.

With the current estimate $V$ instead, the target's expectation is
$(T^\pi V)(s)$, the Bellman operator applied to $V$, which differs from
$V^\pi(s)$ by up to $\gamma \max_{s'}|V(s') - V^\pi(s')|$ (the contraction
bound). That is the bootstrapping bias. At convergence $V = V^\pi$, the
operator's fixed point, and the bias is zero.
</details>

---

## 5. The baseline lemma, with a concrete check

Prove that $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s)] = 0$.
Then verify it on a two-action softmax policy with logits
$\theta = [1, 0]$: compute $\pi$, compute $\nabla_\theta \log \pi(a)$ for
each action, and show the probability-weighted sum is the zero vector.

<details><summary>Solution</summary>

Proof: $\sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) = \sum_a \pi_\theta(a \mid s) \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta 1 = 0$.

Numbers: $\pi = \text{softmax}([1, 0]) = [e/(e+1), 1/(e+1)] \approx [0.731, 0.269]$.
For a softmax, $\nabla_\theta \log \pi(a) = e_a - \pi$:

- $a = 1$: $[1, 0] - [0.731, 0.269] = [0.269, -0.269]$
- $a = 2$: $[0, 1] - [0.731, 0.269] = [-0.731, 0.731]$

Weighted sum: $0.731 \cdot [0.269, -0.269] + 0.269 \cdot [-0.731, 0.731] = [0.197 - 0.197,\ -0.197 + 0.197] = [0, 0]$.
</details>

---

## 6. REINFORCE by hand on a bandit

Two arms, softmax policy, $\theta = [0, 0]$. Arm 1 pays $+1$ with
probability $0.8$, arm 2 pays $+1$ with probability $0.2$. Three pulls
happen: arm 1 pays 1, arm 2 pays 0, arm 1 pays 1. With learning rate
$\alpha = 1$ and no baseline, compute the parameter update from these
three samples (sum of the three per-sample gradients). Repeat with a
baseline equal to the mean reward of the three pulls.

<details><summary>Solution</summary>

At $\theta = [0, 0]$, $\pi = [0.5, 0.5]$ and
$\nabla \log \pi(1) = [0.5, -0.5]$, $\nabla \log \pi(2) = [-0.5, 0.5]$.

No baseline, per-sample gradients $r \cdot \nabla \log \pi(a)$:

- pull 1: $1 \cdot [0.5, -0.5] = [0.5, -0.5]$
- pull 2: $0 \cdot [-0.5, 0.5] = [0, 0]$
- pull 3: $1 \cdot [0.5, -0.5] = [0.5, -0.5]$

Sum: $[1, -1]$. New $\theta = [1, -1]$, new $\pi = [0.88, 0.12]$.

Baseline $b = (1 + 0 + 1)/3 = 2/3$, gradients $(r - b)\nabla \log \pi(a)$:

- pull 1: $(1/3)[0.5, -0.5] = [0.167, -0.167]$
- pull 2: $(-2/3)[-0.5, 0.5] = [0.333, -0.333]$
- pull 3: $[0.167, -0.167]$

Sum: $[0.667, -0.667]$. Same direction, and now the failed pull of arm 2
contributes an explicit push *away* from arm 2 rather than nothing. The
two estimators do not have quite the same expectation: this baseline is
the mean of the same three pulls, so it includes each pull's own reward,
which shrinks the expected gradient by $(G-1)/G = 2/3$. A baseline computed
from the *other* pulls only (leave-one-out) would keep the expectation
unchanged. Either way the baseline version has lower variance, because a
reward of 0 is informative (below average) instead of silent.
</details>

---

## 7. Why the dynamics drop out

Write $\log p_\theta(\tau)$ for a three-step trajectory in full, take its
gradient with respect to $\theta$, and state exactly which terms survive.
Then explain why this makes policy gradient methods "model-free", and
name the price paid for not using the dynamics.

<details><summary>Solution</summary>

$\log p_\theta(\tau) = \log p(s_0) + \log \pi_\theta(a_0 \mid s_0) + \log P(s_1 \mid s_0, a_0) + \log \pi_\theta(a_1 \mid s_1) + \log P(s_2 \mid s_1, a_1) + \log \pi_\theta(a_2 \mid s_2) + \log P(s_3 \mid s_2, a_2)$.

Only the three $\log \pi_\theta$ terms contain $\theta$, so
$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{2} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
The start distribution and every dynamics factor have zero gradient.

Model-free: the update needs only the policy's own log-probabilities and
sampled rewards, never $P$. The price is sample efficiency: the dynamics
are not exploited, every gradient estimate is a high-variance sample
average over trajectories, and nothing learned about the environment
transfers to a new reward function.
</details>

---

## 8. Importance sampling and the PPO ratio

Samples were drawn from $\pi_{old}$ with $\pi_{old}(a_1) = 0.4$,
$\pi_{old}(a_2) = 0.6$. The current policy has $\pi_\theta(a_1) = 0.6$,
$\pi_\theta(a_2) = 0.4$. The advantages are $A(a_1) = +1$, $A(a_2) = -1$.
Compute the importance-weighted estimate of
$\mathbb{E}_{a \sim \pi_\theta}[A(a)]$ from the $\pi_{old}$ samples, and
compare to the direct value. Then compute PPO's clipped objective with
$\epsilon = 0.2$ for each action and say which action's gradient the clip
switches off.

<details><summary>Solution</summary>

Direct: $0.6 \cdot 1 + 0.4 \cdot (-1) = 0.2$.

Importance weighted: $\sum_a \pi_{old}(a) \frac{\pi_\theta(a)}{\pi_{old}(a)} A(a) = 0.4 \cdot 1.5 \cdot 1 + 0.6 \cdot 0.667 \cdot (-1) = 0.6 - 0.4 = 0.2$.
Equal, as the identity guarantees.

PPO: $\min(\rho A,\ \text{clip}(\rho, 0.8, 1.2) A)$.

- $a_1$: $\rho = 1.5$, $A = +1$: $\min(1.5, 1.2) = 1.2$. The clipped
  branch is active, and it is constant in $\theta$, so the gradient is
  zero: no further incentive to raise $\pi_\theta(a_1)$, which has already
  moved more than 20% from $\pi_{old}$.
- $a_2$: $\rho = 0.667$, $A = -1$: $\min(-0.667, -0.8) = -0.8$. Also the
  clipped branch, so again zero gradient: $\pi_\theta(a_2)$ has already
  been lowered by more than 20% and the objective stops pushing.

Both actions have moved past the trust region, so this batch contributes
no gradient. The clip is one-sided in effect: it only removes the
incentive to keep moving in the direction that improves the objective,
never the incentive to move back.
</details>

---

## 9. GRPO advantages and what they mean

A prompt is sampled eight times; the reward model scores the completions
$[0, 1, 1, 0, 0, 0, 1, 0]$. Compute the GRPO advantage of each completion.
Then explain what happens to the advantages, and to the gradient, if all
eight completions receive the same score, and why that is the correct
behavior.

<details><summary>Solution</summary>

Mean $= 3/8 = 0.375$. Variance $= 0.375 \cdot 0.625 = 0.234$, standard
deviation $= 0.484$. Advantages $(r - 0.375)/0.484$: $+1.29$ for the three
successes, $-0.775$ for the five failures. Successes are pushed up and
failures down, with the rarer outcome getting the larger magnitude, so
the pushes balance: $3 \cdot 1.29 - 5 \cdot 0.775 = 0$.

If all eight scores are equal, the mean equals every score, every
advantage is $0$ (the normalization is $0/0$ and implementations add a
small constant, still giving $0$), and the gradient from this prompt is
zero. The eight samples are wasted, which is why some implementations
(DAPO) filter out prompts where every completion scores the same. A
critic-based method would still produce a gradient here, and it can be a
correct one: if the critic predicts $V = 0.5$ and all eight succeed,
pushing them up is the right signal. GRPO's group baseline is close to the
baseline lemma with the group mean as $b$ (not exact, since the mean
includes each sample's own reward), and it needs several samples per prompt
so that the mean is informative.
</details>
