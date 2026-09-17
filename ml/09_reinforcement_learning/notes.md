# Reinforcement learning from first principles

Why this module: RLHF, PPO and GRPO in module 7 are reinforcement learning,
and every lab that trains language models now asks RL fundamentals. DeepMind
asks for the Bellman equation and the policy gradient theorem derived on a
whiteboard; OpenAI and Anthropic ask why PPO has a clipped ratio and what
GRPO replaces the critic with; recommendation and ads teams ask about
bandits. All of it is four ideas: a return, a value function, the Bellman
equation, and the log-derivative trick.

## 1. The setup: Markov decision processes

An agent interacts with an environment in steps. At step $t$ it sees a
**state** $s_t$, picks an **action** $a_t$, receives a **reward**
$r_{t+1}$, and lands in a new state $s_{t+1}$. A **Markov decision process**
(MDP) is the assumption that the next state and reward depend only on the
current state and action, not on the history:

$$P(s_{t+1} = s', r_{t+1} = r \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s', r \mid s_t, a_t)$$

The pieces, with the names used throughout:

- $\mathcal{S}$, the states, and $\mathcal{A}$, the actions.
- $P(s' \mid s, a)$, the **dynamics**: the probability of landing in $s'$
  after taking $a$ in $s$.
- $R(s, a)$, the expected immediate reward for taking $a$ in $s$.
- $\gamma \in [0, 1)$, the **discount factor**, explained below.
- $\pi(a \mid s)$, the **policy**: the probability the agent picks $a$ in
  $s$. This is the thing being learned.

An LLM fits this template exactly: the state is the prompt plus the tokens
generated so far, an action is the next token, the policy is the model's
softmax over the vocabulary, and the reward arrives once at the end of the
completion from a reward model. Module 7 works out the consequences; this
module supplies the theory it relies on.

#### The return, and why it is discounted

The agent's objective is not the next reward but the total reward from here
on, the **return**:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Three reasons for the $\gamma^k$ weights:

- **The sum has to converge.** With rewards bounded by $r_{max}$, the return
  is at most $r_{max}\sum_k \gamma^k = r_{max}/(1-\gamma)$, finite for any
  $\gamma < 1$. Without the discount an endless task has an infinite
  return, and two policies cannot be compared.
- **$\gamma$ sets the horizon.** The weights $\gamma^k$ sum to
  $1/(1-\gamma)$, so the agent effectively cares about roughly that many
  future steps: $\gamma = 0.9$ is a horizon of about 10 steps,
  $\gamma = 0.99$ about 100.
- **It gives the recursion everything else is built on.** Pull the first
  term out of the sum:

$$G_t = r_{t+1} + \gamma\big(r_{t+2} + \gamma r_{t+3} + \dots\big) = r_{t+1} + \gamma\, G_{t+1}$$

The return from now is the next reward plus the discounted return from the
next step. Every equation in sections 2 and 3 is this one line with an
expectation taken.

## 2. Value functions and the Bellman equations

#### Definitions

A policy is judged by the return it earns from each state. The
**state-value function** is the expected return when starting in $s$ and
following $\pi$ thereafter, and the **action-value function** fixes the
first action too:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \qquad Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

The two are related by averaging over the policy's action choice:
$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$.

#### The Bellman expectation equation, derived

Take the expectation of $G_t = r_{t+1} + \gamma G_{t+1}$ given $s_t = s$.
The right side depends on which action the policy picks, which state the
dynamics send the agent to, and then on the return from that state, whose
expectation is $V^\pi(s')$ by definition:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\Big[R(s, a) + \gamma\, V^\pi(s')\Big]$$

This is a system of linear equations, one per state, with the unknowns
$V^\pi(s)$ appearing on both sides. The same step for $Q$:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a')$$

**On numbers.** Two states and a terminal state, $\gamma = 0.9$, one action
(so the policy is fixed):

- From $s_1$: with probability $0.5$ stay in $s_1$ (reward 0), with
  probability $0.5$ move to $s_2$ (reward 0).
- From $s_2$: move to the terminal state with reward 1. Nothing happens
  after the terminal state, so its value is 0.

Write the Bellman equation for each state:

$$V(s_2) = 1 + \gamma \cdot 0 = 1$$

$$V(s_1) = 0.5\big[0 + \gamma V(s_1)\big] + 0.5\big[0 + \gamma V(s_2)\big] = 0.45\, V(s_1) + 0.45$$

Solve the second: $V(s_1)(1 - 0.45) = 0.45$, so $V(s_1) = 0.45/0.55 \approx 0.818$.
The reward of 1 is discounted once per step spent waiting in $s_1$; the
expected wait is two steps, and $0.818$ is what that averages to.

#### The Bellman optimality equation

The **optimal policy** $\pi^*$ earns the highest value in every state, and
its value functions are written $V^*$ and $Q^*$. An optimal agent in state
$s$ picks the action with the largest $Q^*$, so the average over the policy
in the expectation equation becomes a maximum:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a)\Big[R(s, a) + \gamma\, V^*(s')\Big] = \max_a Q^*(s, a)$$

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, \max_{a'} Q^*(s', a')$$

The $\max$ sits outside the sum over $s'$ because the agent chooses its
action before seeing where the dynamics send it. Given $Q^*$, the optimal
policy needs no further computation: $\pi^*(s) = \arg\max_a Q^*(s, a)$.
That is why so many methods estimate $Q$ rather than $V$: $Q$ ranks the
actions directly, while turning $V^*$ into a policy needs the dynamics.

## 3. Solving an MDP when the dynamics are known

#### Value iteration

Treat the optimality equation as an update rule. Start from any $V_0$ and
repeatedly apply the right side:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a)\Big[R(s, a) + \gamma V_k(s')\Big]$$

On the two-state chain from $V_0 = 0$: sweep 1 gives $V(s_2) = 1$,
$V(s_1) = 0$; sweep 2 gives $V(s_1) = 0.45$; sweep 3 gives
$0.45 \cdot 0.45 + 0.45 = 0.6525$; the sequence climbs toward $0.818$.

**Why it converges.** Write the update as an operator $T$ on value
functions. For any two value functions $V$ and $U$, and any state $s$,

$$|(TV)(s) - (TU)(s)| \le \max_a \sum_{s'} P(s' \mid s, a)\,\gamma\,|V(s') - U(s')| \le \gamma \max_{s'} |V(s') - U(s')|$$

The first inequality uses $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$
and the fact that the reward terms cancel; the second uses that a
probability-weighted average of numbers is at most their maximum. So one
sweep shrinks the largest gap between two value functions by a factor
$\gamma$: $T$ is a **$\gamma$-contraction**. Applied to $V_k$ and the fixed
point $V^*$ (which satisfies $TV^* = V^*$),
$\max_s |V_{k+1}(s) - V^*(s)| \le \gamma \max_s |V_k(s) - V^*(s)|$, so the
error shrinks geometrically and the fixed point is unique. The same
argument covers the expectation equation, which is why policy evaluation
by iteration converges too.

#### Policy iteration

Alternate two steps until the policy stops changing:

- **Evaluate**: compute $V^\pi$ for the current policy, by solving the
  linear system or iterating the expectation equation.
- **Improve**: in every state, switch to the greedy action,
  $\pi'(s) = \arg\max_a \sum_{s'} P(s' \mid s, a)[R(s, a) + \gamma V^\pi(s')]$.

The improvement step never makes a policy worse. Picking the greedy action
once and then following $\pi$ earns $\max_a Q^\pi(s, a) \ge V^\pi(s)$;
picking it at every step compounds that inequality, so
$V^{\pi'} \ge V^\pi$ everywhere, with equality only when $\pi$ was already
greedy with respect to its own value function, which is the optimality
equation. There are finitely many deterministic policies, so the loop
terminates at $\pi^*$. In practice it converges in far fewer sweeps than
value iteration, at the cost of a full evaluation per sweep.

## 4. Learning without the dynamics: Monte Carlo and temporal difference

Real problems do not hand over $P(s' \mid s, a)$. The agent has only
sampled transitions $(s, a, r, s')$ from acting. Two ways to estimate
values from samples, and the difference between them is a bias-variance
tradeoff in the sense of module 1, §3.5.

#### Monte Carlo

Run an episode to the end, record the actual return $G_t$ from each state
visited, and average: $V(s_t) \leftarrow V(s_t) + \alpha\,[G_t - V(s_t)]$.
Each sample $G_t$ is an unbiased draw of $V^\pi(s_t)$, so the average is
unbiased. But $G_t$ sums many random rewards, so its variance is large, and
nothing can be learned until the episode ends.

#### Temporal difference

Use the recursion $G_t = r_{t+1} + \gamma G_{t+1}$ and replace the unknown
$G_{t+1}$ with the current estimate $V(s_{t+1})$:

$$V(s_t) \leftarrow V(s_t) + \alpha\,\underbrace{\big[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\big]}_{\text{TD error } \delta_t}$$

The **TD target** $r_{t+1} + \gamma V(s_{t+1})$ is what the Bellman
equation says $V(s_t)$ should equal, evaluated on one sampled transition.
Its expectation over the transition is exactly the right side of the
expectation equation, so the update moves $V$ toward the Bellman fixed
point. The target has low variance (one random reward instead of a whole
episode's worth) but is biased while $V(s_{t+1})$ is still wrong, since the
estimate is built on another estimate: this is called **bootstrapping**.
TD learns online, one step at a time, and in practice usually faster than
Monte Carlo.

#### Q-learning and SARSA

To improve a policy without dynamics, learn $Q$ instead of $V$, so the
greedy action can be read off. Two updates, differing in one term:

- **SARSA** (on-policy) uses the action the policy actually took next:
  $Q(s, a) \leftarrow Q(s, a) + \alpha\big[r + \gamma Q(s', a') - Q(s, a)\big]$.
  It is the TD update for the expectation equation, so it learns
  $Q^\pi$ for the policy being followed, including that policy's
  exploration.
- **Q-learning** (off-policy) uses the best available next action:
  $Q(s, a) \leftarrow Q(s, a) + \alpha\big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big]$.
  It is the TD update for the *optimality* equation, so it learns $Q^*$
  regardless of how the data was collected, as long as every state-action
  pair keeps being visited.

The distinction shows up on a cliff-walking task. An $\varepsilon$-greedy
agent occasionally steps randomly. SARSA accounts for that and learns a
safe path far from the cliff, since its own exploration would sometimes
push it over. Q-learning learns the shortest path along the edge, the
optimal policy for an agent that does not explore, and then falls off
while exploring during training.

#### Function approximation and the deadly triad

Tabular $Q$ needs one entry per state-action pair, impossible for images or
text. Replace the table with a network $Q_\theta(s, a)$ trained to match
the TD target by squared error. Three ingredients together can diverge:
bootstrapping (the target contains $Q_\theta$ itself), off-policy data, and
function approximation. DQN made this work with two changes: a **replay
buffer** of past transitions, sampled at random so consecutive updates are
not correlated, and a **target network**, a frozen copy of $Q_\theta$ used
inside the target and refreshed only every few thousand steps, so the
target is not chasing itself.

## 5. Policy gradient: derived

Value methods learn $Q$ and act greedily. Policy methods skip $Q$ and adjust
the policy's parameters $\theta$ directly to increase expected return. This
is what post-training an LLM does, so the derivation matters.

#### The objective and the log-derivative trick

Let $\tau = (s_0, a_0, r_1, s_1, a_1, \dots)$ be a whole trajectory and
$R(\tau)$ its return. The objective is the expected return under the
policy:

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] = \sum_\tau p_\theta(\tau)\, R(\tau)$$

Differentiating an expectation whose *distribution* depends on $\theta$
needs one identity, which follows from the chain rule applied to
$\log$: $\nabla_\theta \log p = \nabla_\theta p / p$, so

$$\nabla_\theta\, p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)$$

Apply it under the sum:

$$\nabla_\theta J = \sum_\tau \nabla_\theta p_\theta(\tau)\, R(\tau)
  = \sum_\tau p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)
  = \mathbb{E}_{\tau \sim p_\theta}\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big]$$

The gradient is again an expectation under the policy, so it can be
estimated by sampling trajectories. That is the whole trick: a derivative
of a distribution has been turned into a sample average.

#### The dynamics drop out

A trajectory's probability is the start state, then alternately the policy
and the dynamics:

$$p_\theta(\tau) = p(s_0) \prod_t \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$$

Take the log and the product becomes a sum; differentiate with respect to
$\theta$ and every term without $\theta$ vanishes:

$$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

The dynamics $P$ and the start distribution are gone. The agent never
needs to know how the environment works, only how to differentiate its own
policy. Substituting back gives the **policy gradient theorem** in its
simplest form:

$$\nabla_\theta J = \mathbb{E}_{\tau \sim p_\theta}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; R(\tau)\Big]$$

**Reading it.** $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the
direction in parameter space that makes $a_t$ more probable in $s_t$. The
update moves every action taken along that direction, scaled by the return
of the trajectory it belonged to: actions in good trajectories become more
likely, actions in bad ones less. This algorithm, sample trajectories and
take that gradient step, is **REINFORCE**.

**On numbers.** A two-armed bandit (one state, two actions) with a softmax
policy over parameters $\theta = [\theta_1, \theta_2]$, starting at
$\theta = [0, 0]$ so $\pi = [0.5, 0.5]$. For a softmax,
$\nabla_\theta \log \pi(a) = e_a - \pi$, the one-hot of the chosen action
minus the probability vector. Pull arm 1 and receive $r = 1$:

$$\nabla_\theta \log \pi(1) = [1, 0] - [0.5, 0.5] = [0.5, -0.5], \qquad \text{update } \propto r \cdot [0.5, -0.5]$$

With learning rate 1 the parameters become $[0.5, -0.5]$ and the policy
$[0.73, 0.27]$: arm 1 is now favored. Had $r$ been 0 the update would have
been zero, and had it been negative arm 1 would have been suppressed.

#### Two variance reductions that every implementation uses

The estimator is unbiased but noisy: one whole-trajectory return multiplies
every action's gradient. Two fixes, both exact.

**Reward to go.** An action at step $t$ cannot change rewards received
before $t$. Those earlier rewards are constants with respect to $a_t$, and
a constant multiplying $\nabla \log \pi(a_t \mid s_t)$ has expectation zero
(shown next), so they can be dropped. Replace $R(\tau)$ by the return from
$t$ onward:

$$\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t\Big]$$

**A baseline.** Subtract any function of the state, $b(s_t)$, from the
return. This changes nothing in expectation, because for each state

$$\mathbb{E}_{a \sim \pi}\big[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\big]
  = b(s) \sum_a \pi_\theta(a \mid s)\, \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}
  = b(s)\, \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s)\, \nabla_\theta 1 = 0$$

The probabilities sum to 1 for every $\theta$, so their gradient is zero.
The best baseline is close to $V^\pi(s)$, and the quantity
$G_t - V^\pi(s_t)$ has a name, the **advantage** $A(s_t, a_t)$: how much
better this action turned out than the state's average. In the bandit
example, with a baseline of $0.5$ (the average reward so far), the update
becomes $(1 - 0.5)[0.5, -0.5] = [0.25, -0.25]$: the same direction, half
the size, and an arm that earns exactly the average is no longer pushed
either way. Without a baseline, every positive reward pushes every chosen
action up, even the poor ones, and only the *relative* pushes cancel over
many samples.

**Actor-critic.** Estimating $V^\pi$ for the baseline with a second network
(the critic, trained by TD as in section 4) and using
$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ as a one-step estimate of
the advantage gives the actor-critic family. The actor is the policy; the
critic supplies the advantage.

#### From REINFORCE to PPO and GRPO

Module 7's algorithms are this gradient with three modifications, each
answering a specific problem:

- **Reusing samples (importance sampling).** REINFORCE needs fresh
  trajectories from the current $\pi_\theta$ for every gradient step, and
  sampling from an LLM is the expensive part. Samples from an older policy
  $\pi_{old}$ can be reweighted: for any function $f$,
  $\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \sum_a \pi_\theta(a) f(a) = \sum_a \pi_{old}(a)\frac{\pi_\theta(a)}{\pi_{old}(a)} f(a) = \mathbb{E}_{a \sim \pi_{old}}\Big[\frac{\pi_\theta(a)}{\pi_{old}(a)} f(a)\Big]$.
  The **ratio** $\rho = \pi_\theta(a \mid s)/\pi_{old}(a \mid s)$ is what
  appears in PPO's objective, $\mathbb{E}[\rho\, A]$.
- **Not trusting the ratio too far (clipping).** The reweighting is exact
  only in expectation; when $\pi_\theta$ drifts far from $\pi_{old}$ the
  ratios blow up and the estimate's variance with them. PPO clips $\rho$ to
  $[1 - \epsilon, 1 + \epsilon]$ inside the objective, which removes the
  incentive to move the policy more than a small step per batch of
  samples.
- **Replacing the critic (GRPO).** For an LLM, a critic is a second model
  as large as the policy. GRPO drops it: sample a group of $G$ completions
  for the same prompt, score each, and use the group's mean reward as the
  baseline, with the advantage normalized by the group's standard
  deviation. For rewards $[1, 0, 0, 1, 1]$ the mean is $0.6$, the
  standard deviation $0.49$, and the advantages are $+0.82$ for the
  successes and $-1.22$ for the failures. This is exactly the baseline
  lemma above with $b$ estimated from the group instead of a network.

Module 7, sections 3.2 to 3.4, has the full objectives; each term there
is one of the three items here.

## 6. Bandits: the one-state case

A **multi-armed bandit** is an MDP with a single state: each action (arm)
returns a reward from its own unknown distribution, and the agent's only
problem is to find the best arm while not wasting too many pulls on bad
ones. It is the model behind A/B testing with adaptive allocation, ad
selection and recommendation exploration, and it isolates the
**exploration-exploitation** tradeoff from everything else.

- **$\varepsilon$-greedy.** With probability $\varepsilon$ pull a random
  arm, otherwise the arm with the highest average so far. Simple, and
  wasteful: it keeps exploring arms it already knows are bad.
- **UCB (upper confidence bound).** Pull the arm maximizing
  $\bar r_a + \sqrt{2\ln t / n_a}$, the average reward plus a bonus that
  shrinks as the arm is pulled more ($n_a$ pulls out of $t$ total). The
  bonus is a confidence interval from module 1, §3.6: an arm's average is
  uncertain by about $1/\sqrt{n_a}$, so the agent acts optimistically on
  arms it knows little about and stops once their intervals fall below the
  leader.
- **Thompson sampling.** Keep a posterior over each arm's reward (module
  1, §3.2, updated after each pull), sample one value from each posterior,
  and pull the arm whose sample is largest. Arms with wide posteriors get
  pulled in proportion to how plausibly they could be best.

## Common interview questions

1. **Why discount future rewards?** So the infinite sum converges, and to
   set a horizon: the weights sum to $1/(1-\gamma)$, so $\gamma = 0.99$
   means caring about roughly 100 steps. It also gives the recursion
   $G_t = r_{t+1} + \gamma G_{t+1}$ that the Bellman equation comes from.
2. **Derive the Bellman expectation equation.** Take
   $G_t = r_{t+1} + \gamma G_{t+1}$, condition on $s_t = s$, average over
   the policy's action and the dynamics' next state, and note that the
   expected $G_{t+1}$ given $s_{t+1}$ is $V^\pi(s_{t+1})$ by definition.
3. **Why is there a max in the optimality equation but a sum in the
   expectation equation?** The expectation equation describes a fixed
   policy, which averages over its action distribution. The optimal policy
   picks the best action, so the average becomes a maximum.
4. **Why does value iteration converge?** The Bellman operator is a
   $\gamma$-contraction in the max norm: each sweep shrinks the largest
   error by a factor $\gamma$, so the iterates converge geometrically to
   the unique fixed point.
5. **Monte Carlo vs TD?** MC uses actual returns: unbiased, high variance,
   needs complete episodes. TD bootstraps from the current value estimate:
   biased while the estimate is wrong, low variance, learns online. Module
   1's bias-variance tradeoff, applied to value estimation.
6. **Q-learning vs SARSA?** One term: SARSA's target uses the action the
   policy actually took ($Q(s', a')$), Q-learning's uses the best action
   ($\max_{a'} Q(s', a')$). SARSA learns the value of the exploring policy
   it follows; Q-learning learns $Q^*$ off-policy.
7. **Derive the policy gradient.** Write $J = \sum_\tau p_\theta(\tau) R(\tau)$,
   use $\nabla p = p \nabla \log p$ to get
   $\mathbb{E}[\nabla \log p_\theta(\tau) R(\tau)]$, expand
   $\log p_\theta(\tau)$ as a sum in which the dynamics terms carry no
   $\theta$ and vanish, leaving
   $\mathbb{E}[\sum_t \nabla \log \pi_\theta(a_t \mid s_t) R(\tau)]$.
8. **Why does subtracting a baseline not bias the gradient?**
   $\sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = \nabla \sum_a \pi(a \mid s) = \nabla 1 = 0$,
   so any function of the state alone multiplies a zero-mean quantity.
9. **What is the advantage and why use it?** $A(s, a) = Q(s, a) - V(s)$,
   how much better an action is than the state's average under the policy.
   Using it instead of the raw return keeps the gradient unbiased and
   removes the variance from the state's baseline level of reward.
10. **Why does PPO have a probability ratio, and why clip it?** The ratio
    $\pi_\theta / \pi_{old}$ is the importance weight that lets samples
    from an older policy estimate the current policy's gradient. Clipping
    removes the incentive to move the policy far enough in one update that
    the reweighting stops being trustworthy.
11. **What does GRPO change relative to PPO?** It drops the critic. The
    baseline is the mean reward of a group of completions sampled for the
    same prompt, and the advantage is normalized by the group's standard
    deviation. Same baseline lemma, cheaper estimate.
12. **Why is off-policy learning with a neural network unstable, and what
    did DQN do about it?** Bootstrapping, off-policy data and function
    approximation together can diverge (the deadly triad). DQN decorrelates
    updates with a replay buffer and stabilizes the target with a frozen
    target network.
13. **UCB's bonus term: where does it come from?** A confidence interval on
    the arm's average reward: after $n_a$ pulls the average is uncertain by
    about $1/\sqrt{n_a}$, and $\sqrt{2\ln t / n_a}$ is the width that keeps
    the probability of the true mean lying above the bound small as $t$
    grows.
14. **How is an LLM an RL policy?** State: prompt plus tokens so far.
    Action: next token. Policy: the softmax over the vocabulary.
    Reward: the reward model's score, delivered once at the end. Trajectory
    probability is the product of token probabilities, so
    $\nabla \log p_\theta(\tau)$ is the sum of per-token log-prob gradients,
    the same quantity the SFT loss differentiates.
15. **On-policy vs off-policy in one sentence?** On-policy methods learn
    about the policy that generated the data; off-policy methods learn
    about a different policy, usually the greedy one, from data collected
    by any policy.

## Exit checklist

- [ ] Write the Bellman expectation and optimality equations for $V$ and
  $Q$ from $G_t = r_{t+1} + \gamma G_{t+1}$, and solve a two-state chain by
  hand.
- [ ] State why value iteration converges (contraction, factor $\gamma$).
- [ ] Give the TD(0), SARSA and Q-learning updates and say which Bellman
  equation each one samples.
- [ ] Derive the policy gradient theorem in four lines, including why the
  dynamics drop out.
- [ ] Prove the baseline lemma and compute a softmax policy's gradient on a
  bandit by hand.
- [ ] Explain the PPO ratio via importance sampling and GRPO's baseline via
  the lemma.
- [ ] Name the three bandit strategies and the tradeoff each one manages.
