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
The reward of 1 is discounted once per step spent waiting in $s_1$. The
number of steps $k$ until the move to $s_2$ is random, with probability
$(1/2)^k$, so $V(s_1) = \sum_{k \ge 1} (1/2)^k\, 0.9^k = 0.45/0.55 = 0.818$.
The average wait is two steps, but $0.818$ is not $0.9^2 = 0.81$: the average
of $\gamma^k$ is not $\gamma$ raised to the average $k$.

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
tradeoff in the sense of module 1, §3.7.

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
$G_t - V^\pi(s_t)$ is an estimate of the **advantage**
$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$: how much better this action turned out
than the state's average ($G_t$ is one sample of $Q$). In the bandit
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
  ratios blow up and the estimate's variance with them. PPO's objective is
  $\min\big(\rho A,\ \operatorname{clip}(\rho, 1-\epsilon, 1+\epsilon)\,A\big)$.
  The min takes the more pessimistic of the two terms: once $\rho$ has moved
  past $1 \pm \epsilon$ in the direction the advantage favors, the gradient
  is zero, so there is no incentive to move the policy more than a small
  step per batch. Moves in the unfavorable direction are not clipped.
- **Replacing the critic (GRPO).** For an LLM, a critic is a second model
  as large as the policy. GRPO drops it: sample a group of $G$ completions
  for the same prompt, score each, and use the group's mean reward as the
  baseline, with the advantage normalized by the group's standard
  deviation. For rewards $[1, 0, 0, 1, 1]$ the mean is $0.6$, the
  standard deviation $0.49$, and the advantages are $+0.82$ for the
  successes and $-1.22$ for the failures. This is close to the baseline
  lemma with $b$ estimated from the group instead of a network, but not
  exact: the group mean includes each sample's own reward, so the baseline
  depends on the action, which shrinks the expected gradient by a factor
  $(G-1)/G$. Dividing by the standard deviation also reweights prompts.
  Leaving each sample out of its own mean (RLOO) removes the first bias.

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
  1, §3.3, updated after each pull), sample one value from each posterior,
  and pull the arm whose sample is largest. Arms with wide posteriors get
  pulled in proportion to how plausibly they could be best.

## Common interview questions

1. **Why discount future rewards?** Take a task that never ends and pays a
   reward of 1 on every step. With $\gamma = 0.9$ the return is
   $1 + 0.9 + 0.81 + \dots = 1/(1-0.9) = 10$; with $\gamma = 1$ it is
   infinite, and two policies that both earn an infinite return cannot be
   ranked. In general, rewards bounded by $r_{max}$ give a return of at most
   $r_{max}/(1-\gamma)$, and because the weights $\gamma^k$ sum to
   $1/(1-\gamma)$, $\gamma$ also sets an effective horizon: about 10 steps
   at $\gamma = 0.9$, about 100 at $\gamma = 0.99$. The discounted return
   also satisfies $G_t = r_{t+1} + \gamma G_{t+1}$, the recursion every
   Bellman equation is built from.

   **If asked:** *Is $\gamma = 1$ ever used?* Yes, in tasks that always
   terminate, such as generating one LLM completion, because the sum then
   has finitely many terms.

2. **Derive the Bellman expectation equation.** On the chain from §2: from
   $s_1$, stay with probability 0.5 or move to $s_2$ with probability 0.5,
   both with reward 0; from $s_2$, end with reward 1; $\gamma = 0.9$. Then
   $V(s_2) = 1$ and $V(s_1) = 0.5(0 + 0.9\,V(s_1)) + 0.5(0 + 0.9 \cdot 1)$,
   so $V(s_1) = 0.45/0.55 \approx 0.818$. Each term is the reward of one
   transition plus the discounted value of the state it lands in, weighted
   by the probability of that transition. In general, take
   $G_t = r_{t+1} + \gamma G_{t+1}$, condition on $s_t = s$, and average
   over the action $a \sim \pi(\cdot \mid s)$ and the next state
   $s' \sim P(\cdot \mid s, a)$. By the Markov property and the definition
   of $V^\pi$, the expected $G_{t+1}$ given $s_{t+1} = s'$ is $V^\pi(s')$,
   which gives

   $$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\big[R(s, a) + \gamma\, V^\pi(s')\big]$$

   **If asked:** *How do you solve it?* It is one linear equation per state,
   so solve the linear system directly (cost cubic in the number of states)
   or iterate the equation to its fixed point.

3. **Why is there a max in the optimality equation but a sum in the
   expectation equation?** Take a state with two actions, each of which ends
   the episode: action 1 pays 2 and action 2 pays 4. A policy that picks
   each with probability 0.5 has $V = 0.5 \cdot 2 + 0.5 \cdot 4 = 3$, a sum
   over its action probabilities; the optimal policy picks action 2 and has
   $V^* = 4 = \max(2, 4)$. In general the expectation equation describes a
   fixed policy, so it averages $Q^\pi(s, a)$ over $\pi(a \mid s)$, while
   the optimal policy puts all its probability on the best action, so the
   weighted average becomes a maximum.

   **If asked:** *Why is the max outside the sum over next states?* The
   action is chosen before the dynamics pick $s'$, so the maximum can only
   be taken over each action's expected outcome, not over outcomes.

4. **Why does value iteration converge?** On the two-state chain from
   $V_0 = 0$, sweeps 1 to 4 give $V(s_1) = 0, 0.45, 0.6525, 0.744$ on the
   way to $0.818$, so the errors are $0.818, 0.368, 0.166, 0.075$: each is
   $0.45$ times the previous, within the factor $\gamma = 0.9$ that the
   theorem guarantees. In general the update $T$ is a $\gamma$-contraction
   in the max norm, $\max_s |TV(s)-TU(s)| \le \gamma \max_s |V(s)-U(s)|$:
   the reward terms cancel, a difference of two maxima is at most the
   largest difference, and a probability-weighted average is at most its
   largest term. Taking $U = V^*$, which satisfies $TV^* = V^*$, the error
   after $k$ sweeps is at most $\gamma^k$ times the initial error. The fixed
   point is unique, because two fixed points at distance $d$ would satisfy
   $d \le \gamma d$, so $d = 0$.

   **If asked:** *How many sweeps for a 1000-fold error reduction at
   $\gamma = 0.99$?* $\ln 1000 / (-\ln 0.99) \approx 6.91 / 0.01005 \approx 690$.

5. **Monte Carlo vs TD?** Take an episode $s_0 \to s_1 \to s_2 \to$ end
   with rewards $0, 0, 1$, $\gamma = 1$, and a current estimate
   $V(s_1) = 0.3$. Monte Carlo moves $V(s_0)$ toward the observed return 1,
   but only after the episode ends; TD moves it right after the first step,
   toward $r_1 + V(s_1) = 0 + 0.3 = 0.3$. The Monte Carlo target is unbiased
   but sums every random reward in the episode, so its variance is high. The
   TD target contains one random reward, so its variance is low, but it is
   biased while $V(s_1)$ is wrong, because it is built on another estimate
   (bootstrapping). This is module 1's bias-variance tradeoff applied to
   value estimation.

   **If asked:** *Is there anything in between?* An $n$-step return uses $n$
   observed rewards and then $V(s_{t+n})$; PPO's generalized advantage
   estimation (GAE) averages $n$-step advantage estimates with weights
   proportional to $\lambda^{n-1}$, so $\lambda$ sets the tradeoff.

6. **Q-learning vs SARSA?** Take a transition with $r = -1$,
   $\gamma = 0.9$, next-state values $Q(s', \cdot) = [5, 2]$, and an
   $\varepsilon$-greedy policy that happened to take the exploratory second
   action in $s'$. SARSA's target uses the action actually taken:
   $-1 + 0.9 \cdot 2 = 0.8$. Q-learning's target uses the best action:
   $-1 + 0.9 \cdot 5 = 3.5$. So SARSA samples the Bellman expectation
   equation and learns $Q^\pi$ of the exploring policy it follows
   (on-policy), while Q-learning samples the optimality equation and learns
   $Q^*$ from data collected by any policy (off-policy), provided every
   state-action pair keeps being visited.

   **If asked:** *Which one walks closer to the cliff in cliff walking?*
   Q-learning: it learns the shortest path along the edge, which is optimal
   without exploration, and falls off during training when an exploratory
   step is taken; SARSA's targets include those steps, so it learns a path
   further from the edge.

7. **Derive the policy gradient.** On a two-armed bandit with a softmax
   policy at $\theta = [0, 0]$, so $\pi = [0.5, 0.5]$, pulling arm 1 and
   receiving $r = 1$ gives the sample gradient
   $r\,\nabla_\theta \log \pi(1) = [1, 0]-[0.5, 0.5] = [0.5, -0.5]$, a
   step that raises arm 1's probability to $0.73$ at learning rate 1. The
   derivation shows why this sample is an unbiased estimate of $\nabla J$.
   Write $J(\theta) = \sum_\tau p_\theta(\tau) R(\tau)$ and use
   $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$ to get
   $\nabla J = \mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau)\, R(\tau)]$.
   Then $\log p_\theta(\tau)$ is a sum of $\log p(s_0)$, the policy terms
   $\log \pi_\theta(a_t \mid s_t)$ and the dynamics terms
   $\log P(s_{t+1} \mid s_t, a_t)$; only the policy terms depend on
   $\theta$, so

   $$\nabla_\theta J = \mathbb{E}_{\tau \sim p_\theta}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; R(\tau)\Big]$$

   **If asked:** *Why does the environment not need to be differentiable?*
   The gradient passes only through $\log \pi_\theta$; rewards and
   transitions enter as sampled numbers and are never differentiated.

8. **Why does subtracting a baseline not bias the gradient?** In the bandit
   at $\pi = [0.5, 0.5]$, the score $\nabla_\theta \log \pi(a)$ is
   $[0.5, -0.5]$ for arm 1 and $[-0.5, 0.5]$ for arm 2, and its average
   under the policy is $0.5[0.5, -0.5] + 0.5[-0.5, 0.5] = [0, 0]$. So
   subtracting a constant $b$ from every reward changes each sample (with
   $b = 0.5$ and $r = 1$ the arm-1 update shrinks from $[0.5, -0.5]$ to
   $[0.25, -0.25]$) but adds zero in expectation. In general,
   $\sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = \sum_a \nabla \pi(a \mid s) = \nabla 1 = 0$,
   because the probabilities sum to 1 for every $\theta$. Any $b(s)$ that
   depends only on the state factors out of that sum, so its contribution
   is zero.

   **If asked:** *What if the baseline depends on the action?* Then it does
   not factor out of the sum and the gradient is biased; GRPO's group mean
   is a mild case of this (question 11).

9. **What is the advantage and why use it?** Take a state with $V(s) = 10$
   and two equally likely actions with $Q = 11$ and $Q = 9$. Weighting by
   the raw return gives both actions a positive weight of about 10, so both
   become more likely and only the difference across many samples sorts
   them. Weighting by the advantage $A(s, a) = Q(s, a)-V(s)$ gives $+1$ and
   $-1$: the better action is raised, the worse one lowered, and the
   weights are about 10 times smaller, so the estimate has less variance.
   Since $V(s)$ is a baseline, the expected gradient is unchanged. In
   practice $G_t-V(s_t)$ is a sample estimate of the advantage, since $G_t$
   is one sample of $Q(s_t, a_t)$.

   **If asked:** *What else estimates it?* The TD error
   $\delta_t = r_{t+1} + \gamma V(s_{t+1})-V(s_t)$, whose expectation is the
   advantage when $V$ is exact: lower variance than $G_t-V(s_t)$, but biased
   while $V$ is wrong.

10. **Why does PPO have a probability ratio, and why clip it?** Take
    $\epsilon = 0.2$ and an action with advantage $A = +1$. At $\rho = 1.1$
    the objective is $1.1$ and its gradient still raises the action's
    probability; at $\rho = 1.3$ it is $\min(1.3, 1.2) = 1.2$, a constant,
    so the gradient is zero. The ratio
    $\rho = \pi_\theta(a \mid s)/\pi_{old}(a \mid s)$ is the importance
    weight that makes $\mathbb{E}_{a \sim \pi_{old}}[\rho A]$ equal
    $\mathbb{E}_{a \sim \pi_\theta}[A]$, so one batch sampled from
    $\pi_{old}$ can serve several gradient steps. That reweighting has high
    variance once $\pi_\theta$ is far from $\pi_{old}$, and the objective

    $$L = \min\big(\rho A,\ \operatorname{clip}(\rho, 1-\epsilon, 1+\epsilon)\,A\big)$$

    removes the incentive to move $\rho$ past $1 \pm \epsilon$ in the
    direction the advantage favors. Moves in the unfavorable direction are
    not clipped.

    **If asked:** *Why the min, and not the clip alone?* With $A = -1$ and
    $\rho = 1.3$ the clip alone gives a constant $-1.2$ and no gradient,
    although the action became more likely; the min keeps $-1.3$, whose
    gradient lowers $\rho$.

11. **What does GRPO change relative to PPO?** For one prompt, sample
    $G = 5$ completions with rewards $[1, 0, 0, 1, 1]$. The group mean $0.6$
    is the baseline, so the successes get $+0.4$ and the failures $-0.6$;
    dividing by the group's standard deviation $0.49$ gives $+0.82$ and
    $-1.22$. That is the change: GRPO drops the critic, a second network as
    large as the policy, computes advantages from the group, and keeps PPO's
    clipped ratio objective. It is not exactly the baseline lemma, because
    the mean includes the sample's own reward:
    $r_i-\bar r = \frac{G-1}{G}(r_i-\bar r_{-i})$, where $\bar r_{-i}$ is
    the mean of the other $G-1$ rewards. The baseline $\bar r_{-i}$ does
    not depend on sample $i$'s action, so before the division by the
    standard deviation, the expected gradient with the group mean is
    $(G-1)/G = 0.8$ times the true one. Using $\bar r_{-i}$ directly
    (leave-one-out, RLOO) removes this bias: here $+0.5$ for the successes
    and $-0.75$ for the failures.

    **If asked:** *What if every completion in a group gets the same
    reward?* All advantages are 0 (and the standard deviation is 0, so
    implementations add a small constant), so that prompt contributes no
    gradient.

12. **Why is off-policy learning with a neural network unstable, and what
    did DQN do about it?** A standard example: two states share one weight
    $w$, with estimated values $w$ and $2w$, and every reward is 0, so the
    true values are 0. The data contains only the transition from the first
    state to the second, and the TD update on it is
    $w \leftarrow w + \alpha(\gamma \cdot 2w-w)$; with $\gamma = 0.9$ and
    $\alpha = 0.1$ this multiplies $w$ by $1.08$ per update, so any nonzero
    $w$ grows without bound. All three parts of the deadly triad are there:
    bootstrapping (the target contains $2w$), function approximation (one
    shared weight, so raising one value raises the other), and off-policy
    data (the second state is never updated toward its own target of 0).
    DQN reduces the instability with a replay buffer, which samples past
    transitions at random so consecutive updates are not correlated, and a
    target network, a frozen copy of the weights used in the target and
    refreshed every few thousand steps.

    **If asked:** *Removing which ingredient restores convergence?* Any one:
    Monte Carlo targets (no bootstrapping), on-policy TD with linear
    features, and tabular Q-learning all converge.

13. **UCB's bonus term: where does it come from?** After $t = 100$ pulls,
    arm A has average $0.6$ from 50 pulls and arm B average $0.4$ from 5
    pulls. With $2 \ln 100 = 9.21$, A's bonus is $\sqrt{9.21/50} = 0.43$ and
    B's is $\sqrt{9.21/5} = 1.36$, so the scores are $1.03$ and $1.76$ and
    UCB pulls B: 5 pulls are too few to rule it out. The bonus comes from
    Hoeffding's inequality: for $n_a$ independent rewards in $[0, 1]$ with
    true mean $\mu_a$,

    $$P(\mu_a > \bar r_a + c) \le e^{-2 n_a c^2}$$

    Setting the right side to $t^{-4}$ gives $2 n_a c^2 = 4 \ln t$, so
    $c = \sqrt{2 \ln t / n_a}$: the bonus is the width at which the true
    mean lies above the bound with probability at most $t^{-4}$. Because the
    bonus grows with $\ln t$, an arm that is not pulled sees its bound rise
    until it is tried again.

    **If asked:** *How often is a bad arm pulled?* About $8 \ln t / \Delta^2$
    times by step $t$, where $\Delta$ is its gap to the best arm's mean, so
    the regret (reward lost relative to always pulling the best arm) grows
    logarithmically in $t$.

14. **How is an LLM an RL policy?** Prompt "2+2=", completion "4" followed
    by the end-of-sequence token: two actions, with probabilities $0.9$ and
    $0.95$, so the trajectory probability is $0.9 \cdot 0.95 = 0.855$, and a
    reward model scores the finished completion, say $r = 1$. In general the
    state is the prompt plus the tokens so far, an action is the next token,
    the policy is the softmax over the vocabulary, the transition appends
    the token (deterministic), and the reward arrives once at the end.
    Because the trajectory probability is a product of token probabilities,
    $\nabla \log p_\theta(\tau)$ is the sum of per-token log-probability
    gradients, the negative of the SFT loss gradient on that completion. So
    a REINFORCE step on a completion is an SFT step on it scaled by its
    reward (or advantage).

    **If asked:** *What is $\gamma$ here?* Usually 1: every completion
    terminates, so the sum is finite, and with one reward at the end
    $G_t = r$ for every token.

15. **On-policy vs off-policy in one sentence?** From the same transitions
    of an $\varepsilon$-greedy agent with $\varepsilon = 0.1$, SARSA learns
    the value of that $\varepsilon$-greedy policy and Q-learning learns the
    value of the greedy policy. In one sentence: on-policy methods learn
    about the policy that generated the data; off-policy methods learn about
    a different policy, usually the greedy one, from data collected by any
    policy. Only off-policy methods can learn from a replay buffer, whose
    transitions come from older policies.

    **If asked:** *Where does PPO sit?* On-policy, but each batch is reused
    for a few gradient steps while $\pi_\theta$ moves away from the
    $\pi_{old}$ that sampled it; the ratio $\rho$ corrects for that and the
    clip keeps the gap small.

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
