# Probability puzzles: the stats-round warm-ups

Quant-flavored stats rounds (OpenAI's stats and information-theory round,
some Google and trading-adjacent ML loops) open with puzzles like these.
They test whether you reach for the right tool without prompting: a
geometric distribution, linearity of expectation, a Markov chain on
states, Bayes rule with counts, a symmetry argument. Each solution below
names the tool first. Cover the solution, set a five-minute timer, and
write your answer before unfolding. Every number was checked by exact
computation or simulation.

The tools, and which puzzles use them: geometric distribution (1, 12),
linearity of expectation (2, 6, 15), Markov chain on states (7), Bayes
rule with counts (3, 4, 13), symmetry (8, 11, 14), geometry of the sample
space (9, 11), complement counting (5, 12).

---

## 1. Expected rolls to see a 6

How many times, on average, do you roll a fair die until a 6 appears?

<details><summary>Solution</summary>

**Tool: geometric distribution.** Each roll succeeds with probability
$p = 1/6$, independently. The number of rolls until the first success is
geometric with mean $1/p = 6$.

Derivation from a one-line recursion, which generalizes: let $m$ be the
expected number of rolls. The first roll is a 6 with probability $1/6$
(done, 1 roll) or not with probability $5/6$, after which the situation is
identical and $m$ more rolls are expected on top of the one used:
$m = 1 + \tfrac{5}{6} m$, so $m = 6$.
</details>

---

## 2. All six faces

How many rolls, on average, until every face of the die has appeared at
least once?

<details><summary>Solution</summary>

**Tool: linearity of expectation over stages.** Split the process into
stages: stage $k$ starts when $k - 1$ distinct faces have been seen and
ends when a new one appears. In stage $k$ each roll shows a new face with
probability $(6 - (k-1))/6$, so the stage lasts $6/(7-k)$ rolls on
average (puzzle 1's geometric mean). Sum the six stages:

$$6\Big(\tfrac{1}{6} + \tfrac{1}{5} + \tfrac{1}{4} + \tfrac{1}{3} + \tfrac{1}{2} + 1\Big) = 6 \cdot \tfrac{49}{20} = 14.7$$

The last face alone costs 6 expected rolls. This is the coupon collector
problem; for $n$ faces the answer is $n H_n \approx n \ln n$.
</details>

---

## 3. Monty Hall

Three doors, one prize. You pick door 1. The host, who knows where the
prize is, opens door 3 to show it is empty, and offers a switch to door 2.
Switch?

<details><summary>Solution</summary>

**Tool: Bayes rule, or counting.** Switch: it wins with probability
$2/3$.

Counting: your first pick is right $1/3$ of the time, in which case
switching loses. It is wrong $2/3$ of the time, and then the host, forced
to open the other empty door, has pointed at the prize; switching wins.

The step people miss is that the host's choice is not random. If the host
opened a door at random and it happened to be empty, the odds would be
$1/2$ each. The information is in the host's constraint.
</details>

---

## 4. Two children

A family has two children. (a) At least one is a boy. What is the
probability both are? (b) At least one is a boy born on a Tuesday. Now
what is the probability both are boys?

<details><summary>Solution</summary>

**Tool: conditioning by counting equally likely cases.**

(a) Four equally likely pairs: BB, BG, GB, GG. "At least one boy" removes
GG, leaving three, of which one is BB: $1/3$. (Not $1/2$: that would be
the answer to "the *older* child is a boy".)

(b) Each child has 14 equally likely (sex, weekday) types. Pairs with at
least one Tuesday-boy: $14 + 14 - 1 = 27$ (the $-1$ removes the
double-counted pair of two Tuesday-boys). Among them, both boys: the
first is a Tuesday-boy and the second any boy (7), or the reverse (7),
minus the double count (1): 13. Answer $13/27 \approx 0.48$.

Extra information that seems irrelevant moved the answer from $1/3$
toward $1/2$: the more specific the condition, the closer it comes to
identifying *which* child, and identifying the child gives $1/2$.
</details>

---

## 5. The birthday problem

How many people in a room before it is more likely than not that two
share a birthday?

<details><summary>Solution</summary>

**Tool: complement counting.** Compute the probability that all $n$
birthdays differ, and find where it drops below $1/2$:

$$P(\text{all different}) = \frac{365}{365} \cdot \frac{364}{365} \cdots \frac{365 - n + 1}{365}$$

At $n = 23$ this is $0.493$, so a shared birthday has probability $0.507$.
The number is small because it is *pairs* that matter: 23 people make
$\binom{23}{2} = 253$ pairs, each matching with probability $1/365$, and
$253/365 \approx 0.69$ expected matches is enough for one to be likely.
The general rule: about $\sqrt{2N \ln 2} \approx 1.18\sqrt{N}$ draws from
$N$ options before a collision is more likely than not, which is also the
hash-collision rule.
</details>

---

## 6. Expected maximum of two dice

Roll two dice. What is the expected value of the larger one?

<details><summary>Solution</summary>

**Tool: distribution of the max via the CDF, or linearity.** The max is
at most $k$ exactly when both dice are, so
$P(\max \le k) = (k/6)^2$ and
$P(\max = k) = \big(k^2 - (k-1)^2\big)/36 = (2k - 1)/36$. Then

$$\mathbb{E}[\max] = \sum_{k=1}^{6} k \cdot \frac{2k-1}{36} = \frac{1 + 6 + 15 + 28 + 45 + 66}{36} = \frac{161}{36} \approx 4.47$$

Sanity: above the single-die mean of 3.5, as a maximum must be. By
symmetry the expected minimum is $7 - 4.47 = 2.53$ (replace each die $d$
by $7 - d$).
</details>

---

## 7. HH versus HT

Flip a fair coin repeatedly. On average, how many flips until you first
see HH? Until you first see HT? They are not equal.

<details><summary>Solution</summary>

**Tool: Markov chain on "how much of the pattern have I matched".**

HT. States: nothing matched (0), just saw H (1). From 0, a flip goes to 1
(H) or stays at 0 (T). From 1, T finishes and H stays at 1 (the H is
still a valid start). With $e_0, e_1$ the expected remaining flips:
$e_1 = 1 + \tfrac{1}{2} e_1$ gives $e_1 = 2$; $e_0 = 1 + \tfrac{1}{2}e_1 + \tfrac{1}{2}e_0$
gives $e_0 = 4$.

HH. From 1, T sends you back to 0, not to 1:
$e_1 = 1 + \tfrac{1}{2} e_0$, $e_0 = 1 + \tfrac{1}{2}e_1 + \tfrac{1}{2}e_0$.
Substituting, $e_0 = 2 + e_1 = 2 + 1 + \tfrac{1}{2}e_0$, so $e_0 = 6$.

HH takes longer because a failure while matching it wastes the progress
(a T after an H means starting over), while for HT a failure (H after H)
keeps you one step in. Same principle explains why overlapping patterns
are slow: the pattern's self-overlap decides the waiting time.
</details>

---

## 8. Gambler's ruin

You have 10 dollars and bet 1 dollar on a fair coin each round, stopping
at 0 or at 100. What is the probability you reach 100? How many rounds
does it take on average?

<details><summary>Solution</summary>

**Tool: a harmonic (linear) function of position; martingale.** Let
$p_i$ be the win probability from $i$ dollars. One round from $i$ goes to
$i \pm 1$ with equal chance, so $p_i = \tfrac{1}{2}(p_{i-1} + p_{i+1})$:
$p$ is linear in $i$. With $p_0 = 0$ and $p_{100} = 1$, $p_i = i/100$,
so $p_{10} = 0.1$.

Duration: let $d_i$ be the expected rounds. Then
$d_i = 1 + \tfrac{1}{2}(d_{i-1} + d_{i+1})$ with $d_0 = d_{100} = 0$; the
quadratic $d_i = i(100 - i)$ satisfies it, so $d_{10} = 900$ rounds.

Fair game, but a small stake against a large target loses 90% of the
time. With any house edge $p < 1/2$ the win probability is
$\frac{1 - (q/p)^i}{1 - (q/p)^{100}}$, which is exponentially smaller.
</details>

---

## 9. The broken stick

Break a stick at two points chosen uniformly at random. What is the
probability the three pieces form a triangle?

<details><summary>Solution</summary>

**Tool: geometry of the sample space.** Three pieces form a triangle
exactly when no piece is longer than half the stick. Let the break
points be $x < y$ on $[0, 1]$; the pieces are $x$, $y - x$, $1 - y$. The
sample space is the triangle $0 < x < y < 1$ of area $1/2$. The three
conditions $x < 1/2$, $1 - y < 1/2$, $y - x < 1/2$ cut out the smaller
triangle with corners $(0, \tfrac{1}{2})$, $(\tfrac{1}{2}, 1)$,
$(\tfrac{1}{2}, \tfrac{1}{2})$, of area $1/8$. Ratio: $1/4$.

Confirmed by simulation ($0.249$ over 200,000 trials).
</details>

---

## 10. A fair coin from a biased one

You have a coin with unknown bias $p$. Produce a fair bit. How many flips
does it cost on average when $p = 0.3$?

<details><summary>Solution</summary>

**Tool: symmetry.** Flip twice. HT and TH each have probability
$p(1-p)$, so they are equally likely: output 0 on HT, 1 on TH, and
discard HH or TT and repeat. The output is exactly fair for any $p$ (von
Neumann's trick).

Cost: each pair is decisive with probability $2p(1-p)$, so the expected
number of pairs is $1/(2p(1-p))$ and of flips $1/(p(1-p))$. At $p = 0.3$
that is $1/0.21 \approx 4.8$ flips per fair bit.
</details>

---

## 11. Uniforms that add to one

$U_1, U_2$ are independent uniform on $[0, 1]$. (a) $P(U_1 + U_2 \le 1)$?
(b) With three uniforms, $P(U_1 + U_2 + U_3 \le 1)$? (c) Keep drawing
uniforms until the running sum exceeds 1. What is the expected number of
draws?

<details><summary>Solution</summary>

**Tool: geometry of the sample space, then a series.**

(a) The region $u_1 + u_2 \le 1$ in the unit square is the triangle
below the diagonal: area $1/2$.

(b) The region $u_1 + u_2 + u_3 \le 1$ in the unit cube is a corner
tetrahedron: volume $1/6$. In general the probability that $n$ uniforms
sum to at most 1 is $1/n!$.

(c) Let $N$ be the number of draws until the sum exceeds 1. Then
$P(N > n) = P(U_1 + \dots + U_n \le 1) = 1/n!$, and for a
non-negative integer variable
$\mathbb{E}[N] = \sum_{n \ge 0} P(N > n) = \sum_{n \ge 0} \frac{1}{n!} = e \approx 2.718$.
</details>

---

## 12. De Méré's bets

Which is more likely: at least one 6 in four rolls of a die, or at least
one double-6 in twenty-four rolls of two dice?

<details><summary>Solution</summary>

**Tool: complement.** $P(\text{at least one}) = 1 - P(\text{none})$.

Four rolls: $1 - (5/6)^4 = 1 - 0.482 = 0.518$.
Twenty-four double rolls: $1 - (35/36)^{24} = 1 - 0.509 = 0.491$.

The first bet wins more than half the time, the second less, which is
what de Méré noticed at the tables in 1654 and asked Pascal about. The
naive argument "24 chances at 1/36 equals 4 chances at 1/6" fails
because probabilities of overlapping events do not add.
</details>

---

## 13. The double-headed coin

A bag holds two coins: one fair, one with heads on both sides. You draw
one at random, flip it three times, and see three heads. What is the
probability you drew the double-headed coin?

<details><summary>Solution</summary>

**Tool: Bayes rule with counts** (module 1, §3.2). Imagine 16 draws: 8
of each coin. The double-headed coin gives HHH all 8 times; the fair coin
gives HHH with probability $1/8$, so once. Of the 9 draws showing HHH, 8
are the double-headed coin: $8/9 \approx 0.89$.

As a formula: prior odds 1:1, likelihood ratio $1 : 1/8 = 8:1$, posterior
odds 8:1, probability $8/9$. Each additional head multiplies the odds by
2, so after $n$ heads the answer is $2^n/(2^n + 1)$.
</details>

---

## 14. The lost boarding pass

100 passengers board a full plane in order. The first has lost the
boarding pass and sits in a uniformly random seat. Each later passenger
sits in their own seat if it is free, otherwise in a uniformly random
free seat. What is the probability the last passenger gets their own
seat?

<details><summary>Solution</summary>

**Tool: symmetry between two special seats.** Exactly $1/2$, for any
number of passengers.

Every displaced passenger picks a random free seat. Whenever one of them
picks seat 1 (the first passenger's own seat), every later passenger
finds their seat free and the last passenger is fine. Whenever one picks
seat 100, the last passenger is displaced. Every other choice just passes
the problem on. So the process ends the moment someone picks seat 1 or
seat 100, and at every such moment the two seats are equally likely to be
chosen, since both are free and the choice is uniform. The answer is
$1/2$ by that symmetry.

Simulation with 20 passengers: $0.50$ over 20,000 trials.
</details>

---

## 15. Distinct faces in six rolls

Roll a die six times. What is the expected number of distinct faces
seen?

<details><summary>Solution</summary>

**Tool: linearity of expectation with indicators.** Let $I_k = 1$ if face
$k$ appears at least once. Then the number of distinct faces is
$\sum_k I_k$, and by linearity, which needs no independence,
$\mathbb{E}[\sum_k I_k] = \sum_k P(\text{face } k \text{ appears}) = 6\big(1 - (5/6)^6\big) \approx 3.99$.

Computing the distribution of the count directly is hard; the expectation
is one line. Indicators plus linearity solve most "expected number of
things that happen" puzzles the same way.
</details>
