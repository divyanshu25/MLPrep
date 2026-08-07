# Mixed Sets - Interleaved, Unlabeled Practice

Six problem sets with the labels stripped. In the folders, the folder name does
the hardest part of the interview for you - deciding *which* pattern applies.
Here nothing tells you; that discrimination step is the entire point.

**The rules.** Eligible from week 2 (you need at least two patterns learned
before mixing them can train anything). One set per sitting, the full time box
per problem, no notes, talk out loud. Grade each problem with module 8's
Correctness/Speed/Communication 1–4 rubric
([ml/08_interview_drills/README.md](../ml/08_interview_drills/README.md));
anything below 3/3/3 goes into the queue the same day
(`uv run python practice/review.py add <id> <path>`). The answer key at the
bottom is sealed in one block - do not open it mid-set, because knowing set 3
is "mostly stacks" un-mixes the set.

---

## Set 1 (~95 min)

**Problem 1 (20 min).** For each day in a list of daily temperatures, return
how many days until a strictly warmer one (0 if never).

**Problem 2 (25 min).** Koko eats at speed k bananas/hour; each hour she picks
one pile and eats min(k, pile) from it. Find the smallest k finishing all
piles in ≤ h hours.

**Problem 3 (25 min).** Count contiguous subarrays whose sum is exactly k
(negatives allowed).

**Problem 4 (25 min).** Given `[course, prereq]` pairs, return True iff all
courses can be taken.

## Set 2 (~90 min)

**Problem 1 (20 min).** Length of the longest run of consecutive integers in
an unsorted list, O(n).

**Problem 2 (20 min).** Return True iff a binary tree satisfies the BST
invariant.

**Problem 3 (25 min).** Packages must ship in order; a ship of capacity c
loads a prefix of the remaining packages each day without exceeding c. Find
the least capacity that ships everything within `days` days.

**Problem 4 (25 min).** Unlimited coins of given denominations. Return the
fewest coins summing to `amount`, or −1 if impossible.

## Set 3 (~95 min)

**Problem 1 (20 min).** Count connected components of '1' cells
(4-directional) in a grid.

**Problem 2 (35 min).** Largest axis-aligned rectangle fitting under the
bars of a histogram.

**Problem 3 (15 min).** `out[i]` = product of all elements except `nums[i]`;
no division, O(1) extra space.

**Problem 4 (25 min).** Return True if a string can be segmented into a
sequence of dictionary words.

## Set 4 (~95 min)

**Problem 1 (25 min).** Merge k sorted linked lists into one sorted list.

**Problem 2 (15 min).** Find the minimum element of a sorted array rotated at
an unknown pivot, O(log n).

**Problem 3 (35 min).** Smallest substring of `s` containing every character
of `t` with multiplicity; "" if none exists.

**Problem 4 (20 min).** Minimum number of intervals to remove so the rest are
pairwise non-overlapping (touching endpoints do not overlap).

## Set 5 - ML (~55 min)

**Problem 1 (10 min).** On paper: derive the gradient of softmax +
cross-entropy with respect to the logits, from the loss definition.

**Problem 2 (10 min).** Estimate, out loud with the arithmetic visible: memory
to train a 70B model with Adam in mixed precision. How many H100s for the
state alone, and what makes it fit?

**Problem 3 (10 min).** Your training loss goes NaN after a few hundred steps.
Narrate your debugging tree - ordered hypotheses and the check for each.

**Problem 4 (25 min).** Design fragment: a homepage recommender must score
candidates for 10⁸ DAU over a 10⁹-item corpus in a ~100ms budget. Why is one
model impossible, what's the architecture, and where does the 100ms go?

## Set 6 - ML (~55 min)

**Problem 1 (10 min).** On paper: why does attention divide by √d_k? Derive
the variance argument and state what fails without it.

**Problem 2 (10 min).** Estimate: KV-cache size for a 7B model (32 layers,
d=4096, fp16) at batch 32, 4k context. Compare to the weights, then name
three mitigations and their factors.

**Problem 3 (10 min).** Offline AUC improved 2%; the online A/B is flat.
Name at least three distinct causes and what you'd check for each.

**Problem 4 (25 min).** On paper: starting from KL-regularized reward
maximization, derive the DPO loss. State what the closed-form policy is, what
cancels, and why RM + RL collapse into supervised learning.

---

## Answer key - open only after grading a full set

<details><summary>Key: source and the invariant each problem tests</summary>

| Set·# | Source | Invariant / idea tested |
|---|---|---|
| 1·1 | dsa/03 `daily_temperatures` | Monotonic decreasing stack - stack holds the unanswered; push/pop once each ⇒ O(n) |
| 1·2 | dsa/02 `min_eating_speed` | Binary search on the answer - prove p(k) monotone, bracket [1, max(piles)], greedy checker |
| 1·3 | dsa/01 `subarray_sum` | Prefix sum + hashmap seeded {0:1} - NOT sliding window, negatives break shrink-monotonicity |
| 1·4 | dsa/05 `can_finish` | Kahn's peeling - possible iff DAG; leftovers with indegree > 0 form a cycle |
| 2·1 | dsa/01 `longest_consecutive` | Hash set, count only from run left-ends - each run walked once ⇒ O(n) |
| 2·2 | dsa/04 `is_valid_bst` | Global BST invariant via (lo, hi) bounds - child-only comparison is the planted trap |
| 2·3 | dsa/02 `ship_within_days` | Search on answer, bracket [max(w), sum(w)] - justify both ends, greedy feasibility |
| 2·4 | dsa/06 `coin_change` | Unbounded DP - state dp[a], decision = last coin; unboundedness falls out of the state |
| 3·1 | dsa/05 `num_islands` | Grid as implicit graph - flood fill per unvisited land cell, count the starts |
| 3·2 | dsa/03 `largest_rectangle_area` | Monotonic increasing stack - a bar's rectangle is decided at pop time; sentinel drains |
| 3·3 | dsa/01 `product_except_self` | Prefix/suffix products - prefix pass into output, suffix as a running scalar |
| 3·4 | dsa/06 `word_break` | 1-D DP over prefixes - decision = the last word; dp[0] = True |
| 4·1 | dsa/04 `merge_k_lists` | Heap of k heads - next output must be min of heads; (val, i, node) tuples for tie-breaking |
| 4·2 | dsa/02 `find_min_rotated` | Boundary template, compare a[mid] to a[hi] - comparing to a[lo] breaks on unrotated input |
| 4·3 | dsa/01 `min_window` | Variable window, minimizing flavor - expand to cover, shrink while covered, `have` counter |
| 4·4 | dsa/06 `erase_overlap_intervals` | Greedy by earliest finish + exchange argument - greedy is guilty until proven safe |
| 5·1 | ml/03 notes Q4 | ∂L/∂z = p − y; the log cancels the softmax Jacobian - say what cancels and why it matters |
| 5·2 | ml/07 notes Q3 | 16 B/param → 1.12 TB state; ≥14 H100s; ZeRO-3/FSDP is what makes it fit |
| 5·3 | ml/03 notes Q17 | Ordered hypotheses: LR, log(softmax) instability, exploding grads, bad data, missing ε - narrated, not shotgunned |
| 5·4 | ml/06 notes §retrieval/ranking | The 10⁹ × 1µs = 10³s arithmetic ⇒ two-stage; two-tower constraint; 100ms budget split |
| 6·1 | ml/04 notes Q7 | Var(qᵀk) = d_k ⇒ softmax saturation ⇒ vanishing gradients at init |
| 6·2 | ml/07 notes Q15 | 512 KB/token → ~69 GB, 5× weights; GQA ÷8, KV quant ÷2, PagedAttention reclaims fragmentation |
| 6·3 | ml/05 Q18 / ml/06 Q1 | Proxy-metric audit: unseen-pairs ranking, training/serving skew, label ≠ goal, feedback loops |
| 6·4 | ml/07 notes Q10 | π* ∝ π_ref e^{r/β}; invert, substitute into Bradley–Terry, Z cancels |

</details>
