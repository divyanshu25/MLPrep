# Dynamic Programming, Greedy, and Backtracking - From First Principles

This is the module where candidates break, so we go slower than anywhere else.
The reason DP feels hard is that it is usually *taught backwards*: you are shown a
finished table and a recurrence as if they fell from the sky, and then asked to
reproduce the trick on a problem you've never seen. There is no trick. DP is a
mechanical optimization applied to brute-force recursion, and if you can write the
brute force, you can *derive* the DP. That derivation - not memorized recurrences  - 
is what we build here.

The three techniques in this module are all ways of searching a **decision tree**:

- **Backtracking** searches the whole tree (with pruning). Use it when you must
  enumerate solutions, or when subproblems don't repeat.
- **DP** notices that many branches of the tree lead to *identical* subproblems and
  collapses them. Use it when the answer is a count/min/max and subproblems overlap.
- **Greedy** commits to one branch at every node and never looks back. Use it only
  when you can *prove* the locally best choice is safe - and we'll learn the proof
  technique, because guessing greedy is how you fail the interview.

---

## 1. DP is an evolution, not a topic

We derive DP the way it was actually discovered: start from brute force, observe
waste, remove it. We will walk this evolution **twice**, on two problems, because
seeing the same four-step pipeline apply to a second problem is what turns it from
"a thing I saw" into "a thing I do".

### 1.0 The disease: naive Fibonacci

`fib(n) = fib(n-1) + fib(n-2)`, with `fib(0)=0, fib(1)=1`. Code it directly:

```python
def fib(n):
    if n <= 1: return n
    return fib(n - 1) + fib(n - 2)
```

Draw the call tree for `fib(5)`:

```
                        fib(5)
                 ┌────────┴────────┐
              fib(4)             fib(3)
            ┌───┴───┐          ┌───┴───┐
         fib(3)   fib(2)    fib(2)   fib(1)
        ┌──┴──┐   ┌──┴──┐   ┌──┴──┐
     fib(2) fib(1) f(1) f(0) f(1) f(0)
     ┌──┴──┐
   fib(1) fib(0)
```

Count the calls: `fib(3)` is computed **2** times, `fib(2)` **3** times, `fib(1)`
**5** times. The counts are themselves Fibonacci numbers - the tree has ~`φⁿ ≈ 1.618ⁿ`
nodes. For `fib(50)` that's ~10¹⁰ calls to compute a function that has only **51
distinct inputs**. That gap - exponentially many calls, linearly many *distinct*
subproblems - is the entire disease DP cures. Everything else is bookkeeping.

### 1.1 First walk: climbing stairs

> You climb a staircase of `n` steps, taking 1 or 2 steps at a time. How many
> distinct ways to reach the top?

**Step 1 - brute-force recursion.** Stand at the bottom and make the first
*decision*: take 1 step (leaving a staircase of `n-1`) or 2 steps (leaving `n-2`).
Those choices lead to disjoint sets of climbs, so:

```python
def ways(n):
    if n == 0: return 1      # one way to climb nothing: do nothing
    if n < 0:  return 0      # overstepped - not a valid climb
    return ways(n - 1) + ways(n - 2)
```

This is correct and exponential - it's `fib` wearing a costume, with the same
blown-up call tree.

**Step 2 - notice repeated subproblems.** `ways(n-2)` is computed inside
`ways(n-1)` *and* directly. Crucially, `ways(k)` depends only on `k` - not on *how*
we got there. That property ("the future depends only on the current state, not the
path") is what makes memoization sound. There are only `n+1` distinct inputs.

**Step 3 - memoize (top-down DP).** Cache results by argument. **The state IS the
function arguments** - whatever you'd need to pass to the recursive call to make it
self-contained is your state, nothing more:

```python
def ways(n, memo=None):
    if memo is None: memo = {}
    if n == 0: return 1
    if n < 0:  return 0
    if n not in memo:
        memo[n] = ways(n - 1, memo) + ways(n - 2, memo)
    return memo[n]
```

Each of the `n+1` states is computed once, each in O(1): **O(n)** time. The
recursion *structure* is unchanged; we just refuse to redo work.

**Step 4 - tabulate (bottom-up DP).** Memoization computes states lazily in
whatever order the recursion demands. But we can see the dependency order directly:
`ways(k)` needs `ways(k-1)` and `ways(k-2)`, so if we fill an array from small `k`
to large `k`, every dependency is ready when we need it. Tabulation is *the same
computation with the recursion unrolled in dependency order*:

```python
def ways(n):
    dp = [0] * (n + 1)
    dp[0] = 1
    if n >= 1: dp[1] = 1
    for k in range(2, n + 1):
        dp[k] = dp[k - 1] + dp[k - 2]
    return dp[n]
```

Nothing new happened - `memo` became `dp`, and the call stack became a for-loop.

**Step 5 - space-optimize.** Look at what the recurrence *reads*: `dp[k]` touches
only `dp[k-1]` and `dp[k-2]`. Rows older than that are dead. Keep only what the
recurrence reads:

```python
def ways(n):
    prev2, prev1 = 1, 1           # ways(0), ways(1)
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1 if n >= 1 else prev2
```

O(n) time, O(1) space. The full evolution: **brute force → memoize → tabulate →
space-optimize**, each step a mechanical transformation of the previous one.

### 1.2 Second walk: house robber

> Houses `0..n-1` hold loot `nums[i]`. You can't rob two adjacent houses. Maximize
> total loot.

Same pipeline, new problem. Notice this time the answer is a *max*, not a count  - 
the pipeline doesn't care.

**Step 1 - brute force from the decision.** Stand at house `i` and decide:

- **Rob it**: gain `nums[i]`, but house `i+1` is off-limits → best is
  `nums[i] + best(i+2)`.
- **Skip it**: best is `best(i+1)`.

```python
def rob_from(nums, i):
    if i >= len(nums): return 0
    return max(nums[i] + rob_from(nums, i + 2),   # rob house i
               rob_from(nums, i + 1))             # skip house i
```

Two branches per call, depth n → O(2ⁿ).

**Step 2 - spot the overlap.** `rob_from(i+2)` is reached both directly and via
`rob_from(i+1)`'s skip branch. Only `n+2` distinct values of `i`. And again the
Markov-like property holds: the best loot from house `i` onward doesn't depend on
which houses before `i` we robbed - the no-adjacency constraint is fully captured
by "we're allowed to rob house `i`".

**Step 3 - memoize.** State = the argument `i`. Cache on `i`. O(n).

**Step 4 - tabulate.** `dp[i]` = max loot obtainable from houses `i..n-1`.
Dependencies point to *larger* `i`, so fill right-to-left (or flip the state to
"first `i` houses" and fill left-to-right - both are valid; pick the direction
where dependencies are already filled):

```python
def rob(nums):
    n = len(nums)
    dp = [0] * (n + 2)                      # dp[n] = dp[n+1] = 0
    for i in range(n - 1, -1, -1):
        dp[i] = max(nums[i] + dp[i + 2], dp[i + 1])
    return dp[0]
```

**Step 5 - space-optimize.** `dp[i]` reads only `dp[i+1]` and `dp[i+2]`:

```python
def rob(nums):
    ahead1 = ahead2 = 0                     # dp[i+1], dp[i+2]
    for i in range(len(nums) - 1, -1, -1):
        ahead1, ahead2 = max(nums[i] + ahead2, ahead1), ahead1
    return ahead1
```

Two problems, one pipeline. From now on, when a recurrence appears in these notes,
mentally re-run this evolution - every table below is just an unrolled recursion.

---

## 2. The universal DP method - a checklist

Every DP solution answers five questions, in order. In an interview, say them out
loud; interviewers give partial credit for the framework even when the code stalls.

1. **Define the state, precisely, in words.**
   "`dp[i]` = the maximum loot using only the first `i` houses."
   "`dp[i][j]` = the length of the LCS of `A[:i]` and `B[:j]`."
   The sentence must be checkable: given a small example, you can compute `dp[3]`
   by hand from the definition alone, before writing any recurrence.
2. **Write the recurrence as a decision.** "At state X, my last/next move is either
   ... or ...; take the best/sum over choices." A recurrence that isn't backed by a
   decision is a formula you'll misremember under pressure.
3. **Base cases.** The states where no decision remains. Get the empty case right
   (`dp[0]` = 0 items / empty string / amount 0) - off-by-one here is the #1 bug.
4. **Evaluation order.** Fill states so that everything the recurrence reads is
   already computed. Top-down memoization gets this for free; bottom-up you choose
   the loop direction (this is where the knapsack bug in §3.3 lives).
5. **Read off the answer.** Often `dp[n]` or `dp[n][m]`, but not always - for LIS
   it's `max(dp)`, because the state is "ending at `i`", not "among the first `i`".

**The state definition is 90% of the difficulty.** Once "`dp[i][j]` = edit distance
between `word1[:i]` and `word2[:j]`" is written down, the recurrence is almost
forced - you just enumerate what the last operation could have been. When you're
stuck on a DP problem, you are almost never stuck on the recurrence; you're stuck
because you haven't found the state. Debug by asking: *what is the smallest set of
facts about my partial progress that determines the best possible future?* (For
house robber: my position, and whether the previous house was robbed - which we
folded into the position. For stock-with-cooldown: the day, and whether I hold a
share.)

**Counting the cost** (crash section, per the promise in the title): DP time is

> **(number of states) × (cost of one transition)**

- Climbing stairs: n states × O(1) transition = O(n).
- Coin change: (amount+1) states × O(#coins) transition = O(amount · coins).
- LCS: (n+1)(m+1) states × O(1) = O(nm).
- LIS (quadratic): n states × O(n) transition (scan all previous) = O(n²).
- Interval DP (burst balloons): O(n²) states × O(n) choices of split point = O(n³).

Space is (number of states you must keep live) - often reducible to one row or two
scalars once you see what the recurrence reads. If your DP is too slow, either you
have too many states (redefine them) or transitions are too expensive (precompute,
or replace the inner scan with a data structure - that's exactly the LIS O(n log n)
story in §3.5).

---

## 3. The classic state families

Interviewers draw from a small set of decision structures. For each family: the
recognition cues, the *decision* at each state, and the derived recurrence. Never
memorize the recurrence without the decision - the decision is what survives stress.

### 3.1 1-D linear (climb stairs, house robber, decode ways, max subarray)

**Cue:** a sequence processed left to right; each element admits a small fixed set
of choices; the answer for a prefix depends on O(1) previous prefixes.

**Decision:** "what do I do with element `i`, given the best answers for the
prefixes just before it?" Done above in full. Variants: decode ways (is the last
character a 1-digit letter, or do the last two form 10–26?), Kadane's max subarray
(does the best subarray ending at `i` extend the one ending at `i-1`, or start
fresh?).

### 3.2 Unbounded choice - coin change (and the ordering trap)

> Coins of denominations `coins` (unlimited supply), target `amount`. Fewest coins
> to make `amount`? (−1 if impossible.)

**State:** `dp[a]` = fewest coins that sum to exactly `a`.

**Decision:** consider the *last* coin used in an optimal way to make `a`. It is
*some* coin `c` - we don't know which, so we try all of them and take the min.
Whatever it was, the rest of the pile makes `a - c` optimally (if it didn't, we
could swap in a better pile for `a - c` and improve - that's the "optimal
substructure" argument, and it's worth saying out loud once in an interview):

```
dp[a] = 1 + min(dp[a - c] for c in coins if c <= a)        dp[0] = 0
```

This is why it's *min over coins*: the min ranges over the identity of the last
coin. Unbounded supply is captured automatically: `dp[a-c]` may use coin `c`
again, because the state doesn't remember which coins were used - only the amount.
Cost: O(amount) states × O(coins) transition.

**The combinations-vs-permutations ordering trap.** Now change the question to
*count the ways* to make `amount` (coin change II). Two loop orders, two different
answers - this is the classic bug:

```python
# COUNTS COMBINATIONS (correct for "number of ways to pick a multiset of coins"):
for c in coins:                 # outer: coins
    for a in range(c, amount + 1):
        dp[a] += dp[a - c]

# COUNTS PERMUTATIONS (ordered sequences - the "combination sum IV" question):
for a in range(1, amount + 1):  # outer: amounts
    for c in coins:
        if c <= a: dp[a] += dp[a - c]
```

Why: with **coins outer**, the state while processing coin `c` is really "ways to
make `a` using only the coins considered so far" - coin identities are committed in
a fixed order, so `1+2` and `2+1` are the same completed decision sequence, counted
once. With **amount outer**, `dp[a]` sums over "what was the *last* coin", and a
sequence ending in 1 is distinct from one ending in 2 - so `1+2` and `2+1` count
separately. With coins `{1,2}` and amount 3: combinations = 2 (`1+1+1`, `1+2`),
permutations = 3 (`1+1+1`, `1+2`, `2+1`). Know which the problem wants, and say
which your loop order computes.

(For min-coins the loop order doesn't matter - `min` is insensitive to counting the
same pile twice. The trap only bites counting problems.)

### 3.3 0/1 knapsack - take/skip, the 2-D table, and the compression bug

> `n` items, item `i` has weight `w[i]` and value `v[i]`, knapsack capacity `W`.
> Each item used **at most once**. Maximize value.

**Why 2-D.** Try the 1-D state "`dp[c]` = best value with capacity `c`". The
decision "take item `i`" needs to know whether item `i` was already used - but the
state doesn't record it. The fix is to add that information to the state:

**State:** `dp[i][c]` = best value using only the **first `i` items** with capacity
`c`. Now "have I used item `i`?" is answered by construction: within row `i`, item
`i` is being decided right now, and rows below it can't touch it.

**Decision** at `(i, c)` - the last item either goes in or it doesn't:

```
dp[i][c] = max( dp[i-1][c],                          # skip item i
                v[i-1] + dp[i-1][c - w[i-1]] )       # take it (if w[i-1] <= c)
dp[0][c] = 0
```

Both branches recurse into row `i-1`: item `i` is decided exactly once. Contrast
with coin change, where the "take" branch stayed in the same coin-set - that's
exactly the unbounded/0-1 distinction, expressed as which row the take-branch
reads. O(nW) time and space. (O(nW) is *pseudo-polynomial* - polynomial in the
numeric value of W, exponential in its bit-length; worth mentioning at Google.)

**1-D compression, and why the direction matters.** Row `i` reads only row `i-1`,
so keep one array and overwrite in place. But now `dp[c - w]` might mean row `i-1`
(not yet overwritten - what we want) or row `i` (already overwritten - a bug).
Iterate capacity **downward** so that when we write `dp[c]`, the smaller index
`dp[c - w]` is still last iteration's value:

```python
for i in range(n):
    for c in range(W, w[i] - 1, -1):        # DOWNWARD - this is the whole point
        dp[c] = max(dp[c], v[i] + dp[c - w[i]])
```

**The bug, concretely.** Iterate upward with one item, `w=[2], v=[10], W=6`:

```python
for c in range(2, 7):                        # WRONG direction for 0/1
    dp[c] = max(dp[c], 10 + dp[c - 2])
# c=2: dp[2] = 10 + dp[0] = 10
# c=4: dp[4] = 10 + dp[2] = 20   ← dp[2] is already THIS item's row: item taken twice
# c=6: dp[6] = 10 + dp[4] = 30   ← three times
```

Answer 30 with a single copy of a value-10 item. Upward iteration silently turns
0/1 knapsack into unbounded knapsack - which is why the coin-change loops in §3.2
iterate *upward* on purpose. One direction rule to internalize: **compressed 1-D
loop goes downward when each item may be used once, upward when supply is
unlimited.** If you can't remember which, re-derive it from "which row does
`dp[c-w]` need to come from?" - that derivation takes ten seconds and never lies.

### 3.4 Two sequences - LCS and edit distance

**Cue:** two strings/arrays, and the question compares or aligns them.

**Longest common subsequence.** **State:** `dp[i][j]` = length of the LCS of
`A[:i]` and `B[:j]` (first `i` chars of A, first `j` of B - half-open prefixes make
base cases clean: `dp[0][j] = dp[i][0] = 0`).

**Decision:** look at the *last* characters of the two prefixes, `A[i-1]` and
`B[j-1]`:

- If they're **equal**, some optimal LCS matches them to each other (if an optimal
  LCS ignored one of them, you could always extend/replace to use them - a small
  exchange argument), so `dp[i][j] = 1 + dp[i-1][j-1]`.
- If they **differ**, they can't both end the LCS, so at least one is unused:
  `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.

O(nm) states, O(1) transition.

**Edit distance** (insert/delete/replace to turn `A` into `B`). Same state shape:
`dp[i][j]` = min operations to turn `A[:i]` into `B[:j]`. **Decision:** the
three-way choice over what the *last* operation touching the ends was:

```
if A[i-1] == B[j-1]:  dp[i][j] = dp[i-1][j-1]            # free match
else: dp[i][j] = 1 + min( dp[i-1][j-1],   # replace A's last char with B's
                          dp[i-1][j],     # delete A's last char
                          dp[i][j-1] )    # insert B's last char at A's end
```

Base cases are real work now: `dp[i][0] = i` (delete everything), `dp[0][j] = j`
(insert everything). Both tables compress to two rows (or one row plus a saved
diagonal) since row `i` reads only row `i-1`.

### 3.5 Longest increasing subsequence - and buying a better transition

**State (quadratic version):** `dp[i]` = length of the longest strictly increasing
subsequence **ending exactly at index `i`**. Note the "ending at" - the prefix
state "`dp[i]` = LIS within the first `i`" doesn't work, because to extend you need
to know the last element's *value*, and pinning the endpoint encodes it.

**Decision:** which element comes just before `nums[i]` in the subsequence? Any
`j < i` with `nums[j] < nums[i]` (or nothing):

```
dp[i] = 1 + max(dp[j] for j < i if nums[j] < nums[i]; default 0)
answer = max(dp)         # the LIS ends *somewhere*, not necessarily at n-1
```

n states × O(n) transition = O(n²).

**O(n log n): replace the scan with an invariant.** The inner loop asks "best
chain I can extend with `nums[i]`" - we're paying O(n) to answer a query that a
sorted structure answers in O(log n). Maintain a list `tails` with the invariant:

> `tails[k]` = the **smallest possible tail value** of any increasing subsequence
> of length `k+1` seen so far.

Two consequences: (1) `tails` is strictly increasing - if a length-5 chain could
end lower than some length-4 chain, its own prefix of length 4 would end lower
still; (2) processing `x = nums[i]`, the longest chain `x` can extend is found by
binary search: locate the first `tails[k] >= x` (that chain can't precede `x`;
everything left of it can). Replace `tails[k] = x` - we've found a length-`k+1`
chain with a smaller (or equal-position better) tail, which can only help future
elements - or append if `x` beats every tail. `len(tails)` is the LIS length.

Careful: `tails` is **not** an actual LIS - it's a frontier of best-possible tails,
one per length. This is "patience sorting"; in the interview, state the invariant
before writing `bisect_left`, because the invariant is the solution and the code is
transcription. (Longest *non-decreasing*: use `bisect_right`.)

### 3.6 Grid paths

**Cue:** 2-D grid, moves restricted (right/down), count paths or optimize path sum.

**State:** `dp[r][c]` = number of paths (or best cost) from start to cell `(r,c)`.
**Decision:** the last move into `(r,c)` came from above or from the left:
`dp[r][c] = dp[r-1][c] + dp[r][c-1]` (or `min` plus cell cost for path-sum
versions). First row/column are base cases (one way to walk straight). O(mn), one
row of space after compression. Obstacles just zero out cells. This family is the
gentlest 2-D DP and a good warm-up for LCS-shaped tables.

### 3.7 Interval DP, in one paragraph

**Cue:** the natural "process left to right" decision doesn't decompose the problem
(each choice affects both neighbors - burst balloons, matrix-chain multiplication,
polygon triangulation). **State:** `dp[i][j]` = best answer for the contiguous
interval `(i..j)`. **Decision:** pick the *last* event inside the interval (the
last balloon burst, the outermost matrix split); it partitions the interval into
two independent subintervals, giving
`dp[i][j] = best over k in (i..j) of dp[i][k] + dp[k][j] + cost(i,k,j)`.
Choosing the *last* event rather than the first is the key move - it's what makes
the two sides independent of each other. Evaluation order: increasing interval
length, so smaller intervals are ready. O(n²) states × O(n) split points = O(n³).
Burst balloons is drilled in `drills.md`.

---

## 4. Backtracking - DFS over the decision tree

DP collapses a decision tree; backtracking *walks* it. Use backtracking when you
must **enumerate** the solutions themselves (all subsets, all permutations, the
actual N-queens boards) or when subproblems don't repeat, so there's nothing to
cache.

### 4.1 Deriving the template

A partial solution is a path from the root of the decision tree. At each node:
enumerate the legal next choices; for each, **choose** (append to the path),
**explore** (recurse), **unchoose** (pop, restoring state so the next sibling
starts clean). The unchoose step is what lets one shared mutable `path` serve the
entire tree - that's the whole template:

```python
def backtrack(path, choices):
    if is_complete(path):
        results.append(path.copy())      # copy! path keeps mutating after this
        return
    for c in legal(choices, path):
        path.append(c)                   # choose
        backtrack(path, choices)         # explore
        path.pop()                       # unchoose
```

Classic bug: appending `path` itself instead of `path.copy()` - every "solution"
in `results` later mutates into the same empty list.

**Pruning** = refusing to enter a subtree that provably contains no solutions
(constraint already violated, or remaining budget can't reach the target). Pruning
never changes correctness - the tree is still searched exhaustively in principle  - 
it changes whether you finish before the heat death of the universe. N-queens
without pruning is 64-choose-8; with row-by-row placement and column/diagonal
checks it's a few thousand nodes.

### 4.2 Subsets vs combinations vs permutations - same template, different tree

All three are the template above; what differs is **what a tree node's children
are**:

- **Subsets**: each element gets a binary decision, in/out - or equivalently, at
  each node the children are "elements with index ≥ start" (pass `start = i + 1`
  when choosing element `i`). The `start` index enforces one canonical order per
  subset, so `{1,2}` and `{2,1}` are one node, not two. 2ⁿ leaves.
- **Combinations** (size-k subsets): subsets tree + prune paths longer than `k`
  (and paths that can't reach length `k` with the elements remaining).
- **Permutations**: order matters, so *no* `start` index - children are **all
  unused elements** (track a `used` set/array). n! leaves.

Recognition rule: does the problem consider `[1,2]` and `[2,1]` the same? Same →
subsets/combinations machinery (`start` index). Different → permutations machinery
(`used` set). This is the backtracking mirror of §3.2's coin-change ordering trap.

### 4.3 Constraint problems: N-queens

Place `n` queens, none attacking. The decision structure: **one queen per row**
(any solution has exactly that, so placing row by row loses nothing and shrinks the
tree from `n²`-choose-`n` to at most `n` children per node). State to check
constraints in O(1): occupied columns, occupied "/" diagonals (constant `r + c`),
occupied "\" diagonals (constant `r − c`). Choose a column for the current row if
all three sets allow it; recurse to the next row; unchoose (remove from all three
sets). Full code in `problems.py`.

### 4.4 Backtracking or DP? Ask about overlap

The decision tree is common to both. Ask two questions:

1. **Do different paths reach identical subproblems?** In climbing stairs, a
   thousand paths reach "5 steps remain" and the answer from there is the same for
   all of them → collapse with DP. In permutations, the state is *which elements
   are used* - with n! mostly-distinct states, there's (almost) nothing to merge.
2. **What does the problem ask for?** Count/min/max → a number per state → DP can
   summarize. "Return all solutions" → the output alone can be exponential, so no
   state-collapsing can beat exponential - backtrack.

Word break is the instructive hybrid: "*can* this string be segmented?" is DP over
suffixes (booleans, heavy overlap); "*return all* segmentations" is backtracking
(often with a DP feasibility check as pruning).

---

## 5. Greedy - the dangerous pattern

Greedy makes the locally best choice and never revisits it. It's the shortest code
in this module and the easiest to get wrong, because **locally optimal ≠ globally
optimal** by default - a greedy algorithm is *guilty until proven safe*.

**Exhibit A - coin change with arbitrary denominations.** Coins `{1, 3, 4}`,
amount 6. Greedy (largest coin first): 4, then 1, 1 → three coins. Optimal: 3 + 3 →
two coins. Greedy fails because taking the 4 changes what remains (2) into an
amount the coin system covers badly. (US coins happen to be a "canonical" system
where greedy works - which is exactly why the intuition is a trap: it's trained on
a special case.) When greedy fails, the fix is DP: keep *all* subproblem answers
instead of committing to one.

### 5.1 The proof technique: the exchange argument

To trust a greedy choice, show: **any optimal solution can be transformed, one swap
at a time, to agree with the greedy choice, without getting worse.** Then greedy's
first choice is part of *some* optimal solution; induct on the rest. Carry it out
twice, for real:

**Interval scheduling (max number of non-overlapping intervals).** Greedy: sort by
**end time**, repeatedly take the earliest-finishing interval compatible with what's
chosen. *Exchange argument:* let `g` be the earliest-finishing interval overall,
and let OPT be any optimal solution. If `g ∈ OPT`, done. Otherwise let `f` be the
first-to-finish interval in OPT. Every other interval in OPT starts after `f` ends
(they're mutually non-overlapping and finish later). Since `g` finishes no later
than `f`, every other interval in OPT also starts after `g` ends - so
`OPT − {f} + {g}` is still non-overlapping and still has |OPT| intervals. We
exchanged OPT's choice for greedy's at no cost; recurse on the intervals starting
after `g` ends. ∎  - Intuition in one line: *finishing earliest leaves the maximum
room for the future.* (Sorting by start time or by shortest duration both fail  - 
find the 3-interval counterexamples; they're in `drills.md`'s warm-up.)
The interview form is usually the contrapositive: "minimum intervals to **remove**
to make the rest non-overlapping" = n − (max non-overlapping kept). In
`problems.py`.

**Jump game (can you reach the end / min jumps).** Reachability version: sweep
left to right maintaining `furthest = max(furthest, i + nums[i])`; if `i > furthest`
ever, you're stuck. The greedy claim is that tracking only the single number
`furthest` loses nothing. *Exchange argument:* any cell `k ≤ furthest` is reachable
(take any witness path to the `i` that set `furthest`, then jump toward `k`  - 
intermediate cells are all ≤ furthest and an induction makes this precise), and no
path can pass beyond `furthest` without standing on some cell `i ≤ furthest` first,
whose reach is already folded into `furthest`. So the set of reachable positions is
exactly the prefix `[0, furthest]` - any solution that "cleverly" chose different
jumps can be exchanged for one summarized by `furthest`, with no loss. That prefix
structure is *why* one scalar suffices - and it's the difference from coin change,
where committing to a coin genuinely discards options. For *minimum* jumps (jump
game II, in `drills.md`), the same idea becomes a BFS-in-disguise over "frontier of
cells reachable in `j` jumps".

### 5.2 Recognizing greedy vs DP

Cues that greedy *might* work: the problem is about intervals/deadlines/frontiers;
choices can be sorted so that one ordering "dominates"; you can articulate an
exchange argument in a sentence. Protocol under interview pressure: if you can't
sketch the exchange argument in ~30 seconds, **write the DP** - a correct O(n·W)
DP beats a wrong O(n log n) greedy every time, and interviewers specifically probe
greedy answers with counterexample-shaped follow-ups. If greedy fails on your own
small adversarial example (always test coins-{1,3,4}-style cases), that failure is
itself the signal that subproblem answers must be kept, i.e., DP.

---

## 6. The one-page recognition table

| Cue in problem statement | Family | State sketch |
|---|---|---|
| count ways / max-min over a sequence, O(1) lookback | 1-D linear DP | `dp[i]` = answer for prefix `i` (or ending at `i`) |
| unlimited supply, hit a target amount | unbounded DP | `dp[a]` = answer for amount `a`; watch loop order for counting |
| each item once, capacity budget | 0/1 knapsack | `dp[i][c]`; 1-D compress with **downward** capacity loop |
| two strings compared/aligned | 2-seq DP | `dp[i][j]` on prefix pair; decision = last chars |
| "longest increasing …" | LIS | `dp[i]` ending at `i`; upgrade to tails+bisect |
| grid, moves right/down | grid DP | `dp[r][c]` from top/left |
| choice affects both neighbors; "last event" splits range | interval DP | `dp[i][j]` on interval, loop by length, O(n³) |
| "return **all** …", puzzles/constraints | backtracking | choose/explore/unchoose; `start` vs `used` |
| intervals/frontier + a one-line exchange argument | greedy | sort + single pass; otherwise DP |

Everything in this table is exercised in `problems.py` (12 implemented, tested
problems) and `drills.md` (8 timed drills with hint ladders). Do the drills
closed-book: the skill being tested at Google/DeepMind is not knowing these
recurrences - it's deriving them at a whiteboard from the decision, in minutes,
which is precisely the pipeline of §1 and the checklist of §2.

## Exit checklist

- [ ] Cold, blank editor: run the universal DP checklist on house robber - state, transition, base case, order - 4 min
- [ ] Coin change: write both loop orders and say which counts combinations vs permutations, and why - 4 min
- [ ] 0/1 knapsack with 1-D compression: why the capacity loop must run backward (show the double-count bug) - 3 min
- [ ] Write the LCS and edit-distance recurrences from the last-character decision - 4 min
- [ ] LIS: the O(n²) DP, then the O(n log n) tails-array upgrade and its invariant - 5 min
- [ ] Write the backtracking template and say how subsets vs combinations vs permutations change the tree - 4 min
- [ ] Give the exchange argument for interval scheduling and the overlap test for greedy vs DP - 4 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
