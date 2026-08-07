# DP / Greedy / Backtracking — Timed Drills

Ten drills, roughly in increasing order of pain — drill 8 is the summit; drills
9 and 10 are shorter closers that must come out fast and clean. Rules of
engagement:

1. Set a real timer. The time box includes writing *and* hand-testing your code.
2. Work the checklist from `notes.md` §2 out loud: state in words → decision →
   base cases → order → answer location. If stuck, take hints **in order** —
   hint 1 names the pattern family, hint 2 gives the state, hint 3 the
   recurrence. Each hint you need is signal about what to re-study.
3. Only unfold the solution after the timer ends or your code passes your own
   test cases.

Warm-up (untimed, from notes §5.1): find 3-interval counterexamples showing that
interval scheduling by *earliest start* and by *shortest duration* both fail.
<details><summary>Warm-up answer</summary>

Earliest start fails on `[1,10], [2,3], [4,5]` — it takes the long one and keeps 1,
optimal keeps 2. Shortest duration fails on `[1,4], [3,6], [5,8]` — the short middle
interval `[3,6]` blocks both outer ones; optimal keeps the two outer intervals.
</details>

---

## Drill 1 — Decode Ways (25 min, medium)

A message of digits was encoded with A=1 … Z=26. Count the number of ways to
decode string `s` (e.g. `"226"` → 3: BZ, VF, BBF). `"0"` alone decodes to
nothing; `"06"` is invalid but `"10"` is J.

<details><summary>Hint 1 — pattern family</summary>
1-D linear DP, climbing-stairs-shaped: the last decoded letter consumed either
1 or 2 characters.
</details>

<details><summary>Hint 2 — state definition</summary>
`dp[i]` = number of ways to decode the prefix `s[:i]`. `dp[0] = 1` (empty
prefix, one way: decode nothing).
</details>

<details><summary>Hint 3 — recurrence</summary>
Decision on the last letter: it was the single digit `s[i-1]` (valid iff not
`'0'`, contributing `dp[i-1]`) or the two digits `s[i-2:i]` (valid iff in
10–26, contributing `dp[i-2]`). Sum the valid options.
</details>

<details><summary>Solution</summary>

```python
def num_decodings(s: str) -> int:
    if not s or s[0] == "0":
        return 0
    prev2, prev1 = 1, 1            # dp[0], dp[1]
    for i in range(2, len(s) + 1):
        cur = 0
        if s[i - 1] != "0":
            cur += prev1           # last letter = one digit
        if 10 <= int(s[i - 2 : i]) <= 26:
            cur += prev2           # last letter = two digits
        prev2, prev1 = prev1, cur
    return prev1

assert num_decodings("226") == 3
assert num_decodings("06") == 0
assert num_decodings("10") == 1
assert num_decodings("100") == 0   # the middle 0 pairs with 1, the last 0 with nothing
```

Exactly climbing stairs with validity guards on each step size. The zero cases
are the whole difficulty: a `'0'` can only exist as the tail of 10 or 20, and if
neither guard fires, `cur` becomes 0 and correctly poisons everything after.
**O(n) time, O(1) space** (n states × O(1) transition).
</details>

---

## Drill 2 — Maximum Product Subarray (25 min, medium)

Given an integer array (positives, negatives, zeros), return the largest product
of any non-empty contiguous subarray. `[2,3,-2,4]` → 6; `[-2,0,-1]` → 0;
`[-2,3,-4]` → 24.

<details><summary>Hint 1 — pattern family</summary>
1-D linear DP, Kadane-shaped ("best ending at i") — but the natural single
state is not enough. What does a negative number do to "best"?
</details>

<details><summary>Hint 2 — state definition</summary>
A negative flips the sign, so today's *worst* (most negative) product may
become tomorrow's best. Track two states: `hi[i]` / `lo[i]` = max / min product
of a subarray **ending exactly at** `i`.
</details>

<details><summary>Hint 3 — recurrence</summary>
Decision at `i`: the subarray ending here either starts here or extends the one
ending at `i-1`. So both `hi[i]` and `lo[i]` are the max/min of
`{nums[i], nums[i] * hi[i-1], nums[i] * lo[i-1]}`. Answer = max over `hi`.
</details>

<details><summary>Solution</summary>

```python
def max_product(nums: list[int]) -> int:
    hi = lo = best = nums[0]
    for x in nums[1:]:
        cands = (x, x * hi, x * lo)
        hi, lo = max(cands), min(cands)
        best = max(best, hi)
    return best

assert max_product([2, 3, -2, 4]) == 6
assert max_product([-2, 0, -1]) == 0
assert max_product([-2, 3, -4]) == 24
assert max_product([-2]) == -2
```

The lesson: when a transition can *invert* which candidate is desirable, widen
the state to carry both extremes. Zeros are handled for free — `x` alone is
always a candidate, so the run restarts. **O(n) time, O(1) space.**
</details>

---

## Drill 3 — Partition Equal Subset Sum (30 min, medium)

Can `nums` (positive ints) be split into two subsets with equal sums?
`[1,5,11,5]` → true (`{1,5,5}` and `{11}`); `[1,2,3,5]` → false.

<details><summary>Hint 1 — pattern family</summary>
0/1 knapsack in disguise: each number used at most once, and you're asking
whether some subset hits a target "capacity".
</details>

<details><summary>Hint 2 — state definition</summary>
If the total is odd, answer no. Otherwise target = total // 2, and
`dp[s]` = True iff some subset of the numbers seen so far sums to exactly `s`.
</details>

<details><summary>Hint 3 — recurrence</summary>
Take/skip on each number x: `dp[s] |= dp[s - x]`. It's 0/1, so iterate `s`
**downward** — upward lets x be reused (the §3.3 bug: `[1,5,5]` would "reach"
11 by using a 5 twice... actually it reaches any multiple; test `[1, 2]` target
4 to see 2 reused).
</details>

<details><summary>Solution</summary>

```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [True] + [False] * target      # dp[s]: some subset sums to s
    for x in nums:
        for s in range(target, x - 1, -1):   # downward: each x used once
            dp[s] = dp[s] or dp[s - x]
        if dp[target]:                        # early exit, nice to mention
            return True
    return dp[target]

assert can_partition([1, 5, 11, 5]) is True
assert can_partition([1, 2, 3, 5]) is False
assert can_partition([2, 2, 2]) is False     # odd count of equal halves
assert can_partition([1, 1]) is True
```

Boolean knapsack: value dropped, feasibility kept. The direction bug here is
silent — upward iteration returns True on inputs like `[1, 2]` with sum 3...
no odd guard trips first there, so test with `[2, 4]`: upward claims a subset
sums to 3? No — it claims `[1, 3]`-style reuse on e.g. `[1, 4]` (1 used thrice
to reach... ) — simplest concrete check: `[3, 8]` upward marks 6 reachable via
3+3. **O(n · target) time, O(target) space** — pseudo-polynomial, worth saying.
</details>

---

## Drill 4 — Longest Palindromic Substring (30 min, medium)

Return the longest contiguous palindromic substring of `s`. `"babad"` → `"bab"`
(or `"aba"`); `"cbbd"` → `"bb"`.

<details><summary>Hint 1 — pattern family</summary>
Interval-flavored 2-D DP on one string (`dp[i][j]` over substrings) — or skip
the table entirely with expand-around-center, which interviewers accept and
often prefer. Know both.
</details>

<details><summary>Hint 2 — state definition</summary>
DP version: `dp[i][j]` = True iff `s[i..j]` is a palindrome. Center version:
every palindrome has a center — one of n odd centers or n−1 even centers —
so enumerate centers and grow.
</details>

<details><summary>Hint 3 — recurrence</summary>
`dp[i][j] = (s[i] == s[j]) and (j - i < 2 or dp[i+1][j-1])` — the decision is
"do the outer characters match, and is the inside already a palindrome?".
Fill by increasing substring length so the inside is ready.
</details>

<details><summary>Solution</summary>

```python
def longest_palindrome(s: str) -> str:
    best = ""
    def expand(l: int, r: int) -> None:
        nonlocal best
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l, r = l - 1, r + 1
        if (r - 1) - (l + 1) + 1 > len(best):   # window overshot by one each side
            best = s[l + 1 : r]
    for c in range(len(s)):
        expand(c, c)        # odd-length centers
        expand(c, c + 1)    # even-length centers
    return best

assert longest_palindrome("babad") in ("bab", "aba")
assert longest_palindrome("cbbd") == "bb"
assert longest_palindrome("a") == "a"
assert longest_palindrome("ac") in ("a", "c")
```

Expand-around-center: **O(n²) time, O(1) space** — same time as the table but no
memory, and no evaluation-order subtlety. The table version matters as a warm-up
for true interval DP (drill 8), where "fill by increasing length" is mandatory.
</details>

---

## Drill 5 — Combination Sum (30 min, medium)

Given distinct positive `candidates` and a `target`, return **all unique
combinations** (unlimited reuse allowed) summing to target.
`([2,3,6,7], 7)` → `[[2,2,3], [7]]`.

<details><summary>Hint 1 — pattern family</summary>
Backtracking — the problem says "return all", so no DP summary can help
(notes §4.4). Reuse allowed + combinations (not permutations) — which template
knob handles each?
</details>

<details><summary>Hint 2 — state definition</summary>
A tree node is (current path, remaining target, `start` index). Combinations ⇒
keep the `start` index so each multiset is built in one canonical order; reuse
allowed ⇒ when choosing candidate `i`, recurse with `start = i`, NOT `i + 1`.
</details>

<details><summary>Hint 3 — recurrence</summary>
At each node, iterate `i` from `start`: choose `candidates[i]`, recurse with
`remaining - candidates[i]` and `start = i`, unchoose. Record on
`remaining == 0`; prune on `remaining < 0` (or sort candidates and `break` when
`candidates[i] > remaining`).
</details>

<details><summary>Solution</summary>

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()                    # enables the break-prune
    out: list[list[int]] = []
    path: list[int] = []

    def bt(start: int, remaining: int) -> None:
        if remaining == 0:
            out.append(path.copy())      # copy! path keeps mutating
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                    # sorted: everything after is too big
            path.append(candidates[i])   # choose
            bt(i, remaining - candidates[i])  # explore: start=i allows reuse
            path.pop()                   # unchoose

    bt(0, target)
    return out

assert combination_sum([2, 3, 6, 7], 7) == [[2, 2, 3], [7]]
assert combination_sum([2], 1) == []
assert combination_sum([3, 5], 8) == [[3, 5]]
```

The two template knobs, named: `start = i` (not `i+1`) encodes unlimited reuse;
having a `start` at all encodes combinations-not-permutations — this is §4.2's
distinction made executable, and the backtracking mirror of the coin-change loop
order trap. **O(branching^(target/min)) worst case** — exponential, as "return
all" demands; the sort-and-break prune is what keeps real inputs fast.
</details>

---

## Drill 6 — Best Time to Buy and Sell Stock with Cooldown (35 min, medium)

Prices `prices[i]` by day; unlimited transactions, but after selling you must
cool down one day before buying again. Maximize profit. `[1,2,3,0,2]` → 3
(buy 1, sell 2, cooldown, buy 0, sell 2).

<details><summary>Hint 1 — pattern family</summary>
1-D DP where the state needs more than the day index — notes §2 names this
exact problem: "the day, and whether I hold a share". A state machine.
</details>

<details><summary>Hint 2 — state definition</summary>
Three states per day: `hold` (own a share), `sold` (sold today — cooling down
tomorrow), `rest` (no share, free to buy). Each is the max cash achievable in
that condition after day i.
</details>

<details><summary>Hint 3 — recurrence</summary>
Decisions as transitions: `hold = max(hold, rest - price)` (keep holding, or
buy from rest — NOT from sold, that's the cooldown), `sold = hold + price`
(sell today), `rest = max(rest, prev_sold)` (idle, or finish cooling). Update
order matters: use yesterday's values on the right-hand sides.
</details>

<details><summary>Solution</summary>

```python
def max_profit_cooldown(prices: list[int]) -> int:
    hold, sold, rest = float("-inf"), 0, 0   # before day 0: can't be holding
    for p in prices:
        prev_sold = sold
        sold = hold + p                      # sell what we held
        hold = max(hold, rest - p)           # keep, or buy (only from rest!)
        rest = max(rest, prev_sold)          # idle, or cooldown completes
    return max(sold, rest)                   # never end holding a share

assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
assert max_profit_cooldown([1]) == 0
assert max_profit_cooldown([2, 1]) == 0     # falling prices: do nothing
assert max_profit_cooldown([1, 2, 4]) == 3  # no cooldown needed: one long hold
```

The design lesson: when a constraint ("cooldown") reaches across time steps,
widen the state until the constraint becomes a *local* rule about which
transitions exist — the arrow rest→hold exists, sold→hold doesn't. That's the
whole cooldown, expressed structurally. **O(n) time, O(1) space** (3 states ×
O(1) transitions).
</details>

---

## Drill 7 — Word Break II (35 min, hard)

Given `s` and a dictionary, return **all** sentences formed by segmenting `s`
into dictionary words. `("catsanddog", ["cat","cats","and","sand","dog"])` →
`["cats and dog", "cat sand dog"]`.

<details><summary>Hint 1 — pattern family</summary>
The instructive hybrid from notes §4.4: "return all" ⇒ backtracking, but the
feasibility question ("can this suffix be segmented at all?") is the word-break
DP — use it as pruning so dead suffixes are never explored.
</details>

<details><summary>Hint 2 — state definition</summary>
Backtrack on the start index: at position i, try every dictionary word that
prefixes `s[i:]`; a full segmentation is recorded when i reaches len(s).
Precompute `feasible[i]` = can `s[i:]` be segmented (the problems.py #9 DP run
on suffixes).
</details>

<details><summary>Hint 3 — recurrence</summary>
`bt(i)`: if not `feasible[i]`, return immediately (the prune). For each j with
`s[i:j]` in the dictionary and `feasible[j]`, choose the word, recurse on j,
unchoose. Without the prune, an adversarial input like `"aaaa...b"` with
dictionary `{"a","aa","aaa"}` explores exponentially many doomed prefixes.
</details>

<details><summary>Solution</summary>

```python
def word_break_all(s: str, word_dict: list[str]) -> list[str]:
    words = set(word_dict)
    max_len = max(map(len, words), default=0)
    n = len(s)

    feasible = [False] * (n + 1)             # feasible[i]: s[i:] segmentable
    feasible[n] = True
    for i in range(n - 1, -1, -1):
        feasible[i] = any(
            s[i:j] in words and feasible[j]
            for j in range(i + 1, min(n, i + max_len) + 1)
        )

    out: list[str] = []
    path: list[str] = []

    def bt(i: int) -> None:
        if i == n:
            out.append(" ".join(path))
            return
        for j in range(i + 1, min(n, i + max_len) + 1):
            if s[i:j] in words and feasible[j]:   # prune doomed suffixes
                path.append(s[i:j])               # choose
                bt(j)                             # explore
                path.pop()                        # unchoose

    bt(0)
    return out

assert sorted(word_break_all("catsanddog",
                             ["cat", "cats", "and", "sand", "dog"])) == \
    ["cat sand dog", "cats and dog"]
assert word_break_all("aaab", ["a", "aa", "aaa"]) == []   # prune earns its keep
assert word_break_all("ab", ["a", "b"]) == ["a b"]
```

The output itself can be exponential (think `"aaaa"` with `{"a","aa"}`), so no
algorithm beats exponential worst case — the DP prune's job is to make the work
proportional to the *actual* output, not the doomed search space. Saying that
distinction is the point of the drill. **O(n · max_len) for the DP; output-bound
for the search.**
</details>

---

## Drill 8 — Burst Balloons (40 min, hard)

Balloons `nums[0..n-1]`. Bursting balloon i scores
`nums[left] * nums[i] * nums[right]` where left/right are the *currently
adjacent* surviving balloons (out-of-bounds counts as 1). Burst all balloons,
maximize the total. `[3,1,5,8]` → 167.

<details><summary>Hint 1 — pattern family</summary>
Interval DP (notes §3.7). Left-to-right decisions don't decompose — bursting a
balloon changes both neighbors. The §3.7 key move: condition on the LAST event
in an interval, not the first.
</details>

<details><summary>Hint 2 — state definition</summary>
Pad: `a = [1] + nums + [1]`. `dp[i][j]` = best score for bursting everything
strictly BETWEEN i and j (exclusive endpoints, which stay alive as the walls).
</details>

<details><summary>Hint 3 — recurrence</summary>
Decide which balloon k in (i, j) bursts LAST: at that moment only i and j
survive beside it, so it scores `a[i]*a[k]*a[j]` — independent of the order
inside — and the two sides `(i,k)`, `(k,j)` become fully independent:
`dp[i][j] = max over k of dp[i][k] + dp[k][j] + a[i]*a[k]*a[j]`. Fill by
increasing gap j − i.
</details>

<details><summary>Solution</summary>

```python
def max_coins(nums: list[int]) -> int:
    a = [1] + nums + [1]
    m = len(a)
    dp = [[0] * m for _ in range(m)]         # dp[i][j]: burst all in (i, j)
    for gap in range(2, m):                  # smallest interval with an inside
        for i in range(m - gap):
            j = i + gap
            dp[i][j] = max(
                dp[i][k] + dp[k][j] + a[i] * a[k] * a[j]
                for k in range(i + 1, j)     # k bursts LAST in (i, j)
            )
    return dp[0][m - 1]

assert max_coins([3, 1, 5, 8]) == 167
assert max_coins([1, 5]) == 10
assert max_coins([7]) == 7
assert max_coins([]) == 0
```

Why "first burst" fails and "last burst" works: bursting k *first* leaves its
two sides coupled (they become adjacent to each other), but bursting k *last*
pins the walls of both subintervals to i and j for their entire lifetimes —
independence is manufactured by conditioning on the final event. This inversion
is the transferable idea (same trick: matrix-chain, polygon triangulation).
**O(n²) states × O(n) split points = O(n³) time, O(n²) space.**
</details>

---

## Drill 9 — Jump Game II (25 min, medium)

Same setup as jump game (`problems.py` #14) but reaching the last index is
guaranteed; return the **minimum** number of jumps. `[2,3,1,1,4]` → 2
(0 → 1 → 4).

<details><summary>Hint 1 — pattern family</summary>
Greedy — but see it as BFS in disguise (notes §5.1): layer j is the set of
indices reachable in j jumps. The answer is the layer containing the last
index, and layers are contiguous ranges, so no queue is needed.
</details>

<details><summary>Hint 2 — state definition</summary>
Track the current layer as a range `[.., cur_end]` and `furthest` = the
furthest index reachable from anywhere in the current layer — that furthest
value is exactly next layer's right edge.
</details>

<details><summary>Hint 3 — recurrence</summary>
Sweep i, folding `i + nums[i]` into `furthest`. When i hits `cur_end`, the
current layer is exhausted: jumps += 1 and `cur_end = furthest`. Don't process
the last index (no jump is made FROM it).
</details>

<details><summary>Solution</summary>

```python
def jump(nums: list[int]) -> int:
    jumps = cur_end = furthest = 0
    for i in range(len(nums) - 1):           # last index never jumps
        furthest = max(furthest, i + nums[i])
        if i == cur_end:                     # current BFS layer exhausted
            jumps += 1
            cur_end = furthest               # next layer's right edge
    return jumps

assert jump([2, 3, 1, 1, 4]) == 2
assert jump([1, 1, 1, 1]) == 3
assert jump([0]) == 0                        # already there
assert jump([2, 1]) == 1
```

Why layers are intervals (the fact that deletes the queue): if index x is
reachable in j jumps, so is every index < x at the same or smaller cost —
any path to x passes positions from which all earlier indices are also
reachable. So each BFS frontier is `[old_end + 1, furthest]`, and one scalar
per side represents it. Contrast with real BFS: same layer-counting proof of
minimality, O(1) space instead of a queue. **O(n) time, O(1) space.**
</details>

---

## Drill 10 — Palindromic Substrings (20 min, medium)

Count the palindromic substrings of `s` (each occurrence counts — same letters
at different positions are different substrings). `"abc"` → 3; `"aaa"` → 6.

<details><summary>Hint 1 — pattern family</summary>
Same machinery as drill 4 (longest palindromic substring): expand around
center — but now *count* every successful expansion instead of tracking the
widest.
</details>

<details><summary>Hint 2 — state definition</summary>
Every palindrome has a unique center: one of n single characters (odd length)
or n−1 gaps (even length) — 2n−1 centers. Each successful outward step at a
center is exactly one distinct palindromic substring.
</details>

<details><summary>Hint 3 — recurrence</summary>
`expand(l, r)`: while in bounds and `s[l] == s[r]`, add 1 and step outward.
Sum over all centers `(i, i)` and `(i, i+1)`. The DP-table alternative fills
`dp[i][j] = (s[i]==s[j]) and (j-i < 2 or dp[i+1][j-1])` and counts Trues —
same O(n²) time but O(n²) space, vs O(1) here.
</details>

<details><summary>Solution</summary>

```python
def count_substrings(s: str) -> int:
    def expand(l: int, r: int) -> int:
        found = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            found += 1                       # each expansion = one palindrome
            l, r = l - 1, r + 1
        return found

    return sum(expand(i, i) + expand(i, i + 1) for i in range(len(s)))

assert count_substrings("abc") == 3
assert count_substrings("aaa") == 6          # a,a,a,aa,aa,aaa
assert count_substrings("") == 0
assert count_substrings("abba") == 6         # +abba, +bb
```

The center enumeration is exhaustive and non-overlapping — every palindrome
has exactly one center, so nothing is missed or double-counted; that sentence
is the correctness proof. When to prefer the DP table instead: when a *later*
computation needs "is s[i..j] a palindrome?" as an O(1) oracle (e.g.
palindrome partitioning) — the table is a reusable artifact, the expansion is
a one-shot count. **O(n²) time, O(1) space** (Manacher's O(n) exists; name it,
don't write it).
</details>