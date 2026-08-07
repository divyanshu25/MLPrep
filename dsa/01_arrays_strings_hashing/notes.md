# Arrays, Strings, Hashing - Pattern Notes

Seven patterns generate essentially every array/string problem at the interview level.
For each: the invariant stated precisely, why the pattern is *correct* (not just what
it is), a tiny example traced by hand, the complexity derived, and the recognition
cues that tell you to reach for it.

---

## 1. Hashing - why lookup is O(1)

**What a hash table actually does.** It's an array of `m` buckets plus a hash
function `h(key) → {0..m-1}`. To store `key: value`, compute `i = h(key)` and put the
pair in bucket `i`; to look up, recompute `i` and check bucket `i`. Two keys can
collide (same bucket), so each bucket holds a small list (chaining). Lookup cost =
one hash computation + scanning one bucket. If the `n` stored keys spread evenly
across `m` buckets, a bucket holds `n/m` items on average - a constant, as long as we
keep `m` proportional to `n`.

**Amortized analysis in one paragraph.** How do we keep `m ∝ n` when `n` grows?
When the load factor `n/m` exceeds a threshold, allocate a table twice as big and
rehash everything - an O(n) operation. Doubling means that after a resize to size
`2m`, you get `m` cheap inserts before the next resize; the O(n) rehash cost, spread
over those `n/2` inserts that "paid" for it, is O(1) each. Total cost of `n` inserts:
n + (1 + 2 + 4 + ... + n) ≤ 3n, so **O(1) amortized** per operation. (Same argument
as Python's `list.append`.) Worst case per single operation is O(n) - adversarial
keys all colliding, or landing on the resize - which is why we say *expected,
amortized* O(1), not guaranteed.

**Tiny example.** `m = 4`, `h(k) = k % 4`. Insert 5, 9, 3: bucket 1 gets [5, 9]
(collision - both ≡ 1 mod 4), bucket 3 gets [3]. Lookup 9: hash → bucket 1, scan two
items, found. Lookup 7: hash → bucket 3, scan, `3 ≠ 7`, not present.

**What hashing buys you in problems.** It converts "have I seen X before?" and
"how many of X so far?" from O(n) scans into O(1) queries. Nearly every "find a pair /
count occurrences / detect duplicate" problem is a linear scan plus one of:
- `set` - membership (seen it?)
- `dict` value = index - membership + *where* (two sum)
- `dict` value = count - frequency (anagrams, ransom note)
- `dict` value = count of *prefix sums* - pattern 4 below.

**Recognition cues:** "find two/k elements that...", "count occurrences",
"is there a duplicate", "group by some property", any O(n²) brute force whose inner
loop is just a lookup.

**Common bugs:** mutating a dict while iterating it; using a list as a dict key
(unhashable - convert to `tuple`); forgetting that `Counter`/dict equality is your
friend for "same multiset?" questions.

---

## 2. Two pointers - the monotonic elimination argument

**The pattern:** two indices `l = 0`, `r = n-1` walk toward each other; each step,
some rule moves exactly one of them inward; stop when they meet. That's obviously
O(n) - the question a good interviewer probes is **why you're allowed to skip the
pairs you never look at**. There are ~n²/2 pairs; you inspect ~n of them. The move
rule must come with a *proof of safe elimination*: when you move `l` inward, you must
argue that **no pair `(l, x)` with `x ≤ r` can beat what you've already seen** - you
discard n candidates in one step, provably losing nothing.

**Worked example - pair with target sum in a sorted array.** `a = [1, 3, 5, 8]`,
target 11. Invariant: *if a solution exists, both its indices lie in `[l, r]`*.
- `l=0, r=3`: 1+8 = 9 < 11. Move `l`? We must show no pair `(0, x)` works: `a[0] +
  a[x] ≤ a[0] + a[r] = 9 < 11` for every `x ≤ r`, because `a[r]` is the **largest**
  remaining partner. All pairs involving index 0 are dead. `l ← 1`. Invariant holds.
- `l=1, r=3`: 3+8 = 11. Found.

The proof leans on sortedness: monotonicity is what lets one comparison kill a whole
row of the pair-matrix. That's the general shape: **two pointers works exactly when
one comparison gives you a monotone reason to discard everything on one side.**

For *container with most water* the same argument runs on `min`: if
`height[l] < height[r]`, every container `(l, x)` is capped by `height[l]` and is
*narrower* than `(l, r)`, so `(l, r)` was already the best container using `l`  - 
discard `l`. (Trapping rain water uses a similar bound-from-the-shorter-side
argument; see `problems.py`.)

**Complexity:** each step moves `l` or `r` inward by ≥ 1 and they never cross back,
so ≤ n steps: O(n) time, O(1) space.

**Recognition cues:** sorted input (or you can sort); "pair/triple with
sum/difference/property"; maximize something over pairs where a monotone bound
exists; palindromes; merging two sequences; in-place partitioning.

**Common bugs:** claiming two pointers on *unsorted* data with no elimination
argument (the classic wrong answer); duplicate handling in 3sum (skip equal
neighbors); off-by-one on the stop condition (`l < r` vs `l <= r` - for pairs you
want `l < r`).

---

## 3. Sliding window - invariant maintenance

**The view that makes it one idea, not twenty problems:** you maintain a window
`[l, r]` and a **predicate P** that the window must satisfy ("no repeated chars",
"≤ k zeros", "sum < target"). The loop is always:

```
for r in range(n):          # admit a[r] into the window
    ... update state ...
    while P is violated:    # restore the invariant
        evict a[l]; l += 1
    # invariant holds here → window [l, r] is the LARGEST valid window ending at r
answer = best over all r
```

**Why it's correct:** for the answer to be findable this way, P must be
**monotone under shrinking**: if a window is valid, every sub-window (or at least
every suffix of it) is valid too - equivalently, if `[l, r]` is *invalid*, then any
`[l', r]` with `l' < l` (a bigger window with the same right end) is also invalid.
Then for each `r`, the set of valid left ends is a suffix `{L_r, ..., r}`, and - key
fact - **L_r is non-decreasing in r** (growing the window can only break P, never fix
it, so the smallest valid `l` never moves left). That's why `l` can be a pointer that
only advances, and why the largest valid window ending at each `r` is exactly
`[L_r, r]`: nothing is missed.

**Complexity:** `r` advances n times, `l` advances at most n times total (it never
retreats), each advance does O(1) state updates: **O(n)**, even though there's a
nested `while`. Say this in the interview - "amortized: each index enters and leaves
the window once" - it's a favorite follow-up.

**Fixed vs variable window.** Fixed size k (max sum of k elements, anagram find):
no predicate needed, slide by add-one/remove-one, O(1) per step instead of
recomputing the window in O(k) - that recompute is the O(nk) brute force you're
beating. Variable: the invariant loop above.

**Tiny trace - longest substring without repeats,** `s = "abba"`. State: set of
chars in window.
- r=0 'a': {a}, window "a", best 1
- r=1 'b': {a,b}, "ab", best 2
- r=2 'b': violated → evict 'a' (l=1), still violated → evict 'b' (l=2), now {b}, best stays 2
- r=3 'a': {b,a}, "ba", best 2. Answer 2. Note l went 0→2, never back - monotone.

**When it fails - negatives with sums.** "Shortest subarray with sum ≥ k" with
negative numbers allowed: shrinking the window no longer monotonically decreases the
sum, so "invalid window stays invalid when grown" breaks, `L_r` is no longer
monotone, and the pointer argument collapses. A window you'd discard can become
valid again after admitting a negative-then-large pair. Fix: prefix sums + monotonic
deque (LC 862), or just recognize it's not a sliding-window problem. **Interview
checkpoint: before claiming sliding window, ask "are elements non-negative / is the
predicate monotone under shrinking?"**

**Recognition cues:** "longest/shortest **contiguous** subarray/substring such
that...", "max/min over every window of size k", stream-like phrasing. Contiguity is
the giveaway - subsequences (non-contiguous) are DP, not windows.

**Common bugs:** updating the answer *before* restoring the invariant; forgetting to
evict from frequency maps (decrement, and delete at zero if you compare dicts);
shrink condition `while` written as `if` (only evicts one element).

---

## 4. Prefix sums - range queries become two lookups

**The identity.** Define `pre[i] = a[0] + ... + a[i-1]` with `pre[0] = 0` (the empty
prefix - including it is what handles subarrays starting at index 0). Then any range
sum collapses:

```
sum(a[i..j]) = pre[j+1] - pre[i]
```

One O(n) pass builds `pre`; every range query afterward is O(1). You've traded a
per-query O(n) scan for two array reads - the same "precompute structure, answer
queries cheaply" move as hashing.

**+ hashmap: subarray sum equals k - derive it.** Count subarrays with sum exactly
k. Brute force: all O(n²) ranges. Instead, walk left to right maintaining the running
prefix `pre[j+1]`; a subarray ending at `j` and starting at `i` has sum k iff

```
pre[j+1] - pre[i] = k   ⟺   pre[i] = pre[j+1] - k
```

So at each `j`, the number of valid start points is *the number of earlier prefix
values equal to* `pre[j+1] - k` - a counting query, i.e. a hashmap of
`prefix value → times seen`, seeded with `{0: 1}` for the empty prefix.

**Tiny trace.** `a = [1, 2, 1, 3]`, k = 3. Seen = {0:1}, run = 0, count = 0.
- +1 → run 1. Need 1−3 = −2: 0 hits. Seen {0:1, 1:1}
- +2 → run 3. Need 0: **1 hit** ([1,2]). count 1. Seen {0:1, 1:1, 3:1}
- +1 → run 4. Need 1: **1 hit** ([2,1]). count 2. Seen {..., 4:1}
- +3 → run 7. Need 4: **1 hit** ([3]). count 3. ✓ (subarrays [1,2], [2,1], [3])

Works with negatives - no monotonicity needed, unlike sliding window. The same
skeleton with a set instead of a counter solves "subarray sum divisible by k"
(store `pre % k`) and "longest subarray with equal 0s and 1s" (map 0 → −1, look for
equal prefix values).

**Complexity:** O(n) time, O(n) space for the map.

**Recognition cues:** "subarray **sum/count/average** equals / divisible by /
at most k", many range queries on a static array, "equal number of X and Y in a
range". If elements can be negative and it's about sums → prefix+hashmap, not window.

**Common bugs:** forgetting to seed `{0: 1}` (misses subarrays starting at 0);
updating the map *before* querying at the same step (lets a subarray of length 0
count itself when k = 0 - query first, then insert).

---

## 5. String patterns - frequency counting and anagrams

**The core fact:** two strings are anagrams ⟺ they have the same character
multiset ⟺ their frequency maps are equal. So "anagram" questions are dict-equality
questions, and the canonical forms of a string are:
- `tuple(sorted(s))` - O(L log L) to build, works for any alphabet;
- a 26-tuple of counts - O(L), when the alphabet is known and small.

Either canonical form is a valid **hashable group key**: *group anagrams* is one pass
building `key → list of strings`, O(total chars) with the count key.

**Sliding a frequency window (find all anagrams of p in s):** fixed window of
`len(p)`; keep window counts, add the entering char, remove the leaving one, compare
to `p`'s counts. Naive compare is O(26) per slide - fine - or maintain a single
`matched` counter of how many of the 26 letters currently agree, for true O(1) slides.
The general skill: **frequency maps support O(1) incremental update**; never rebuild
a count from scratch inside a loop when one element changed.

**Recognition cues:** "anagram", "permutation of", "same letters", "can be built
from" (ransom note - one-sided count ≤ check), "first unique character".

**Common bugs:** comparing counts with `==` while stale zero-count entries linger
(delete keys at zero, or use `Counter` and compare after cleanup); assuming
lowercase-only without asking - say the assumption out loud.

---

## 6. In-place array tricks - swap-to-partition and cyclic sort

**Swap-to-partition.** Maintain a *boundary* invariant: `a[0..w-1]` holds the "good"
elements in order, `a[w..i-1]` the rejected ones, `a[i..]` unexamined. Scan `i`; when
`a[i]` is good, `swap(a[w], a[i]); w += 1`. One pass, O(1) extra space. Move-zeros
trace, `a = [0, 1, 0, 3]`: i=0 skip; i=1 swap→`[1,0,0,3]`, w=1; i=2 skip; i=3
swap→`[1,3,0,0]`, w=2. Done. This is the partition step of quicksort and the answer
to "remove/move X in place". The Dutch national flag (sort 0s/1s/2s) is the same idea
with two boundaries.

**Cyclic sort idea.** When the array holds a permutation-like range (values 1..n or
0..n), each value has a *home index* (`value − 1`). Repeatedly swap `a[i]` to its home
until `a[i]` is home or a duplicate sits there. Why O(n) despite nested loops: **every
swap puts at least one element into its final home**, and there are only n homes, so
≤ n swaps total across the whole array. Afterward, one scan reveals the anomaly: the
first index where `a[i] ≠ i+1` exposes the missing/duplicate number. This is the O(1)-
space answer to "first missing positive", "find the duplicate", "find all
disappeared numbers".

**Recognition cues:** "in place", "O(1) extra space", and *especially* "array
contains numbers in range 1..n" - that range phrase is nearly always cyclic sort (or
the sign-marking variant: negate `a[value-1]` as a "seen" bit).

**Common bugs:** in cyclic sort, advancing `i` after a swap that brought a new
unexamined element into position `i` (only advance when `a[i]` is home or swap target
equals `a[i]`); index/value off-by-one between 0-based indices and 1-based values.

---

## 7. Intervals - sort, then only ever look at the last one

**Why sorting is the whole trick.** An interval problem hands you n ranges
`[start, end]` in arbitrary order, and overlap is a pairwise question - n²/2 pairs.
Sorting **by start** collapses it to a *local* question. The invariant:

> After sorting by start, if interval `i` does not overlap the merged interval with
> the **largest end so far**, it overlaps nothing earlier at all.

Proof in one line: every earlier interval starts no later than `i` does, so the only
way any of them reaches `i` is by extending far enough right - and the furthest-right
reach among them is exactly the current merged interval's end. One comparison against
the last merged interval decides everything; that's why merge is a single sweep.

**Merge, traced.** `[[1,3],[2,6],[8,10]]` (already sorted): take [1,3]; 2 ≤ 3 →
overlap, extend end to max(3,6)=6 → [1,6]; 8 > 6 → gap, start a new interval. The
sweep is "extend or append", nothing else (code in `problems.py` #11). Touching
counts as overlap when the problem says so - decide `<=` vs `<` out loud.

**Insert interval - the three phases.** Given already-sorted disjoint intervals and
one new interval, don't re-sort and re-merge (O(n log n) for no reason). The sorted
input splits into three contiguous zones relative to the newcomer `[s, e]`:

1. **Strictly left:** every interval with `end < s` - copy through untouched.
2. **Overlapping:** every interval with `start <= e` - each one folds into the
   newcomer: `s = min(s, start)`, `e = max(e, end)`. They're consecutive because the
   input was sorted and disjoint.
3. **Strictly right:** everything else - copy through untouched.

The phase boundaries are monotone (once you leave a phase you never return), so one
pass, O(n). The classic bug is phase-2's condition: it's `start <= e` with the
*running* merged `e`, not the original one.

**Meeting Rooms I** (can one person attend all meetings?) is the degenerate case:
sort by start; any overlap at all means a conflict, and by the invariant it would
show up between *adjacent* intervals - check `a[i].start < a[i-1].end` for each i.

**Meeting Rooms II - minimum rooms, derived both ways.** The quantity being computed
is the **maximum number of meetings alive at once** (max concurrency). You can never
use fewer rooms than that - at the busiest moment, each alive meeting needs its own
room. Both standard solutions compute exactly this number:

- **Min-heap of end times.** Sweep meetings sorted by start; the heap holds the end
  times of meetings currently occupying rooms. A new meeting first asks the room
  that frees **earliest** (the heap min): if `heap[0] <= start`, reuse that room
  (pop, then push the new end); else every existing room is still busy - allocate
  one (push). Why the earliest-freeing room is the right one to ask: if *it* isn't
  free, no room is. The heap's size only grows at moments when the new meeting
  overlaps every currently-alive meeting - i.e. exactly when concurrency hits a new
  high - so max heap size = max concurrency. O(n log n).
- **Endpoint sweep (+1/−1).** Flatten to events: `(start, +1)` and `(end, −1)`,
  sort, running-sum. The running sum *is* the number of meetings alive; the answer
  is its maximum. Tie-break: at equal times process ends before starts (a meeting
  ending at t frees its room for one starting at t - matching the `<=` in the heap
  version). O(n log n), and it generalizes to "max overlap of anything" (calendar
  booking, CPU load, staffing).

Same count, two vocabularies: the heap simulates room *assignment* and its size
tracks concurrency; the sweep counts concurrency directly and skips the assignment.
If the interviewer asks "which meeting is in which room", only the heap version
answers.

**Recognition cues:** the word "intervals/meetings" itself; "merge", "insert",
"minimum rooms/resources", "maximum overlap", "free time". First move is always the
same - *sort (usually by start) and state what the sweep maintains*.

**Common bugs:** comparing against the current interval instead of the merged
`out[-1]` (misses chained merges like `[1,4],[2,3],[3,5]`); wrong tie handling at
touching endpoints; in insert-interval, forgetting the running `e` in the phase-2
condition; in the sweep, sorting `(t, +1)` before `(t, −1)` and inventing a phantom
room.

---

## Choosing under pressure - a 20-second triage

| Signal in the problem | Reach for |
|---|---|
| "have I seen / how many so far" | hashmap/set (1) |
| sorted + pairs/monotone bound | two pointers (2) |
| longest/shortest **contiguous**, non-negative or monotone predicate | sliding window (3) |
| subarray sums, negatives allowed, or "divisible by" | prefix sum + hashmap (4) |
| anagram / permutation / same letters | frequency map (5) |
| in place, O(1) space, values in 1..n | partition swap / cyclic sort (6) |
| intervals / meetings / max overlap | sort by start + sweep, heap for rooms (7) |

State the invariant, state the complexity, then code. In that order.

## Exit checklist

- [ ] Cold, blank editor: write the sliding-window skeleton with the invariant stated first, shrink-while loop included - 3 min
- [ ] State the monotonic-elimination argument for two pointers: why moving the losing pointer discards nothing optimal - 3 min
- [ ] Write prefix-sum + hashmap for "count subarrays summing to k", including the seed {0: 1} - 5 min
- [ ] Explain why hash lookup is O(1): buckets, load factor, resizing, and when it degrades - 2 min
- [ ] Write the anagram check via a 26-slot frequency array - 2 min
- [ ] Run the 20-second triage on three prompt cues (contiguous window / pair in sorted / in-place values 1..n) - 1 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
