# Binary Search - From First Principles

Binary search is the shortest algorithm with the highest bug density in interviews.
The cure is not memorizing a template - it's holding a **loop invariant** so precisely
that the template writes itself. This file builds that, then generalizes to the view
that unifies half of LeetCode: *binary search on the answer*.

---

## 1. The loop invariant treatment

Take the fundamental form - not "find x" but **"find the first index where a
predicate becomes true"**. Assume `p` is monotone over indices: `F F F F T T T`
(once true, true forever). Find the leftmost `T`.

```python
lo, hi = 0, n          # candidate range [lo, hi]; hi = n means "maybe no T exists"
while lo < hi:
    mid = lo + (hi - lo) // 2
    if p(mid):
        hi = mid       # mid could be the answer: keep it in range
    else:
        lo = mid + 1   # mid is F: the answer is strictly right of mid
return lo              # == hi; equals n if no T exists
```

**The invariant, stated precisely:** at every iteration,

> every index `< lo` is F, and every index `≥ hi` is T (or `hi = n`, vacuously).
> The answer - the first T - always lies in `[lo, hi]`.

Check it's maintained: if `p(mid)` is true, then by monotonicity everything `≥ mid`
is T, so setting `hi = mid` keeps the invariant. If false, everything `≤ mid` is F,
so `lo = mid + 1` keeps it. At termination `lo == hi`, and the invariant pins that
single index as the boundary: first T, or `n` if none.

**Tiny trace.** `a = [1, 3, 5, 7]`, predicate `a[i] >= 5` (F F T T), so we're
finding the leftmost element ≥ 5 (`bisect_left`).
- lo=0, hi=4 → mid=2, a[2]=5 ≥ 5: T → hi=2
- lo=0, hi=2 → mid=1, a[1]=3 ≥ 5: F → lo=2
- lo == hi == 2. Answer index 2. ✓

**Why `lo < hi` here, and when `lo <= hi`?** Two coherent styles, and mixing them
is the bug factory:

- **Half-open boundary search (above):** range `[lo, hi)`, condition `lo < hi`,
  moves `hi = mid` / `lo = mid + 1`, answer is `lo` after the loop. Use for
  first/last-occurrence, search-on-answer, "find the boundary".
- **Closed-interval membership test:** range `[lo, hi]` with `hi = n - 1`,
  condition `lo <= hi`, moves `hi = mid - 1` / `lo = mid + 1`, `return mid` on an
  exact hit inside the loop. Use only for "is x present, return its index".

Pick one per problem and hold its invariant. The classic crash: writing `lo < hi`
with `hi = mid - 1` (can skip over the answer) or `lo <= hi` with `hi = mid`
(infinite loop - see bugs section).

**Why `mid = lo + (hi - lo) // 2`?** In fixed-width languages `(lo + hi) / 2`
overflows for large indices; the subtract form can't. Python ints don't overflow, so
`(lo + hi) // 2` is fine here - but write the safe form anyway and *say why*: it
signals you know the classic bug (this exact overflow lived in the JDK's
`Arrays.binarySearch` for nine years). What actually matters in Python: `//` rounds
**down**, so `mid` can equal `lo` but never `hi` (when `lo < hi`) - which is exactly
why `hi = mid` always shrinks the range and the loop terminates. A mirrored variant
that moves `lo = mid` must use the rounding-up `mid = lo + (hi - lo + 1) // 2` for
the same reason.

**Complexity, derived:** the range length drops from `L` to at most `⌈L/2⌉` each
iteration; it hits 1 after `⌈log₂ n⌉` halvings. O(log n) time, O(1) space.

---

## 2. Binary search on the answer - THE unifying view

Forget arrays. The general statement:

> You have a monotonic predicate `p` over a range of candidate answers:
> `p(x)` false for small `x`, true for all `x` past some threshold.
> Binary search finds the smallest `x` with `p(x)` true, using O(log range)
> *evaluations of p* - even when `p` costs O(n) each.

The array case is the special case `p(i) = a[i] >= target`. But nothing requires an
array - only monotonicity. This turns "optimize a min-max / max-min quantity" into
"binary search + write a feasibility checker":

- **Koko eating bananas:** `p(speed) = "can finish all piles within h hours at this
  speed"`. Faster ⇒ still finishes: monotone. Answer = smallest feasible speed.
- **Ship packages in D days:** `p(capacity) = "greedy loading fits in ≤ D days"`.
  Bigger ship ⇒ fewer or equal days: monotone.
- **Split array to minimize largest sum, aggressive cows, min days to make
  bouquets** - same shape.

**The recognition cue is the phrase "minimize the maximum" or "maximize the
minimum",** or any "smallest value such that ⟨checkable condition⟩". The interview
script: (1) name the candidate range `[lo, hi]` (its endpoints need justification  - 
e.g. speeds 1..max(piles)); (2) *prove monotonicity in one sentence* - this is the
step that licenses the whole method; (3) write the checker; (4) drop in the
boundary-search template unchanged. Complexity: O(checker · log range).

Note the checker is usually **greedy** ("just load packages until full, then start a
new day") - and the greedy is correct precisely because you only need a yes/no
feasibility answer, not an optimal packing.

---

## 3. Rotated sorted arrays

A sorted array rotated at an unknown pivot: `[4,5,6,7,0,1,2]`. Global order is
broken; what survives is: **cut anywhere, and at least one side is properly
sorted.** That's enough for O(log n), because a sorted half answers "is the target
in you?" in O(1) (range check on its endpoints) - and whichever question it answers,
you discard half.

```
a = [4,5,6,7,0,1,2], target 0
lo=0 hi=6 mid=3: a[lo..mid]=[4..7] sorted (a[lo]<=a[mid]); 0 not in [4,7] -> go right
lo=4 hi=6 mid=5: left half [0,1] sorted; 0 in [0,1] -> go left
lo=4 hi=4: found a[4]=0 ✓
```

**Find-minimum variant:** compare `a[mid]` to `a[hi]`. If `a[mid] > a[hi]`, the
"cliff" (rotation point) is strictly right of mid → `lo = mid + 1`; else the minimum
is at mid or left of it → `hi = mid`. This is exactly the boundary template with
predicate `p(i) = a[i] <= a[hi_original]` - F on the high shelf before the cliff, T
after. Comparing to `a[lo]` instead breaks on already-unrotated arrays; comparing to
`a[hi]` doesn't. Duplicates (`[3,3,1,3]`) can make `a[mid] == a[hi]` uninformative  - 
then `hi -= 1` is the only safe move, degrading worst case to O(n); say so.

---

## 4. Search on real numbers

Same invariant, but the range is continuous - there's no `mid + 1`, and you don't
stop at `lo == hi`. Iterate until the bracket is smaller than the tolerance (or a
fixed iteration count):

```python
lo, hi = 0.0, max(1.0, x)          # bracket the answer, e.g. sqrt(x)
for _ in range(100):               # 100 halvings: bracket shrinks by 2^100 - plenty
    mid = (lo + hi) / 2
    if mid * mid < x:
        lo = mid
    else:
        hi = mid
return lo
```

Invariant: `lo² ≤ x ≤ hi²` throughout - a continuous bisection on the monotone
predicate `mid² ≥ x`. Each step halves the bracket width: after k steps the error is
`(hi₀ − lo₀)/2ᵏ`, so ~50 iterations reach 1e-15 relative precision. A fixed
iteration count is safer than `while hi - lo > eps`, which can loop forever when
`eps` is below float resolution near large numbers. Same pattern: minimize
average/maximize distance with real-valued answers.

---

## 5. Common bugs - the failing case and the fix

**Bug 1: the `lo = mid` infinite loop.** With rounding-down mid, `mid` can equal
`lo`; if the branch then sets `lo = mid`, nothing changes:

```python
# find LAST index with a[i] <= target, a = [1, 2], target = 5
lo, hi = 0, 1
while lo < hi:
    mid = (lo + hi) // 2      # mid = 0
    if a[mid] <= 5: lo = mid  # lo = 0 -> lo = 0. Forever.
    else: hi = mid - 1
```

Fix: when the moves are `lo = mid` / `hi = mid - 1`, round **up**:
`mid = lo + (hi - lo + 1) // 2`. Rule of thumb - *the pointer assigned `mid` bare
must be unreachable by the rounding*: round down when `hi = mid`, up when `lo = mid`.
(Or sidestep entirely: find last-T as `first_index_of_F − 1`, reusing the one
round-down template.)

**Bug 2: off-by-one at the boundaries.** `hi = len(a) - 1` in the half-open
template silently excludes the answer `n` ("not found" / "insert at end"), and
`hi = len(a)` in the closed template reads `a[n]` out of bounds. Failing case for
the first: `a = [1, 2, 3]`, insert position of 4 - correct answer 3 is outside
`[0, 2]`, and the loop returns 2. The fix is discipline, not tweaking: write the
invariant comment first ("everything < lo is F, everything ≥ hi is T"), and the
initial values are forced - `[0, n)` must start with both regions empty.

**Bug 3: `lo <= hi` with `hi = mid`.** `lo == hi == mid` then loops forever  - 
`[5]`, any predicate: mid=0, `hi = 0`, state unchanged. The pairs that work:
`lo < hi` with `hi = mid`, or `lo <= hi` with `hi = mid - 1`. Never cross-breed.

**Bug 4: returning `mid` from a boundary search.** The loop's final `lo` is the
answer; whatever `mid` was last tested is one branch stale. Failing case: the trace
in section 1 - last mid computed is 1, answer is 2. Return `lo`.

**Bug 5 (search-on-answer): unjustified bracket.** Setting Koko's `lo = 0` crashes
the checker (division by zero) and speed 0 is meaningless; `lo = 1` is the smallest
sane candidate. Always state *why* the answer must lie in `[lo, hi]` - that
statement is half the correctness proof, and interviewers listen for it.

---

## What interviewers listen for

1. The invariant said aloud before coding ("left of lo is F, right of hi is T").
2. Monotonicity *proved*, not assumed - one sentence, before any search-on-answer.
3. Termination argument ("range shrinks every step because mid < hi").
4. The complexity split: O(log n) probes × O(cost per probe).
5. Edge cases named unprompted: empty array, target absent, all-T, all-F,
   duplicates, single element.

## Exit checklist

- [ ] Cold, blank editor: write bug-free classic binary search with the loop invariant stated before coding - 3 min
- [ ] Write the first-True boundary search (lower/upper bound) and say what each pointer is guaranteed to maintain - 4 min
- [ ] Frame "binary search on the answer": state the monotonic predicate for Koko-bananas or ship-capacity - 3 min
- [ ] Rotated sorted array: state the which-half-is-sorted decision rule and code the branch - 4 min
- [ ] Name the classic bugs (mid overflow/rounding, l < r vs l <= r, the mid = l update infinite loop) with fixes - 3 min
- [ ] Real-number search: set the termination rule (fixed iterations vs epsilon) and justify it - 2 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
