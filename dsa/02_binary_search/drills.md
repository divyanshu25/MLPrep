# Binary Search - Timed Drills

Protocol (from the [README](../README.md)): start a timer, attempt cold, no hints
until the box expires. Then take hints one at a time - each is a smaller give-away
than the next. Unfold the solution only after a real attempt, and put any miss on
your 2-day re-attempt list. For every drill: state the invariant and the predicate
OUT LOUD before writing the loop - that discipline is what this folder trains.

---

## Drill 1 - Search Insert Position · Easy · 10 min

Sorted distinct array, target. Return the index of target, or the index where it
would be inserted to keep the array sorted. Example: `[1,3,5,6]`, 5 → 2; 2 → 1;
7 → 4. O(log n) required.

<details><summary>Hint 1 (pattern)</summary>
This is the boundary-search template from notes.md §1, verbatim - no membership
test needed.
</details>
<details><summary>Hint 2 (invariant)</summary>
Predicate p(i) = a[i] >= target, which is F...F T...T. The insert position IS the
first true index - whether or not the target is present.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Half-open range [0, n) with hi starting at n (NOT n-1 - "insert at the end" must be
a reachable answer); `lo < hi`; `hi = mid` on true, `lo = mid + 1` on false;
return lo.
</details>
<details><summary>Solution</summary>

```python
def search_insert(nums, target):
    lo, hi = 0, len(nums)          # hi = n: "insert past the end" stays reachable
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] >= target:
            hi = mid               # mid could be the boundary
        else:
            lo = mid + 1           # everything <= mid is too small
    return lo
```

Complexity: O(log n) time, O(1) space. This is `bisect.bisect_left` - say so.
The classic miss is `hi = len(nums) - 1` (bug 2 in notes.md §5): test with
`([1, 2, 3], 4)`, expected 3.
</details>

---

## Drill 2 - Valid Perfect Square · Easy · 10 min

Given a positive integer `num`, return True iff it is a perfect square. No `sqrt`,
no floating point. Example: 16 → True, 14 → False.

<details><summary>Hint 1 (pattern)</summary>
Binary search on the answer over the integers 1..num: the "array" is the candidate
roots, no array needed.
</details>
<details><summary>Hint 2 (invariant)</summary>
p(x) = x*x >= num is monotone (F...F T...T). Find the first true x; then num is a
perfect square iff x*x == num exactly.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Boundary template on [1, num + 1); after the loop, check `lo * lo == num`. Python
ints don't overflow, but say why `mid * mid` would be the overflow site in C/Java.
</details>
<details><summary>Solution</summary>

```python
def is_perfect_square(num):
    lo, hi = 1, num + 1            # bracket: 1 <= root <= num for num >= 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid * mid >= num:
            hi = mid
        else:
            lo = mid + 1
    return lo * lo == num
```

Complexity: O(log num) time, O(1) space. Same skeleton solves sqrt(x)-rounded-down
(return `lo` or `lo - 1`) and, with the real-number variant of notes.md §4,
sqrt to a tolerance. Test 1 (True), 2 (False), and a large square like 2**40.
</details>

---

## Drill 3 - Split Array Largest Sum · Hard · 35 min

Split an array of non-negative integers into `k` non-empty **contiguous** parts,
minimizing the largest part-sum. Return that minimized largest sum.
Example: `[7,2,5,10,8]`, k=2 → 18 (split `[7,2,5] | [10,8]`).

<details><summary>Hint 1 (pattern)</summary>
"Minimize the maximum" - the exact recognition phrase from notes.md §2. Binary
search on the answer; the work is writing the feasibility checker.
</details>
<details><summary>Hint 2 (invariant)</summary>
p(cap) = "the array can be split into <= k parts, each with sum <= cap". Monotone:
a bigger cap can only make splitting easier (any valid split for cap is valid for
cap' > cap). Bracket: lo = max(nums) (the biggest element must fit somewhere),
hi = sum(nums) (one part takes all).
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
The checker is greedy: fill the current part until adding the next element would
exceed cap, then start a new part; count parts. Greedy is safe because you only
need a yes/no - packing each part as full as possible minimizes the part count for
this cap. Then the boundary template unchanged.
</details>
<details><summary>Solution</summary>

```python
def split_array(nums, k):
    def parts_needed(cap):
        parts, load = 1, 0
        for x in nums:
            if load + x > cap:     # this part is full: start a new one
                parts += 1
                load = 0
            load += x
        return parts

    lo, hi = max(nums), sum(nums)  # justify BOTH ends aloud
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if parts_needed(mid) <= k:
            hi = mid               # feasible: try tighter caps
        else:
            lo = mid + 1
    return lo
```

Complexity: O(n log(sum − max)) time, O(1) space. This is ship-packages
(`problems.py` #6) with the story stripped off - recognizing that identity IS the
skill. Why the returned `lo` is achievable (not just a bound): `lo` passed the
checker (it's the first true), and every value below it failed, so some split
realizes exactly <= k parts with max part-sum <= lo, and no split does better.
The O(nk·something) DP exists; mention it, then don't write it.
</details>

---

## Drill 4 - Time-Based Key-Value Store: the GET · Medium · 20 min

A store receives `set(key, value, timestamp)` calls with strictly increasing
timestamps per key. Implement `get(key, timestamp)`: the value with the largest
`ts <= timestamp`, or `""` if none. You hold `store[key] = [(ts, value), ...]` in
arrival order. (The full class design - why append-only lists stay sorted - is
folder 03's design-problems section; this drill is the search inside it.)

<details><summary>Hint 1 (pattern)</summary>
Arrival order + increasing timestamps means each key's list is ALREADY sorted by
ts - no sorting ever happens. "Largest element <= x" is a boundary search.
</details>
<details><summary>Hint 2 (invariant)</summary>
Find the first index with ts > timestamp (predicate ts_i > timestamp, monotone
F...F T...T); the answer sits one to its left. Index 0 there means "all entries are
from the future" - return "".
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
That first-true index is `bisect_right(pairs, timestamp, key=first element)` - or
hand-roll the round-down template with p(i) = pairs[i][0] > timestamp and return
index lo - 1.
</details>
<details><summary>Solution</summary>

```python
def get(store, key, timestamp):
    pairs = store.get(key, [])
    lo, hi = 0, len(pairs)
    while lo < hi:                  # first index with ts > timestamp
        mid = lo + (hi - lo) // 2
        if pairs[mid][0] > timestamp:
            hi = mid
        else:
            lo = mid + 1
    return pairs[lo - 1][1] if lo else ""
```

Complexity: O(log m) per get for m entries under the key, O(1) space. The
"last true" is found as "first false minus one" - reusing the single round-down
template instead of writing the fragile `lo = mid` round-up variant (notes.md §5,
bug 1). Test: query before the first ts (""), exactly on a ts, between two, after
the last.
</details>

---

## Drill 5 - Search in Rotated Sorted Array II (duplicates) · Medium · 25 min

Like the distinct-values rotated search (`problems.py` #2), but the array may
contain duplicates. Return True iff target is present.
Example: `[2,5,6,0,0,1,2]`, 0 → True. State the worst-case complexity honestly.

<details><summary>Hint 1 (pattern)</summary>
Same sorted-half elimination - but the test `nums[lo] <= nums[mid]` that identified
the sorted half can now be fooled. When, exactly?
</details>
<details><summary>Hint 2 (invariant)</summary>
When `nums[lo] == nums[mid] == nums[hi]`, the comparison carries zero information:
`[1,1,1,0,1]` and `[1,0,1,1,1]` are indistinguishable at those three probes, and
the target could be in either half. No O(1) test can pick a half to discard.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
In that one uninformative case, shrink by a step you can justify: `lo += 1` and
`hi -= 1` (safe - those positions equal nums[mid], which you already checked
against target). Otherwise proceed exactly as the distinct-values version. The
shrink step is why the worst case degrades to O(n): an all-equal array forces n/2
single steps.
</details>
<details><summary>Solution</summary>

```python
def search_rotated_dups(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return True
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1                        # both equal mid, already checked:
            hi -= 1                        # discard them, information-free case
        elif nums[lo] <= nums[mid]:        # left half genuinely sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                              # right half genuinely sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False
```

Complexity: O(log n) expected on "mostly distinct" inputs, **O(n) worst case**  - 
and that degradation is provably unavoidable: with all elements equal except one,
any algorithm must in the worst case probe linearly many positions to find the
odd one out. Saying that lower-bound sentence is the difference between knowing
the fix and understanding it.
</details>

---

## Drill 6 - Single Element in a Sorted Array · Medium · 20 min

A sorted array where every element appears exactly twice except one, which appears
once. Find it in O(log n) time, O(1) space.
Example: `[1,1,2,3,3,4,4,8,8]` → 2.

<details><summary>Hint 1 (pattern)</summary>
O(log n) demands a monotone predicate over indices. XOR-scan is O(n) - nice to
mention, not the answer. What global structure does the single element break?
</details>
<details><summary>Hint 2 (invariant)</summary>
Before the single element, pairs start at EVEN indices (a[0]=a[1], a[2]=a[3], ...);
after it, every pair is shifted and starts at an ODD index. "Pair alignment is
broken at or before index i" is monotone F...F T...T - the flip happens exactly at
the single element.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Probe only even mids (clear the low bit: `mid -= mid % 2`). If a[mid] == a[mid+1],
alignment still holds there - the single element is strictly right: lo = mid + 2.
Else it's at mid or left: hi = mid. Loop on `lo < hi`, return a[lo].
</details>
<details><summary>Solution</summary>

```python
def single_non_duplicate(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        mid -= mid % 2                 # probe pair starts: even indices only
        if nums[mid] == nums[mid + 1]:
            lo = mid + 2               # alignment intact: singleton is rightward
        else:
            hi = mid                   # alignment broken at or before mid
    return nums[lo]
```

Complexity: O(log n) time, O(1) space. Invariant: the singleton always lies in
[lo, hi] and lo stays even, so `nums[mid + 1]` never reads out of bounds (hi is the
last index, and mid < hi after evening). This drill is the purest example of the
notes' thesis: binary search needs no target and no comparisons against a value  - 
only a monotone predicate, here PARITY of pair alignment.
</details>
