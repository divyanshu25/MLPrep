# Arrays / Strings / Hashing - Timed Drills

Protocol (from the [README](../README.md)): start a timer, attempt cold, no hints
until the box expires. Then take hints one at a time - each is a smaller give-away
than the next. Unfold the solution only after a real attempt, and put any miss on
your 2-day re-attempt list. Talk out loud; state complexity before you code.

---

## Drill 1 - Valid Anagram of a Substring (a.k.a. Find All Anagrams) · Medium · 20 min

Given strings `s` and `p`, return the start indices of all substrings of `s` that
are anagrams of `p`. Example: `s="cbaebabacd"`, `p="abc"` → `[0, 6]`.

<details><summary>Hint 1 (pattern)</summary>
Fixed-size sliding window with frequency counts.
</details>
<details><summary>Hint 2 (invariant)</summary>
Maintain the character counts of the current length-|p| window; slide by add one
char / remove one char - O(1) updates, never rebuild the count.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Compare the window Counter to p's Counter after each slide; delete keys when their
count hits zero so `==` is honest. (Or keep a `matched` counter over 26 letters.)
</details>
<details><summary>Solution</summary>

```python
from collections import Counter

def find_anagrams(s, p):
    if len(p) > len(s):
        return []
    need = Counter(p)
    window = Counter(s[:len(p)])
    out = [0] if window == need else []
    for r in range(len(p), len(s)):
        window[s[r]] += 1
        l = r - len(p)
        window[s[l]] -= 1
        if window[s[l]] == 0:
            del window[s[l]]          # keep == comparison honest
        if window == need:
            out.append(l + 1)
    return out
```

Complexity: O(|s| · 26) time in this form - the `==` compares at most 26 keys  - 
i.e. O(|s|) with a constant-size alphabet; O(1) space. The `matched`-counter variant
makes the per-slide work strictly O(1).
</details>

---

## Drill 2 - Longest Repeating Character Replacement · Medium · 25 min

Given a string `s` and integer `k`, you may replace at most `k` characters. Return
the length of the longest substring you can make all-equal. Example: `s="AABABBA"`,
`k=1` → 4.

<details><summary>Hint 1 (pattern)</summary>
Variable sliding window over a "cost ≤ k" predicate.
</details>
<details><summary>Hint 2 (invariant)</summary>
A window is fixable iff (window length − count of its most frequent char) ≤ k.
Expand r; when violated, advance l. The predicate is monotone under shrinking, so
the window argument from notes.md applies.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Track `max_freq` = max count seen in any window so far and never decrease it when l
advances. A stale (too large) max_freq only makes the window look *more* valid than
it is, letting it stay at the best length already achieved - the answer (a max over
window lengths) is still correct because the window length never exceeds the best
achievable. This is the subtle bit interviewers probe.
</details>
<details><summary>Solution</summary>

```python
from collections import defaultdict

def character_replacement(s, k):
    count = defaultdict(int)
    l = max_freq = best = 0
    for r, ch in enumerate(s):
        count[ch] += 1
        max_freq = max(max_freq, count[ch])
        if (r - l + 1) - max_freq > k:   # window unfixable: slide (not shrink-to-valid)
            count[s[l]] -= 1
            l += 1
        best = max(best, r - l + 1)
    return best
```

Complexity: O(n) time, O(26) space. Note the `if` instead of `while`: the window
never shrinks below the best length found, it only slides - a legitimate variant
when you only need the maximum length, not each valid window.
</details>

---

## Drill 3 - Sort Colors (Dutch National Flag) · Medium · 20 min

Array of 0s, 1s, 2s. Sort it in place, one pass, O(1) space, without counting sort.

<details><summary>Hint 1 (pattern)</summary>
Swap-to-partition with TWO boundaries (notes.md pattern 6).
</details>
<details><summary>Hint 2 (invariant)</summary>
Maintain: a[0..lo-1] all 0s, a[lo..i-1] all 1s, a[hi+1..] all 2s, a[i..hi]
unexamined. Every step either advances i or shrinks hi.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
On a[i]==0 swap with lo, advance both; on 1 just advance i; on 2 swap with hi and
decrement hi but do NOT advance i - the swapped-in element is unexamined.
</details>
<details><summary>Solution</summary>

```python
def sort_colors(a):
    lo, i, hi = 0, 0, len(a) - 1
    while i <= hi:
        if a[i] == 0:
            a[lo], a[i] = a[i], a[lo]
            lo += 1
            i += 1               # swapped-in element from lo is a known 1 (or i==lo)
        elif a[i] == 2:
            a[i], a[hi] = a[hi], a[i]
            hi -= 1              # a[i] is now unexamined: don't advance i
        else:
            i += 1
```

Complexity: O(n) time (each step moves i or hi toward each other), O(1) space.
The classic bug is advancing `i` after the 2-swap - the failing case is `[2,0,1]`.
</details>

---

## Drill 4 - First Missing Positive · Hard · 30 min

Unsorted integer array (any values, any sign). Return the smallest positive integer
not present. O(n) time, O(1) extra space. Example: `[3,4,-1,1]` → 2.

<details><summary>Hint 1 (pattern)</summary>
The answer lies in 1..n+1, so this is a values-in-range problem: cyclic sort.
</details>
<details><summary>Hint 2 (invariant)</summary>
Send each value v in 1..n to its home index v−1 by repeated swaps; each swap homes
at least one element, so ≤ n swaps total. Ignore out-of-range values.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
`while 1 <= a[i] <= n and a[a[i]-1] != a[i]: swap a[i] to its home`. The duplicate
guard `a[a[i]-1] != a[i]` prevents infinite swap loops. Then scan for the first
i with a[i] != i+1.
</details>
<details><summary>Solution</summary>

```python
def first_missing_positive(a):
    n = len(a)
    for i in range(n):
        while 1 <= a[i] <= n and a[a[i] - 1] != a[i]:
            home = a[i] - 1
            a[i], a[home] = a[home], a[i]
    for i in range(n):
        if a[i] != i + 1:
            return i + 1
    return n + 1
```

Complexity: O(n) time - the nested while executes at most n times *total* because
every iteration homes one element permanently - and O(1) space.
</details>

---

## Drill 5 - Continuous Subarray Sum (multiple of k) · Medium · 25 min

Given an integer array and k, return True iff some subarray of length ≥ 2 has a sum
that is a multiple of k. Example: `[23,2,4,6,7]`, k=6 → True (`[2,4]`).

<details><summary>Hint 1 (pattern)</summary>
Prefix sums + hashmap - the "divisible by" cue from notes.md pattern 4.
</details>
<details><summary>Hint 2 (invariant)</summary>
sum(i..j) ≡ 0 (mod k) ⟺ pre[j+1] ≡ pre[i] (mod k). Store the FIRST index where
each residue appeared.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Map residue → earliest prefix index, seeded {0: 0} (pre[0] = 0). At prefix index
j+1 with residue r, answer True if the stored index for r is ≤ j+1 − 2 (length ≥ 2).
</details>
<details><summary>Solution</summary>

```python
def check_subarray_sum(nums, k):
    first = {0: 0}                 # residue -> earliest prefix index
    run = 0
    for j, x in enumerate(nums, start=1):   # j = prefix index
        run = (run + x) % k
        if run in first:
            if j - first[run] >= 2:
                return True
        else:
            first[run] = j         # keep the EARLIEST occurrence only
    return False
```

Complexity: O(n) time, O(min(n, k)) space. Two classic bugs: seeding {0:0} (misses
subarrays from index 0) and overwriting the residue's first index (shrinks lengths).
</details>

---

## Drill 6 - Encode and Decode Strings · Medium · 20 min

Design `encode(list[str]) -> str` and `decode(str) -> list[str]` so any list of
strings (which may contain ANY characters, including your delimiter) round-trips.
This is a reported practical-round question - graded on robustness, not cleverness.

<details><summary>Hint 1 (pattern)</summary>
No delimiter can be safe by itself - think length-prefix framing, like network
protocols.
</details>
<details><summary>Hint 2 (invariant)</summary>
Emit `f"{len(s)}#{s}"` per string. While decoding, the digits before the next '#'
are trusted as a length; the next `length` chars are consumed as payload verbatim,
so payload contents can never be confused with framing.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Decode with an index pointer: find '#' from the pointer, parse the int, slice the
payload, jump the pointer past it. No splitting, no escaping.
</details>
<details><summary>Solution</summary>

```python
def encode(strs):
    return "".join(f"{len(s)}#{s}" for s in strs)

def decode(text):
    out, i = [], 0
    while i < len(text):
        j = text.index("#", i)
        length = int(text[i:j])
        out.append(text[j + 1: j + 1 + length])
        i = j + 1 + length
    return out
```

Complexity: O(total chars) both ways, O(1) extra space beyond output. Test the
adversarial cases: empty list, empty string in the list, strings containing "#"
and digits (e.g. `["5#hello"]`).
</details>

---

## Drill 7 - Longest Palindromic Substring · Medium · 30 min

Return the longest palindromic substring of `s`. Example: `"babad"` → `"bab"` (or
`"aba"`).

<details><summary>Hint 1 (pattern)</summary>
Two pointers, expanding OUTWARD from a center instead of inward from the ends.
</details>
<details><summary>Hint 2 (invariant)</summary>
Every palindrome has a center: one of n single characters or n−1 gaps between
characters (even length). Expanding while s[l]==s[r] finds the maximal palindrome
at that center - 2n−1 centers, each expansion O(n).
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Write one helper `expand(l, r)` returning the widest palindrome bounds from that
seed; call it with (i, i) and (i, i+1) for every i and keep the longest.
</details>
<details><summary>Solution</summary>

```python
def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return l + 1, r          # last valid bounds, half-open
    best = (0, 0)
    for i in range(len(s)):
        for seed in (expand(i, i), expand(i, i + 1)):
            if seed[1] - seed[0] > best[1] - best[0]:
                best = seed
    return s[best[0]:best[1]]
```

Complexity: O(n²) time worst case (e.g. `"aaaa..."`), O(1) space. Manacher's
algorithm achieves O(n) but is not expected in interviews - *saying it exists* and
coding expand-around-center is the right move.
</details>

---

## Drill 8 - Subarrays With K Different Integers · Hard · 35 min

Given an integer array and k, count subarrays with EXACTLY k distinct values.
Example: `[1,2,1,2,3]`, k=2 → 7.

<details><summary>Hint 1 (pattern)</summary>
Sliding window - but "exactly k" is not a monotone predicate, so not directly.
Reduce it to two monotone problems.
</details>
<details><summary>Hint 2 (invariant)</summary>
exactly(k) = atMost(k) − atMost(k−1). "At most k distinct" IS monotone under
shrinking, so the standard window applies to each term.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
For atMost(k): maintain the window with ≤ k distinct; for each r, every left end in
[l, r] gives a valid subarray ending at r, so add (r − l + 1) to the count.
</details>
<details><summary>Solution</summary>

```python
from collections import defaultdict

def subarrays_with_k_distinct(nums, k):
    def at_most(k):
        if k < 0:
            return 0
        count = defaultdict(int)
        l = total = 0
        for r, x in enumerate(nums):
            count[x] += 1
            while len(count) > k:
                count[nums[l]] -= 1
                if count[nums[l]] == 0:
                    del count[nums[l]]
                l += 1
            total += r - l + 1     # subarrays ending at r with <= k distinct
        return total
    return at_most(k) - at_most(k - 1)
```

Complexity: O(n) time (two window passes), O(k) space. The exactly = atMost(k) −
atMost(k−1) reduction is a reusable trick - it converts any non-monotone "exactly"
constraint on windows into two monotone ones.
</details>
