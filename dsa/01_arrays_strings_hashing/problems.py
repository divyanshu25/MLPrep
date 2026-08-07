"""17 classic array/string/hashing problems, each solved with the pattern derived
in notes.md. Read the docstring, attempt on paper, then read the code - the key
insight is in the comment above each implementation.

Run: uv run python dsa/01_arrays_strings_hashing/problems.py
"""

import heapq
from collections import Counter, defaultdict, deque


# ---------------------------------------------------------------------------
# 1. Two Sum - pattern: hashing (value -> index)
# ---------------------------------------------------------------------------
def two_sum(nums, target):
    """Given a list of integers nums and an integer target, return indices
    [i, j], i < j, with nums[i] + nums[j] == target. Exactly one solution
    exists, and the same element may not be used twice (the same value at two
    different indices is fine).

    Example:
        two_sum([2, 7, 11, 15], 9) -> [0, 1]   # 2 + 7 == 9
        two_sum([3, 3], 6) -> [0, 1]           # equal values, distinct indices

    Note: nums may contain negative numbers.

    Pattern: hashmap. Key insight: while scanning, asking "does target - x
    exist among earlier elements?" is an O(1) lookup instead of an O(n) scan
    - the hashmap turns the O(n^2) pair search into one pass.
    Complexity: O(n) time, O(n) space.
    """
    seen = {}  # value -> index
    for j, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], j]
        seen[x] = j  # insert AFTER the query so we never pair x with itself
    return []


# ---------------------------------------------------------------------------
# 2. Longest Substring Without Repeating Characters - pattern: sliding window
# ---------------------------------------------------------------------------
def longest_unique_substring(s):
    """Given a string s, return the length of the longest contiguous substring
    in which no character repeats. Return 0 for the empty string.

    Example:
        longest_unique_substring("abcabcbb") -> 3   # "abc"
        longest_unique_substring("abba") -> 2       # "ab" or "ba"

    Pattern: variable sliding window. Key insight: the predicate "no repeats"
    is monotone under shrinking, so the smallest valid left end never moves
    left - l only advances, giving O(n) despite the nested while.
    Complexity: O(n) time, O(min(n, alphabet)) space.
    """
    window = set()
    l = best = 0
    for r, ch in enumerate(s):
        while ch in window:      # restore the invariant: no duplicates
            window.remove(s[l])
            l += 1
        window.add(ch)
        best = max(best, r - l + 1)
    return best


# ---------------------------------------------------------------------------
# 3. Container With Most Water - pattern: two pointers
# ---------------------------------------------------------------------------
def max_area(height):
    """Given a list height where height[i] is a vertical line at x = i, choose
    two lines that together with the x-axis form the container holding the most
    water: area = (r - l) * min(height[l], height[r]). Return that maximum
    area.

    Example:
        max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) -> 49   # i=1, i=8: width 7 * height 7
        max_area([1, 1]) -> 1

    Constraints: len(height) >= 2; the container may not be tilted.

    Pattern: two pointers. Key insight (elimination argument): if
    h[l] < h[r], every container (l, x) with x < r is capped by h[l] and is
    narrower, so (l, r) already realized the best container using l - index l
    is provably exhausted and can be discarded. n steps kill n^2 pairs.
    Complexity: O(n) time, O(1) space.
    """
    l, r = 0, len(height) - 1
    best = 0
    while l < r:
        best = max(best, (r - l) * min(height[l], height[r]))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return best


# ---------------------------------------------------------------------------
# 4. 3Sum - pattern: sort + two pointers
# ---------------------------------------------------------------------------
def three_sum(nums):
    """Given a list of integers nums, return all unique triplets of values (three
    distinct indices) that sum to 0. Each triplet must be sorted ascending, no
    triplet may repeat, and the order of triplets in the output does not matter.
    Return [] if none exist.

    Example:
        three_sum([-1, 0, 1, 2, -1, -4]) -> [[-1, -1, 2], [-1, 0, 1]]
        three_sum([0, 0, 0, 0]) -> [[0, 0, 0]]   # duplicates collapse to one

    Pattern: sort, fix the smallest element, then two-pointer the rest for a
    pair summing to -nums[i]. Key insight: sorting makes duplicates adjacent
    (skip equal neighbors for uniqueness) and enables the pair elimination
    argument. Complexity: O(n^2) time (n pointer sweeps), O(1) extra space
    beyond the output; the sort is O(n log n), dominated by n^2.
    """
    nums = sorted(nums)
    n, out = len(nums), []
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue                       # skip duplicate anchors
        if nums[i] > 0:
            break                          # smallest element positive -> no zero sums
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                out.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1                 # skip duplicate pairs
    return out


# ---------------------------------------------------------------------------
# 5. Product of Array Except Self - pattern: prefix/suffix products
# ---------------------------------------------------------------------------
def product_except_self(nums):
    """Given a list of integers nums, return a list out where out[i] is the
    product of every element of nums except nums[i]. Division is not allowed,
    and the solution must run in O(n) time.

    Example:
        product_except_self([1, 2, 3, 4]) -> [24, 12, 8, 6]
        product_except_self([-1, 1, 0, -3, 3]) -> [0, 0, 9, 0, 0]   # one zero

    Note: nums may contain zeros and negative numbers.

    Pattern: prefix sums, multiplicative flavor. Key insight: the answer at i
    is (product of the prefix left of i) * (product of the suffix right of i);
    build prefixes left-to-right into out, then sweep right-to-left carrying a
    running suffix product. Complexity: O(n) time, O(1) space beyond output.
    """
    n = len(nums)
    out = [1] * n
    for i in range(1, n):                  # out[i] = product of nums[0..i-1]
        out[i] = out[i - 1] * nums[i - 1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]
    return out


# ---------------------------------------------------------------------------
# 6. Subarray Sum Equals K - pattern: prefix sum + hashmap
# ---------------------------------------------------------------------------
def subarray_sum(nums, k):
    """Given a list of integers nums and an integer k, return the number of
    non-empty contiguous subarrays whose elements sum to exactly k. Subarrays
    are counted by position, so identical contents at different positions count
    separately.

    Example:
        subarray_sum([1, 1, 1], 2) -> 2    # [1,1] starting at 0 and at 1
        subarray_sum([1, -1, 0], 0) -> 3   # [1,-1], [0], and [1,-1,0]

    Note: negatives and zeros are allowed, so a running sum can repeat.

    Pattern: prefix sum + hashmap. Key insight: sum(i..j) = pre[j+1] - pre[i],
    so a subarray ending here has sum k iff some earlier prefix equals
    (current prefix - k) - a counting lookup. Seed {0:1} for the empty prefix.
    Complexity: O(n) time, O(n) space.
    """
    count = 0
    run = 0
    seen = defaultdict(int)
    seen[0] = 1
    for x in nums:
        run += x
        count += seen[run - k]  # query BEFORE inserting the current prefix
        seen[run] += 1
    return count


# ---------------------------------------------------------------------------
# 7. Longest Consecutive Sequence - pattern: hashing (set membership)
# ---------------------------------------------------------------------------
def longest_consecutive(nums):
    """Given an unsorted list of integers nums (duplicates possible), return the
    length of the longest run of consecutive integer VALUES, e.g. 1, 2, 3, 4  - 
    the values need not sit next to each other in the list. Must run in O(n);
    return 0 for an empty list.

    Example:
        longest_consecutive([100, 4, 200, 1, 3, 2]) -> 4   # the run 1, 2, 3, 4
        longest_consecutive([]) -> 0

    Pattern: hash set. Key insight: only start counting from a run's LEFT end
    (x such that x-1 is absent); then each element is visited O(1) times total
    - the inner while walks each run exactly once, so O(n) overall despite the
    nested loop. Complexity: O(n) time, O(n) space.
    """
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 in s:
            continue                       # not a run start
        y = x
        while y + 1 in s:
            y += 1
        best = max(best, y - x + 1)
    return best


# ---------------------------------------------------------------------------
# 8. Group Anagrams - pattern: canonical hashable key
# ---------------------------------------------------------------------------
def group_anagrams(strs):
    """Given a list of lowercase words strs, group the words that are anagrams
    of each other and return the groups as a list of lists. Every word lands in
    exactly one group; the order of groups, and of words within a group, does
    not matter.

    Example:
        group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
            -> [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]  # any order
        group_anagrams([""]) -> [[""]]

    Constraints: words use only lowercase a-z and may be empty strings.

    Pattern: frequency counting. Key insight: anagram-ness is equality of
    character multisets, so a canonical form of the multiset (26-tuple of
    counts) is a valid dict key - grouping becomes one pass of key -> bucket.
    Complexity: O(total chars) time, O(total chars) space.
    """
    groups = defaultdict(list)
    for w in strs:
        key = [0] * 26
        for ch in w:
            key[ord(ch) - ord("a")] += 1
        groups[tuple(key)].append(w)
    return list(groups.values())


# ---------------------------------------------------------------------------
# 9. Trapping Rain Water - pattern: two pointers
# ---------------------------------------------------------------------------
def trap(height):
    """Given a list height of non-negative bar heights (each bar has width 1),
    return the total units of water trapped between the bars after it rains.
    Return 0 for an empty list or a profile that holds nothing.

    Example:
        trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) -> 6
        trap([3, 2, 1]) -> 0   # strictly downhill traps nothing

    Pattern: two pointers. Key insight: water above bar i is
    min(max_left, max_right) - h[i]. Working from the side with the SMALLER
    running max, that side's max is guaranteed to be the binding minimum
    (the other side already has something at least as tall), so the cell's
    water is decided without knowing the far side exactly.
    Complexity: O(n) time, O(1) space.
    """
    if not height:
        return 0
    l, r = 0, len(height) - 1
    max_l, max_r = height[l], height[r]
    water = 0
    while l < r:
        if max_l <= max_r:                 # left max is the binding bound
            l += 1
            max_l = max(max_l, height[l])
            water += max_l - height[l]
        else:
            r -= 1
            max_r = max(max_r, height[r])
            water += max_r - height[r]
    return water


# ---------------------------------------------------------------------------
# 10. Minimum Window Substring - pattern: sliding window + frequency counts
# ---------------------------------------------------------------------------
def min_window(s, t):
    """Given strings s and t, return the smallest contiguous substring of s that
    contains every character of t with at least t's multiplicity (extra
    characters in the window are fine). Return "" if no such window exists.

    Example:
        min_window("ADOBECODEBANC", "ABC") -> "BANC"
        min_window("aa", "aa") -> "aa"   # multiplicity counts: one 'a' won't do

    Pattern: variable sliding window, minimizing flavor: expand r until the
    window covers t, then shrink l while coverage holds, recording the best.
    Key insight: track a single `have` counter of satisfied characters
    (need[ch] fulfilled) so validity checks are O(1), not O(alphabet).
    Complexity: O(|s| + |t|) time, O(alphabet) space.
    """
    if not t or len(t) > len(s):
        return ""
    need = Counter(t)
    window = defaultdict(int)
    have, want = 0, len(need)
    best = (float("inf"), 0, 0)            # (length, l, r)
    l = 0
    for r, ch in enumerate(s):
        window[ch] += 1
        if window[ch] == need[ch]:         # this char just became satisfied
            have += 1
        while have == want:                # valid -> shrink for minimality
            if r - l + 1 < best[0]:
                best = (r - l + 1, l, r)
            window[s[l]] -= 1
            if s[l] in need and window[s[l]] < need[s[l]]:
                have -= 1
            l += 1
    return "" if best[0] == float("inf") else s[best[1]: best[2] + 1]


# ---------------------------------------------------------------------------
# 11. Merge Intervals - pattern: sort + linear sweep
# ---------------------------------------------------------------------------
def merge_intervals(intervals):
    """Given a list of intervals [start, end] in any order, merge every group of
    overlapping intervals and return the result sorted by start, all intervals
    disjoint. Intervals that merely touch (one's end == the next's start) also
    merge.

    Example:
        merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
            -> [[1, 6], [8, 10], [15, 18]]
        merge_intervals([[1, 4], [4, 5]]) -> [[1, 5]]   # touching merges

    Pattern: sort by start, then sweep. Key insight: after sorting, an
    interval can only overlap the LAST merged interval (anything earlier
    ended before that one did), so one pass with "extend or append" suffices.
    Complexity: O(n log n) time (sort), O(n) space for output.
    """
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)   # overlap -> extend
        else:
            out.append([start, end])            # gap -> new interval
    return out


# ---------------------------------------------------------------------------
# 12. Sliding Window Maximum - pattern: monotonic deque
# ---------------------------------------------------------------------------
def max_sliding_window(nums, k):
    """Given a list of integers nums and a window size k (1 <= k <= len(nums)),
    return the maximum of every contiguous window of size k, scanning left to
    right. The output has len(nums) - k + 1 entries. Aim for O(n), not O(nk).

    Example:
        max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) -> [3, 3, 5, 5, 6, 7]
        max_sliding_window([1, 2, 3, 4], 4) -> [4]   # k == n: one window

    Pattern: monotonic (decreasing) deque of indices. Key insight: when x
    enters, any smaller element already in the deque can never be a future
    window max (x is newer AND bigger) - pop it forever. Each index is pushed
    and popped at most once, so O(n) total; the deque front is always the
    current window's max. Complexity: O(n) time, O(k) space.
    """
    dq = deque()                            # indices; values decreasing
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:     # smaller-and-older: dead forever
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                  # front slid out of the window
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


# ---------------------------------------------------------------------------
# 13. Insert Interval - pattern: three-phase sweep over sorted intervals
# ---------------------------------------------------------------------------
def insert_interval(intervals, new_interval):
    """Given a list of disjoint intervals sorted by start and a new_interval
    [s, e], insert the newcomer and merge any overlaps so the result is again
    sorted and disjoint. Return the new list; the input list may be empty.

    Example:
        insert_interval([[1, 3], [6, 9]], [2, 5]) -> [[1, 5], [6, 9]]
        insert_interval([[1, 5]], [2, 3]) -> [[1, 5]]   # newcomer swallowed

    Pattern: three phases (notes.md pattern 7). Key insight: relative to
    [s, e], the sorted input splits into strictly-left (end < s), overlapping
    (start <= running e - each folds into the newcomer), and strictly-right;
    the phase boundaries are monotone, so one pass suffices and no re-sort is
    needed. Complexity: O(n) time, O(n) space for output.
    """
    s, e = new_interval
    out = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < s:        # phase 1: strictly left
        out.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= e:       # phase 2: overlap - fold in
        s = min(s, intervals[i][0])
        e = max(e, intervals[i][1])             # RUNNING e, not the original
        i += 1
    out.append([s, e])
    out.extend(intervals[i:])                   # phase 3: strictly right
    return out


# ---------------------------------------------------------------------------
# 14. Meeting Rooms I - pattern: sort + adjacent overlap check
# ---------------------------------------------------------------------------
def can_attend_meetings(intervals):
    """Given a list of meeting intervals [start, end] in any order, return True
    iff one person can attend every meeting, i.e. no two meetings overlap. A
    meeting ending exactly when the next starts is NOT an overlap. An empty
    schedule returns True.

    Example:
        can_attend_meetings([[0, 30], [5, 10], [15, 20]]) -> False  # [0,30] clashes
        can_attend_meetings([[1, 2], [2, 3]]) -> True               # touching ok

    Pattern: sort by start. Key insight: after sorting, any overlap must show
    up between ADJACENT meetings - an earlier meeting reaching a later one
    would reach (or engulf) everything in between first - so n-1 local checks
    replace n^2/2 pairwise ones. Touching endpoints (end == next start) are
    fine. Complexity: O(n log n) time, O(1) extra space.
    """
    intervals = sorted(intervals)
    return all(intervals[i][0] >= intervals[i - 1][1]
               for i in range(1, len(intervals)))


# ---------------------------------------------------------------------------
# 15. Meeting Rooms II - pattern: min-heap of end times
# ---------------------------------------------------------------------------
def min_meeting_rooms(intervals):
    """Given a list of meeting intervals [start, end] in any order, return the
    minimum number of conference rooms needed so that no two overlapping
    meetings share a room. A room frees up exactly at its meeting's end time,
    so back-to-back meetings can reuse it. Return 0 for no meetings.

    Example:
        min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) -> 2   # [0,30] overlaps both
        min_meeting_rooms([[1, 2], [2, 3], [3, 4]]) -> 1       # back-to-back reuse

    Pattern: sweep by start + min-heap of end times (notes.md pattern 7).
    Key insight: the answer is the maximum number of meetings alive at once
    (each alive meeting needs its own room, and that many always suffice).
    Ask the earliest-freeing room (heap min): if even IT is busy at this
    start, no room is free - allocate. Heap size only grows exactly when
    concurrency hits a new high, so max heap size = the answer.
    Complexity: O(n log n) time, O(n) space.
    """
    ends = []                                   # end times of occupied rooms
    for start, end in sorted(intervals):
        if ends and ends[0] <= start:           # earliest room is free: reuse
            heapq.heapreplace(ends, end)
        else:
            heapq.heappush(ends, end)           # all busy: new room
    return len(ends)


# ---------------------------------------------------------------------------
# 16. Best Time to Buy and Sell Stock - pattern: one-pass running minimum
# ---------------------------------------------------------------------------
def max_profit(prices):
    """Given a list prices where prices[i] is a stock's price on day i, choose
    one day to buy and a strictly later day to sell so profit is maximized.
    Return that maximum profit, or 0 if every trade would lose money (you may
    decline to trade).

    Example:
        max_profit([7, 1, 5, 3, 6, 4]) -> 5   # buy at 1 (day 1), sell at 6 (day 4)
        max_profit([7, 6, 4, 3, 1]) -> 0      # prices only fall: don't trade

    Pattern: running minimum. Key insight: the best sell at day i uses the
    cheapest buy among days 0..i - a single running min. Track it while
    scanning and score prices[i] - min_so_far each day; every (buy, sell)
    pair is thereby considered without the O(n^2) enumeration.
    Complexity: O(n) time, O(1) space.
    """
    best = 0
    min_so_far = float("inf")
    for p in prices:
        min_so_far = min(min_so_far, p)         # cheapest buy so far
        best = max(best, p - min_so_far)        # best sale ending today
    return best


# ---------------------------------------------------------------------------
# 17. Maximum Subarray (Kadane) - pattern: extend-or-restart
# ---------------------------------------------------------------------------
def max_subarray(nums):
    """Given a non-empty list of integers nums (negatives allowed), return the
    largest sum achievable by any non-empty contiguous subarray. With an
    all-negative input the answer is the largest single element, not 0.

    Example:
        max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) -> 6   # [4, -1, 2, 1]
        max_subarray([-3, -1, -2]) -> -1                     # best single element

    Pattern: Kadane's algorithm. Key insight - derive the decision: let
    cur = best sum of a subarray ENDING exactly at i. That subarray either
    extends the best one ending at i-1 (worth cur + x) or starts fresh at i
    (worth x). Extending is better iff the previous cur is positive - a
    negative prefix can only drag x down, so drop it. Hence
    cur = max(cur + x, x), answer = max over all i (the best subarray ends
    somewhere). Complexity: O(n) time, O(1) space.
    """
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(cur + x, x)                   # extend, or restart at x
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. two_sum
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]          # duplicate values
    assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]  # negatives

    # 2. longest_unique_substring
    assert longest_unique_substring("abcabcbb") == 3
    assert longest_unique_substring("bbbbb") == 1
    assert longest_unique_substring("") == 0     # edge: empty
    assert longest_unique_substring("abba") == 2  # l must jump past repeat

    # 3. max_area
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert max_area([1, 1]) == 1
    assert max_area([4, 3, 2, 1, 4]) == 16
    assert max_area([1, 2, 1]) == 2

    # 4. three_sum
    assert sorted(three_sum([-1, 0, 1, 2, -1, -4])) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 1, 1]) == []
    assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]  # edge: all zeros, dedup
    assert three_sum([]) == []                     # edge: empty

    # 5. product_except_self
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]  # zero
    assert product_except_self([2, 3]) == [3, 2]

    # 6. subarray_sum
    assert subarray_sum([1, 1, 1], 2) == 2
    assert subarray_sum([1, 2, 3], 3) == 2
    assert subarray_sum([1, -1, 0], 0) == 3       # negatives and zeros
    assert subarray_sum([3], 3) == 1              # single element = k

    # 7. longest_consecutive
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert longest_consecutive([]) == 0           # edge: empty
    assert longest_consecutive([5]) == 1

    # 8. group_anagrams
    got = sorted(sorted(g) for g in group_anagrams(
        ["eat", "tea", "tan", "ate", "nat", "bat"]))
    assert got == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([""]) == [[""]]         # edge: empty string
    assert group_anagrams(["a"]) == [["a"]]

    # 9. trap
    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap([4, 2, 0, 3, 2, 5]) == 9
    assert trap([]) == 0                          # edge: empty
    assert trap([3, 2, 1]) == 0                   # monotone: traps nothing

    # 10. min_window
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""            # edge: impossible
    assert min_window("aa", "aa") == "aa"         # multiplicity matters

    # 11. merge_intervals
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == \
        [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]   # touching merge
    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]   # containment
    assert merge_intervals([[5, 6]]) == [[5, 6]]

    # 12. max_sliding_window
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    assert max_sliding_window([9, 8, 7, 6], 2) == [9, 8, 7]  # decreasing
    assert max_sliding_window([1, 2, 3, 4], 4) == [4]        # k == n

    # 13. insert_interval
    assert insert_interval([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]],
                           [4, 8]) == [[1, 2], [3, 10], [12, 16]]
    assert insert_interval([], [5, 7]) == [[5, 7]]           # edge: empty input
    assert insert_interval([[1, 5]], [2, 3]) == [[1, 5]]     # engulfed newcomer
    assert insert_interval([[3, 4]], [1, 2]) == [[1, 2], [3, 4]]  # no overlap

    # 14. can_attend_meetings
    assert not can_attend_meetings([[0, 30], [5, 10], [15, 20]])
    assert can_attend_meetings([[7, 10], [2, 4]])
    assert can_attend_meetings([[1, 2], [2, 3]])             # touching is fine
    assert can_attend_meetings([])                           # edge: no meetings

    # 15. min_meeting_rooms
    assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_meeting_rooms([[7, 10], [2, 4]]) == 1
    assert min_meeting_rooms([[1, 5], [2, 6], [3, 7]]) == 3  # all overlap
    assert min_meeting_rooms([[1, 2], [2, 3], [3, 4]]) == 1  # back-to-back
    assert min_meeting_rooms([]) == 0                        # edge: no meetings

    # 16. max_profit
    assert max_profit([7, 1, 5, 3, 6, 4]) == 5               # buy 1, sell 6
    assert max_profit([7, 6, 4, 3, 1]) == 0                  # only losses: skip
    assert max_profit([2, 4, 1, 7]) == 6                     # min AFTER a peak
    assert max_profit([5]) == 0                              # edge: one day

    # 17. max_subarray
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6   # [4,-1,2,1]
    assert max_subarray([1]) == 1
    assert max_subarray([5, 4, -1, 7, 8]) == 23              # take everything
    assert max_subarray([-3, -1, -2]) == -1                  # all negative

    print("all 17 problems passed")
