"""8 binary search problems, each an instance of the invariant treatment in
notes.md: maintain "everything left of lo fails the predicate, everything right
of hi passes", and the search writes itself. The key insight comment above each
function names the predicate.

Run: uv run python dsa/02_binary_search/problems.py
"""


# ---------------------------------------------------------------------------
# 1. First and Last Position of Target — pattern: boundary search x2
# ---------------------------------------------------------------------------
def search_range(nums, target):
    """Sorted array with duplicates: return [first, last] index of target,
    or [-1, -1] if absent.

    Pattern: two boundary searches. Key insight: first occurrence is the
    smallest i with the monotone predicate a[i] >= target; the last is
    (smallest i with a[i] > target) - 1 — reuse ONE round-down template twice
    instead of writing a fragile round-up variant.
    Complexity: O(log n) time, O(1) space.
    """
    def first_true(pred):          # smallest i in [0, n] with pred(i) true
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if pred(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    left = first_true(lambda i: nums[i] >= target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    right = first_true(lambda i: nums[i] > target) - 1
    return [left, right]


# ---------------------------------------------------------------------------
# 2. Search in Rotated Sorted Array — pattern: sorted-half elimination
# ---------------------------------------------------------------------------
def search_rotated(nums, target):
    """Distinct sorted array rotated at an unknown pivot; return target's
    index or -1.

    Pattern: modified binary search. Key insight: any cut leaves at least one
    properly sorted half, and a sorted half answers "is target inside you?"
    with an O(1) range check — either way half the array is eliminated.
    Complexity: O(log n) time, O(1) space.
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:              # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                  # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


# ---------------------------------------------------------------------------
# 3. Find Minimum in Rotated Sorted Array — pattern: boundary search
# ---------------------------------------------------------------------------
def find_min_rotated(nums):
    """Minimum of a distinct sorted array rotated at an unknown pivot.

    Pattern: boundary search with predicate p(i) = nums[i] <= nums[-1] —
    false on the high shelf before the rotation cliff, true after. Key
    insight: compare mid to the RIGHT end (comparing to the left end breaks
    on unrotated input); a[mid] > a[hi] puts the cliff strictly right of mid.
    Complexity: O(log n) time, O(1) space.
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1                       # min strictly right of mid
        else:
            hi = mid                           # mid could be the min
    return nums[lo]


# ---------------------------------------------------------------------------
# 4. Search a 2D Matrix — pattern: flatten the index space
# ---------------------------------------------------------------------------
def search_matrix(matrix, target):
    """m x n matrix, rows sorted, each row's first element > previous row's
    last. Return True iff target present.

    Pattern: plain binary search on a virtual 1D array. Key insight: the
    sortedness condition means row-major order IS sorted order, so index k
    maps to matrix[k // n][k % n] and one search over m*n items suffices —
    no need for a two-phase row-then-column search.
    Complexity: O(log(m*n)) time, O(1) space.
    """
    m, n = len(matrix), len(matrix[0])
    lo, hi = 0, m * n - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        val = matrix[mid // n][mid % n]
        if val == target:
            return True
        if val < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


# ---------------------------------------------------------------------------
# 5. Koko Eating Bananas — pattern: binary search on the answer
# ---------------------------------------------------------------------------
def min_eating_speed(piles, h):
    """Koko eats at speed k bananas/hour; each hour she picks one pile and
    eats min(k, pile) from it. Find the smallest k finishing all piles in
    <= h hours.

    Pattern: search on the answer. Key insight: p(k) = "finishes within h at
    speed k" is monotone (faster never hurts), so find the smallest k where
    it flips to true. Hours for one pile = ceil(pile / k); bracket is
    [1, max(piles)] — speed above max(piles) can't help (one pile per hour).
    Complexity: O(n log max(piles)) time, O(1) space.
    """
    def feasible(k):
        return sum((p + k - 1) // k for p in piles) <= h

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# 6. Capacity to Ship Packages Within D Days — pattern: search on the answer
# ---------------------------------------------------------------------------
def ship_within_days(weights, days):
    """Packages must ship IN ORDER; a ship of capacity c loads a prefix of
    the remaining packages each day without exceeding c. Find the least
    capacity that ships everything within `days` days.

    Pattern: search on the answer with a greedy checker. Key insight:
    p(c) = "greedy loading needs <= days" is monotone in c; the greedy
    (pack until the next package would overflow, then start a new day) is
    valid because we only need feasibility, not an optimal packing. Bracket:
    lo = max(weights) (must fit the biggest single package), hi = sum(weights)
    (one day ships all). Complexity: O(n log sum) time, O(1) space.
    """
    def days_needed(c):
        d, load = 1, 0
        for w in weights:
            if load + w > c:
                d += 1
                load = 0
            load += w
        return d

    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if days_needed(mid) <= days:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# 7. Median of Two Sorted Arrays — pattern: binary search on a partition
# ---------------------------------------------------------------------------
def find_median_sorted_arrays(a, b):
    """Median of two sorted arrays in O(log(min(m, n))) time.

    Pattern: binary search on how many elements of the shorter array go to
    the combined left half. Key insight: a valid partition takes i from a and
    j = half - i from b such that every left element <= every right element;
    since j is determined by i, the ONE degree of freedom i can be binary
    searched — a[i-1] > b[j] means i is too big, b[j-1] > a[i] means too
    small. Sentinels (+-inf) handle empty sides.
    Complexity: O(log min(m, n)) time, O(1) space.
    """
    if len(a) > len(b):
        a, b = b, a                            # search over the shorter array
    m, n = len(a), len(b)
    half = (m + n + 1) // 2                    # size of the combined left part
    lo, hi = 0, m                              # i = elements of a on the left
    while lo <= hi:
        i = lo + (hi - lo) // 2
        j = half - i
        a_left = a[i - 1] if i > 0 else float("-inf")
        a_right = a[i] if i < m else float("inf")
        b_left = b[j - 1] if j > 0 else float("-inf")
        b_right = b[j] if j < n else float("inf")
        if a_left > b_right:
            hi = i - 1                         # too many from a
        elif b_left > a_right:
            lo = i + 1                         # too few from a
        else:                                  # valid partition found
            if (m + n) % 2:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2.0
    raise ValueError("inputs not sorted")


# ---------------------------------------------------------------------------
# 8. Find Peak Element — pattern: follow the uphill gradient
# ---------------------------------------------------------------------------
def find_peak_element(nums):
    """A peak is strictly greater than its neighbors (nums[-1] and nums[n]
    are conceptually -inf; adjacent elements differ). Return any peak's index
    in O(log n).

    Pattern: binary search without global order. Key insight: if
    nums[mid] < nums[mid+1] the slope goes up to the right, and walking
    uphill must hit a peak before the -inf wall at the end — so a peak
    surely exists in [mid+1, hi]; otherwise one exists in [lo, mid]. The
    invariant "a peak lies in [lo, hi]" holds even though the array is
    unsorted. Complexity: O(log n) time, O(1) space.
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1                       # uphill to the right
        else:
            hi = mid                           # mid could be the peak
    return lo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. search_range
    assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    assert search_range([], 0) == [-1, -1]            # edge: empty
    assert search_range([2, 2, 2], 2) == [0, 2]       # all equal
    assert search_range([1, 3], 4) == [-1, -1]        # past the end

    # 2. search_rotated
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search_rotated([1], 1) == 0                # single element
    assert search_rotated([1, 3], 3) == 1             # no rotation
    assert search_rotated([5, 1, 3], 5) == 0          # target at pivot side

    # 3. find_min_rotated
    assert find_min_rotated([3, 4, 5, 1, 2]) == 1
    assert find_min_rotated([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_min_rotated([11, 13, 15, 17]) == 11   # edge: not rotated
    assert find_min_rotated([2, 1]) == 1

    # 4. search_matrix
    assert search_matrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3)
    assert not search_matrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13)
    assert search_matrix([[1]], 1)                    # edge: 1x1
    assert not search_matrix([[1]], 2)

    # 5. min_eating_speed
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30   # h == len(piles)
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
    assert min_eating_speed([1], 1) == 1              # edge: single pile

    # 6. ship_within_days
    assert ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
    assert ship_within_days([3, 2, 2, 4, 1, 4], 3) == 6
    assert ship_within_days([1, 2, 3, 1, 1], 4) == 3
    assert ship_within_days([10], 1) == 10            # edge: single package

    # 7. find_median_sorted_arrays
    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
    assert find_median_sorted_arrays([], [1]) == 1.0  # edge: one empty
    assert find_median_sorted_arrays([2], []) == 2.0
    assert find_median_sorted_arrays([1, 2, 5], [3, 4, 6, 7]) == 4.0

    # 8. find_peak_element
    assert find_peak_element([1, 2, 3, 1]) == 2
    assert find_peak_element([1, 2, 1, 3, 5, 6, 4]) in (1, 5)  # any peak valid
    assert find_peak_element([1]) == 0                # edge: single element
    assert find_peak_element([1, 2, 3, 4]) == 3       # peak at the wall

    print("all 8 problems passed")
