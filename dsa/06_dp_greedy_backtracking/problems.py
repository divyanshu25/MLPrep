"""15 canonical DP / backtracking / greedy problems, solved the way notes.md
teaches them: state the state, derive the recurrence from a decision, then code.

Run:  uv run python dsa/06_dp_greedy_backtracking/problems.py
"""

from bisect import bisect_left


# ---------------------------------------------------------------- 1. climbing stairs
def climb_stairs(n: int) -> int:
    """You climb a staircase of n steps, taking 1 or 2 steps at a time.
    Return the number of distinct ways to reach the top.

    State: dp[k] = number of ways to climb exactly k steps.
    Decision: the LAST move was a 1-step (from k-1) or a 2-step (from k-2),
    so dp[k] = dp[k-1] + dp[k-2]; dp[0] = 1 (the empty climb).
    Space-optimized: the recurrence reads only the last two values.
    Time O(n), space O(1).
    """
    prev2, prev1 = 1, 1  # dp[0], dp[1]
    if n == 0:
        return 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1


# ------------------------------------------------------------------ 2. house robber
def rob(nums: list[int]) -> int:
    """Houses hold loot nums[i]; adjacent houses can't both be robbed.
    Return the maximum total loot.

    State: dp[i] = max loot obtainable from houses i..n-1.
    Decision at house i: rob it (nums[i] + dp[i+2]) or skip it (dp[i+1]).
    Filled right-to-left; compressed to two scalars (dp[i+1], dp[i+2]).
    Time O(n), space O(1).
    """
    ahead1 = ahead2 = 0  # dp[i+1], dp[i+2]
    for i in range(len(nums) - 1, -1, -1):
        ahead1, ahead2 = max(nums[i] + ahead2, ahead1), ahead1
    return ahead1


# ------------------------------------------------------------------- 3. coin change
def coin_change(coins: list[int], amount: int) -> int:
    """Unlimited coins of the given denominations. Return the fewest coins
    summing to `amount`, or -1 if impossible.

    State: dp[a] = fewest coins making exactly amount a.
    Decision: the LAST coin used was some c, so dp[a] = 1 + min(dp[a-c]).
    Unbounded supply falls out of the state not remembering coins used.
    Time O(amount * len(coins)), space O(amount).
    """
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1


# --------------------------------------------------- 4. longest common subsequence
def longest_common_subsequence(a: str, b: str) -> int:
    """Return the length of the longest subsequence common to strings a and b.

    State: dp[i][j] = LCS length of prefixes a[:i] and b[:j].
    Decision on the last characters: equal -> 1 + dp[i-1][j-1];
    unequal -> at least one is unused -> max(dp[i-1][j], dp[i][j-1]).
    Row i reads only row i-1, so keep two rows.
    Time O(nm), space O(m).
    """
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = 1 + prev[j - 1]
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


# ------------------------------------------------------------------ 5. edit distance
def edit_distance(a: str, b: str) -> int:
    """Return the minimum number of insert/delete/replace operations to turn
    string a into string b (Levenshtein distance).

    State: dp[i][j] = min ops turning a[:i] into b[:j].
    Decision (three-way, on the last operation): match/replace via
    dp[i-1][j-1], delete a's last char via dp[i-1][j], insert b's last char
    via dp[i][j-1]. Base cases: dp[i][0] = i, dp[0][j] = j.
    Time O(nm), space O(m).
    """
    prev = list(range(len(b) + 1))  # dp[0][j] = j
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)  # dp[i][0] = i
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


# ------------------------------------------- 6. longest increasing subsequence (x2)
def lis_quadratic(nums: list[int]) -> int:
    """Return the length of the longest strictly increasing subsequence.

    State: dp[i] = LIS length ENDING exactly at index i (pinning the endpoint
    encodes the last value, which extension needs).
    Decision: which j < i (with nums[j] < nums[i]) comes just before i.
    Answer is max(dp), since the LIS ends somewhere, not necessarily at n-1.
    Time O(n^2), space O(n).
    """
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def lis_nlogn(nums: list[int]) -> int:
    """Same problem in O(n log n) via patience sorting.

    Invariant: tails[k] = smallest possible tail value of any increasing
    subsequence of length k+1 seen so far; tails is strictly increasing.
    For each x, bisect_left finds the longest chain x can extend; overwrite
    the next slot (a better tail for that length) or append (a new record).
    len(tails) is the LIS length; tails itself is NOT an LIS.
    Time O(n log n), space O(n).
    """
    tails: list[int] = []
    for x in nums:
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
    return len(tails)


# ----------------------------------------------------------------- 7. 0/1 knapsack
def knapsack_2d(weights: list[int], values: list[int], capacity: int) -> int:
    """Each item used at most once; maximize value within the weight capacity.

    State: dp[i][c] = best value using only the FIRST i items with capacity c
    (the extra dimension records which items are still undecided).
    Decision at (i, c): skip item i (dp[i-1][c]) or take it
    (values[i-1] + dp[i-1][c - weights[i-1]]) -- both branches read row i-1,
    which is exactly what makes it 0/1 rather than unbounded.
    Time O(n * capacity), space O(n * capacity).
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                dp[i][c] = max(dp[i][c], v + dp[i - 1][c - w])
    return dp[n][capacity]


def knapsack_1d(weights: list[int], values: list[int], capacity: int) -> int:
    """1-D compression of knapsack_2d: row i reads only row i-1, so overwrite
    one array in place -- but iterate capacity DOWNWARD so dp[c - w] is still
    the previous item's row. Upward iteration would let an item be taken
    repeatedly (it silently becomes unbounded knapsack -- the classic bug).
    Time O(n * capacity), space O(capacity).
    """
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):  # downward: each item at most once
            dp[c] = max(dp[c], v + dp[c - w])
    return dp[capacity]


# ------------------------------------------------------------------ 8. unique paths
def unique_paths(m: int, n: int) -> int:
    """A robot at the top-left of an m x n grid moves only right or down.
    Return the number of distinct paths to the bottom-right.

    State: dp[c] (current row) = number of paths reaching cell (r, c).
    Decision: the last move came from above (dp[c] before overwrite) or from
    the left (dp[c-1]), so dp[c] += dp[c-1]. First row/column = 1.
    Time O(mn), space O(n).
    """
    dp = [1] * n
    for _ in range(1, m):
        for c in range(1, n):
            dp[c] += dp[c - 1]
    return dp[-1]


# -------------------------------------------------------------------- 9. word break
def word_break(s: str, word_dict: list[str]) -> bool:
    """Return True if s can be segmented into a sequence of dictionary words.

    State: dp[i] = True iff the prefix s[:i] can be segmented.
    Decision: what was the LAST word? Some s[j:i] in the dictionary with
    dp[j] True. dp[0] = True (empty prefix).
    Time O(n^2 * L) with L the max word-length cost of hashing slices,
    space O(n).
    """
    words = set(word_dict)
    max_len = max(map(len, words), default=0)
    dp = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[len(s)]


# ---------------------------------------------------------- 10. subsets (backtrack)
def subsets(nums: list[int]) -> list[list[int]]:
    """Return all 2^n subsets of a list of distinct integers.

    Backtracking: each node's children are the elements with index >= start;
    the start index enforces one canonical order per subset, so {1,2} and
    {2,1} are the same node. Every node (not just leaves) is a subset.
    Choose / explore / unchoose; copy the shared path when recording.
    Time O(n * 2^n), space O(n) beyond the output.
    """
    out: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        out.append(path.copy())
        for i in range(start, len(nums)):
            path.append(nums[i])  # choose
            backtrack(i + 1)      # explore (only later elements: canonical order)
            path.pop()            # unchoose

    backtrack(0)
    return out


# ------------------------------------------------------ 11. permutations (backtrack)
def permutations(nums: list[int]) -> list[list[int]]:
    """Return all n! orderings of a list of distinct integers.

    Backtracking: order matters, so there is no start index -- children are
    ALL unused elements, tracked with a used[] array. Contrast with subsets:
    the start-vs-used distinction is exactly combinations-vs-permutations.
    Time O(n * n!), space O(n) beyond the output.
    """
    out: list[list[int]] = []
    path: list[int] = []
    used = [False] * len(nums)

    def backtrack() -> None:
        if len(path) == len(nums):
            out.append(path.copy())
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])  # choose
            backtrack()           # explore
            path.pop()            # unchoose
            used[i] = False

    backtrack()
    return out


# ----------------------------------------------------------- 12a. N-queens (count)
def n_queens(n: int) -> tuple[int, list[str] | None]:
    """Place n queens on an n x n board so none attack. Return
    (number of solutions, one solution as a list of strings, or None).

    Backtracking, one queen per row (every solution has exactly one, so this
    loses nothing and bounds branching by n). O(1) constraint checks via sets:
    columns, "/" diagonals (r + c constant), "\\" diagonals (r - c constant).
    Time O(n!) with heavy pruning in practice, space O(n).
    """
    cols: set[int] = set()
    diag1: set[int] = set()  # r + c
    diag2: set[int] = set()  # r - c
    placement: list[int] = []  # placement[r] = column of the queen in row r
    count = 0
    first: list[str] | None = None

    def backtrack(r: int) -> None:
        nonlocal count, first
        if r == n:
            count += 1
            if first is None:
                first = ["." * c + "Q" + "." * (n - c - 1) for c in placement]
            return
        for c in range(n):
            if c in cols or (r + c) in diag1 or (r - c) in diag2:
                continue  # prune: constraint already violated
            cols.add(c); diag1.add(r + c); diag2.add(r - c)
            placement.append(c)               # choose
            backtrack(r + 1)                  # explore
            placement.pop()                   # unchoose
            cols.discard(c); diag1.discard(r + c); diag2.discard(r - c)

    backtrack(0)
    return count, first


# ------------------------------------------- 12b. non-overlapping intervals (greedy)
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    """Return the minimum number of intervals to remove so the rest are
    pairwise non-overlapping (touching endpoints do not overlap).

    Greedy: minimizing removals = maximizing the kept set, and interval
    scheduling by EARLIEST FINISH TIME is optimal (exchange argument: the
    earliest-finishing interval can replace the first interval of any optimal
    solution without reducing its size -- see notes.md 5.1). Sort by end,
    sweep, keep whatever starts at/after the last kept end.
    Time O(n log n), space O(1) beyond the sort.
    """
    intervals = sorted(intervals, key=lambda iv: iv[1])
    kept = 0
    last_end = float("-inf")
    for start, end in intervals:
        if start >= last_end:
            kept += 1
            last_end = end
    return len(intervals) - kept


# ------------------------------------------------------- 13. word search (backtrack)
def word_search(board: list[list[str]], word: str) -> bool:
    """Can `word` be traced in the grid through adjacent (up/down/left/right)
    cells, using each cell at most once?

    Backtracking on a grid — the choose/explore/unchoose template of notes.md
    4.1 where "choose" = step onto a cell and mark it visited IN PLACE
    (board[r][c] = '#'), and "unchoose" = restore the letter so sibling paths
    see a clean board. The visited-restore is the whole discipline: forget it
    and either paths illegally reuse cells (never marking) or later starts see
    a corrupted grid (never restoring). No DP here — paths don't overlap as
    subproblems because "which cells are used" is part of the state.
    Time O(m*n * 3^L) (4 choices at the first step, then at most 3 — you never
    go back the way you came), space O(L) recursion.
    """
    rows, cols = len(board), len(board[0])

    def bt(r: int, c: int, i: int) -> bool:
        if board[r][c] != word[i]:
            return False                       # prune: mismatch
        if i == len(word) - 1:
            return True                        # last letter matched
        saved, board[r][c] = board[r][c], "#"  # choose: mark visited
        found = any(
            0 <= r + dr < rows and 0 <= c + dc < cols
            and board[r + dr][c + dc] != "#"
            and bt(r + dr, c + dc, i + 1)      # explore
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        board[r][c] = saved                    # unchoose: restore the cell
        return found

    return any(bt(r, c, 0) for r in range(rows) for c in range(cols))


# ------------------------------------------------------------ 14. jump game (greedy)
def can_jump(nums: list[int]) -> bool:
    """From index i you may jump up to nums[i] forward. Starting at index 0,
    can you reach the last index?

    Greedy furthest-reach sweep. The single scalar `furthest` loses nothing:
    the reachable set is exactly the prefix [0, furthest] — the exchange
    argument proving that (any witness path can be swapped for one the scalar
    summarizes) is worked out in notes.md 5.1; this is its transcription.
    If the scan ever stands past `furthest`, no path exists.
    Time O(n), space O(1).
    """
    furthest = 0
    for i, x in enumerate(nums):
        if i > furthest:
            return False                       # unreachable gap: stuck
        furthest = max(furthest, i + x)
        if furthest >= len(nums) - 1:
            return True                        # early exit, nice to mention
    return True                                # single-element array


# ---------------------------------------------------------- 15. gas station (greedy)
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """gas[i] fuel is gained at station i; cost[i] is spent driving to station
    i+1 (circular). Return the unique starting station from which the full
    circuit can be driven, or -1 if impossible.

    Greedy restart. Sweep with a running tank; whenever the tank goes negative
    at station i, restart the candidate at i+1. Two claims prove it:
    (1) If a candidate start s fails first at i, no station in (s, i] can
    succeed either: any such station t was reached from s with tank >= 0, so
    starting AT t (tank exactly 0) has no more fuel at every point up to i and
    fails there too — skipping straight to i+1 discards nothing.
    (2) If total gas >= total cost, a solution exists — so checking the final
    candidate against the whole sum suffices, no second lap needed. Proof:
    let s be the last restart. Every earlier candidate block [s_k, s_{k+1})
    ended with a NEGATIVE partial sum; total = (negative blocks) + (s's tail
    sum from s to n-1) >= 0 forces the tail to be positive enough that adding
    each negative block back, in order, while wrapping around, never dips the
    running tank below zero — i.e., s completes the circuit.
    Time O(n), space O(1).
    """
    if sum(gas) < sum(cost):
        return -1                              # necessary; (2) shows sufficient
    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:                           # (1): no start in (start, i] works
            start, tank = i + 1, 0
    return start


# ---------------------------------------------------------------------------- tests
if __name__ == "__main__":
    # 1. climbing stairs: 1,1,2,3,5,8...
    assert [climb_stairs(n) for n in range(7)] == [1, 1, 2, 3, 5, 8, 13]
    assert climb_stairs(35) == 14930352

    # 2. house robber
    assert rob([]) == 0
    assert rob([5]) == 5
    assert rob([1, 2, 3, 1]) == 4          # 1 + 3
    assert rob([2, 7, 9, 3, 1]) == 12      # 2 + 9 + 1
    assert rob([2, 1, 1, 2]) == 4          # 2 + 2, skip two in the middle

    # 3. coin change
    assert coin_change([1, 2, 5], 11) == 3     # 5 + 5 + 1
    assert coin_change([2], 3) == -1
    assert coin_change([1], 0) == 0
    assert coin_change([1, 3, 4], 6) == 2      # 3 + 3 -- greedy (4+1+1) says 3

    # 4. LCS
    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("abc", "def") == 0
    assert longest_common_subsequence("", "abc") == 0
    assert longest_common_subsequence("AGGTAB", "GXTXAYB") == 4  # GTAB

    # 5. edit distance
    assert edit_distance("horse", "ros") == 3
    assert edit_distance("intention", "execution") == 5
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0

    # 6. LIS, both versions agree
    cases = [
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),   # 2,3,7,18 (or 101)
        ([0, 1, 0, 3, 2, 3], 4),
        ([7, 7, 7, 7], 1),                    # strictly increasing
        ([], 0),
        ([3, 1, 2], 2),
    ]
    for arr, want in cases:
        assert lis_quadratic(arr) == want, (arr, want)
        assert lis_nlogn(arr) == want, (arr, want)

    # 7. knapsack: 2-D and compressed 1-D must agree
    w, v = [1, 3, 4, 5], [1, 4, 5, 7]
    for cap in range(0, 11):
        assert knapsack_2d(w, v, cap) == knapsack_1d(w, v, cap), cap
    assert knapsack_2d(w, v, 7) == 9        # items of weight 3 and 4
    assert knapsack_1d([2], [10], 6) == 10  # single item used once, not thrice

    # 8. unique paths
    assert unique_paths(3, 7) == 28
    assert unique_paths(3, 2) == 3
    assert unique_paths(1, 1) == 1

    # 9. word break
    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("applepenapple", ["apple", "pen"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    assert word_break("", ["a"]) is True

    # 10. subsets
    ss = subsets([1, 2, 3])
    assert len(ss) == 8 and len({tuple(sorted(s)) for s in ss}) == 8
    assert [] in ss and sorted([1, 2, 3]) in [sorted(s) for s in ss]
    assert subsets([]) == [[]]

    # 11. permutations
    ps = permutations([1, 2, 3])
    assert len(ps) == 6 and len({tuple(p) for p in ps}) == 6
    assert all(sorted(p) == [1, 2, 3] for p in ps)

    # 12a. N-queens: counts 1,0,0,2,10,4,40 for n = 1..7
    for n_, want in [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (8, 92)]:
        cnt, board = n_queens(n_)
        assert cnt == want, (n_, cnt)
        if want:
            assert board is not None and len(board) == n_
            qcols = [row.index("Q") for row in board]
            assert len(set(qcols)) == n_                       # distinct columns
            assert len({r + c for r, c in enumerate(qcols)}) == n_  # "/" diagonals
            assert len({r - c for r, c in enumerate(qcols)}) == n_  # "\" diagonals
        else:
            assert board is None

    # 12b. non-overlapping intervals
    assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0
    assert erase_overlap_intervals([]) == 0

    # 13. word search
    grid = [["A", "B", "C", "E"],
            ["S", "F", "C", "S"],
            ["A", "D", "E", "E"]]
    assert word_search(grid, "ABCCED") is True
    assert word_search(grid, "SEE") is True
    assert word_search(grid, "ABCB") is False      # would need to reuse B
    assert grid[0][0] == "A"                       # board restored afterward
    assert word_search([["a"]], "a") is True
    assert word_search([["a", "a"]], "aaa") is False  # runs out of cells

    # 14. jump game
    assert can_jump([2, 3, 1, 1, 4]) is True
    assert can_jump([3, 2, 1, 0, 4]) is False      # every path lands on the 0
    assert can_jump([0]) is True                   # already at the last index
    assert can_jump([1, 0, 1]) is False            # stuck on the middle zero
    assert can_jump([2, 0, 0]) is True             # jump over the zeros

    # 15. gas station
    assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert can_complete_circuit([2, 3, 4], [3, 4, 3]) == -1  # total gas short
    assert can_complete_circuit([5, 1, 2, 3, 4], [4, 4, 1, 5, 1]) == 4
    assert can_complete_circuit([3], [3]) == 0     # exact fuel, single station
    assert can_complete_circuit([2, 0], [0, 5]) == -1

    print("all 15 problems passed")
