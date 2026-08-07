# Bits / Math / Matrix — Timed Drills

Protocol (from the [README](../README.md)): start a timer, attempt cold, no hints
until the box expires. Then take hints one at a time — each is a smaller give-away
than the next. Unfold the solution only after a real attempt, and put any miss on
your 2-day re-attempt list. Talk out loud; state complexity before you code.

---

## Drill 1 — Counting Bits, Recurrence From Scratch · Easy · 15 min

For a given n, return a list where entry i is the number of 1 bits in i, for all
i in 0..n — in O(n) TOTAL time. You saw this in problems.py; the drill is deriving
the recurrence yourself from a blank page, then defending why the table order is
valid. Example: n=5 → `[0, 1, 1, 2, 1, 2]`.

<details><summary>Hint 1 (pattern)</summary>
Per-number counting is O(n log n). To reach O(n) you need each answer to come from
an earlier answer — find a smaller number whose bit count almost equals i's.
</details>
<details><summary>Hint 2 (invariant)</summary>
What operation removes exactly one known bit from i? Two candidates work: dropping
the last bit (i >> 1) or clearing the lowest set bit (i & (i-1)). Either gives a
strictly smaller index, so increasing fill order is safe.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
bits(i) = bits(i >> 1) + (i & 1) — i >> 1 is i minus its last bit, and (i & 1) is
that bit. Or: bits(i) = bits(i & (i-1)) + 1, since the AND clears exactly one set
bit (proved in notes.md).
</details>
<details><summary>Solution</summary>

```python
def count_bits(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)   # or: dp[i & (i - 1)] + 1
    return dp
```

Complexity: O(n) time, O(n) space (the output itself). The defense to say out
loud: `i >> 1 < i` for every `i >= 1`, so in increasing fill order the subproblem
is always already computed — that one sentence is the whole correctness argument,
and it's the same "recurse on a smaller representation" move as digit DP.
</details>

---

## Drill 2 — Single Number II (every element thrice) · Medium · 25 min

Every element appears exactly three times except one, which appears once. Find it
in O(n) time and O(1) space. Example: `[2, 2, 3, 2]` → 3. (XOR alone fails here —
three copies don't annihilate.)

<details><summary>Hint 1 (pattern)</summary>
Think per bit position: across the array, what is (count of 1s in bit k) mod 3
for a bit where the unique element has a 1? Where it has a 0?
</details>
<details><summary>Hint 2 (invariant)</summary>
Each bit needs a counter that counts 0 → 1 → 2 → 0 (mod 3). Two bits of state per
position suffice — keep them as two integers `ones` and `twos`, where bit k of
`ones`/`twos` encodes that position's count mod 3. Maintain: after processing a
prefix, bit k of `ones` is set iff its count ≡ 1, of `twos` iff ≡ 2.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
`ones = (ones ^ x) & ~twos` then `twos = (twos ^ x) & ~ones`. Check the cycle for
one bit of x=1: (0,0) → (1,0) → (0,1) → (0,0). The `& ~` term is what forces the
2 → 0 reset instead of 2 → 3. The answer is `ones`.
</details>
<details><summary>Solution</summary>

```python
def single_number_ii(nums):
    ones = twos = 0
    for x in nums:
        ones = (ones ^ x) & ~twos    # advance count; suppress if already at 2
        twos = (twos ^ x) & ~ones    # record "seen twice"; reset after third
    return ones
```

Complexity: O(n) time, O(1) space. Verify the state machine per bit with x's bit
set: (ones,twos) goes (0,0)→(1,0)→(0,1)→(0,0) — a mod-3 counter — so after all
input, bits seen 3k times read 0 and the unique element's bits read 1: `ones` IS
the answer. Works for negatives in Python because ints behave as infinite-width
two's complement (`~` included). Generalization worth saying: "appears k times
except one" needs a mod-k counter, i.e. ceil(log2 k) state integers.
</details>

---

## Drill 3 — Plus One · Easy · 10 min

A non-negative integer is given as a list of digits, most significant first, no
leading zeros. Add one and return the digit list. Example: `[1, 2, 9]` →
`[1, 3, 0]`; `[9, 9]` → `[1, 0, 0]`.

<details><summary>Hint 1 (pattern)</summary>
Grade-school carry propagation from the least significant end.
</details>
<details><summary>Hint 2 (invariant)</summary>
Walking right to left: a digit < 9 absorbs the carry and you're done immediately;
a 9 becomes 0 and the carry continues. So the loop only runs over the trailing 9s.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
If the loop falls off the left end, every digit was a 9 — the answer is 1 followed
by all zeros: `[1] + digits`.
</details>
<details><summary>Solution</summary>

```python
def plus_one(digits):
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits            # carry absorbed: done early
        digits[i] = 0                # 9 + 1 rolls over, carry continues
    return [1] + digits              # all nines: 999 + 1 = 1000
```

Complexity: O(n) worst case (all nines), O(1) extra space except the all-nines
reallocation. The trap graded here: converting to int and back — it works in
Python but dodges the point (and overflows elsewhere); interviewers ask this
BECAUSE they want carry discipline, which is also the warm-up for drill 4.
</details>

---

## Drill 4 — Multiply Strings · Medium · 30 min

Multiply two non-negative integers given as strings, returning the product as a
string — without int(), and without any big-integer shortcuts. Example:
`"123" * "456"` → `"56088"`.

<details><summary>Hint 1 (pattern)</summary>
Grade-school multiplication into a digit array: a product of an m-digit and an
n-digit number has at most m + n digits.
</details>
<details><summary>Hint 2 (invariant)</summary>
Digit i of a (from the right) times digit j of b lands at position i + j in the
result — plus a carry into i + j + 1. Accumulate first, normalize carries as you
go or in a final pass.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Index from the ends: with pos = [0]*(m+n) indexed so that pos[i+j+1] is the "ones"
slot for digits a[i], b[j] (left-indexed strings), do
`s = a[i]*b[j] + pos[i+j+1]`, write `s % 10` there and add `s // 10` into
pos[i+j]. Strip leading zeros at the end (but leave a single "0").
</details>
<details><summary>Solution</summary>

```python
def multiply_strings(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    pos = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            s = int(num1[i]) * int(num2[j]) + pos[i + j + 1]
            pos[i + j + 1] = s % 10
            pos[i + j] += s // 10          # carry; normalized on later visits
    out = "".join(map(str, pos)).lstrip("0")
    return out
```

Complexity: O(m·n) time, O(m + n) space. Why `pos[i + j]` never overflows a later
normalization: every slot is normalized each time it appears as the "ones" slot,
and the leftmost slot only ever receives carries (< 10·min(m,n), fine as a Python
int; in C you'd argue the bound). Classic bugs: forgetting the "0" special case
(lstrip eats everything), and building strings with += in the inner loop (O(n²)
string churn — use the array).
</details>

---

## Drill 5 — Word Search I · Medium · 25 min

Given an m×n grid of letters and a word, return True iff the word can be traced
through adjacent (up/down/left/right) cells without reusing a cell. Example:
grid `[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]`, word `"ABCCED"` →
True. This is grid backtracking — the bridge from matrix simulation to folder
06's backtracking pattern.

<details><summary>Hint 1 (pattern)</summary>
DFS from every cell that matches word[0]; each recursive step tries to extend the
match by one character into the four neighbors.
</details>
<details><summary>Hint 2 (invariant)</summary>
The recursion carries (cell, index k) meaning "word[:k] is already traced, ending
at this cell, using each cell at most once". You need to forbid revisiting cells
on the CURRENT path only — not globally.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Mark in place: overwrite grid[i][j] with a sentinel ('#') before recursing,
restore it after — O(1) space visited-tracking, and the restore is what makes it
backtracking rather than flood fill.
</details>
<details><summary>Solution</summary>

```python
def word_search(board, word):
    m, n = len(board), len(board[0])

    def dfs(i, j, k):                     # word[:k] matched, now need word[k]
        if k == len(word):
            return True
        if not (0 <= i < m and 0 <= j < n) or board[i][j] != word[k]:
            return False
        board[i][j] = "#"                 # claim the cell for this path
        found = (dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or
                 dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1))
        board[i][j] = word[k]             # restore: other paths may use it
        return found

    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

Complexity: O(m·n · 3^L) time worst case (L = len(word); 3, not 4 — you never
walk back into the cell you came from since it's marked), O(L) recursion space.
Classic bug: a global visited set never un-marked — rejects words that need a
cell freed by an abandoned branch. The mark/restore pair IS the backtracking
pattern of folder 06, just wearing grid clothes.
</details>

---

## Drill 6 — Game of Life · Medium · 25 min

An m×n board of 0/1 cells evolves one step by Conway's rules: a live cell
survives with 2–3 live neighbors (all 8 directions), a dead cell becomes live
with exactly 3. Compute the next state IN PLACE — every cell must be updated
"simultaneously", so you may not overwrite a cell's current value while neighbors
still need to read it.

<details><summary>Hint 1 (pattern)</summary>
Each cell only ever holds 0 or 1 — a whole int of scratch space is going unused.
Store two generations in one cell.
</details>
<details><summary>Hint 2 (invariant)</summary>
Encode: bit 0 = current state, bit 1 = next state. During the sweep, every cell's
CURRENT state is always recoverable as `cell & 1`, no matter how many cells have
already had their next state written into bit 1.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Pass 1: count live neighbors with `& 1`, and if the cell will live, `|= 2`.
Pass 2: `cell >>= 1` shifts the future into the present.
</details>
<details><summary>Solution</summary>

```python
def game_of_life(board):
    m, n = len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            live = sum(board[x][y] & 1              # & 1: current gen only
                       for x in range(max(0, i - 1), min(m, i + 2))
                       for y in range(max(0, j - 1), min(n, j + 2))
                       if (x, y) != (i, j))
            if (board[i][j] & 1 and live in (2, 3)) or \
               (not board[i][j] & 1 and live == 3):
                board[i][j] |= 2                    # bit 1 = next gen
    for i in range(m):
        for j in range(n):
            board[i][j] >>= 1                       # future becomes present
    return board
```

Complexity: O(m·n) time (8 neighbors each), O(1) space. This is the bit-flavored
twin of set-matrix-zeroes' marker trick from notes.md: both smuggle the "next"
state into storage whose "current" reading is still recoverable — spare bits here,
determined-anyway cells there. Classic bug: reading neighbors without `& 1`, so
already-updated cells leak their future into the current count (failing case: any
blinker — `[[0,1,0],[0,1,0],[0,1,0]]` must flip to a horizontal bar).
</details>
