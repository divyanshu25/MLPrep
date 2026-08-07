# Bits, Math, Matrix Simulation - From First Principles

Three pattern families that share a grading rubric: there is no clever data
structure to hide behind, so the interviewer is watching whether you *understand
the representation* - bits as two's complement, numbers as digit/bit sequences,
matrices as index maps. Each trick below is derived, because the derivation is
what survives when the interviewer flips a constraint ("now do it without
strings", "now in place").

---

## 1. Bit manipulation

### Two's complement, from first principles

Fix a width, say 4 bits. There are 16 patterns; arithmetic is mod 16. Define
`-x` as the pattern that makes `x + (-x) ≡ 0 (mod 16)`, i.e. `-x = 16 - x`.
Now notice what flipping every bit does: `~x = 15 - x`, because each flipped
bit contributes its full place value (`1111 = 15`, and `x + ~x = 1111`). So

```
-x = 16 - x = (15 - x) + 1 = ~x + 1
```

Work it for `x = 3`: `0011 → ~ → 1100 (12) → +1 → 1101 (13)`, and
`3 + 13 = 16 ≡ 0`. ✓ So `1101` *is* −3. The sign bit falls out for free: the
top half of the patterns (`1xxx`) represent negatives, and addition hardware
needs no special case - everything is mod-2ⁿ arithmetic. That single identity
`-x = ~x + 1` generates the whole toolbox below.

### The operator toolbox - what each one is FOR

| Tool | What it does | Why (one line) |
|---|---|---|
| `x & (x - 1)` | clears the lowest set bit | proof below |
| `x & -x` | isolates the lowest set bit | proof below |
| `x ^ y` | addition without carry; parity | self-inverse: `x ^ x = 0`, `x ^ 0 = x` |
| `x << k`, `x >> k` | `x * 2**k`, `x // 2**k` | shifting the place values |
| `x | (1 << k)` / `x & ~(1 << k)` / `(x >> k) & 1` | set / clear / read bit k | masks |

**Prove `x & (x-1)` clears the lowest set bit.** Write `x = a·1·0…0` - some
prefix `a`, then the lowest set bit, then `k` zeros. Subtracting 1 borrows
through the trailing zeros: `x - 1 = a·0·1…1` (prefix untouched, that bit
flipped to 0, the zeros to 1s). AND them: the prefix survives (`a & a = a`),
the boundary bit dies (`1 & 0`), the tail dies (`0 & 1`). Result `a·0·0…0`  - 
lowest set bit gone, everything else intact. ∎

**Prove `x & -x` isolates it.** With `x = a·1·0…0`, we get
`-x = ~x + 1 = (ā·0·1…1) + 1 = ā·1·0…0` - the +1 ripples back through the
trailing 1s and stops exactly at the boundary bit. AND with `x`: prefix
`a & ā = 0`, boundary `1 & 1 = 1`, tail zeros. Result `0…0·1·0…0`. ∎ (This is
the primitive under Fenwick trees, and the O(1) "which bucket" step in many
packing problems.)

**Derive XOR-fold → single number.** XOR is associative and commutative, so
folding a list computes one big XOR in *any* order. Group the pairs: each pair
annihilates (`x ^ x = 0`), and `0` is the identity - the fold equals the
unpaired element. Same argument gives **missing number**: XOR all indices
`0..n` with all values; every present value pairs with its index, the missing
one's index survives. No hash set, O(1) space.

### Brian Kernighan counting

Count set bits by repeatedly clearing the lowest one:

```python
count = 0
while x:
    x &= x - 1      # one set bit dies per iteration (proved above)
    count += 1
```

Loop count = number of set bits exactly - O(number of 1s), not O(width). Trace
`x = 13 = 1101`: `1101 → 1100 → 1000 → 0000`, three iterations. ✓

### Bit masks as sets

An n-bit integer is a subset of `{0..n-1}`: membership `(m >> i) & 1`, insert
`m | (1 << i)`, union `|`, intersection `&`, complement `^ full`. That makes
`2^n` subsets enumerable as `range(1 << n)` - the substrate of bitmask DP
(folder 06's TSP-flavored problems). To iterate the subsets of a *specific*
mask:

```python
sub = mask
while sub:
    ...                      # use sub
    sub = (sub - 1) & mask
```

Why it works: `sub - 1` borrows through `sub`'s low zeros; `& mask` throws away
any bits outside `mask` that the borrow turned on. Each step yields the next
smaller integer that is a subset of `mask`, so all `2^popcount(mask)` subsets
appear in strictly decreasing order.

### The DP-on-bits idea - counting bits, derived

Want `bits(i)` for all `i` in `0..n`. Don't count each independently
(O(n log n)); find a recurrence. `i >> 1` is `i` with its last bit dropped, so

```
bits(i) = bits(i >> 1) + (i & 1)
```

and since `i >> 1 < i` for `i ≥ 1`, filling an array in increasing order means
the subproblem is always ready: `dp[i] = dp[i >> 1] + (i & 1)`, O(n) total.
This is the smallest possible example of "DP = recursion on a smaller
representation of the same object" - the same move as digit DP.

### Python-specific traps

Python ints are **infinite-precision**: there is no width, hence no sign bit,
hence no wraparound. Consequences:

- `~x` is *always* `-x - 1` (the identity holds with n = ∞), so `~5 == -6` and
  `~-1 == 0` - but there's no `0xFFFF...` pattern to see; `bin(-6)` prints
  `-0b110`, a minus sign on the magnitude, not two's complement bits.
- `>>` on negatives is arithmetic and floors *forever*: `-1 >> 1 == -1`. A
  C-style `while x:` shift loop never terminates on negative input.
- To simulate 32-bit behavior, mask every intermediate with
  `MASK = 0xFFFFFFFF`, and convert back at the end: a masked value `v ≥ 2^31`
  represents the negative number `v - 2^32` (equivalently `~(v ^ MASK)`).
  This is exactly what "sum of two integers without +" requires (problem 6).

**Why this matters here:** DeepMind loops have reportedly included bit-packing
LC-hards (state squeezed into machine words, masks as visited-sets). And for
ML engineers the topic isn't trivia: quantization *is* bit manipulation  - 
packing int4 weights two-per-byte, extracting them with shifts and masks,
scale/zero-point arithmetic - see `ml/07_llm_stack`. Being fluent here reads
as "has touched real low-level ML systems".

**Recognition cues:** "without using +/*", "appears once/twice/three times",
"count bits", "O(1) space" on problems about duplicates or missing values,
"subsets" with n ≤ 20, anything with 32-bit in the statement.

**Classic bugs:** off-by-one in mask width (`(1 << n) - 1` is n ones, not
`1 << n`); shifting a negative and expecting termination; forgetting operator
precedence - `==` binds tighter than `&`, so write `(x & 1) == 1`, never
`x & 1 == 1`; mutating a mask while iterating its subsets.

---

## 2. Math patterns

### Fast exponentiation - from the binary decomposition of the exponent

`x^n` by n multiplications is O(n). Write `n` in binary:
`n = Σ b_i · 2^i`, so

```
x^n = Π over set bits i of x^(2^i)
```

The factors `x^(2^i)` come free from **repeated squaring** - each is the
square of the previous - so one pass over n's bits suffices:

```python
def fast_pow(x, n):          # n >= 0
    result = 1
    while n:
        if n & 1:            # this bit is set: multiply the current square in
            result *= x
        x *= x               # x^(2^i) -> x^(2^(i+1))
        n >>= 1
    return result
```

Trace `3^13`, `13 = 1101₂`: multiply in `3^1`, skip `3^2`, multiply `3^4`,
multiply `3^8` → `3·81·6561 = 1594323`. ✓ O(log n) multiplications, because
n loses a bit per iteration. Negative n: `x^(-n) = (1/x)^n` - flip once at
the top. This same decomposition is matrix power (Fibonacci in O(log n)) and
modular exponentiation (RSA); *say that* - it shows you see the pattern, not
the problem.

### Floyd cycle-finding on implicit sequences

Happy number: repeatedly replace n by the sum of squared digits; does it reach
1? The step function makes an **implicit linked list**: every node (number)
has exactly one successor. For `n < 10^k` the step value is ≤ `81k`, so the
orbit falls into a bounded set and by pigeonhole must eventually repeat - the
sequence is a ρ shape: a tail into a cycle. That cycle either contains 1
(happy - 1 maps to itself) or doesn't. Detect it exactly as in
`dsa/03_stacks_queues_linked_lists`: tortoise moves one step, hare two; inside
the cycle the gap shrinks by 1 per tick, so they must meet. O(1) space, no
visited-set. The transferable idea: **any deterministic `x → f(x)` iteration
is a linked list**, and every linked-list cycle tool applies.

### Digit manipulation - palindrome number without strings

Peel digits with `% 10` and `// 10`. For palindrome checking, don't reverse
the whole number - reverse *half*, stopping when the reversed part catches up:

```python
rev = 0
while x > rev:
    rev = rev * 10 + x % 10
    x //= 10
return x == rev or x == rev // 10    # even / odd length
```

The odd case drops the middle digit with `rev // 10` (a middle digit needs no
mirror). Two guards up front: negatives are never palindromes (the sign has no
mirror), and a trailing zero (`x % 10 == 0` with `x != 0`) can't be - the
leading digit would have to be 0.

### The overflow talk-track

Reversing half instead of all is not an aesthetic choice: a full reverse of a
valid 32-bit int can **overflow** in C++/Java (2147483647 reversed exceeds the
range). Python ints can't overflow - but say the sentence anyway: *"In Python
this can't overflow; in C++ or Java I'd either check against INT_MAX before
multiplying by 10, or use this half-reverse specifically to stay in range."*
One sentence, said unprompted, is a seniority signal at Google/DeepMind  - 
the same reflex as the `lo + (hi - lo) // 2` line in folder 02.

**Recognition cues:** "implement pow / without library", "digits of", "will
this sequence repeat", explicit 32-bit bounds in the constraints.

**Classic bugs:** forgetting `n = 0 → 1` in pow; infinite loop on negative n
(negate before the loop, not inside); happy-number with a visited-set when the
interviewer asked for O(1) space; `x //= 10` vs `x /= 10` (float creep).

---

## 3. Matrix simulation

These problems have no algorithmic depth - O(mn) is obvious and optimal. What's
graded is **state discipline**: in-place index gymnastics without trampling
data you still need. Interviewers here want clean state management, not tricks;
a readable two-pass solution with named invariants beats a clever one-liner
you can't defend.

### Rotate image - derive it from the index map

Rotating an n×n matrix 90° clockwise sends row i to column n−1−i:

```
(i, j) → (j, n - 1 - i)
```

(Check on 3×3: top-left (0,0) → (0,2) top-right. ✓) Doing that map directly
in place is a nasty 4-cycle of swaps. Instead **factor** it:

- transpose: `(i, j) → (j, i)`
- reverse each row: `(i, j) → (i, n - 1 - j)`

Compose (transpose first, then reverse rows): `(i,j) → (j,i) → (j, n−1−i)`. ✓
Each factor is a set of independent swaps - trivially in place. The classic
bug: transposing with both loops over the full range swaps every pair *twice*
(net identity) - iterate `j in range(i + 1, n)` only.

### Boundary-walk simulation - spiral matrix

Keep four pointers `top, bottom, left, right` bracketing the not-yet-emitted
rectangle. **Invariant after every full pass: everything outside
`[top..bottom] × [left..right]` has been emitted exactly once, everything
inside not at all.** Each pass walks one edge, then shrinks its pointer:
top row left→right (`top += 1`), right column top→bottom (`right -= 1`),
bottom row right→left (`bottom -= 1`), left column bottom→top (`left += 1`).

**The classic single-row/column bug:** after walking the top row and right
column, the rectangle may already be exhausted (a 1×n or m×1 remnant). Walking
the bottom row then re-emits the top row backwards. The fix is two guards mid-
loop - `if top <= bottom` before the bottom row, `if left <= right` before the
left column. Failing case to say out loud: `[[1, 2, 3]]` - without the guard,
output is `[1, 2, 3, 2, 1]`.

### O(1)-marker tricks - set matrix zeroes

"Zero every row and column containing a zero, O(1) extra space." The obvious
two sets of flags are O(m + n); the trick is that **the first row and first
column can BE the flag store**: `matrix[i][0] = 0` means "row i dies",
`matrix[0][j] = 0` means "column j dies". They're about to be zeroed anyway if
they're marked - the information isn't lost, it's relocated.

Two subtleties, both about **order of zeroing**:

1. `matrix[0][0]` is shared between the first row and first column - one extra
   boolean (e.g. `first_col_has_zero`) disambiguates.
2. Zero the **interior first, marker lines last**. The markers are live state;
   zeroing the first row early destroys the column flags before the interior
   has read them. (Derive, don't memorize: any marker scheme forces
   "consumers before the store is cleared".)

That's the general shape of O(1)-marker tricks: find cells whose final value
you already know, and use them as scratch space until the last pass. Game of
life (drill 6) does the same with spare *bits* instead of spare *cells*.

**Recognition cues:** "in place", "rotate / spiral / diagonal traversal",
"O(1) space" on a grid problem, "simulate".

**Classic bugs:** the double-swap transpose; the spiral re-emission on thin
remnants; zeroing markers before the interior; row/column index swap (`m` vs
`n` on non-square inputs - always test a 3×4).

---

## What interviewers listen for

1. The identity *derived*, not asserted ("subtracting 1 borrows through the
   trailing zeros, so the AND clears exactly that bit").
2. The Python-width caveat said unprompted when bits meet negatives.
3. Overflow acknowledged out loud even though Python is immune.
4. On matrices: the invariant of the walk stated before coding, and a thin
   (1×n) test case named without prompting.
5. Complexity stated before and after - O(log n) for pow with the
   bits-of-the-exponent argument, O(mn) with O(1) space for the grid work.

---

## Exit checklist

- [ ] Derive `-x = ~x + 1` on 4 bits and prove `x & (x - 1)` clears the lowest
  set bit - cold, in 3 minutes.
- [ ] Explain why XOR-fold finds the single number, then write single number
  and missing number - cold, in 5 minutes.
- [ ] Write `pow(x, n)` by repeated squaring, negative n included, with the
  binary-decomposition complexity argument said aloud - cold, in 5 minutes.
- [ ] State both Python bit traps (no width: `~` on negatives, arithmetic `>>`
  never reaching 0) and write 32-bit addition with the 0xFFFFFFFF mask  - 
  cold, in 5 minutes.
- [ ] Rotate an n×n matrix in place starting from the index map
  `(i, j) → (j, n - 1 - i)`, naming the double-swap transpose bug - cold, in
  7 minutes.
- [ ] Write spiral order with the single-row/column guards, and set matrix
  zeroes with first-row/col markers, stating why the interior is zeroed before
  the marker lines - cold, in 12 minutes.

All boxes cold before moving on - a miss goes into practice/queue.tsv.
