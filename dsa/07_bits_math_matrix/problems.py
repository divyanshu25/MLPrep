"""12 bits / math / matrix problems, each an instance of a derivation in
notes.md: bits are two's complement (so -x = ~x + 1 generates the toolbox),
numbers are digit/bit sequences (so pow decomposes over the exponent's bits),
and matrices are index maps (so in-place transforms factor into swaps). The
key insight comment above each function names the identity it leans on.

Run: uv run python dsa/07_bits_math_matrix/problems.py
"""


# ---------------------------------------------------------------------------
# 1. Single Number — pattern: XOR fold
# ---------------------------------------------------------------------------
def single_number(nums):
    """Every element appears twice except one; find it in O(n) time,
    O(1) space.

    Pattern: XOR fold. Key insight: XOR is associative + commutative with
    x ^ x = 0 and x ^ 0 = x, so folding the whole array annihilates every
    pair in whatever order and leaves exactly the unpaired element.
    Complexity: O(n) time, O(1) space.
    """
    acc = 0
    for x in nums:
        acc ^= x
    return acc


# ---------------------------------------------------------------------------
# 2. Number of 1 Bits — pattern: Brian Kernighan
# ---------------------------------------------------------------------------
def hamming_weight(n):
    """Count the set bits of a non-negative integer.

    Pattern: Brian Kernighan. Key insight: x & (x - 1) clears exactly the
    lowest set bit (subtracting 1 borrows through the trailing zeros, flips
    the boundary bit to 0, and the AND kills the flipped tail) — so the loop
    runs once per set bit, O(#ones) rather than O(width).
    Complexity: O(#set bits) time, O(1) space.
    """
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


# ---------------------------------------------------------------------------
# 3. Counting Bits — pattern: DP on bits
# ---------------------------------------------------------------------------
def count_bits(n):
    """Return a list where entry i is the number of set bits in i, for
    i in 0..n, in O(n) total.

    Pattern: DP on the bit representation. Key insight: i >> 1 is i with its
    last bit dropped, so bits(i) = bits(i >> 1) + (i & 1); since i >> 1 < i,
    filling the table in increasing order always finds the subproblem ready.
    Complexity: O(n) time, O(n) space (the output).
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


# ---------------------------------------------------------------------------
# 4. Missing Number — pattern: XOR pairing / sum formula (two derivations)
# ---------------------------------------------------------------------------
def missing_number_xor(nums):
    """nums contains n distinct values from 0..n; find the missing one.

    Pattern: XOR fold over indices AND values. Key insight: XOR the indices
    0..n with every value; each present value pairs with its own index and
    annihilates, so only the missing value's index survives.
    Complexity: O(n) time, O(1) space.
    """
    acc = len(nums)                       # the index n has no array slot
    for i, x in enumerate(nums):
        acc ^= i ^ x
    return acc


def missing_number_sum(nums):
    """Same problem via arithmetic. Key insight: sum(0..n) = n(n+1)/2 in
    closed form, so the missing value is that minus the actual sum. Same
    O(1)-space idea, different algebra — in fixed-width languages the XOR
    variant dodges overflow, worth saying aloud.
    Complexity: O(n) time, O(1) space.
    """
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)


# ---------------------------------------------------------------------------
# 5. Reverse Bits — pattern: rebuild in the mirror order
# ---------------------------------------------------------------------------
def reverse_bits(n):
    """Reverse the 32 bits of an unsigned 32-bit integer.

    Pattern: shift out of one end, into the other. Key insight: popping the
    lowest bit of n (n & 1, then n >>= 1) and pushing it onto the result
    (out = out << 1 | bit) reverses order for free — exactly 32 iterations,
    a fixed width the problem grants us (Python ints have none of their own).
    Complexity: O(32) = O(1) time, O(1) space.
    """
    out = 0
    for _ in range(32):
        out = (out << 1) | (n & 1)
        n >>= 1
    return out


# ---------------------------------------------------------------------------
# 6. Sum of Two Integers Without + or - — pattern: XOR + carry, 32-bit masked
# ---------------------------------------------------------------------------
def get_sum(a, b):
    """Add two integers (either may be negative) using only bitwise ops.

    Pattern: XOR is addition without carry; AND-shift is the carry. Key
    insight: a + b = (a ^ b) + ((a & b) << 1), and iterating pushes the
    carry left until it falls off — but ONLY in fixed width. Python ints
    are infinite, so a negative operand has infinitely many sign bits and
    the carry never falls off: mask every intermediate to 32 bits with
    0xFFFFFFFF, then reinterpret the final pattern (>= 2^31 means negative:
    subtract 2^32, written here as ~(v ^ MASK)).
    Complexity: O(32) iterations worst case, O(1) space.
    """
    MASK = 0xFFFFFFFF
    a &= MASK
    b &= MASK
    while b:
        a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK
    return a if a < 0x80000000 else ~(a ^ MASK)


# ---------------------------------------------------------------------------
# 7. Pow(x, n) — pattern: fast exponentiation
# ---------------------------------------------------------------------------
def my_pow(x, n):
    """Compute x**n for integer n (possibly negative) in O(log |n|)
    multiplications.

    Pattern: binary decomposition of the exponent. Key insight:
    n = sum of set bits b_i * 2^i, so x^n is the product of x^(2^i) over set
    bits — and repeated squaring produces each x^(2^i) from the previous one.
    Negative n: x^(-n) = (1/x)^n, flip once before the loop.
    Complexity: O(log |n|) time, O(1) space.
    """
    if n < 0:
        x, n = 1.0 / x, -n
    result = 1.0
    while n:
        if n & 1:
            result *= x
        x *= x
        n >>= 1
    return result


# ---------------------------------------------------------------------------
# 8. Happy Number — pattern: Floyd on an implicit sequence
# ---------------------------------------------------------------------------
def is_happy(n):
    """n -> sum of squared digits, repeated. Return True iff it reaches 1.

    Pattern: cycle detection on an implicit linked list. Key insight: the
    step function gives every number exactly one successor, and orbits are
    bounded (a k-digit number steps to <= 81k), so by pigeonhole the sequence
    enters a cycle — either the fixed point 1 or a cycle avoiding it. Floyd's
    tortoise/hare (same proof as dsa/03 list cycles) detects it in O(1) space,
    no visited-set.
    Complexity: O(cycle + tail) steps, O(1) space.
    """
    def step(x):
        total = 0
        while x:
            d = x % 10
            total += d * d
            x //= 10
        return total

    slow, fast = n, step(n)
    while fast != slow:
        slow = step(slow)
        fast = step(step(fast))
    return fast == 1


# ---------------------------------------------------------------------------
# 9. Palindrome Number — pattern: reverse half, no strings
# ---------------------------------------------------------------------------
def is_palindrome_number(x):
    """Return True iff the decimal digits of x read the same both ways,
    without converting to a string.

    Pattern: digit peeling with % 10 and // 10. Key insight: reverse only the
    trailing half (stop once rev >= x); compare x == rev for even length or
    x == rev // 10 for odd (the middle digit needs no mirror). Reversing only
    half is the overflow-safe form in fixed-width languages — say so.
    Guards: negatives never match (sign has no mirror); a trailing zero with
    x != 0 can't match (leading digit would be 0).
    Complexity: O(#digits) time, O(1) space.
    """
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    rev = 0
    while x > rev:
        rev = rev * 10 + x % 10
        x //= 10
    return x == rev or x == rev // 10


# ---------------------------------------------------------------------------
# 10. Rotate Image — pattern: factor the index map into swaps
# ---------------------------------------------------------------------------
def rotate_image(matrix):
    """Rotate an n x n matrix 90 degrees clockwise, in place.

    Pattern: in-place transform via factored index map. Key insight: the
    rotation is (i, j) -> (j, n-1-i), which factors as transpose
    ((i, j) -> (j, i)) then reverse each row ((i, j) -> (i, n-1-j)) — each
    factor is a set of independent swaps, hence in place. Transpose must
    iterate j > i only, or every pair is swapped twice (net identity).
    Complexity: O(n^2) time, O(1) space.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):          # j > i only: avoid the double swap
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()


# ---------------------------------------------------------------------------
# 11. Spiral Matrix — pattern: boundary-walk simulation
# ---------------------------------------------------------------------------
def spiral_order(matrix):
    """Return all elements of an m x n matrix in spiral (clockwise) order.

    Pattern: four shrinking boundary pointers. Key insight: the invariant —
    everything outside [top..bottom] x [left..right] is already emitted
    exactly once — is restored after each edge walk by shrinking that edge's
    pointer. The guards before the bottom row and left column are mandatory:
    a remaining single row/column would otherwise be re-emitted backwards
    (failing case: [[1, 2, 3]] -> [1, 2, 3, 2, 1]).
    Complexity: O(m*n) time, O(1) space beyond output.
    """
    out = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):           # top row ->
            out.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):           # right column v
            out.append(matrix[i][right])
        right -= 1
        if top <= bottom:                          # guard: a row may remain
            for j in range(right, left - 1, -1):   # bottom row <-
                out.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:                          # guard: a column may remain
            for i in range(bottom, top - 1, -1):   # left column ^
                out.append(matrix[i][left])
            left += 1
    return out


# ---------------------------------------------------------------------------
# 12. Set Matrix Zeroes — pattern: O(1) markers in the first row/column
# ---------------------------------------------------------------------------
def set_zeroes(matrix):
    """If an element is 0, set its entire row and column to 0 — in place,
    O(1) extra space.

    Pattern: reuse cells whose final value is already determined as the flag
    store. Key insight: matrix[i][0] = 0 records "row i dies" and
    matrix[0][j] = 0 records "column j dies" — those cells were going to be
    zeroed anyway when marked, so no information is lost. matrix[0][0] is
    shared by both stores, so the first column gets its own boolean. Order
    matters: zero the interior FIRST (it reads the markers), the marker
    lines LAST (clearing them early destroys live state).
    Complexity: O(m*n) time, O(1) space.
    """
    m, n = len(matrix), len(matrix[0])
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))

    for i in range(m):                     # record into the marker stores
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    for i in range(1, m):                  # interior first: markers are live
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    if matrix[0][0] == 0:                  # marker lines last
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. single_number
    assert single_number([2, 2, 1]) == 1
    assert single_number([4, 1, 2, 1, 2]) == 4
    assert single_number([7]) == 7                    # edge: single element
    assert single_number([-3, 1, 1]) == -3            # negative survives XOR

    # 2. hamming_weight
    assert hamming_weight(0b1011) == 3
    assert hamming_weight(0) == 0                     # edge: no bits
    assert hamming_weight(0xFFFFFFFF) == 32           # all 32 set
    assert hamming_weight(1 << 40) == 1               # beyond 32 bits: fine

    # 3. count_bits
    assert count_bits(5) == [0, 1, 1, 2, 1, 2]
    assert count_bits(0) == [0]                       # edge: n = 0
    assert count_bits(8)[8] == 1                      # power of two
    assert count_bits(7)[7] == 3                      # all-ones value

    # 4. missing_number — assert BOTH variants agree
    for nums, want in [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8), ([0], 1)]:
        assert missing_number_xor(nums) == want
        assert missing_number_sum(nums) == want

    # 5. reverse_bits
    assert reverse_bits(0b00000010100101000001111010011100) == 964176192
    assert reverse_bits(0) == 0                       # edge: zero
    assert reverse_bits(1) == 1 << 31                 # lowest -> highest
    assert reverse_bits(0xFFFFFFFF) == 0xFFFFFFFF     # palindromic pattern

    # 6. get_sum — negatives are the whole point
    assert get_sum(1, 2) == 3
    assert get_sum(-2, 3) == 1                        # mixed signs
    assert get_sum(-1, 1) == 0                        # cancels to zero
    assert get_sum(-12, -8) == -20                    # both negative
    assert get_sum(0, 0) == 0                         # edge: zeros

    # 7. my_pow
    assert my_pow(2.0, 10) == 1024.0
    assert my_pow(2.0, -2) == 0.25                    # negative exponent
    assert my_pow(3.0, 0) == 1.0                      # edge: n = 0
    assert abs(my_pow(2.1, 3) - 9.261) < 1e-9

    # 8. is_happy
    assert is_happy(19)
    assert not is_happy(2)
    assert is_happy(1)                                # edge: already 1
    assert is_happy(7)

    # 9. is_palindrome_number
    assert is_palindrome_number(121)
    assert not is_palindrome_number(-121)             # sign has no mirror
    assert not is_palindrome_number(10)               # trailing-zero trap
    assert is_palindrome_number(0)                    # edge: zero
    assert is_palindrome_number(1221)                 # even length

    # 10. rotate_image
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_image(m)
    assert m == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    m = [[1]]
    rotate_image(m)
    assert m == [[1]]                                 # edge: 1x1
    m = [[1, 2], [3, 4]]
    rotate_image(m)
    assert m == [[3, 1], [4, 2]]

    # 11. spiral_order
    assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == \
        [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    assert spiral_order([[1, 2, 3]]) == [1, 2, 3]     # single row: guard test
    assert spiral_order([[1], [2], [3]]) == [1, 2, 3]  # single column
    assert spiral_order([[5]]) == [5]                 # edge: 1x1

    # 12. set_zeroes
    m = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes(m)
    assert m == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    m = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    set_zeroes(m)
    assert m == [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
    m = [[0]]
    set_zeroes(m)
    assert m == [[0]]                                 # edge: 1x1 zero
    m = [[1, 0]]
    set_zeroes(m)
    assert m == [[0, 0]]                              # zero in first row

    print("all 12 problems passed")
