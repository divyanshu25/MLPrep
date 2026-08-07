"""13 stack/queue/linked-list problems in the patterns of notes.md. Each
docstring states the problem; the key insight comment states why the pattern
is correct. The LRU cache and the two design classes after it are the
folder's capstones.

Run: uv run python dsa/03_stacks_queues_linked_lists/problems.py
"""

import random
from bisect import bisect_right
from collections import defaultdict


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def make_list(vals):
    """Build a linked list from a Python list; return the head (or None)."""
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def to_pylist(head, limit=100):
    """Read a linked list back into a Python list (limit guards cycles)."""
    out = []
    while head and len(out) < limit:
        out.append(head.val)
        head = head.next
    return out


# ---------------------------------------------------------------------------
# 1. Valid Parentheses - pattern: stack matching
# ---------------------------------------------------------------------------
def is_valid(s):
    """Given a string s containing only the characters ()[]{} , return True iff
    it is well-nested: every opener is closed by the matching closer type, in
    the correct (last-opened, first-closed) order. The empty string is valid.

    Example:
        is_valid("()[]{}") -> True
        is_valid("(]") -> False   # wrong closer type
        is_valid(")(") -> False   # closer arrives before any opener

    Pattern: stack. Key insight: in well-nested input, the bracket that must
    close next is the one that opened most recently - LIFO order IS nesting
    order. Fail on: closer with empty stack, mismatched top, leftovers at end.
    Complexity: O(n) time, O(n) space.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs:                     # a closer
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:                               # an opener
            stack.append(ch)
    return not stack


# ---------------------------------------------------------------------------
# 2. Min Stack - pattern: stack with parallel invariant
# ---------------------------------------------------------------------------
class MinStack:
    """Design a stack supporting push(val), pop(), top(), and get_min() - the
    minimum element currently on the stack - with every operation O(1). pop()
    removes and returns the top; duplicate minima must survive a pop of one
    copy.

    Example:
        ms.push(-2); ms.push(0); ms.push(-3)
        ms.get_min() -> -3
        ms.pop() -> -3
        ms.get_min() -> -2   # min reverts once -3 leaves

    Pattern: auxiliary stack. Key insight: the minimum can only change at a
    push/pop boundary, so a second stack recording "min of everything at or
    below this height" stays synchronized with the main stack - get_min is
    its top. Complexity: O(1) per operation, O(n) space.
    """

    def __init__(self):
        self.stack = []
        self.mins = []                      # mins[i] = min(stack[:i+1])

    def push(self, val):
        self.stack.append(val)
        self.mins.append(min(val, self.mins[-1]) if self.mins else val)

    def pop(self):
        self.mins.pop()
        return self.stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.mins[-1]


# ---------------------------------------------------------------------------
# 3. Daily Temperatures - pattern: monotonic (decreasing) stack
# ---------------------------------------------------------------------------
def daily_temperatures(temps):
    """Given a list temps of daily temperatures, return a list ans of the same
    length where ans[i] is how many days after day i the first STRICTLY warmer
    temperature occurs, or 0 if no warmer day ever comes.

    Example:
        daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])
            -> [1, 1, 4, 2, 1, 1, 0, 0]   # day 2 (75) waits 4 days for 76
        daily_temperatures([5, 4, 3]) -> [0, 0, 0]   # never warmer

    Pattern: monotonic stack of indices, values decreasing. Key insight: when
    a warmer temp arrives, every stacked colder day just found its answer  - 
    pop it permanently. Each index pushes and pops at most once, so the
    nested while is O(n) total. Complexity: O(n) time, O(n) space.
    """
    ans = [0] * len(temps)
    stack = []                              # indices; temps decreasing
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            ans[j] = i - j
        stack.append(i)
    return ans                              # still stacked -> never warmer -> 0


# ---------------------------------------------------------------------------
# 4. Largest Rectangle in Histogram - pattern: monotonic (increasing) stack
# ---------------------------------------------------------------------------
def largest_rectangle_area(heights):
    """Given a list heights of non-negative histogram bar heights (every bar
    has width 1), return the area of the largest axis-aligned rectangle that
    fits entirely under the bars. Return 0 for an empty histogram.

    Example:
        largest_rectangle_area([2, 1, 5, 6, 2, 3]) -> 10   # height 5 over bars 5, 6
        largest_rectangle_area([3, 3, 3]) -> 9             # width 3 at height 3

    Pattern: monotonic increasing stack. Key insight: a bar's maximal
    rectangle is decided the moment a SHORTER bar arrives - at pop time its
    right boundary is the arrival i and its left boundary is the new stack
    top (everything between was taller, hence popped). A sentinel height 0
    appended at the end drains the stack. Complexity: O(n) time, O(n) space.
    """
    stack = []                              # indices; heights increasing
    best = 0
    for i, h in enumerate(list(heights) + [0]):     # sentinel flush
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            left = stack[-1] if stack else -1       # first shorter bar leftward
            best = max(best, height * (i - left - 1))
        stack.append(i)
    return best


# ---------------------------------------------------------------------------
# 5. Evaluate Reverse Polish Notation - pattern: operand stack
# ---------------------------------------------------------------------------
def eval_rpn(tokens):
    """Given a list of string tokens forming a valid Reverse Polish Notation
    (postfix) expression over integers and the binary operators + - * /,
    evaluate it and return the integer result. Division truncates toward zero,
    and operand order matters for - and /.

    Example:
        eval_rpn(["2", "1", "+", "3", "*"]) -> 9   # (2 + 1) * 3
        eval_rpn(["6", "-3", "/"]) -> -2           # toward zero, not floor (-2, not -3)

    Pattern: stack. Key insight: in postfix, an operator always applies to
    the two MOST RECENT unconsumed values - exactly what a stack's top two
    slots hold; no parentheses or precedence needed. Order matters for - and
    /: the right operand is popped first. Complexity: O(n) time, O(n) space.
    """
    stack = []
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: int(a / b),       # truncate toward zero, not floor
    }
    for tok in tokens:
        if tok in ops:
            b = stack.pop()                 # right operand popped FIRST
            a = stack.pop()
            stack.append(ops[tok](a, b))
        else:
            stack.append(int(tok))
    return stack[0]


# ---------------------------------------------------------------------------
# 6. Reverse Linked List - pattern: pointer surgery (both ways)
# ---------------------------------------------------------------------------
def reverse_list(head):
    """Given the head of a singly linked list (possibly None), reverse it in
    place by rewiring the next pointers - no new nodes, O(1) extra space - and
    return the new head.

    Example:
        reverse_list(1 -> 2 -> 3 -> 4 -> 5) -> 5 -> 4 -> 3 -> 2 -> 1
        reverse_list(None) -> None

    Pattern: one-arrow-per-step surgery. Key insight (invariant): prev heads
    the already-reversed prefix, cur heads the untouched suffix; each step
    saves cur.next BEFORE flipping the arrow (sever-after-save), then both
    walkers advance. Complexity: O(n) time, O(1) space.
    """
    prev, cur = None, head
    while cur:
        nxt = cur.next                      # save before you sever
        cur.next = prev                     # flip one arrow
        prev, cur = cur, nxt                # advance both walkers
    return prev


def reverse_list_recursive(head):
    """Given the head of a singly linked list (possibly None), reverse it
    recursively and return the new head - the standard follow-up to the
    iterative version.

    Example:
        reverse_list_recursive(1 -> 2 -> 3) -> 3 -> 2 -> 1
        reverse_list_recursive(None) -> None

    Key insight: trust reverse(head.next) to return the reversed tail; head's
    old successor is now the tail's LAST node, so head.next.next = head hooks
    head on, and head.next = None makes it the new terminus.
    Complexity: O(n) time, O(n) space (call stack) - say this trade-off aloud.
    """
    if head is None or head.next is None:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


# ---------------------------------------------------------------------------
# 7. Merge Two Sorted Lists - pattern: dummy head
# ---------------------------------------------------------------------------
def merge_two_lists(l1, l2):
    """Given the heads of two sorted linked lists l1 and l2 (either may be
    None), splice them into one sorted list REUSING the existing nodes - no new
    nodes - and return its head. On ties l1's node goes first.

    Example:
        merge_two_lists(1 -> 2 -> 4, 1 -> 3 -> 4) -> 1 -> 1 -> 2 -> 3 -> 4 -> 4
        merge_two_lists(None, None) -> None

    Pattern: dummy head. Key insight: the dummy guarantees every real node
    has a predecessor, so "which list provides the head?" stops being a
    special case; each step attaches the smaller front node and advances that
    list. Leftovers attach in one assignment - they're already sorted.
    Complexity: O(m + n) time, O(1) space.
    """
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2                    # at most one list has leftovers
    return dummy.next


# ---------------------------------------------------------------------------
# 8. Linked List Cycle - pattern: fast/slow pointers (Floyd)
# ---------------------------------------------------------------------------
def has_cycle(head):
    """Given the head of a linked list (possibly None), return True iff
    following next pointers eventually revisits a node, i.e. the list contains
    a cycle. Must use O(1) space - no visited set.

    Example:
        3 -> 2 -> 0 -> -4 with -4.next wired back to the 2-node -> True
        has_cycle(None) -> False   # empty list: no cycle

    Pattern: Floyd's tortoise and hare. Key insight: once both pointers are
    in a cycle of length C, the gap between them shrinks by exactly 1 per
    step (fast gains +1 mod C), so it must hit 0 within C steps - they can't
    jump over each other. No cycle: fast hits None in n/2 steps.
    Complexity: O(n) time, O(1) space.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:                    # identity, not value equality
            return True
    return False


# ---------------------------------------------------------------------------
# 9. Reorder List - pattern: middle + reverse + interleave
# ---------------------------------------------------------------------------
def reorder_list(head):
    """Given the head of a singly linked list L0 -> L1 -> ... -> Ln, rewire it
    IN PLACE into L0 -> Ln -> L1 -> Ln-1 -> L2 -> ... , alternating front and
    back nodes. Only next pointers may change (no copying values into new
    nodes); the function returns nothing.

    Example:
        [1, 2, 3, 4] becomes [1, 4, 2, 3]
        [1, 2, 3, 4, 5] becomes [1, 5, 2, 4, 3]   # odd length: middle ends up last

    Pattern: composition of three list primitives. Key insight: the target
    order interleaves the first half with the REVERSED second half - so:
    (1) fast/slow to find the middle, (2) split and reverse the back half,
    (3) alternate-splice. Each primitive is one already-mastered pattern.
    Complexity: O(n) time, O(1) space.
    """
    if not head or not head.next:
        return
    slow, fast = head, head.next            # slow lands on end of first half
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    second = reverse_list(slow.next)        # reverse back half
    slow.next = None                        # split the halves
    first = head
    while second:                           # interleave; first is >= second
        f_nxt, s_nxt = first.next, second.next
        first.next = second
        second.next = f_nxt
        first, second = f_nxt, s_nxt


# ---------------------------------------------------------------------------
# 10. LRU Cache - pattern: hashmap + doubly linked list
# ---------------------------------------------------------------------------
class LRUCache:
    """Design a fixed-capacity least-recently-used cache. get(key) returns the
    stored value or -1 on a miss; put(key, value) inserts or updates a pair,
    evicting the least recently used key when an INSERT would exceed capacity.
    Both get and put count as a "use" of the key, and both must run in O(1).

    Example:
        cache = LRUCache(2); cache.put(1, 1); cache.put(2, 2)
        cache.get(1) -> 1    # refreshes key 1, so key 2 is now the LRU
        cache.put(3, 3)      # evicts key 2
        cache.get(2) -> -1

    Note: updating an existing key must refresh it, never trigger an eviction.

    Pattern: hashmap + doubly linked list (derived in notes.md). Key insight:
    the map gives O(1) lookup-by-key but no order; the DLL gives O(1)
    reordering (unlink any node given a pointer to it) but no lookup - each
    covers the other's blind spot. Map values ARE the DLL nodes; nodes carry
    their key so eviction can delete its map entry. Sentinel head/tail remove
    every "empty/first/last" branch. Invariant: list order = recency order
    (head side most recent); map keys = list contents, exactly.
    Complexity: O(1) per operation, O(capacity) space.
    """

    class _Node:
        __slots__ = ("key", "val", "prev", "next")

        def __init__(self, key=None, val=None):
            self.key, self.val = key, val
            self.prev = self.next = None

    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}                       # key -> node
        self.head = self._Node()            # sentinel: most-recent side
        self.tail = self._Node()            # sentinel: least-recent side
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._unlink(node)                  # reads refresh recency too
        self._push_front(node)
        return node.val

    def put(self, key, value):
        if key in self.map:                 # update: refresh, don't evict
            node = self.map[key]
            node.val = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self.map) == self.cap:       # evict least recent (tail side)
            lru = self.tail.prev
            self._unlink(lru)
            del self.map[lru.key]           # node carries key for this line
        node = self._Node(key, value)
        self.map[key] = node
        self._push_front(node)


# ---------------------------------------------------------------------------
# 11. Add Two Numbers - pattern: dummy head + carry propagation
# ---------------------------------------------------------------------------
def add_two_numbers(l1, l2):
    """Two non-negative integers are stored as linked lists with their digits
    in REVERSE order (ones digit first), one digit per node. Return their sum
    as a new linked list in the same reversed-digit format. The lists may have
    different lengths, and the sum may need one more digit than either input.
    (The single most-reported Google phone-screen question.)

    Example:
        add_two_numbers(2 -> 4 -> 3, 5 -> 6 -> 4) -> 7 -> 0 -> 8   # 342 + 465 = 807
        add_two_numbers(9 -> 9 -> 9, 1) -> 0 -> 0 -> 0 -> 1        # carry ripples

    Pattern: dummy head + simultaneous walk. Key insight: reverse order means
    both lists start at the ones digit, so grade-school addition runs in one
    forward pass - sum the current digits plus carry, emit sum % 10, carry
    sum // 10. The loop condition `l1 or l2 or carry` covers unequal lengths
    (treat a missing node as digit 0) and the final carry (343 + 657 = 1000
    needs a node the inputs never had). Complexity: O(max(m, n)) time,
    O(max(m, n)) space for the output list.
    """
    dummy = tail = ListNode()
    carry = 0
    while l1 or l2 or carry:
        s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
        carry, digit = divmod(s, 10)
        tail.next = ListNode(digit)
        tail = tail.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next


# ---------------------------------------------------------------------------
# 12. Insert Delete GetRandom O(1) - pattern: array + hashmap, swap-with-last
# ---------------------------------------------------------------------------
class RandomizedSet:
    """Design a set of integers supporting insert(val), remove(val), and
    get_random(), each in O(1) average time. insert and remove return True iff
    they changed the set (inserting a present value or removing an absent one
    returns False); get_random returns a uniformly random element of the
    current, non-empty set.

    Example:
        rs.insert(1); rs.insert(2); rs.insert(3)   # all return True
        rs.insert(2) -> False                      # already present
        rs.remove(2) -> True   # afterwards get_random() yields only 1 or 3

    Pattern: array + hashmap (derived in notes.md 6). Key insight: a hashmap
    alone can't do uniform random (no O(1) "element number k" over scattered
    buckets); an array alone can't do O(1) remove (find + shift). Combine
    them - arr holds the elements, pos maps val -> its index - and make
    removal O(1) by SWAPPING the victim with the last element before popping:
    the last slot is the only O(1)-removable one, and order was never a
    requirement. Patch the moved element's pos entry BEFORE deleting the
    victim's (handles victim == last element).
    Complexity: O(1) per operation, O(n) space.
    """

    def __init__(self):
        self.arr = []
        self.pos = {}                       # val -> index in arr

    def insert(self, val):
        if val in self.pos:
            return False
        self.pos[val] = len(self.arr)
        self.arr.append(val)
        return True

    def remove(self, val):
        if val not in self.pos:
            return False
        i, last = self.pos[val], self.arr[-1]
        self.arr[i] = last                  # move last element into the hole
        self.pos[last] = i                  # patch BEFORE deleting val's entry
        self.arr.pop()
        del self.pos[val]
        return True

    def get_random(self):
        return random.choice(self.arr)      # uniform: arr is dense


# ---------------------------------------------------------------------------
# 13. Time-Based Key-Value Store - pattern: append-only lists + binary search
# ---------------------------------------------------------------------------
class TimeMap:
    """Design a time-based key-value store. set(key, value, timestamp) stores a
    value under key; successive calls for the same key arrive with strictly
    increasing timestamps. get(key, timestamp) returns the value stored with
    the largest ts <= timestamp, or "" if the key is unknown or all its values
    were stored later. (Near-verbatim Anthropic CodeSignal territory - see
    notes.md 6.)

    Example:
        tm.set("foo", "bar", 1); tm.set("foo", "bar2", 4)
        tm.get("foo", 4) -> "bar2"   # exact hit
        tm.get("foo", 3) -> "bar"    # latest value at ts <= 3
        tm.get("foo", 0) -> ""       # nothing stored yet

    Pattern: per-key append-only lists + boundary search. Key insight:
    increasing timestamps mean each key's (ts, value) list is sorted BY
    CONSTRUCTION - set is a plain O(1) append, and get is folder 02's
    boundary search: first index with ts > query, answer one to its left.
    Complexity: O(1) set, O(log m) get, O(n) space.
    """

    def __init__(self):
        self.store = defaultdict(list)      # key -> [(ts, value), ...] sorted

    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        pairs = self.store.get(key, [])
        # first index with ts > timestamp; (timestamp, chr(0x10FFFF)) sorts
        # after every real (timestamp, value) pair, so ties resolve rightward
        i = bisect_right(pairs, (timestamp, chr(0x10FFFF)))
        return pairs[i - 1][1] if i else ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. is_valid
    assert is_valid("()[]{}")
    assert not is_valid("(]")
    assert not is_valid("(")                     # unclosed opener
    assert not is_valid(")(")                    # closer with empty stack
    assert is_valid("")                          # edge: empty is valid

    # 2. MinStack
    ms = MinStack()
    ms.push(-2); ms.push(0); ms.push(-3)
    assert ms.get_min() == -3
    assert ms.pop() == -3
    assert ms.top() == 0
    assert ms.get_min() == -2                    # min restored after pop
    ms.push(-2)
    assert ms.get_min() == -2                    # duplicate mins handled

    # 3. daily_temperatures
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == \
        [1, 1, 4, 2, 1, 1, 0, 0]
    assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert daily_temperatures([5, 4, 3]) == [0, 0, 0]   # never warmer
    assert daily_temperatures([7]) == [0]               # edge: single day

    # 4. largest_rectangle_area
    assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10
    assert largest_rectangle_area([2, 4]) == 4
    assert largest_rectangle_area([3, 3, 3]) == 9        # flat: width wins
    assert largest_rectangle_area([]) == 0               # edge: empty
    assert largest_rectangle_area([5]) == 5

    # 5. eval_rpn
    assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
    assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
    assert eval_rpn(["6", "-3", "/"]) == -2              # truncation toward 0
    assert eval_rpn(["42"]) == 42                        # edge: single operand

    # 6. reverse_list (both ways)
    assert to_pylist(reverse_list(make_list([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
    assert to_pylist(reverse_list(make_list([1]))) == [1]
    assert to_pylist(reverse_list(None)) == []           # edge: empty
    assert to_pylist(reverse_list_recursive(make_list([1, 2, 3]))) == [3, 2, 1]
    assert to_pylist(reverse_list_recursive(None)) == []
    assert to_pylist(reverse_list_recursive(make_list([7]))) == [7]

    # 7. merge_two_lists
    assert to_pylist(merge_two_lists(make_list([1, 2, 4]),
                                     make_list([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
    assert to_pylist(merge_two_lists(None, None)) == []  # edge: both empty
    assert to_pylist(merge_two_lists(None, make_list([0]))) == [0]
    assert to_pylist(merge_two_lists(make_list([1, 5]),
                                     make_list([2]))) == [1, 2, 5]

    # 8. has_cycle
    a = make_list([3, 2, 0, -4])
    assert not has_cycle(a)
    tail = a
    while tail.next:
        tail = tail.next
    tail.next = a.next                           # cycle back to index 1
    assert has_cycle(a)
    single = make_list([1])
    assert not has_cycle(single)
    single.next = single                         # self-loop
    assert has_cycle(single)
    assert not has_cycle(None)                   # edge: empty list

    # 9. reorder_list
    lst = make_list([1, 2, 3, 4])
    reorder_list(lst)
    assert to_pylist(lst) == [1, 4, 2, 3]
    lst = make_list([1, 2, 3, 4, 5])             # odd length
    reorder_list(lst)
    assert to_pylist(lst) == [1, 5, 2, 4, 3]
    lst = make_list([1])                         # edge: single node
    reorder_list(lst)
    assert to_pylist(lst) == [1]

    # 10. LRUCache
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1                     # refreshes key 1
    cache.put(3, 3)                              # evicts key 2 (LRU)
    assert cache.get(2) == -1
    cache.put(4, 4)                              # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    cache = LRUCache(1)                          # edge: capacity 1
    cache.put(9, 9)
    cache.put(9, 99)                             # update must NOT evict
    assert cache.get(9) == 99
    cache.put(8, 8)
    assert cache.get(9) == -1

    # 11. add_two_numbers
    assert to_pylist(add_two_numbers(make_list([2, 4, 3]),
                                     make_list([5, 6, 4]))) == [7, 0, 8]  # 342+465
    assert to_pylist(add_two_numbers(make_list([0]), make_list([0]))) == [0]
    assert to_pylist(add_two_numbers(make_list([9, 9, 9]),
                                     make_list([1]))) == [0, 0, 0, 1]  # carry ripples
    assert to_pylist(add_two_numbers(make_list([3, 4, 3]),
                                     make_list([7, 5, 6]))) == [0, 0, 0, 1]  # final carry
    assert to_pylist(add_two_numbers(make_list([1, 8]),
                                     make_list([0]))) == [1, 8]  # unequal lengths

    # 12. RandomizedSet - delete a middle element, then check getRandom coverage
    rs = RandomizedSet()
    assert rs.insert(1) and rs.insert(2) and rs.insert(3)
    assert not rs.insert(2)                      # duplicate insert refused
    assert rs.remove(2)                          # middle removal: swap-with-last
    assert not rs.remove(2)                      # already gone
    assert sorted(rs.arr) == [1, 3]              # internal state consistent
    assert rs.pos == {rs.arr[0]: 0, rs.arr[1]: 1}
    seen = {rs.get_random() for _ in range(200)}
    assert seen == {1, 3}                        # covers ALL survivors, never 2
    assert rs.remove(3) and rs.remove(1)         # remove last element case
    assert rs.insert(2)                          # reusable after emptying
    assert rs.get_random() == 2

    # 13. TimeMap - gets between, before, and exactly on stored timestamps
    tm = TimeMap()
    tm.set("foo", "bar", 1)
    assert tm.get("foo", 1) == "bar"             # exact hit
    assert tm.get("foo", 3) == "bar"             # between/after: latest <= 3
    tm.set("foo", "bar2", 4)
    assert tm.get("foo", 4) == "bar2"
    assert tm.get("foo", 5) == "bar2"
    assert tm.get("foo", 3) == "bar"             # older query still sees v1
    assert tm.get("foo", 0) == ""                # before any set
    assert tm.get("nope", 9) == ""               # unknown key

    print("all 13 problems passed")
