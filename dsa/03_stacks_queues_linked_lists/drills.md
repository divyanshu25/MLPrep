# Stacks / Queues / Linked Lists — Timed Drills

Protocol (from the [README](../README.md)): start a timer, attempt cold, no hints
until the box expires. Then take hints one at a time — each is a smaller give-away
than the next. Unfold the solution only after a real attempt, and put any miss on
your 2-day re-attempt list. For linked-list drills: draw the boxes and arrows
BEFORE coding; for design drills: derive the structure from the requirements out
loud, LRU-style.

---

## Drill 1 — Implement Queue Using Stacks · Easy · 15 min

Build a FIFO queue (`push`, `pop`, `peek`, `empty`) using only two stacks. Every
operation must be **amortized** O(1) — and you must be able to defend that word.

<details><summary>Hint 1 (pattern)</summary>
One stack receives pushes; the other serves pops. Two LIFO reversals compose into
FIFO.
</details>
<details><summary>Hint 2 (invariant)</summary>
`out` holds the queue's front segment in reversed (poppable) order; `inb` holds the
back segment in arrival order. The full queue is out[::-1]-then-inb... i.e. pop
from `out`, and only when `out` is EMPTY, pour all of `inb` into it.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
The amortized argument is per-element accounting: each element is pushed onto inb
once, moved to out at most once, and popped once — 3 stack operations over its
lifetime, regardless of the interleaving. An O(n) pour is paid for by the n
elements it moves.
</details>
<details><summary>Solution</summary>

```python
class MyQueue:
    def __init__(self):
        self.inb, self.out = [], []

    def push(self, x):
        self.inb.append(x)

    def _pour(self):                    # only when out is empty!
        if not self.out:
            while self.inb:
                self.out.append(self.inb.pop())

    def pop(self):
        self._pour()
        return self.out.pop()

    def peek(self):
        self._pour()
        return self.out[-1]

    def empty(self):
        return not self.inb and not self.out
```

Complexity: amortized O(1) per operation (each element: ≤ 3 pushes/pops total),
O(n) worst case for a single pop that triggers a pour. The classic bug is pouring
while `out` is non-empty — that interleaves old and new elements and breaks FIFO;
test push 1,2 / pop / push 3 / pop / pop → 1, 2, 3.
</details>

---

## Drill 2 — Basic Calculator II · Medium · 30 min

Evaluate a string like `" 3+5 / 2 "` containing non-negative integers, `+ - * /`
(integer division truncating toward zero), and spaces. No parentheses, no `eval`.
This is the OpenAI-flavored parsing question: graded on clean tokenizing and
precedence handling, not cleverness.

<details><summary>Hint 1 (pattern)</summary>
One pass with a stack of terms whose SUM is the answer. Precedence means * and /
must combine with the previous term immediately; + and - can defer.
</details>
<details><summary>Hint 2 (invariant)</summary>
Process each number together with the operator BEFORE it: `+` push the number,
`-` push its negation, `*` or `/` pop the previous term and push the combined
product/quotient. The stack always holds fully-resolved additive terms.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Track `num` (current number being built digit by digit) and `op` (last operator
seen, initialized '+'). Flush num when you hit a non-digit or the end of string —
the end-of-string flush is the classic miss. Python gotcha: -3 // 2 == -2
(floors), so truncate with int(a / b).
</details>
<details><summary>Solution</summary>

```python
def calculate(s):
    stack, num, op = [], 0, "+"
    for i, ch in enumerate(s):
        if ch.isdigit():
            num = num * 10 + int(ch)
        if ch in "+-*/" or i == len(s) - 1:   # operator or end: flush num
            if op == "+":
                stack.append(num)
            elif op == "-":
                stack.append(-num)
            elif op == "*":
                stack.append(stack.pop() * num)
            else:
                prev = stack.pop()
                stack.append(int(prev / num))  # truncate toward zero, not floor
            num, op = 0, ch
    return sum(stack)
```

Complexity: O(n) time, O(n) space. Test: `"3+2*2"` → 7, `" 3/2 "` → 1,
`"14-3/2"` → 13, and a negative intermediate like `"1-2*3"` → -5 (the -6 term).
Follow-up they often add: parentheses (Basic Calculator I) — recurse or push
(result, sign) frames on the stack at '('.
</details>

---

## Drill 3 — Resumable Multi-File Iterator (progressive) · Hard · 35 min

The Anthropic/OpenAI progressive-design shape (notes.md §6). Work the levels in
order; each level's code should survive into the next.

**Level 1:** `FileIterator(files)` where `files` is a list of lists of records —
`next()` / `has_next()` yielding all records of file 0, then file 1, ... Skip
empty files.
**Level 2:** add `remove_file(i)` — subsequent iteration never yields records from
file i (already-yielded records stay yielded).
**Level 3:** add `get_state()` returning a serializable token, and
`set_state(token)` so a NEW iterator resumes exactly where the old one stood.

<details><summary>Hint 1 (pattern)</summary>
Reify the cursor: (file index, offset within file), advanced by hand. A generator
gives you next() for free but makes level 3 impossible — you can't serialize a
paused stack frame.
</details>
<details><summary>Hint 2 (invariant)</summary>
Keep the cursor NORMALIZED: after every advance, push it forward past empty,
exhausted, or removed files so it either points at a real unread record or sits at
the end. Then has_next() is a pure O(1) read and get_state() always snapshots a
meaningful position.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
State token = (file_idx, offset, removed_files). All three are needed: two
different resume points must never share a token, and removals are part of "where
the iteration stands". set_state restores the fields then re-normalizes.
</details>
<details><summary>Solution</summary>

```python
class FileIterator:
    def __init__(self, files):
        self.files = files
        self.removed = set()
        self.f = 0          # current file index
        self.i = 0          # offset within current file
        self._normalize()

    def _normalize(self):   # advance past removed/exhausted files
        while self.f < len(self.files) and (
                self.f in self.removed or self.i >= len(self.files[self.f])):
            self.f += 1
            self.i = 0

    def has_next(self):
        return self.f < len(self.files)

    def next(self):
        rec = self.files[self.f][self.i]
        self.i += 1
        self._normalize()
        return rec

    def remove_file(self, idx):          # level 2
        self.removed.add(idx)
        self._normalize()                # current file may be the one removed

    def get_state(self):                 # level 3: plain data, serializable
        return (self.f, self.i, tuple(sorted(self.removed)))

    def set_state(self, state):
        self.f, self.i, removed = state
        self.removed = set(removed)
        self._normalize()
```

Complexity: O(1) amortized per next (each file is skipped past once), O(#removed)
state size. What the interviewer is probing: (1) the eager `_normalize` discipline
— lazy versions make `has_next` mutate state or lie; (2) that the state token is
*complete* (forgetting `removed` means a restored iterator resurrects deleted
files); (3) saying explicitly why a generator-based level 1 dead-ends at level 3.
</details>

---

## Drill 4 — Remove Nth Node From End of List · Medium · 15 min

Delete the nth node from the END of a singly linked list in ONE pass, and return
the head. Example: `1→2→3→4→5`, n=2 → `1→2→3→5`.

<details><summary>Hint 1 (pattern)</summary>
Two pointers with a fixed gap — the fast/slow family, offset flavor.
</details>
<details><summary>Hint 2 (invariant)</summary>
Send `lead` n nodes ahead, then advance `lag` and `lead` together; when lead falls
off the end, lag is exactly n from the end. To DELETE, lag must stop at the
PREDECESSOR of the victim.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
Start lag at a dummy head and walk lead n steps from the real head: when lead hits
None, lag.next is the victim (`lag.next = lag.next.next`). The dummy also handles
"delete the head" (n == length) with zero special-casing.
</details>
<details><summary>Solution</summary>

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    lead = head
    for _ in range(n):                   # open a gap of n
        lead = lead.next
    lag = dummy                          # trails the gap, lands on predecessor
    while lead:
        lead, lag = lead.next, lag.next
    lag.next = lag.next.next             # splice the victim out
    return dummy.next
```

Complexity: O(L) time, O(1) space, single pass. Invariant to state: lead is always
exactly n nodes ahead of lag.next. Edge cases to name unprompted: n equals the
list length (head removed — the dummy earns its keep), single-node list.
</details>

---

## Drill 5 — Palindrome Linked List · Medium · 20 min

Is a singly linked list a palindrome? O(n) time, **O(1) space** (so no copying
into an array — mention that O(n)-space version, then beat it).

<details><summary>Hint 1 (pattern)</summary>
Composition of the folder's primitives: find the middle, reverse in place, walk
two fronts — the reorder-list toolkit pointed at a yes/no question.
</details>
<details><summary>Hint 2 (invariant)</summary>
Fast/slow to the middle; reverse the SECOND half; compare first half vs reversed
second half node by node. Odd length: the middle node belongs to neither half and
compares against nothing — make sure your split leaves it out of the comparison.
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
With `fast = head.next` at the start, slow stops at the END of the first half, so
`second = reverse(slow.next)` covers exactly the back half (odd middle stays with
the front and is skipped when second runs out). Compare while `second` is not None.
</details>
<details><summary>Solution</summary>

```python
def is_palindrome(head):
    if not head or not head.next:
        return True
    slow, fast = head, head.next
    while fast and fast.next:            # slow -> end of first half
        slow, fast = slow.next, fast.next.next
    second = reverse_list(slow.next)     # reverse back half (problems.py #6)
    first, ok = head, True
    while second:                        # second is the shorter (or equal) side
        if first.val != second.val:
            ok = False
            break
        first, second = first.next, second.next
    return ok
```

Complexity: O(n) time, O(1) space. Interview polish: offer to restore the list
(re-reverse the second half and re-attach) — mutating the input is worth flagging
even when accepted. Test odd (`1,2,1`), even (`1,2,2,1`), failing (`1,2`), and
single-node lists.
</details>

---

## Drill 6 — Copy List with Random Pointer · Medium · 25 min

Each node has `next` and a `random` pointer to an arbitrary node (or None). Return
a deep copy. Then the follow-up: do it without the hashmap, O(1) extra space.

<details><summary>Hint 1 (pattern)</summary>
The obstacle: when you copy node X, X.random's copy may not exist yet. You need a
map old → new that's complete before any random pointer is wired.
</details>
<details><summary>Hint 2 (invariant)</summary>
Two passes: pass 1 creates all clones and records old → clone; pass 2 wires
clone.next = map[old.next], clone.random = map[old.random] (None maps to None —
`dict.get` handles it).
</details>
<details><summary>Hint 3 (near-spoiler)</summary>
O(1)-space version: make the LIST ITSELF the map. Interleave clones
(A→A'→B→B'→...); then old.random's clone is old.random.next — an O(1) lookup by
position; finally unzip the two lists. Three passes, no dict.
</details>
<details><summary>Solution</summary>

```python
def copy_random_list(head):
    clone = {None: None}                 # old -> new; None maps to None
    cur = head
    while cur:                           # pass 1: create every clone
        clone[cur] = Node(cur.val)
        cur = cur.next
    cur = head
    while cur:                           # pass 2: wire pointers via the map
        clone[cur].next = clone[cur.next]
        clone[cur].random = clone[cur.random]
        cur = cur.next
    return clone[head]
```

Complexity: O(n) time, O(n) space. The interleaving version trades the dict for
pointer discipline: (1) insert A' after each A, (2) set
`A.next.random = A.random.next` when A.random exists, (3) unzip — restoring the
original list exactly (interviewers check). Same O(n) time, O(1) extra space.
The seeded `{None: None}` entry is the trick that removes every null check —
worth stating.
</details>
