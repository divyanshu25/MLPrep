# Stacks, Queues, Linked Lists - Pattern Notes

Three structures, six patterns. As always: the invariant first, then the proof of
why the pattern is correct and why the complexity holds, then the recognition cues.
The LRU cache is the payoff - it combines everything in this folder and is a
reported OpenAI question - and the design-problems section after it reuses its
derivation method three more times.

---

## 1. Stacks - why LIFO fits matching and nesting

A stack does two O(1) things: push and pop-the-most-recent. That's exactly the
access pattern of **nested structure**: in well-nested input (brackets, HTML tags,
function calls, directory paths), *the thing that must close next is always the
thing that opened most recently*. That sentence is the whole correctness argument
for the stack solution to matching problems - LIFO order IS nesting order.

**Valid parentheses, traced.** `s = "([])"`:
- `(` push → stack `(`
- `[` push → stack `( [`
- `]` matches top `[` → pop → stack `(`
- `)` matches top `(` → pop → stack empty. Valid. ✓

Reject on: a closer with an empty stack (`")("` - nothing open), a closer that
mismatches the top (`"(]"`), or a non-empty stack at the end (`"(("` - unclosed).
Those three failure modes are the whole edge-case checklist; name them out loud.

The same "most recent unfinished thing" logic drives expression evaluation (operands
wait on a stack for their operator - see RPN in `problems.py`), undo histories,
DFS iteratively, and call stacks - a recursive solution is a stack solution where the
language manages the stack for you. **Recognition cues:** nesting, matching pairs,
"most recent", undo/backtrack, parsing.

---

## 2. Monotonic stack - the "next greater element" argument

**Problem shape:** for each element, find the nearest element to the right (or left)
that is greater (or smaller). Brute force is O(n²). The monotonic stack does it in
O(n), and deriving *why* is a favorite interview probe.

**The derivation.** Scan left to right, keeping a stack of indices whose "next
greater" is not yet known. When `x = a[i]` arrives:

> Any stacked element **smaller than x** just found its answer - it's x. Pop it,
> record the answer. Repeat until the top is ≥ x (or empty), then push i.

Why may we pop `y < x` *permanently*? Because y's question is now answered - x is
the first greater element to its right (everything between y and x was smaller,
else y would have been popped earlier). And why is the stack always decreasing top
to bottom? Because we only push x after removing everything smaller - the invariant
**"stack values are decreasing from bottom to top"** is maintained by construction.
Each element still on the stack is still waiting; each popped element is done.

**Why O(n) despite the nested while:** each index is pushed exactly once and popped
at most once, so the total work of ALL the while-loops across the whole scan is ≤ n
pops. Amortized O(1) per element. This is the same accounting argument as sliding
window's left pointer - say "each element enters and leaves the stack once".

**Tiny trace - daily temperatures** `[73, 74, 71, 76]` (answer: days until warmer):
- 73: stack empty → push 0. stack [0]
- 74: 74 > 73 → pop 0, ans[0] = 1−0 = 1. push 1. stack [1]
- 71: 71 < 74 → push 2. stack [1, 2] (values 74, 71 - decreasing ✓)
- 76: pop 2 (ans[2] = 1), pop 1 (ans[1] = 2), push 3. Answer [1, 2, 0, 0]. ✓

**Increasing or decreasing?** Derive it, don't memorize: the stack holds the
*unanswered*, so it's monotone in the direction that keeps them waiting.
Next **greater** ⇒ pop smaller ⇒ stack **decreasing**. Next **smaller** ⇒ pop
bigger ⇒ stack **increasing**. Largest-rectangle-in-histogram uses the increasing
stack: a bar pops when a shorter one arrives, and at pop time you know both of its
boundaries - left boundary is the new stack top, right boundary is the arrival  - 
so its maximal rectangle is decided (see `problems.py`).

**Recognition cues:** "next/previous greater/smaller", "days until", "span",
"visible from the left", histogram areas, "remove k digits to make smallest".

**Common bugs:** storing values instead of indices (you almost always need the
distance or boundary); forgetting the sentinel flush at the end (elements still
stacked have no next-greater - either default answers or append a sentinel like
height 0 to force a final drain); `<` vs `<=` in the pop condition (decide how ties
behave and say it).

---

## 3. Queues and deques

A queue pops the *oldest* - FIFO. It fits problems where order of arrival is order
of service: BFS frontiers (process distance-d nodes before d+1 - this ordering is
the entire proof that BFS finds shortest paths in unweighted graphs), task
scheduling, rate limiting by sliding time window ("count requests in the last N
seconds": push arrivals, pop from the front while too old - each event enters and
leaves once, amortized O(1)).

Python note: `list.pop(0)` is O(n) - every element shifts. Use
`collections.deque`, which gives O(1) `append`/`popleft` at both ends.

A **deque** (double-ended queue) pushes/pops both ends in O(1) and unlocks the
**monotonic deque**: sliding-window maximum keeps indices with decreasing values  - 
new elements kill smaller ones from the back (they're older AND smaller: dominated,
dead forever), the window boundary retires them from the front. Same
push-once/pop-once accounting → O(n). It's a monotonic stack (section 2) that can
also expire the oldest - use it when the "next greater" logic must coexist with a
window constraint.

---

## 4. Linked lists - pointer surgery discipline

Linked-list problems aren't algorithmically deep; they are *error-prone*, and
interviewers use them to watch your discipline. The rules that prevent 90% of bugs:

1. **Draw before code.** Boxes and arrows, 3 nodes, do the rewiring on paper. Every
   pointer assignment in code should correspond to one arrow you redrew. If you
   can't draw it, you can't code it.
2. **Save before you sever.** `nxt = cur.next` *before* `cur.next = prev`. Once
   overwritten, the rest of the list is unreachable - this is THE linked-list bug.
3. **Dummy head trick.** Operations that might replace the head (delete first node,
   merge, partition) all have an ugly special case. Kill it with
   `dummy = ListNode(0, head)`, work on `dummy.next`, return `dummy.next`. One
   sentence of justification: it guarantees every real node has a predecessor, so
   the head is no longer special.

**Reversal, the canonical surgery.** Iterative: walk with `prev, cur`, flipping one
arrow per step - invariant: *prev heads the already-reversed prefix, cur heads the
untouched suffix* (trace in `problems.py`). O(n) time, O(1) space. Recursive:
`reverse(head)` trusts the recursion to reverse the tail, then hooks
`head.next.next = head`. Know both; the follow-up "now do it without recursion" (or
the reverse) is standard.

### Fast/slow pointers

Two walkers on the same list, different speeds. Two theorems to be able to prove:

**Middle of the list.** `fast` moves 2, `slow` moves 1. When fast hits the end,
slow has taken half as many steps → the middle. Invariant: `slow` has moved exactly
`⌊steps/2⌋` when `fast` has moved `steps`. O(n), O(1), no length precomputation.

**Cycle detection (Floyd) - prove the meeting.** If there's a cycle of length C,
both pointers eventually enter it. Once both are inside, look at the gap from slow
to fast (measured forward around the cycle): each step, fast gains +1 on slow
(moves 2 vs 1), so the gap changes by exactly 1 modulo C. A quantity that decreases
by 1 each step in a finite cyclic group must hit 0 - **they meet within C steps of
slow entering the cycle**; the pointers cannot "jump over" each other because the
relative step is 1. No cycle ⇒ fast hits None in n/2 steps. Total O(n), O(1) space.

The follow-up (*find where the cycle starts*, LC 142): let the head-to-cycle
distance be F and the meeting point be M steps into the cycle. Slow traveled
F + M; fast traveled 2(F + M); the difference F + M is a multiple of C, so
F ≡ −M (mod C): walking F more steps from the meeting point lands exactly on the
cycle entrance. Hence the two-phase trick - after meeting, restart one pointer at
the head, advance both at speed 1; they meet at the entrance.

**Recognition cues:** "middle/kth from end" (fast/slow or two pointers offset by
k), "cycle", "palindrome list" (middle + reverse second half), "in place, O(1)
space" on a list, "reorder/rotate".

---

## 5. LRU cache - deriving the hashmap + doubly-linked-list design

(Reported OpenAI question; also the archetype of Anthropic's "design a stateful
class" levels.) Requirements: `get(key)` and `put(key, value)` in **O(1)**, evicting
the least-recently-used entry when capacity is exceeded.

**Derive the data structure from the requirements**, don't announce it:

- O(1) `get` by key ⇒ need a **hashmap**. But a hashmap has no notion of order, so
  it can't tell us who is least recent.
- Eviction needs a **recency order** we can update in O(1). What are the order
  operations? (a) move an arbitrary element to the "most recent" end on every
  access; (b) remove the "least recent" end; (c) insert at the "most recent" end.
- An array/Python list: move-to-front is O(n) - out. A heap ordered by timestamp:
  O(log n) - out (and update-in-place is awkward). A **singly** linked list: removal
  of a node requires its *predecessor*, which is an O(n) hunt - out. A **doubly**
  linked list: given a pointer to a node, unlink it in O(1) (`node.prev.next =
  node.next; node.next.prev = node.prev`), and both ends are O(1) with sentinel
  head/tail nodes. In.
- Remaining gap: given a *key*, find its *node* in O(1). That's a lookup - so the
  hashmap's values should be the DLL nodes themselves: `key → node`, node carries
  `(key, value, prev, next)`. The node stores the key too, so eviction can delete
  the map entry from the node.

**The invariant:** the list holds all cached entries ordered by recency (head side
= most recent, tail side = least); the map holds exactly the keys in the list,
pointing at their nodes. Every operation is compose-of-O(1): `get` = map lookup +
unlink + insert-at-head; `put` = (update or insert) + possible evict-at-tail.

Neither structure alone suffices - the map gives O(1) *lookup by key*, the DLL
gives O(1) *reordering by recency*; each covers the other's blind spot. That
sentence is what the interviewer wants to hear. (Python's `OrderedDict` - and
dicts' insertion order plus `move_to_end`-style re-insertion - is this exact combo
built in; mention it, then build it by hand anyway, because that's the point of the
question.) Full implementation with sentinels in `problems.py`.

**Common bugs:** forgetting to move-to-front on `get` (recency updates on *reads*
too); on `put` of an existing key, evicting anyway; unlink order (update neighbors
before losing the node's own pointers); evicting when capacity is merely *reached*
rather than exceeded. Sentinel head/tail nodes remove all the "is it the first/last
node?" branches - same philosophy as the dummy head.

---

## 6. Design problems & iterators - the LRU method, reused

LRU taught the method: *derive the structure from the O(1) requirements by
eliminating candidates*. Three more designs that show up nearly verbatim in
practical rounds (Anthropic's CodeSignal screens and OpenAI's onsite coding both
lean this way) - run the same derivation for each.

### Insert-Delete-GetRandom O(1)

Requirements: `insert(val)`, `remove(val)`, `getRandom()` (uniform over current
elements), all O(1).

- **Hashmap/set alone fails `getRandom`.** Uniform random needs "give me element
  number k" for a random k - an *indexing* operation. A hash table scatters its
  elements over buckets with gaps; picking a uniform element means scanning, O(n).
  (`random.choice(list(s))` hides an O(n) copy.)
- **Array alone fails `remove`.** `list.append` is O(1) and `random.choice(arr)`
  is O(1) - but removing a value means finding it (O(n)) and closing the gap
  (O(n) shift).
- **The combination, plus one trick.** Keep `arr` (the elements) and
  `pos: val → index in arr`. Insert: append + record index. Remove: the only
  O(1)-removable slot in an array is the *last* one - so **swap the victim with
  the last element**, patch the moved element's `pos` entry, then pop. Order gets
  scrambled; nothing in the requirements asked for order, and uniformity over a
  set doesn't care. That observation - *sacrifice what wasn't required* - is the
  design move.

Invariant: `arr` holds exactly the current elements; `pos[v]` is v's index in
`arr`. The classic bug: updating `pos` for the swapped-in element *after* deleting
the victim's entry, which breaks when the victim IS the last element (patch first,
or guard the case). Implementation and tests in `problems.py`.

### Time-Based Key-Value Store

Requirements: `set(key, value, timestamp)`, `get(key, timestamp)` returning the
value with the largest stored ts ≤ query ts ("what was the config at time T?").
This is near-verbatim Anthropic CodeSignal territory, and it composes two folders:

- Timestamps arrive **strictly increasing per key** (say you're relying on this!),
  so `store[key].append((ts, value))` keeps each per-key list sorted *by
  construction* - append-only, no sort ever runs. O(1) per set.
- "Largest ts ≤ query" on a sorted list is a boundary search: first index with
  ts > query, answer one to its left; index 0 means "no value existed yet" →
  `""`. O(log m) per get (folder 02's template; the search itself is drilled in
  `02_binary_search/drills.md`).

The derivation to narrate: a dict of dicts (`key → ts → value`) gives O(1) exact-ts
gets but can't answer "≤" queries; a globally sorted structure pays O(log n) on
writes for no reason. Per-key append-only lists exploit the input's monotonicity  - 
the sortedness is free. If the interviewer drops the increasing-ts guarantee,
`bisect.insort` (O(m) worst) or a balanced structure is the honest fallback - name
the trade-off.

### Resumable iterators - getState / setState

The reported OpenAI shape: build an iterator over nested or multi-source data  - 
`[1, [2, 3], [[4]]]`, or K files/streams round-robin - with `next()`/`has_next()`,
then the twist: **`get_state()` returning a token, and `set_state(token)`
resuming an identical iterator there** (checkpoint/restore, like a crashed job
resuming a scan).

- **What the state must capture:** the *position at every level of the traversal*,
  not just a count. For nested lists: the stack of (container, index) pairs  - 
  i.e., the path from the root to the cursor. For K sources: which source is
  current plus the offset within each. Test your proposed state with: "could two
  different resume points produce the same state token?" If yes, it's not enough
  state.
- **Why generators alone can't checkpoint:** a Python generator's "state" is a
  live stack frame - instruction pointer plus locals. It can be *paused* (that's
  what `yield` is) but not *serialized*: there's no supported way to copy or
  pickle the frame, so a generator gives you `next()` for free and makes
  `get_state` impossible. The fix is to **reify the state**: replace the implicit
  frame with an explicit structure you own (the index stack), advanced by hand.
  "Reify the implicit state" is the same move as converting recursion to an
  explicit stack in tree traversal - say that connection out loud.
- Design detail that separates candidates: make `has_next()` idempotent by
  normalizing the cursor *forward past exhausted levels* after every advance
  (skip empty sublists / drained files eagerly), so `get_state` always snapshots
  a cursor that either points at a real element or is terminal.

A progressive 3-level version (plain → multi-file → get/setState) is drill 3 in
`drills.md`.

---

## What interviewers listen for

1. The LIFO/FIFO *fit* stated as a reason ("what must close next is what opened
   last"), not just the structure named.
2. The amortized argument, unprompted: "nested loop, but each element is pushed and
   popped at most once - O(n) total."
3. On lists: narrating the drawing ("save next, flip the arrow, advance") and
   reaching for the dummy head before the edge case bites.
4. On LRU: deriving map+DLL from the O(1) requirements by eliminating the
   alternatives, and stating the invariant before coding.

## Exit checklist

- [ ] Cold, blank editor: write monotonic-stack next-greater-element and argue O(n) via push/pop-once - 4 min
- [ ] Reverse a linked list iteratively with the three-pointer dance, no reference slips - 3 min
- [ ] Fast/slow pointers: argue why they must meet in a cycle, then find the cycle start - 4 min
- [ ] Derive LRU cache = hashmap + doubly-linked list from the O(1) requirements by eliminating alternatives - 4 min
- [ ] Valid parentheses with a stack, plus the one-line reason LIFO matches nesting - 2 min
- [ ] Say when a deque beats a queue and sketch sliding-window maximum with a monotonic deque - 3 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
