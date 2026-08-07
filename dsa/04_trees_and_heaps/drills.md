# Trees and Heaps — Timed Drills

Nine problems, each with a time box calibrated to a real interview pace. Work
the problem cold; use the hint ladder only when stuck (each hint gives away
strictly more); unfold the solution only when done or out of time. Re-attempt
any drill you needed hint 3 for, a day later, from scratch.

---

## Drill 1 — Binary Tree Right Side View
**Time box: 15 min · Medium**

Standing to the right of a binary tree, return the values of the nodes you can
see, top to bottom. Example: `[1,2,3,None,5,None,4]` → `[1, 3, 4]`.

**Hint ladder**

1. "Top to bottom, one value per level" — which traversal naturally groups
   nodes by level?
2. Within one level of a BFS, which position in the level is visible from the
   right?
3. Do standard level-order BFS with the frozen-`len(q)` loop; append the value
   of the *last* node processed in each level.

<details>
<summary>Solution</summary>

```python
from collections import deque

def right_side_view(root):
    if not root:
        return []
    out, q = [], deque([root])
    while q:
        n = len(q)
        for i in range(n):
            node = q.popleft()
            if i == n - 1:          # last node of this level = rightmost
                out.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
    return out
```

Why the last node in the level is the rightmost: BFS enqueues left before
right for every parent, and parents appear in left-to-right order by
induction, so each level comes off the queue left-to-right.

A DFS alternative: recurse right-first carrying depth; the first node reached
at each depth is the answer for that level (`if depth == len(out): out.append`).

**Complexity:** O(n) time, O(w) space (queue) or O(h) for the DFS version.
</details>

---

## Drill 2 — Count Good Nodes in Binary Tree
**Time box: 15 min · Medium**

A node is *good* if no node on the path from the root to it has a value
greater than it. Count the good nodes. Example: `[3,1,4,3,None,1,5]` → 4.

**Hint ladder**

1. Rephrase: a node is good iff `node.val >= max(values on root path)`. What
   single number, carried downward, decides goodness?
2. This is top-down state (like the BST bounds check), not bottom-up combine.
   Pass the running path-max as a parameter.
3. `dfs(node, path_max)`: count 1 if `node.val >= path_max`, recurse with
   `max(path_max, node.val)`.

<details>
<summary>Solution</summary>

```python
def good_nodes(root):
    def dfs(node, path_max):
        if not node:
            return 0
        good = 1 if node.val >= path_max else 0
        m = max(path_max, node.val)
        return good + dfs(node.left, m) + dfs(node.right, m)
    return dfs(root, float('-inf'))
```

Correctness: `path_max` is by construction the maximum on the root-to-parent
path (induction: starts at −∞, updated with each node passed). Goodness of a
node depends only on that maximum, so the local comparison is exactly the
definition.

**Complexity:** O(n) time, O(h) space. Recognize the shape: information flows
*down* (parameters), counts flow *up* (return values) — many tree problems
combine both directions.
</details>

---

## Drill 3 — Path Sum III
**Time box: 25 min · Medium (hard to make O(n))**

Count downward paths (start anywhere, go only parent→child) whose values sum
to `target`. Example: tree `[10,5,-3,3,2,None,11,3,-2,None,1]`, target 8 → 3.

**Hint ladder**

1. Brute force: from every node, DFS summing downward — O(n²). To beat it,
   think about *prefix sums along the root path*, like subarray-sum-equals-k
   but on a tree.
2. If `prefix[node]` is the sum from root to node, then a path ending at
   `node` with sum `target` exists for each ancestor `a` with
   `prefix[a] = prefix[node] − target`. Count ancestors with a hashmap.
3. DFS carrying a dict `count[prefix_value]`. At each node: add
   `count[cur − target]` to the answer, increment `count[cur]`, recurse, then
   **decrement `count[cur]` on the way out** (backtrack — the map must only
   contain the current root path).

<details>
<summary>Solution</summary>

```python
from collections import defaultdict

def path_sum(root, target):
    count = defaultdict(int)
    count[0] = 1                      # empty prefix: paths starting at root
    ans = 0

    def dfs(node, cur):
        nonlocal ans
        if not node:
            return
        cur += node.val
        ans += count[cur - target]    # ancestors a with prefix[a] = cur - target
        count[cur] += 1
        dfs(node.left, cur)
        dfs(node.right, cur)
        count[cur] -= 1               # backtrack: leave the map as we found it

    dfs(root, 0)
    return ans
```

Why it's correct: a downward path a→node (a an ancestor, exclusive) has sum
`prefix[node] − prefix[a]`. Setting that to `target` and counting matching
ancestors counts exactly the paths ending at `node`; summing over all nodes
counts every path once. `count[0] = 1` handles paths starting at the root.
The decrement is the classic bug: without it you'd count "ancestors" from
sibling subtrees.

**Complexity:** O(n) time, O(n) space (map + stack).
</details>

---

## Drill 4 — Top K Frequent Elements
**Time box: 15 min · Medium**

Given an integer array and k, return the k most frequent elements (any order).
Example: `[1,1,1,2,2,3]`, k=2 → `[1, 2]`.

**Hint ladder**

1. First count frequencies (hashmap). Now it's "k largest by frequency" —
   which heap pattern, and which polarity of heap?
2. A *min*-heap of size k keyed on frequency: the root is the weakest keeper;
   a new candidate either beats it or is discarded. O(n log k).
3. Push `(freq, num)` tuples; use `heappushpop` once the heap has k entries.
   (Or: `heapq.nlargest(k, counts, key=counts.get)`. Or bucket sort by
   frequency for O(n).)

<details>
<summary>Solution</summary>

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    counts = Counter(nums)
    heap = []                          # min-heap of (freq, num), size <= k
    for num, freq in counts.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heappushpop(heap, (freq, num))
    return [num for _, num in heap]
```

Why min-heap: with the k best kept, only the *smallest* of them is ever at
risk from a newcomer — a min-heap's root is exactly that element, checkable
in O(1) and replaceable in O(log k).

**Complexity:** O(n log k) time, O(n) space for the counter. The bucket-sort
alternative (index buckets by frequency, 1..n) achieves O(n) — worth
mentioning, rarely required to code.
</details>

---

## Drill 5 — Task Scheduler
**Time box: 25 min · Medium**

Given tasks `['A','A','A','B','B','B']` and cooldown `n` (same task must be
≥ n slots apart), return the minimum number of slots (idles count).
Example above with n=2 → 8 (`A B _ A B _ A B`).

**Hint ladder**

1. Greedy intuition: always schedule the *most frequent remaining* task that's
   off cooldown — burning down the biggest pile first minimizes forced idles.
   What structure gives "most frequent remaining" fast?
2. Max-heap of counts (negate for Python). Simulate in rounds of `n+1` slots:
   pop up to `n+1` distinct tasks, run each once, push back decremented counts.
3. Alternative O(1) math: let `f` = max frequency, `m` = number of tasks with
   that frequency. The answer is `max(len(tasks), (f-1)*(n+1) + m)`. Derive
   why.

<details>
<summary>Solution</summary>

Heap simulation:

```python
import heapq
from collections import Counter

def least_interval(tasks, n):
    heap = [-c for c in Counter(tasks).values()]
    heapq.heapify(heap)
    time = 0
    while heap:
        batch = []
        for _ in range(n + 1):            # one cooldown window
            if heap:
                batch.append(heapq.heappop(heap))
        for c in batch:
            if c + 1 < 0:                 # count was >1: task still has runs left
                heapq.heappush(heap, c + 1)
        time += (n + 1) if heap else len(batch)  # last window: no trailing idles
    return time
```

Why rounds of n+1 are safe: any two runs of the same task placed in different
windows of length n+1 are automatically ≥ n apart, and filling each window
with the currently-most-frequent distinct tasks never creates a shortage
later (exchange argument: swapping a rarer task earlier for a more frequent
one never increases idles).

The closed form: the frequency-f task forces a skeleton of f runs separated
by n gaps → `(f-1)(n+1) + 1` slots, plus one slot for each *other* task tied
at frequency f (they must extend the final block) → `(f-1)(n+1) + m`. If
there are enough distinct tasks to fill every gap, no idles are needed and
the answer is just `len(tasks)`. Hence the `max`.

**Complexity:** heap version O(T log 26) = O(T); formula O(T) to count,
O(1) after. Space O(1) (26 letters).
</details>

---

## Drill 6 — Smallest Range Covering Elements from K Lists
**Time box: 35 min · Hard**

Given k sorted lists, find the smallest range `[a, b]` that contains at least
one number from each list. Example: `[[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]`
→ `[20, 24]`.

**Hint ladder**

1. Any candidate range is `[min of chosen elements, max of chosen]`, one per
   list. Fix the current minimum: the only way to shrink the range is to
   *advance the list holding the minimum*. Why? Advancing any other list can
   only raise the max or leave it — never lowers the span.
2. This is k-way merge in disguise: heap of (value, list index, element
   index), plus a running max of the current heads.
3. Pop the min; the range (heap-min, running-max) is a candidate; advance
   that list and push its next element, updating the running max. Stop when
   any list is exhausted — the range can no longer cover it.

<details>
<summary>Solution</summary>

```python
import heapq

def smallest_range(lists):
    heap = [(row[0], i, 0) for i, row in enumerate(lists)]
    heapq.heapify(heap)
    cur_max = max(row[0] for row in lists)
    best = (float('-inf'), float('inf'))

    while True:
        lo, i, j = heapq.heappop(heap)
        if cur_max - lo < best[1] - best[0]:
            best = (lo, cur_max)
        if j + 1 == len(lists[i]):        # list i exhausted: no range beyond here
            return list(best)
        nxt = lists[i][j + 1]
        cur_max = max(cur_max, nxt)
        heapq.heappush(heap, (nxt, i, j + 1))
```

Correctness: the heap always holds one "current" element per list, so
`[heap-min, cur_max]` always covers all k lists. Every optimal range has some
element as its minimum; the algorithm enumerates, in increasing order, *every*
possible minimum (each element of each list gets its turn as heap-min before
being advanced past), and for each records the tightest max achievable with
that minimum — which is forced: it's the smallest current element ≥ heap-min
from every other list, exactly what the running heads maintain. So the
optimum is among the candidates checked.

**Complexity:** every element pushed/popped once: O(N log k) for N total
elements, O(k) space. Recognition cue: "one element from each of k sorted
lists" ⇒ heap of heads.
</details>

---

## Drill 7 — Kth Smallest Element in a BST
**Time box: 15 min · Medium**

Return the kth smallest value (1-indexed) in a BST. Example:
`[3,1,4,None,2]`, k=1 → 1. Then the follow-up: how would you support
`next()`/`has_next()` so a caller can pull values one at a time?

**Hint ladder**

1. Which traversal visits a BST's keys in sorted order (notes §4)? The kth
   smallest is just the kth node that traversal touches.
2. Don't traverse everything and index — *stop* at the kth visit. A counter in
   the recursion works; an explicit stack gives finer control.
3. Controlled inorder: push the left spine onto a stack; pop = the next smallest;
   after a pop, push the popped node's right child's left spine. Repeat k times.

<details>
<summary>Solution</summary>

```python
def kth_smallest(root, k):
    stack = []
    node = root
    while True:
        while node:                 # push the left spine
            stack.append(node)
            node = node.left
        node = stack.pop()          # next smallest in the whole tree
        k -= 1
        if k == 0:
            return node.val
        node = node.right           # continue with the right subtree's spine
```

Why the pop order is sorted: the stack holds the path of "ancestors whose left
side is being visited" — exactly the pending nodes of an inorder traversal; the
top is always the smallest unvisited key (its left subtree is exhausted by
construction).

**Complexity:** O(h + k) time — the spine costs O(h), then each of the k pops
does amortized O(1) work (every node is pushed once) — and O(h) space.

The follow-up is the *BST Iterator* (LC 173): the loop body above, split into
`has_next` (stack non-empty) and `next` (pop + push right spine). This is the
same "reify the traversal state into an explicit structure" move as the
resumable iterators in folder 03's design-problems notes — the call stack
becomes a data structure you can pause, resume, and (if asked) checkpoint.
</details>

---

## Drill 8 — Binary Tree Maximum Path Sum
**Time box: 25 min · Hard**

A path is any sequence of nodes connected by edges (no node twice); it need not
pass through the root. Return the maximum sum of values along any path.
Example: `[-10,9,20,None,None,15,7]` → 42 (path 15→20→7).

**Hint ladder**

1. This is diameter (notes §1) with values instead of edge counts: "return one
   thing, track another". What does a parent need from a subtree, and what does
   the global answer need at each node?
2. The parent can extend a path through only ONE child arm (a path can't fork),
   so the recursion returns the best *single downward arm*. The global best at a
   node may join BOTH arms through the node.
3. The negative-clamp decision: an arm with a negative best sum is worse than not
   taking that arm at all — clamp each child's contribution to
   `max(arm, 0)`, which is exactly the "extend or drop" choice made per child.

<details>
<summary>Solution</summary>

```python
def max_path_sum(root):
    best = float('-inf')

    def arm(node):                     # best downward path STARTING at node
        nonlocal best
        if not node:
            return 0
        left = max(arm(node.left), 0)    # negative arm: better to drop it
        right = max(arm(node.right), 0)
        best = max(best, node.val + left + right)  # join both arms HERE
        return node.val + max(left, right)         # parent may extend one arm
    arm(root)
    return best
```

The two quantities are genuinely different — `node.val + left + right` (a
complete path peaking at this node) versus `node.val + max(left, right)` (an
extendable arm) — and conflating them is *the* classic failure. The clamp
`max(arm, 0)` handles all-negative subtrees; the all-negative *tree* still works
because `best` is seeded at −inf and every node's own value (with both arms
clamped to 0) is considered.

**Complexity:** O(n) time, O(h) space. Test: single negative node (`[-3]` → −3),
all-negative tree, and the example above.
</details>

---

## Drill 9 — K Closest Points to Origin
**Time box: 15 min · Medium**

Given points `[[x, y], ...]` and k, return the k points closest to the origin
(any order). Example: `[[1,3],[-2,2]]`, k=1 → `[[-2,2]]`.

**Hint ladder**

1. This is top-k from notes §8 — but top-k *smallest*, so which heap polarity
   keeps the gatekeeper at the root?
2. Mirror image of the min-heap-for-k-largest derivation: keep a MAX-heap of the
   k closest so far; its root is the *farthest keeper* — the only one a newcomer
   could displace. Python: negate the distance.
3. Compare squared distances (`x*x + y*y`) — monotone in true distance, no
   `sqrt`, no floats. Push `(-d, x, y)`; use `heappushpop` once size hits k.

<details>
<summary>Solution</summary>

```python
import heapq

def k_closest(points, k):
    heap = []                              # max-heap by distance via negation
    for x, y in points:
        d = x * x + y * y                  # sqrt is monotone: skip it
        if len(heap) < k:
            heapq.heappush(heap, (-d, x, y))
        elif -d > heap[0][0]:              # closer than the farthest keeper
            heapq.heappushpop(heap, (-d, x, y))
    return [[x, y] for _, x, y in heap]
```

The polarity argument, said aloud: for k *largest* you keep a min-heap (root =
weakest keeper); for k *smallest* you keep a max-heap — either way the root is
the element on the bubble, the only comparison a newcomer needs. Distances can
tie, and the tuple falls back to comparing x then y — plain ints, so no
`TypeError` (the notes §9 tie-breaking quirk, dodged by construction).

**Complexity:** O(n log k) time, O(k) space. Mention the alternatives and when
they win: `heapq.nsmallest(k, points, key=...)` (same bound, one line),
quickselect (average O(n), order not guaranteed), full sort (O(n log n) —
simplest when n is small or k ≈ n).
</details>
