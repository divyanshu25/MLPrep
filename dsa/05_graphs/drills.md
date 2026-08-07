# Graphs — Timed Drills

Six problems at interview pace. Attempt cold; take hints one at a time only
when stuck; unfold the solution when done or when the clock runs out. If you
needed hint 3, redo the drill from scratch the next day.

---

## Drill 1 — Surrounded Regions
**Time box: 20 min · Medium**

Given a grid of `'X'` and `'O'`, flip every region of `'O'`s that is fully
surrounded by `'X'` (a region touching the border is *not* surrounded).

**Hint ladder**

1. Deciding "is this region surrounded?" while flood-filling it is awkward —
   you discover a border cell mid-fill and must undo. Invert the question:
   which `'O'`s are definitely *safe*?
2. Safe = connected to a border `'O'`. Where should the traversal start?
3. Flood fill from every border `'O'`, marking survivors with a temp value
   `'S'`. Then one sweep: `'O'` → `'X'` (surrounded), `'S'` → `'O'` (safe).

<details>
<summary>Solution</summary>

```python
from collections import deque

def solve(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r in (0, rows - 1) or c in (0, cols - 1)) and board[r][c] == 'O':
                board[r][c] = 'S'
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == 'O':
                board[nr][nc] = 'S'          # mark on enqueue
                q.append((nr, nc))
    for r in range(rows):
        for c in range(cols):
            board[r][c] = 'O' if board[r][c] == 'S' else 'X'
```

Correctness: a region is unsurrounded iff it contains a border `'O'` iff every
cell of it is reachable from some border `'O'` (regions are connected) — and
the multi-source flood fill marks exactly the cells reachable from border
`'O'`s. The complement is exactly the surrounded cells.

**Complexity:** O(rows·cols) time, O(rows·cols) queue worst case. Lesson:
when "surrounded/enclosed" is hard to test directly, traverse from the
*escape boundary* and take the complement.
</details>

---

## Drill 2 — Graph Valid Tree
**Time box: 15 min · Medium**

Given `n` nodes (0..n−1) and a list of undirected edges, return whether they
form a valid tree.

**Hint ladder**

1. A tree = connected + acyclic. There's also a pure counting fact about
   trees relating edges to nodes — what is it?
2. A tree on n nodes has exactly n−1 edges. Given exactly n−1 edges, connected
   ⟺ acyclic (either one implies the other) — so check the count plus just
   one property.
3. Union-find: if any edge's union returns "already connected", there's a
   cycle. With len(edges) == n−1 checked first, no-cycle ⟹ valid tree.

<details>
<summary>Solution</summary>

```python
def valid_tree(n, edges):
    if len(edges) != n - 1:
        return False
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False        # edge inside a component: cycle
        parent[ru] = rv
    return True
```

Why n−1 edges + acyclic ⟹ connected: each union that succeeds reduces the
component count by exactly 1. Starting from n components, n−1 successful
unions leave exactly one component. If all n−1 edges union successfully
(no cycle), the graph is connected. Conversely a cycle wastes an edge, so
with only n−1 edges connectivity would fail.

**Complexity:** O(n α(n)) ≈ O(n) time, O(n) space. (BFS/DFS from node 0
checking `visited == n` also works — same complexity, either is acceptable.)
</details>

---

## Drill 3 — Walls and Gates
**Time box: 20 min · Medium**

Grid of `-1` (wall), `0` (gate), `INF = 2**31 − 1` (empty room). Fill each
empty room with the distance to its nearest gate (leave INF if unreachable).

**Hint ladder**

1. "Distance to nearest gate for *every* cell" — one BFS per gate is
   O(gates · rows · cols). What pattern from rotting oranges collapses this
   to one pass?
2. Multi-source BFS: seed the queue with all gates at distance 0. The §2
   frontier invariant then delivers nearest-gate distances directly.
3. The grid itself can be the visited set: only step into cells still
   holding INF, and write the distance as you enqueue.

<details>
<summary>Solution</summary>

```python
from collections import deque

INF = 2**31 - 1

def walls_and_gates(rooms):
    if not rooms:
        return
    rows, cols = len(rooms), len(rooms[0])
    q = deque((r, c) for r in range(rows) for c in range(cols)
              if rooms[r][c] == 0)
    while q:
        r, c = q.popleft()
        for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == INF:
                rooms[nr][nc] = rooms[r][c] + 1   # write = mark visited
                q.append((nr, nc))
```

Correctness is exactly multi-source BFS: imagine a super-source wired to all
gates; BFS layers from it are "distance to nearest gate". Walls are never
INF so never entered; writing the distance on enqueue doubles as the visited
mark (a cell is written once, with its first — hence smallest — distance).

**Complexity:** O(rows·cols) time and queue space. Signature to remember:
"nearest X for every cell" ⇒ seed all X's at once.
</details>

---

## Drill 4 — Cheapest Flights Within K Stops
**Time box: 30 min · Medium-Hard**

`n` cities, directed flights `(u, v, price)`, source, destination, and `k`
max stops (so ≤ k+1 edges). Return the cheapest such route's price, or −1.

**Hint ladder**

1. Why can't vanilla Dijkstra be used as-is? Because a *pricier* partial
   route with *fewer stops used* may still win later — settling a node by
   price alone discards routes that are better under the hop budget.
2. Bellman–Ford insight: one relaxation round over all edges extends best
   costs by at most one edge. What does running exactly k+1 rounds give you?
3. Run k+1 rounds; in each, relax from a *frozen snapshot* of the previous
   round's costs (else a route could use two new edges in one round and
   break the hop bound).

<details>
<summary>Solution</summary>

```python
def find_cheapest_price(n, flights, src, dst, k):
    INF = float('inf')
    cost = [INF] * n
    cost[src] = 0
    for _ in range(k + 1):                 # k stops = at most k+1 edges
        prev = cost[:]                     # snapshot: cap edges per round at 1
        for u, v, w in flights:
            if prev[u] + w < cost[v]:
                cost[v] = prev[u] + w
        # (skipping relaxations where prev[u] is INF is implicit: INF + w < x is False)
    return cost[dst] if cost[dst] < INF else -1
```

Invariant (induction on rounds): after round i, `cost[v]` is the cheapest
route to v using ≤ i edges. Base: 0 edges reaches only src at cost 0. Step:
the cheapest (≤ i+1)-edge route to v is either a (≤ i)-edge route (already in
`cost`) or a (≤ i)-edge route to some u plus edge (u,v) — exactly what
relaxing from the *snapshot* `prev` considers. Without the snapshot, an
in-round update to `cost[u]` could be reused for (u,v) in the same round,
sneaking in i+2 edges — the classic bug here.

**Complexity:** O(k · E) time, O(n) space. This *is* Bellman–Ford with the
round count capped at k+1 instead of n−1 — which is also why Bellman–Ford
tolerates negative edges: it never greedily commits, it just re-relaxes.
</details>

---

## Drill 5 — Alien Dictionary
**Time box: 35 min · Hard**

A list of words is sorted by an unknown alphabet. Recover a possible letter
order, or return `""` if the list is inconsistent. Example:
`["wrt","wrf","er","ett","rftt"]` → `"wertf"`.

**Hint ladder**

1. What does one adjacent pair of words tell you? Compare them
   character-by-character: the *first mismatch* gives one fact "x before y".
   Later characters tell you nothing (sorted order is decided at the first
   difference).
2. Facts "x before y" are directed edges. "Find an order consistent with all
   edges" is topological sort; "inconsistent" = cycle. Use Kahn's.
3. Edge case that fails silently: `["abc", "ab"]` — a word followed by its
   own proper prefix is invalid (no mismatch found, but the order is
   impossible). Check for it explicitly.

<details>
<summary>Solution</summary>

```python
from collections import defaultdict, deque

def alien_order(words):
    graph = defaultdict(set)
    indeg = {c: 0 for w in words for c in w}

    for w1, w2 in zip(words, words[1:]):
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:      # dedupe: count each edge once
                    graph[c1].add(c2)
                    indeg[c2] += 1
                break                        # only the FIRST mismatch is a fact
        else:                                # no mismatch in the overlap
            if len(w1) > len(w2):
                return ""                    # "abc" before "ab": impossible

    q = deque(c for c in indeg if indeg[c] == 0)
    order = []
    while q:
        c = q.popleft()
        order.append(c)
        for nxt in graph[c]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return "".join(order) if len(order) == len(indeg) else ""
```

Correctness: adjacent-pair constraints are sufficient — sortedness of the
whole list is equivalent to every adjacent pair being ordered (transitivity),
so the edge set captures all information. Kahn's then yields an order
respecting every edge, and its cycle detection (emitted < total letters)
catches contradictions like a→b→a.

Bugs this drill exists to teach: taking facts beyond the first mismatch;
forgetting letters that never appear in any edge (they must still be in
`indeg` and the output); double-counting a repeated edge's indegree; the
prefix case.

**Complexity:** O(total characters) to build (+ O(V + E) for Kahn's on ≤ 26
letters), O(1)-ish space.
</details>

---

## Drill 6 — Swim in Rising Water
**Time box: 35 min · Hard**

`grid[r][c]` is the elevation. At time t you can occupy any cell with
elevation ≤ t and move between adjacent occupiable cells. Return the least t
at which you can travel from (0,0) to (n−1,n−1).

**Hint ladder**

1. Reframe: a path's cost is the *maximum* elevation on it (you must wait
   for the water to reach the path's highest cell). You want the path
   minimizing that maximum — a bottleneck ("minimax") shortest path, not a
   sum.
2. Dijkstra still works if you swap the relaxation: instead of
   `dist[u] + w`, a path through u to v costs `max(dist[u], elev[v])`. The
   cut argument survives because `max` is, like `+ with w ≥ 0`,
   non-decreasing as the path extends.
3. Heap of `(best_max_so_far, r, c)`; pop-min, settle, relax neighbors with
   `max(cur, grid[nr][nc])`. (Alternative: binary search on t with a
   BFS/union-find feasibility check.)

<details>
<summary>Solution</summary>

```python
import heapq

def swim_in_water(grid):
    n = len(grid)
    heap = [(grid[0][0], 0, 0)]
    seen = {(0, 0)}
    while heap:
        t, r, c = heapq.heappop(heap)
        if (r, c) == (n - 1, n - 1):
            return t
        for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in seen:
                seen.add((nr, nc))           # settle on push: safe here, see below
                heapq.heappush(heap, (max(t, grid[nr][nc]), nr, nc))
    return -1  # unreachable (cannot happen on a valid grid)
```

Why Dijkstra's argument transfers: the proof in the notes needs only that
extending a path never decreases its cost. For bottleneck cost,
`max(cost, elev) ≥ cost` always — so the first time a cell is popped, its
cost is optimal, and the first pop of the target is the answer. Marking seen
on push is safe in this variant because every later push of the same cell
carries a ≥ cost (heap pops in nondecreasing order, and `max` can't shrink);
in sum-cost Dijkstra you'd instead use lazy deletion.

**Complexity:** O(n² log n) time, O(n²) space. The binary-search alternative
is O(n² log(max elevation)) — mention it; the minimax-Dijkstra is the
sharper tool. Recognition cue: "minimize the maximum along a path" ⇒
bottleneck Dijkstra or binary search + reachability.
</details>
