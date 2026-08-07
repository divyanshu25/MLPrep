# Graphs - From First Principles

A graph is just "things and relationships": nodes V, edges E. Almost every
graph interview problem is one of five algorithms - BFS, DFS, topological
sort, union-find, Dijkstra - applied to a graph that may be handed to you
explicitly, hidden in a grid, or implicit in a state space. This module
derives all five, with the correctness argument for each, then gives you the
recognition guide.

---

## 1. Representation: adjacency lists (and why not matrices, usually)

```python
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)   # drop this line for directed graphs
```

An adjacency **matrix** is an n×n table with `M[u][v] = 1` iff the edge
exists. Compare costs:

|  | adjacency list | adjacency matrix |
|---|---|---|
| space | O(V + E) | O(V²) |
| "is (u,v) an edge?" | O(deg(u)) | O(1) |
| "iterate u's neighbors" | O(deg(u)) | O(V) |

The space argument is the decider: real graphs are almost always **sparse**  - 
E is closer to V than to V². A social graph with 10⁶ users and ~200 friends
each has 2×10⁸ list entries but would need 10¹² matrix cells, mostly zeros.
Worse, BFS/DFS spend all their time iterating neighbors, which the matrix
makes O(V) per node → O(V²) traversal even on a sparse graph, versus the
list's O(V + E). Use a matrix only when the graph is dense, tiny, or you
genuinely need O(1) edge queries (e.g. Floyd–Warshall).

Traversal complexity, once and for all: BFS and DFS on an adjacency list are
**O(V + E)** - each vertex is processed once (the V term) and each edge is
examined once per endpoint (the E term). Every traversal complexity below
falls out of this.

---

## 2. BFS, derived: why a queue finds shortest paths

Goal: from a source s, find the fewest-edges distance to every node in an
*unweighted* graph. Idea: explore in expanding rings - all nodes at distance
0, then all at distance 1, then 2, … A FIFO queue implements exactly this.

```python
from collections import deque

def bfs_distances(graph, s):
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in dist:        # mark when ENQUEUED, not when dequeued
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
```

**The frontier-by-distance invariant, proved.** Claim: at every moment, the
queue contains nodes of at most two consecutive distances d, d+1, with all
the d's ahead of all the d+1's; and when a node is first inserted, its
recorded `dist` is its true shortest distance. Proof: initially the queue is
{s} at distance 0 - true. Inductively, suppose it holds and we pop u with
dist d (the front, hence minimal in queue). Every neighbor v we insert gets
dist d+1 and joins the back, behind any remaining d's and alongside other
d+1's - the two-level structure is preserved. Is d+1 really v's shortest
distance? v is unvisited, so no path of length ≤ d reached it earlier - but
any shorter path to v would end in an edge from a node at distance ≤ d−1,
and all such nodes were dequeued and had all their neighbors inserted before
u was popped (they precede u in FIFO order). So no shorter path exists, and
d+1 is exact. ∎

This is why BFS = shortest path **only for unweighted graphs**: the argument
leans on every edge having length 1, so "fewer edges" = "shorter". Add
weights and the ring structure collapses (that's what Dijkstra fixes, §6).

**The classic bug: mark visited on enqueue, not on dequeue.** If you check
`v not in visited` when *popping* instead of adding to `visited` when
*pushing*, a node can be enqueued many times - once per neighbor that sees
it before any copy is popped. In a grid, each cell has 4 neighbors, so cells
enter the queue up to 4 times, their neighbors up to 4 times per copy… the
queue balloons and on dense graphs the blowup is exponential in the worst
case. Correct answers, catastrophic performance. The rule: a node's fate is
decided the moment it's first *seen* - seal it then.

---

## 3. DFS: components and cycle detection

DFS follows one path as deep as possible, backtracking when stuck  - 
recursion, or an explicit stack.

```python
def dfs(u, graph, visited):
    visited.add(u)
    for v in graph[u]:
        if v not in visited:
            dfs(v, graph, visited)
```

**Connected components:** loop over all nodes; each time you find an
unvisited one, that's a new component - DFS from it marks the whole
component (a node is reachable from the start iff DFS visits it, by
induction on path length). Count the outer-loop triggers. O(V + E) total,
since the visited set ensures each node's DFS body runs once.

**Cycle detection in a *directed* graph - the three colors, derived.** In a
directed graph, "visited" is not enough information: you need to distinguish
*"currently on my recursion path"* from *"fully explored earlier"*. Use
three states:

- **white** - never seen
- **gray** - DFS has entered but not yet exited (it's on the current
  recursion stack)
- **black** - DFS has fully finished it

**Why hitting a gray node means a cycle:** if from u you see an edge to a
gray node v, then v is on the current recursion path - there is a directed
path v ⇝ u (the stack segment from v down to u), and the edge u → v closes
it into a cycle. Conversely, if a cycle exists, consider the first cycle
node the DFS enters, v: DFS cannot finish v until it explores the cycle path
out of v, which leads back to v while v is still gray - a gray hit occurs.
So gray-hit ⟺ cycle. Hitting a **black** node is *not* a cycle - it's just a
path re-converging on finished work (a "cross" or "forward" edge); treating
black like gray is the classic false-positive bug.

```python
WHITE, GRAY, BLACK = 0, 1, 2

def has_cycle(graph, n):
    color = [WHITE] * n
    def dfs(u):
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:                 # back edge: cycle
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False
    return any(color[u] == WHITE and dfs(u) for u in range(n))
```

For an **undirected** graph, cycles are simpler: during DFS, seeing any
visited neighbor other than the node you just came from means a cycle (an
undirected edge would otherwise be its own trivial "cycle", so you must
exclude the parent).

---

## 4. Topological sort, both ways

Given a DAG (directed acyclic graph - think "course prerequisites"), a
topological order lists nodes so every edge points forward. Two independent
derivations; know both.

**Way 1 - Kahn's algorithm (indegree peeling).** A node with indegree 0 has
no prerequisites, so it can safely go first. Remove it; its outgoing edges
disappear, possibly dropping other indegrees to 0. Repeat.

```python
from collections import deque

def topo_kahn(graph, n):
    indeg = [0] * n
    for u in range(n):
        for v in graph[u]:
            indeg[v] += 1
    q = deque(u for u in range(n) if indeg[u] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None   # None: cycle detected
```

Why it works: a DAG always *has* an indegree-0 node (walk edges backward; in
a finite graph with no cycles the walk must terminate, and it terminates at
indegree 0). Emitting it first violates no edge, and what remains is a
smaller DAG - induction finishes the argument. The bonus: if the queue
empties before emitting all n nodes, every remaining node has indegree ≥ 1,
which (walking backward within the remainder) forces a cycle. So Kahn's is
simultaneously a **cycle detector** - this is the cleanest way to solve
Course Schedule.

**Way 2 - DFS postorder, reversed.** Run DFS; when a node *finishes* (turns
black), append it to a list; reverse the list at the end.

Why that works: for any edge u → v, we must show u finishes *after* v, so
that reversing postorder puts u before v. Two cases when DFS at u examines
the edge: (a) v is white - DFS enters v now, and v finishes before u does
(nested calls); (b) v is black - v already finished, u hasn't. The
impossible case, v gray, would mean a cycle (§3) - excluded in a DAG. So in
all cases finish(v) < finish(u), and reversed finishing order respects every
edge. ∎

```python
def topo_dfs(graph, n):
    color = [0] * n
    out = []
    def dfs(u):
        color[u] = 1
        for v in graph[u]:
            if color[v] == 1: raise ValueError("cycle")
            if color[v] == 0: dfs(v)
        color[u] = 2
        out.append(u)              # postorder append
    for u in range(n):
        if color[u] == 0: dfs(u)
    return out[::-1]               # the reversal is the point - don't forget it
```

Both are O(V + E). Kahn's is usually the interview-safer choice: iterative,
gives cycle detection for free, and easily yields "the" order (or
lexicographically-smallest, by swapping the queue for a heap - that's Alien
Dictionary's follow-up).

---

## 5. Union-Find (Disjoint Set Union), from scratch

Problem: maintain a collection of disjoint sets under two operations  - 
`find(x)` (which set is x in?) and `union(x, y)` (merge their sets) - with a
long sequence of interleaved calls. DFS answers connectivity for a *static*
graph; union-find answers it for a graph whose edges arrive *incrementally*,
without re-traversing.

**The forest representation.** Each set is a tree; each element points at a
parent; the root names the set. `find` walks parent pointers to the root;
`union` hooks one root under the other. Naively the trees can degenerate
into chains → O(n) finds. Two fixes:

- **Union by rank:** attach the *shorter* tree under the *taller* one (track
  an upper bound "rank" per root). Height only grows when two equal-rank
  trees merge, and then the merged rank is +1 - so a rank-r tree has ≥ 2^r
  nodes (induction: two rank-(r−1) trees of ≥ 2^(r−1) nodes each). Hence
  height ≤ log₂ n.
- **Path compression:** during `find`, repoint every node on the walked path
  directly at the root. Future finds through those nodes are O(1)-ish.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False                      # already connected - no merge
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
```

(The loop uses *path halving* - point each node at its grandparent - which
gets the same asymptotics as full compression without recursion.)

**Complexity, stated honestly:** with both optimizations, m operations on n
elements take O(m · α(n)), where α is the inverse Ackermann function - a
function that grows so slowly it is ≤ 4 for every n that fits in the
universe. The full proof (Tarjan's amortized potential argument) is beyond
interview scope and it's fine to say so; what you should be able to sketch
is the union-by-rank height bound above, which alone gives O(log n) per op.
Treat union-find as "amortized effectively O(1)".

**When union-find over DFS:** edges arriving online (Redundant Connection:
the first edge whose `union` returns False closes a cycle); repeated
connectivity queries interleaved with edge insertions; Kruskal's MST;
grouping problems where nodes are keys, not indices (Accounts Merge - map
emails to ids first). For one-shot connectivity on a fixed graph, plain
DFS/BFS is simpler and equally O(V+E).

---

## 6. Dijkstra, derived: greedy shortest paths with non-negative weights

Weighted graph, non-negative edge weights, single source. BFS's rings break
because one edge can be "longer" than three others. Dijkstra repairs the
idea: instead of expanding by hop count, always expand the **unsettled node
with the smallest tentative distance**.

**Why greedy is safe - the cut argument.** Let S be the set of nodes whose
shortest distance is already finalized (initially {s}). Let u be the
unsettled node with the smallest tentative distance d(u) (tentative = best
known via settled nodes). Claim: d(u) is u's true shortest distance.
Suppose some shorter path P to u existed. P starts inside S and ends
outside, so it crosses the S-boundary at some first node y ∉ S. The length
of P up to y is a tentative distance for y (its prefix uses only settled
nodes), and since weights are **non-negative**, length(P up to y) ≤
length(P) < d(u). But then y is an unsettled node with a smaller tentative
distance than u - contradicting u's minimality. So no shorter path exists,
and u can be settled. ∎ Induction over |S| finishes the proof.

**Why negative edges break it:** the underlined step - "the prefix to y is
no longer than all of P" - fails if the rest of P has negative total weight.
A negative edge later can make a currently-worse path win, so settling
greedily commits too early. (Bellman–Ford handles negatives by relaxing all
edges V−1 times instead of committing; see the cheapest-flights drill.)

**Lazy-deletion heapq implementation.** Python's heapq has no decrease-key,
so when we find a better distance to v we just push a new (dist, v) entry
and let the stale one rot; on pop, skip entries worse than the recorded
best. This is the same lazy-deletion trick from the heaps notes.

```python
import heapq

def dijkstra(graph, s):            # graph[u] = [(v, w), ...]
    dist = {s: 0}
    heap = [(0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue               # stale entry - lazy deletion
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

Each edge can push at most one heap entry → heap size O(E), giving
**O(E log E)** = O(E log V) time (log E ≤ 2 log V), O(V + E) space. Sanity
check: with all weights 1, Dijkstra degenerates to BFS with an unnecessary
log factor - which is why you should notice "unweighted" and not reach for
Dijkstra.

---

## 7. Grids as graphs

A grid is a graph in disguise: each cell is a node, edges connect
(usually 4-directional) neighbors. No adjacency list needed - neighbors are
computed on the fly:

```python
for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        ...
```

**Number of Islands** = connected components on the grid: scan all cells;
each unvisited land cell starts a flood fill (DFS or BFS) that marks its
whole component; count the starts. O(rows·cols).

**Rotting Oranges - multi-source BFS.** "Rot spreads to adjacent fresh
oranges each minute; how many minutes until all rot?" This asks for the
maximum over fresh oranges of the distance to the *nearest* rotten one.
Instead of one BFS per rotten orange, seed the queue with **all** rotten
oranges at distance 0 and run a single BFS. Why that's correct: it's
ordinary BFS on the graph augmented with a virtual super-source connected
to every rotten cell - distances from the super-source (minus the free
first hop) are exactly distances to the nearest source, and the §2
invariant applies unchanged. One O(rows·cols) pass instead of k of them.
Multi-source BFS is a pattern to keep loaded: "distance to nearest X for
every cell" (Walls and Gates, 01-Matrix) is its signature.

Grid-specific bug: marking cells visited by mutating the grid (`grid[r][c] =
'0'`) is fine and O(1) space, but say you're doing it - some interviewers
want the input unmodified; then carry a visited set.

---

## 8. Implicit graphs: states as nodes

Sometimes no graph is given at all - the graph is a *state space*. Word
Ladder: nodes are words, edges connect words differing by one letter,
question is "shortest transformation sequence" = shortest path = BFS. You
never build the graph; you generate neighbors on demand (for each position,
try all 26 letters, keep those in the dictionary - O(26·L) per word, better
than comparing all word pairs when the list is long).

The same lens covers puzzle problems (states of a lock/board, moves as
edges - Open the Lock, Sliding Puzzle) and "minimum operations to reach X"
problems. The tell: *"minimum number of steps/moves/transformations"* ⇒
shortest path in an implicit graph ⇒ BFS (or Dijkstra if steps have
weights). The only design work is defining the state, the neighbor
function, and hashing states into the visited set.

---

## 9. How to recognize which traversal

| Problem says… | Reach for | Because |
|---|---|---|
| shortest path / fewest moves, unweighted | **BFS** | frontier-by-distance invariant (§2) |
| shortest path, weighted non-negative | **Dijkstra** | cut argument (§6) |
| negative edges, or "at most k hops" | **Bellman–Ford** style DP | Dijkstra's greedy commits too early |
| "distance to nearest X" for many cells | **multi-source BFS** | super-source trick (§7) |
| connected components, flood fill, reachability | **DFS** (or BFS - either) | any traversal covers a component |
| ordering with prerequisites / dependencies | **topological sort** | DAG order (§4) |
| "can you finish" / cycle in directed graph | **Kahn's** (count) or **colors** | gray-hit ⟺ cycle (§3) |
| edges arriving one by one, "first edge that…" | **union-find** | incremental connectivity (§5) |
| merge/group items sharing a key | **union-find** | sets under union (§5) |
| minimize the *maximum* edge on a path | Dijkstra variant or binary search + BFS/union-find | bottleneck shortest path |
| "minimum steps" but no explicit graph | **BFS on states** | implicit graph (§8) |

And the two bugs to name before an interviewer does: (1) visited-on-enqueue,
not on-dequeue (§2) - the silent queue blowup; (2) gray vs black in directed
cycle detection (§3) - black hits are re-convergence, not cycles.

## Exit checklist

- [ ] Cold: state why BFS with a queue yields shortest paths in unweighted graphs, and the visited-add-on-enqueue rule - 2 min
- [ ] Directed cycle detection with white/gray/black: why a gray hit is a cycle and a black hit is not - 3 min
- [ ] Blank editor: write topo sort both ways - Kahn's in-degree queue and DFS postorder reversed - 5 min
- [ ] Write Union-Find with path compression and union by rank/size - 4 min
- [ ] State Dijkstra's greedy invariant, why non-negative weights are required, and the lazy-deletion heap trick - 4 min
- [ ] Grid BFS: directions array, bounds check, mark-visited-on-enqueue, no bugs - 3 min
- [ ] Triage four prompts to BFS/DFS/topo/Union-Find/Dijkstra with a one-line justification each - 2 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
