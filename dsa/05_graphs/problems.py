"""Graphs - 10 classic problems, solved from first principles.

Each solution names the algorithm it instantiates from notes.md: BFS
(frontier invariant), multi-source BFS, DFS flood fill, Kahn's topological
sort, union-find, Dijkstra with lazy deletion. Run this file; it asserts
every test and prints a success line.
"""

import heapq
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# 1. Number of Islands
# ---------------------------------------------------------------------------
def num_islands(grid):
    """Given a 2D grid (list of lists of the CHARACTERS '1' for land and '0'
    for water), return the number of islands. An island is a maximal group of
    '1' cells connected 4-directionally (up/down/left/right - no diagonals),
    surrounded by water or the grid edge. An empty grid has 0 islands.

    Example:
        Input:  [list(r) for r in ["11000",
                                   "11000",
                                   "00100",
                                   "00011"]]
        Output: 3

    Note: you may modify the grid in place.

    Key insight: the grid is a graph in disguise; each unvisited land cell
    starts a flood fill that consumes its whole component. Count the starts.
    Marks visited by mutating the grid to '0' (say so in interviews; use a
    visited set if the input must survive).
    Complexity: O(rows * cols) time, O(rows * cols) worst-case stack/queue.
    """
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def flood(r, c):  # iterative DFS: no recursion-limit worries on big grids
        stack = [(r, c)]
        grid[r][c] = "0"
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                    grid[nr][nc] = "0"  # mark when pushed, not when popped
                    stack.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                flood(r, c)
    return count


# ---------------------------------------------------------------------------
# 2. Rotting Oranges (multi-source BFS)
# ---------------------------------------------------------------------------
def oranges_rotting(grid):
    """Given a 2D grid of INTEGERS where 0 is an empty cell, 1 a fresh orange,
    and 2 a rotten orange, every minute each rotten orange rots the fresh
    oranges 4-directionally adjacent to it. Return the number of minutes until
    no fresh orange remains. If some fresh orange can never rot (walled off by
    empty cells), return -1; if there are no fresh oranges at all, return 0.

    Example:
        Input:  [[2, 1, 1], [1, 1, 0], [0, 1, 1]]   Output: 4
        Input:  [[2, 1, 1], [0, 1, 1], [1, 0, 1]]   Output: -1
                (the orange at the bottom-left is unreachable)

    Key insight: "minutes until all rot" = max over fresh cells of distance
    to the NEAREST rotten cell. Seed the BFS queue with ALL rotten cells at
    time 0 (a virtual super-source) - one pass computes every
    nearest-source distance at once.
    Complexity: O(rows * cols) time and space.
    """
    rows, cols = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                q.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    minutes = 0
    while q:
        r, c, t = q.popleft()
        minutes = max(minutes, t)
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2  # visited-on-enqueue: seal fate when seen
                fresh -= 1
                q.append((nr, nc, t + 1))
    return minutes if fresh == 0 else -1


# ---------------------------------------------------------------------------
# 3. Clone Graph
# ---------------------------------------------------------------------------
class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    """Given a reference to one node of a connected undirected graph, return a
    deep copy of the whole graph. Nodes are GraphNode objects with an integer
    .val and a .neighbors list of adjacent GraphNode references (edges appear
    in both endpoints' lists; cycles are possible). The copy must contain
    brand-new nodes - no returned node may be an original - with the same
    values and the same adjacency structure. Return None for a None input.

    Example:
        Input:  node 1 of the 4-cycle 1-2-3-4-1
        Output: a new node with val 1 whose neighbor vals are {2, 4}, part of
                an identical 4-cycle built entirely from new nodes

    Key insight: the visited set must double as an old->new map, and a
    node's clone must be created (and registered) BEFORE recursing into
    neighbors - otherwise cycles recurse forever. Same trick as
    copy-list-with-random-pointer.
    Complexity: O(V + E) time, O(V) space.
    """
    if node is None:
        return None
    clones = {}

    def dfs(u):
        if u in clones:
            return clones[u]
        copy = GraphNode(u.val)
        clones[u] = copy  # register before recursing: breaks cycles
        copy.neighbors = [dfs(v) for v in u.neighbors]
        return copy

    return dfs(node)


# ---------------------------------------------------------------------------
# 4. Course Schedule (cycle detection via Kahn's)
# ---------------------------------------------------------------------------
def can_finish(num_courses, prerequisites):
    """There are num_courses courses labeled 0 .. num_courses-1. prerequisites
    is an edge list of pairs [course, prereq] meaning you must take prereq
    before course (pairs may share courses; the list may be empty). Return
    True iff it is possible to order all the courses so every prerequisite is
    respected, False otherwise.

    Example:
        Input:  num_courses = 2, prerequisites = [[1, 0]]          Output: True
        Input:  num_courses = 2, prerequisites = [[1, 0], [0, 1]]  Output: False
                (each course requires the other first)

    Key insight: possible iff the prerequisite digraph is acyclic. Kahn's
    peeling detects cycles for free: if fewer than n nodes ever reach
    indegree 0, the leftovers all have a prerequisite among themselves  - 
    a cycle.
    Complexity: O(V + E) time, O(V + E) space.
    """
    graph = defaultdict(list)
    indeg = [0] * num_courses
    for course, pre in prerequisites:
        graph[pre].append(course)
        indeg[course] += 1

    q = deque(u for u in range(num_courses) if indeg[u] == 0)
    taken = 0
    while q:
        u = q.popleft()
        taken += 1
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return taken == num_courses


# ---------------------------------------------------------------------------
# 5. Course Schedule II (topological order)
# ---------------------------------------------------------------------------
def find_order(num_courses, prerequisites):
    """Same setup as Course Schedule: num_courses courses labeled
    0 .. num_courses-1 and prerequisites as [course, prereq] pairs (take
    prereq before course). Return a list of all num_courses labels in ANY
    order that respects every prerequisite; if no such order exists (the
    prerequisites are contradictory), return [].

    Example:
        Input:  num_courses = 4, prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
        Output: [0, 1, 2, 3] or [0, 2, 1, 3] - either is accepted
        Input:  num_courses = 2, prerequisites = [[1, 0], [0, 1]]   Output: []

    Key insight: identical to problem 4, but keep the pop order - Kahn's
    emission order IS a topological order (an indegree-0 node violates no
    edge; removing it leaves a smaller DAG; induct).
    Complexity: O(V + E) time and space.
    """
    graph = defaultdict(list)
    indeg = [0] * num_courses
    for course, pre in prerequisites:
        graph[pre].append(course)
        indeg[course] += 1

    q = deque(u for u in range(num_courses) if indeg[u] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == num_courses else []


# ---------------------------------------------------------------------------
# 6. Pacific Atlantic Water Flow
# ---------------------------------------------------------------------------
def pacific_atlantic(heights):
    """Given an m x n grid of integer heights, rain water on a cell can flow
    to any 4-directionally adjacent cell whose height is LESS THAN OR EQUAL to
    the current cell's, and from there onward. The Pacific ocean touches the
    top and left edges; the Atlantic touches the bottom and right edges.
    Return every cell from which water can reach BOTH oceans, as a sorted list
    of (row, col) tuples. Return [] for an empty grid.

    Example:
        Input:  [[1, 2, 2, 3, 5],
                 [3, 2, 3, 4, 4],
                 [2, 4, 5, 3, 1],
                 [6, 7, 1, 4, 5],
                 [5, 1, 1, 2, 4]]
        Output: [(0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)]

    Key insight: "which cells reach the ocean" run forward is one traversal
    per cell - O((mn)^2). Reverse the flow: BFS FROM each ocean's coastline
    climbing uphill (neighbor height >= current). Two multi-source
    traversals; answer = intersection.
    Complexity: O(rows * cols) time and space.
    """
    if not heights:
        return []
    rows, cols = len(heights), len(heights[0])

    def reachable(starts):
        seen = set(starts)
        q = deque(starts)
        while q:
            r, c = q.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in seen
                        and heights[nr][nc] >= heights[r][c]):  # climb uphill
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return seen

    pacific = [(0, c) for c in range(cols)] + [(r, 0) for r in range(rows)]
    atlantic = ([(rows - 1, c) for c in range(cols)]
                + [(r, cols - 1) for r in range(rows)])
    return sorted(reachable(pacific) & reachable(atlantic))


# ---------------------------------------------------------------------------
# 7. Word Ladder
# ---------------------------------------------------------------------------
def ladder_length(begin_word, end_word, word_list):
    """Given begin_word, end_word, and word_list (all lowercase strings of the
    same length), a transformation sequence changes exactly one letter per
    step, and every word after begin_word - including end_word - must appear
    in word_list (begin_word itself need not). Return the NUMBER OF WORDS in
    the shortest such sequence, counting both endpoints; return 0 if end_word
    is not in word_list or no sequence exists.

    Example:
        Input:  begin = "hit", end = "cog",
                word_list = ["hot", "dot", "dog", "lot", "log", "cog"]
        Output: 5   ("hit" -> "hot" -> "dot" -> "dog" -> "cog")
        Without "cog" in the list                        Output: 0

    Key insight: an implicit graph - words are nodes, one-letter edits are
    edges, "shortest sequence" = BFS. Generate neighbors by trying 26
    letters per position (O(26L) per word) instead of comparing all pairs.
    Complexity: O(N * L * 26) time for N words of length L, O(N) space.
    """
    words = set(word_list)
    if end_word not in words:
        return 0
    q = deque([(begin_word, 1)])
    seen = {begin_word}
    while q:
        word, steps = q.popleft()
        if word == end_word:
            return steps
        for i in range(len(word)):
            for ch in "abcdefghijklmnopqrstuvwxyz":
                nxt = word[:i] + ch + word[i + 1:]
                if nxt in words and nxt not in seen:
                    seen.add(nxt)  # on enqueue - the §2 bug otherwise
                    q.append((nxt, steps + 1))
    return 0


# ---------------------------------------------------------------------------
# Union-Find (shared by problems 8 and 10)
# ---------------------------------------------------------------------------
class UnionFind:
    """Disjoint sets: path halving + union by rank => amortized ~O(1) ops."""

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
            return False  # already connected: this edge closes a cycle
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# ---------------------------------------------------------------------------
# 8. Redundant Connection (union-find)
# ---------------------------------------------------------------------------
def find_redundant_connection(edges):
    """Given an undirected graph that began as a tree on nodes 1..n and had
    exactly one extra edge added, as an edge list `edges` of n pairs [u, v],
    return an edge whose removal leaves a tree on all n nodes. If several
    edges qualify, return the one that appears LAST in the input, in its
    original [u, v] form.

    Example:
        Input:  [[1, 2], [1, 3], [2, 3]]                    Output: [2, 3]
        Input:  [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]    Output: [1, 4]

    Key insight: edges arrive incrementally - union-find territory. Union
    each edge; the first union that returns False joins two already-
    connected nodes, i.e. closes the unique cycle. That edge is the answer.
    Complexity: O(E * alpha(V)) time, O(V) space.
    """
    uf = UnionFind(len(edges) + 1)  # nodes are 1..n
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
    return []


# ---------------------------------------------------------------------------
# 9. Network Delay Time (Dijkstra)
# ---------------------------------------------------------------------------
def network_delay_time(times, n, k):
    """A network has n nodes labeled 1..n. `times` is a list of directed edges
    (u, v, w): a signal sent from u reaches v after w time units. A signal
    starts at node k at time 0 and propagates along every outgoing edge from
    each node it reaches. Return the time at which the LAST node receives the
    signal, or -1 if any node never receives it.

    Example:
        Input:  times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]], n = 4, k = 2
        Output: 2   (node 4 is reached last, via 2 -> 3 -> 4)
        Input:  times = [[1, 2, 1]], n = 2, k = 2           Output: -1

    Note: edge weights are non-negative; multiple edges between the same pair
    of nodes may appear.

    Key insight: single-source shortest paths, non-negative weights =>
    Dijkstra. heapq has no decrease-key, so use lazy deletion: push
    duplicates, skip stale pops. Answer = max settled distance.
    Complexity: O(E log E) time, O(V + E) space.
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = {k: 0}
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue  # stale entry
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return max(dist.values()) if len(dist) == n else -1


# ---------------------------------------------------------------------------
# 10. Accounts Merge (union-find)
# ---------------------------------------------------------------------------
def accounts_merge(accounts):
    """Given a list `accounts` where accounts[i] = [name, email1, email2, ...]
    (a person's name followed by one or more email strings), merge every pair
    of accounts that share at least one email - merging is transitive across
    accounts. Two accounts with the same name but no shared email stay
    separate. Return the merged accounts, each as [name, sorted emails...];
    this implementation also sorts the accounts themselves for determinism.

    Example:
        Input:  [["John", "js@mail.com", "ny@mail.com"],
                 ["John", "js@mail.com", "j00@mail.com"],
                 ["Mary", "mary@mail.com"]]
        Output: [["John", "j00@mail.com", "js@mail.com", "ny@mail.com"],
                 ["Mary", "mary@mail.com"]]

    Key insight: emails are nodes; each account unions its emails into one
    set. Union-find shines because nodes are string keys arriving
    incrementally - assign each new email an integer id, union within each
    account, then bucket emails by root.
    Complexity: O(T * alpha(T) + T log T) for T total emails (sort
    dominates), O(T) space.
    """
    email_id = {}
    email_name = {}
    for name, *emails in accounts:
        for e in emails:
            if e not in email_id:
                email_id[e] = len(email_id)
                email_name[e] = name

    uf = UnionFind(len(email_id))
    for _, *emails in accounts:
        first = email_id[emails[0]]
        for e in emails[1:]:
            uf.union(first, email_id[e])

    groups = defaultdict(list)
    for e, i in email_id.items():
        groups[uf.find(i)].append(e)

    return sorted(
        [email_name[emails[0]]] + sorted(emails) for emails in groups.values()
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. number of islands
    grid1 = [list(row) for row in ["11110", "11010", "11000", "00000"]]
    assert num_islands(grid1) == 1
    grid2 = [list(row) for row in ["11000", "11000", "00100", "00011"]]
    assert num_islands(grid2) == 3
    assert num_islands([]) == 0
    # component reachable only via an up/left step: catches down/right-only floods
    grid3 = [list(row) for row in ["01", "11"]]
    assert num_islands(grid3) == 1

    # 2. rotting oranges
    assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1  # unreachable
    assert oranges_rotting([[0, 2]]) == 0  # nothing fresh: 0 minutes

    # 3. clone graph - 4-cycle: 1-2-3-4-1
    n1, n2, n3, n4 = (GraphNode(i) for i in (1, 2, 3, 4))
    n1.neighbors = [n2, n4]
    n2.neighbors = [n1, n3]
    n3.neighbors = [n2, n4]
    n4.neighbors = [n1, n3]
    c1 = clone_graph(n1)
    assert c1 is not n1 and c1.val == 1
    assert sorted(x.val for x in c1.neighbors) == [2, 4]
    c2 = next(x for x in c1.neighbors if x.val == 2)
    assert c2.neighbors[0] is c1  # cycle closes onto the SAME clone
    c3 = next(x for x in c2.neighbors if x.val == 3)
    assert sorted(x.val for x in c3.neighbors) == [2, 4]
    assert clone_graph(None) is None

    # 4. course schedule
    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False  # 2-cycle
    assert can_finish(5, [[1, 0], [2, 1], [3, 2], [4, 3]]) is True

    # 5. course schedule II - verify returned order respects all edges
    def valid_order(order, n, prereqs):
        if sorted(order) != list(range(n)):
            return False
        pos = {c: i for i, c in enumerate(order)}
        return all(pos[pre] < pos[course] for course, pre in prereqs)

    prereqs = [[1, 0], [2, 0], [3, 1], [3, 2]]
    assert valid_order(find_order(4, prereqs), 4, prereqs)
    assert find_order(1, []) == [0]
    assert find_order(2, [[1, 0], [0, 1]]) == []

    # 6. pacific atlantic
    heights = [
        [1, 2, 2, 3, 5],
        [3, 2, 3, 4, 4],
        [2, 4, 5, 3, 1],
        [6, 7, 1, 4, 5],
        [5, 1, 1, 2, 4],
    ]
    assert pacific_atlantic(heights) == [
        (0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)
    ]
    assert pacific_atlantic([[1]]) == [(0, 0)]

    # 7. word ladder
    assert ladder_length("hit", "cog",
                         ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
    assert ladder_length("a", "c", ["a", "b", "c"]) == 2

    # 8. redundant connection
    assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert find_redundant_connection(
        [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]) == [1, 4]

    # 9. network delay time
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert network_delay_time([[1, 2, 1]], 2, 1) == 1
    assert network_delay_time([[1, 2, 1]], 2, 2) == -1  # node 1 unreachable
    # lazy deletion exercised: a longer direct edge then a shorter 2-hop path
    assert network_delay_time([[1, 3, 10], [1, 2, 1], [2, 3, 1]], 3, 1) == 2

    # 10. accounts merge
    merged = accounts_merge([
        ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
        ["John", "johnsmith@mail.com", "john00@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "johnnybravo@mail.com"],
    ])
    assert merged == sorted([
        ["John", "john00@mail.com", "john_newyork@mail.com",
         "johnsmith@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "johnnybravo@mail.com"],
    ])
    # transitive merge across three accounts sharing pairwise emails
    merged2 = accounts_merge([
        ["A", "a@x.com", "b@x.com"],
        ["A", "b@x.com", "c@x.com"],
        ["A", "c@x.com", "d@x.com"],
    ])
    assert merged2 == [["A", "a@x.com", "b@x.com", "c@x.com", "d@x.com"]]

    print("all 10 problems passed")
