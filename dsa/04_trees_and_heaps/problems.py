"""Trees and heaps — 14 classic problems, solved from first principles.

Every solution follows the template from notes.md: base case for the empty
tree, recurse on subtrees, combine at the node. Heap problems use heapq
(min-heap only — negate for max); the trie problems at the end instantiate
notes.md section 10. Run this file; it asserts every test and prints a
success line.
"""

import heapq
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(values):
    """Build a tree from a level-order list with None gaps (LeetCode format)."""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    q = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            q.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            q.append(node.right)
        i += 1
    return root


# ---------------------------------------------------------------------------
# 1. Maximum Depth of Binary Tree
# ---------------------------------------------------------------------------
def max_depth(root):
    """Return the number of nodes on the longest root-to-leaf path.

    Key insight: the purest instance of the tree template — empty tree has
    depth 0; otherwise 1 + the deeper subtree.
    Complexity: O(n) time, O(h) recursion stack.
    """
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


# ---------------------------------------------------------------------------
# 2. Diameter of Binary Tree
# ---------------------------------------------------------------------------
def diameter(root):
    """Return the length (in edges) of the longest path between any two nodes.

    Key insight: the best path through a node is depth(left) + depth(right),
    but the answer may live anywhere — so return depth to the parent while a
    nonlocal tracks the global best ("return one thing, track another").
    Complexity: O(n) time, O(h) space.
    """
    best = 0

    def depth(node):
        nonlocal best
        if node is None:
            return 0
        left, right = depth(node.left), depth(node.right)
        best = max(best, left + right)
        return 1 + max(left, right)

    depth(root)
    return best


# ---------------------------------------------------------------------------
# 3. Invert Binary Tree
# ---------------------------------------------------------------------------
def invert(root):
    """Mirror the tree: swap left and right subtrees everywhere.

    Key insight: invert(tree) = node with children invert(right), invert(left).
    Swap first or recurse first — both work; just don't lose a pointer.
    Complexity: O(n) time, O(h) space.
    """
    if root is None:
        return None
    root.left, root.right = invert(root.right), invert(root.left)
    return root


# ---------------------------------------------------------------------------
# 4. Binary Tree Level Order Traversal
# ---------------------------------------------------------------------------
def level_order(root):
    """Return values grouped by level, top to bottom.

    Key insight: BFS with a frozen len(q) snapshot per outer iteration — the
    queue holds exactly one level at the top of each pass.
    Complexity: O(n) time, O(w) space (w = max width).
    """
    if root is None:
        return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):  # freeze: only current level's nodes
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out


# ---------------------------------------------------------------------------
# 5. Validate Binary Search Tree
# ---------------------------------------------------------------------------
def is_valid_bst(root):
    """Return True iff the tree satisfies the global BST invariant.

    Key insight: comparing each node only to its children is the classic bug
    (the invariant constrains whole subtrees, not just children). Carry down
    an (lo, hi) open interval: left turn tightens hi, right turn tightens lo.
    Complexity: O(n) time, O(h) space.
    """
    def valid(node, lo, hi):
        if node is None:
            return True
        if not (lo < node.val < hi):
            return False
        return valid(node.left, lo, node.val) and valid(node.right, node.val, hi)

    return valid(root, float("-inf"), float("inf"))


# ---------------------------------------------------------------------------
# 6. Lowest Common Ancestor of a Binary Tree
# ---------------------------------------------------------------------------
def lowest_common_ancestor(root, p, q):
    """Return the deepest node having both p and q in its subtree.

    Key insight: ask each subtree "did you find p or q?". If both sides
    report a find, this node is the split point — the LCA. If you land on p
    or q itself, return it immediately: if the other target is below it, it
    is its own ancestor and still the right answer.
    Complexity: O(n) time, O(h) space.
    """
    if root is None or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right


# ---------------------------------------------------------------------------
# 7. Construct Binary Tree from Preorder and Inorder Traversal
# ---------------------------------------------------------------------------
def build_tree(preorder, inorder):
    """Rebuild the unique tree with the given traversals (distinct values).

    Key insight: preorder[0] is the root; its position in inorder splits
    inorder into left/right subtrees, and those sizes split the rest of
    preorder. A value->index map plus index arithmetic avoids O(n) slicing
    per call, giving O(n) total instead of O(n^2).
    Complexity: O(n) time, O(n) space (map + recursion).
    """
    idx = {v: i for i, v in enumerate(inorder)}
    pre_i = 0

    def rec(in_lo, in_hi):  # build from inorder[in_lo:in_hi]
        nonlocal pre_i
        if in_lo >= in_hi:
            return None
        root = TreeNode(preorder[pre_i])
        pre_i += 1
        mid = idx[root.val]
        root.left = rec(in_lo, mid)      # left consumed before right —
        root.right = rec(mid + 1, in_hi)  # matches preorder's order
        return root

    return rec(0, len(inorder))


# ---------------------------------------------------------------------------
# 8. Serialize and Deserialize Binary Tree
# ---------------------------------------------------------------------------
def serialize(root):
    """Encode a tree to a comma-separated string.

    Key insight: preorder WITH null markers is unambiguous — the encoding of
    a subtree is self-delimiting, so decode can mirror encode exactly.
    Complexity: O(n) time and space, both directions.
    """
    out = []

    def go(node):
        if node is None:
            out.append("#")
            return
        out.append(str(node.val))
        go(node.left)
        go(node.right)

    go(root)
    return ",".join(out)


def deserialize(data):
    tokens = iter(data.split(","))

    def rec():
        t = next(tokens)
        if t == "#":
            return None
        node = TreeNode(int(t))
        node.left = rec()
        node.right = rec()
        return node

    return rec()


# ---------------------------------------------------------------------------
# 9. Kth Largest Element in a Stream
# ---------------------------------------------------------------------------
class KthLargest:
    """Maintain the kth largest element as values stream in.

    Key insight: keep a MIN-heap of the k largest seen — its root is the
    gatekeeper (smallest keeper), which is exactly the kth largest. A
    max-heap's root says nothing about whether a newcomer makes the cut.
    Complexity: O(log k) per add, O(k) space.
    """

    def __init__(self, k, nums):
        self.k = k
        self.heap = nums[:]
        heapq.heapify(self.heap)          # O(n) build
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heapreplace(self.heap, val)  # pop root, push val: one sift
        return self.heap[0]


# ---------------------------------------------------------------------------
# 10. Merge k Sorted Lists
# ---------------------------------------------------------------------------
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_k_lists(lists):
    """Merge k sorted linked lists into one sorted list.

    Key insight: the next output must be the min of the k current heads
    (lists are sorted, nothing later can be smaller). Heap of heads gives it
    in O(log k). Tuples carry (val, i, node); the index i breaks ties so
    Python never tries to compare two ListNodes.
    Complexity: O(N log k) time for N total nodes, O(k) space.
    """
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapq.heapify(heap)
    dummy = tail = ListNode()
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next


# ---------------------------------------------------------------------------
# 11. Find Median from Data Stream
# ---------------------------------------------------------------------------
class MedianFinder:
    """Support add_num and O(1) find_median over a growing stream.

    Key insight: two heaps — max-heap `lo` (smaller half, negated values)
    and min-heap `hi` (larger half), with len(lo) in {len(hi), len(hi)+1}.
    Routing every push through lo then rebalancing keeps max(lo) <= min(hi)
    no matter where the new value belongs.
    Complexity: O(log n) per add, O(1) per median, O(n) space.
    """

    def __init__(self):
        self.lo = []  # max-heap via negation: smaller half
        self.hi = []  # min-heap: larger half

    def add_num(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))  # lo's max -> hi
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2.0


# ---------------------------------------------------------------------------
# 12. Implement Trie (Prefix Tree)
# ---------------------------------------------------------------------------
class Trie:
    """insert(word), search(word), starts_with(prefix) — all O(L).

    Key insight (notes.md section 10): a hashset answers exact membership but
    hashing destroys prefix structure. Store words as root-paths instead —
    node = children dict + terminal flag — and both queries become the same
    walk, differing only in whether the terminal flag is consulted. Cost is
    O(L) independent of the number of words; shared prefixes are stored once.
    Complexity: O(L) per operation, O(total chars) space.
    """

    def __init__(self):
        self.root = {}                    # char -> child dict; "$" marks ends

    def _walk(self, s):
        node = self.root
        for ch in s:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True                  # terminal flag: a word ENDS here

    def search(self, word):
        node = self._walk(word)
        return node is not None and "$" in node

    def starts_with(self, prefix):
        return self._walk(prefix) is not None


# ---------------------------------------------------------------------------
# 13. Design Add and Search Words (wildcard '.')
# ---------------------------------------------------------------------------
class WordDictionary:
    """add_word(word); search(word) where '.' matches any single character.

    Key insight: a literal character follows one trie edge, but '.' must try
    EVERY child — so search generalizes from a walk to a DFS over
    (node, position). Wildcards trade away the O(L) guarantee: an all-dots
    pattern can touch every node, and saying so is part of the answer.
    Complexity: O(L) add; search O(L) without dots, O(total nodes) worst case.
    """

    def __init__(self):
        self.root = {}

    def add_word(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True

    def search(self, word):
        def dfs(node, i):
            if i == len(word):
                return "$" in node
            ch = word[i]
            if ch != ".":
                return ch in node and dfs(node[ch], i + 1)
            return any(dfs(child, i + 1)          # '.' branches over children
                       for key, child in node.items() if key != "$")

        return dfs(self.root, 0)


# ---------------------------------------------------------------------------
# 14. Word Search II
# ---------------------------------------------------------------------------
def find_words(board, words):
    """Return every word (from a dictionary of k words) that can be traced in
    the m x n letter grid through adjacent (up/down/left/right) cells, each
    cell used at most once per word.

    Key insight: per-word grid backtracking re-explores the same paths k
    times. Build ONE trie of all the words and DFS the grid walking the trie
    in lockstep — a path is abandoned the moment it stops being a prefix of
    any word, and all words are matched in a single sweep. Pop the terminal
    flag on a match (each word reported once) and prune childless trie nodes
    on the way out, so the search space shrinks as words are found.
    Complexity: O(total word chars) to build; DFS O(m*n * 3^L) worst case
    (first step has 4 choices, later steps at most 3 — no going back).
    """
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = w                          # store the word at its terminal

    out = []
    rows, cols = len(board), len(board[0])

    def dfs(r, c, node):
        ch = board[r][c]
        child = node.get(ch)
        if child is None:
            return                             # not a prefix of any word left
        word = child.pop("$", None)            # report once, then unset
        if word is not None:
            out.append(word)
        board[r][c] = "#"                      # choose: mark visited
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, child)             # explore
        board[r][c] = ch                       # unchoose: restore the cell
        if not child:                          # dead leaf: no words need it
            del node[ch]                       # prune — future DFS skips it

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. max depth
    assert max_depth(build([3, 9, 20, None, None, 15, 7])) == 3
    assert max_depth(build([1, None, 2])) == 2
    assert max_depth(None) == 0

    # 2. diameter
    assert diameter(build([1, 2, 3, 4, 5])) == 3      # path 4-2-1-3 (or 5-2-1-3)
    assert diameter(build([1, 2])) == 1
    assert diameter(None) == 0

    # 3. invert
    inv = invert(build([4, 2, 7, 1, 3, 6, 9]))
    assert level_order(inv) == [[4], [7, 2], [9, 6, 3, 1]]
    assert invert(None) is None

    # 4. level order
    assert level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [9, 20], [15, 7]]
    assert level_order(None) == []

    # 5. validate BST — including the local-check trap tree
    assert is_valid_bst(build([2, 1, 3])) is True
    assert is_valid_bst(build([5, 1, 4, None, None, 3, 6])) is False
    trap = build([5, 3, 8, None, None, 2, 9])  # 2 valid vs parent 8, not vs root 5
    assert is_valid_bst(trap) is False
    assert is_valid_bst(build([1, 1])) is False  # duplicates not allowed

    # 6. LCA
    root = build([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p = root.left                # 5
    q = root.right               # 1
    assert lowest_common_ancestor(root, p, q).val == 3
    q = root.left.right.right    # 4, inside 5's subtree
    assert lowest_common_ancestor(root, p, q).val == 5  # node is its own ancestor

    # 7. build from preorder + inorder
    t = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    assert level_order(t) == [[3], [9, 20], [15, 7]]
    assert build_tree([1], [1]).val == 1

    # 8. serialize / deserialize round-trip
    original = build([1, 2, 3, None, None, 4, 5])
    s = serialize(original)
    assert serialize(deserialize(s)) == s
    assert level_order(deserialize(s)) == [[1], [2, 3], [4, 5]]
    assert serialize(deserialize(serialize(None))) == "#"

    # 9. kth largest in stream
    kl = KthLargest(3, [4, 5, 8, 2])
    assert [kl.add(v) for v in [3, 5, 10, 9, 4]] == [4, 5, 5, 8, 8]
    kl2 = KthLargest(1, [])
    assert kl2.add(-3) == -3
    assert kl2.add(0) == 0

    # 10. merge k sorted lists
    def from_list(xs):
        dummy = tail = ListNode()
        for x in xs:
            tail.next = ListNode(x)
            tail = tail.next
        return dummy.next

    def to_list(node):
        out = []
        while node:
            out.append(node.val)
            node = node.next
        return out

    merged = merge_k_lists([from_list([1, 4, 5]), from_list([1, 3, 4]), from_list([2, 6])])
    assert to_list(merged) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert merge_k_lists([]) is None
    assert merge_k_lists([None, from_list([1])]).val == 1

    # 11. median from data stream
    mf = MedianFinder()
    mf.add_num(1)
    mf.add_num(2)
    assert mf.find_median() == 1.5
    mf.add_num(3)
    assert mf.find_median() == 2.0
    mf2 = MedianFinder()
    for v in [5, 15, 1, 3]:
        mf2.add_num(v)
    assert mf2.find_median() == 4.0

    # 12. Trie
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple")
    assert not trie.search("app")            # prefix, but no word ends there
    assert trie.starts_with("app")
    trie.insert("app")
    assert trie.search("app")                # now a word AND a prefix
    assert not trie.starts_with("apx")
    assert not trie.search("apples")         # walking past a terminal

    # 13. WordDictionary
    wd = WordDictionary()
    for w in ["bad", "dad", "mad"]:
        wd.add_word(w)
    assert not wd.search("pad")
    assert wd.search("bad")
    assert wd.search(".ad")                  # wildcard first char
    assert wd.search("b..")
    assert not wd.search("b.")               # length must match exactly
    assert not wd.search("...x")
    assert wd.search("...")                  # all dots

    # 14. word search II
    board = [["o", "a", "a", "n"],
             ["e", "t", "a", "e"],
             ["i", "h", "k", "r"],
             ["i", "f", "l", "v"]]
    got = find_words(board, ["oath", "pea", "eat", "rain"])
    assert sorted(got) == ["eat", "oath"]
    assert board[0][0] == "o"                # grid restored after backtracking
    assert sorted(find_words([["a", "b"]], ["ba", "ab", "abc"])) == ["ab", "ba"]
    assert find_words([["a"]], ["aa"]) == []  # cell reuse forbidden
    dup_board = [["a", "a"]]
    assert find_words(dup_board, ["aa"]) == ["aa"]  # reported once, not per path

    print("all 14 problems passed")
