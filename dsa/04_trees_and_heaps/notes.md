# Trees and Heaps — From First Principles

Trees are the first data structure where recursion stops being a technique and
becomes the *definition*. A binary tree is either empty, or a node with two
binary trees hanging off it. That's it. Every tree algorithm falls out of taking
that definition seriously.

---

## 1. The one template that solves most tree problems

Because a tree is defined recursively, *any* function on a tree can be defined
recursively: to compute something about a tree, compute it for the left subtree,
compute it for the right subtree, and **combine the answers at the root**.

```python
def solve(node):
    if node is None:
        return BASE_CASE          # answer for the empty tree
    left  = solve(node.left)      # trust recursion: full answer for left subtree
    right = solve(node.right)     # full answer for right subtree
    return combine(left, right, node.val)
```

This is not "one pattern among many" — it is THE pattern. The design questions
for a new problem are only:

1. **What does the empty tree answer look like?** (base case)
2. **Given complete answers for both subtrees, how do I build the answer for
   the whole tree?** (combine)
3. **Is the final answer at the root, or could it be hidden anywhere in the
   tree?** If it's "anywhere", keep a global/outer variable updated inside the
   recursion, and have the function return the *local* quantity subtrees need.

Watch it instantiate three times:

**Max depth.** Empty tree has depth 0. Given depths of the two subtrees, the
whole tree's depth is one more than the deeper one.

```python
def depth(node):
    if node is None: return 0
    return 1 + max(depth(node.left), depth(node.right))
```

**Diameter** (longest path between any two nodes, in edges). The longest path
either passes through the root — in which case it's `depth(left) +
depth(right)` — or it lives entirely inside one subtree. So the answer isn't
necessarily at the root: this is design question 3. The recursion returns depth
(what parents need), while a nonlocal `best` tracks the through-this-node path
at every node:

```python
def diameter(root):
    best = 0
    def depth(node):
        nonlocal best
        if node is None: return 0
        L, R = depth(node.left), depth(node.right)
        best = max(best, L + R)   # best path THROUGH this node
        return 1 + max(L, R)      # what the parent needs
    depth(root)
    return best
```

This "return one thing, track another" split is the single most reusable trick
in tree interviews (diameter, max path sum, longest univalue path all use it).

**Balance check.** A tree is balanced iff both subtrees are balanced *and*
their depths differ by at most 1. So combine needs both a boolean and a depth
— return both, or use the classic trick of returning depth normally and `-1`
as a poison value meaning "unbalanced below me":

```python
def is_balanced(root):
    def check(node):              # returns depth, or -1 if unbalanced
        if node is None: return 0
        L = check(node.left)
        if L == -1: return -1
        R = check(node.right)
        if R == -1: return -1
        if abs(L - R) > 1: return -1
        return 1 + max(L, R)
    return check(root) != -1
```

All three are O(n) time (each node visited once, O(1) work each) and O(h) space
for the recursion stack, where h is the height — O(log n) for balanced trees,
O(n) worst case for a degenerate chain. Say this in interviews; the stack space
is the part people forget.

---

## 2. The three DFS orders, and what each is *for*

DFS on a binary tree does three things at each node: process the node, recurse
left, recurse right. The three orders are just the three places you can put
"process the node":

| Order | Sequence | Natural use |
|---|---|---|
| **Preorder** | node, left, right | Copying/serializing a tree (you must know the root before you can attach children); prefix notation |
| **Inorder** | left, node, right | BSTs — visits keys in sorted order (proof below); anything "in sorted order" on a BST |
| **Postorder** | left, right, node | Computing bottom-up values (depth, size, delete-a-tree) — you need children's answers before the node's |

Notice the template in section 1 *is* postorder: children first, combine at the
node. That's why postorder is the default shape of tree problems — most
questions ask for a bottom-up quantity.

Iterative preorder with an explicit stack (know this cold — interviewers ask
for it when they want to check you understand what recursion is doing):

```python
def preorder(root):
    out, stack = [], [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.right: stack.append(node.right)  # right first, so left pops first
        if node.left:  stack.append(node.left)
    return out
```

The recursion-to-stack translation: the call stack *is* a stack; making it
explicit trades O(h) implicit space for O(h) explicit space, but frees you from
Python's ~1000-frame recursion limit (relevant for trees with 10^4+ nodes in a
chain — `sys.setrecursionlimit` is the quick fix, an explicit stack is the
robust one).

---

## 3. BFS / level order

DFS goes deep; BFS goes wide. Use a queue: process a node, enqueue its
children. Since children are exactly one level deeper than their parent, the
queue always holds nodes from at most two adjacent levels, in level order —
so nodes come off the queue level by level.

The interview-standard version processes one *whole level* per outer iteration
by snapshotting the queue length:

```python
from collections import deque

def level_order(root):
    if not root: return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):        # len(q) = size of current level, frozen
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

**Why the `for _ in range(len(q))` works:** at the top of the outer loop the
queue contains exactly the current level (invariant: true initially with just
the root; the inner loop removes exactly those nodes and adds exactly the next
level). Freezing `len(q)` before the inner loop is what separates levels —
forgetting to freeze it (iterating while appending) is the classic bug.

BFS is the answer whenever the problem says "level", "nearest", "minimum
depth", or "zigzag". O(n) time, O(w) space where w is max width — up to n/2
for a complete tree, so BFS can use more memory than DFS on bushy trees.

---

## 4. Binary Search Trees

**The invariant** (get this exactly right): for *every* node, ALL keys in its
left subtree are `< node.val`, and ALL keys in its right subtree are
`> node.val`. Not just the children — the entire subtrees.

**Why inorder traversal of a BST is sorted:** proof by induction on tree size.
Empty tree: trivially sorted. Otherwise, inorder emits (inorder of left), then
node.val, then (inorder of right). By induction each part is sorted internally;
by the invariant everything in the left part is < node.val and everything in
the right part is > node.val, so the concatenation is sorted. ∎

This gives you free algorithms: kth smallest = stop inorder after k nodes;
validate BST = check inorder is strictly increasing.

**Validate BST — the bounds method, and the classic bug.** The tempting local
check is:

```python
# WRONG
def bad_validate(node):
    if node is None: return True
    if node.left and node.left.val >= node.val: return False
    if node.right and node.right.val <= node.val: return False
    return bad_validate(node.left) and bad_validate(node.right)
```

This only compares each node to its *children*. But the invariant is global:

```
      5
     / \
    3   8
       / \
      2   9      ← 2 < 8 locally fine, but 2 is in 5's RIGHT subtree: not a BST
```

Every node is fine relative to its parent, yet the tree is invalid. The fix:
carry down the range of allowed values. The root can be anything in
(−∞, +∞); going left tightens the upper bound to the node's value; going right
tightens the lower bound:

```python
def is_valid_bst(root):
    def valid(node, lo, hi):
        if node is None: return True
        if not (lo < node.val < hi): return False
        return valid(node.left, lo, node.val) and valid(node.right, node.val, hi)
    return valid(root, float('-inf'), float('inf'))
```

**Why bounds are correct:** the interval (lo, hi) at a node is by construction
exactly the set of values the BST invariant permits there — each left turn
records "everything below here must be less than this ancestor", each right
turn the mirror. A tree satisfies the invariant iff every node's value lies in
its interval.

BST search/insert are O(h): each comparison discards one entire subtree.
That's O(log n) only if the tree is balanced — a sorted-order insertion
sequence produces a linked list and O(n) operations. (Self-balancing trees fix
this; for interviews you name AVL/red-black and move on, you don't implement
them.)

---

## 5. Lowest Common Ancestor, derived

LCA(p, q) = the deepest node that has both p and q in its subtree (a node
counts as its own ancestor).

Derive it from the template. Ask each subtree: "do you contain p or q (or an
already-found LCA)?" Return the found node, else None. At the root, three
cases:

- Both left and right return something → p and q are in different subtrees →
  this node is the *first* place they're both below you → it's the LCA.
- Only one side returns something → both targets (or their LCA) are on that
  side → pass its answer up.
- Neither → return None.

```python
def lca(root, p, q):
    if root is None or root is p or root is q:
        return root
    L = lca(root.left, p, q)
    R = lca(root.right, p, q)
    if L and R: return root
    return L or R
```

The subtle line is the base case `root is p or root is q`: if you hit p, you
can return p immediately without searching below it — if q is inside p's
subtree, then p itself is the LCA, and returning p is still correct because
the other side will return None and p propagates up. Both cases are covered by
the same return. O(n) time, O(h) space.

(For a *BST* there's a cheaper walk: start at root; if both p, q < node go
left, if both > go right, otherwise you're standing on the LCA — the first
node where the paths split. O(h) with no recursion.)

---

## 6. Serialization

To turn a tree into a string and back, you need a traversal that determines
the tree uniquely. Preorder alone doesn't (many trees share a preorder) —
*unless you record the Nones*. With null markers, preorder is unambiguous:
the first token is the root, then the encoding of the left subtree (which is
self-delimiting — it consumes exactly the tokens it needs), then the right.

```python
def serialize(root):
    out = []
    def go(node):
        if node is None:
            out.append('#'); return
        out.append(str(node.val))
        go(node.left); go(node.right)
    go(root)
    return ','.join(out)

def deserialize(s):
    tokens = iter(s.split(','))
    def build():
        t = next(tokens)
        if t == '#': return None
        node = TreeNode(int(t))
        node.left = build()
        node.right = build()
        return node
    return build()
```

**Why decode works:** decode mirrors encode exactly — encode wrote root, left
encoding, right encoding; decode reads the root token, then recursively
consumes precisely the left encoding, then the right. Induction on subtree
size makes this airtight. The `iter` + `next` sharing of one token stream is
the clean way to express "each recursive call consumes what it needs".

Related classic: rebuild a tree from **preorder + inorder** (no null markers,
but requires distinct values). Preorder's first element is the root; find it
in inorder — everything left of it is the left subtree's inorder, everything
right is the right's, and the sizes tell you how to split the rest of
preorder. Recurse. With a hashmap from value → inorder index, O(n) total.

---

## 7. Heaps from first principles

A heap answers one question fast: "what's the minimum?" — with O(log n)
insert and O(log n) extract-min. It does this with two ideas.

**Idea 1 — the heap property:** every node ≤ its children (min-heap). This
puts the minimum at the root, but says *nothing* about siblings' order — a
heap is much weaker than sorted, which is exactly why it's cheaper to
maintain.

**Idea 2 — the complete-binary-tree array trick.** Keep the tree *complete*
(every level full except possibly the last, filled left to right). A complete
tree can be stored in a plain array with no pointers, reading level by level:

```
index:      0  1  2  3  4  5 ...
parent(i)  = (i - 1) // 2
children(i) = 2i + 1, 2i + 2
```

Check the arithmetic once: children of 0 are 1, 2; children of 1 are 3, 4;
`(3-1)//2 = (4-1)//2 = 1`. ✓ Completeness guarantees no holes, so the array
is dense and cache-friendly, and "last node" = last array element.

**Sift up (after insert):** append at the end (keeps completeness), then swap
with parent while smaller than parent. Each swap fixes the only possible
violation and moves it one level up; height is ⌊log₂ n⌋, so O(log n).

**Sift down (after extract-min):** pop the root by moving the *last* element
into index 0 (keeps completeness), then repeatedly swap it with its *smaller*
child while it's bigger than either child. Must be the smaller child: swapping
with the larger one would put a value above a smaller sibling, re-breaking the
property. O(log n).

**Why build-heap is O(n), not O(n log n).** Heapify an arbitrary array by
sifting *down* every node from the last internal node backward. Naively
that's n sifts × O(log n) = O(n log n). But sift-down costs the node's
*height*, not log n, and most nodes are near the bottom with tiny heights.
Count honestly: about n/2 nodes have height 0 (cost 0), n/4 have height 1,
n/8 have height 2, … so the total is

```
Σ_{h≥0}  (n / 2^{h+1}) · h  =  (n/2) Σ h/2^h  =  (n/2) · 2  =  n
```

The series Σ h/2ʰ = 2 converges (it's the derivative-of-geometric-series
trick), so build-heap is O(n). This is a real interview question; the
one-line answer is "cost per node is its height, and heights sum to O(n)
because half the nodes are leaves."

---

## 8. Heap patterns

**Top-k with a min-heap of size k — and why min, not max.** To keep the k
*largest* items of a stream, maintain a **min**-heap of size k. Counter-
intuitive until you see the role of the root: the root of a min-heap of the
current top-k is the *smallest of the keepers* — the gatekeeper. A new item
either beats the gatekeeper (pop root, push item) or it can't be in the top k
(ignore it). A max-heap's root is the *largest*, which tells you nothing about
whether a newcomer makes the cut — you'd have to store everything. Cost:
n operations on a heap of size k → O(n log k) time, O(k) space, versus
O(n log n) for sorting. In Python: `heapq.nlargest(k, xs)` does exactly this.

**K-way merge.** To merge k sorted lists: the next output element must be the
minimum of the k current heads (each list is sorted, so nothing later in any
list can be smaller). A heap of the k heads gives that minimum in O(log k);
pop it, push that list's next element. Total N elements × O(log k) =
O(N log k). This pattern is also "kth smallest in a sorted matrix" and
"smallest range covering k lists".

**Two-heap running median.** Keep the smaller half of the numbers in a
max-heap `lo` and the larger half in a min-heap `hi`, with
`len(lo) ∈ {len(hi), len(hi)+1}`. The median is then `lo`'s root (odd count)
or the average of the two roots (even). Insert: push onto `lo`, then move
`lo`'s max over to `hi`, then if `hi` got bigger than `lo`, move `hi`'s min
back. Why this maintains max(lo) ≤ min(hi): the element pushed to `hi` is the
max of `lo`-including-the-newcomer, so it is ≥ everything remaining in `lo`;
any element moved back is `hi`'s min, so it is ≤ everything remaining in `hi`
— the halves never interleave, regardless of where the new value belonged.
Each insert is O(log n), median read is O(1). This beats
re-sorting (O(n log n) per query) and is the canonical "two heaps" cue:
*any* problem asking for a running max-of-lower-half / min-of-upper-half.

---

## 9. Python `heapq` quirks (memorize these)

- **Min-heap only.** There is no max-heap flag. For a max-heap, **negate**:
  push `-x`, and negate again on pop. For tuples, negate the key field.
- Operates on a plain list in place: `heapq.heappush(h, x)`,
  `heapq.heappop(h)`, `h[0]` peeks. `heapq.heapify(h)` is the O(n) build.
- **Tuple keys:** heap elements compare with `<`, and tuples compare
  lexicographically, so `(priority, item)` orders by priority with `item` as
  tiebreak. If items aren't comparable (e.g. ListNode), insert a counter:
  `(priority, count, item)` — the counter breaks all ties before Python ever
  compares the items. Forgetting this raises `TypeError` the first time two
  priorities collide.
- `heapq.heappushpop(h, x)` (push then pop) is faster than the two calls and
  is exactly the top-k inner loop; `heapq.heapreplace(h, x)` is pop-then-push.
- No delete-arbitrary or decrease-key. The standard workaround is **lazy
  deletion**: leave stale entries in the heap and skip them on pop (used in
  Dijkstra — see the graphs notes).

---

## 10. Tries — prefix trees from first principles

**Derive the structure from the query.** A hashset of N words answers "is `w` a
word?" in O(L) (hash the L characters), but it is *helpless* on "is `w` a
**prefix** of any word?" — hashing destroys prefix structure, so the only recourse
is scanning all N keys. The fix: store the words as **paths in a tree**. A node
per prefix; an edge per character; words that share a prefix share the path.

```python
class TrieNode:
    def __init__(self):
        self.children = {}     # char -> TrieNode
        self.terminal = False  # does a word END here?
```

That's the whole structure: a node is a dict of children plus one flag. The flag
is load-bearing — without it you can't distinguish "app is a word" from "app is
merely a prefix of apple"; the node for `app` exists either way.

- **insert(word):** walk from the root, creating missing children; mark the last
  node terminal. O(L).
- **search(word):** walk; fail on a missing child; at the end, return the
  `terminal` flag. O(L).
- **startsWith(prefix):** same walk; at the end return True unconditionally —
  reaching the node at all means some word passes through it. O(L).

**Why the trie wins.** Both lookups cost O(L) — *independent of N*, the number of
words. The shared-prefix compression is also a space story: a million words
starting with `pre` store those three nodes once. Against the alternative
"hashset of every prefix of every word" (which does answer startsWith), the trie
uses strictly less space and adds per-node structure you can traverse — which is
exactly what the next two problems exploit.

**Wildcard search = DFS branching.** Add `.` matching any single character
(WordDictionary, LC 211). A literal character follows one edge; a `.` must try
**every** child — so search becomes a DFS over (node, position-in-pattern):

```
match(node, i): if i == len(word): return node.terminal
                if word[i] != '.': one child or fail
                else: any(match(child, i+1) for child in node.children.values())
```

Worst case for a pattern of all dots is O(total nodes), and saying that honestly
matters: wildcards buy expressiveness by giving back the O(L) guarantee.

**Word Search II — why trie + backtracking beats per-word search.** Find which of
k dictionary words appear in an m×n letter grid (paths of adjacent cells). Per-word
grid backtracking costs k full searches — the k words re-explore the same grid
paths k times. Instead, build a trie of ALL the words and run ONE grid DFS that
walks the trie in lockstep with the grid: from each cell, step only into children
that exist in the trie. The trie *is* the pruning — a grid path is abandoned the
moment it stops being a prefix of any word, and all k words are matched
simultaneously. Two refinements interviewers look for:

1. **Report at terminal nodes, then unset the flag** (or collect into a set) so a
   word isn't reported once per occurrence.
2. **Remove dead leaves:** after a word is found, its trie path may serve no other
   word — delete childless nodes on the way back up (`if not child.children:
   del children[ch]`). The trie shrinks as words are found, and the grid DFS
   prunes harder and harder; on adversarial inputs this is the difference between
   passing and timing out.

**Recognition cues:** "prefix", "autocomplete", "starts with", "add and search
words" (with wildcards), "many words against one grid/stream", "longest common
prefix over many strings". If the query is exact-membership only, a hashset is
simpler — reach for the trie when *prefix structure* is queried or shared.

**Common bugs:** forgetting the terminal flag (search == startsWith); marking
terminal on the wrong node (one short of the full word); in Word Search II,
unmarking the visited cell but forgetting to restore/prune trie state; treating
`.` as a literal child lookup.

---

## 11. Recognition cheat sheet

| Cue in problem | Reach for |
|---|---|
| "depth / path / subtree property" | Postorder template (§1), maybe with a nonlocal best |
| "level by level", "nearest leaf", "right side view" | BFS |
| "kth smallest in BST", "sorted order" | Inorder |
| "validate", "range" on BST | Bounds-passing DFS |
| "k largest / smallest / most frequent" | Size-k min-heap (or max, mirrored) |
| "merge k sorted anything" | Heap of heads |
| "running median / percentile" | Two heaps |
| "schedule most frequent first", "greedy by priority" | Max-heap (negate) |
| "prefix / starts with / autocomplete" | Trie (§10) |
| "many words vs one grid/stream" | Trie + one shared DFS (§10) |

Classic bugs to name-check: local-vs-global BST validation (§4); forgetting to
freeze `len(q)` in level order (§3); using a max-heap for top-k (§8);
un-comparable tuple ties in heapq (§9); recursion depth limits on degenerate
trees (§2).

## Exit checklist

- [ ] Cold, blank editor: write the universal tree-recursion template, then instantiate it for max depth and diameter — 4 min
- [ ] State what each DFS order is for (pre/in/post) with one canonical use apiece — 2 min
- [ ] Validate a BST correctly by passing bounds down — and say why the local parent-child check is wrong — 3 min
- [ ] Derive the LCA recursion and state its return-value contract — 4 min
- [ ] Explain heap sift-up/sift-down and why build-heap is O(n), not O(n log n) — 4 min
- [ ] Top-k: which heap, the size-k invariant, and the O(n log k) complexity — 2 min
- [ ] Recite the Python heapq quirks: min-heap only, negation for max, tuple tie-breaking — 2 min

All boxes cold before moving on — a miss goes into practice/queue.tsv.
