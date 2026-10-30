# Timed Drills

Thirteen drills, each simulating a reported interview round. Protocol: read the spec,
start the timer, write code in an empty file, run it, *then* unfold the
solution and diff. Every reference solution below has been executed and tested.
"What good looks like" describes the bar under time pressure, not perfection.

Numbering matches the round-mapping table in [README.md](README.md).

---

## Drill 1 - Multi-head causal self-attention in numpy

**Time box: 25 min · Simulates: OpenAI / Anthropic "attention from scratch" · Reference: module 4, `attention_scratch.ipynb`**

Isolates: the single most-asked implementation - can you produce
`softmax(QKᵀ/√d_h + mask)V` with correct head reshapes, from a blank buffer.

Spec: `mha_causal(x, Wq, Wk, Wv, Wo, n_head)` with `x: (T, d)` and all weights
`(d, d)`. Return `(T, d)`. Causal, numerically stable softmax, no loops over
heads or positions.

What good looks like: you say the plan first ("project, split heads, scores,
mask, softmax, weighted sum, merge, output-project"), annotate every shape,
and finish with a *causality test*: perturb the last token, assert earlier
outputs unchanged. The three classic time-pressure bugs: softmax on the wrong
axis, `reshape` where a `transpose` is needed on the head split, forgetting √d_h.

<details><summary>Reference solution</summary>

```python
import numpy as np

def mha_causal(x, Wq, Wk, Wv, Wo, n_head):
    T, d = x.shape
    hd = d // n_head
    q, k, v = x @ Wq, x @ Wk, x @ Wv                    # (T, d) each

    def heads(a):                                       # (T, d) -> (h, T, hd)
        return a.reshape(T, n_head, hd).transpose(1, 0, 2)

    q, k, v = heads(q), heads(k), heads(v)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)     # (h, T, T)
    mask = np.tril(np.ones((T, T), dtype=bool))         # keep j <= i
    scores = np.where(mask, scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)        # stable softmax
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)                  # rows sum to 1
    out = w @ v                                         # (h, T, hd)
    out = out.transpose(1, 0, 2).reshape(T, d)          # concat heads
    return out @ Wo

# --- the test an interviewer wants to see, unprompted ---
rng = np.random.default_rng(0)
T, d, h = 6, 8, 2
x = rng.standard_normal((T, d))
Ws = [rng.standard_normal((d, d)) * 0.3 for _ in range(4)]
y1 = mha_causal(x, *Ws, h)
x2 = x.copy(); x2[-1] += 10.0            # perturb the LAST token
y2 = mha_causal(x2, *Ws, h)
assert np.allclose(y1[:-1], y2[:-1]), "causality violated"
assert not np.allclose(y1[-1], y2[-1])
```

The `-np.inf` before max-subtraction is safe: each row keeps at least the
diagonal, so no all-`-inf` row exists and no NaN appears. The head split
`reshape(T, h, hd).transpose(1, 0, 2)` is correct; `reshape(h, T, hd)` is the
dangerous variant: it shuffles channels across heads and still runs without
errors.
</details>

---

## Drill 2 - BPE tokenizer: train + encode + decode

**Time box: 40 min · Simulates: Anthropic from-scratch build · Reference: module 4 notes (tokenization)**

Isolates: greedy merge logic, and the encode-time subtlety that merges must be
applied in *training order*, not by count on the new text.

Spec, pure Python, byte-level: `train_bpe(text, num_merges) -> merges`;
`encode(text, merges) -> list[int]`; `decode(ids, merges) -> str`. Round-trip
must be exact, including on text with characters never seen in training
(that's why byte-level: base vocab = 256 bytes, no OOV possible).

What good looks like: round-trip test on training text AND unseen text AND
non-ASCII, plus a compression print (bytes → tokens). The classic bug:
encoding by "most frequent pair in this text" instead of "earliest-learned
merge" - states the difference out loud.

<details><summary>Reference solution</summary>

```python
def pair_counts(ids):
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train_bpe(text, num_merges):
    ids = list(text.encode("utf-8"))            # base vocab: 256 bytes
    merges = {}                                 # (a, b) -> new_id, in order
    for j in range(num_merges):
        counts = pair_counts(ids)
        if not counts:
            break
        best = max(counts, key=counts.get)
        if counts[best] < 2:
            break                               # nothing left worth merging
        merges[best] = 256 + j
        ids = merge(ids, best, 256 + j)
    return merges

def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = pair_counts(ids)
        # apply the EARLIEST-LEARNED merge present, not the most frequent
        best = min(counts, key=lambda p: merges.get(p, float("inf")))
        if best not in merges:
            break
        ids = merge(ids, best, merges[best])
    return ids

def bpe_decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), nid in merges.items():          # dict preserves merge order
        vocab[nid] = vocab[a] + vocab[b]        # children already resolved
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# --- tests ---
text = "low lower lowest newest newest newest widest widest"
merges = train_bpe(text, 20)
ids = bpe_encode(text, merges)
assert bpe_decode(ids, merges) == text
assert len(ids) < len(text.encode())
assert bpe_decode(bpe_encode("lowest wide new", merges), merges) == "lowest wide new"
assert bpe_decode(bpe_encode("héllo ☃", merges), merges) == "héllo ☃"
```

Why encode uses `min` over merge *rank*: training produced a merge ordering;
tokens like `est` only exist because `es` was learned first. Applying a later
merge first can strand the text in a state the training sequence never saw.
Decode works in one pass because `merges` insertion order guarantees children
of every merge are already in `vocab`. Complexity is O(merges · len); fine for
an interview - mention (don't build) the priority-queue version.
</details>

---

## Drill 3 - Temperature / top-k / top-p sampling

**Time box: 20 min · Simulates: Anthropic sampling-from-scratch · Reference: module 4 notes (decoding)**

Isolates: the exact order of operations (temperature → top-k on logits →
softmax → top-p on probabilities → renormalize) and off-by-ones in "smallest
set with mass ≥ p".

Spec: `sample_logits(logits, temperature=1.0, top_k=None, top_p=None, rng)`
returning an int. `temperature=0` means greedy. Stable softmax mandatory.

What good looks like: you articulate that top-p keeps the *smallest* prefix of
the sorted distribution whose mass reaches p (so the boundary token is
included), and you test the degenerate cases: `top_k=1` ≡ greedy,
`top_p→0` ≡ greedy, huge temperature ≈ uniform.

<details><summary>Reference solution</summary>

```python
import numpy as np

def sample_logits(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    logits = np.array(logits, dtype=np.float64)
    if temperature == 0:                       # greedy shortcut
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None:
        kth = np.sort(logits)[-top_k]          # k-th largest value
        logits = np.where(logits < kth, -np.inf, logits)
    logits -= logits.max()                     # stable softmax
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p is not None:
        order = np.argsort(-probs)             # descending
        csum = np.cumsum(probs[order])
        keep = order[: np.searchsorted(csum, top_p) + 1]  # smallest set >= p
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()          # renormalize survivors
    return int(rng.choice(len(probs), p=probs))

# --- tests ---
rng = np.random.default_rng(0)
logits = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
assert sample_logits(logits, temperature=0) == 0
assert all(sample_logits(logits, top_k=2, rng=rng) in (0, 1) for _ in range(200))
assert all(sample_logits(logits, top_p=0.01, rng=rng) == 0 for _ in range(50))
counts = np.bincount([sample_logits(logits, temperature=100.0, rng=rng)
                      for _ in range(2000)], minlength=5)
assert (counts > 0).all()                      # high T reaches the tail
```

Order matters and interviewers probe it: temperature reshapes the distribution
*before* truncation decides what survives; top-k operates on logits (a value
threshold), top-p on the cumulative probability mass. `searchsorted(csum, p)+1`
implements "include the token that crosses p" - dropping the `+1` silently
turns `top_p=0.9` into a harsher filter than specified.
</details>

---

## Drill 4 - Logistic regression with a manual backward pass

**Time box: 25 min · Simulates: general ML screen, "no autograd" · Reference: modules 1–2, `linear_models_scratch.ipynb`**

Isolates: the derivation you must own cold - `∂L/∂z = (p − y)/N` for
sigmoid + BCE - and turning it into two matrix products.

Spec: torch tensors, but **no** `requires_grad`, no `.backward()`, no
`torch.optim`. Generate separable-ish data, train full-batch gradient descent
to >90% accuracy, and verify your manual gradient against autograd at the end
(that check is allowed - it's the test, not the method).

What good looks like: you *derive* the gradient on paper-in-comments - say
"sigmoid+BCE collapses to p−y, same as softmax+CE" - rather than pattern-match
it, and you know why the `1e-12` clamps live in the loss print but nowhere in
the gradient (the p−y form is already stable).

<details><summary>Reference solution</summary>

```python
import torch
torch.manual_seed(0)

N, D = 400, 5
w_true = torch.randn(D)
X = torch.randn(N, D)
y = (X @ w_true + 0.3 * torch.randn(N) > 0).float()

w, b = torch.zeros(D), torch.zeros(1)
lr = 0.5
for step in range(300):
    z = X @ w + b                       # (N,)   forward
    p = torch.sigmoid(z)                # (N,)
    # Backward, by hand. L = -mean(y log p + (1-y) log(1-p)).
    # dL/dp = (p-y) / (p(1-p)N);  dp/dz = p(1-p)  =>  dL/dz = (p-y)/N.
    g = (p - y) / N                     # (N,)
    gw = X.T @ g                        # (D,)  dz/dw = X
    gb = g.sum()                        #       dz/db = 1
    w -= lr * gw
    b -= lr * gb

p = torch.sigmoid(X @ w + b)
acc = ((p > 0.5).float() == y).float().mean()
assert acc > 0.9

# --- verify against autograd (the allowed check) ---
w_a = w.clone().requires_grad_(True)
b_a = b.clone().requires_grad_(True)
L = torch.nn.functional.binary_cross_entropy(torch.sigmoid(X @ w_a + b_a), y)
L.backward()
assert torch.allclose(w_a.grad, X.T @ ((torch.sigmoid(X @ w + b) - y) / N),
                      atol=1e-5)
```

The p(1−p) factors cancel between dL/dp and dp/dz - that cancellation is the
whole reason the pairing "sigmoid + BCE" (and "softmax + CE") is standard: the
gradient never divides by a probability, so saturation doesn't destroy it.
Module 1 derives this in full; module 3 does the softmax version.
</details>

---

## Drill 5 - Numerically stable softmax + cross-entropy, with tests

**Time box: 15 min · Simulates: DeepMind "code must run" numerics screen · Reference: module 1 notes (log-sum-exp), module 3**

Isolates: the max-subtraction trick and log-sum-exp - writing `log(softmax)`
without ever materializing a probability.

Spec: numpy. `log_softmax(z)` for `(N, V)`, `cross_entropy(logits, targets)`
returning mean NLL. Must survive logits of ±1e4 with no inf/nan, and match
`torch.nn.functional.cross_entropy` to 1e-9.

What good looks like: 15 minutes is generous - spend the surplus writing the
three tests below. Saying "softmax is shift-invariant, so I subtract the row
max; then log-softmax is `z − max − log Σ exp`" is the whole interview.

<details><summary>Reference solution</summary>

```python
import numpy as np

def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)      # shift-invariance: exp <= 1 now
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def cross_entropy(logits, targets):
    ls = log_softmax(np.asarray(logits, dtype=np.float64))
    return float(-ls[np.arange(len(targets)), targets].mean())

# --- tests ---
z = np.array([[1e4, 0.0, -1e4], [5.0, 5.0, 5.0]])
assert np.isfinite(log_softmax(z)).all()                       # no overflow
assert abs(cross_entropy(np.zeros((4, 7)), np.array([0, 1, 2, 3]))
           - np.log(7)) < 1e-9                                 # uniform -> ln V

import torch
zt = torch.randn(8, 10, dtype=torch.float64)
tt = torch.randint(10, (8,))
assert abs(cross_entropy(zt.numpy(), tt.numpy())
           - torch.nn.functional.cross_entropy(zt, tt).item()) < 1e-9
```

Never compute `log(softmax(z))` as two steps: softmax underflows to exact 0
for logits ~-750 apart in float64 (~-87 in float32), and `log(0) = -inf`. The
fused form subtracts before exponentiating, so the largest exp is exactly 1.
Same identity powers the `ln V` init check in the debugging playbook.
</details>

---

## Drill 6 - k-means in numpy

**Time box: 20 min · Simulates: classical-ML screen · Reference: module 2, `kmeans_pca_scratch.ipynb`**

Isolates: vectorized pairwise distances via broadcasting, and knowing the two
gotchas - empty clusters and local optima - before being asked.

Spec: `kmeans(X, k)` → centers, assignments, inertia. Assignment step must be
loop-free (broadcast `(N,1,D)` vs `(1,k,D)`). Handle empty clusters. Add
restarts keeping best inertia.

What good looks like: you name the objective (within-cluster sum of squares),
note each step monotonically decreases it (hence convergence), and volunteer
"random init can split one blob and merge two - that's why restarts /
k-means++". If you finish early: mention the O(NkD) per-iteration cost.

<details><summary>Reference solution</summary>

```python
import numpy as np

def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]         # (k, D) init
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # (N, k)
        z = d2.argmin(axis=1)                                # (N,) assign
        newC = np.array([
            X[z == j].mean(axis=0) if (z == j).any() else C[j]  # empty guard
            for j in range(k)])
        if np.allclose(newC, C):
            break                                            # converged
        C = newC
    return C, z, d2[np.arange(len(X)), z].sum()              # inertia

def kmeans_restarts(X, k, n_init=5, seed=0):
    """k-means only finds a local optimum -- restart, keep best inertia."""
    best = None
    for s in range(n_init):
        out = kmeans(X, k, seed=seed + s)
        if best is None or out[2] < best[2]:
            best = out
    return best

# --- test: three well-separated blobs must be recovered ---
rng = np.random.default_rng(1)
blob = np.concatenate([rng.normal(c, 0.3, size=(50, 2))
                       for c in [(0, 0), (5, 5), (0, 5)]])
C, z, inertia = kmeans_restarts(blob, 3)
for s in range(3):
    assert len(set(z[s * 50:(s + 1) * 50])) == 1   # each blob -> one cluster
```

Honest detail: with a single seed-0 init this test *fails* - two centers land
in one blob and k-means happily converges there. That's not a bug, it's the
algorithm; the restarts wrapper is part of a correct answer, not a flourish.
Assignment step decreases inertia (nearest center), update step decreases it
(mean minimizes squared distance), so it converges - to a local optimum only.
</details>

---

## Drill 7 - AUC from scores, pure Python

**Time box: 15 min · Simulates: metrics screen · Reference: module 5, `metrics_scratch.ipynb` and notes**

Isolates: knowing AUC's rank-statistic identity - P(random positive scores
above random negative) - and handling ties with average ranks, no curve built.

Spec: `auc(scores, labels)` in pure Python. Must match
`sklearn.roc_auc_score` exactly, including tied scores. Formula:
`AUC = (Σ ranks(pos) − n₊(n₊+1)/2) / (n₊ n₋)` with 1-based average ranks.

What good looks like: you state the probabilistic meaning first, derive why
the formula works (Σ ranks of positives counts, for each positive, how many
items sit at-or-below it), and treat ties by giving the whole tied group the
mean of its rank range - which makes each tied pos/neg pair contribute ½.

<details><summary>Reference solution</summary>

```python
def auc(scores, labels):
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])   # ascending
    ranks = [0.0] * n                     # 1-based, ties share average rank
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + j + 1) / 2             # mean of ranks i+1 .. j
        for t in range(i, j):
            ranks[order[t]] = avg
        i = j
    n_pos = sum(labels)
    n_neg = n - n_pos
    r_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

# --- tests ---
assert auc([0.1, 0.9], [0, 1]) == 1.0     # perfect ranking
assert auc([0.9, 0.1], [0, 1]) == 0.0     # perfectly wrong
assert auc([0.5, 0.5], [0, 1]) == 0.5     # tie -> coin flip
```

Why it works: rank of a positive = 1 + (#items strictly below it). Summing
over positives counts positive-below-positive pairs too; subtracting
`n₊(n₊+1)/2` (the ranks the positives would have if they only counted each
other) leaves exactly #(pos above neg) pairs, ties counting ½ via average
ranks. Divide by total pairs `n₊n₋`. This is the Mann–Whitney U statistic  - 
module 5 builds the same number from the ROC curve and shows they agree.
Verified against sklearn on random tied data to 1e-12.
</details>

---

## Drill 8 - LRU cache

**Time box: 20 min · Simulates: OpenAI SWE-style staple · Reference: none needed - this is pure data-structure fluency**

Isolates: O(1) `get`/`put` via hashmap + doubly linked list, with sentinels so
there are zero edge cases on empty/full.

Spec: `LRUCache(capacity)`, `get(key) -> value | -1`, `put(key, value)` with
eviction of the least-recently-used on overflow. Both O(1). `get` and
`put`-update both count as "use".

What good looks like: sentinels head/tail declared immediately ("so I never
branch on None"), two private helpers (`_unlink`, `_push_front`), and the
eviction detail said aloud: delete from the map by `lru.key` - the node
remembers its key precisely so eviction can find its map entry. Mention
`OrderedDict.move_to_end` exists, then build it manually anyway (that's the
point of the question).

<details><summary>Reference solution</summary>

```python
class Node:
    __slots__ = ("key", "val", "prev", "next")
    def __init__(self, key=None, val=None):
        self.key, self.val, self.prev, self.next = key, val, None, None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}               # key -> Node
        self.head = Node()          # sentinel: most-recent side
        self.tail = Node()          # sentinel: least-recent side
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node):
        node.next, node.prev = self.head.next, self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._unlink(node)          # refresh recency
        self._push_front(node)
        return node.val

    def put(self, key, val):
        if key in self.map:
            node = self.map[key]
            node.val = val
            self._unlink(node)
            self._push_front(node)
            return
        if len(self.map) == self.cap:
            lru = self.tail.prev
            self._unlink(lru)
            del self.map[lru.key]   # node stores key for exactly this moment
        node = Node(key, val)
        self.map[key] = node
        self._push_front(node)

# --- tests ---
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)                         # evicts 2: key 1 was refreshed by get
assert c.get(2) == -1 and c.get(1) == 1 and c.get(3) == 3
c.put(1, 10)                        # update also refreshes
c.put(4, 4)                         # evicts 3
assert c.get(3) == -1 and c.get(1) == 10 and c.get(4) == 4
```

Follow-ups to expect: thread safety (one lock around both ops; or sharding),
TTL (store expiry in the node, lazily evict on get), and "why not a heap?"
(heap gives O(log n) and stale-entry headaches; the DLL is O(1) because
recency order *is* the list order).
</details>

---

## Drill 9 - Progressive in-memory KV store (3 levels)

**Time box: 45 min total (10 / 10 / 25) · Simulates: Anthropic CodeSignal-style progressive build**

Isolates: the real skill this format tests - writing level 1 so levels 2–3
drop in without a rewrite. The interviewers reveal levels one at a time; you
should *ask* "will requirements grow?" and design for it.

Spec, revealed progressively:
- **Level 1:** `set(key, value)`, `get(key)`, `delete(key)`.
- **Level 2:** `set(..., now, ttl)` - key expires at `now + ttl`; `get(key,
  now)` returns `None` for expired keys. Time is a passed-in parameter
  (testability - say this out loud), expiry is lazy.
- **Level 3:** `begin()`, `commit()`, `rollback()` with **nesting**: rollback
  undoes the innermost transaction; commit folds it into its parent.

What good looks like: level 1 stores `(value, expiry)` tuples from the start
*or* refactors calmly when level 2 lands; level 3 uses an **undo log** (record
each key's prior state on first write per transaction) rather than snapshotting
the whole dict - and you name the tradeoff: snapshot is O(size) per `begin`,
undo log is O(writes).

<details><summary>Reference solution</summary>

```python
class KVStore:
    _MISSING = object()             # sentinel: "key did not exist"

    def __init__(self):
        self.data = {}              # key -> (value, expiry_or_None)
        self.txns = []              # stack of undo dicts: key -> old entry

    # -- transaction plumbing: record prior state once per key per txn
    def _record(self, key):
        if self.txns:
            undo = self.txns[-1]
            if key not in undo:                       # first write only
                undo[key] = self.data.get(key, self._MISSING)

    # -- level 1 + 2
    def set(self, key, value, now=0, ttl=None):
        self._record(key)
        expiry = None if ttl is None else now + ttl
        self.data[key] = (value, expiry)

    def get(self, key, now=0):
        entry = self.data.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and now >= expiry:
            return None                               # lazy expiry
        return value

    def delete(self, key):
        if key in self.data:
            self._record(key)
            del self.data[key]
            return True
        return False

    # -- level 3
    def begin(self):
        self.txns.append({})

    def rollback(self):
        if not self.txns:
            return False
        for key, old in self.txns.pop().items():
            if old is self._MISSING:
                self.data.pop(key, None)
            else:
                self.data[key] = old
        return True

    def commit(self):
        if not self.txns:
            return False
        undo = self.txns.pop()
        if self.txns:               # nested: fold undo info into parent
            parent = self.txns[-1]
            for key, old in undo.items():
                if key not in parent:
                    parent[key] = old
        return True

# --- tests, per level ---
kv = KVStore()
kv.set("a", 1); assert kv.get("a") == 1
assert kv.delete("a") and not kv.delete("a")

kv.set("s", "x", now=10, ttl=5)
assert kv.get("s", now=14) == "x" and kv.get("s", now=15) is None

kv.set("k", 1)
kv.begin(); kv.set("k", 2); kv.set("new", 9)
kv.begin(); kv.set("k", 3); assert kv.get("k") == 3
kv.rollback(); assert kv.get("k") == 2            # inner txn undone
kv.commit();  assert kv.get("k") == 2 and kv.get("new") == 9
kv.begin(); kv.delete("k"); assert kv.get("k") is None
kv.rollback(); assert kv.get("k") == 2            # delete undone
```

The two subtle points, both about `commit` of a nested transaction: (1) the
parent's undo entry must keep the value from *before the parent's own first
write* - hence "only if `key not in parent`"; (2) reads go straight to
`self.data` because writes are applied immediately and the undo log is the
rollback path - this makes `get` O(1) regardless of nesting depth, unlike
layered-overlay designs where `get` walks the stack.
</details>

---

## Drill 10 - Batched inference queue simulation

**Time box: 30 min · Simulates: Anthropic systems-flavored coding - the serving-stack version of "concurrency without threads"**

Isolates: dynamic batching, the policy at the heart of every LLM serving
system - dispatch when B requests are queued *or* the oldest has waited T ms  - 
simulated deterministically with event logic, no threads, no sleep.

Spec: pure Python. `simulate(arrivals, B, max_wait, base_ms, per_item_ms)`
where a single worker serves batches; a batch costs `base_ms + per_item_ms ×
size`; the worker must be free to dispatch. Return per-request latencies
(completion − arrival). Then report mean/p95 for B=1 vs B=8 on Poisson
arrivals and explain the difference.

What good looks like: you resist threads entirely - state that the next
dispatch time is computable: `t = max(worker_free, min(oldest + max_wait,
arrival_of_Bth))`, then take everything that has arrived by `t`, up to B.
You interpret the numbers: with `base_ms` dominating, B=1 means 6 ms of
service per 2 ms inter-arrival - an overloaded queue whose latency grows
without bound; batching amortizes `base_ms` across 8 requests and the system
becomes stable. That's the actual reason GPU serving batches.

<details><summary>Reference solution</summary>

```python
import numpy as np

def simulate(arrivals, B, max_wait, base_ms, per_item_ms):
    """Single worker. A batch dispatches when the worker is free AND
    (B requests are queued OR the oldest queued has waited max_wait)."""
    arrivals = sorted(arrivals)
    n, i = len(arrivals), 0            # i = first unserved request
    free_at = 0.0
    latencies = []
    while i < n:
        oldest = arrivals[i]
        t_bth = arrivals[i + B - 1] if i + B - 1 < n else float("inf")
        trigger = min(oldest + max_wait, t_bth)   # whichever fires first
        t = max(free_at, trigger, oldest)         # not early, not busy
        size = 0                                  # take up to B arrived by t
        while size < B and i + size < n and arrivals[i + size] <= t:
            size += 1
        finish = t + base_ms + per_item_ms * size
        latencies.extend(finish - arrivals[i + j] for j in range(size))
        free_at = finish
        i += size
    return latencies

def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q / 100 * (len(xs) - 1))))]

# --- experiment: same arrivals, two policies ---
rng = np.random.default_rng(3)
arr = np.cumsum(rng.exponential(2.0, size=500)).tolist()   # ~1 req / 2 ms
for B, w in [(1, 0.0), (8, 10.0)]:
    lat = simulate(arr, B=B, max_wait=w, base_ms=5.0, per_item_ms=1.0)
    print(f"B={B}: mean {np.mean(lat):6.1f} ms   p95 {pctl(lat, 95):6.1f} ms")
# measured: B=1 mean ~951 ms, p95 ~1839 ms (unstable queue, latency ~ backlog)
# measured: B=8 mean  ~18 ms, p95   ~26 ms (stable: 13 ms service per 8 reqs)
```

The load-arithmetic that interviewers want said aloud: offered load is 0.5
req/ms. At B=1, service is 6 ms/req → capacity 0.167 req/ms < 0.5: utilization
ρ > 1, the queue grows linearly, and mean latency is dominated by backlog (the
951 ms measured is just "half the run's backlog"). At B=8, service is 13 ms
per 8 → 0.615 req/ms capacity, ρ ≈ 0.81 < 1: stable, latency = a little
queueing + ≤10 ms batching delay + 13 ms service. Batching trades a bounded
per-request wait for amortizing the fixed cost - module 6's latency notes do
this same arithmetic for GPUs, where `base_ms` is kernel-launch + weight-read
cost and `per_item_ms` shrinks until memory bandwidth saturates.
</details>

---

## Drill 11 - A training loop from scratch: AdamW, warmup + cosine, clipping

**Time box: 30 min · Simulates: OpenAI / Anthropic "write the training loop" round · Reference: module 3 (Adam), module 7 §2.9 (stability)**

Isolates: the three pieces every LLM training loop has, written by hand
instead of imported: the optimizer update, the learning-rate schedule, and
gradient clipping.

Spec: PyTorch, autograd allowed for gradients, but no `torch.optim` and no
`clip_grad_norm_`. Write `lr_at(step, max_lr, warmup, total, min_lr)`
(linear warmup, then cosine decay to `min_lr`), `clip_grad_norm(params,
max_norm)` (global norm, in place, returns the norm before clipping), and an
`AdamW` class with `step()` and `zero_grad()`. Train a small MLP and show the
loss falls.

What good looks like: you test against the library, not by eye. Match
`torch.optim.AdamW` parameter-for-parameter after 20 steps and match
`clip_grad_norm_` exactly. The classic bugs: applying weight decay through
the gradient (that is Adam with L2, not AdamW; AdamW shrinks the weights
directly), forgetting bias correction, clipping each tensor separately
instead of by the global norm, and an off-by-one that starts warmup at
learning rate 0.

<details><summary>Reference solution</summary>

```python
import math
import torch

def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    """Linear warmup to max_lr over `warmup` steps, then cosine decay to min_lr at `total`."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def clip_grad_norm(params, max_norm):
    """Global-norm clipping in place. Returns the norm before clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    scale = max_norm / (total + 1e-6)
    if scale < 1:
        for g in grads:
            g.mul_(scale)
    return total

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                 # decoupled weight decay
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

def train(model, X, y, steps=300, max_lr=1e-2, warmup=30, clip=1.0):
    opt = AdamW(model.parameters(), lr=max_lr)
    losses = []
    for step in range(steps):
        opt.lr = lr_at(step, max_lr, warmup, steps, min_lr=max_lr / 10)
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(list(model.parameters()), clip)
        opt.step()
        losses.append(loss.item())
    return losses

# --- tests ---
torch.manual_seed(0)
# 1. schedule shape
assert abs(lr_at(9, 1.0, 10, 100) - 1.0) < 1e-12            # end of warmup = max
assert abs(lr_at(100, 1.0, 10, 100, 0.1) - 0.1) < 1e-12      # end = min
assert lr_at(0, 1.0, 10, 100) < lr_at(5, 1.0, 10, 100)       # rising during warmup

# 2. AdamW matches torch.optim.AdamW
def make():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
X, y = torch.randn(64, 4), torch.randn(64, 1)
a, b = make(), make()
mine = AdamW(a.parameters(), lr=1e-2, weight_decay=0.1)
ref = torch.optim.AdamW(b.parameters(), lr=1e-2, weight_decay=0.1)
for _ in range(20):
    for net, opt in ((a, mine), (b, ref)):
        opt.zero_grad(); ((net(X) - y) ** 2).mean().backward(); opt.step()
for pa, pb in zip(a.parameters(), b.parameters()):
    assert torch.allclose(pa, pb, atol=1e-6)

# 3. clipping matches torch
a, b = make(), make()
for net in (a, b):
    (net(X * 100) ** 2).mean().backward()                    # large gradients
n1 = clip_grad_norm(list(a.parameters()), 1.0)
n2 = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
assert torch.allclose(n1, n2)
for pa, pb in zip(a.parameters(), b.parameters()):
    assert torch.allclose(pa.grad, pb.grad, atol=1e-6)

# 4. it trains
Xt = torch.randn(256, 4); yt = (Xt[:, :1] * 2 - Xt[:, 1:2]).tanh()
losses = train(make(), Xt, yt)
assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
```

Say why each piece exists: warmup because Adam's second-moment estimate is
noisy for the first steps and early updates can diverge; global clipping
because one bad batch can produce a gradient 50 times the usual size;
decoupled weight decay because with Adam, a decay term added to the gradient
gets divided by $\sqrt{v}$ and shrinks weights unevenly.
</details>

---

## Drill 12 - GAE and the PPO clipped loss

**Time box: 25 min · Simulates: Anthropic / OpenAI post-training coding round · Reference: module 9 §5, module 7 §3.2**

Isolates: the two formulas inside every PPO implementation: generalized
advantage estimation (GAE) and the clipped policy objective.

Spec: PyTorch. `gae(rewards, values, dones, last_value, gamma, lam)` returns
advantages and returns for one rollout of length T, where `dones[t] = 1`
means the episode ended after step t. `ppo_policy_loss(logp_new, logp_old,
adv, eps)` returns the loss to minimize.

What good looks like: you write the backward recursion
$A_t = \delta_t + \gamma\lambda A_{t+1}$ with
$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, and test the two limits:
$\lambda = 0$ gives the one-step TD error and $\lambda = 1$ gives the
discounted return minus $V$. Then you show the clip works: with ratio 1.5,
$\epsilon = 0.2$ and a positive advantage the gradient is zero, and with a
negative advantage it is not. The classic bugs: not cutting the bootstrap at
episode ends, using `max` instead of `min`, and forgetting the minus sign.

<details><summary>Reference solution</summary>

```python
import torch

def gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards, values, dones: (T,). dones[t] = 1 if the episode ended after step t.
    last_value: V(s_T), the value of the state after the last step.
    Returns (advantages, returns), each (T,)."""
    T = len(rewards)
    adv = torch.zeros(T)
    next_value, next_adv = last_value, 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]   # TD error
        next_adv = delta + gamma * lam * not_done * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values

def ppo_policy_loss(logp_new, logp_old, adv, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()       # minimize the negative objective

# --- tests ---
r = torch.tensor([1.0, 0.0, 2.0, 1.0]); v = torch.tensor([0.5, 0.4, 0.3, 0.2])
d = torch.zeros(4); g = 0.9

# lam = 0: advantage is the one-step TD error
adv0, _ = gae(r, v, d, last_value=0.1, gamma=g, lam=0.0)
td = r + g * torch.cat([v[1:], torch.tensor([0.1])]) - v
assert torch.allclose(adv0, td)

# lam = 1: advantage is the discounted return minus V
adv1, ret1 = gae(r, v, d, last_value=0.1, gamma=g, lam=1.0)
G = torch.zeros(4); running = 0.1
for t in reversed(range(4)):
    running = r[t] + g * running; G[t] = running
assert torch.allclose(adv1, G - v) and torch.allclose(ret1, G)

# a done flag cuts the bootstrap: step 1 ends the episode
dd = torch.tensor([0.0, 1.0, 0.0, 0.0])
adv_d, _ = gae(r, v, dd, last_value=0.1, gamma=g, lam=1.0)
assert torch.allclose(adv_d[1], r[1] - v[1])

# clipping: ratio 1.5 with eps 0.2
logp_old = torch.zeros(1)
for a, expect_zero in ((1.0, True), (-1.0, False)):
    logp_new = torch.log(torch.tensor([1.5])).requires_grad_()
    ppo_policy_loss(logp_new, logp_old, torch.tensor([a])).backward()
    assert (logp_new.grad.abs() < 1e-12).item() == expect_zero
# A > 0, ratio past 1+eps: no gradient. A < 0, same ratio: gradient still flows.
```

The `min` makes the objective pessimistic: once the ratio has moved past
$1 + \epsilon$ in the direction the advantage favors, the clipped term is the
smaller one and it is constant, so the gradient is zero. Moves in the
unfavorable direction keep their full gradient.
</details>

---

## Drill 13 - DPO loss from logits

**Time box: 20 min · Simulates: post-training coding round · Reference: module 7 §3.3**

Isolates: turning model logits into per-sequence log-probabilities with the
right shift and mask, then the DPO loss on top.

Spec: PyTorch. `sequence_logp(logits, labels, mask)` with logits `(B, T, V)`
for the whole prompt plus response, labels `(B, T)`, and mask `(B, T)` equal
to 1 on response tokens; return the summed log-probability of the response
tokens, `(B,)`. Then `dpo_loss(pi_chosen, pi_rejected, ref_chosen,
ref_rejected, beta)`.

What good looks like: you say the shift out loud (position t's logits
predict token t + 1), test against a plain loop, and test that changing
predictions of prompt or padding tokens leaves the result unchanged. Then
you check the loss is $\log 2$ when the policy equals the reference, and
that the gradient raises the chosen log-probability and lowers the rejected
one. The classic bugs: no shift (each position scored on its own token),
summing over prompt or padding tokens, and using `log(sigmoid(x))` instead of
`logsigmoid(x)`, which returns $-\infty$ for large negative $x$.

<details><summary>Reference solution</summary>

```python
import math
import torch
import torch.nn.functional as F

def sequence_logp(logits, labels, mask):
    """logits: (B, T, V) from the model on the full prompt+response.
    labels: (B, T) the same token ids. mask: (B, T), 1 on response tokens.
    Returns (B,): sum of log p(token_t | tokens before t) over response tokens."""
    logp = F.log_softmax(logits[:, :-1], dim=-1)               # position t predicts t+1
    tok = torch.gather(logp, 2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (tok * mask[:, 1:]).sum(-1)

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean()

# --- tests ---
torch.manual_seed(0)
B, T, V = 2, 6, 11
labels = torch.randint(V, (B, T))
mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0]], dtype=torch.float)

# sequence_logp against a plain loop
logits = torch.randn(B, T, V)
lp = sequence_logp(logits, labels, mask)
for b in range(B):
    s = sum(F.log_softmax(logits[b, t - 1], -1)[labels[b, t]] for t in range(1, T) if mask[b, t])
    assert torch.allclose(lp[b], s)

# changing predictions of prompt or padding tokens must not change the result
logits2 = logits.clone()
logits2[:, 0] = 10 * torch.randn(B, V)     # position 0 predicts token 1: prompt in both rows
logits2[1, 4] = 10 * torch.randn(V)        # row 1, position 4 predicts token 5: padding
assert torch.allclose(sequence_logp(logits2, labels, mask), lp)
logits3 = logits.clone(); logits3[0, 2] = 10 * torch.randn(V)   # predicts token 3: a response token
assert not torch.allclose(sequence_logp(logits3, labels, mask)[0], lp[0])

# policy == reference: margin 0, loss = log 2
x = torch.randn(4)
assert abs(dpo_loss(x, x - 1, x, x - 1).item() - math.log(2)) < 1e-6

# gradient raises the chosen log-prob and lowers the rejected one
pc = torch.zeros(1, requires_grad=True); pr = torch.zeros(1, requires_grad=True)
dpo_loss(pc, pr, torch.zeros(1), torch.zeros(1)).backward()
assert pc.grad.item() < 0 < pr.grad.item()                    # descent moves pc up, pr down
```

The loss is $-\log \sigma(\beta \cdot \text{margin})$, where the margin is
how much more the policy prefers the chosen response than the reference
does. At the start the policy equals the reference, the margin is 0, and the
loss is $\log 2 \approx 0.693$.
</details>
