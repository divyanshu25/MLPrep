# DSA Interview Prep — From First Principles

The companion to [ml/](../ml/README.md) for the coding rounds. Same rules apply:
derive, don't memorize; tiny numbers first; code is the proof. A memorized LeetCode
solution collapses the moment the interviewer changes one constraint — a pattern you
can *re-derive* doesn't. Every pattern here comes with the argument for why it's
correct and why the complexity holds, because that argument is exactly what you say
out loud in the interview.

Pure Python stdlib, no dependencies:

```bash
uv run python dsa/01_arrays_strings_hashing/problems.py   # any problems file; prints "all N problems passed"
```

## What each company's coding round actually looks like

These are different sports. Train for the one you're playing.

| Company | Coding-round fingerprint | What you're graded on |
|---|---|---|
| **OpenAI** | Practical speed-coding, often *easier* than LC-medium in raw algorithmic difficulty, but multi-part: a base problem that grows through 2–4 progressive extensions ("now support deletes", "now make it an iterator"). Real-code flavor (parsers, caches, intervals, log processing) over puzzle flavor. | **Speed and polish.** Finishing all parts with clean, tested code beats a clever half-solution. Write a quick test after each part before moving on. Sloppy variable names and untested edge cases cost you here, not big-O. |
| **Anthropic** | Progressive 4-level builds (CodeSignal style): level 1 is trivial, each level adds requirements on top of your own earlier code. Heavy on OOP and state (design a class, maintain invariants across operations), sometimes concurrency-flavored (timestamps, TTLs, ordering guarantees). | **Near-100% correctness** across all levels. Refactorability of your level-1 code determines whether level 3 is easy or a rewrite. Tests-first iteration is explicitly rewarded. LRU-cache-like "hashmap + ordering structure" designs recur. |
| **Google / DeepMind** | Classic LC medium-hard DSA: graphs, DP, binary-search-on-answer, heaps. 1–2 problems in 45 min. | **Code must run** (or be run-ready line-by-line), and you must **state complexity out loud, unprompted** — before coding (planned) and after (confirmed). Communication of the invariant is scored, not just the answer. |

Practical consequence: for OpenAI/Anthropic, drill *finishing* — set a timer, write
tests, refactor as requirements grow (module 8 of `ml/` has the progressive-build
drills). For Google/DeepMind, drill the harder patterns in folders 04–06 and narrate
every solution as if teaching it.

## The practice protocol

Designed the way a tutor would, around two facts about skill acquisition: recognition
must come before recall, and retrieval under pressure is a separate skill from
understanding.

1. **Pattern first, then grind the pattern.** Read a folder's `notes.md` and make sure
   you can state each pattern's *invariant* precisely (e.g. "everything left of `lo`
   fails the predicate, everything right of `hi` passes"). Only then do the problems.
   Grinding random problems without the invariant is how people do 300 problems and
   still fail — they're memorizing solutions, not learning the 15 ideas that generate
   them.
2. **Timed solo attempt BEFORE reading any solution.** 25 minutes for a medium,
   35–40 for a hard. No hints until the timer ends. The struggle is the training
   signal; reading a solution first produces the illusion of competence ("yes, that's
   what I would have done") that evaporates under interview pressure.
3. **Re-attempt every failure after 2 days.** Keep a plain list of problems you
   missed. Two days later, attempt cold, from the blank page. Spaced retrieval is the
   single highest-leverage habit in this whole protocol — a problem you solved once
   after peeking is not learned; a problem you re-derived after forgetting is.
4. **Talk out loud, always** — even alone. "The window invariant is: no repeated
   character inside [l, r]. Expanding r can break it; I restore it by advancing l."
   The interview is a verbal performance; if you've never narrated a monotonic stack,
   your first attempt will be in the room.
5. **State complexity before writing code, and re-confirm after.** "This is O(n)
   time, O(k) space for the window's set — let me code it." This is scored directly
   at Google/DeepMind and signals seniority everywhere. It also catches wrong plans
   early: if your plan is O(n²) and n is 10⁵, stop and rethink before typing.

## Pattern → folder map

| Folder | Patterns | Signature problems |
|---|---|---|
| [01_arrays_strings_hashing](01_arrays_strings_hashing/notes.md) | Hashing, two pointers, sliding window, prefix sums, string frequency tricks, in-place partitioning | Two sum, longest substring w/o repeats, subarray sum = k, trapping rain water, minimum window substring |
| [02_binary_search](02_binary_search/notes.md) | Loop-invariant binary search, binary search *on the answer* (monotonic predicate), rotated arrays, real-valued search | First/last position, rotated array search, Koko eating bananas, median of two sorted arrays |
| [03_stacks_queues_linked_lists](03_stacks_queues_linked_lists/notes.md) | Matching/nesting with stacks, monotonic stack, deques, pointer surgery, fast/slow pointers, LRU cache design | Daily temperatures, largest rectangle in histogram, reverse list, cycle detection, LRU cache |
| [04_trees_and_heaps](04_trees_and_heaps/notes.md) | Traversals, BST invariant, recursion on trees, heaps and k-way problems, tries | Validate BST, LCA, serialize/deserialize, merge k lists, implement trie, word search II |
| [05_graphs](05_graphs/notes.md) | BFS/DFS, topological sort, union-find, shortest paths, grids as graphs | Number of islands, course schedule, word ladder, Dijkstra |
| [06_dp_greedy_backtracking](06_dp_greedy_backtracking/notes.md) | 1D/2D DP from recursion + memo, greedy exchange arguments, backtracking with pruning | Coin change, LIS, edit distance, word break, N-queens, word search |
| [07_bits_math_matrix](07_bits_math_matrix/notes.md) | Bit manipulation from two's complement up, fast exponentiation, Floyd on implicit sequences, in-place matrix simulation | Single number, counting bits, pow(x,n), rotate image, spiral matrix, set matrix zeroes |

Folders 01 and 03 also carry the interval pattern (insert interval, meeting rooms
I/II) and the design/iterator problems (LRU, RandomizedSet, time-based KV store,
resumable iterators) that OpenAI/Anthropic practical rounds favor.

## A 3-week schedule (alongside ML prep)

Assumes you're running the `ml/` 6-week ML plan concurrently — DSA gets ~1–1.5
hours/day, treated as a daily workout rather than a separate phase. Adjust weights by
target company: Google/DeepMind loops justify doubling this; OpenAI/Anthropic loops
should tilt time toward timed, tested, multi-part finishing practice.

**Week 1 — Foundations of iteration (folders 01–03).**
Day 1–2: folder 01 notes + problems, then the drills timed. Day 3: folder 02 (binary
search is small but the #1 source of off-by-one deaths — do the bugs section
carefully). Day 4–5: folder 03, including implementing the LRU cache cold twice.
Day 6: re-attempt this week's failures (the 2-day rule). Day 7: rest or one mock.

**Week 2 — Structure and search (folders 04–05).**
Trees and heaps days 1–2, graphs days 3–5 (graphs are the highest-frequency
Google/DeepMind topic — give them the full three days). Day 6: failure re-attempts.
Interleave: redo two folder-01 problems cold mid-week so week 1 doesn't decay.

## The practice system

The protocol's habits — the 2-day failure list, blind draws, invariant note cards —
have tooling in [practice/](../practice/README.md): `practice/review.py` is the
failure list with expanding intervals (1→2→4→8→16 days, reset on a miss),
`practice/mixed_sets.md` serves problems with the folder labels stripped (the
pattern-recognition step these folders can't train, because their names give it
away), and `practice/mock_protocol.md` runs the week-3 mock rounds with a partner
script and scoresheet. Queue every miss the day it happens.

**Week 3 — DP + the long tail + full-loop simulation (folders 06–07 + mocks).**
Days 1–3: DP/greedy/backtracking — derive each DP from the brute-force recursion,
never start from the table. Day 4: folder 07 (bits, math, matrix simulation) — one
sitting covers it; the bit toolbox and the matrix index maps are memorization-light
once derived. Days 5–6: full mock rounds under real conditions: 45-minute
timer, talk out loud, code must run, one problem from each of folders 04–07 drawn
blind. Day 7: final failure sweep — anything missed twice goes on a note card with
its invariant, and you re-derive it the morning of the interview, not the solution.
