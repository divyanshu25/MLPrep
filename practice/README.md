# The Practice System

The `ml/` and `dsa/` tracks teach you to derive. This folder makes sure you can
still do it three weeks later, out of order, with someone watching. Three
mechanisms, each earning its place from a specific fact about memory - nothing
here is ritual.

## Why spaced retrieval

A memory has two strengths. **Storage strength** - how deeply it's encoded  - 
only grows, and grows most when retrieval is *effortful*. **Retrieval strength** - how accessible it is right now - decays fast, and is what an interview tests.
The consequence is uncomfortable: reviewing something while it's still fresh
feels good and builds almost nothing; recalling it just as it's about to slip
away feels bad and builds the most. So the queue schedules each item at
expanding intervals - 1 → 2 → 4 → 8 → 16 days - each successful cold retrieval
buying a longer gap, because higher storage strength slows the decay. A miss
resets the interval to 1: failure means retrieval strength collapsed and the
item needs the short-gap ladder again, not another turn at the long end.
`review.py` implements exactly this and nothing more.

## Why interleaving after blocking

Blocked practice (a whole day of binary search) is right for first contact  - 
you need the pattern in front of you to learn its invariant. But blocking hides
the hardest interview skill: **deciding which pattern applies**, because the
folder name has already told you. Interview problems arrive unlabeled. Mixed
sets strip the labels and shuffle topics, forcing the discrimination step  - 
"contiguous + monotone predicate → window; negatives + sums → prefix hashmap"  - 
that blocked practice never trains. It feels worse and scores worse in the
session; it transfers far better. That's the signature of real learning, and
why `mixed_sets.md` exists.

## Why mocks under observation

Solving alone and solving while narrating to an evaluator are different skills;
the second degrades under stress unless stressed during practice. Mock
interviews are stress inoculation: controlled exposure to the clock, the
silence, the being-watched, so the real round is a rehearsal rather than a
first exposure. They also train the one rubric axis (Communication) that solo
drills can't score honestly. `mock_protocol.md` is the full kit;
`communication_drills.md` trains the narration muscle between mocks.

## The weekly rhythm

Runs alongside the `ml/` 6-week and `dsa/` 3-week plans - it schedules *how*
you revisit, not *what* you learn.

| Day | Work |
|---|---|
| Mon–Tue | New material, **blocked** (current ml module + current dsa folder) - first contact needs the label |
| Wed | `review.py due` queue sweep + one 10-min flashcard pass |
| Thu | One **mixed set** (from week 2), full time box - discrimination training |
| Fri | New material, blocked |
| Sat | Queue sweep + flashcards + second mixed set or one big module-8 drill |
| Sun | Mock (final 2 weeks) or rest - consolidation needs the off day |

Daily, regardless: every failed exercise, drill, or mixed-set problem goes into
the queue the same day (`review.py add`), because a miss not scheduled for
re-retrieval is a miss you'll repeat in the room.

## The files

| File | What it is |
|---|---|
| [`review.py`](review.py) + [`queue.tsv`](queue.tsv) | The spaced-retrieval queue: `due`, `add`, `pass`, `fail`, `stats`. Plain TSV - edit it by hand freely |
| [`flashcards.md`](flashcards.md) | ~150 rapid-recall cards distilled from the modules' interview-question banks + the must-be-automatic numbers; [`anki.tsv`](anki.tsv) is the same deck, Anki-importable |
| [`mixed_sets.md`](mixed_sets.md) | 6 unlabeled interleaved problem sets (4 DSA, 2 ML), answer key sealed at the bottom |
| [`mock_protocol.md`](mock_protocol.md) | Partner script, blind problem draw, printable scoresheet, solo-recording fallback |
| [`communication_drills.md`](communication_drills.md) | The 5 sentence templates + 3 narration drills for the Communication axis |
| [`taper.md`](taper.md) | The last 7 days: what to do, what to refuse to do, and why |
