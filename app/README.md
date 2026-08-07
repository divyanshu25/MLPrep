# app/ - Local LeetCode-style Study App

A single-file server + single-file SPA over the repo: the ~94 `dsa/` problems as a
judge-backed problem table, all `ml/`/`dsa/`/`practice/` notes, the ~164 flashcards,
and a dashboard over the `practice/queue.tsv` spaced-repetition queue.

Run: `make study`, or `PORT=8888 uv run python app/server.py`, then open
http://127.0.0.1:8888. Server is Python-stdlib-only; the UI pulls Tailwind,
marked, and CodeMirror from CDNs (a banner appears if you're offline).

**The judge:** your submission is parsed with `ast` and your def is swapped into the
original `problems.py` in place of the reference implementation (extra helper
defs/imports you write are inserted too). The rebuilt module runs as a temp file in
the same folder, so the file's own `__main__` assert suite - all problems, including
every assert for yours - is the test, with a 30 s timeout.

Tutor design decisions:
- **Interleave toggle** hides pattern labels and daily-shuffles the list - choosing the pattern unprompted is the skill labeled lists can't train.
- **Solution gate**: opening the reference solution takes a confirm step and auto-logs a `fail`, scheduling the cold re-attempt that makes it count.
- **Timer vs time box** per difficulty (15/25/35 min), amber at 80%, red past - hints are for after the box, not instead of it.
- **Queue integration**: every pass/fail (problems, flashcards, solution peeks) writes `practice/queue.tsv` with exactly `review.py`'s 1→2→4→8→16-day mechanics.
- **Exit checklists** in notes render as live checkboxes kept in localStorage - repo files are never modified by browsing.
