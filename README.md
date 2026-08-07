# MLPrep

Interview preparation for ML engineer / research engineer roles at frontier labs
(OpenAI, Anthropic, Google DeepMind) and big-tech ML teams - built on one principle:
**derive everything from first principles, then drill it under time pressure.**

## The three tracks

| Track | What it is |
|---|---|
| **[ml/](ml/README.md)** | 8-module ML curriculum: foundations → classical ML → neural networks → deep learning → evaluation → ML systems → the LLM stack → interview drills. Every module: first-principles notes, runnable from-scratch code, exercises with folded solutions. |
| **[dsa/](dsa/README.md)** | 7-pattern DSA track for the coding rounds: pattern derivations (why each invariant works), ~90 implemented + tested problems covering Blind 75 and the most-reported Google/OpenAI/Anthropic questions, timed drills with hint ladders in every folder. |
| **[practice/](practice/README.md)** | The retention layer over both: a spaced-repetition queue (`review.py`, expanding 1→16-day intervals), ~160 flashcards + Anki deck, unlabeled mixed problem sets, a mock-interview kit with partner script and scoresheet, communication drills, and the last-7-days taper plan. |

Curriculum contents were cross-checked against 2024–2026 interview reports from the
target companies and the major published syllabi; the drills recreate actual reported
round formats (buggy-transformer debugging, progressive builds, rapid-fire quizzes).

## Setup

```bash
make install        # uv sync - creates .venv with numpy/torch/sklearn/matplotlib
make study          # launch the LeetCode-style study app at http://localhost:8888
```

The study app ([app/](app/README.md)) is a local LeetCode-style UI over the whole
repo: solve the DSA problems in an in-browser editor judged against the real test
suites, browse the notes, run flashcards, and work the review queue - with the
tutor logic built in (interleave mode, solution-gate that schedules a cold
re-attempt, per-problem timers).

Every script runs on CPU in seconds-to-minutes, with fixed seeds and no downloads:

```bash
uv run jupyter execute ml/03_neural_networks/micrograd_plus.ipynb
uv run python dsa/05_graphs/problems.py
```

## Where to start

- New to prep: read [ml/README.md](ml/README.md) - it has the 6-week plan and the
  per-company preparation map.
- Interview soon: go straight to [ml/08_interview_drills/](ml/08_interview_drills/README.md)
  and the `drills.md` files in [dsa/](dsa/README.md) - timed, closed-book.
- Also in the repo: [src/](src/) holds earlier scratch work (ROC curve notebook,
  RNN gradient-explosion demo) that the curriculum references.
