# Mock Interview Protocol

Solo drills train correctness and speed; only observation trains performing
under stress. This kit makes a mock cheap to run - hand a partner this file and
they can interview you competently in five minutes of prep - and makes solo
mocks honest when no partner exists.

**Cadence:** 3–5 full-dress mocks in the final 2 weeks, none in the last 2 days
(the taper - see [taper.md](taper.md); a late bad mock costs confidence and
buys nothing consolidation can use in time). Full-dress means: camera on or in
person, real timer, whiteboard or bare editor with no autocomplete, no phone,
and you dressed and seated as you will be on the day.

## Blind problem draw

The interviewer draws; the candidate never sees the pool before the round.
Roll a die (or `uv run python -c "import random; print(random.choice(range(1, N+1)))"`)
over the pool for the round type:

| Round type | Pool | Draw |
|---|---|---|
| Coding (DSA) | `dsa/*/drills.md` - 8+6+6+5 timed drills with time boxes attached | Pick a folder 01–06 by die, then a drill number within it |
| Coding (speed / progressive) | `ml/08_interview_drills/drills.md` #8, #9, #10 (LRU, KV store, inference queue) | One of the three |
| ML implementation | `ml/08_interview_drills/drills.md` #1–7 | Die over 1–7 |
| ML debugging (flagship) | `ml/08_interview_drills/buggy_transformer.py` | Fixed - run at most twice ever, it doesn't survive a third exposure |
| ML breadth | `ml/08_interview_drills/rapid_fire_quiz.md` (draw 10 of 40) + 2 cards from `debugging_scenarios.md` (of 12) | random.choice per item |
| ML system design | A prompt from `ml/06_ml_systems/exercises.md`, or a fresh variant of the `design_walkthroughs.md` scenarios | Interviewer picks one the candidate hasn't done |

The interviewer reads only the problem statement aloud - never the pattern,
folder name, or time box source.

## Partner script (read verbatim where quoted)

**Opening (sets the frame, 30 seconds):**

> "Thanks for coming in. We have 45 minutes. I'll give you a problem; please
> think out loud as much as you can - I care about how you get there, not just
> the answer. You can ask me clarifying questions at any point. Ready?"

Then read the problem statement, verbatim, once. Answer clarifying questions
about the *problem*; deflect questions about the *approach* ("that's up to
you").

**Hint policy - when and how much.** Hints are for keeping the round
generative, not for rescuing it; each hint drops the Correctness score anchor
by one (note it on the scoresheet).

- **Silence tolerance: ~2 minutes.** Say nothing during productive silence
  (writing, drawing). After ~2 minutes of unproductive silence - no writing,
  no talking - prompt with the smallest possible nudge, questions not answers:
  > "Can you tell me what you're thinking?"
- If stuck on approach after that: one recognition-level cue only  - 
  > "What does the problem's structure remind you of?"
- If stuck on a bug: point at the region, never the fix  - 
  > "Walk me through this loop with a two-element input."
- Never give a second hint within 5 minutes of the first. Two unproductive
  hints ⇒ let the round fail; a failed mock graded honestly is worth more
  than a rescued one graded 3.

**Time calls (exactly two, neutral tone):**

- At 50%: > "About halfway."
- At 80%: > "About ten minutes left - if you're mid-implementation, consider
  what you want working at the end."

**Close:**

> "Time. Where were you, and what would you do with ten more minutes?"

(Score their answer to that under Communication - accurate self-location is a
signal interviewers report using.)

Then give feedback: scoresheet first, then the single highest-leverage fix,
then one thing that was genuinely strong. One fix, not five - the candidate
can only train one thing before the next mock.

## Partner scoresheet

Adapted from module 8's rubric so mock scores are comparable with solo drill
scores. Circle one per axis; a round passes at 3/3/3.

| Score | Correctness | Speed | Communication |
|---|---|---|---|
| 4 | Runs first/second try; edge cases handled; wrote a sanity test unprompted; zero hints | Done with time to spare, spare time spent testing or extending | Plan stated before code; shapes/invariants said aloud; narrated hypotheses when stuck; named tradeoffs unprompted |
| 3 | Correct after own debugging inside the box; ≤1 recognition-level hint | Done at the buzzer | Stated the plan and key invariants; narration mostly held |
| 2 | Core idea right; a bug survived the box, or needed a pointed hint | Needed +25% over the box | Narrated only when prompted |
| 1 | No running code / wrong approach held despite hints | Would need a second sitting | Silent coding |

Below the table, tally: clarifying questions asked before solving (n), hints
given (n, level), minutes of unproductive silence (est.).

## Solo fallback - recorded mock

No partner: same draw, same timer, and **record yourself** (screen + voice; a
phone propped up is fine). Solve as if narrating to an interviewer. Then - this
is the part people skip, and it's where the training lives - watch the
recording the same day and tally the five reported failure modes:

| Failure mode | Tally | What it looks like on tape |
|---|---|---|
| Coding too soon | | Typing before any stated plan |
| Half-thoughts | | "So maybe we could... actually no. Hmm. What if..." - nothing completed |
| No clarifying questions | | Solving the assumed problem within 30 seconds of reading |
| Not stating a plan | | The approach exists only in the code |
| Silent struggling | | >60s of dead air while visibly stuck |

Each nonzero tally names your drill for the week in
[communication_drills.md](communication_drills.md). Log the round in the queue
(`add mock:2026-08-20 practice/mock_protocol.md`) with pass/fail by the 3/3/3
bar, and put every failed problem on its own queue item.
