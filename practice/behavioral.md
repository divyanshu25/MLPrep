# Behavioral rounds, the project deep-dive, and the values round

The `ml/` track prepares the technical rounds. This file prepares the
three rounds that are not technical and that decide offers at least as
often: the behavioral interview every company runs, the project deep-dive
that research labs weight most heavily, and the values or safety
conversation that Anthropic treats as a gate. None of them can be
improvised well. All of them reward the same thing the technical rounds
reward: specific, honest, first-hand detail.

## 1. The behavioral interview

### What is being scored

The interviewer has a rubric with four to six axes and is listening for
evidence on each. Across companies the axes are close to:

- **Impact.** Did your work change something measurable, and can you say
  what and by how much?
- **Ownership.** Did you drive it, or were you present while it happened?
- **Judgment under ambiguity.** When the problem was unclear, what did you
  decide, and why that?
- **Collaboration and conflict.** How do you disagree, and what happens
  after?
- **Learning from failure.** Can you describe something that went wrong
  without blaming others or minimizing it?
- **Growth.** What are you worse at than you would like, and what are you
  doing about it?

A story that hits one axis well beats a story that gestures at four.

### The format: STAR with numbers

Every answer has the same shape, and the interviewer expects it:

- **Situation**: two sentences of context. What was the system, the team,
  the constraint.
- **Task**: what you specifically were responsible for. Say "I", not "we",
  for the parts that were yours.
- **Action**: what you did, in enough detail that a peer could tell it was
  real. This is most of the answer.
- **Result**: what happened, with a number. Latency from 300 ms to 90 ms.
  Offline AUC from 0.81 to 0.86, and the online metric that moved with it.
  A launch date met, or missed by how much and why.

Then one more line the format does not name: what you would do differently.
It signals reflection and it preempts the follow-up.

Two to three minutes. If the interviewer wants more they will ask.

### Prepare six stories, not thirty answers

Questions vary; the underlying stories do not need to. Prepare six, each
written out in the STAR shape with real numbers, chosen so that together
they cover every axis above:

1. Your highest-impact project (impact, ownership).
2. A time the problem was unclear and you had to decide what to build
   (judgment).
3. A disagreement with a colleague or manager about a technical decision,
   how it resolved, and whether you were right (conflict, collaboration).
4. A failure that was your fault, and what changed afterward (learning).
5. A time you had to deliver under a deadline or resource constraint and
   what you cut (prioritization).
6. Something you taught, mentored, or improved for the team beyond your
   own deliverables (leadership without a title).

Most questions map onto one of these. "Tell me about a time you had to
influence without authority" is story 3 or 6. "Describe a project that
did not go as planned" is story 4 or 5.

### The question bank

Answer each aloud, timed, from your six stories. The parenthetical says
what the interviewer is probing.

- Tell me about the project you are proudest of. (impact; can you quantify)
- Tell me about a time you failed. (ownership; do you blame others)
- Describe a technical disagreement. How did it end? (conflict; can you
  lose gracefully, and did you push when you were right)
- Tell me about a time you had to make a decision with incomplete
  information. (judgment)
- Tell me about a time you received hard feedback. (growth)
- Describe a time you had to push back on a request from above. (judgment,
  courage)
- Tell me about a time you shipped something you were not fully happy
  with. (prioritization, honesty)
- What is the most complex thing you have explained to a non-expert?
  (communication)
- What would your last manager say you should work on? (self-awareness;
  give a real answer, not a strength in disguise)
- Why this company, and why now? (motivation; needs specifics about the
  team's actual work)

### Things that lose points

- Stories where nothing went wrong. Interviewers know every project has a
  mess in it, and a story without one reads as either edited or
  unobservant.
- "We" throughout. If your contribution was a part, say which part.
- Results without numbers, or numbers without a baseline.
- Blaming a teammate, a manager, or "the org" for a failure.
- A weakness that is a strength ("I care too much about quality").
- Answers longer than three minutes without checking in.

## 2. The project deep-dive

At research labs this is the round with the highest weight, and for a
research scientist track it is close to the whole decision. You present
one piece of your own work for 15 to 20 minutes, then defend it for 30 to
40 minutes against people whose job is to find the weak point. It cannot
be prepared the day before.

### Choosing the project

Pick the project you understand most deeply, not the one with the biggest
name. You will be asked why every decision was made and what would have
happened otherwise; you must be able to answer from memory. If you led a
part of a large effort, present your part and be precise about the
boundary.

### The ten-slide structure

1. **The problem**, in one slide, with why it mattered to whoever paid for
   it.
2. **Why it was hard.** What made the obvious approach fail or not apply.
3. **What existed before.** Prior work or the previous system, and its
   specific limitation.
4. **Your approach**, at the level of the key idea. One diagram.
5. **The critical decision.** The one choice that mattered most, the
   alternatives you considered, and why you chose as you did. This slide
   gets the most questions.
6. **Results**, with baselines, error bars where you have them, and the
   evaluation setup stated exactly.
7. **What did not work.** Dead ends, and what each one taught you.
8. **Limitations.** What the method cannot do, and where it would break.
9. **What you would do next**, or do differently now.
10. **Your contribution vs the team's**, stated plainly.

### The questions you will get

Prepare answers to these before the room asks them:

- Why did the baseline fail? Did you actually run it, tuned properly?
- How do you know the improvement is not noise? (Module 1, §3.8: seeds,
  confidence intervals, how many runs.)
- What is the simplest thing that would have gotten most of the gain?
- What happens if assumption X does not hold?
- Where does the compute go, and what is the cost at ten times the scale?
- What did you try that did not make it into the writeup?
- If you had one more month, what would you do?
- What part of this was yours?

The best possible outcome of the deep-dive is that the panel finds a real
weakness and you already know it, have thought about it, and say so. The
worst is being surprised by a limitation of your own work.

### Rehearsal

Give the talk to someone technical who is willing to be hostile. Time it.
Do it at least three times. Record one and watch it. The `mock_protocol.md`
scoring rubric applies; add one axis: did you defend the decisions with
evidence, or with authority?

## 3. The values and safety conversation

Anthropic runs a round specifically on the candidate's views about AI
safety and the company's mission, and treats it as a gate rather than a
tiebreaker. OpenAI and DeepMind ask related questions inside other rounds.
The material below is written for the Anthropic version; the preparation
transfers.

### What the round is actually assessing

Not agreement. The interviewer is trying to establish whether you have
thought seriously about why the work is risky and what a responsible
approach looks like, whether your view is your own, and whether you would
raise a concern rather than defer. Three failure modes are common and all
of them are easy to detect:

- **Recitation.** Repeating the company's public positions back to it.
  The interviewer wrote some of those positions and can tell.
- **Dismissal.** Treating safety concerns as marketing, or as a problem
  for some future decade. This ends the process at Anthropic.
- **Performance.** Alarm without substance: strong claims about risk that
  fall apart under a single follow-up question.

What passes is a specific, moderately held, well-informed view that
includes at least one point of disagreement or uncertainty, expressed
without hedging into mush.

### Preparation that works

- **Read the primary sources**, not summaries: Anthropic's Responsible
  Scaling Policy, the Constitutional AI paper (module 7, §3.5, covers the
  method), the core views on AI safety essay, and at least one
  interpretability or alignment paper from the last year. Know what each
  claims, and what it does not claim.
- **Form a position on three questions**, written down in a paragraph
  each: How likely is it that models at the next few scales cause serious
  harm, and through what path? What is the strongest argument against
  the company's approach (for instance, that building frontier models to
  make them safe accelerates the thing it is trying to control)? What
  would you personally want to see before a model is deployed?
- **Connect it to your own work.** Where in your past projects did safety,
  misuse, evaluation honesty, or deployment risk actually come up, and
  what did you do? Concrete beats abstract here as everywhere else.
- **Prepare one disagreement.** A place where you think the company, or
  the field, is getting something wrong, stated respectfully with a
  reason. Its presence is evidence that the rest of your views are real.

### Questions to expect

- Why do you want to work on this, specifically, rather than on
  capabilities somewhere else?
- What do you think is the most likely way advanced AI goes badly?
- Where do you disagree with our public positions?
- Is it responsible to build frontier models at all? Defend either answer.
- A colleague wants to ship a model that fails an internal evaluation,
  arguing the evaluation is too conservative. What do you do?
- How would you evaluate whether a model is honest?
- What would change your mind about the level of risk?
- Tell me about a time you raised a concern that was unpopular.

### The one rule

Say what you think. The round is designed to distinguish people who have
views from people who have prepared answers, and the only reliable way to
pass it is to be in the first group. Preparation here means reading and
thinking until you have a position, not memorizing one.
