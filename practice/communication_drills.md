# Communication Drills

Module 8 scores three axes; most prep trains only Correctness and Speed.
Communication is the axis interviewers can't average away - it's all they can
observe while you think - and it's trainable in isolation, like scales. Five
sentence templates first (the moves), then three drills (the reps).

## The five templates

Each template exists because its absence is a reported failure mode. Bad/good
pairs are transcripts of the same moment.

**1. The plan statement - before any code.**
Failure it prevents: coding too soon.

> **Bad:** *(reads problem, starts typing)* "So I'll make a dict here..."
>
> **Good:** "Three parts: build the prefix counts, then one pass querying the
> map for `pre − k`, then handle the seed entry for subarrays starting at
> zero. That's O(n) time, O(n) space. Coding now."

**2. The complete-thought contrast - "I considered X, but constraint Y rules it out."**
Failure it prevents: half-thoughts - trailing off mid-option leaves the
interviewer with fragments and you with no committed direction.

> **Bad:** "Maybe sliding window... hmm, or prefix sums... actually wait..."
>
> **Good:** "I considered a sliding window, but negatives break the
> shrink-monotonicity the window invariant needs - so prefix sums with a
> hashmap instead."

**3. The stuck-narration - know / don't know / next probe.**
Failure it prevents: silent struggling. Being stuck out loud is a skill
demonstration; being stuck silently is dead air the interviewer must interpret.

> **Bad:** *(90 seconds of silence, scrolling own code)*
>
> **Good:** "Here's what I know: the loss is flat at ln(V), so logits carry no
> signal. What I don't know: whether the embedding or the head is at fault.
> My next probe: check whether embedding gradients are nonzero."

**4. The tradeoff naming - unprompted.**
Failure it prevents: looking like you know one way rather than choosing among
ways; it's the 4-anchor behavior on the rubric.

> **Bad:** "I'll copy the dict each transaction." *(no reason given)*
>
> **Good:** "Dict copy is O(n) per transaction; an undo log is O(writes).
> I'll do the log - transactions here are small and frequent."

**5. The clarifying-question opener - before solution thinking.**
Failure it prevents: solving the assumed problem. Different round types need
different openers:

- *Coding:* "Before I start - input size and value ranges? Can the array be
  empty, can values be negative, and is the input sorted or sortable?"
- *ML design:* "Let me pin the requirements: how many users and items, what
  latency budget, and what's the actual business metric - engagement, revenue,
  or safety?"

> **Bad:** *(reads "design a fraud detector", immediately)* "So, an XGBoost
> model on transaction features..."
>
> **Good:** "Three questions first: what's the fraud base rate, how late do
> labels arrive, and is the decision block/step-up/approve or binary?"

## Drill A - re-solve for narration (recorded)

Take a problem you have **already solved and passed** - correctness is off the
table by construction, so all remaining effort is narration. Set the original
time box, record screen + voice, and solve it again as a performance. Then
watch the tape and score yourself, one point each:

1. Plan stated before the first line of code (template 1).
2. Every abandoned direction closed with a complete thought (template 2).
3. Invariant/shapes said aloud at least twice mid-implementation.
4. Every stuck moment narrated within ~20 seconds (template 3).
5. At least one tradeoff named unprompted (template 4).
6. Complexity stated before coding and re-confirmed after.

5–6: narration is round-ready for this pattern. ≤4: same drill, different
already-solved problem, tomorrow. Queue it: `add comm:drillA practice/communication_drills.md`.

## Drill B - the 60-second whiteboard summary

Explain one module's core derivation aloud in 60 seconds, timed, standing if
you'll interview standing. The constraint is the drill: 60 seconds forces the
load-bearing sentence of the derivation to the front, which is exactly what a
"can you briefly explain X?" follow-up wants. Record; a listener who doesn't
know the topic should still catch the *why*. Eight topics, one per day:

1. Backprop = chain rule + memoization (ml/03)
2. Why softmax+CE gives gradient p − y, and why MSE fails there (ml/03)
3. The √d_k scaling argument (ml/04)
4. Why the KV cache works for generation and not training (ml/04, ml/07)
5. Bias–variance decomposition, with the two vanishing cross terms (ml/05)
6. Why AUC hides the base rate; PR for rare classes (ml/05)
7. C = 6ND and the Chinchilla 20-tokens/param consequence (ml/07)
8. Two-stage retrieval-then-rank from the latency arithmetic (ml/06)

## Drill C - the clarifying-questions drill

For each vague prompt below, produce **3 clarifying questions before any
solution thinking** - write them down before unfolding. The discipline being
trained: questions come from the requirements axes (scale, latency, labels,
costs, constraints), not from stalling.

1. "Design a system to detect fraud."
2. "Build a search feature for our app."
3. "Given a stream of events, find the anomalies."
4. "Implement a rate limiter."
5. "Make our model faster."

<details><summary>Sample strong questions</summary>

1. Fraud: What's the base rate and the money distribution per fraud type?
   How late do confirmed labels arrive? Is the action space binary or
   block/step-up/review - and what does each error cost?
2. Search: Searching what corpus, at what size and update rate? Latency budget
   end-to-end? Is relevance judged by clicks (position-biased) or by human
   ratings - what's the actual success metric?
3. Anomalies: Anomalous relative to what - a point's history, its cohort, or a
   global distribution? What's the event rate and detection-latency budget?
   Do we ever get confirmed labels, or is this purely unsupervised?
4. Rate limiter: Per-user, per-IP, or per-API-key - and what are the limit
   semantics (fixed window, sliding window, token bucket with bursts)?
   Single process or distributed - must the count be exact across nodes?
   What happens on rejection - 429, queue, or degrade?
5. Faster: Faster at what - training throughput, serving latency, or serving
   cost? What's the current bottleneck - is decode bandwidth-bound or are we
   compute-bound at prefill? What quality loss is acceptable, measured how?

</details>
