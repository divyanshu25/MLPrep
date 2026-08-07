# ML System Design - Interview Prep Notes

These notes build ML system design from first principles. The rule throughout: never
name a component until we've derived the problem it solves. If you can re-derive the
architecture from the constraints, you can answer any variant the interviewer throws
at you. If you've only memorized boxes and arrows, one follow-up question ends you.

---

## Part A - The Universal Answer Framework

An ML system design interview is not a modeling interview. The model is maybe 20% of
the answer. The other 80% is: what problem are we solving, where does the data come
from, how do we know it's working, and what breaks in production. The framework below
is ten steps. For each, we explain *why the step exists* - i.e., what failure mode it
prevents. That's the difference between reciting a checklist and understanding one.

### Step 1: Clarify the product goal

**What you do:** Ask what the business actually wants. "Design a recommendation
system" - for what? Watch time? Purchases? Diversity of consumption? New-user
retention? Who are the users, how many, what surface (homepage? notification? search
results page?), what's the current baseline (popularity list? nothing?).

**Why this step exists:** Every downstream decision - label, metric, model, latency
budget - is a function of the product goal. The classic failure it prevents:
optimizing a proxy that's easy to measure instead of the thing that matters. A team
that optimizes clicks builds a clickbait engine; a team that optimizes watch time
builds an autoplay trap. If you skip this step, you might build a technically
excellent system that solves the wrong problem, and no amount of modeling skill
recovers from that. Interviewers deliberately give vague prompts to see if you ask.

Concrete numbers to establish here: number of users (10M? 1B?), number of items
(10K products? $10^9$ videos?), traffic (100 QPS? 100K QPS?), and freshness needs
(are new items useful within minutes, like news, or days, like movies?).

### Step 2: Define the ML problem - is it even ML? what's the label?

**What you do:** Translate the product goal into a learnable task: input, output,
and - most importantly - *where the label comes from*. "Predict $P(\text{click}
\mid \text{user}, \text{item})$, labels from logged impressions" is a real ML
problem. "Recommend good videos" is not.

**Why this step exists:** Two failure modes.

1. *Using ML where a rule suffices.* If "block accounts that send 10K messages in a
   minute" solves 90% of spam, ship the rule first. ML adds training pipelines,
   drift, monitoring, and opacity - you should pay that cost only when rules run
   out. Saying this out loud is a strong senior signal.
2. *No viable label.* A model is a function fit to labeled examples. If you can't
   say precisely what a positive example is and how you'd collect a million of
   them, you don't have an ML problem yet - you have a labeling problem. Most
   real-world ML projects die here, not at the modeling stage.

Also decide the *unit of prediction* (per user-item pair? per session? per
transaction?) and the *prediction target's horizon* (click in this session?
purchase within 7 days? churn within 30 days?). The horizon determines label delay,
which will haunt you in Step 9 (monitoring).

### Step 3: Metrics - offline AND online, and how they disagree

**What you do:** Define both:

- **Offline metrics** - computed on held-out logged data before deployment: AUC,
  log loss, recall@k, NDCG, precision at fixed recall.
- **Online metrics** - computed from a live experiment: CTR, watch time,
  conversion, revenue, retention, complaint rate. Plus **guardrail metrics** that
  must not regress: latency, error rate, content-diversity, publisher fairness.

**Why this step exists:** Because offline and online metrics *disagree*, routinely,
and a candidate who doesn't know this has never shipped a model. Concrete reasons
they diverge:

- **Logged data is biased by the old policy.** Your training data contains only
  items the *previous* system chose to show ("clicked because shown"). Offline, you
  evaluate "can I re-rank what the old system retrieved"; online, your model
  changes what gets retrieved. A model can be better at the first and worse at the
  second.
- **Offline metric ≠ product metric.** AUC measures ranking of positives over
  negatives across the whole population; the product cares about the top 10 slots
  for each user. A model can improve average AUC while getting worse in the head.
- **Feedback loops.** Your model changes user behavior, which changes the data
  distribution the metric was computed on.

Interview-ready line: "I'd use offline metrics as a cheap filter - anything that
doesn't beat baseline offline doesn't ship - but the launch decision comes from an
A/B test on the product metric, with guardrails." Also state the *counter-metric*
explicitly: if optimizing CTR, watch for clickbait via bounce rate / dwell time.

### Step 4: Data - sources, labeling, sampling, feedback loops

**What you do:** Enumerate data sources (impression logs, user profiles, item
catalogs, third-party signals), how labels are produced (implicit from behavior,
explicit ratings, human annotation, delayed outcomes like chargebacks), and how
you'll sample (you cannot train on $10^{11}$ impressions/day; how do you
downsample negatives, and how do you correct the prior afterward?).

**Why this step exists:** Data problems are silent; model problems are loud. A
buggy model architecture won't converge and you'll notice. A biased sample or leaky
label converges *great* - better than the honest version - and you find out via an
online failure weeks later. Specific failures this step prevents:

- **Selection bias:** training only on shown items (see Part C).
- **Label leakage:** a feature that encodes the label (e.g., "number of fraud
  reports" as a feature for predicting fraud - reports happen *after* fraud).
- **Feedback loops:** the model trains on data its predecessor generated. A
  recommender that only shows popular items generates logs proving popular items
  get clicks. Mitigation: reserve a small slice of traffic (1%) for exploration
  (random or epsilon-greedy injection), and log *propensities* - the probability
  each item had of being shown - so you can importance-weight later.
- **Negative sampling arithmetic:** if you downsample negatives by $w$, your
  model's output odds are inflated by $w$; recalibrate with
  $p = \frac{p'}{p' + (1-p')/w}$ where $p'$ is the model's raw output. If you
  forget this, your predicted CTRs are 10x reality and anything downstream that
  consumes the probability (ad pricing, expected-value ranking) is wrong.

### Step 5: Features

**What you do:** Group features by entity: user (demographics, long-term
aggregates like 30-day category preferences), item (content embeddings, age,
historical CTR), context (time of day, device, surface), and **cross/interaction**
features (user's historical CTR *on this item's category* - usually the strongest
signals). Distinguish batch-computed features (daily aggregates) from real-time
features (clicks in the last 5 minutes).

**Why this step exists:** Features are where the two worst production bugs live:

1. **Training/serving skew** - the feature computed at training time differs from
   the one computed at serving time (different code path, different data
   freshness). Detailed in Part C; it's the most common reason a model that looked
   great offline does nothing online.
2. **Point-in-time leakage** - a training feature computed with information from
   after the prediction moment. Also Part C.

Stating "for every feature I ask: can I compute this at serving time, with the same
code, from data available *before* the prediction moment?" prevents both.

### Step 6: Model choice - start simple, justify complexity

**What you do:** Baseline first (popularity / logistic regression / gradient
boosted trees), then complexity only where the baseline demonstrably fails.
Standard ladder for tabular ranking problems: heuristic → LR with crosses → GBDT
→ neural nets (needed once you want learned embeddings of users/items/text) →
specialized architectures (two-tower for retrieval, sequence models for user
history, transformers for content understanding).

**Why this step exists:** From first principles: the value of a model is
$\Delta(\text{product metric})$ minus operational cost, and complexity raises
operational cost (latency, training cost, debugging opacity, more knobs to drift).
A simple baseline also gives you a *yardstick* - without it you can't tell whether
your fancy model's 0.82 AUC is good, because maybe LR gets 0.81. The failure
prevented: months spent on a deep model that beats nothing, or a deep model whose
2% gain costs 10x serving hardware. Interviewers explicitly listen for "I'd start
with X because..." - jumping straight to a transformer is a red flag.

### Step 7: Training pipeline

**What you do:** Describe the loop: log collection → label joining (wait for the
label horizon; a 7-day conversion label means training data is at least 7 days
stale) → feature generation (same definitions as serving - ideally the same code)
→ **temporal train/validation split** → training → offline evaluation against the
current production model → model registry.

**Why this step exists:** Two failures:

1. **Random splits on temporal data leak the future.** If Tuesday's examples are
   in train and Monday's in validation, the model has effectively seen the future
   (trends, item popularity spikes). Offline numbers inflate; online reality
   disappoints. Always split by time: train on weeks 1–3, validate on week 4.
2. **Non-reproducibility.** When the online metric tanks at 3am, you need to answer
   "what exactly is this model?" - code version, data snapshot, feature
   definitions, hyperparameters, seed. If retraining is a manual notebook, you
   can't roll back or diff. Hence pipelines-as-code and a model registry (Step 9).

### Step 8: Serving - latency budgets, batch vs online inference

**What you do:** State the latency budget and derive the architecture from it.
Decide **batch inference** (precompute predictions for all users nightly, serve
from a key-value store - works when the input space is enumerable and freshness in
hours is fine) vs **online inference** (compute at request time - required when
the input depends on the request: query text, real-time context, session
history). Many systems mix: batch-precompute candidate embeddings, online-compute
the final ranking.

**Why this step exists:** Latency is a hard product constraint, not an
afterthought - added latency measurably reduces engagement (the classic industry
findings: ~100ms of delay costs ~1% of revenue/engagement). If you design the model
first and discover it takes 400ms, you redesign everything. Deriving the
architecture from the budget (see Part F for the worked math) prevents proposing a
system that cannot physically serve traffic. It also forces the funnel insight: you
cannot run a big model on $10^9$ items per request, hence multi-stage retrieval
(Part B).

### Step 9: Monitoring - drift and degradation

**What you do:** Monitor three layers (detailed in Part D):
- **Inputs:** feature distributions, null rates, cardinalities vs a training-time
  reference.
- **Outputs:** prediction distribution (mean score, histogram) vs historical.
- **Outcomes:** online metric, and label-based metrics once delayed labels arrive.

**Why this step exists:** An ML system fails *silently*. A crashed service pages
you; a model scoring garbage because an upstream team renamed a log field just
returns numbers. Real incident shape: upstream changes country codes from "US" to
"USA", the feature hashes to an unseen bucket, model quietly degrades 5%, nobody
notices for three weeks until a quarterly business review. Input monitoring
catches it in an hour. Because labels are delayed (Step 2's horizon), input and
prediction monitoring are your *only* fast signals.

### Step 10: Iteration and the flywheel

**What you do:** Close the loop: A/B test results and error analysis feed the next
version. Plan retraining cadence (Part D), exploration traffic to keep collecting
counterfactual data, and a roadmap (v1 GBDT on batch features → v2 add real-time
features → v3 sequence model).

**Why this step exists:** Because v1 is never the end state, and the world is
non-stationary - a frozen model decays. Also an interview meta-point: presenting a
staged roadmap ("here's v1 shipping in a month, here's what I'd add") signals you've
actually shipped things, versus designing a five-year system on day one.

**Framework recap (memorize this spine):** goal → ML formulation → metrics → data
→ features → model → training → serving → monitoring → iteration. Spend your 45
minutes roughly: 5 on goal/formulation, 5 on metrics, 10 on data/features, 10 on
model+training, 10 on serving+scale math, 5 on monitoring/iteration.

---

## Part B - Deep Dives on the Classic Questions

### B1. Design a recommendation system (e.g., video/product recs)

**Setup with numbers:** $10^8$ daily active users, $10^9$ items in the corpus,
homepage shows 30 recommendations, peak 100K requests/sec, latency budget 200ms
end-to-end for the page, of which recs get ~100ms.

**The funnel math - why two (or three) stages, from first principles.** Suppose
our best model scores one (user, item) pair in 1ms of CPU (a modest neural net
with cross features). Scoring the full corpus per request: $10^9$ items ×
1ms $= 10^6$ seconds of compute *per request*. Even at 1µs per item (a big
speedup), that's $10^3$ seconds. At 100K QPS this is beyond any hardware budget.
Conclusion, forced by arithmetic: **you cannot run an expressive model on the full
corpus.** You need a cheap stage that cuts $10^9$ to $10^2$–$10^3$, then an
expensive stage that ranks those. That's the funnel:

$$10^9 \xrightarrow{\text{candidate generation}} \sim 10^3 \xrightarrow{\text{ranking}} \sim 10^2 \xrightarrow{\text{re-ranking}} 30$$

Each stage trades expressiveness for throughput: stage 1 is dumb-but-fast over
everything, stage 3 is smart-but-slow over almost nothing.

**Candidate generation - deriving the two-tower model.** We need to find, for a
user, the ~1000 most relevant items out of $10^9$, in ~10ms. What's the fastest
"relevance" function that can be evaluated against a billion items? A dot product - because dot products against a fixed set of vectors can be served by
approximate nearest neighbor (ANN) indexes in sublinear time. So we *constrain the
model's form* to force this: score must be $s(u, i) = f(u) \cdot g(i)$, where
$f$ embeds the user and $g$ embeds the item, with **no interaction between user
and item features until the final dot product**. Two separate networks - two
towers. That constraint is the whole point:

- All item embeddings $g(i)$ can be **precomputed in batch** (items don't change
  per-request) and loaded into an ANN index (HNSW graphs or quantization-based
  methods - both answer "top-k by dot product" in sub-10ms over $10^9$ vectors).
- At request time you run one forward pass to get $f(u)$ (a few ms), then one ANN
  query.

Training: this is a retrieval problem, so train with contrastive/softmax loss  - 
positives are (user, engaged-item) pairs; negatives are sampled, typically
**in-batch negatives** (other items in the same batch serve as negatives for each
user, which is computationally free) plus corrections for popularity bias (popular
items appear as negatives too often; apply $\log q(i)$ correction to the logits).
The cost of the two-tower constraint: it cannot express feature crosses like
"user's CTR on this creator" - that's exactly what the next stage is for. In
practice you run **multiple candidate sources** in parallel - two-tower ANN,
recently-popular, followed-creators' new uploads, "users like you watched" - and
union them, because each source has different blind spots (e.g., pure embedding
retrieval is bad at brand-new items with no interaction history).

**Ranking.** Now ~1000 candidates, budget ~30–50ms. Here you can afford a model
that sees user features and item features *together*: cross features, user
history attended against the candidate, GBDT or a deep net. Predict the
product-relevant events - commonly multi-task: $P(\text{click})$,
$P(\text{long watch})$, expected watch time, $P(\text{like/share})$ - combined
into a final score $s = \sum_k w_k \cdot \hat{y}_k$ where the weights $w_k$ are a
*product decision*, not learned (this is where "we value shares 10x clicks"
lives). Batch all 1000 candidates through the network in one forward pass (this is
why ranking is throughput-bound, not latency-bound per item).

**Re-ranking.** The ranker scores items independently, so the top 30 might be 30
near-identical videos. The *slate* has properties no pointwise score captures:
diversity, freshness quotas, deduplication against recent impressions, business
rules (demote borderline content), exploration injection. This stage is often
heuristics + a small model, and it's cheap ($10^2$ items).

**What to monitor here specifically:** funnel health - what fraction of finally
watched items were present in each candidate source (source recall); if one source
degrades, overall metrics dip mysteriously.

### B2. Design a feed ranking system (social feed)

Same funnel skeleton as B1, but the differences are the interview substance:

- **Candidates come mostly from the social graph**, not global retrieval: posts by
  followed accounts + shared/liked by them + some out-of-network discovery. Corpus
  per user is $10^2$–$10^4$, so candidate generation is a *fan-out* problem, not
  an ANN problem. Classic tradeoff: **fan-out-on-write** (push each new post into
  followers' precomputed feeds; cheap reads, but a celebrity with $10^8$ followers
  triggers $10^8$ writes per post) vs **fan-out-on-read** (assemble at request
  time; expensive reads). Real systems hybridize: push for normal users, pull for
  celebrities.
- **Multi-objective is unavoidable.** A feed optimized purely for clicks becomes
  ragebait. Model heads: click, like, comment, share, hide/report (a *negative*
  head with negative weight), dwell. Value function
  $s = \sum_k w_k \hat{y}_k$; expect interviewer follow-ups about picking $w_k$
  (answer: offline it's not learnable - it's a product choice tuned via A/B tests
  and constrained by guardrails like report-rate).
- **Freshness matters at minutes granularity**, so real-time features (engagement
  velocity of a post in its first 10 minutes) and frequent model/feature updates
  matter more than in movie recs.
- **Position bias is severe** (top slot gets clicked partly *because* it's on
  top). Handle in training: add position as a feature during training, set it to a
  constant (or slot 1) at serving; or weight examples by inverse propensity. If
  you ignore it, the model learns "position 1 → click" and you've laundered the
  old ranker's choices into your labels.

### B3. Design fraud detection (payments)

Three properties make fraud its own genre; a generic classifier answer misses all
of them.

**1. Extreme class imbalance.** Fraud rate ~0.1–0.5% of transactions. At $10^7$
transactions/day, ~$10^4$–$10^5$ positives/day. Consequences:

- **Accuracy is meaningless** (99.8% accuracy = predict "legit" always). Metrics:
  precision/recall at the operating threshold, PR-AUC (not ROC-AUC - ROC-AUC is
  flattered by the huge true-negative pool: with 0.2% positives, a false positive
  rate of 1% *sounds* tiny but means five false alarms per true fraud), and
  **money-weighted recall** - catching a \$10,000 fraud matters more than a \$5
  one, so weight by transaction amount.
- **Train-time handling:** downsample negatives (with the calibration correction
  from Part A Step 4) or upweight positives; the deeper fix is good features, not
  resampling tricks.

**2. Asymmetric and business-defined costs.** A false negative costs the fraud
amount + chargeback fee (~\$15–30) + fraud-ratio penalties from card networks. A
false positive costs a blocked legitimate customer - lost sale plus lifetime-value
damage (a good customer falsely declined may never return). These costs differ per
transaction, so don't bake them into the loss - train a calibrated
$P(\text{fraud})$ and choose the *action* by expected value:
block if $p \cdot \text{loss}_{\text{fraud}} > (1-p) \cdot \text{cost}_{\text{block}}$.
And the output isn't binary: three-way action space - approve / step-up
authentication (SMS, 3DS challenge) / block - where the middle option buys
information at small friction cost. Proposing the step-up tier is a strong signal.

**3. Adversarial drift.** Unlike movie tastes, fraudsters *respond to your
model*. Ship a rule blocking >\$500 electronics from new accounts, and next week
fraud arrives as \$499 gift cards from aged accounts. Consequences:

- **Retraining must be frequent** (daily-ish) and triggered by drift alarms, not
  just calendar (Part D).
- **Features should encode invariants** that are expensive for attackers to
  change - device fingerprints, graph features (this card was used across 12
  accounts; shared shipping addresses), velocity features (transactions per card
  per hour) - rather than surface attributes that cost nothing to vary.
- **Never fully trust the model's blind spots:** keep a rules layer for known
  patterns and manual-review queues; use the review team's decisions as a
  continuous labeling stream.

**Labels are delayed and censored:** chargebacks arrive 30–90 days after the
transaction, so today's training data reflects fraud tactics from two months ago - which is exactly when adversaries have already moved. Partial mitigations:
proxy early labels (customer fraud reports arrive faster), manual review labels,
and evaluating recent models on *fresh* label cohorts as they mature. Also:
transactions you *blocked* never get labels (you'll never know if they were
really fraud) - a censoring/selection problem; mitigate by letting a small random
fraction of would-be blocks through (expensive exploration, hence tiny) or via
review-team adjudication.

**Serving:** synchronous decision in the payment path, budget ~100–200ms,
real-time features mandatory (velocity counters need second-level freshness - a
card being rapidly cycled is *the* signal, and a daily-batch feature misses it
entirely).

### B4. Design search ranking

Distinguishing property: there's an explicit **query**, so relevance is
query-dependent, and intent satisfaction beats engagement.

- **Two notions of matching.** *Lexical:* classic inverted index + BM25 - for a
  query term, look up the posting list of documents containing it, score by term
  frequency/rarity. Fast, exact, 60-year-old tech, still the backbone; fails on
  vocabulary mismatch ("how to fix a flat" vs "bicycle tire repair"). *Semantic:*
  embed query and documents in the same vector space (a two-tower again - same
  ANN-forced structure as B1, with the query tower replacing the user tower),
  retrieve by similarity; handles synonymy, but can drift into "related but not
  matching". Real systems retrieve from **both** and union (hybrid retrieval).
- **Ranking features** come in three families: query-only (length, predicted
  intent), document-only (quality, PageRank-ish authority, freshness), and  - 
  carrying most of the signal - **query-document interaction**: BM25 score,
  embedding cosine, term overlap in title, and historical CTR of this (query,
  doc) pair. The heavy stage can be a **cross-encoder**: feed query and document
  *together* through a transformer so tokens attend to each other. Why it beats
  two-tower on accuracy: it can represent token-level interactions ("flat" in
  query aligning with "tire" in doc); why it can't retrieve: it must run per
  (query, doc) pair - no precomputation - so it only fits the funnel's narrow
  end (~100 docs × ~10ms/doc, batched).
- **Labels:** clicks with position-bias correction (Part C), dwell time
  ("short-click" = clicked then bounced back in 5s = *negative* signal, not
  positive), plus a smaller set of human relevance judgments (graded 0–4) used
  to compute NDCG and to anchor/calibrate the click models. NDCG from first
  principles: gain of a result discounted by $1/\log_2(\text{rank}+1)$, because
  a perfect result at rank 8 is worth far less than at rank 1 - attention
  decays roughly logarithmically.
- **Evaluation:** offline NDCG on judged sets + online interleaving experiments
  (mix results from rankers A and B in one list, see whose results get clicked  - 
  much more sensitive than A/B at same traffic) + query-segment slicing (head
  queries vs tail; navigational vs informational).

### B5. Design an LLM-powered feature (support bot / doc Q&A) - RAG from first principles

**Why retrieval at all? Derive it.** An LLM's knowledge has two hard limits:

1. **Staleness/coverage:** parameters encode the training corpus - not your
   internal wiki, not yesterday's policy change. Fine-tuning on your docs is
   slow (hours-days per update), expensive, doesn't reliably store *specific
   facts* verbatim, and can't cite sources.
2. **Context limits:** you *could* paste documents into the prompt - the model
   reads them at inference. But your corpus is, say, 100K docs × 2KB ≈ 200MB of
   text ≈ 50M tokens; context windows are $10^5$–$10^6$ tokens, and even within
   the window, cost scales linearly and attention quality degrades with length
   ("lost in the middle"). You can't paste everything.

So: paste only the *relevant* part → you need to find the relevant part → that's
retrieval. **RAG = search engine (B4) stapled to an LLM**: retrieve top-k relevant
chunks, insert into the prompt with the question, generate a grounded, citable
answer. Updating knowledge becomes an *index update* (seconds), not a training
run.

**Chunking - why and how.** Why chunk at all? (a) Embedding models compress input
into one fixed-size vector - a 50-page doc squeezed into 768 floats averages away
the specific fact you need; retrieval precision demands small, topically-coherent
units. (b) You want to paste only the relevant *passage*, not the whole doc.
Practical recipe: 200–500 token chunks, split on structural boundaries (headers,
paragraphs) rather than mid-sentence, 10–20% overlap so facts straddling a
boundary appear intact in at least one chunk, and prepend document title/section
path to each chunk (a bare paragraph saying "it defaults to 30 days" is
unretrievable without the context "Refund policy →").

**The pipeline:** ingest → chunk → embed (dense) + index (also keep BM25  - 
hybrid retrieval matters even more here because users query with exact product
names, error codes, and IDs that embeddings handle poorly) → at query time:
embed query → retrieve top ~50 → **rerank with a cross-encoder** to top ~5 (the
B4 funnel again, in miniature) → build prompt with chunks + instructions ("answer
only from the provided context; cite; say 'I don't know' if absent") → generate.

**Evaluating RAG - decompose it, because end-to-end scores don't localize
failures:**
- **Retrieval quality:** recall@k on (question, gold-passage) pairs - if the
  answer isn't in the retrieved set, generation cannot succeed. Measure this
  first; it's cheap and it's the most common failure point.
- **Faithfulness/groundedness:** is every claim in the answer supported by the
  retrieved context? Judged by humans on a sample, and at scale by an LLM judge
  (with the judge itself validated against human labels on a few hundred
  examples before you trust it).
- **Answer quality:** correctness/completeness vs a golden Q&A set; plus the
  "should have abstained" rate on questions the corpus can't answer.
- **Online:** thumbs up/down (sparse, biased), deflection rate (ticket avoided),
  escalation-to-human rate, and - careful - *unanswered-question rate*.

**Production concerns unique to this design:** latency dominated by generation
(retrieval ~50ms; generation of 300 tokens at ~20–50 tokens/sec/stream  - 
streaming the output is the UX fix, time-to-first-token is the metric); cost per
query is $10^2$–$10^3\times$ a classic model, so cache frequent questions;
prompt-injection via retrieved documents (a doc containing "ignore your
instructions" enters the prompt - sanitize and privilege-separate); index
freshness pipeline as a first-class system.

---

## Part C - Data Engineering Realities

### C1. Training/serving skew

**Definition:** any systematic difference between the features (or preprocessing)
the model sees at training time and at serving time. The model learned
$f(\text{train features})$; you're calling it on subtly different inputs, so its
learned mapping partly misfires. Symptom: great offline, mediocre online, no bug
in the model itself.

**Three concrete causes:**

1. **Two implementations of the same feature.** Training features computed in a
   Spark/SQL batch job; serving features computed in a Java service. The spec
   says "user's 7-day click count" but one counts by event time and the other by
   processing time; one lowercases strings, the other doesn't; they handle nulls
   differently (0 vs missing). Each tiny divergence is a distribution shift.
2. **Freshness mismatch.** Training joins against a snapshot where the "7-day
   count" was fully materialized; at serving, the aggregation job runs at 2am, so
   at 11pm the feature is up to 21 hours stale - the model was trained on fresher
   features than it serves on (or vice versa).
3. **Different preprocessing/statistics.** Normalization stats (mean/std),
   vocabulary/hash mappings, or imputation values computed on the training set
   but not exported with the model - serving recomputes them on different data or
   hardcodes stale ones. (This includes the classic "fit the scaler on all data"
   variant - see `feature_pipeline_demo.py` in this directory for a runnable
   demonstration.)

**How a feature store addresses it - derived, not name-dropped.** The root cause
of #1 and #2 is *duplicated feature logic and uncoordinated freshness*. The fix:
define each feature **once**, in one system that (a) runs the definition over
historical data to *generate training sets*, (b) runs the same definition
continuously to keep an online low-latency store fresh for serving, and (c) when
generating training data, performs **point-in-time joins** (below). That system
is what people call a feature store: one definition, two materializations (offline
for training, online for serving), guaranteed consistent. It's not a database;
it's a consistency mechanism.

**Cheap detector even without one:** log the actual served feature values at
prediction time, and periodically diff them against what the training pipeline
would have computed for the same entities/timestamps. Any mismatch is skew,
found before it costs you a quarter.

### C2. Point-in-time correctness

**The rule:** every training feature must be computed using only information
available **strictly before** that example's prediction timestamp. Training asks
the model to reproduce a prediction that was (hypothetically) made at time $t$;
if the features encode anything from after $t$, you're training a model with a
time machine and deploying it without one.

**Concrete leak:** predicting fraud for a transaction at 2:00pm using the
feature "card's transaction count *today*" computed over the full day (includes
2pm–midnight). Fraudulent cards get bursts of transactions *after* the fraud
starts, so the full-day count is partly an encoding of the label. Offline AUC
jumps; online, the feature (correctly computed from the past only) has far less
signal, and performance craters. The fix is the **point-in-time join** (a.k.a.
as-of join): for a label event at time $t$, join each feature at its latest value
with timestamp $< t$ (optionally $< t - \delta$ to model real pipeline delay).
This requires storing feature *history*, not just current values - you can't
as-of join against a table that only holds the latest snapshot, which is exactly
why naive "join features from the production DB" training sets are leaky by
default. Runnable demonstration: `feature_pipeline_demo.py`.

### C3. Labels from user behavior and their biases

Implicit labels (clicks, watches, purchases) are free and plentiful - and biased,
because users can only interact with what your *existing system* chose to show.

- **Position bias:** items in slot 1 get clicked more *because they're in slot
  1* (attention decays ~logarithmically with rank). Naive training learns
  "whatever the old ranker put on top is good" - you're distilling the old
  ranker, not user preference. Fixes: (a) include position as a training feature
  and freeze it at serving; (b) inverse propensity weighting - weight each
  example by $1/P(\text{examined at its position})$, with propensities estimated
  from randomization experiments (occasionally swap adjacent results and
  measure the click-rate change); (c) models that factor
  $P(\text{click}) = P(\text{examined}) \cdot P(\text{relevant})$.
- **Selection bias - the "clicked because shown" problem:** your logs contain
  labels *only for shown items*. The other $10^9 - 30$ items per request have no
  label at all - and they're missing *not at random*: the old policy chose
  against them. Train on this and (a) you can never learn that an unshown item
  is good, (b) offline evaluation only measures re-ranking of the old policy's
  choices, (c) the ecosystem ossifies (feedback loop). Mitigations: exploration
  traffic (small % of randomized slates), logging propensities for off-policy
  correction, and treating candidate-generation training specially (sampled
  softmax over the full corpus rather than learning only from shown items).
- **Interpretation bias:** a click isn't satisfaction (clickbait), a skip isn't
  disinterest (maybe seen already), no-click isn't negative (maybe never
  scrolled to it - hence "examined" modeling, and using only items *above* a
  clicked item as implicit negatives).

---

## Part D - Production Concerns

### D1. Data drift vs concept drift - precise definitions

Let the model estimate $P(y \mid x)$, and the world generate data from
$P(x, y) = P(y \mid x)\,P(x)$.

- **Data drift (covariate shift):** $P(x)$ changes; $P(y \mid x)$ stays fixed.
  The inputs move to regions the model saw rarely. *Example:* your product
  launches in Brazil - new locales, currencies, usage hours. The relationship
  "these features → this fraud probability" is unchanged, but you're now
  extrapolating where training data was thin. Retraining on recent data fixes it
  without rethinking the problem.
- **Concept drift:** $P(y \mid x)$ itself changes - the same input now means
  something different. *Examples:* fraudsters change tactics, so a feature
  pattern that meant "legit" last month now means "fraud" (adversarial concept
  drift); COVID made "flight booking sprees" flip meaning overnight; "corona"
  changed meaning as a search query. Retraining helps only once labels
  reflecting the new concept arrive - which, with delayed labels, can lag the
  drift by weeks. This is why fraud systems keep fast-twitch rule layers.

Distinguishing them in practice: data drift is visible in *input* monitoring
immediately, label-free. Concept drift may show **no input change at all** - the
first symptoms are prediction-outcome divergence (calibration decay) and online
metric drops, confirmed only when labels mature.

### D2. Monitoring - what, concretely

- **Input distributions:** per-feature summaries vs a training-set reference:
  null/default rate (a feature suddenly 40% null = broken upstream join  - 
  the single most common production incident), mean/quantiles for numerics,
  category frequencies and *unseen-category* rate for categoricals. Distance
  metrics: PSI (population stability index; alert threshold conventionally
  ~0.2) or KL/JS divergence per feature. Cheap, fast, label-free - this layer
  catches the majority of real incidents.
- **Prediction distributions:** mean predicted score, histogram of scores,
  fraction above the action threshold. If mean predicted fraud probability
  doubles overnight, either the world changed or a feature broke; either way you
  want the page *before* the labels arrive. Also monitor *per-segment* (new vs
  tenured users, per-country): aggregate stability can hide offsetting
  segment-level breakage.
- **Delayed labels / outcome metrics:** as labels mature, compute rolling AUC /
  precision-recall / calibration ($\frac{\sum \hat{p}}{\sum y}$ near 1.0) on
  recent cohorts, sliced by segment. This is the ground truth, but it's slow  - 
  it confirms what input/prediction monitoring suspected. Design dashboards by
  *label-maturity cohort* so you don't mix 90-day-mature data with 3-day data.
- **System metrics:** feature-fetch latency and timeout rate (timeouts silently
  substitute default values = an input-distribution problem wearing a systems
  costume), model latency (p50/p99), fallback rate.

### D3. Retraining cadence and triggers

Retraining exists because the world is non-stationary - model quality decays at a
rate set by *how fast $P(x,y)$ moves in your domain*. Fashion/fraud/news: days.
Movie taste, credit risk: months. Two policies, use both:

- **Scheduled:** retrain every $N$ days on a rolling window. Choose $N$
  empirically: backtest - train on data up to $T$, evaluate on $T+1, T+7,
  T+30$; the measured decay curve tells you how much AUC a week of staleness
  costs, which converts "how often should we retrain" from vibes into
  arithmetic (weigh against retraining cost + rollout risk).
- **Triggered:** retrain (or page a human) when monitoring alarms - PSI over
  threshold, calibration drift, online metric regression. Necessary for
  adversarial domains where drift doesn't wait for your schedule.

Non-negotiable guard: an automated retrain **must pass the same offline
evaluation gate + canary rollout as a manual release**. Automatically retraining
on drifted/broken data and auto-deploying is how you turn a data incident into a
model incident. If the world truly changed, the newly-trained model will pass the
gate on *fresh* eval data; if a pipeline broke, the gate is what saves you.

### D4. Safe rollout: shadow → canary → A/B

Derived from the core fact that *offline evaluation cannot prove online safety*
(Part A Step 3), so exposure to reality must be incremental:

1. **Shadow deployment:** new model runs on live traffic, predictions logged,
   **never acted on**. Verifies: latency at real QPS, feature availability at
   serving time, prediction-distribution sanity, and prediction-diff vs the
   incumbent (if the new model disagrees with prod on 40% of traffic, you want
   to know *before* users do). Zero user risk; also zero outcome signal  - 
   shadow can't tell you if the model is *better*, only that it's not broken.
   Limitation to mention: in interactive systems, shadow can't capture feedback
   effects (its predictions never change user behavior).
2. **Canary:** route a small slice (1–5%) of *real decisions* to the new model;
   watch guardrails (errors, latency, complaint/report rates, revenue) with
   automatic rollback triggers. Catches harms fast, cheaply. Not yet a
   measurement of improvement - the slice is too small/short for that.
3. **A/B test:** randomized user-level split at a size/duration computed by
   power analysis (detecting a 0.5% CTR lift at 80% power takes far more
   traffic than people intuit - often weeks; state this). Randomize by *user*,
   not by request (a user seeing both variants contaminates both, and
   request-level randomization breaks independence assumptions). Watch for
   novelty effects (new-model bump that decays; run long enough) and, in
   marketplaces, cross-arm interference (both arms compete for the same
   inventory/attention).

Roll forward gradually (5% → 20% → 50% → 100%), keeping instant rollback to the
previous registered model at every step.

### D5. Model registry and reproducibility

The incident that motivates it: 3am, fraud losses spiking; "what model is
serving, what was it trained on, and how do I get back to the last good one?"
If the answer involves someone's notebook, you have no answer. A model registry
is a versioned catalog where each entry binds: model artifact + training code
version + data snapshot/range + feature definitions version + hyperparameters +
offline eval results + lineage (which run produced it) + stage
(staging/production/archived). What it buys: **instant rollback** (repoint
serving to version $n-1$), **auditability** (finance/regulatory: "why was this
loan declined" requires knowing exactly which model decided), and
**debuggability** (diff v12 vs v13 in data and code, not in vibes). Rule of
thumb: anything not reproducible from the registry entry - retrain from the
recorded inputs and get an equivalent model - doesn't ship.

---

## Part E - Scaling Training

**Why scale at all:** dataset $10^9$ examples or model $10^9$+ parameters →
single-GPU training takes weeks or the model doesn't fit in one GPU's memory
(~80GB on a big accelerator). Two orthogonal axes, chosen by *which resource ran
out*:

**Data parallelism - when compute/time ran out but the model fits.** Copy the
full model to $N$ GPUs, split each global batch $N$ ways, each GPU computes
gradients on its shard, then **all-reduce**: average gradients across GPUs so
every replica applies the identical update and weights stay in lockstep.
Intuition: gradient of a sum = sum of gradients, so averaging shard-gradients
exactly equals the gradient of the full batch - data parallelism is
*mathematically* just big-batch SGD. Cost: an all-reduce of every gradient
(= one model-size of communication) per step; overlap communication with the
backward pass to hide it. Scaling efficiency dies when communication time
approaches compute time - which is why interconnect bandwidth is the spec that
matters.

**Model parallelism - when memory ran out.** The model itself is split.
*Pipeline parallelism:* layers 1–12 on GPU 0, 13–24 on GPU 1, etc.; micro-batches
stream through so all stages work concurrently (else all-but-one GPU idles  - 
the "bubble"; bubble fraction shrinks as micro-batches per stage grow).
*Tensor parallelism:* a single layer's weight matrix is sliced across GPUs - a
$d \times 4d$ matmul split column-wise, partial results combined; needs
communication *inside every layer*, so it's reserved for GPUs on the same
high-bandwidth node. Large-model training composes all three (data × pipeline ×
tensor), plus sharded optimizer states (ZeRO/FSDP-style: even in "data parallel"
mode, shard the optimizer state + params across replicas and gather on demand,
since Adam states are ~2 extra copies of the model).

**Why batch size interacts with learning rate - the linear scaling rule.** Go
from batch $B$ to $kB$ (as data parallelism pushes you to): the averaged
gradient's *expectation* is unchanged but its **variance drops by** $k$ - the
step direction is more trustworthy, and you take $k\times$ *fewer* steps per
epoch. To make roughly the same optimization progress per epoch, take
proportionally bigger steps: $\eta \to k\eta$. Sanity check the intuition: $k$
small-batch steps ≈ one big step of $k$-times the size when the loss surface is
locally flat enough. Caveats that show real understanding: it breaks at large
$k$ (curvature makes huge steps invalid - hence LR **warmup**, starting small
while weights are random and gradients unrepresentative); and gradient noise is
itself a regularizer, so very large batches can generalize slightly worse at
equal loss. There's a *critical batch size* beyond which more parallelism stops
buying time-to-accuracy - you're averaging noise that was already averaged.

---

## Part F - Latency Math and Back-of-Envelope Estimation

Interviewers want to see you *derive* feasibility with arithmetic. Memorize a few
constants, then multiply out loud.

**Constants worth knowing (orders of magnitude):**
- Same-datacenter network round trip: ~0.5ms. Cross-region: 50–150ms.
- In-memory KV store (Redis-class) read: ~0.3–1ms. SSD random read: ~0.1ms
  device-level, ~1–5ms through a database.
- GBDT (a few hundred trees) inference: ~0.1–1ms/example on CPU.
- Small NN (a few M params) batched on CPU: ~1–5ms for hundreds of examples;
  GPU: sub-ms compute + ~1–5ms of overhead/queueing.
- ANN query over $10^9$ vectors: ~5–10ms.
- LLM generation: time-to-first-token 100–500ms; then ~20–100 tokens/sec per
  stream depending on model size and batching.
- 1 day ≈ $86{,}400$s ≈ $10^5$s. Peak QPS ≈ 2–5× average QPS.

**Worked example - a 100ms ranking budget.** Feed ranking service; SLA: p99 ≤
100ms for scoring ~500 candidates.

| Step | Budget | Why |
|---|---|---|
| Request parsing, auth, routing | 5ms | overhead floor |
| Candidate generation (ANN + graph sources, parallel) | 15ms | precomputed indexes |
| Feature fetch: user features (1 key) + 500 items × ~50 features | 20ms | see below - the danger zone |
| Model forward: 500 candidates in one batch | 25ms | one batched pass, never 500 calls |
| Re-ranking, business logic, response | 10ms | cheap, $10^2$ items |
| **Total (p50 path)** | **75ms** | leaves ~25ms p99 headroom |

The feature fetch is where designs die. Naive: 500 items × 1 KV lookup × 0.5ms  - 
fine if parallel (~one round trip), catastrophic if sequential (250ms). So:
**batch/multi-get** the 500 keys in one request; store each item's features as
one serialized blob (1 key per item, not 50); cache hot items in-process (item
feature reads are Zipfian - a tiny cache absorbs most reads); and define
**timeout-and-default** behavior (if features miss the 20ms budget, serve with
defaults and *log it* - recall D2: silent defaults are a model-quality incident).
p99 discipline: with 500 parallel lookups you hit the store's per-key p99 tail on
nearly every request (tail at scale), another reason for one multi-get, hedged
requests, or replicas.

**Throughput/hardware estimation - the interview recipe:**

1. **Demand:** DAU × actions/user/day ÷ $10^5$s × peak factor. E.g. $10^8$ DAU ×
   10 feed loads = $10^9$ req/day ≈ $10^4$/s average → ~30K QPS peak.
2. **Per-request compute:** 30K QPS × 500 candidates = $1.5 \times 10^7$
   model scores/sec.
3. **Per-core (or per-GPU) capacity:** GBDT at ~0.2ms/score → 5K scores/s/core →
   3,000 cores ≈ ~100 32-core machines (before 30–50% headroom + redundancy).
   Same math says a 10ms/candidate cross-encoder is $10^5$ GPUs - i.e.,
   *impossible*, which is the point of doing the math: it prunes designs.
4. **Memory sanity:** embedding table $10^8$ users × 64 dims × 4B = 25.6GB  - 
   fits one server; $10^9$ items × 256 dims × 4B ≈ 1TB - must shard or
   quantize (int8 → 256GB). ANN index over $10^9$ × 64B-quantized codes ≈
   64GB - one big box.
5. **LLM case:** 7B params in fp16 = 14GB weights; KV cache per token ≈
   2 bytes/param-layer-fraction (order 0.5MB/token for 7B) - memory, not
   compute, caps concurrent streams. Generation ≈ 2 × params FLOPs/token =
   14 GFLOPs/token; a modern accelerator's usable throughput then gives
   tokens/sec; divide demand by it. (Worked fully in `exercises.md`.)

State assumptions out loud, round aggressively to powers of 10, and sanity-check
the result against something real ("~100 machines for feed ranking at $10^8$ DAU - plausible; $10^5$ GPUs - redesign").

---

## Common Interview Questions

**Q1. Your model's offline AUC improved but the online A/B shows no gain. Name
three reasons.**
A: (1) Offline eval is on logged data biased by the old policy - the gain is in
re-ranking already-shown items, which doesn't transfer once the model changes
what's shown. (2) Training/serving skew - the serving features differ from
training features, so the offline model isn't the online model. (3) AUC measures
global ranking, but the product only exposes the top-k slots - the improvement
lives in a region users never see. (Bonus: underpowered A/B test.)

**Q2. Why do recommendation systems use two stages instead of one great model?**
A: Arithmetic. An expressive model at even 1µs/item over $10^9$ items is $10^3$s
per request. So a cheap stage (dot-product retrieval via ANN, ~10ms over $10^9$)
cuts the corpus to $10^3$, and the expressive model runs only there. Each stage
trades expressiveness for throughput.

**Q3. What structural constraint defines a two-tower model, and what does it buy
and cost?**
A: The score must be $f(u) \cdot g(i)$ - user and item features never interact
before the final dot product. Buys: item embeddings precomputable and servable by
an ANN index → sublinear retrieval over the full corpus. Costs: no cross features
(e.g., user's history with this specific creator), so accuracy is ceilinged  - 
which is fine, because the ranking stage handles that.

**Q4. Data drift vs concept drift?**
A: Data drift: $P(x)$ changes, $P(y|x)$ fixed - inputs move (new market
launches); retrain on recent data. Concept drift: $P(y|x)$ changes - same input,
new meaning (fraud tactics shift); retraining helps only after labels reflecting
the new concept arrive, so you also need fast label proxies and rule layers.

**Q5. What is training/serving skew and its most common cause?**
A: Systematic difference between features at training vs serving time. Most
common cause: the feature is implemented twice - a batch pipeline for training
and a service for serving - and the implementations diverge (time semantics,
nulls, casing, freshness). Fix: single feature definition materialized to both
offline and online stores (a feature store), plus logging served features and
diffing against training-pipeline recomputation.

**Q6. What is point-in-time correctness?**
A: Every training feature must be computed only from data timestamped strictly
before that example's prediction time. Violating it (e.g., a full-day aggregate
for a 2pm prediction) leaks the label into features: offline metrics inflate,
online performance craters. Enforced by as-of joins over feature *history*.

**Q7. Why is accuracy the wrong metric for fraud, and what's right?**
A: At 0.2% fraud rate, always-predict-legit is 99.8% accurate. Use
precision/recall at the operating threshold, PR-AUC (ROC-AUC is flattered by the
enormous true-negative pool), money-weighted recall (fraud caught in dollars),
and calibration - because the block/step-up/approve decision is an
expected-value computation needing honest probabilities.

**Q8. How do you handle position bias in click labels?**
A: Clicks confound relevance with exposure - slot 1 gets clicked partly for
being slot 1. Options: (a) position as a training feature, frozen to a constant
at serving; (b) inverse-propensity weighting with propensities from small
randomization experiments; (c) factor click into examination × relevance. Doing
nothing means distilling the old ranker into the new one.

**Q9. Shadow vs canary vs A/B - what does each uniquely tell you?**
A: Shadow (predictions logged, never acted on): latency at real traffic, feature
availability, prediction-diff vs prod - safety, zero outcome signal. Canary
(1–5% of real decisions): guardrail harms - errors, complaints, latency - with
fast rollback. A/B (powered, user-randomized, weeks): the only stage that
*measures improvement* on the product metric.

**Q10. Why does increasing batch size call for increasing learning rate?**
A: $k\times$ the batch cuts gradient variance by $k$ while leaving its
expectation unchanged, and gives you $k\times$ fewer updates per epoch. To make
equivalent progress you take proportionally larger steps: $\eta \to k\eta$
(linear scaling rule), with warmup because the rule fails early in training and
at very large $k$ where curvature bounds step size.

**Q11. When would you choose batch inference over online inference?**
A: When the input space is enumerable in advance and staleness of hours is
acceptable: precompute scores for all (user, item-shortlist) pairs nightly and
serve from a KV store - trivial latency, no serving model. Online inference is
forced when the input exists only at request time (search query, session
context, real-time velocity features).

**Q12. Why does RAG exist instead of fine-tuning knowledge in, or pasting all
docs into the prompt?**
A: Fine-tuning is slow/expensive per update, unreliable for verbatim facts, and
can't cite. Pasting everything breaks on context limits, linear cost in prompt
length, and long-context attention degradation. Retrieval gives the model just
the relevant passage: knowledge updates become index updates, and answers carry
citations.

**Q13. How do you evaluate a RAG system?**
A: Decompose. Retrieval: recall@k against (question, gold passage) pairs - if
the passage isn't retrieved, nothing downstream can work. Generation:
faithfulness to retrieved context (human + validated LLM judge) and correctness
on a golden Q&A set, plus abstention rate on unanswerable questions. Online:
deflection/escalation rates and user feedback.

**Q14. What do you monitor when labels arrive 60 days late?**
A: The label-free layers: per-feature input distributions (null rates,
PSI/JS-divergence vs training reference, unseen-category rates) and prediction
distributions (mean score, histogram, above-threshold rate), sliced by segment  - 
plus fast proxy labels (user fraud reports, manual-review decisions) as an early
outcome signal, with true-label metrics computed by maturity cohort as they land.

**Q15. Your input features look stable but online performance is decaying. What
is it and what do you do?**
A: Signature of concept drift: $P(x)$ stable, $P(y|x)$ moving - likely
adversarial adaptation or a world shift. Confirm via calibration decay on the
freshest labels/proxies. Respond with faster-horizon labels, more frequent
retraining on recent windows, and (in adversarial domains) rules for the new
pattern while the labels catch up.

**Q16. How do you serve a model that must score 500 items in under 100ms?**
A: Budget it: ~15ms candidate fetch, ~20ms features, ~25ms model, ~10ms
re-rank, ~25ms headroom. Keys: one batched multi-get for all 500 items' features
(one blob per item), one batched forward pass (never 500 calls), hot-item
caching, timeout-with-logged-defaults, and hedged requests for tail latency.

**Q17. Why keep a rules engine alongside an ML fraud model?**
A: Speed of response (a rule ships in minutes; a retrain waits for labels that
lag 30–90 days), coverage of the model's blind spots against adversaries who
probe them, hard business/regulatory constraints (sanctions lists), and
interpretability for the review team. Rules are also the emergency brake when
the model degrades.

**Q18. The PM asks for "just one metric" for the new recommender. What do you
say?**
A: Push back with structure: one *primary* online metric aligned to the product
goal (e.g., long-watch rate), plus guardrails that must not regress (reports,
diversity, latency, revenue) and a counter-metric for the primary's known
failure mode (dwell/bounce against clickbait). A single unguarded metric will be
Goodharted - the system will find the degenerate optimum.

## Exit checklist

- [ ] Cold, blank editor: write the 10-step universal framework, one sentence per step - 5 min
- [ ] Sketch the two-stage recsys funnel (retrieval → ranking) with candidate counts, models, and latency budget per stage - 5 min
- [ ] Define training/serving skew and point-in-time correctness, one concrete failure example each - 4 min
- [ ] Data drift vs concept drift: precise definitions plus what you'd monitor to catch each - 3 min
- [ ] Walk shadow → canary → A/B rollout and name what each stage catches that the previous can't - 3 min
- [ ] Design RAG end to end: chunking, embedding, retrieval, grounding - and its top three failure modes - 5 min
- [ ] Do the back-of-envelope latency math for scoring 1k candidates under a 100 ms budget - 4 min

All boxes cold before moving on - a miss goes into practice/queue.tsv.
