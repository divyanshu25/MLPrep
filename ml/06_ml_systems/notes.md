# ML System Design: Interview Prep Notes

These notes build ML system design from first principles. The rule throughout: never
name a component until we've derived the problem it solves. If you can re-derive the
architecture from the constraints, you can answer any variant the interviewer asks.
If you've only memorized boxes and arrows, a single follow-up question exposes the gap.

---

## Part A: The Universal Answer Framework

An ML system design interview is not a modeling interview. The model is maybe 20% of
the answer. The other 80% is: what problem are we solving, where does the data come
from, how do we know it's working, and what breaks in production. The framework below
is ten steps. For each, we explain *why the step exists*, i.e., what failure mode it
prevents. That's the difference between reciting a checklist and understanding one.

### Step 1: Clarify the product goal

**What you do:** Ask what the business actually wants. "Design a recommendation
system": for what? Watch time? Purchases? Diversity of consumption? New-user
retention? Who are the users, how many, what surface (homepage? notification? search
results page?), what's the current baseline (popularity list? nothing?).

**Why this step exists:** Every downstream decision (label, metric, model, latency
budget) is a function of the product goal. The classic failure it prevents:
optimizing a proxy that's easy to measure instead of the thing that matters. A team
that optimizes clicks ends up ranking clickbait first; a team that optimizes watch
time ends up favoring autoplay chains that hold attention without satisfying users. If you skip this step, you might build a technically
excellent system that solves the wrong problem, and no amount of modeling skill
recovers from that. Interviewers deliberately give vague prompts to see if you ask.

Concrete numbers to establish here: number of users (10M? 1B?), number of items
(10K products? $10^9$ videos?), traffic (100 QPS? 100K QPS?), and freshness needs
(are new items useful within minutes, like news, or days, like movies?).

### Step 2: Define the ML problem: is it even ML? What's the label?

**What you do:** Translate the product goal into a learnable task: input, output,
and, most importantly, *where the label comes from*. "Predict $P(\text{click}
\mid \text{user}, \text{item})$, labels from logged impressions" is a real ML
problem. "Recommend good videos" is not.

**Why this step exists:** Two failure modes.

1. *Using ML where a rule suffices.* If "block accounts that send 10K messages in a
   minute" solves 90% of spam, ship the rule first. ML adds training pipelines,
   drift, monitoring, and opacity; you should pay that cost only when rules run
   out. Saying this out loud is a strong senior signal.
2. *No viable label.* A model is a function fit to labeled examples. If you can't
   say precisely what a positive example is and how you'd collect a million of
   them, you don't have an ML problem yet; you have a labeling problem. Most
   real-world ML projects stall at this step, not at the modeling stage.

Also decide the *unit of prediction* (per user-item pair? per session? per
transaction?) and the *prediction target's horizon* (click in this session?
purchase within 7 days? churn within 30 days?). The horizon determines label delay
(how long after a prediction its true label arrives), which limits how fast
label-based monitoring can detect a problem.

### Step 3: Metrics (offline AND online) and how they disagree

**What you do:** Define both:

- **Offline metrics**, computed on held-out logged data before deployment: AUC
  (area under the ROC curve), log loss, recall@k (fraction of relevant items in
  the top k), NDCG (a rank-discounted relevance score), precision at fixed recall.
- **Online metrics**, computed from a live experiment: CTR (click-through rate), watch time,
  conversion, revenue, retention, complaint rate. Plus **guardrail metrics** that
  must not regress: latency, error rate, content-diversity, publisher fairness.

**Why this step exists:** Because offline and online metrics *disagree*, routinely,
and not knowing this signals that a candidate has never shipped a model. Concrete reasons
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

Interview-ready line: "I'd use offline metrics as a cheap filter (anything that
doesn't beat baseline offline doesn't ship), but the launch decision comes from an
A/B test on the product metric, with guardrails." Also state the *counter-metric*
explicitly: if optimizing CTR, watch for clickbait via bounce rate / dwell time.

### Step 4: Data (sources, labeling, sampling, feedback loops)

**What you do:** Enumerate data sources (impression logs, user profiles, item
catalogs, third-party signals), how labels are produced (implicit from behavior,
explicit ratings, human annotation, delayed outcomes like chargebacks), and how
you'll sample (you cannot train on $10^{11}$ impressions/day; how do you
downsample negatives, and how do you correct the prior afterward?).

**Why this step exists:** Data problems raise no errors; model problems usually do.
A buggy model architecture won't converge and you'll notice. A biased sample or
leaky label converges well, often to better offline numbers than the honest
version, and you find out via an online failure weeks later. Specific failures this step prevents:

- **Selection bias:** training only on shown items, which leaves every unshown
  item without a label.
- **Label leakage:** a feature that encodes the label (e.g., "number of fraud
  reports" as a feature for predicting fraud; reports happen *after* fraud).
- **Feedback loops:** the model trains on data its predecessor generated. A
  recommender that only shows popular items generates logs proving popular items
  get clicks. Mitigation: reserve a small slice of traffic (1%) for exploration
  (random or epsilon-greedy injection), and log *propensities* (the probability
  each item had of being shown) so you can importance-weight later.
- **Negative sampling arithmetic:** if you downsample negatives by $w$, your
  model's output odds are inflated by $w$; recalibrate with
  $p = \frac{p'}{p' + (1-p')/w}$ where $p'$ is the model's raw output. If you
  forget this with $w = 10$, your predicted CTRs are about 10x reality (odds scale
  by $w$, and for small $p$ odds ≈ probability), and anything downstream that
  consumes the probability (ad pricing, expected-value ranking) is wrong.

### Step 5: Features

**What you do:** Group features by entity: user (demographics, long-term
aggregates like 30-day category preferences), item (content embeddings, age,
historical CTR), context (time of day, device, surface), and **cross/interaction**
features (user's historical CTR *on this item's category*, usually the strongest
signals). Distinguish batch-computed features (daily aggregates) from real-time
features (clicks in the last 5 minutes).

**Why this step exists:** Features are where the two worst production bugs live:

1. **Training/serving skew**: the feature computed at training time differs from
   the one computed at serving time (different code path, different data
   freshness). It's the most common reason a model that looked
   great offline does nothing online.
2. **Point-in-time leakage**: a training feature computed with information from
   after the prediction moment.

Stating "for every feature I ask: can I compute this at serving time, with the same
code, from data available *before* the prediction moment?" prevents both.

### Step 6: Model choice (start simple, justify complexity)

**What you do:** Baseline first (popularity / logistic regression (LR) / gradient
boosted decision trees (GBDT)), then complexity only where the baseline demonstrably fails.
Standard ladder for tabular ranking problems: heuristic → LR with crosses → GBDT
→ neural nets (needed once you want learned embeddings of users/items/text) →
specialized architectures (two-tower for retrieval, meaning separate user and item
networks compared by a dot product; sequence models for user history; transformers
for content understanding).

**Why this step exists:** From first principles: the value of a model is
$\Delta(\text{product metric})$ minus operational cost, and complexity raises
operational cost (latency, training cost, debugging opacity, more knobs to drift).
A simple baseline also gives you a *yardstick*; without it you can't tell whether
your fancy model's 0.82 AUC is good, because maybe LR gets 0.81. The failure
prevented: months spent on a deep model that beats nothing, or a deep model whose
2% gain costs 10x serving hardware. Interviewers explicitly listen for "I'd start
with X because..."; jumping straight to a transformer is a red flag.

### Step 7: Training pipeline

**What you do:** Describe the loop: log collection → label joining (wait for the
label horizon; a 7-day conversion label means training data is at least 7 days
stale) → feature generation (same definitions as serving, ideally the same code)
→ **temporal train/validation split** → training → offline evaluation against the
current production model → model registry.

**Why this step exists:** Two failures:

1. **Random splits on temporal data leak the future.** If Tuesday's examples are
   in train and Monday's in validation, the model has effectively seen the future
   (trends, item popularity spikes). Offline numbers come out higher than what
   the model achieves online. Always split by time: train on weeks 1–3, validate on week 4.
2. **Non-reproducibility.** When the online metric drops sharply at 3am, you need to answer
   "what exactly is this model?": code version, data snapshot, feature
   definitions, hyperparameters, seed. If retraining is a manual notebook, you
   can't roll back or diff. Hence pipelines-as-code and a model registry (a versioned
   catalog recording each model's code, data, and settings).

### Step 8: Serving (latency budgets, batch vs online inference)

**What you do:** State the latency budget and derive the architecture from it.
Decide **batch inference** (precompute predictions for all users nightly, serve
from a key-value store; works when the input space is enumerable and freshness in
hours is fine) vs **online inference** (compute at request time; required when
the input depends on the request: query text, real-time context, session
history). Many systems mix: batch-precompute candidate embeddings, online-compute
the final ranking.

**Why this step exists:** Latency is a hard product constraint, not an
afterthought: added latency measurably reduces engagement (widely cited industry
measurements put ~100ms of added delay at ~1% of revenue or engagement). If you design the model
first and discover it takes 400ms, you redesign everything. Deriving the
architecture from the budget prevents proposing a system that cannot physically
serve traffic. It also forces the funnel insight: you
cannot run a big model on $10^9$ items per request, hence multi-stage retrieval
(a cheap stage narrows the corpus to a few thousand candidates, then an expensive
model ranks only those).

### Step 9: Monitoring (drift and degradation)

**What you do:** Monitor three layers:
- **Inputs:** feature distributions, null rates, cardinalities vs a training-time
  reference.
- **Outputs:** prediction distribution (mean score, histogram) vs historical.
- **Outcomes:** online metric, and label-based metrics once delayed labels arrive.

**Why this step exists:** An ML system fails *silently*. A crashed service pages
you; a model producing wrong scores because an upstream team renamed a log field
still returns numbers, with no error. A typical incident: upstream changes country
codes from "US" to "USA", the feature hashes to an unseen bucket, model accuracy
drops 5% with no error raised, nobody notices for three weeks until a quarterly business review. Input monitoring
catches it in an hour. Because labels are delayed (Step 2's horizon), input and
prediction monitoring are your *only* fast signals.

### Step 10: Iteration and the flywheel

**What you do:** Close the loop: A/B test results and error analysis feed the next
version. Plan retraining cadence, exploration traffic to keep collecting
counterfactual data (labels for items the current model would not have shown), and a roadmap (v1 GBDT on batch features → v2 add real-time
features → v3 sequence model).

**Why this step exists:** Because v1 is never the end state, and the world is
non-stationary: a model that is never retrained loses accuracy as the data
distribution changes. Also an interview meta-point: presenting a
staged roadmap ("here's v1 shipping in a month, here's what I'd add") signals you've
actually shipped things, versus designing a five-year system on day one.

**Framework recap (memorize this sequence):** goal → ML formulation → metrics → data
→ features → model → training → serving → monitoring → iteration. Spend your 45
minutes roughly: 5 on goal/formulation, 5 on metrics, 10 on data/features, 10 on
model+training, 10 on serving+scale math, 5 on monitoring/iteration.

---

## Part B: Deep Dives on the Classic Questions

### B1. Design a recommendation system (e.g., video/product recs)

**Setup with numbers:** $10^8$ daily active users, $10^9$ items in the corpus,
homepage shows 30 recommendations, peak 100K requests/sec, latency budget 200ms
end-to-end for the page, of which recs get ~100ms.

**The funnel math: why two (or three) stages, from first principles.** Suppose
our best model scores one (user, item) pair in 1ms of CPU (a modest neural net
with cross features). Scoring the full corpus per request: $10^9$ items ×
1ms $= 10^6$ seconds of compute *per request*. Even at 1µs per item (a big
speedup), that's $10^3$ seconds. At 100K QPS this is $10^3 \times 10^5 = 10^8$ cores busy
continuously, beyond any hardware budget.
Conclusion, forced by arithmetic: **you cannot run an expressive model on the full
corpus.** You need a cheap stage that cuts $10^9$ to $10^2$–$10^3$, then an
expensive stage that ranks those. That's the funnel:

$$10^9 \xrightarrow{\text{candidate generation}} \sim 10^3 \xrightarrow{\text{ranking}} \sim 10^2 \xrightarrow{\text{re-ranking}} 30$$

Each stage trades expressiveness for throughput: stage 1 runs a cheap model over
every item, stage 3 runs an expensive model over about a hundred items.

**Candidate generation: deriving the two-tower model.** We need to find, for a
user, the ~1000 most relevant items out of $10^9$, in ~10ms. What's the fastest
"relevance" function that can be evaluated against a billion items? A dot product, because dot products against a fixed set of vectors can be served by
approximate nearest neighbor (ANN) indexes in sublinear time. So we *constrain the
model's form* to force this: score must be $s(u, i) = f(u) \cdot g(i)$, where
$f$ embeds the user and $g$ embeds the item, with **no interaction between user
and item features until the final dot product**. Two separate networks, hence
"two towers." That constraint is the whole point:

- All item embeddings $g(i)$ can be **precomputed in batch** (items don't change
  per-request) and loaded into an ANN index (HNSW, a layered nearest-neighbor graph, or
  quantization-based methods; both answer "top-k by dot product" in sub-10ms over $10^9$ vectors).
- At request time you run one forward pass to get $f(u)$ (a few ms), then one ANN
  query.

Training: this is a retrieval problem, so train with contrastive/softmax loss:
positives are (user, engaged-item) pairs; negatives are sampled, typically
**in-batch negatives** (other items in the same batch serve as negatives for each
user, which is computationally free) plus corrections for popularity bias (popular
items appear as negatives too often; subtract $\log q(i)$, where $q(i)$ is item $i$'s sampling
frequency, from its logit).
The cost of the two-tower constraint: it cannot express feature crosses like
"user's CTR on this creator"; the next stage (ranking) handles those. In
practice you run **multiple candidate sources** in parallel (two-tower ANN,
recently-popular, followed-creators' new uploads, "users like you watched") and
union them, because each source has different blind spots (e.g., pure embedding
retrieval is bad at brand-new items with no interaction history).

**Ranking.** Now ~1000 candidates, budget ~30–50ms. Here you can afford a model
that sees user features and item features *together*: cross features, user
history attended against the candidate, GBDT or a deep net. Predict the
product-relevant events, commonly with a multi-task model: $P(\text{click})$,
$P(\text{long watch})$, expected watch time, $P(\text{like/share})$, combined
into a final score $s = \sum_k w_k \cdot \hat{y}_k$ where the weights $w_k$ are a
*product decision*, not learned (this is where "we value shares 10x clicks"
lives). Batch all 1000 candidates through the network in one forward pass (this is
why ranking is throughput-bound, not latency-bound per item).

**Re-ranking.** The ranker scores items independently, so the top 30 might be 30
near-identical videos. The *slate* has properties no pointwise score captures:
diversity, freshness quotas, deduplication against recent impressions, business
rules (demote borderline content), exploration injection. This stage is often
heuristics + a small model, and it's cheap ($10^2$ items).

**What to monitor here specifically:** funnel health, meaning what fraction of finally
watched items were present in each candidate source (source recall); if one source
degrades, overall metrics drop with no obvious cause.

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
- **Multi-objective is unavoidable.** A feed optimized purely for clicks tends to
  rank outrage-provoking posts highest. Model heads: click, like, comment, share, hide/report (a *negative*
  head with negative weight), dwell. Value function
  $s = \sum_k w_k \hat{y}_k$; expect interviewer follow-ups about picking $w_k$
  (answer: offline it's not learnable; it's a product choice tuned via A/B tests
  and constrained by guardrails like report-rate).
- **Freshness matters at minutes granularity**, so real-time features (engagement
  velocity of a post in its first 10 minutes) and frequent model/feature updates
  matter more than in movie recs.
- **Position bias is severe** (top slot gets clicked partly *because* it's on
  top). Handle in training: add position as a feature during training, set it to a
  constant (or slot 1) at serving; or weight examples by inverse propensity. If
  you ignore it, the model learns "position 1 → click" and the old ranker's
  placement decisions end up encoded in your labels as relevance.

### B3. Design fraud detection (payments)

Three properties make fraud its own genre; a generic classifier answer misses all
of them.

**1. Extreme class imbalance.** Fraud rate ~0.1–0.5% of transactions. At $10^7$
transactions/day, ~$10^4$–$5\times10^4$ positives/day. Consequences:

- **Accuracy is meaningless** (99.8% accuracy = predict "legit" always). Metrics:
  precision/recall at the operating threshold, PR-AUC (not ROC-AUC, which stays high
  because of the huge true-negative pool: with 0.2% positives, a false positive
  rate of 1% *sounds* tiny but per 1,000 transactions means 998 × 1% ≈ 10 false
  alarms against 2 true frauds, five per fraud), and
  **money-weighted recall**: catching a \$10,000 fraud matters more than a \$5
  one, so weight by transaction amount.
- **Train-time handling:** downsample negatives (with the calibration correction
  from Part A Step 4) or upweight positives; the deeper fix is good features, not
  resampling tricks.

**2. Asymmetric and business-defined costs.** A false negative costs the fraud
amount + chargeback fee (~\$15–30) + fraud-ratio penalties from card networks. A
false positive costs a blocked legitimate customer: lost sale plus lifetime-value
damage (a good customer falsely declined may never return). These costs differ per
transaction, so don't bake them into the loss; train a calibrated
$P(\text{fraud})$ and choose the *action* by expected value:
block if $p \cdot \text{loss}_{\text{fraud}} > (1-p) \cdot \text{cost}_{\text{block}}$.
And the output isn't binary: three-way action space: approve / step-up
authentication (SMS, or a 3DS challenge, the card issuer's 3-D Secure check) /
block, where the middle option gains information at a small friction cost. Proposing the step-up tier is a strong signal.

**3. Adversarial drift.** Unlike movie tastes, fraudsters *respond to your
model*. Ship a rule blocking >\$500 electronics from new accounts, and next week
fraud arrives as \$499 gift cards from aged accounts. Consequences:

- **Retraining must be frequent** (roughly daily) and triggered by drift alarms,
  not just the calendar.
- **Features should encode invariants** that are expensive for attackers to
  change, such as device fingerprints, graph features (this card was used across
  12 accounts; shared shipping addresses), and velocity features (transactions
  per card per hour), rather than surface attributes that cost nothing to vary.
- **Cover the model's blind spots:** keep a rules layer for known
  patterns and manual-review queues; use the review team's decisions as a
  continuous labeling stream.

**Labels are delayed and censored:** chargebacks arrive 30–90 days after the
transaction, so today's training data reflects fraud tactics from two months ago, and adversaries have
typically changed tactics since then. Partial mitigations:
proxy early labels (customer fraud reports arrive faster), manual review labels,
and evaluating recent models on *fresh* label cohorts as they mature. Also:
transactions you *blocked* never get labels (you'll never know if they were
really fraud), a censoring/selection problem; mitigate by letting a small random
fraction of would-be blocks through (expensive exploration, hence tiny) or via
review-team adjudication.

**Serving:** synchronous decision in the payment path, budget ~100–200ms,
real-time features mandatory (velocity counters need second-level freshness; a
card used many times in quick succession is *the* signal, and a daily-batch feature misses it
entirely).

### B4. Design search ranking

Distinguishing property: there's an explicit **query**, so relevance is
query-dependent, and intent satisfaction beats engagement.

- **Two notions of matching.** *Lexical:* classic inverted index + BM25 scoring: for a
  query term, look up the posting list of documents containing it, score by term
  frequency/rarity. Fast, exact, decades-old, still the base of most search systems; fails on
  vocabulary mismatch ("how to fix a flat" vs "bicycle tire repair"). *Semantic:*
  embed query and documents in the same vector space (a two-tower again, the same
  structure as B1 chosen so ANN retrieval works, with the query tower replacing
  the user tower), retrieve by similarity; handles synonymy, but can return
  related-but-not-matching results. Real systems retrieve from **both** and union (hybrid retrieval).
- **Ranking features** come in three families: query-only (length, predicted
  intent), document-only (quality, PageRank-ish authority, freshness), and,
  carrying most of the signal, **query-document interaction**: BM25 score,
  embedding cosine, term overlap in title, and historical CTR of this (query,
  doc) pair. The heavy stage can be a **cross-encoder**: feed query and document
  *together* through a transformer so tokens attend to each other. Why it beats
  two-tower on accuracy: it can represent token-level interactions ("flat" in
  query aligning with "tire" in doc); why it can't retrieve: it must run per
  (query, doc) pair with no precomputation, so it only fits the funnel's narrow
  end (~100 docs × ~10ms/doc, batched).
- **Labels:** clicks with position-bias correction (higher slots get clicked more regardless
  of relevance), dwell time
  ("short-click" = clicked then bounced back in 5s = *negative* signal, not
  positive), plus a smaller set of human relevance judgments (graded 0–4) used
  to compute NDCG and to anchor/calibrate the click models. NDCG from first
  principles: gain of a result discounted by $1/\log_2(\text{rank}+1)$, because
  a perfect result at rank 8 is worth far less than at rank 1: user
  attention falls off roughly logarithmically with rank.
- **Evaluation:** offline NDCG on judged sets + online interleaving experiments
  (mix results from rankers A and B in one list, see whose results get clicked;
  much more sensitive than A/B at same traffic) + query-segment slicing (head
  queries vs tail; navigational vs informational).

### B5. Design an LLM-powered feature (support bot / doc Q&A): retrieval-augmented generation (RAG) from first principles

**Why retrieval at all? Derive it.** An LLM's knowledge has two hard limits:

1. **Staleness/coverage:** parameters encode the training corpus, not your
   internal wiki, not yesterday's policy change. Fine-tuning on your docs is
   slow (hours-days per update), expensive, doesn't reliably store *specific
   facts* verbatim, and can't cite sources.
2. **Context limits:** you *could* paste documents into the prompt; the model
   reads them at inference. But your corpus is, say, 100K docs × 2KB ≈ 200MB of
   text ≈ 50M tokens (at ~4 bytes per token); context windows are $10^5$–$10^6$ tokens, and even within
   the window, cost scales linearly and attention quality degrades with length
   ("lost in the middle"). You can't paste everything.

So: paste only the *relevant* part → you need to find the relevant part → that's
retrieval. **RAG = a search engine (B4) whose results go into an LLM's prompt**: retrieve top-k relevant
chunks, insert into the prompt with the question, generate a grounded, citable
answer. Updating knowledge becomes an *index update* (seconds), not a training
run.

**Chunking: why and how.** Why chunk at all? (a) Embedding models compress input
into one fixed-size vector; a 50-page doc compressed into 768 floats averages out
the specific fact you need; retrieval precision demands small, topically-coherent
units. (b) You want to paste only the relevant *passage*, not the whole doc.
Practical recipe: 200–500 token chunks, split on structural boundaries (headers,
paragraphs) rather than mid-sentence, 10–20% overlap so facts straddling a
boundary appear intact in at least one chunk, and prepend document title/section
path to each chunk (a bare paragraph saying "it defaults to 30 days" is
unretrievable without the context "Refund policy →").

**The pipeline:** ingest → chunk → embed (dense) + index (also keep a BM25 index;
hybrid retrieval matters even more here because users query with exact product
names, error codes, and IDs that embeddings handle poorly) → at query time:
embed query → retrieve top ~50 → **rerank with a cross-encoder** to top ~5 (the
B4 funnel again, in miniature) → build prompt with chunks + instructions ("answer
only from the provided context; cite; say 'I don't know' if absent") → generate.

**Evaluating RAG: decompose it, because end-to-end scores don't localize
failures.**
- **Retrieval quality:** recall@k on (question, gold-passage) pairs; if the
  answer isn't in the retrieved set, generation cannot succeed. Measure this
  first; it's cheap and it's the most common failure point.
- **Faithfulness/groundedness:** is every claim in the answer supported by the
  retrieved context? Judged by humans on a sample, and at scale by an LLM judge
  (with the judge itself validated against human labels on a few hundred
  examples before you trust it).
- **Answer quality:** correctness/completeness vs a golden Q&A set; plus the
  "should have abstained" rate on questions the corpus can't answer.
- **Online:** thumbs up/down (sparse, biased), deflection rate (ticket avoided),
  escalation-to-human rate, and *unanswered-question rate* (interpret with care: a low rate can
  mean the bot answers questions it should decline).

**Production concerns unique to this design:** latency dominated by generation
(retrieval ~50ms; generation of 300 tokens at ~20–50 tokens/sec/stream takes 300/50 to 300/20 =
6–15 s, so streaming the output is the UX fix and time-to-first-token is the
metric); cost per
query is $10^2$–$10^3\times$ a classic model, so cache frequent questions;
prompt-injection via retrieved documents (a doc containing "ignore your
instructions" enters the prompt; sanitize retrieved text and keep it separate
from the instructions); index
freshness pipeline as a first-class system.

---

## Part C: Data Engineering Realities

### C1. Training/serving skew

**Definition:** any systematic difference between the features (or preprocessing)
the model sees at training time and at serving time. The model learned
$f(\text{train features})$; you're calling it on subtly different inputs, so its
outputs are partly wrong. Symptom: great offline, mediocre online, no bug
in the model itself.

**Three concrete causes:**

1. **Two implementations of the same feature.** Training features computed in a
   Spark/SQL batch job; serving features computed in a Java service. The spec
   says "user's 7-day click count" but one counts by event time and the other by
   processing time; one lowercases strings, the other doesn't; they handle nulls
   differently (0 vs missing). Each tiny divergence is a distribution shift.
2. **Freshness mismatch.** Training joins against a snapshot where the "7-day
   count" was fully materialized; at serving, the aggregation job runs at 2am, so
   at 11pm the feature is up to 21 hours stale (2am to 11pm), so the model was trained on fresher
   features than it serves on (or vice versa).
3. **Different preprocessing/statistics.** Normalization stats (mean/std),
   vocabulary/hash mappings, or imputation values computed on the training set
   but not exported with the model; serving recomputes them on different data or
   hardcodes stale ones. (This includes the classic "fit the scaler on all data"
   variant; Demo 1 of `feature_pipeline_demo.ipynb` in this directory runs it
   and shows the inflated offline AUC.)

**How a feature store addresses it (derived, not name-dropped).** The root cause
of #1 and #2 is *duplicated feature logic and uncoordinated freshness*. The fix:
define each feature **once**, in one system that (a) runs the definition over
historical data to *generate training sets*, (b) runs the same definition
continuously to keep an online low-latency store fresh for serving, and (c) when
generating training data, performs **point-in-time joins** (each example gets only feature values
timestamped before its prediction time). That system
is what people call a feature store: one definition, two materializations (offline
for training, online for serving), guaranteed consistent. It's not a database;
it's a consistency mechanism.

**Cheap detector even without one:** log the actual served feature values at
prediction time, and periodically diff them against what the training pipeline
would have computed for the same entities/timestamps. Any mismatch is skew,
found before it affects months of production traffic.

### C2. Point-in-time correctness

**The rule:** every training feature must be computed using only information
available **strictly before** that example's prediction timestamp. Training asks
the model to reproduce a prediction that was (hypothetically) made at time $t$;
if the features encode anything from after $t$, the model learns to rely on
information that will not exist at serving time.

**Concrete leak:** predicting fraud for a transaction at 2:00pm using the
feature "card's transaction count *today*" computed over the full day (includes
2pm–midnight). Fraudulent cards get bursts of transactions *after* the fraud
starts, so the full-day count is partly an encoding of the label. Offline AUC
jumps; online, the feature can only be computed from the past, carries less
signal, and the model scores well below its offline estimate (in the notebook
demo, 0.99 offline vs 0.94 served). The fix is the **point-in-time join** (a.k.a.
as-of join): for a label event at time $t$, join each feature at its latest value
with timestamp $< t$ (optionally $< t - \delta$ to model real pipeline delay).
This requires storing feature *history*, not just current values; you can't
as-of join against a table that only holds the latest snapshot, which is exactly
why naive "join features from the production DB" training sets are leaky by
default. Demo 2 of `feature_pipeline_demo.ipynb` runs this leak on simulated card
data and compares it with the past-only count.

### C3. Labels from user behavior and their biases

Implicit labels (clicks, watches, purchases) are free and plentiful, and biased,
because users can only interact with what your *existing system* chose to show.

- **Position bias:** items in slot 1 get clicked more *because they're in slot
  1* (attention decays ~logarithmically with rank). Naive training learns
  "whatever the old ranker put on top is good", so the model reproduces the old
  ranker, not user preference. Fixes: (a) include position as a training feature
  and freeze it at serving; (b) inverse propensity weighting (IPW): weight each
  example by $1/P(\text{examined at its position})$, with propensities estimated
  from randomization experiments (occasionally swap adjacent results and
  measure the click-rate change); (c) models that factor
  $P(\text{click}) = P(\text{examined}) \cdot P(\text{relevant})$.
- **Selection bias (the "clicked because shown" problem):** your logs contain
  labels *only for shown items*. The other $10^9 - 30$ items per request have no
  label at all, and they're missing *not at random*: the old policy chose not
  to show them. Train on this and (a) you can never learn that an unshown item
  is good, (b) offline evaluation only measures re-ranking of the old policy's
  choices, (c) the set of recommended items narrows over time (feedback loop). Mitigations: exploration
  traffic (small % of randomized slates), logging propensities for off-policy
  correction, and treating candidate-generation training specially (sampled
  softmax over the full corpus rather than learning only from shown items).
- **Interpretation bias:** a click isn't satisfaction (clickbait), a skip isn't
  disinterest (maybe seen already), no-click isn't negative (maybe never
  scrolled to it; hence "examined" modeling, and using only items *above* a
  clicked item as implicit negatives).

---

## Part D: Production Concerns

### D1. Data drift vs concept drift: precise definitions

Let the model estimate $P(y \mid x)$, and the world generate data from
$P(x, y) = P(y \mid x)\,P(x)$.

- **Data drift (covariate shift):** $P(x)$ changes; $P(y \mid x)$ stays fixed.
  The inputs move to regions the model saw rarely. *Example:* your product
  launches in Brazil: new locales, currencies, usage hours. The relationship
  "these features → this fraud probability" is unchanged, but you're now
  extrapolating where training data was thin. Retraining on recent data fixes it
  without rethinking the problem.
- **Concept drift:** $P(y \mid x)$ itself changes: the same input now means
  something different. *Examples:* fraudsters change tactics, so a feature
  pattern that meant "legit" last month now means "fraud" (adversarial concept
  drift); COVID made "flight booking sprees" flip meaning overnight; "corona"
  changed meaning as a search query. Retraining helps only once labels
  reflecting the new concept arrive, which with delayed labels can be weeks
  after the drift. This is why fraud systems keep rule layers that analysts can
  update within hours.

Distinguishing them in practice: data drift is visible in *input* monitoring
immediately, label-free. Concept drift may show **no input change at all**; the
first symptoms are prediction-outcome divergence (calibration decay) and online
metric drops, confirmed only when labels mature.

### D2. Monitoring: what, concretely

- **Input distributions:** per-feature summaries vs a training-set reference:
  null/default rate (a feature suddenly 40% null usually means a broken upstream
  join, one of the most common production incidents), mean/quantiles for numerics,
  category frequencies and *unseen-category* rate for categoricals. Distance
  metrics: PSI (population stability index; alert threshold conventionally
  ~0.2) or KL/JS (Kullback-Leibler / Jensen-Shannon) divergence per feature. Cheap,
  fast, label-free; this layer catches most real incidents.
- **Prediction distributions:** mean predicted score, histogram of scores,
  fraction above the action threshold. If mean predicted fraud probability
  doubles overnight, either the world changed or a feature broke; either way you
  want the page *before* the labels arrive. Also monitor *per-segment* (new vs
  tenured users, per-country): aggregate stability can hide offsetting
  segment-level breakage.
- **Delayed labels / outcome metrics:** as labels mature, compute rolling AUC /
  precision-recall / calibration ($\frac{\sum \hat{p}}{\sum y}$ near 1.0) on
  recent cohorts, sliced by segment. This is the ground truth, but it's slow:
  it confirms what input/prediction monitoring suspected. Design dashboards by
  *label-maturity cohort* so you don't mix 90-day-mature data with 3-day data.
- **System metrics:** feature-fetch latency and timeout rate (timeouts silently
  substitute default values, an input-distribution problem that first shows up
  in systems metrics), model latency (p50/p99), fallback rate.

### D3. Retraining cadence and triggers

Retraining exists because the world is non-stationary: model quality decays at a
rate set by *how fast $P(x,y)$ moves in your domain*. Fashion/fraud/news: days.
Movie taste, credit risk: months. Two policies, use both:

- **Scheduled:** retrain every $N$ days on a rolling window. Choose $N$
  empirically: backtest (train on data up to $T$, evaluate on $T+1, T+7,
  T+30$); the measured decay curve tells you how much AUC a week of staleness
  costs, which converts "how often should we retrain" from a guess into
  arithmetic (weigh against retraining cost + rollout risk).
- **Triggered:** retrain (or page a human) when monitoring alarms (PSI over
  threshold, calibration drift, online metric regression). Necessary for
  adversarial domains where drift can happen between scheduled retrains.

Non-negotiable guard: an automated retrain **must pass the same offline
evaluation gate + canary rollout as a manual release**. Automatically retraining
on drifted/broken data and auto-deploying is how you turn a data incident into a
model incident. If the world truly changed, the newly-trained model will pass the
gate on *fresh* eval data; if a pipeline broke, the gate blocks the bad model.

### D4. Safe rollout: shadow → canary → A/B

Derived from the core fact that *offline evaluation cannot prove online safety*
(Part A Step 3), so exposure to reality must be incremental:

1. **Shadow deployment:** new model runs on live traffic, predictions logged,
   **never acted on**. Verifies: latency at real QPS, feature availability at
   serving time, prediction-distribution sanity, and prediction-diff vs the
   incumbent (if the new model disagrees with prod on 40% of traffic, you want
   to know *before* users do). Zero user risk, and also zero outcome signal:
   shadow can't tell you if the model is *better*, only that it's not broken.
   Limitation to mention: in interactive systems, shadow can't capture feedback
   effects (its predictions never change user behavior).
2. **Canary:** route a small slice (1–5%) of *real decisions* to the new model;
   watch guardrails (errors, latency, complaint/report rates, revenue) with
   automatic rollback triggers. Catches harms fast, cheaply. Not yet a
   measurement of improvement; the slice is too small and short for that.
3. **A/B test:** randomized user-level split at a size/duration computed by
   power analysis (detecting a 0.5% CTR lift at 80% power takes far more
   traffic than people intuit, often weeks; state this). Randomize by *user*,
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
(staging/production/archived). What it provides: **instant rollback** (repoint
serving to version $n-1$), **auditability** (finance/regulatory: "why was this
loan declined" requires knowing exactly which model decided), and
**debuggability** (diff v12 vs v13 in data and code, not by guessing). Rule of
thumb: anything not reproducible from the registry entry (retrain from the
recorded inputs and get an equivalent model) doesn't ship.

---

## Part E: Scaling Training

**Why scale at all:** dataset $10^9$ examples or model $10^9$+ parameters →
single-GPU training takes weeks or the model doesn't fit in one GPU's memory
(~80GB on a big accelerator). Two orthogonal axes, chosen by *which resource ran
out*:

**Data parallelism: when compute/time ran out but the model fits.** Copy the
full model to $N$ GPUs, split each global batch $N$ ways, each GPU computes
gradients on its shard, then **all-reduce**: average gradients across GPUs so
every replica applies the identical update and weights stay in lockstep.
Intuition: gradient of a sum = sum of gradients, so averaging shard-gradients
exactly equals the gradient of the full batch, so data parallelism is
*mathematically* just big-batch SGD. Cost: an all-reduce of every gradient per
step (with a ring all-reduce each GPU sends and receives about 2× the gradient
size); overlap communication with the
backward pass to hide it. Scaling efficiency drops when communication time
approaches compute time, which is why interconnect bandwidth is the spec that
matters.

**Model parallelism: when memory ran out.** The model itself is split.
*Pipeline parallelism:* layers 1–12 on GPU 0, 13–24 on GPU 1, etc.; micro-batches
stream through so all stages work concurrently (otherwise all but one GPU sit idle,
the "bubble"; bubble fraction shrinks as micro-batches per stage grow).
*Tensor parallelism:* a single layer's weight matrix is sliced across GPUs, e.g. a
$d \times 4d$ matmul split column-wise, partial results combined; needs
communication *inside every layer*, so it's reserved for GPUs on the same
high-bandwidth node. Large-model training composes all three (data × pipeline ×
tensor), plus sharded optimizer states (ZeRO/FSDP-style: even in "data parallel"
mode, shard the optimizer state + params across replicas and gather on demand,
since Adam states are ~2 extra copies of the model).

**Why batch size interacts with learning rate: the linear scaling rule.** Go
from batch $B$ to $kB$ (as data parallelism pushes you to): the averaged
gradient's *expectation* is unchanged but its **variance drops by** $k$: the
step direction is more reliable, and you take $k\times$ *fewer* steps per
epoch. To make roughly the same optimization progress per epoch, take
proportionally bigger steps: $\eta \to k\eta$. Sanity check the intuition: $k$
small-batch steps ≈ one big step of $k$-times the size when the loss surface is
locally flat enough. Caveats that show real understanding: it breaks at large
$k$ (curvature makes huge steps invalid; hence LR **warmup**, which starts small
while weights are random and gradients unrepresentative); and gradient noise is
itself a regularizer, so very large batches can generalize slightly worse at
equal loss. There's a *critical batch size* beyond which more parallelism stops
reducing time-to-accuracy: the gradient estimate is already accurate, so more
averaging changes little.

---

## Part F: Latency Math and Back-of-Envelope Estimation

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

**Worked example: a 100ms ranking budget.** Feed ranking service; SLA: p99 ≤
100ms for scoring ~500 candidates.

| Step | Budget | Why |
|---|---|---|
| Request parsing, auth, routing | 5ms | overhead floor |
| Candidate generation (ANN + graph sources, parallel) | 15ms | precomputed indexes |
| Feature fetch: user features (1 key) + 500 items × ~50 features | 20ms | see below; the riskiest step |
| Model forward: 500 candidates in one batch | 25ms | one batched pass, never 500 calls |
| Re-ranking, business logic, response | 10ms | cheap, $10^2$ items |
| **Total (p50 path)** | **75ms** | leaves ~25ms p99 headroom |

The feature fetch is the step most likely to exceed its budget. Naive: 500 items ×
1 KV lookup × 0.5ms is fine if parallel (~one round trip), catastrophic if
sequential (500 × 0.5ms = 250ms). So:
**batch/multi-get** the 500 keys in one request; store each item's features as
one serialized blob (1 key per item, not 50); cache hot items in-process (item
feature reads are Zipfian: a few popular items get most requests, so a small
cache serves most reads); and define
**timeout-and-default** behavior (if features miss the 20ms budget, serve with
defaults and *log it*; recall D2: silent defaults are a model-quality incident).
p99 discipline: with 500 parallel lookups you hit the store's per-key p99 tail on
nearly every request ($1 - 0.99^{500} \approx 99\%$; tail at scale), another
reason for one multi-get, hedged requests, or replicas.

**Throughput/hardware estimation (the interview recipe):**

1. **Demand:** DAU × actions/user/day ÷ $10^5$s × peak factor. E.g. $10^8$ DAU ×
   10 feed loads = $10^9$ req/day ≈ $10^4$/s average → × 3 peak factor ≈ 30K
   QPS peak.
2. **Per-request compute:** 30K QPS × 500 candidates = $1.5 \times 10^7$
   model scores/sec.
3. **Per-core (or per-GPU) capacity:** GBDT at ~0.2ms/score → 5K scores/s/core →
   3,000 cores ≈ ~100 32-core machines (before 30–50% headroom + redundancy).
   Same math says a 10ms/candidate cross-encoder needs
   $1.5\times10^7 \times 0.01$s $= 1.5\times10^5$ GPU-seconds per second, i.e.
   ~$10^5$ GPUs, *impossible*, which is the point of doing the math: it rules out
   designs.
4. **Memory sanity:** embedding table $10^8$ users × 64 dims × 4B = 25.6GB,
   fits one server; $10^9$ items × 256 dims × 4B ≈ 1TB, so it must shard or
   quantize (int8 → 256GB). ANN index over $10^9$ × 64B-quantized codes ≈
   64GB, one big box.
5. **LLM case:** 7B params in fp16 = 14GB weights; KV cache per token =
   2 (K and V) × layers × width × 2 bytes = 2 × 32 × 4096 × 2 ≈ 0.5MB for a 7B
   (32 layers, width 4096), so memory, not compute, caps concurrent streams: an
   80GB GPU has 66GB free after weights, and 66GB ÷ 0.5MB ≈ 130K tokens of
   context in total. Generation ≈ 2 × params FLOPs/token = 14 GFLOPs/token, but
   at moderate batch each decode step reads all 14GB of weights, so memory
   bandwidth ÷ 14GB gives steps/sec; multiply by batch size for tokens/sec and
   divide demand by it. (Worked fully in `exercises.md`.)

State assumptions out loud, round aggressively to powers of 10, and sanity-check
the result against something real ("~100 machines for feed ranking at $10^8$ DAU: plausible; $10^5$ GPUs: redesign").

---

## Common Interview Questions

**Q1. Your model's offline AUC improved but the online A/B shows no gain. Name
three reasons.**
A: Say AUC rises from 0.80 to 0.82 on last month's logs and the A/B test is
flat. (1) The logs contain only items the old model chose to show, so the
offline gain measures a better ordering of those items; once the new model
changes what is shown, that comparison no longer applies (selection bias, C3).
(2) Training/serving skew: the features computed at serving differ from the
training features, so the model in the A/B test is not the model that was
evaluated offline. (3) AUC is the probability that a random positive scores
above a random negative, over all pairs, while users see only the top 30
slots; the gain can sit in ranks 200 to 1000, which nobody views. A fourth
reason is an underpowered test: detecting a 0.5% lift can take weeks of
traffic (D4).

**If asked:** *How do you check for skew?* Log the feature values served at
prediction time and diff them against the training pipeline's values for the
same users and timestamps; any mismatch is skew.

**Q2. Why do recommendation systems use two stages instead of one great model?**
A: Take $10^9$ items and an expressive model at 1 µs per item: one request
costs $10^9 \times 10^{-6}\,\text{s} = 10^3$ s of compute, and at 100K requests
per second that is $10^8$ CPU-seconds every second. So the first stage restricts
the score to a dot product $f(u) \cdot g(i)$, which an approximate nearest
neighbor (ANN) index answers in about 10 ms over $10^9$ precomputed item
vectors, and keeps about $10^3$ candidates. The expressive model then scores
only those, $10^3 \times 1\,\mu\text{s} = 1$ ms. The general rule: each stage
trades expressiveness for throughput, and the expensive stage sees one
millionth of the corpus.

**If asked:** *Why several candidate sources?* Each misses different items
(embedding retrieval misses new items with no interactions), so systems union
two-tower ANN, recently popular, and followed-creator sources.

**Q3. What structural constraint defines a two-tower model, and what does it buy
and cost?**
A: The user tower outputs a vector $f(u)$, the item tower outputs $g(i)$, say
64 numbers each, and the score is $f(u) \cdot g(i)$: user and item features
never interact before that final dot product. Buys: $g(i)$ does not depend on
the user, so all $10^9$ item vectors are computed offline and loaded into an
ANN index, and a request costs one user-tower forward pass (a few ms) plus one
ANN query (under 10 ms). Costs: a cross feature such as "this user's click rate
on this creator" depends on both sides, so neither tower can compute it, and
accuracy is capped. The ranking stage adds those cross features back on the
$10^3$ retrieved candidates.

**If asked:** *How is it trained?* Softmax over (user, engaged item) pairs with
the other items in the batch as negatives, plus a $\log q(i)$ correction to the
logits, where $q(i)$ is how often item $i$ appears in batches, because popular
items appear as negatives too often.

**Q4. Data drift vs concept drift?**
A: Write $P(x, y) = P(y \mid x)\,P(x)$. A fraud model launches in Brazil and
20% of transactions are now in a new currency at new hours: $P(x)$ moved, but
the same features still imply the same fraud probability. That is data drift;
input monitoring shows it immediately, and retraining on recent data fixes it.
Concept drift is $P(y \mid x)$ changing: fraudsters change tactics, so a
feature pattern that meant "legit" last month now means "fraud". Inputs can
look unchanged, and retraining helps only after labels for the new pattern
arrive (chargebacks take 30 to 90 days), so fraud systems also use fast proxy
labels and a rules layer.

**If asked:** *How do you detect each?* Data drift: per-feature PSI against the
training distribution, no labels needed. Concept drift: calibration
$\sum \hat{p} / \sum y$ moving away from 1 on the freshest labeled cohort.

**Q5. What is training/serving skew and its most common cause?**
A: The spec says "user's 7-day click count". The training job in Spark counts
by event time and stores missing users as 0; the serving service in Java counts
by processing time and stores missing users as null. That is training/serving
skew: a systematic difference between the features the model saw in training
and the features it receives at serving. The most common cause is this one,
the feature implemented twice in two systems that diverge. The fix is to define
each feature once and have one system compute it both for training sets and for
the online store (a feature store, C1), and to log served values and diff them
against the training pipeline.

**If asked:** *Other causes?* Freshness mismatch (a 2am aggregation job leaves
the feature 21 hours stale at 11pm) and normalization statistics or
vocabularies that were not exported with the model.

**Q6. What is point-in-time correctness?**
A: Predict fraud for a 2:00pm transaction using "card's transaction count
today" computed over the full day. Fraudulent cards get bursts of transactions
after the fraud starts, so the 2pm to midnight part of that count encodes the
label: offline AUC jumps, and online, where the count covers only the hours
before 2pm, the feature carries much less signal. The rule: every training
feature uses only data timestamped strictly before the example's prediction
time $t$. It is enforced with an as-of join, which takes each feature's latest
value with timestamp $< t$; that requires storing feature history, not only
current values.

**If asked:** *Why use $< t - \delta$ instead?* Serving receives features with
a pipeline delay $\delta$ (for example a counter updated every 5 minutes), so
training should see only what serving would have had.

**Q7. Why is accuracy the wrong metric for fraud, and what's right?**
A: At a 0.2% fraud rate, predicting "legit" for every transaction is 99.8%
accurate and catches no fraud. Use precision and recall at the operating
threshold, and PR-AUC rather than ROC-AUC: per 100,000 transactions there are
200 frauds and 99,800 legit ones, so a 1% false positive rate, which looks small
on a ROC curve, is 998 false alarms, about 5 per fraud. Add money-weighted
recall, since catching a \$10,000 fraud matters more than a \$5 one. Finally
require calibrated probabilities, because the action is an expected-value
comparison: block when
$p \cdot \text{loss}_{\text{fraud}} > (1-p) \cdot \text{cost}_{\text{block}}$.

**If asked:** *Is the output binary?* No: a middle action, step-up
authentication (SMS code or 3DS challenge), gathers more information at a small
friction cost for mid-range $p$.

**Q8. How do you handle position bias in click labels?**
A: Suppose the same item gets a 10% click rate in slot 1 and 4% in slot 5.
Clicks measure relevance times the chance of being seen, so training on raw
clicks teaches the new model to reproduce the old ranker's order. Three fixes.
(a) Use position as a training feature and set it to a constant at serving.
(b) Inverse propensity weighting: weight each click by
$1/P(\text{examined at its position})$, estimated from small randomization
experiments that swap adjacent results; with the numbers above, slot 5 is
examined 0.4 times as often as slot 1, so a click there gets weight 2.5
relative to slot 1. (c) Model
$P(\text{click}) = P(\text{examined}) \cdot P(\text{relevant})$ and rank by
$P(\text{relevant})$.

**If asked:** *Why a constant position at serving?* Position is what serving
decides, so it is unknown; a constant gives every candidate the same position
term, and the ranking depends only on the other features.

**Q9. Shadow vs canary vs A/B: what does each uniquely tell you?**
A: Take a new fraud model. In shadow it scores all live traffic and its
predictions are logged but never acted on: this measures latency at real QPS,
whether serving-time features exist, and how often it disagrees with the
production model (40% disagreement needs explaining before launch). It has zero
user risk and zero outcome signal. In canary it makes 1 to 5% of real decisions
while guardrails (errors, latency, complaints, revenue) trigger automatic
rollback; this detects harm, but the slice is too small and short to measure
improvement. The A/B test, randomized by user and sized by power analysis
(often weeks), is the only stage that measures whether the product metric
improves.

**If asked:** *What can shadow miss in a recommender?* Feedback effects: its
recommendations are never shown, so it cannot reveal how users respond to them.

**Q10. Why does increasing batch size call for increasing learning rate?**
A: Go from batch 256 to 1024, so $k = 4$. The averaged gradient has the same
expectation, its variance drops by 4, and an epoch now has a quarter as many
steps. Four small steps of size $\eta$ along nearly the same gradient move the
weights about as far as one step of size $4\eta$, so to make similar progress per epoch
set $\eta \to k\eta$ (the linear scaling rule). This assumes the gradient
changes little over those $k$ steps. It fails early in training, when weights
are random and curvature is high (hence learning-rate warmup), and at very
large $k$, where curvature caps the stable step size and past the critical
batch size extra examples stop reducing time to accuracy.

**If asked:** *Does it hold for Adam?* Adam divides each step by a running RMS
of the gradient, so the linear rule does not carry over; the common heuristic is
square-root scaling, $\eta \to \sqrt{k}\,\eta$.

**Q11. When would you choose batch inference over online inference?**
A: A daily email recommends 10 videos to each of $10^8$ users. Scoring a
1,000-item shortlist per user overnight is $10^{11}$ scores; at 5,000 GBDT
scores per second per core that is $2 \times 10^7$ core-seconds, about 700
cores for an 8-hour window. The top 10 per user go into a key-value store and
are served with one read of about 1 ms, with no model in the request path. The
rule: choose batch when the inputs can be enumerated in advance and hours of
staleness are acceptable. Online inference is required when the input exists
only at request time: a search query, the current session, real-time velocity
features for fraud.

**If asked:** *Main cost of batch?* Compute spent on users who do not visit
that day, and scores that ignore anything the user did since the last run.

**Q12. Why does RAG exist instead of fine-tuning knowledge in, or pasting all
docs into the prompt?**
A: A support corpus of 100K docs at 2 KB each is 200 MB, about 50M tokens at 4
bytes per token. Context windows hold $10^5$ to $10^6$ tokens, so pasting it
all does not fit, and even within the window cost grows linearly with prompt
length and models use facts in the middle of long prompts less reliably.
Fine-tuning takes hours to days per update, does not reliably store specific
facts verbatim, and cannot cite a source. Retrieval inserts the top 5 chunks of
200 to 500 tokens, 1,000 to 2,500 tokens per question; a knowledge update
becomes re-indexing the changed documents, and the answer can cite the chunk it
used.

**If asked:** *When is fine-tuning the right tool?* For behavior and format
(tone, output schema, domain style), not for facts that change.

**Q13. How do you evaluate a RAG system?**
A: Build 200 (question, gold passage) pairs and measure recall@5, the fraction
of questions whose gold passage is among the top 5 retrieved chunks. If it is
60%, then 40% of questions reach the model without the passage that answers
them, so measure retrieval first. Then score generation: faithfulness (every
claim supported by the retrieved context, judged by humans on a sample and by an
LLM judge validated against a few hundred human labels), correctness on a
golden Q&A set, and the abstention rate on questions the corpus cannot answer.
Online, track deflection rate (tickets avoided), escalation-to-human rate, and
user ratings.

**If asked:** *Why not one end-to-end score?* A 70% end-to-end score does not
say whether retrieval or generation failed, so it does not say what to fix.

**Q14. What do you monitor when labels arrive 60 days late?**
A: Take a fraud model whose chargeback labels arrive after 30 to 90 days.
Monitor inputs per feature against the training reference: null rate (a
feature jumping to 40% null means a broken upstream join), PSI with an alert
near 0.2, and the rate of unseen categories. Monitor predictions: mean score,
score histogram, fraction above the action threshold; if the mean predicted
fraud probability doubles overnight, either the world or a feature changed.
Slice both by segment. Add proxy labels that arrive in days (customer fraud
reports, manual-review decisions), and compute true-label metrics by
label-maturity cohort as labels arrive.

**If asked:** *Why slice by segment?* A drop in one country and a rise in
another can cancel in the aggregate.

**Q15. Your input features look stable but online performance is decaying. What
is it and what do you do?**
A: Suppose every feature has PSI below 0.1 while precision at the threshold
falls from 60% to 45% over a month. $P(x)$ is stable and $P(y \mid x)$ is
moving: concept drift, in fraud usually attackers adapting to the model.
Confirm it with calibration $\sum \hat{p} / \sum y$ moving away from 1 on the
freshest labeled or proxy-labeled cohort, and rule out a change in the label
pipeline. Respond with faster labels (proxies, manual review), more frequent
retraining on a recent window, rules for the new pattern while labels arrive,
and features that are costly for attackers to change (device fingerprints, card
and account graph features).

**If asked:** *Why not auto-retrain and auto-deploy?* If the cause is a broken
pipeline, the retrained model learns the broken data; the same offline gate and
canary as a manual release catch that.

**Q16. How do you serve a model that must score 500 items in under 100ms?**
A: Budget it: 5 ms parsing and routing, 15 ms candidate generation, 20 ms
feature fetch, 25 ms model, 10 ms re-ranking, 75 ms in total, leaving 25 ms for
the p99 tail. The feature fetch decides feasibility: 500 sequential lookups at
0.5 ms is 250 ms, so issue one multi-get for all 500 keys and store each item's
features as one blob (500 keys, not 25,000). Score all 500 in one batched
forward pass, not 500 calls. With 500 parallel lookups, the chance that none
lands in the store's slowest 1% is $0.99^{500} \approx 0.7\%$, so nearly every
request waits on a tail lookup; hedged requests (resend a slow lookup to a
second replica, take the first reply) and caching hot items reduce that.

**If asked:** *What if features miss the 20 ms budget?* Serve with default
values and log it, because unlogged defaults change the input distribution the
model sees.

**Q17. Why keep a rules engine alongside an ML fraud model?**
A: A fraud ring switches to \$499 gift cards from aged accounts on Monday. A
rule for that pattern ships the same day; the chargeback labels that would teach
a retrained model arrive 30 to 90 days later. Beyond speed, rules cover patterns
attackers find by probing the model, enforce hard constraints such as sanctions
lists that must hold regardless of the score, and give the review team a
readable reason for each block. They are also the fallback control when the
model degrades.

**If asked:** *Cost of rules?* Each one is maintained by hand and attackers
route around it (a \$500 rule produced \$499 gift cards), so rules need owners,
hit-rate monitoring and expiry dates.

**Q18. The PM asks for "just one metric" for the new recommender. What do you
say?**
A: Optimize click rate alone and the model learns thumbnails that get clicks
followed by 5-second bounces. Give one primary online metric aligned to the
goal (for example long-watch rate), which is the number the launch decision
uses, plus guardrails that must not regress (report rate, diversity, latency,
revenue) and a counter-metric for the primary's known failure mode (bounce rate
against clickbait). Goodhart's law: once a measure becomes the optimization
target, the system finds the cheapest way to raise it, and the measure stops
tracking the goal.

**If asked:** *How are the weights in a multi-head score set?* As a product
decision tuned by A/B tests under guardrails, because logged data has no label
for how much a share is worth relative to a click.

**Q19. Why does memory, not compute, cap how many conversations one GPU can
serve for a 7B LLM?**
A: A 7B model in fp16 is 14 GB, leaving 66 GB on an 80 GB GPU. Each token of
context needs 0.5 MB of KV cache ($2 \times 32$ layers $\times 4096 \times 2$
bytes), so a 2,000-token conversation holds 1 GB, and about 66 conversations
fill the GPU. Compute is far from the limit: each decode step reads the 14 GB
of weights plus 66 GB of cache, about 80 GB, so at 3 TB/s it runs about 37
steps per second, which needs $66 \times 14$ GFLOPs $\times 37 \approx 34$
TFLOP/s, a small fraction of a modern GPU's hundreds of TFLOP/s. So the fixes target memory:
grouped-query attention shares K and V across heads (4 to 8 times smaller),
KV quantization halves it, and paged allocation removes fragmentation.

**If asked:** *Then why batch at all?* Each decode step reads the weights
once for the whole batch, so tokens per second grow with batch size until
the KV cache fills memory.

## Exit checklist

- [ ] Cold, blank editor: write the 10-step universal framework, one sentence per step (5 min)
- [ ] Sketch the two-stage recsys funnel (retrieval → ranking) with candidate counts, models, and latency budget per stage (5 min)
- [ ] Define training/serving skew and point-in-time correctness, one concrete failure example each (4 min)
- [ ] Data drift vs concept drift: precise definitions plus what you'd monitor to catch each (3 min)
- [ ] Walk shadow → canary → A/B rollout and name what each stage catches that the previous can't (3 min)
- [ ] Design RAG end to end: chunking, embedding, retrieval, grounding, and its top three failure modes (5 min)
- [ ] Do the back-of-envelope latency math for scoring 1k candidates under a 100 ms budget (4 min)

All boxes cold before moving on; a miss goes into practice/queue.tsv.
