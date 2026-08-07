# Two Complete Mock Interviews - Annotated Transcripts

Format: full dialogue, with blockquote annotations explaining *why* the
candidate's move works (or what a weaker answer looks like). Read these after
`notes.md`; the candidate is executing that framework under time pressure.

---

## Walkthrough 1: "Design a video recommendation system"

**Interviewer:** Let's do a design question. Design the recommendation system
for a video app - think of something like the homepage of YouTube or a
short-video feed. You have 45 minutes.

**Candidate:** Great. Before I touch architecture I want to pin down the product,
because the label, the metrics, and the latency budget all fall out of that. A
few questions: Is this the logged-in homepage - so we have user history - or
also cold-start users? What's the primary business goal: total watch time,
session frequency, subscriptions? And roughly what scale: how many users, how
many videos, and does the catalog churn fast - are new uploads important within
minutes?

> **Why this is good:** The candidate refuses to design in a vacuum and asks
> questions that *change the design* (cold-start changes candidate sources,
> freshness changes indexing, goal changes labels). A weak candidate asks
> generic questions ("what's the scale?") without saying why they matter, or
> worse, starts drawing boxes immediately.

**Interviewer:** Logged-in homepage. Say 200M DAU, about 1 billion videos,
uploads matter within an hour or so. The business cares about total time spent,
but they've been burned by clickbait before.

**Candidate:** That last sentence is important. I'll optimize *satisfied* watch
time, not clicks - concretely, my primary online metric will be something like
total watch time per user per day, with a "long-watch rate" flavor (fraction of
plays watched past, say, 50% or 30 seconds), and I'll carry counter-metrics:
report/dislike rate and early-abandon rate, so clickbait that gets clicked and
abandoned in 5 seconds scores as a failure, not a success.

Let me set up scale numbers, since they'll force the architecture. 200M DAU,
assume ~5 homepage loads/day → $10^9$ requests/day ≈ $10^4$/sec average, call
it 30–50K QPS at peak. Corpus $10^9$ videos. Latency: homepage recs should
render within a couple hundred ms; I'll take a 150ms budget for the rec service.

> **Why this is good:** Numbers first, out loud, rounded to powers of ten. The
> $10^9$-corpus number is about to *justify* the entire multi-stage
> architecture. Weak answers name-drop "candidate generation and ranking"
> because that's what the blog posts say; strong answers derive it.

**Candidate:** Now the central constraint. Any model good enough to personalize
well - something that looks at user history crossed with video features - costs
on the order of 0.1–1ms per (user, video) score. Even at 1µs per score, scoring
$10^9$ videos per request is $10^3$ seconds of compute. It's not that one model
is inelegant - it's arithmetically impossible. So the design is a funnel:

- **Candidate generation:** $10^9 \to$ a few thousand, ~10–15ms, using models
  whose structure permits precomputation.
- **Ranking:** a few thousand $\to$ top ~100 with an expressive model, ~40ms.
- **Re-ranking:** slate-level logic - diversity, freshness, policy - to the
  final ~30, ~10ms.

**Interviewer:** OK. How does candidate generation actually find 1000 good
videos out of a billion in 10 milliseconds?

**Candidate:** By restricting the scoring function to something indexable. If I
force the score to be a dot product between a user vector and a video vector  - 
$s(u,v) = f(u)\cdot g(v)$ where the two sides never interact until that dot
product - then all $10^9$ video vectors can be embedded offline, in batch, and
put into an approximate nearest-neighbor index. ANN structures answer "top-k by
inner product" over a billion vectors in single-digit milliseconds. At request
time I compute one user embedding $f(u)$ - one small forward pass using the
user's history - then one ANN lookup. That's the two-tower model, and the thing
to be clear about is that its defining property is a *limitation we chose on
purpose*: no user-video cross features, in exchange for precomputability.

For training it, I'd use engagement pairs - (user, video they watched
substantially) - with a sampled-softmax/contrastive objective, using in-batch
negatives with a popularity correction, since popular videos otherwise appear as
negatives far too often.

I wouldn't rely on a single source, though. Embedding retrieval is weak exactly
where interaction history is missing - brand-new uploads, and new users. So
candidate generation is a union of sources, each covering another's blind spot:
two-tower ANN for personalization, subscriptions/follows, "trending now" (which
also handles the one-hour-freshness requirement, since a popularity source can
update in minutes from a stream of play events), and co-watch neighbors of the
user's recent videos. Maybe 300–500 from each, deduped to ~2000.

> **Why this is good:** Two-tower is *derived* (constraint → precomputation →
> ANN), its cost is stated honestly, and its blind spots immediately generate
> the multi-source design and the freshness answer. A weak answer says "I'd use
> a two-tower model" as a vocabulary word and gets dismantled by the follow-up
> "why not one tower?" or "how do new videos get recommended?"

**Interviewer:** You mentioned new users. Your two-tower needs history. What
does a brand-new user see?

**Candidate:** Cold start degrades gracefully through the source mix. With zero
history: trending + regional/language popularity + whatever signup signals exist
(country, language, chosen topics, device). The user tower should be built on
features that *exist at signup* plus history aggregates that default cleanly to
"empty," so it produces a usable - if generic - embedding from day one rather
than garbage. And I'd bias early sessions toward exploration: a new user's
first-day clicks are worth a lot of information, so showing a deliberately
diverse slate beats exploiting a weak prior. The metric for this cohort is
different too - early retention, not watch time - and I'd slice every dashboard
by user tenure so the average doesn't hide a broken new-user experience.

**Interviewer:** Fine. Tell me about the ranking stage - what's the model, and
what exactly is the label?

**Candidate:** Now that we're down to ~2000 candidates, I can afford real
features: user long-term topic preferences, user recent-session history, video
features (topic, length, age, historical engagement), and - where most of the
lift usually is - crosses like "user's historical long-watch rate on this
creator" and "similarity of this video to the last 10 watched."

The label question is where the clickbait constraint bites. If I train on
$P(\text{click})$, I build the clickbait machine the business already got burned
by. So the ranker is multi-task: predict $P(\text{click})$, $P(\text{watch} >
30s)$, expected watch fraction, $P(\text{like})$, $P(\text{report/hide})$. The
final ordering score is a weighted combination - something like $s = w_1
\hat{p}_{\text{long}} + w_2 \mathbb{E}[\text{watch}] + w_3 \hat{p}_{\text{like}}
- w_4 \hat{p}_{\text{report}}$ - and I want to be explicit that those weights
are a *product decision* tuned through A/B tests, not something the model
learns. That's the knob where "we care about satisfaction over clicks" actually
lives.

Model class: I'd start with gradient-boosted trees over these features as v1  - 
strong on tabular crosses, trivial to serve at 0.2ms/candidate, easy to debug  - 
and move to a deep multi-task model in v2 when I want learned embeddings of
watch sequences. All 2000 candidates go through in one batched forward pass.

Two label hygiene points, because they're where offline numbers lie. First,
position bias: homepage slot 1 gets clicked because it's slot 1. I'd feed
position as a feature during training and freeze it at serving, so the model
doesn't launder the old ranker's placements into "relevance." Second, my
training data only contains videos previous systems chose to *show* - so offline
eval only tells me about re-ranking the old policy's choices. I'd hold out a
small exploration slice - order 1% of slates with some randomization, logged
with propensities - as the closest thing to unbiased ground truth we can get.

> **Why this is good:** The candidate connects labels to the stated business
> pain (clickbait), places the value-weights decision with the product team,
> starts with GBDT *with a reason*, and raises position/selection bias unprompted
> - the two things that most reliably distinguish people who have shipped
> rankers. A weak answer: "train a DNN on clicks, optimize AUC."

**Interviewer:** Let's stress the serving path. Walk me through the 150ms
budget, and tell me what breaks first as traffic doubles.

**Candidate:** Budget, per request, targeting p50 ~80ms so p99 fits in 150:

| Stage | Budget |
|---|---|
| Routing/auth/overhead | 5ms |
| User feature + history fetch (1–2 keyed reads) | 5ms |
| Candidate sources in parallel (ANN, trending, subs) | 15ms |
| Item feature fetch for ~2000 candidates | 20ms |
| Ranker forward pass, batched | 25ms |
| Re-rank + assemble response | 10ms |
| **Total** | **80ms** |

The fragile line is the item-feature fetch. 2000 candidates × dozens of features
must be *one* batched multi-get against an online store, with each video's
features packed as a single blob - 2000 sequential lookups at 0.5ms would be a
full second. Item reads are heavily Zipfian, so an in-process cache of hot
videos absorbs most of that fan-out. And I need an explicit timeout policy: if
features aren't back in 20ms, score with defaults and log that we did - silently
serving defaults is a model-quality incident that looks like a systems hiccup.

What breaks first as load doubles: the ranker's compute. 50K QPS × 2000
candidates = $10^8$ scores/sec; at 0.2ms/score that's ~20K cores - already a
few hundred machines, and it scales linearly with both traffic and candidate
count. So my first levers are cutting candidates (is 2000 → 800 measurably
worse? test it), caching scores for (user, video) pairs re-requested within
minutes, and precomputing slates for the long tail of users who open the app
once a day - their homepage can be batch-computed hourly and served from KV
storage for near-zero marginal cost, keeping online ranking for active sessions.

> **Why this is good:** A concrete table, the sequential-vs-batched fetch trap
> called out, timeout-with-logging (production scar tissue), and a
> capacity answer that includes *which lever to pull*, including the
> batch-precompute hybrid. Weak answers hand-wave "we'd scale horizontally."

**Interviewer:** Last five minutes. You ship v1 and offline AUC on the ranker
improves in v2, but the A/B shows watch time flat. What do you do?

**Candidate:** First, distrust the offline number for the standard reasons:
it's computed on logged, old-policy-biased data, and it measures global pairwise
ranking rather than the top-30 slots users actually see. Then debug in order of
cheapness. One: check training/serving skew - I'd have shadow-logged v2's served
features; diff them against the training pipeline's recomputation for the same
requests. A mismatch there explains everything and is the most common culprit.
Two: check where the AUC gain lives - recompute offline metrics restricted to
top-k (NDCG@30) and per segment; a gain concentrated in the tail of the ranking
or in one segment can be invisible online. Three: check the experiment itself  - 
power analysis (is the test even sized to detect the expected lift?), novelty
effects, whether guardrails moved. And four: check the funnel - if v2 ranks
differently, maybe candidate generation is now the binding constraint: the
videos v2 would rank highly aren't in its candidate set. Source-recall
monitoring - "what fraction of watched videos each source retrieved" - tells me
whether to spend the next quarter on the ranker or the retriever.

**Interviewer:** Good place to stop.

> **Why this is good:** The close mirrors the metrics section of the framework  - 
> the candidate has a *diagnostic ordering* (skew → metric locality → experiment
> validity → funnel), each step with a concrete artifact to check. This is the
> single most common follow-up in ML system design interviews; have this
> four-step answer loaded.

---

## Walkthrough 2: "Design a spam/fraud detector"

**Interviewer:** Your product is a payments platform - think online checkout
for lots of merchants. Design the system that decides whether a transaction is
fraudulent.

**Candidate:** Let me get the decision context first, because fraud design is
dominated by three things: what actions we can take, what mistakes cost, and how
fast we must decide. Questions: Is this a synchronous decision in the payment
path - the customer is waiting - and what's the latency budget? What actions
are available: just approve/decline, or can we challenge - 3DS, SMS
verification? And do we have a manual review team? Also scale and base rate:
transactions per day, and roughly what fraud rate?

**Interviewer:** Synchronous, the processor gives you about 200ms. You can
approve, decline, or send to a step-up challenge. There's a small review team.
Say 20M transactions/day, fraud is around 0.2% of transactions.

**Candidate:** Good - the three-way action space matters a lot; I'll come back
to it. Scale: 20M/day ≈ 230/sec average, maybe 1–2K QPS at peak with bursts
(sales events). That's modest QPS with a tight 200ms budget - so the challenge
here is not throughput, it's feature freshness, class imbalance, and the fact
that the adversary adapts. Let me state the three properties that make this
problem its own genre, because each one changes a design decision:

First, **imbalance**: 0.2% positives means ~40K fraud events/day in 20M - so
accuracy is meaningless and even ROC-AUC will look flattering, because the
false-positive *rate* is measured against 19.96M legit transactions. I'll use
precision/recall at the chosen operating points and PR-AUC, and I'd report
recall in *dollars* - catching a \$5,000 fraud is not the same as a \$5 one.

Second, **asymmetric costs**: a missed fraud costs the amount plus a ~\$20
chargeback fee plus network penalties if our fraud ratio rises; a false decline
costs the sale plus, worse, customer lifetime value - people don't come back to
a checkout that rejected their card. These costs vary *per transaction*, so I
don't want to bake them into training. I want a **calibrated**
$\hat{p} = P(\text{fraud})$ and a decision layer on top computing expected
value: decline when $\hat{p} \cdot \text{loss}_{\text{fraud}}$ exceeds
$(1-\hat{p}) \cdot \text{cost}_{\text{decline}}$, and - this is where the
step-up action earns its keep - challenge in the ambiguous middle band, because
a challenge converts uncertainty into information for ~10 seconds of friction
instead of a lost customer.

Third, **adversarial drift**: unlike a movie recommender, my errors are being
actively searched for. Whatever pattern I block, next week's fraud will be
shaped around it. That commits me, downstream, to fast retraining, drift-triggered
alerts, a rules layer for fast response, and features that are *expensive for
the attacker to change*.

> **Why this is good:** Before any architecture, the candidate names the three
> structural properties and ties each to a concrete commitment. The
> separation of calibrated probability from business-cost decisioning is a
> senior move - a weak answer picks "a threshold of 0.5" or bakes costs into
> class weights and can't adapt when the chargeback fee changes.

**Interviewer:** You said labels are a problem in fraud. Spell that out - what
exactly do you train on?

**Candidate:** The ground-truth label is a chargeback, and it arrives 30–90
days late. Three nasty consequences. One, my freshest reliable training data
describes fraud tactics from two months ago - precisely the tactics the
adversary has already abandoned. Two, at any moment, the last ~60 days of data
are *partially* labeled - a transaction with no chargeback yet isn't confirmed
legit - so I have to train either on mature cohorts only, or treat recent
negatives as noisy. Three, and subtlest: transactions we **declined** never get
a label at all. We'll never observe whether they were actually fraud. That's
selection bias with teeth - train naively on approved-only traffic and the
model slowly forgets what the fraud it already catches looks like, then
"discovers" it's safe to approve it again.

Mitigations, in the order I'd implement them: use faster proxy labels  - 
customer-reported fraud and issuer alerts arrive in days, manual-review
verdicts in hours - as an early label stream, reconciled later against
chargebacks. Feed the review team strategically: their labels are expensive but
fast and unbiased-ish, so spend them on the model's uncertain band - which the
step-up/review queue naturally produces. And for the censoring problem, let a
very small random fraction of would-be declines through - it costs real money,
so it's small and capped by amount, but it's the only source of ground truth in
the declined region. I'd frame that explicitly as paying for label coverage,
and get the budget agreed with the fraud-loss owners rather than smuggling it in.

> **Why this is good:** "What do you train on?" is *the* fraud follow-up, and
> the candidate covers delay, partial maturity, and censoring-by-our-own-
> decisions - plus mitigations with their costs stated honestly. A weak answer:
> "we train on chargebacks" with no mention that the model's own actions censor
> the labels.

**Interviewer:** OK - features and model. What signals actually catch fraud?

**Candidate:** The high-value features are mostly about *behavior over time and
across entities*, not the transaction in isolation, which drives the serving
design. Four families:

- **Velocity features:** counts and sums over sliding windows per entity  - 
  transactions per card in the last 10 min / hour / day, distinct cards per
  device, distinct shipping addresses per card, amount vs the card's trailing
  average. Card-testing attacks are literally "one card, many rapid small
  charges" - a feature computed by a nightly batch job misses the entire event.
  So these need an online store updated within seconds from the transaction
  stream, with windowed counters.
- **Graph/linkage features:** does this device/email/address connect to
  previously confirmed fraud? Fraud comes in rings - shared fingerprints across
  "unrelated" accounts. These are expensive for attackers to vary, which is
  exactly the property I want, given adversarial drift.
- **Mismatch/context features:** IP-geo vs billing vs shipping distance, account
  age, time-of-day vs the user's pattern, price-category anomaly.
- **Merchant-side features:** merchant's category and historical fraud rate  - 
  some of the signal is "where," not "who."

Every one of these must be **point-in-time correct** in training: for a
transaction at time $t$, features computed only from events strictly before
$t$. The tempting bug is joining training data against "today's totals" - a
card's full-day transaction count is inflated *by the fraud burst itself*, so
the leaky feature looks brilliant offline and evaporates online. This is the
strongest argument for a feature store here: one definition per feature,
materialized online for serving and as-of-joined over history for training  - 
otherwise the training SQL and the serving counters will drift apart and we get
skew on top of leakage.

Model: **GBDT first**, and in fraud I might stop there for a long time - tabular
features with heavy interactions is its home turf, it's fast (sub-ms), and the
review team can be given reason codes from feature attributions, which they'll
demand. Handle imbalance by downsampling negatives ~10:1 and then **correcting
the calibration** - if I downsample by $w$, raw scores are inflated; map back
with $p = p'/(p' + (1-p')/w)$ - because my whole decision layer assumed honest
probabilities. Deep models earn a slot later for sequence modeling (the *shape*
of a card's recent event sequence) and shared embeddings of devices/emails, as
a second score feeding the same decision layer, not a rip-and-replace.

> **Why this is good:** Features are tied to attacker economics ("expensive to
> change"), the point-in-time trap is explained with the exact mechanism by
> which it flatters offline metrics, and the calibration correction after
> downsampling - which almost everyone forgets - is stated with the formula.
> The model choice is justified by ops needs (reason codes, latency), not
> fashion.

**Interviewer:** Put the serving path together for me under 200ms. And where do
rules fit - do we even need ML here, or do rules suffice?

**Candidate:** Honestly, v0 *is* rules - velocity caps, geo mismatches,
blocklists probably catch a large share of fraud at near-zero complexity cost,
and we should ship that while the ML pipeline is being built; it also generates
labels for review. But rules alone plateau: they're brittle points in feature
space, the adversary walks around them in days, and combinatorial interactions - "slightly odd amount + slightly new device + slightly unusual hour" - are
exactly what a model integrates and rules can't. So the steady state is layered,
and the layers have different jobs:

1. **Hard rules first** (~5ms): sanctions/blocklists, impossible values  - 
   non-negotiable, interpretable, instant to update. Also the emergency brake:
   when a new attack pattern shows up at 2am, an analyst ships a rule in
   minutes while the model waits for labels.
2. **Feature assembly** (~40ms): one batched read of precomputed entity blobs
   (card, device, account, merchant) from the online store, plus streaming
   counters. Timeouts substitute defaults *and log loudly* - a degraded score
   beats a timed-out payment, but silent defaults are how models rot.
3. **Model scoring** (~10ms): GBDT forward, calibrated.
4. **Decision layer** (~5ms): expected-value thresholds → approve / challenge /
   decline (+ queue for review). Thresholds are config, not code - the fraud
   ops team will tune them weekly as costs and attack volumes shift, and they
   must not need a deploy to do it.

That's ~60–70ms with big headroom against the 200ms SLA - which I want,
because the tail matters: a timed-out fraud check becomes someone's default
policy, and I'd rather define that policy deliberately (fail-open below \$X
with logging, fail-closed above).

**Interviewer:** Last question. It's Monday morning and the chargeback team
says losses doubled over the weekend, but your model's offline metrics - on the
data you have - look unchanged. Walk me through your response.

**Candidate:** Classic fraud incident, and the trap in the question is
"offline metrics look unchanged" - of course they do; my labeled data is weeks
old and the attack started Saturday. Chargebacks doubling *reported today*
actually describes fraud from weeks ago maturing - so first I'd check proxy
signals with short lag: issuer alerts, customer fraud reports, review-team hit
rate over the weekend. Those tell me if there's a live ongoing attack versus an
old cohort maturing badly.

In parallel, the label-free monitors: prediction-score distribution over the
weekend - did the mean score drop or shift? - and per-feature input drift
(PSI vs reference), sliced by segment: merchant, country, amount band, new-vs-old
accounts. Fraud attacks are concentrated, so segment slicing usually lights up
one cell - say, one merchant category from one region with brand-new accounts.
Also the boring hypothesis first: did a feature break? A null-rate spike in a
velocity counter - a stream job that died Friday night - makes the model blind
in exactly the way an attack would exploit, and "feature outage" is more common
than "brilliant new adversary."

If it's a genuine new pattern: rule out the specific pattern *today* via the
rules layer, tighten the challenge band (shift the step-up thresholds so the
gray zone gets 3DS instead of approval - friction is cheap during an active
attack), feed confirmed weekend cases to the review team to produce fast
labels, and kick off retraining on the freshest proxy-labeled data - gated, as
always, through offline eval and a canary before it takes over decisions. Then
the postmortem question: which monitor *should* have paged us Saturday, and why
didn't it.

> **Why this is good:** The candidate spots that the offline metric is
> structurally incapable of showing the incident (label delay), checks the cheap
> boring cause (broken feature) before the exciting one (new attack), uses the
> layered system as designed - rules for hours-scale response, thresholds for
> risk posture, retraining for weeks-scale - and closes with the monitoring
> postmortem. A weak answer jumps straight to "retrain the model," which does
> nothing for days and may train on contaminated data.

**Interviewer:** That's time. Nice work.

---

## Patterns across both transcripts (steal these moves)

1. **Open with questions that change the design**, and say why each answer
   matters before you get it.
2. **Do the disqualifying arithmetic early** ($10^9$ items × 1µs; 2000
   candidates × 0.2ms × QPS) - it justifies architecture better than any
   name-drop.
3. **Derive components from constraints**: two-tower from "must precompute for
   ANN"; feature store from "one definition, two materializations";
   step-up tier from "uncertainty has a price between approve and decline."
4. **Separate model from decision**: calibrated probability in, business costs
   and thresholds outside the model, in config.
5. **Volunteer the ways your offline numbers lie** (position bias, selection
   bias, leakage, label delay) *before* the interviewer raises them.
6. **Have the "offline good / online bad" debug ordering memorized**: skew →
   metric locality → experiment validity → funnel/candidate recall.
7. **In incidents, check broken pipelines before clever adversaries**, and use
   fast layers (rules, thresholds) while slow layers (labels, retraining) catch
   up.
