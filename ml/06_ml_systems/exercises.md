# ML System Design — Exercises

Ten open-ended exercises. Attempt each out loud for 10–15 minutes (whiteboard or
paper) *before* opening the sample answer. The sample answers are deliberately
opinionated — a strong candidate's sketch, not the only valid design.

---

## 1. Design push-notification ranking

Your app can send each user push notifications about new content, replies, and
promotions. Design the system that decides **whether** to send, **what** to
send, and **when**. What makes this different from feed ranking?

<details><summary>Sample answer</summary>

**What's different from a feed:** a feed is *pull* (user asked; worst case they
scroll past), a notification is *push* (you interrupt; worst case they disable
notifications or uninstall — a near-irreversible loss). So the dominant design
feature is the **asymmetric, delayed cost of annoyance**, and the core decision
is send/don't-send, not just ranking.

**Formulation:** two-part model. (1) $P(\text{positive engagement} \mid
\text{send, user, item, time})$ — open/click within, say, 4h. (2) A cost side:
$P(\text{disable or uninstall})$ as a function of recent notification load.
Decision: send the best candidate only if expected value clears a bar,
$\hat{p}_{\text{open}} \cdot v_{\text{open}} - \hat{p}_{\text{annoy}} \cdot
v_{\text{churn}} > \tau$, with a hard **frequency cap** (e.g., ≤2/day) and
quiet hours as rules outside the model. The bar $\tau$ effectively prices the
user's attention.

**Labels and their traps:** opens are position/timing-confounded (a 9am
notification opens more because the user is awake, not because content is
better — include send-time features and compare within-time cohorts). Silent
harm is invisible per-notification: disabling happens after the *tenth* annoying
push, so attribute with a survival/uplift framing on notification *load*, and
run **holdout groups** (users who get none / fewer) permanently to measure
long-run retention impact — the online metric is retention/DAU, not opens.
Optimizing opens alone is the classic Goodhart trap here.

**Serving:** not request-driven — a candidate event stream + a scheduler.
Batch-score candidates when events occur, hold them, deliver at each user's
predicted-receptive time. Latency budget is minutes, so batch inference is fine;
the hard infra is per-user state (recent sends, caps, quiet hours). Numbers:
$10^8$ users × ~5 candidate events/day = $5\times10^8$ scores/day ≈ 6K
scores/sec average — a handful of machines with a GBDT.

**Monitoring:** opt-out rate and notification-disable rate are the guardrails
that page you; opens can rise while you're burning the user base.
</details>

---

## 2. Online CTR dropped, offline AUC unchanged — debug it

Your ranking model has served for months. This week online CTR is down 8%, but
when you evaluate the current model on fresh logged data, offline AUC is the
same as ever. Walk through your debugging process.

<details><summary>Sample answer</summary>

Key deduction up front: offline AUC unchanged means *given the features it
receives in the eval pipeline*, the model ranks as well as ever. So either the
serving path differs from the eval path, or the world/product changed in a way
AUC doesn't see. Order of investigation, cheapest and most likely first:

1. **Non-ML causes (check before touching ML):** did the app/UI ship a change
   (layout, card size, new module stealing clicks)? Did latency regress
   (slower page → fewer clicks at identical ranking quality)? Did traffic mix
   shift (a marketing push flooding in low-intent users — CTR drops with model
   quality constant)? Segment the CTR drop: by platform, country, user tenure,
   surface. A drop isolated to one app version answers the question.
2. **Training/serving skew, freshly introduced:** diff *served* feature values
   (from serving-time logs) against the eval pipeline's recomputation for the
   same requests. A stale/broken online feature (dead streaming job → counter
   frozen at 0) degrades serving, while offline eval — recomputing the feature
   correctly from logs — looks fine. This exact signature (offline fine, online
   bad) is skew's fingerprint. Check feature null/default rates and
   feature-fetch timeout rates this week vs last.
3. **Upstream funnel:** ranking AUC is conditioned on candidates. If candidate
   generation degraded (a source down, index stale), the ranker re-ranks a
   worse pool perfectly — AUC on shown items unchanged, CTR down. Check
   per-source candidate volume and source-recall dashboards.
4. **Score consumption:** did the calibration or a downstream consumer change —
   ad load, blending weights with other content types, threshold configs?
5. **Only then, drift:** input PSI and prediction-distribution shifts — though
   true drift usually *does* dent offline AUC on fresh labels, so it's the
   least consistent explanation for this signature.

A weak answer starts with "retrain the model" — which fixes none of the above
and delays the real diagnosis.
</details>

---

## 3. Design labels for "helpfulness" of an AI assistant

You run an LLM assistant product. Leadership wants the model improved for
"helpfulness." Design the labeling/measurement system that defines and collects
this signal at scale.

<details><summary>Sample answer</summary>

First, refuse the undefined word: decompose "helpful" into measurable claims —
(a) the answer is correct, (b) it addresses the user's actual intent, (c) it's
complete/actionable, (d) appropriately calibrated (says "I don't know" when it
should). Different failures, different fixes; a single scalar hides which one
you have.

**Layered label sources, cheap-and-biased to expensive-and-clean:**

1. **Implicit behavioral signals (free, huge, biased):** thumbs up/down
   (~sub-1% response rate, selection-biased toward extremes), regeneration and
   "stop generation" (mild negatives), copy-to-clipboard / code-run (positives),
   user rephrasing the same question (strong negative — the answer missed
   intent), conversation abandonment vs task completion. Treat these as weak
   labels; calibrate them against human labels below (e.g., learn how much a
   regenerate is "worth").
2. **Human ratings on a sampled slice (thousands/week, the anchor):** trained
   raters score sampled conversations on the decomposed rubric (correctness /
   intent / completeness / calibration, each 1–5 with written guidelines and
   ~10% double-rating to track inter-rater agreement — if raters can't agree,
   the rubric, not the raters, is broken). Sample *stratified*: uniform random
   plus oversampled low-implicit-signal and high-stakes categories.
3. **LLM-as-judge (the scaling bridge):** a judge model applies the same rubric
   to a much larger sample. Crucially, **validate the judge against the human
   slice** (agreement rate per rubric axis) before trusting it, re-validate
   after every judge/prompt change, and watch known judge biases (verbosity,
   self-preference, position bias in pairwise comparisons — randomize order).
4. **Ground-truth subsets where they exist:** coding (did tests pass), math
   (checkable answers), retrieval-backed factual QA — objective islands that
   anchor the subjective sea.

**Bias handling:** thumbs are missing-not-at-random — model the response
propensity or at minimum report metrics on the human-rated random sample, never
on the self-selected feedback. **Aggregate metric:** report the rubric axes
separately + a composite for headlines; track per-topic slices (helpfulness on
tax questions vs poetry is not one number). These labels then feed evaluation
first, and preference-tuning (pairwise comparisons are more reliable than
absolute scores — humans rank better than they rate) second.
</details>

---

## 4. Estimate hardware to serve a 7B LLM at 1000 QPS

Requests average 500 input tokens and 200 output tokens. Latency target: first
token < 1s, full response < 10s. Do the estimate out loud; state assumptions.

<details><summary>Sample answer</summary>

**Demand:** 1000 req/s × 200 output tokens = $2\times10^5$ output tokens/s to
generate; 1000 × 500 = $5\times10^5$ input tokens/s to prefill.

**Weights:** 7B params × 2 bytes (fp16) = **14 GB** — fits on one 80 GB
accelerator with room to spare; the question is throughput, not fitting.

**Decode is memory-bandwidth-bound.** Each decoding step must read all weights:
14 GB/step. At ~3 TB/s HBM bandwidth → ~200 forward steps/s per GPU,
*regardless of batch size* — so batch hard: with a batch of 64 concurrent
streams, one GPU produces ~$200 \times 64 \approx 13\text{K}$ tokens/s. Need
$2\times10^5$ → **~15–16 GPUs for decode**. (Sanity check per-stream speed: 200
steps/s ≈ 200 tokens/s? No — each stream advances once per step only if
scheduled every step; realistically ~30–60 tok/s/stream → 200 tokens in ~4–7s.
Meets the 10s target.)

**Prefill is compute-bound:** ~$2 \times 7\times10^9 = 14$ GFLOPs/token ×
$5\times10^5$ tok/s = 7 PFLOP/s. At ~1 PFLOP/s peak and ~40% utilization →
**~18 GPUs for prefill**. Same order as decode.

**KV cache — the hidden constraint.** Concurrency = arrival rate × residence
time ≈ 1000 req/s × ~5 s = **5000 concurrent requests**, each holding ~700
tokens of context. Naive fp16 KV for a 7B (32 layers × 4096 dim × 2 (K,V) × 2
bytes) ≈ 0.5 MB/token → 700 × 0.5 MB × 5000 ≈ **1.75 TB** — dwarfs weights!
This is why serving stacks exist: grouped-query attention (~4–8× smaller KV),
KV quantization (2×), paged allocation (no fragmentation) → ~200–400 GB total,
spread over the same ~30 GPUs (66 GB free per GPU after weights) — feasible,
but KV, not weights, decides the GPU count's memory side.

**Answer:** ~30–40 H100-class GPUs (≈16 decode + ≈18 prefill, with KV fitting
in their spare memory), plus ~30–50% headroom for peaks and redundancy →
**~50 GPUs / ~6–7 8-GPU nodes**. Levers if that's too expensive: int8/int4
quantization (2–4× on both bandwidth and memory), smaller/distilled model,
caching frequent responses, and speculative decoding. The interview point isn't
the exact count — it's showing decode = bandwidth-bound, prefill =
compute-bound, and KV cache = the memory budget everyone forgets.
</details>

---

## 5. Design "similar items" for e-commerce, including brand-new items

Every product page shows a "similar items" module. The catalog has $10^8$
products; 1% turn over daily (new listings). Design it, handling new items with
zero interaction history.

<details><summary>Sample answer</summary>

**Two notions of "similar" — pick per product goal:** substitutes (other items
you'd buy *instead*: same category, comparable price) vs complements (buy
*together*). A product page usually wants substitutes; post-purchase wants
complements. Say this; it changes the labels.

**Behavioral similarity (the strong signal, when it exists):** co-engagement —
items co-viewed/co-bought in sessions. Train item embeddings so items with
shared engagement contexts land nearby (a skip-gram over session sequences, or
item tower from a two-tower). Serve via ANN: $10^8$ items × 128 dims × 4 B ≈
51 GB — one index server, or int8-quantize to ~13 GB. Precompute top-50
neighbors for head items nightly (cheap KV lookups at serve time); ANN online
only for the tail.

**The cold-start hole:** a new listing has zero sessions, so behavioral
embeddings don't exist. This is where content comes in — **learn a mapping from
content to the behavioral space**: train a model that takes title/description
text, image, category, brand, price and predicts the *behavioral* embedding,
trained on established items (which have both). New items get a predicted
embedding at listing time → immediately servable in the same index; as real
interactions accrue, blend toward the behavioral embedding, e.g.
$e = \alpha(n)\, e_{\text{behav}} + (1-\alpha(n))\, e_{\text{content}}$ with
$\alpha$ ramping over the first ~100 interactions. This beats the naive
"fallback to same-category rules" because it lives in one index and one
similarity space.

**Freshness pipeline:** 1%/day = $10^6$ new items/day ≈ 12/s — embed on listing
(a content-model forward pass, trivially cheap) and upsert into the ANN index;
index must support streaming upserts or use a daily-rebuild + delta-index
pattern. **Evaluation:** offline recall of "co-bought within 7 days" pairs;
online A/B on module CTR → add-to-cart, sliced by item age — the *new-item
slice* is the whole point, so report it separately. Guardrail: don't show
same-item duplicates/counterfeits (dedupe by near-identical embedding + image
hash).
</details>

---

## 6. Design a content moderation system (policy violations at scale)

$10^8$ pieces of user content/day; policy violations (~0.5%) must come down
fast, but wrongful removals damage trust and invite regulatory pain. You have a
human review team of ~500. Design the system.

<details><summary>Sample answer</summary>

**Frame it as a triage/routing problem, not a classifier.** 500 reviewers × ~500
reviews/day ≈ $2.5\times10^5$ human decisions/day — that's 0.25% of content.
The ML system's job is to spend that scarce, high-quality resource where it
matters, and to act autonomously only where confidence × severity justifies it.

**Decision policy by (score, severity):** per policy area (not one lump —
CSAM/terror vs spam vs borderline harassment have different cost structures and
legal requirements): high-confidence severe → auto-remove immediately (recall
matters most; appeals as safety valve); high-confidence mild (spam) →
auto-remove/limit cheaply; uncertain band → human review queue, **ordered by
expected harm = $\hat p \cdot \text{predicted reach}$** — a maybe-violating
post about to go viral outranks a certain-violation post nobody will see
(prioritizing by virality × probability is the key move); low score → no
action, but sampled for QA.

**Models:** multi-modal (text, image, video frames) per-policy classifiers; a
cheap first-pass filter over everything ($10^8$/day ≈ 1.2K/s average — fine on
modest GPU fleet at ~10ms/item) with heavier models only on the flagged slice.
Hash-matching (known-bad content fingerprints) sits *before* ML — exact,
instant, and legally mandated in some areas.

**Labels:** reviewer decisions are the label stream — which makes label quality
a first-class system: calibrated rubrics, double-review sampling for agreement,
and *appeals outcomes* as labels for the wrongful-removal side. Beware feedback
loop: reviewers only see what the model flags → train-on-flagged ossifies blind
spots → maintain a small uniform random sample reviewed regardless of score
(this also gives the unbiased violation-rate estimate = your prevalence metric).

**Metrics:** prevalence (violating views ÷ total views, from the random sample
— the honest north star), median time-to-action on violations weighted by
views accrued, wrongful-removal rate (appeal overturn rate), reviewer
queue latency. **Adversarial reality:** evasion is constant (text in images,
leetspeak, re-uploads with pixel shifts) → perceptual hashing, OCR in the
pipeline, fast retrain cadence, and rules/hashes for hours-scale response while
models catch up.
</details>

---

## 7. Retraining strategy for a delivery-ETA model

Your food-delivery ETA model (predicts minutes-to-arrival) slowly degrades:
MAE has crept from 4.1 to 5.3 minutes over six months. Design the retraining
strategy: cadence, data windows, triggers, and safe rollout.

<details><summary>Sample answer</summary>

**Diagnose before scheduling:** decompose the degradation. ETA labels arrive
within an hour (the delivery completes) — a luxury; use it. Slice MAE by city,
daypart, restaurant cohort, weather. Creep usually isn't uniform: new cities
(data drift — model extrapolating), new restaurant types, seasonal shifts
(winter traffic = concept drift in the time-given-distance relationship). If
80% of the creep is three new markets, the fix is representation in training
data, not frequency.

**Measure the staleness-cost curve:** backtest — train on data up to time $T$,
evaluate at $T+1\text{wk}, T+4\text{wk}, T+12\text{wk}$. This yields "MAE
degrades ~0.15 min/month of staleness" — now cadence is arithmetic: weekly
retrains cost ~nothing in compute vs ~0.04 min MAE saved... whereas if the
curve is steep only after 8 weeks, monthly suffices. Don't guess; measure.

**Windows:** rolling window long enough to cover seasonality (12+ months) with
recency upweighting, *not* "last month only" (you'd forget winter in July).
Emphasize: labels here are fast and unbiased — so unlike fraud, aggressive
automation is safe.

**Triggers on top of cadence:** (a) input drift alarms (new-city traffic share,
courier-supply shifts), (b) online MAE by segment crossing thresholds — with
1-hour labels you can monitor *true* performance nearly live, (c) event
calendar (holidays, storms — consider a fast contextual correction layer
rather than emergency retrains: a per-city online bias term $\hat y' = \hat y +
b_{\text{city,hour}}$ updated hourly from the freshest errors absorbs shocks
faster than any retrain).

**Safe rollout, automated but gated:** every retrain passes (1) offline gate:
MAE on the freshest held-out weeks must beat incumbent overall *and not regress
any major segment*, (2) shadow for a day (prediction diff distribution vs
incumbent — a 20-minute median diff means a data bug, halt), (3) canary 5% by
*city* (ETA has marketplace effects; randomize by geography), (4) full rollout
with one-click rollback via the registry. Also monitor **calibration of the
promise, not just MAE**: users experience lateness asymmetrically — being 10
min late is much worse than 10 min early — so track quantile loss / P(late) and
consider predicting a distribution and quoting its ~70th percentile rather than
the mean.
</details>

---

## 8. Your fraud model melts down on Black Friday — false positives spike 5×

It's the biggest sales day of the year. Alerting shows the decline rate tripled
and support is flooded with "my card was rejected" complaints, but confirmed
fraud (from proxy labels) looks flat. What happened, and what do you do — in
the next hour, day, and month?

<details><summary>Sample answer</summary>

**What happened — reason it from the features:** Black Friday shifts *every*
velocity and anomaly feature toward what training data calls fraud: users make
many rapid purchases (velocity counters spike), buy atypically large amounts
(amount-anomaly features fire), shop at 2am from new merchants, ship to
relatives. This is **data drift into the model's fraud region** — $P(x)$ moved
en masse toward high-score territory while actual $P(y|x)$ *on this day* is
different (a 6-transactions-per-hour card on Black Friday is normal). The model
is behaving exactly as trained; the training distribution just doesn't contain
"the whole world behaves like a card tester day." If last year's Black Friday
wasn't in (or was drowned out of) the training window, this was predictable.

**Next hour (config, not code):** shift decision thresholds — raise the decline
threshold and widen the step-up band, so borderline scores get a 3DS challenge
instead of a decline (friction is vastly cheaper than lost checkout today; this
is why thresholds live in config with fraud-ops ownership). Add a temporary
rule/context adjustment discounting pure-velocity signals for
established-account + verified-device traffic. Watch the proxy-fraud rate as
you loosen — you're deliberately moving along the precision/recall curve, and
attackers *also* know it's Black Friday, so loosen surgically (tenured
accounts), not globally.

**Next day:** monitor the loosened posture's fraud leakage via fast proxies
(issuer alerts, reports); prepare a day-after tightening (fraudsters ride the
post-loosening window); feed the weekend's review-team verdicts back as labels.

**Next month (make it structural):** add calendar/event features
(is_promo_period, merchant-level promo flags) and *relative* velocity features
(this card's rate ÷ population rate *today*) — normalizing by the crowd makes
the feature drift-invariant: on Black Friday everyone's raw velocity rises, so
the ratio stays informative; ensure past holiday periods are represented (even
upweighted) in the training window; add a pre-holiday load-and-drift rehearsal
(replay last year's peak through the current model in shadow) to the runbook;
and add a drift alarm on *score distribution* — mean predicted fraud
probability doubling in an hour should page before support tickets do.
</details>

---

## 9. Design an internal-docs Q&A assistant (RAG) for a 5,000-person company

Sources: wiki (~200K pages), Slack archives, Google Drive, an engineering
handbook. Employees ask questions in Slack; the bot answers. Design it, with
special attention to evaluation and to what will actually go wrong.

<details><summary>Sample answer</summary>

**Why RAG (derive, don't assert):** the corpus is ~200K docs and changes daily
— fine-tuning knowledge in is stale on arrival and can't cite; pasting the
corpus into context is $10^7$+ tokens — impossible and unaffordable. So:
retrieve the few relevant chunks per question, generate from them with
citations. Knowledge updates = index updates (minutes).

**Ingestion (where most real-world failure lives):** per-source connectors with
**permission awareness** — this is the #1 landmine: Drive docs and Slack
channels have ACLs, and a bot that leaks the M&A folder into anyone's answer is
a company-level incident. Store ACLs with each chunk; filter retrieval by the
*asker's* permissions at query time (not at index time — permissions change).
Chunking: 300–500 tokens on structural boundaries, headers prepended
("Handbook → Oncall → Escalation policy: ..."), 15% overlap. Freshness
metadata on every chunk (staleness is endemic in wikis — prefer recent, show
last-updated in answers, and let "this doc is outdated" feedback flow to doc
owners).

**Retrieval:** hybrid — BM25 (employees search exact tokens: service names,
error codes, acronyms embeddings mangle) + dense embeddings, union of ~50 →
cross-encoder rerank → top 5–8 chunks. Slack questions are messy
("anyone know how the thing w/ oncall payouts works?") → a cheap query-rewrite
step (LLM turns it into a clean search query) pays for itself.

**Generation:** answer *only* from context, cite chunks (linked), abstain
explicitly when the context doesn't contain the answer — for an internal tool,
a confident wrong answer about, say, expense policy is worse than "I couldn't
find this; try #finance." Treat retrieved text as data, not instructions
(prompt-injection via a wiki page is a real vector).

**Evaluation (decomposed):** build a ~300-question golden set from real Slack
questions with links to the gold pages (get team leads to contribute). Measure
(1) retrieval recall@k against gold pages — check *this first*; it's where
most quality is lost and it's cheap to iterate on; (2) faithfulness — sampled
human review + an LLM judge validated against those humans; (3) abstention
correctness on deliberately unanswerable questions; (4) online: thumbs,
re-ask rate, escalation-to-human-channel rate, and weekly review of a random
answer sample. **Scale reality:** 5,000 employees × a few questions/day ≈
$10^4$/day ≈ well under 1 QPS — this is not a throughput problem; spend the
entire engineering budget on ingestion quality, permissions, and evaluation,
not on serving infrastructure. Saying that out loud is the senior move.
</details>

---

## 10. A/B testing a new ranker: design the experiment and its pitfalls

Your new feed ranker beat the incumbent offline. Design the online experiment:
randomization unit, size/duration reasoning, metrics, and the ways this
specific experiment can silently lie to you.

<details><summary>Sample answer</summary>

**Randomization unit: user, not request.** Request-level randomization shows
one user both rankers — their behavior in one arm contaminates the other (and
personalization state leaks across). Hash user-id into arms; keep assignment
sticky for the whole experiment.

**Size/duration from power, not vibes:** to detect a relative lift $\delta$ in
a metric with per-user standard deviation $\sigma$, you need roughly $n \approx
16\,\sigma^2/(\delta\bar{x})^2$ per arm (80% power, 5% significance,
two-sided). Engagement metrics are extremely high-variance across users
(heavy-tailed usage), so detecting a 0.5% lift routinely needs millions of
users for 2+ weeks. State the practical consequences: run *at least* two full
weeks (weekday/weekend cycles), don't peek-and-stop at the first significant
day (sequential peeking inflates false positives — use fixed horizon or proper
sequential tests), and reduce variance where possible (CUPED/regression
adjustment with pre-experiment engagement as covariate — often cuts required
sample by 30–50%).

**Metrics:** one pre-registered primary (e.g., satisfied-watch-time/user/day);
guardrails: report rate, hide rate, latency, revenue, creator-side
concentration; plus diagnostics (CTR, dwell) that inform but don't decide.
Decide the launch rule *before* looking.

**Ways this experiment lies:**
- **Novelty/primacy effects:** new ranking surfaces different content; users
  click from novelty (inflates early lift) or are disoriented (masks real
  lift). Mitigation: run longer; compare week-1 vs week-2 effects; watch
  new-user cohorts (no habits to disrupt) separately.
- **Interference between arms:** both arms share one ecosystem. Treatment users
  engaging more with certain creators changes those creators' behavior
  (posting more), which leaks into control — biasing effects toward zero. For
  ecosystem-level questions, use cluster/geo randomization or
  creator-side experiments.
- **Training-data contamination:** if the model retrains during the experiment
  on data generated mostly by control-arm policy (95% of traffic), the
  treatment model is trained off-policy relative to its own serving — decide
  and document the retraining policy during the test.
- **Segment masking:** a flat average can hide +5% for tenured users and −5%
  for new users; pre-register key segments.
- **Sample ratio mismatch:** if arm sizes deviate from the planned split
  (crash-on-treatment bias, bot filtering asymmetry), the randomization is
  broken — check SRM *first*, it invalidates everything downstream.

Roll out via canary → 50/50 → holdback (keep 1–5% on the old ranker for weeks
post-launch to measure long-term effects the two-week test can't see).
</details>
