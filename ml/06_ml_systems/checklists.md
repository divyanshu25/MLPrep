# ML System Design - Compact Reference

## The 10-step framework (one page)

| # | Step | Say / decide | Failure it prevents |
|---|------|--------------|---------------------|
| 1 | **Product goal** | Who's the user, what's the business metric, what surface, what scale ($10^?$ users/items, QPS), freshness needs | Optimizing the wrong thing brilliantly |
| 2 | **ML formulation** | Input → output → *where labels come from*; unit of prediction; label horizon; "is a rule enough?" | ML without a viable label; ML where a rule suffices |
| 3 | **Metrics** | Offline (AUC/NDCG/recall@k) as filter; online (CTR/watch/conversion) as launch gate; guardrails + counter-metric | Trusting offline gains that don't transfer; Goodharting |
| 4 | **Data** | Sources, sampling (negative downsampling + recalibration $p = \frac{p'}{p'+(1-p')/w}$), feedback loops, exploration slice + propensity logging | Silent bias; model distilling its predecessor |
| 5 | **Features** | User / item / context / cross; batch vs real-time; "computable at serving time, from data strictly before $t$, with the same code?" | Training/serving skew; point-in-time leakage |
| 6 | **Model** | Baseline → GBDT → NN, each step justified by a measured gap; complexity has ops cost | Unbenchmarked fancy model; unserveable model |
| 7 | **Training pipeline** | Label join (wait the horizon), **temporal split**, eval gate vs prod model, registry entry | Future leakage via random splits; unreproducible models |
| 8 | **Serving** | Latency budget table; batch vs online inference; batched feature multi-get; batched forward pass; timeout-and-default (logged) | Designs that can't physically serve; sequential-lookup death |
| 9 | **Monitoring** | Inputs (nulls, PSI), predictions (mean/histogram), delayed-label outcomes by cohort, per-segment slices | Silent degradation for weeks |
| 10 | **Iteration** | Retrain cadence + triggers; shadow → canary → A/B; staged roadmap v1/v2/v3 | Frozen decaying model; big-bang risky launches |

Time budget in a 45-min interview: ~5 goal/formulation, 5 metrics, 10
data/features, 10 model/training, 10 serving/scale math, 5 monitoring/iteration.

---

## Pre-interview cram list

- [ ] Recite the 10 steps cold, with the *why* for each.
- [ ] The funnel math: why $10^9$ items forces retrieval → ranking → re-ranking.
- [ ] Two-tower: the constraint ($f(u)\cdot g(i)$, no early interaction), what it
      buys (ANN over precomputed item vectors), what it costs (no crosses).
- [ ] Cross-encoder vs bi-encoder: accuracy vs precomputability.
- [ ] Position bias fixes: position-as-feature (frozen at serving), IPW,
      examination × relevance factoring.
- [ ] Negative downsampling recalibration formula.
- [ ] Data drift vs concept drift: $P(x)$ vs $P(y|x)$, one example each.
- [ ] Shadow / canary / A/B: what each uniquely proves.
- [ ] The "offline up, online flat" debug order: skew → metric locality →
      experiment validity → funnel recall.
- [ ] Fraud triad: imbalance (PR-AUC, money-weighted recall), asymmetric costs
      (calibrated $p$ + EV decision layer + step-up tier), adversarial drift
      (rules layer, fast retrain, hard-to-fake features).
- [ ] RAG: derive retrieval from context limits + staleness; chunking rationale;
      hybrid retrieval; evaluate retrieval (recall@k) separately from generation
      (faithfulness).
- [ ] Linear scaling rule: batch $\times k$ → variance $/k$ → LR $\times k$,
      with warmup caveat.
- [ ] Latency constants: DC round trip ~0.5ms, KV read ~1ms, GBDT ~0.2ms/row,
      ANN over $10^9$ ~10ms, LLM ~20–100 tok/s. Day ≈ $10^5$s.
- [ ] One full worked estimate (feed ranking cores; 7B LLM serving) done on
      paper the night before.

---

## Red flags interviewers listen for

1. **Designing before asking** - no product goal, no scale numbers.
2. **Jumping to the fanciest model** - no baseline, no justification ladder.
3. **"Optimize accuracy"** on an imbalanced problem, or any single unguarded
   metric.
4. **No answer for where labels come from**, or labels that contain the future.
5. **Random train/test split** on temporal data.
6. **Ignoring the offline/online gap** - treating offline AUC as the launch
   criterion.
7. **Unserveable designs** - big model over the whole corpus per request; never
   mentions latency.
8. **No monitoring story** - ships the model and walks away; can't say what
   pages them when an upstream field goes null.
9. **Name-dropping without mechanism** - "use a feature store / two-tower /
   vector DB" but can't derive the problem it solves.
10. **No feedback-loop awareness** - trains on logs the old model produced,
    never mentions position/selection bias or exploration.
11. **Big-bang launch** - 0% → 100% with no shadow/canary/rollback.
12. **Can't do arithmetic** - refuses or fumbles back-of-envelope QPS/memory
    estimates.

---

## Glossary (~30 production-ML terms, one line each)

| Term | One-liner |
|---|---|
| **Training/serving skew** | Any systematic difference between features/preprocessing at training vs serving time; classic cause of "great offline, flat online." |
| **Point-in-time correctness** | Training features computed only from data timestamped strictly before the example's prediction moment. |
| **As-of join** | Join that picks each feature's latest value *before* the label event's timestamp; requires feature history. |
| **Feature store** | Single feature definition materialized twice - offline (point-in-time training sets) and online (low-latency serving) - to guarantee consistency. |
| **Data drift (covariate shift)** | $P(x)$ changes while $P(y\|x)$ holds; inputs move to thin-data regions. |
| **Concept drift** | $P(y\|x)$ changes; the same input now means something else (fraud tactics, pandemic behavior). |
| **Label leakage** | A feature that (partly) encodes the label, usually via future information; inflates offline metrics. |
| **Position bias** | Higher slots get clicked partly for being higher; naive click training distills the old ranker. |
| **Selection bias ("clicked because shown")** | Labels exist only for items the old policy chose to show; unshown items are unlabeled, not-at-random. |
| **Propensity / IPW** | Logged probability an item was shown; weighting by its inverse de-biases learning from logged feedback. |
| **Exploration traffic** | Small randomized slice of serving used to collect counterfactual labels and break feedback loops. |
| **Feedback loop** | Model's outputs shape the data it later trains on, ossifying its own past decisions. |
| **Candidate generation / retrieval** | Cheap first funnel stage cutting $10^9$ items to $10^3$ using precomputable scores. |
| **Two-tower (bi-encoder)** | Score constrained to $f(u)\cdot g(i)$ so item vectors precompute and ANN handles retrieval; no cross features. |
| **ANN index** | Approximate nearest-neighbor structure answering top-k by similarity over $10^9$ vectors in ~ms. |
| **Cross-encoder** | Model that reads query/user and item together (full interaction); accurate but must run per pair - final-stage only. |
| **Re-ranking** | Slate-level pass after pointwise ranking: diversity, dedupe, freshness, business rules. |
| **In-batch negatives** | Contrastive training trick: other positives in the batch serve as free negatives (needs popularity correction). |
| **Negative downsampling recalibration** | After downsampling negatives by $w$, map scores back: $p = p'/(p'+(1-p')/w)$. |
| **Calibration** | Predicted probabilities match observed frequencies ($\sum\hat p / \sum y \approx 1$); required when decisions use expected value. |
| **NDCG** | Ranking metric: relevance gain discounted by $1/\log_2(\text{rank}+1)$, normalized by the ideal ordering. |
| **PR-AUC** | Area under precision-recall; the imbalance-honest alternative to ROC-AUC for rare positives. |
| **Interleaving** | Online ranker comparison mixing two rankers' results in one list; more sensitive than A/B per unit traffic. |
| **Shadow deployment** | New model scores live traffic but actions are never taken; proves latency/feature-availability, not quality. |
| **Canary** | Small real-traffic slice on the new model with guardrail monitoring and instant rollback. |
| **A/B test** | User-randomized, power-analyzed experiment; the only stage that measures product-metric improvement. |
| **Model registry** | Versioned catalog binding artifact + code + data + features + eval + stage; enables rollback and audit. |
| **Batch vs online inference** | Precompute predictions to a KV store (enumerable inputs, staleness OK) vs compute at request time (request-dependent inputs). |
| **PSI** | Population stability index; per-feature train-vs-live distribution distance, conventional alert ~0.2. |
| **Label horizon / delay** | Time until ground truth exists (7-day conversion, 60-day chargeback); bounds training freshness and forces proxy labels. |
| **Data parallelism** | Replicate model, shard the batch, all-reduce (average) gradients; mathematically = large-batch SGD. |
| **Model parallelism (pipeline/tensor)** | Split the model itself across devices - by layers (pipeline, micro-batched) or within layers (tensor). |
| **Linear scaling rule** | Batch $\times k$ ⇒ LR $\times k$ (gradient variance drops by $k$); needs warmup, breaks at very large $k$. |
| **KV cache** | Stored attention keys/values for generated tokens so each new token avoids recomputing the past; makes LLM serving memory-bound. |
| **RAG** | Retrieve relevant chunks, paste into the prompt, generate a grounded answer; knowledge updates become index updates. |
