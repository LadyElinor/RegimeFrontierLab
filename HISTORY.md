# HISTORY.md — RegimeFrontierLab Development Log

This file tracks major phases, architectural decisions, empirical learnings, and pivots in RegimeFrontierLab (memetic gelation / symbiogenesis simulator + AI safety transfer).

## Early Feb 2026 (Week 1) — Project Initialization & Symbiogenesis Core
- Started with a symbiogenesis-inspired toy model (`src/sim_memetic_gelation.py`).
- Core mechanics: token/motif soup, fusion/merge dynamics, graph-based LCC tracking, resource caps, replication.
- Initial metrics: `lcc_fraction`, `type_entropy`, `stable_mass_fraction`.
- Early insight: fusion (not mutation) is the dominant driver of sustained connectivity (Lesson #1).
- Reference scripts/results: early baselines in `scripts/` and `results/`.

## Mid Feb 2026 (Week 2) — Metric & Controller Refinements
- Added nontrivial persistence metric: `stable_nontrivial_mass_fraction` after monomer artifact discovery (Lesson #5).
- Introduced hub/risk telemetry: `hub_gini`, `hub_top5_degree_share`, `risk_mass_fraction`, `risk_lcc_fraction`.
- Implemented first intervention ladder: baseline, soft, targeted prune modes.
- Pilot v1 finding: targeted prune strongly reduced risk proxy with minimal LCC movement.
- Reference: `scripts/run_followup_gfs_patch_ab_v1.py`.

## Mid–Late Feb 2026 (Week 3) — Universality & External Evaluation
- **U1** (Universality Test 1): replication matrix across controller configs.
- **U2** (Universality Test 2): intervention invariance across internal text datasets.
- **HF adversarial eval v1** (HuggingFace prompt-injection corpora): 3 datasets, ~2000 rows each.
- Targeted variants drove `risk_mass_fraction` toward 0.0 with small LCC shift.
- Identified bottleneck: semantic risk proxy sensitivity/coverage lagged lexical controls.
- Reference: `scripts/run_targeted_followup_gfs_e1_e2_v1.py` (later consolidated package).

## Late Feb 2026 (Week 4) — Flourishing Pivot & Positive Control
- Shifted from pure suppression to positive motif promotion (coop-score-weighted replication).
- Tested: gentle promotion, staged generation, fusion affinity modulation.
- Repeated clean negatives highlighted a framing mismatch:
  - adversarial corpora optimize for suppression robustness,
  - flourishing objective requires merge-compatible prosocial motif scaffolding.
- Conclusion: same controller could not simultaneously maximize both regimes without explicit tradeoff handling.
- References: `scripts/run_followup_gfs_frontier_push_v1.py`, `results/e2/*`.

## Late Feb 2026 (Week 4) — Pareto Frontier Pivot (Current Regime)
- Replaced binary pass/fail framing with Pareto frontier analysis (risk↓ vs utility retention vs flourishing retention).
- Evaluated dual environments: adversarial (safety stress) vs prosocial/neutral (flourishing capacity).
- Policy families compared: safety-first, balanced, flourish-first.
- Key result: no single global optimum; real, unavoidable tradeoffs.
- Library/narrative environment: highest tractability for flourishing (HV 0.1043).
- GFS environment: near-degenerate frontier (HV 0.0109) under current representation.
- Transfer is partial: safety-selected policy yielded strong narrative flourishing gains (+0.3508 Δflourish) but weak GFS lift (+0.0742).
- Frontier push on patched GFS corpus delivered +24.6% HV uplift while safety-nonworse.
- References: `results/e1/summary.csv`, `results/e1/frontier_pareto.csv`, `results/e1/plan_comparison.csv`, `results/e2/summary_with_ci.csv`.

## Lessons (Model-Bounded)
1. Fusion drives connectivity; no-fusion regimes collapse.
2. Source composition strongly shapes attractor/diversity dynamics.
3. Real tradeoffs exist; no universal improvement lever appears in current model.
4. Narrative sources scaffold cooperative bridging better than technical/reflective sources.
5. Metrics matter: incorrect persistence proxies can hide true effects.
6. Claims remain simulator + corpus specific; no universal biological/AI proof is implied.

## Next Horizon (as of 2026-02-27)
- Contextual gating across policy families.
- Representation refinement for GFS-style abstract flourishing.
- Deeper transfer testing (safety→prosocial, prosocial→safety).
- Larger-seed confirmatory passes with CIs before hard recommendations.

This log is updated after each major cycle or methodological pivot.
