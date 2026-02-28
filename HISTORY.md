# HISTORY.md — RegimeFrontierLab Development Log

This file tracks the major phases, architectural decisions, empirical learnings, and pivots in the RegimeFrontierLab project (memetic gelation / symbiogenesis simulator + AI safety transfer).

## 2026-02-xx — Project Initialization & Symbiogenesis Core
- Started with symbiogenesis-inspired toy model (`mysterium/sim_memetic_gelation.py`)
- Core mechanics: token/motif soup, fusion/merger dynamics, graph-based LCC tracking, resource caps, replication
- Initial metrics: `lcc_fraction`, `type_entropy`, `stable_mass_fraction`
- Early insight: Fusion (not mutation) is the dominant driver of sustained connectivity (Lesson #1)

## 2026-02-xx — Metric & Controller Refinements
- Added nontrivial persistence metric (`stable_nontrivial_mass_fraction`) after monomer artifact discovered (Lesson #5)
- Introduced hub/risk telemetry: `hub_gini`, `hub_top5_degree_share`, `risk_mass_fraction`, `risk_lcc_fraction`
- Implemented first intervention ladder: baseline, soft, targeted prune modes
- Pilot v1 results: Targeted prune strongly reduced risk proxy with minimal LCC movement

## 2026-02-xx — Universality & External Eval
- U1: Replication matrix across configs
- U2: Intervention invariance across internal text datasets
- HF adversarial eval v1: Prompt-injection corpora (3 datasets, 2000 rows each)
- Targeted variants drove `risk_mass_fraction` → 0.0 with small LCC shift
- Semantic proxy weak → flagged as bottleneck

## 2026-02-xx — Flourishing Pivot & Positive Control
- Shifted focus to positive motif promotion (coop_score boost during replication)
- Tested gentle promotion, staged generation, fusion affinity modulation
- Repeated clean negatives → revealed framing mismatch: incompatible objectives on adversarial corpora

## 2026-02-xx — Pareto Frontier Pivot (Current Regime)
- Abandoned binary gates → adopted Pareto frontier analysis (risk↓ vs utility retention vs flourishing retention)
- Dual-environment eval: adversarial (safety stress) vs prosocial/neutral (flourishing capacity)
- Policy families: safety-first, balanced, flourish-first
- Key result: No global optimum; real tradeoffs
- Library/narrative env most tractable for flourishing (highest HV 0.1043)
- GFS env near-degenerate (HV 0.0109)
- Safety-selected policy transfers well to narrative flourishing (+0.3508 Δflourish) but weakly to GFS (+0.0742)
- Latest: Frontier push on patched GFS corpus → +24.6% HV uplift while safety-nonworse

## Lessons (Model-Bounded)
1. Fusion drives connectivity; no-fusion regimes collapse
2. Source composition strongly shapes attractor / diversity dynamics
3. Real tradeoffs exist — no universal improvement lever
4. Narrative sources scaffold bridging better than technical/reflective
5. Metrics matter — wrong persistence metric hides true effects
6. Claims remain simulator + corpora specific; no universal biological/AI proof

## Next Horizon (as of 2026-02-27)
- Contextual gating across policy families
- Representation refinement for GFS-style abstract flourishing
- Deeper transfer testing (safety → prosocial, prosocial → safety)

This log will be updated after each major cycle or insight.
