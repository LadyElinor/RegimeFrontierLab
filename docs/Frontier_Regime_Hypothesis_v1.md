# Frontier Regime Hypothesis v1

Date: 2026-02-27
Project: mysterium
Status: Working hypothesis (model-bounded)

## Thesis
Shared flourishing persistence is strongly conditional on the **idea substrate**. The system is not failing to find a universal optimum; it is revealing real tradeoffs driven by coupling among:
1. Corpus motif structure,
2. Representation/embedding geometry,
3. Merge-compatibility dynamics under safety gating.

Implication: optimize **policy families per environment class** rather than forcing one controller across all environments.

## Observational anchors (from current runs)
- Frontiers are non-degenerate in safety stress environments (multiple nondominated points).
- Narrative/library-like corpora show tractable flourishing landscapes with materially larger frontier area and uplift potential.
- GFS-style reflective/prosocial corpora remain near-degenerate under current representation.
- Transfer is partial: safety-selected policies can significantly improve narrative flourishing while only weakly lifting GFS flourishing.
- Latest patched-GFS frontier push improved frontier proxy (~+24.6%) while staying safety-nonworse, indicating local leverage exists even in constrained substrates.

## Regime framing
### Regime A: Narrative-rich substrate
- Motifs are naturally merge-compatible for cooperative bridging.
- Promotion and affinity shaping can move flourish substantially without violating safety envelope.
- Expect broader Pareto frontiers and larger hypervolume gains.

### Regime B: Reflective/abstract-social substrate (GFS-like)
- Motifs are semantically prosocial but weakly merge-compatible in current encoding.
- Controller tuning alone gives limited movement.
- Gains require representation/corpus shaping that increases compatible cooperative motif density.

## Testable predictions
1. **Policy specialization wins:** per-regime tuned policies outperform any single global policy on combined objective (risk↓, utility↔, flourish↑).
2. **Representation patch sensitivity:** adding anchored abstract-social prototypes increases coop-score density and motif-length survival in GFS-like environments more than pure knob tuning.
3. **Frontier shape divergence:** narrative environments retain wider nondominated sets under stricter safety gates; GFS frontiers collapse faster unless representation is patched.
4. **Transfer asymmetry persists:** safety-trained policy transfers better to narrative than reflective/abstract environments unless merge-compatibility is explicitly repaired.

## Experiment plan (next 3 runs)

### E1 — Regime-conditioned policy bank (A/B/C)
Goal: compare a global controller vs per-regime controllers.
- A: single global best policy (current baseline for transfer).
- B: narrative-optimized policy family.
- C: GFS-optimized policy family (patched representation + constrained promotion).
Metrics:
- Per-env hypervolume proxy,
- Frontier size,
- Safety envelope deltas vs anchor,
- Aggregate weighted score across regimes.
Success condition:
- B+C strictly improves aggregate flourish/hypervolume at non-worse safety vs A.

### E2 — Representation ablation ladder on GFS
Goal: isolate what part of patching matters.
Arms:
- A0: original GFS corpus,
- A1: minimal anchored abstract-social prototypes,
- A2: expanded prototypes + bridge templates,
- A3: expanded + anti-fragmentation lexical anchors.
Keep controller fixed (same prune/promotion settings).
Metrics:
- Coop-score density,
- Motif-length survival,
- Flourishing persistence,
- Semantic-risk mass.
Success condition:
- Monotonic or near-monotonic improvement in flourish metrics from A0→A2 with safety non-worse.

### E3 — Safety-gate phase diagram by regime
Goal: map robustness of frontier width to gate strictness.
Sweep:
- risk_penalty_scale ∈ {0.0, 0.5, 1.0, 1.5}
- targeted_prune_fraction ∈ {0.04, 0.08, 0.12}
Run separately for narrative and GFS-patched corpora.
Metrics:
- Frontier size,
- Hypervolume proxy,
- Safety envelope pass rate,
- Utility retention.
Success condition:
- Identify stable operating bands per regime (not single-point tuning).

## Operational policy recommendation (v1)
- Maintain **separate frontier maps** by environment class.
- Use **gating policy families** selected by detected substrate type.
- Treat cross-regime transfer as a bonus, not a design assumption.
- Prioritize representation-quality interventions for GFS-like regimes before further aggressive controller tuning.

## Risks / caveats
- All claims are simulator-bounded and representation-dependent.
- Hypervolume proxy is useful but not complete; pair with envelope and persistence metrics.
- Avoid overfitting to current corpora token distributions; keep holdout seeds and out-of-source checks.

## Immediate next action
Implement E1 first (highest leverage): produce one comparative table with per-regime policy winners and safety-gated aggregate score.
