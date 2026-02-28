# RegimeFrontierLab

Simulation toolkit for mapping **safety–utility–flourishing** tradeoff frontiers across environment regimes.

## What this is
RegimeFrontierLab helps you:
- map Pareto/frontier structure per environment,
- compare global vs regime-specific policy families,
- test representation patches for abstract/prosocial corpora,
- enforce explicit safety-envelope checks while optimizing flourishing signals.

## What this is not
- Not a universal controller.
- Not a deployment guarantee.
- All findings are **model-bounded** and corpus/representation dependent.

## Repository layout
- `src/sim_memetic_gelation.py` — core simulator
- `scripts/` — reproducible run scripts (E1/E2 and follow-ups)
- `docs/Frontier_Regime_Hypothesis_v1.md` — current hypothesis and experiment plan
- `results/` — sample outputs from recent runs (summaries, frontiers, CIs)
- `data/demo/demo_corpus.txt` — tiny demo corpus for smoke tests

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

Run a smoke test:
```bash
python src/sim_memetic_gelation.py --seed 42 --iters 2000 --corpus-file data/demo/demo_corpus.txt --output-json smoke.json
```

Run full packaged follow-up (from this repo root):
```bash
python scripts/run_targeted_followup_gfs_e1_e2_v1.py
```

## Current status (v0.1.0 baseline)
- E1 supports **regime-conditioned policy banks** over one global policy.
- E2 confirmatory ablation shows strong coop-density uplift directionally on patched representation, with safety non-worse in current runs.
- Frontier width and hypervolume vary sharply by substrate type.

## Reproducibility
Each run should emit:
- raw metrics tables,
- summary tables (with CIs where relevant),
- frontier/Pareto tables,
- safety envelope checks,
- receipts/manifests.

## License
MIT (see `LICENSE`).
