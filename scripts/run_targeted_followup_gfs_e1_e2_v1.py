import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def dominates(a, b):
    no_worse = (a["risk"] <= b["risk"]) and (a["utility"] >= b["utility"]) and (a["flourish"] >= b["flourish"])
    strictly = (a["risk"] < b["risk"]) or (a["utility"] > b["utility"]) or (a["flourish"] > b["flourish"])
    return no_worse and strictly


def hv_point(risk, utility, flourish):
    return max(0.0, min(1.0, 1.0 - risk)) * max(0.0, min(1.0, utility)) * max(0.0, min(1.0, flourish))


def parse_runs(run_glob, regime, source_tag):
    rows = []
    receipts = []
    for fp in sorted(run_glob):
        data = json.loads(fp.read_text(encoding="utf-8"))
        final = data.get("final", {})
        stem = fp.stem
        prefix = f"{source_tag}_"
        tail = stem[len(prefix):] if stem.startswith(prefix) else stem
        policy, seed_s = tail.rsplit("_seed", 1)
        seed = int(seed_s)

        utility = 0.5 * float(final.get("lcc_fraction", 0.0)) + 0.5 * float(final.get("stable_nontrivial_mass_fraction", 0.0))
        coop_density = mean(
            [
                float(final.get("score_bins", {}).get("1_3", {}).get("coop", {}).get("mean", 0.0)),
                float(final.get("score_bins", {}).get("4_6", {}).get("coop", {}).get("mean", 0.0)),
                float(final.get("score_bins", {}).get("7_plus", {}).get("coop", {}).get("mean", 0.0)),
            ]
        )
        rows.append(
            {
                "regime": regime,
                "source": source_tag,
                "policy": policy,
                "seed": seed,
                "risk": float(final.get("risk_mass_fraction", 0.0)),
                "semantic_risk": float(final.get("semantic_risk_mass_fraction", 0.0)),
                "utility": utility,
                "flourish": float(final.get("flourishing_persistence", 0.0)),
                "coop_score_density": coop_density,
                "motif_length_survival": float(final.get("stable_nontrivial_mass_fraction", 0.0)),
            }
        )
        receipts.append({"source_file": str(fp)})
    return rows, receipts


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["regime"], r["source"], r["policy"])].append(r)

    out = []
    for (regime, source, policy), rs in sorted(grouped.items()):
        out.append(
            {
                "regime": regime,
                "source": source,
                "policy": policy,
                "n": len(rs),
                "risk_mean": mean([x["risk"] for x in rs]),
                "semantic_risk_mean": mean([x["semantic_risk"] for x in rs]),
                "utility_mean": mean([x["utility"] for x in rs]),
                "flourish_mean": mean([x["flourish"] for x in rs]),
                "coop_score_density_mean": mean([x["coop_score_density"] for x in rs]),
                "motif_length_survival_mean": mean([x["motif_length_survival"] for x in rs]),
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def compute_pareto(summary_rows):
    pareto = []
    by_regime = defaultdict(list)
    for r in summary_rows:
        by_regime[r["regime"]].append(r)

    for regime, rs in by_regime.items():
        for i, a in enumerate(rs):
            dom = False
            for j, b in enumerate(rs):
                if i == j:
                    continue
                if dominates(
                    {"risk": b["risk_mean"], "utility": b["utility_mean"], "flourish": b["flourish_mean"]},
                    {"risk": a["risk_mean"], "utility": a["utility_mean"], "flourish": a["flourish_mean"]},
                ):
                    dom = True
                    break
            pareto.append(
                {
                    "regime": regime,
                    "source": a["source"],
                    "policy": a["policy"],
                    "risk_mean": a["risk_mean"],
                    "utility_mean": a["utility_mean"],
                    "flourish_mean": a["flourish_mean"],
                    "pareto_nondominated": int(not dom),
                }
            )
    return pareto


def e1(base: Path, out: Path):
    runs_u3 = base / "u3_pareto_frontier_v1" / "runs"
    runs_step4 = base / "targeted_followup_gfs_v1" / "step4_frontier_push_v1" / "runs"

    lib_rows, lib_receipts = parse_runs(runs_u3.glob("flourish_library_*.json"), "narrative", "flourish_library")
    gfs_rows, gfs_receipts = parse_runs(runs_step4.glob("*.json"), "gfs_patched", "")

    raw = lib_rows + gfs_rows
    summary = summarize(raw)
    pareto = compute_pareto(summary)

    # Safety gates vs anchor within each regime/source
    by_regime = defaultdict(list)
    for r in summary:
        by_regime[r["regime"]].append(r)

    envelope = []
    selected = {}
    for regime, rs in by_regime.items():
        anchor = next(x for x in rs if x["policy"] == "anchor_baseline")
        gated = [
            x
            for x in rs
            if (x["risk_mean"] <= anchor["risk_mean"] + 1e-12)
            and (x["semantic_risk_mean"] <= anchor["semantic_risk_mean"] + 1e-12)
        ]
        for r in rs:
            envelope.append(
                {
                    "regime": regime,
                    "policy": r["policy"],
                    "delta_risk_vs_anchor": r["risk_mean"] - anchor["risk_mean"],
                    "delta_semantic_risk_vs_anchor": r["semantic_risk_mean"] - anchor["semantic_risk_mean"],
                    "delta_utility_vs_anchor": r["utility_mean"] - anchor["utility_mean"],
                    "delta_flourish_vs_anchor": r["flourish_mean"] - anchor["flourish_mean"],
                    "safety_nonworse": int(r in gated),
                }
            )
        # maximize flourish among safety-gated policies
        selected[regime] = max(gated, key=lambda x: x["flourish_mean"]) if gated else anchor

    # Compare global A vs regime-conditioned B+C
    global_narr = next(x for x in by_regime["narrative"] if x["policy"] == "anchor_baseline")
    global_gfs = next(x for x in by_regime["gfs_patched"] if x["policy"] == "anchor_baseline")

    bank_narr = selected["narrative"]
    bank_gfs = selected["gfs_patched"]

    comparison = [
        {
            "plan": "A_global_anchor",
            "narrative_policy": global_narr["policy"],
            "gfs_policy": global_gfs["policy"],
            "narrative_hv_point": hv_point(global_narr["risk_mean"], global_narr["utility_mean"], global_narr["flourish_mean"]),
            "gfs_hv_point": hv_point(global_gfs["risk_mean"], global_gfs["utility_mean"], global_gfs["flourish_mean"]),
            "aggregate_hv_point_mean": mean(
                [
                    hv_point(global_narr["risk_mean"], global_narr["utility_mean"], global_narr["flourish_mean"]),
                    hv_point(global_gfs["risk_mean"], global_gfs["utility_mean"], global_gfs["flourish_mean"]),
                ]
            ),
            "aggregate_flourish_mean": mean([global_narr["flourish_mean"], global_gfs["flourish_mean"]]),
            "aggregate_utility_mean": mean([global_narr["utility_mean"], global_gfs["utility_mean"]]),
        },
        {
            "plan": "BplusC_regime_conditioned",
            "narrative_policy": bank_narr["policy"],
            "gfs_policy": bank_gfs["policy"],
            "narrative_hv_point": hv_point(bank_narr["risk_mean"], bank_narr["utility_mean"], bank_narr["flourish_mean"]),
            "gfs_hv_point": hv_point(bank_gfs["risk_mean"], bank_gfs["utility_mean"], bank_gfs["flourish_mean"]),
            "aggregate_hv_point_mean": mean(
                [
                    hv_point(bank_narr["risk_mean"], bank_narr["utility_mean"], bank_narr["flourish_mean"]),
                    hv_point(bank_gfs["risk_mean"], bank_gfs["utility_mean"], bank_gfs["flourish_mean"]),
                ]
            ),
            "aggregate_flourish_mean": mean([bank_narr["flourish_mean"], bank_gfs["flourish_mean"]]),
            "aggregate_utility_mean": mean([bank_narr["utility_mean"], bank_gfs["utility_mean"]]),
        },
    ]

    e1_dir = out / "e1_regime_conditioned_policy_bank_v1"
    e1_dir.mkdir(parents=True, exist_ok=True)
    write_csv(e1_dir / "raw_metrics.csv", raw)
    write_csv(e1_dir / "summary.csv", summary)
    write_csv(e1_dir / "frontier_pareto.csv", pareto)
    write_csv(e1_dir / "safety_envelope.csv", envelope)
    write_csv(e1_dir / "plan_comparison.csv", comparison)

    receipt = {
        "model_bounded": True,
        "inputs": {
            "narrative_runs_count": len(lib_rows),
            "gfs_runs_count": len(gfs_rows),
            "source_dirs": [str(runs_u3), str(runs_step4)],
        },
        "receipts": lib_receipts + gfs_receipts,
    }
    (e1_dir / "receipts.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    delta_hv = comparison[1]["aggregate_hv_point_mean"] - comparison[0]["aggregate_hv_point_mean"]
    delta_fl = comparison[1]["aggregate_flourish_mean"] - comparison[0]["aggregate_flourish_mean"]

    report = [
        "# E1 Regime-conditioned Policy Bank (v1)",
        "",
        "Model-bounded outputs from local simulator runs only.",
        "",
        f"- Global A: anchor_baseline across both regimes.",
        f"- Narrative winner (B): {bank_narr['policy']} (safety-gated vs narrative anchor).",
        f"- GFS winner (C): {bank_gfs['policy']} (safety-gated vs patched GFS anchor).",
        f"- Aggregate Δhv_point_mean (B+C vs A): {delta_hv:.6f}",
        f"- Aggregate Δflourish_mean (B+C vs A): {delta_fl:.6f}",
        "",
        "Recommendation: use substrate-conditioned bank (B for narrative, C for GFS-like) under per-regime safety gates.",
    ]
    (e1_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    return {
        "dir": str(e1_dir),
        "narrative_policy": bank_narr["policy"],
        "gfs_policy": bank_gfs["policy"],
        "delta_hv": delta_hv,
        "delta_flourish": delta_fl,
    }


def run_sim(base, out_json, seed, corpus_file, args):
    cmd = [
        "python",
        "sim_memetic_gelation.py",
        "--seed",
        str(seed),
        "--iters",
        "9000",
        "--merge-prob",
        "0.02",
        "--resource-cap",
        "500",
        "--corpus-file",
        str(corpus_file),
        "--corpus-max-tokens",
        "220",
        "--semantic-risk-threshold",
        "0.35",
        "--output-json",
        str(out_json),
    ] + args
    subprocess.run(cmd, cwd=str(base), check=True)
    return cmd


def e2(base: Path, out: Path):
    e2_dir = out / "e2_representation_ablation_sweep_v1"
    run_dir = e2_dir / "runs"
    e2_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    a0 = base / "gfs_flourishing_balanced.txt"
    a1 = e2_dir / "A1_minimal_anchored.txt"
    a2 = e2_dir / "A2_expanded_bridge_templates.txt"

    base_txt = a0.read_text(encoding="utf-8", errors="ignore").rstrip() + "\n\n"
    a1_patch = "\n".join(
        [
            "### A1_minimal_anchored_abstract_social",
            "shared flourishing through cooperation and reciprocity",
            "mutual alignment with dignity and trust",
            "community benefit through fairness and care",
        ]
    )
    a2_patch = "\n".join(
        [
            "### A2_expanded_prototypes_and_bridges",
            "collective agency with prosocial norms and belonging",
            "bridge conflict through respect listening and solidarity",
            "non domination justice compassion reciprocity",
            "coordination template: cooperate -> verify -> repair -> stabilize",
            "bridge lexicon: together, shared, reciprocal, co-create, mutual aid",
        ]
    )
    a1.write_text(base_txt + a1_patch + "\n", encoding="utf-8")
    a2.write_text(base_txt + a1_patch + "\n" + a2_patch + "\n", encoding="utf-8")

    corpora = {"A0": a0, "A1": a1, "A2": a2}
    fixed_policy_name = "fixed_promote_frontier_alpha12_boost24"
    fixed_args = [
        "--gentle-promotion",
        "--boost-factor",
        "2.4",
        "--affinity-scale",
        "1.2",
        "--risk-penalty-scale",
        "0.5",
        "--targeted-prune",
        "--targeted-prune-mode",
        "hybrid",
        "--targeted-prune-interval",
        "1200",
        "--targeted-prune-fraction",
        "0.08",
        "--repetition-penalty",
        "0.46",
        "--replication-scale",
        "1.9",
    ]
    seeds = [6101, 6102, 6103]

    raw = []
    receipts = []
    for arm, corpus in corpora.items():
        for seed in seeds:
            out_json = run_dir / f"{arm}_{fixed_policy_name}_seed{seed}.json"
            cmd = run_sim(base, out_json, seed, corpus, fixed_args)
            receipts.append({"arm": arm, "seed": seed, "cmd": cmd, "corpus": str(corpus)})
            final = json.loads(out_json.read_text(encoding="utf-8"))["final"]
            raw.append(
                {
                    "arm": arm,
                    "policy": fixed_policy_name,
                    "seed": seed,
                    "risk": float(final.get("risk_mass_fraction", 0.0)),
                    "semantic_risk": float(final.get("semantic_risk_mass_fraction", 0.0)),
                    "flourish": float(final.get("flourishing_persistence", 0.0)),
                    "coop_score_density": mean(
                        [
                            float(final.get("score_bins", {}).get("1_3", {}).get("coop", {}).get("mean", 0.0)),
                            float(final.get("score_bins", {}).get("4_6", {}).get("coop", {}).get("mean", 0.0)),
                            float(final.get("score_bins", {}).get("7_plus", {}).get("coop", {}).get("mean", 0.0)),
                        ]
                    ),
                    "motif_length_survival": float(final.get("stable_nontrivial_mass_fraction", 0.0)),
                }
            )

    summary = []
    for arm in ["A0", "A1", "A2"]:
        rs = [x for x in raw if x["arm"] == arm]
        summary.append(
            {
                "arm": arm,
                "n": len(rs),
                "risk_mean": mean([x["risk"] for x in rs]),
                "semantic_risk_mean": mean([x["semantic_risk"] for x in rs]),
                "flourish_mean": mean([x["flourish"] for x in rs]),
                "coop_score_density_mean": mean([x["coop_score_density"] for x in rs]),
                "motif_length_survival_mean": mean([x["motif_length_survival"] for x in rs]),
            }
        )

    write_csv(e2_dir / "raw_metrics.csv", raw)
    write_csv(e2_dir / "summary.csv", summary)
    (e2_dir / "receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")

    report = [
        "# E2 Representation Ablation Ladder (minimal sweep)",
        "",
        "Model-bounded local simulation; fixed controller across arms.",
        "",
    ]
    a0s = next(x for x in summary if x["arm"] == "A0")
    for row in summary:
        report.append(
            f"- {row['arm']}: flourish={row['flourish_mean']:.6f}, coop_density={row['coop_score_density_mean']:.6f}, motif_survival={row['motif_length_survival_mean']:.6f}, semantic_risk={row['semantic_risk_mean']:.6f}"
        )
    report.append("")
    report.append(
        f"- ΔA2-A0 flourish={next(x for x in summary if x['arm']=='A2')['flourish_mean'] - a0s['flourish_mean']:.6f}, Δsemantic_risk={next(x for x in summary if x['arm']=='A2')['semantic_risk_mean'] - a0s['semantic_risk_mean']:.6f}"
    )
    (e2_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    return {"dir": str(e2_dir)}


def main():
    base = Path(__file__).resolve().parent
    out = base / "targeted_followup_gfs_v1" / "run_2026-02-27_e1_policy_bank_e2_minisweep"
    out.mkdir(parents=True, exist_ok=True)

    e1_info = e1(base, out)
    e2_info = e2(base, out)

    manifest = {
        "hypothesis_guide": str(base / "Frontier_Regime_Hypothesis_v1.md"),
        "e1": e1_info,
        "e2": e2_info,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
