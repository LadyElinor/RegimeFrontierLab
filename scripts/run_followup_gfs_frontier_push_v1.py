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


def hypervolume_proxy(points):
    vals = []
    for p in points:
        vals.append(max(0.0, min(1.0, 1.0 - p["risk"])) * max(0.0, min(1.0, p["utility"])) * max(0.0, min(1.0, p["flourish"])))
    return mean(vals)


def run_sim(base, out_json, seed, corpus_file, args):
    cmd = [
        "python",
        "sim_memetic_gelation.py",
        "--seed",
        str(seed),
        "--iters",
        "12000",
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


def main():
    base = Path(__file__).resolve().parent
    parent = base / "targeted_followup_gfs_v1"
    out = parent / "step4_frontier_push_v1"
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    corpus = parent / "gfs_flourishing_balanced_patched_v1.txt"
    if not corpus.exists():
        raise FileNotFoundError(f"Missing patched corpus: {corpus}")

    policies = {
        "anchor_baseline": ["--repetition-penalty", "0.48", "--replication-scale", "1.8"],
        "promote_balanced_ref": [
            "--gentle-promotion", "--boost-factor", "2.0", "--affinity-scale", "1.0", "--risk-penalty-scale", "0.5",
            "--targeted-prune", "--targeted-prune-mode", "hybrid", "--targeted-prune-interval", "1200", "--targeted-prune-fraction", "0.08",
            "--repetition-penalty", "0.48", "--replication-scale", "1.8",
        ],
        "promote_frontier_alpha12_boost24": [
            "--gentle-promotion", "--boost-factor", "2.4", "--affinity-scale", "1.2", "--risk-penalty-scale", "0.5",
            "--targeted-prune", "--targeted-prune-mode", "hybrid", "--targeted-prune-interval", "1200", "--targeted-prune-fraction", "0.08",
            "--repetition-penalty", "0.46", "--replication-scale", "1.9",
        ],
        "promote_frontier_alpha15_boost30": [
            "--gentle-promotion", "--boost-factor", "3.0", "--affinity-scale", "1.5", "--risk-penalty-scale", "0.5",
            "--targeted-prune", "--targeted-prune-mode", "hybrid", "--targeted-prune-interval", "1200", "--targeted-prune-fraction", "0.08",
            "--repetition-penalty", "0.46", "--replication-scale", "1.9",
        ],
    }
    seeds = [5311, 5312, 5313, 5314]

    raw, receipts = [], []
    for policy, pargs in policies.items():
        for seed in seeds:
            outj = runs / f"{policy}_seed{seed}.json"
            cmd = run_sim(base, outj, seed, corpus, pargs)
            receipts.append({"policy": policy, "seed": seed, "cmd": cmd})
            final = json.loads(outj.read_text(encoding="utf-8"))["final"]
            utility = 0.5 * float(final.get("lcc_fraction", 0.0)) + 0.5 * float(final.get("stable_nontrivial_mass_fraction", 0.0))
            raw.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "risk": float(final.get("risk_mass_fraction", 0.0)),
                    "semantic_risk": float(final.get("semantic_risk_mass_fraction", 0.0)),
                    "utility": utility,
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

    with (out / "raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)

    grouped = defaultdict(list)
    for r in raw:
        grouped[r["policy"]].append(r)

    summary = []
    for policy, rs in sorted(grouped.items()):
        summary.append(
            {
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

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    pareto_rows, frontier = [], []
    for i, a in enumerate(summary):
        dom = False
        for j, b in enumerate(summary):
            if i == j:
                continue
            if dominates(
                {"risk": b["risk_mean"], "utility": b["utility_mean"], "flourish": b["flourish_mean"]},
                {"risk": a["risk_mean"], "utility": a["utility_mean"], "flourish": a["flourish_mean"]},
            ):
                dom = True
                break
        pareto_rows.append(
            {
                "policy": a["policy"],
                "risk_mean": a["risk_mean"],
                "utility_mean": a["utility_mean"],
                "flourish_mean": a["flourish_mean"],
                "pareto_nondominated": int(not dom),
            }
        )
        if not dom:
            frontier.append({"risk": a["risk_mean"], "utility": a["utility_mean"], "flourish": a["flourish_mean"]})

    with (out / "pareto_points.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
        w.writeheader()
        w.writerows(pareto_rows)

    hv = {
        "hypervolume_proxy": hypervolume_proxy(frontier),
        "frontier_size": len(frontier),
        "points": len(summary),
    }
    with (out / "hypervolume.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hv.keys()))
        w.writeheader()
        w.writerow(hv)

    anchor = next(x for x in summary if x["policy"] == "anchor_baseline")
    envelope = []
    for row in summary:
        envelope.append(
            {
                "policy": row["policy"],
                "delta_risk_vs_anchor": row["risk_mean"] - anchor["risk_mean"],
                "delta_semantic_risk_vs_anchor": row["semantic_risk_mean"] - anchor["semantic_risk_mean"],
                "delta_utility_vs_anchor": row["utility_mean"] - anchor["utility_mean"],
                "delta_flourish_vs_anchor": row["flourish_mean"] - anchor["flourish_mean"],
                "safety_nonworse": int(
                    (row["risk_mean"] <= anchor["risk_mean"] + 1e-12)
                    and (row["semantic_risk_mean"] <= anchor["semantic_risk_mean"] + 1e-12)
                ),
            }
        )

    with (out / "safety_envelope.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(envelope[0].keys()))
        w.writeheader()
        w.writerows(envelope)

    (out / "receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")

    viable = [x for x in envelope if x["policy"] != "anchor_baseline" and x["safety_nonworse"] == 1]
    best = max(viable, key=lambda x: x["delta_flourish_vs_anchor"], default=None)

    report = [
        "# Step4 frontier push on patched GFS corpus",
        "",
        "Model-bounded simulation evidence only.",
        "",
        f"- Hypervolume proxy: {hv['hypervolume_proxy']:.6f} ({hv['frontier_size']}/{hv['points']} on frontier)",
    ]
    if best:
        report.append(
            f"- Best safety-nonworse flourish uplift: {best['policy']} (Δflourish={best['delta_flourish_vs_anchor']:.6f}, Δutility={best['delta_utility_vs_anchor']:.6f}, Δsemantic_risk={best['delta_semantic_risk_vs_anchor']:.6f})"
        )
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")

    print("done", out)


if __name__ == "__main__":
    main()
