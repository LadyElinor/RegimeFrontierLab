import csv, json, subprocess
from pathlib import Path
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def dominates(a, b):
    no_worse = (a['risk'] <= b['risk']) and (a['utility'] >= b['utility']) and (a['flourish'] >= b['flourish'])
    strictly = (a['risk'] < b['risk']) or (a['utility'] > b['utility']) or (a['flourish'] > b['flourish'])
    return no_worse and strictly


def hypervolume_proxy(points):
    vals = []
    for p in points:
        vals.append(max(0.0, min(1.0, 1.0 - p['risk'])) * max(0.0, min(1.0, p['utility'])) * max(0.0, min(1.0, p['flourish'])))
    return mean(vals)


def run_sim(base, out_json, seed, corpus_file, args):
    cmd = [
        'python', 'sim_memetic_gelation.py',
        '--seed', str(seed), '--iters', '12000', '--merge-prob', '0.02',
        '--resource-cap', '500', '--corpus-file', str(corpus_file), '--corpus-max-tokens', '220',
        '--semantic-risk-threshold', '0.35', '--output-json', str(out_json)
    ] + args
    subprocess.run(cmd, cwd=str(base), check=True)
    return cmd


def step1_signal_audit(base: Path, out: Path):
    run_dir = base / 'u3_pareto_frontier_v1' / 'runs'
    rows = []
    coverage = defaultdict(set)
    for fp in sorted(run_dir.glob('flourish_gfs_*.json')):
        data = json.loads(fp.read_text(encoding='utf-8'))
        final = data.get('final', {})
        cfg = data.get('config', {})
        name = fp.stem
        tail = name.replace('flourish_gfs_', '')
        policy, seed_s = tail.rsplit('_seed', 1)
        seed = int(seed_s)

        bins = final.get('score_bins', {})
        coop_density = mean([
            float(bins.get('1_3', {}).get('coop', {}).get('mean', 0.0)),
            float(bins.get('4_6', {}).get('coop', {}).get('mean', 0.0)),
            float(bins.get('7_plus', {}).get('coop', {}).get('mean', 0.0)),
        ])

        rows.append({
            'policy': policy,
            'seed': seed,
            'coop_score_density': coop_density,
            'motif_length_survival': float(final.get('stable_nontrivial_mass_fraction', 0.0)),
            'stable_mass_fraction': float(final.get('stable_mass_fraction', 0.0)),
            'mean_len': float(final.get('mean_len', 0.0)),
            'max_len': float(final.get('max_len', 0.0)),
            'risk_mass_fraction': float(final.get('risk_mass_fraction', 0.0)),
            'semantic_risk_mass_fraction': float(final.get('semantic_risk_mass_fraction', 0.0)),
            'flourishing_persistence': float(final.get('flourishing_persistence', 0.0)),
        })
        coverage[policy].add(seed)

    audit_dir = out / 'step1_gfs_signal_audit'
    audit_dir.mkdir(exist_ok=True)

    with (audit_dir / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r['policy']].append(r)

    summary = []
    for policy, rs in sorted(grouped.items()):
        summary.append({
            'policy': policy,
            'n': len(rs),
            'seed_coverage': len(coverage[policy]),
            'coop_score_density_mean': mean([x['coop_score_density'] for x in rs]),
            'motif_length_survival_mean': mean([x['motif_length_survival'] for x in rs]),
            'stable_mass_fraction_mean': mean([x['stable_mass_fraction'] for x in rs]),
            'mean_len_mean': mean([x['mean_len'] for x in rs]),
            'max_len_mean': mean([x['max_len'] for x in rs]),
            'risk_mass_fraction_mean': mean([x['risk_mass_fraction'] for x in rs]),
            'semantic_risk_mass_fraction_mean': mean([x['semantic_risk_mass_fraction'] for x in rs]),
            'flourishing_persistence_mean': mean([x['flourishing_persistence'] for x in rs]),
        })

    with (audit_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    return audit_dir


def step2_rep_patch(base: Path, out: Path):
    src = base / 'gfs_flourishing_balanced.txt'
    dst = out / 'gfs_flourishing_balanced_patched_v1.txt'
    text = src.read_text(encoding='utf-8', errors='ignore').rstrip() + '\n\n'
    patch_block = '\n'.join([
        '### representation_patch_v1_abstract_social_flourishing',
        'shared flourishing through cooperation and reciprocity',
        'mutual alignment, dignity, and trust across differences',
        'community benefit with fairness, belonging, and care',
        'collective agency, purpose, and meaning in social life',
        'bridge conflict through respect, listening, and solidarity',
        'prosocial norms: compassion, justice, and non-domination',
    ]) + '\n'
    dst.write_text(text + patch_block, encoding='utf-8')
    return dst


def step3_compact_ab(base: Path, out: Path, patched_corpus: Path):
    run_dir = out / 'step3_compact_ab_flourish_gfs'
    runs = run_dir / 'runs'
    run_dir.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)

    corpora = {
        'A_original': base / 'gfs_flourishing_balanced.txt',
        'B_patched': patched_corpus,
    }
    policies = {
        'anchor_baseline': ['--repetition-penalty', '0.48', '--replication-scale', '1.8'],
        'anchor_hybrid': ['--repetition-penalty', '0.48', '--replication-scale', '1.8', '--targeted-prune', '--targeted-prune-mode', 'hybrid', '--targeted-prune-interval', '1200', '--targeted-prune-fraction', '0.08'],
        'promote_balanced': ['--gentle-promotion', '--boost-factor', '2.0', '--affinity-scale', '1.0', '--risk-penalty-scale', '0.5', '--targeted-prune', '--targeted-prune-mode', 'hybrid', '--targeted-prune-interval', '1200', '--targeted-prune-fraction', '0.08', '--repetition-penalty', '0.48', '--replication-scale', '1.8'],
        'promote_strict': ['--gentle-promotion', '--boost-factor', '2.0', '--affinity-scale', '1.5', '--risk-penalty-scale', '1.0', '--targeted-prune', '--targeted-prune-mode', 'hybrid', '--targeted-prune-interval', '1200', '--targeted-prune-fraction', '0.08', '--repetition-penalty', '0.48', '--replication-scale', '1.8'],
    }
    seeds = [4301, 4302, 4303, 4304]

    raw, receipts = [], []
    for arm, corpus in corpora.items():
        for policy, pargs in policies.items():
            for seed in seeds:
                outj = runs / f'{arm}_{policy}_seed{seed}.json'
                cmd = run_sim(base, outj, seed, corpus, pargs)
                receipts.append({'arm': arm, 'policy': policy, 'seed': seed, 'cmd': cmd})
                f = json.loads(outj.read_text(encoding='utf-8'))['final']
                utility = 0.5 * float(f.get('lcc_fraction', 0.0)) + 0.5 * float(f.get('stable_nontrivial_mass_fraction', 0.0))
                raw.append({
                    'arm': arm, 'policy': policy, 'seed': seed,
                    'risk': float(f.get('risk_mass_fraction', 0.0)),
                    'semantic_risk': float(f.get('semantic_risk_mass_fraction', 0.0)),
                    'utility': utility,
                    'flourish': float(f.get('flourishing_persistence', 0.0)),
                    'coop_score_density': mean([
                        float(f.get('score_bins', {}).get('1_3', {}).get('coop', {}).get('mean', 0.0)),
                        float(f.get('score_bins', {}).get('4_6', {}).get('coop', {}).get('mean', 0.0)),
                        float(f.get('score_bins', {}).get('7_plus', {}).get('coop', {}).get('mean', 0.0)),
                    ]),
                    'motif_length_survival': float(f.get('stable_nontrivial_mass_fraction', 0.0)),
                })

    with (run_dir / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r['arm'], r['policy'])].append(r)

    summary = []
    for (arm, policy), rs in sorted(grouped.items()):
        summary.append({
            'arm': arm, 'policy': policy, 'n': len(rs),
            'risk_mean': mean([x['risk'] for x in rs]),
            'semantic_risk_mean': mean([x['semantic_risk'] for x in rs]),
            'utility_mean': mean([x['utility'] for x in rs]),
            'flourish_mean': mean([x['flourish'] for x in rs]),
            'coop_score_density_mean': mean([x['coop_score_density'] for x in rs]),
            'motif_length_survival_mean': mean([x['motif_length_survival'] for x in rs]),
        })

    with (run_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    hv_rows, pareto_rows = [], []
    for arm in corpora:
        pts = [x for x in summary if x['arm'] == arm]
        frontier = []
        for i, a in enumerate(pts):
            dom = False
            for j, b in enumerate(pts):
                if i == j:
                    continue
                if dominates(
                    {'risk': b['risk_mean'], 'utility': b['utility_mean'], 'flourish': b['flourish_mean']},
                    {'risk': a['risk_mean'], 'utility': a['utility_mean'], 'flourish': a['flourish_mean']}
                ):
                    dom = True
                    break
            pareto_rows.append({
                'arm': arm, 'policy': a['policy'],
                'risk_mean': a['risk_mean'], 'utility_mean': a['utility_mean'], 'flourish_mean': a['flourish_mean'],
                'pareto_nondominated': int(not dom)
            })
            if not dom:
                frontier.append({'risk': a['risk_mean'], 'utility': a['utility_mean'], 'flourish': a['flourish_mean']})

        hv_rows.append({
            'arm': arm,
            'hypervolume_proxy': hypervolume_proxy(frontier),
            'frontier_size': len(frontier),
            'points': len(pts),
        })

    with (run_dir / 'pareto_points.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys())); w.writeheader(); w.writerows(pareto_rows)

    with (run_dir / 'hypervolume.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(hv_rows[0].keys())); w.writeheader(); w.writerows(hv_rows)

    # safety envelope: compare best flourish policy in each arm vs anchor_baseline (same arm)
    envelope = []
    for arm in corpora:
        arm_rows = [x for x in summary if x['arm'] == arm]
        base_row = next(x for x in arm_rows if x['policy'] == 'anchor_baseline')
        best_flourish = max(arm_rows, key=lambda r: r['flourish_mean'])
        envelope.append({
            'arm': arm,
            'best_policy': best_flourish['policy'],
            'delta_risk_vs_anchor': best_flourish['risk_mean'] - base_row['risk_mean'],
            'delta_semantic_risk_vs_anchor': best_flourish['semantic_risk_mean'] - base_row['semantic_risk_mean'],
            'delta_utility_vs_anchor': best_flourish['utility_mean'] - base_row['utility_mean'],
            'delta_flourish_vs_anchor': best_flourish['flourish_mean'] - base_row['flourish_mean'],
        })

    with (run_dir / 'safety_envelope.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(envelope[0].keys())); w.writeheader(); w.writerows(envelope)

    (run_dir / 'receipts.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')

    report = [
        '# Targeted Follow-up: GFS patch A/B',
        '',
        'Model-bounded simulation evidence only.',
        '',
        '## Hypervolume proxy (frontier area proxy)'
    ]
    for h in hv_rows:
        report.append(f"- {h['arm']}: hv={h['hypervolume_proxy']:.6f}, frontier={h['frontier_size']}/{h['points']}")
    report.append('')
    report.append('## Safety envelope check (best flourish policy vs anchor baseline in same arm)')
    for e in envelope:
        report.append(
            f"- {e['arm']} ({e['best_policy']}): Δrisk={e['delta_risk_vs_anchor']:.6f}, Δsemantic_risk={e['delta_semantic_risk_vs_anchor']:.6f}, Δutility={e['delta_utility_vs_anchor']:.6f}, Δflourish={e['delta_flourish_vs_anchor']:.6f}"
        )

    (run_dir / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    return run_dir


def main():
    base = Path(__file__).resolve().parent
    out = base / 'targeted_followup_gfs_v1'
    out.mkdir(exist_ok=True)

    audit_dir = step1_signal_audit(base, out)
    patched = step2_rep_patch(base, out)
    ab_dir = step3_compact_ab(base, out, patched)

    manifest = {
        'step1_audit': str(audit_dir),
        'step2_patched_corpus': str(patched),
        'step3_compact_ab': str(ab_dir),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('done', json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
