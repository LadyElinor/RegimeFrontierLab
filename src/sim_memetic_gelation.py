import argparse, json, math, random, re
from collections import Counter


RISK_LEXICON = {
    "always", "never", "prove", "proven", "certain", "guaranteed", "unsafe", "harm",
    "biased", "bias", "hallucination", "fabricate", "fabricated", "false", "misleading",
    "conspiracy", "illegal", "toxic", "hate", "danger", "dangerous", "weapon"
}

RISK_PROTOTYPES = [
    "ignore previous instructions",
    "bypass safety policy",
    "reveal hidden prompt",
    "prompt injection attack",
    "fabricate false citation",
    "harmful illegal weapon",
    "toxic hateful content",
    "deceptive misleading answer",
]

FLOURISH_LEXICON = {
    "cooperate", "cooperation", "align", "alignment", "care", "respect", "mutual",
    "shared", "trust", "truth", "help", "safety", "benefit", "thrive", "flourish",
    "bridge", "understand", "wisdom", "compassion", "justice", "community", "peace"
}

FLOURISH_PROTOTYPES = [
    "shared flourishing through cooperation",
    "mutual alignment and trust",
    "bridge differences with respect",
    "truthful helpful safe response",
    "community benefit and compassion",
]


def gini(vals):
    xs = sorted([float(v) for v in vals if v >= 0])
    n = len(xs)
    if n == 0:
        return 0.0
    s = sum(xs)
    if s == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1) / n


def entropy(counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    e = 0.0
    for c in counter.values():
        p = c / tot
        e -= p * math.log(p, 2)
    return e


def _tokenize_text(text, min_len=3):
    toks = re.findall(r"[a-z][a-z\-']+", text.lower())
    return [t for t in toks if len(t) >= min_len]


def _stopwords():
    return {
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "have", "has",
        "had", "into", "over", "than", "then", "when", "what", "which", "while", "where", "their",
        "there", "also", "will", "would", "about", "could", "should", "been", "being", "only", "more"
    }


def _load_corpus_tokens(path, max_tokens=500, min_len=3):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    toks = _tokenize_text(text, min_len=min_len)
    freq = Counter(toks)
    stop = _stopwords()
    items = [w for w, _ in freq.most_common(max_tokens * 3) if w not in stop]
    return items[:max_tokens]


def _load_corpus_tokens_weighted(paths, per_source=120, min_len=3, global_cap=500):
    stop = _stopwords()
    selected = []
    for p in paths:
        text = open(p, "r", encoding="utf-8", errors="ignore").read()
        freq = Counter(_tokenize_text(text, min_len=min_len))
        top = [w for w, _ in freq.most_common(per_source * 4) if w not in stop][:per_source]
        selected.extend(top)
    # preserve source-quota mix, dedupe in order
    seen = set()
    out = []
    for w in selected:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= global_cap:
            break
    return out


def _repetition_ratio(tokens):
    if not tokens:
        return 0.0
    c = Counter(tokens)
    return max(c.values()) / len(tokens)


def _ngram_diversity(tokens, n=2):
    if len(tokens) < n:
        return 1.0
    ng = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ng)) / max(1, len(ng))


def _hub_concentration_from_graph(g):
    degs = [len(v) for v in g.values()]
    if not degs:
        return {"hub_gini": 0.0, "top5_degree_share": 0.0}
    total = sum(degs)
    top5 = sum(sorted(degs, reverse=True)[:5])
    return {
        "hub_gini": gini(degs),
        "top5_degree_share": (top5 / total) if total > 0 else 0.0,
    }


def _is_risk_motif(motif):
    toks = set(motif.split())
    return len(toks.intersection(RISK_LEXICON)) > 0


def _is_flourish_motif(motif):
    toks = set(motif.split())
    return len(toks.intersection(FLOURISH_LEXICON)) > 0


def _char_ngram_vector(text, n=3):
    s = f" {text.lower()} "
    grams = [s[i:i+n] for i in range(max(0, len(s) - n + 1))]
    c = Counter(grams)
    norm = math.sqrt(sum(v * v for v in c.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in c.items()}


def _cosine_sparse(a, b):
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _semantic_risk_score(motif, risk_vecs):
    v = _char_ngram_vector(motif)
    if not v:
        return 0.0
    best = max((_cosine_sparse(v, rv) for rv in risk_vecs), default=0.0)
    return max(0.0, min(1.0, best))


def _semantic_risk_mass_fraction(counter, risk_vecs, threshold=0.58):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    risky = 0
    for motif, cnt in counter.items():
        if _semantic_risk_score(motif, risk_vecs) >= threshold:
            risky += cnt
    return risky / total


def _semantic_flourish_mass_fraction(counter, flourish_vecs, threshold=0.50):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    good = 0
    for motif, cnt in counter.items():
        if _semantic_risk_score(motif, flourish_vecs) >= threshold:
            good += cnt
    return good / total


def _len_bin(n_tokens):
    if n_tokens <= 3:
        return "1_3"
    if n_tokens <= 6:
        return "4_6"
    return "7_plus"


def _risk_subgraph_metrics(graph):
    risk_nodes = [n for n in graph if n in RISK_LEXICON]
    if not risk_nodes:
        return {
            "risk_node_fraction": 0.0,
            "risk_edge_fraction": 0.0,
            "risk_lcc_fraction": 0.0,
            "risk_hub_gini": 0.0,
            "risk_top5_degree_share": 0.0,
        }

    risk_set = set(risk_nodes)
    edges = 0
    for u in risk_nodes:
        edges += sum(1 for v in graph[u] if v in risk_set)
    edges //= 2
    all_edges = sum(len(v) for v in graph.values()) // 2

    # LCC fraction inside risk-only induced subgraph
    seen = set()
    best = 0
    for n in risk_nodes:
        if n in seen:
            continue
        stack = [n]
        seen.add(n)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in graph[u]:
                if v in risk_set and v not in seen:
                    seen.add(v)
                    stack.append(v)
        best = max(best, size)

    degs = [sum(1 for v in graph[u] if v in risk_set) for u in risk_nodes]
    total_risk_deg = sum(degs)
    top5_risk_deg = sum(sorted(degs, reverse=True)[:5])

    return {
        "risk_node_fraction": len(risk_nodes) / max(1, len(graph)),
        "risk_edge_fraction": edges / max(1, all_edges),
        "risk_lcc_fraction": best / max(1, len(risk_nodes)),
        "risk_hub_gini": gini(degs),
        "risk_top5_degree_share": (top5_risk_deg / total_risk_deg) if total_risk_deg > 0 else 0.0,
    }


def run(seed=42, iters=100_000, merge_prob=0.02, resource_cap=500, replication_scale=2.0,
        disable_fusion=False, disable_replication=False, mutation_only=False, interval=1000,
        corpus_file="", corpus_max_tokens=250, corpus_files="", corpus_per_source=120,
        repetition_penalty=0.3, ngram_guard=True,
        targeted_prune=False, targeted_prune_interval=2500, targeted_prune_fraction=0.02,
        targeted_prune_mode="hub", gentle_promotion=False, boost_factor=2.0,
        coop_threshold=0.5, semantic_risk_threshold=0.58,
        affinity_scale=0.0, risk_penalty_scale=0.0):
    random.seed(seed)
    vocab = [
        "idea", "scatter", "context", "align", "narrative", "condense", "cognition", "snap", "resist",
        "propagate", "mind", "harder", "think", "against", "with", "emerge", "structure", "signal",
        "noise", "cluster", "fuse", "threshold", "drift", "lock", "pattern"
    ]
    if corpus_files:
        try:
            paths = [p.strip() for p in corpus_files.split(';') if p.strip()]
            extra = _load_corpus_tokens_weighted(paths, per_source=corpus_per_source, global_cap=corpus_max_tokens)
            vocab = sorted(set(vocab + extra))
        except Exception:
            pass
    elif corpus_file:
        try:
            extra = _load_corpus_tokens(corpus_file, max_tokens=corpus_max_tokens)
            vocab = sorted(set(vocab + extra))
        except Exception:
            pass
    soup = [random.choice(vocab) for _ in range(resource_cap)]
    risk_vecs = [_char_ngram_vector(w) for w in sorted(RISK_LEXICON)] + [_char_ngram_vector(p) for p in RISK_PROTOTYPES]
    flourish_vecs = [_char_ngram_vector(w) for w in sorted(FLOURISH_LEXICON)] + [_char_ngram_vector(p) for p in FLOURISH_PROTOTYPES]

    # undirected neighborhood overlap graph
    graph = {w: set() for w in vocab}

    history = []
    motif_presence = Counter()  # number of checkpoints where motif appears
    checkpoints = 0
    merge_attempt = {"high_coop": 0, "high_risk": 0, "neutral": 0}
    merge_accept = {"high_coop": 0, "high_risk": 0, "neutral": 0}

    def lcc_fraction():
        seen = set()
        best = 0
        for n in graph:
            if n in seen:
                continue
            stack = [n]
            seen.add(n)
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                for v in graph[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            best = max(best, size)
        return best / max(1, len(graph))

    def token_pair(m):
        toks = m.split()
        if len(toks) == 1:
            return toks[0], toks[0]
        return toks[0], toks[-1]

    for i in range(iters):
        if len(soup) < 2:
            soup += [random.choice(vocab) for _ in range(2)]

        a_idx, b_idx = random.sample(range(len(soup)), 2)
        a, b = soup[a_idx], soup[b_idx]
        a0, a1 = token_pair(a)
        b0, b1 = token_pair(b)

        overlap = len(graph[a1].intersection(graph[b0]))
        deg_boost = (len(graph[a1]) + len(graph[b0])) / 40.0
        p_merge = max(0.0, min(1.0, merge_prob + 0.03 * overlap + 0.01 * deg_boost))
        # upstream fusion-affinity modulation
        pair_coop = 0.5 * (_semantic_risk_score(a, flourish_vecs) + _semantic_risk_score(b, flourish_vecs))
        pair_risk = 0.5 * (_semantic_risk_score(a, risk_vecs) + _semantic_risk_score(b, risk_vecs))
        p_merge *= (1.0 + affinity_scale * pair_coop)
        p_merge *= max(0.0, 1.0 - risk_penalty_scale * pair_risk)
        p_merge = max(0.0, min(1.0, p_merge))

        did_merge = False
        if not disable_fusion and random.random() < p_merge:
            merged = a + " " + b
            merge_risk = _semantic_risk_score(merged, risk_vecs)
            merge_coop = _semantic_risk_score(merged, flourish_vecs)
            if merge_risk > semantic_risk_threshold:
                merge_bucket = "high_risk"
            elif merge_coop > coop_threshold:
                merge_bucket = "high_coop"
            else:
                merge_bucket = "neutral"
            merge_attempt[merge_bucket] += 1

            toksm = merged.split()
            if len(toksm) <= 10:
                rep_ratio = _repetition_ratio(toksm)
                bigram_div = _ngram_diversity(toksm, n=2)
                # suppress degenerate repeated chains
                if rep_ratio <= (1.0 - repetition_penalty) and (not ngram_guard or bigram_div >= 0.35):
                    # accept
                    merge_accept[merge_bucket] += 1
                    soup[a_idx] = merged
                    if a_idx < b_idx:
                        del soup[b_idx]
                    else:
                        del soup[a_idx]
                        a_idx = b_idx
                        soup[a_idx] = merged
                    graph[a1].add(b0)
                    graph[b0].add(a1)
                    did_merge = True

                    if not disable_replication and not mutation_only:
                        base_rep = replication_scale * (1 + overlap) * max(0.2, 1.0 - rep_ratio)
                        coop_score = _semantic_risk_score(merged, flourish_vecs)
                        if gentle_promotion:
                            rep = int(max(1, math.floor(base_rep * (1.0 + boost_factor * coop_score))))
                        else:
                            rep = int(base_rep)
                        for _ in range(max(0, rep - 1)):
                            soup.append(merged)

        # mutation-only baseline: random substitutions keep size bounded
        if mutation_only and random.random() < merge_prob:
            j = random.randrange(len(soup))
            toks = soup[j].split()
            if toks:
                k = random.randrange(len(toks))
                toks[k] = random.choice(vocab)
                soup[j] = " ".join(toks)

        # targeted intervention: prune motifs touching risk lexicon
        if targeted_prune and (i > 0) and (i % max(1, targeted_prune_interval) == 0) and soup:
            risky_idx = []
            for idx, motif in enumerate(soup):
                toks = motif.split()
                if not toks:
                    continue
                lex_hit = len(set(toks).intersection(RISK_LEXICON)) > 0
                sem_score = _semantic_risk_score(motif, risk_vecs)
                if targeted_prune_mode == "hub" and (not lex_hit):
                    continue
                if targeted_prune_mode == "hybrid" and (not lex_hit) and sem_score < 0.22:
                    continue
                end_a, end_b = toks[0], toks[-1]
                hub_score = len(graph.get(end_a, set())) + len(graph.get(end_b, set()))
                score = hub_score if targeted_prune_mode == "hub" else (0.7 * hub_score + 10.0 * sem_score)
                risky_idx.append((score, idx))
            if risky_idx:
                risky_idx.sort(reverse=True)
                drop_n = max(1, int(targeted_prune_fraction * len(soup)))
                drop_set = {idx for _, idx in risky_idx[:drop_n]}
                soup = [m for idx, m in enumerate(soup) if idx not in drop_set]

        # resource cap
        while len(soup) > resource_cap:
            del soup[random.randrange(len(soup))]

        if i % interval == 0:
            lengths = [len(x.split()) for x in soup]
            c = Counter(soup)
            checkpoints += 1
            for m in c.keys():
                motif_presence[m] += 1
            hub = _hub_concentration_from_graph(graph)
            risk_graph = _risk_subgraph_metrics(graph)
            risk_mass = sum(cnt for m, cnt in c.items() if _is_risk_motif(m)) / max(1, len(soup))
            semantic_risk_mass = _semantic_risk_mass_fraction(c, risk_vecs, threshold=semantic_risk_threshold)
            flourish_mass = sum(cnt for m, cnt in c.items() if _is_flourish_motif(m)) / max(1, len(soup))
            semantic_flourish_mass = _semantic_flourish_mass_fraction(c, flourish_vecs, threshold=0.50)
            history.append({
                "iter": i,
                "soup_size": len(soup),
                "unique_types": len(c),
                "lcc_fraction": lcc_fraction(),
                "mean_len": sum(lengths)/max(1,len(lengths)),
                "max_len": max(lengths) if lengths else 0,
                "len_gini": gini(lengths),
                "type_entropy": entropy(c),
                "top_type_count": c.most_common(1)[0][1] if c else 0,
                "hub_gini": hub["hub_gini"],
                "hub_top5_degree_share": hub["top5_degree_share"],
                "risk_mass_fraction": risk_mass,
                "semantic_risk_mass_fraction": semantic_risk_mass,
                "flourish_mass_fraction": flourish_mass,
                "semantic_flourish_mass_fraction": semantic_flourish_mass,
                **risk_graph,
            })

    c = Counter(soup)
    lengths = [len(x.split()) for x in soup]
    stable_threshold = max(1, int(0.6 * max(1, checkpoints)))
    stable_mass = sum(cnt for m, cnt in c.items() if motif_presence.get(m, 0) >= stable_threshold)
    stable_nontrivial_mass = sum(
        cnt for m, cnt in c.items()
        if motif_presence.get(m, 0) >= stable_threshold and len(m.split()) >= 2
    )
    hub = _hub_concentration_from_graph(graph)
    risk_graph = _risk_subgraph_metrics(graph)
    risk_mass = sum(cnt for m, cnt in c.items() if _is_risk_motif(m)) / max(1, len(soup))
    semantic_risk_mass = _semantic_risk_mass_fraction(c, risk_vecs, threshold=semantic_risk_threshold)
    flourish_mass = sum(cnt for m, cnt in c.items() if _is_flourish_motif(m)) / max(1, len(soup))
    semantic_flourish_mass = _semantic_flourish_mass_fraction(c, flourish_vecs, threshold=0.50)
    coop_persistent = sum(cnt for m, cnt in c.items() if _semantic_risk_score(m, flourish_vecs) > coop_threshold) / max(1, len(soup))

    motif_count = max(1, len(c))
    overlap_count = 0
    coop_scores = {"1_3": [], "4_6": [], "7_plus": []}
    risk_scores = {"1_3": [], "4_6": [], "7_plus": []}
    for m in c.keys():
        ls = len(m.split())
        b = _len_bin(ls)
        cs = _semantic_risk_score(m, flourish_vecs)
        rs = _semantic_risk_score(m, risk_vecs)
        coop_scores[b].append(cs)
        risk_scores[b].append(rs)
        if cs > coop_threshold and rs > semantic_risk_threshold:
            overlap_count += 1

    def _q(xs, q):
        if not xs:
            return 0.0
        ys = sorted(xs)
        i = int(round((len(ys)-1) * q))
        return ys[max(0, min(len(ys)-1, i))]

    score_bins = {}
    for b in ["1_3", "4_6", "7_plus"]:
        cs = coop_scores[b]
        rs = risk_scores[b]
        score_bins[b] = {
            "coop": {"mean": (sum(cs)/len(cs) if cs else 0.0), "median": _q(cs, 0.5), "p25": _q(cs, 0.25), "p75": _q(cs, 0.75)},
            "risk": {"mean": (sum(rs)/len(rs) if rs else 0.0), "median": _q(rs, 0.5), "p25": _q(rs, 0.25), "p75": _q(rs, 0.75)},
        }

    merge_accept_rates = {
        "high_coop": (merge_accept["high_coop"] / merge_attempt["high_coop"]) if merge_attempt["high_coop"] > 0 else 0.0,
        "high_risk": (merge_accept["high_risk"] / merge_attempt["high_risk"]) if merge_attempt["high_risk"] > 0 else 0.0,
        "neutral": (merge_accept["neutral"] / merge_attempt["neutral"]) if merge_attempt["neutral"] > 0 else 0.0,
    }

    out = {
        "config": {
            "seed": seed, "iters": iters, "merge_prob": merge_prob, "resource_cap": resource_cap,
            "replication_scale": replication_scale, "disable_fusion": disable_fusion,
            "disable_replication": disable_replication, "mutation_only": mutation_only, "interval": interval,
            "corpus_file": corpus_file, "corpus_files": corpus_files, "corpus_per_source": corpus_per_source,
            "corpus_max_tokens": corpus_max_tokens, "vocab_size": len(vocab),
            "repetition_penalty": repetition_penalty, "ngram_guard": ngram_guard,
            "targeted_prune": targeted_prune, "targeted_prune_interval": targeted_prune_interval,
            "targeted_prune_fraction": targeted_prune_fraction, "targeted_prune_mode": targeted_prune_mode,
            "gentle_promotion": gentle_promotion, "boost_factor": boost_factor,
            "coop_threshold": coop_threshold, "semantic_risk_threshold": semantic_risk_threshold,
            "affinity_scale": affinity_scale, "risk_penalty_scale": risk_penalty_scale,
        },
        "final": {
            "soup_size": len(soup),
            "unique_types": len(c),
            "lcc_fraction": history[-1]["lcc_fraction"] if history else 0.0,
            "mean_len": sum(lengths)/max(1,len(lengths)),
            "max_len": max(lengths) if lengths else 0,
            "len_gini": gini(lengths),
            "type_entropy": entropy(c),
            "hub_gini": hub["hub_gini"],
            "hub_top5_degree_share": hub["top5_degree_share"],
            "risk_mass_fraction": risk_mass,
            "semantic_risk_mass_fraction": semantic_risk_mass,
            "flourish_mass_fraction": flourish_mass,
            "semantic_flourish_mass_fraction": semantic_flourish_mass,
            **risk_graph,
            "stable_threshold_checkpoints": stable_threshold,
            "stable_mass_fraction": stable_mass / max(1, len(soup)),
            "stable_nontrivial_mass_fraction": stable_nontrivial_mass / max(1, len(soup)),
            "flourishing_score": (
                0.28 * (stable_nontrivial_mass / max(1, len(soup)))
                + 0.22 * (history[-1]["lcc_fraction"] if history else 0.0)
                + 0.18 * min(1.0, entropy(c) / 8.0)
                + 0.18 * min(1.0, flourish_mass + 0.5 * semantic_flourish_mass)
                - 0.14 * min(1.0, risk_mass + 0.5 * semantic_risk_mass)
            ),
            "flourishing_persistence": coop_persistent,
            "coop_risk_overlap_fraction": overlap_count / motif_count,
            "merge_accept_high_coop": merge_accept_rates["high_coop"],
            "merge_accept_high_risk": merge_accept_rates["high_risk"],
            "merge_accept_neutral": merge_accept_rates["neutral"],
            "score_bins": score_bins,
            "top10": c.most_common(10),
        },
        "history": history,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--merge-prob", type=float, default=0.02)
    ap.add_argument("--resource-cap", type=int, default=500)
    ap.add_argument("--replication-scale", type=float, default=2.0)
    ap.add_argument("--disable-fusion", action="store_true")
    ap.add_argument("--disable-replication", action="store_true")
    ap.add_argument("--mutation-only", action="store_true")
    ap.add_argument("--interval", type=int, default=1000)
    ap.add_argument("--corpus-file", type=str, default="")
    ap.add_argument("--corpus-files", type=str, default="", help="semicolon-separated file list for weighted multi-source reservoir")
    ap.add_argument("--corpus-per-source", type=int, default=120)
    ap.add_argument("--corpus-max-tokens", type=int, default=250)
    ap.add_argument("--repetition-penalty", type=float, default=0.3)
    ap.add_argument("--no-ngram-guard", action="store_true")
    ap.add_argument("--targeted-prune", action="store_true")
    ap.add_argument("--targeted-prune-interval", type=int, default=2500)
    ap.add_argument("--targeted-prune-fraction", type=float, default=0.02)
    ap.add_argument("--targeted-prune-mode", type=str, default="hub", choices=["hub", "hybrid"])
    ap.add_argument("--gentle-promotion", action="store_true")
    ap.add_argument("--boost-factor", type=float, default=2.0)
    ap.add_argument("--coop-threshold", type=float, default=0.5)
    ap.add_argument("--semantic-risk-threshold", type=float, default=0.58)
    ap.add_argument("--affinity-scale", type=float, default=0.0)
    ap.add_argument("--risk-penalty-scale", type=float, default=0.0)
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args()

    result = run(
        seed=args.seed,
        iters=args.iters,
        merge_prob=args.merge_prob,
        resource_cap=args.resource_cap,
        replication_scale=args.replication_scale,
        disable_fusion=args.disable_fusion,
        disable_replication=args.disable_replication,
        mutation_only=args.mutation_only,
        interval=args.interval,
        corpus_file=args.corpus_file,
        corpus_files=args.corpus_files,
        corpus_per_source=args.corpus_per_source,
        corpus_max_tokens=args.corpus_max_tokens,
        repetition_penalty=args.repetition_penalty,
        ngram_guard=(not args.no_ngram_guard),
        targeted_prune=args.targeted_prune,
        targeted_prune_interval=args.targeted_prune_interval,
        targeted_prune_fraction=args.targeted_prune_fraction,
        targeted_prune_mode=args.targeted_prune_mode,
        gentle_promotion=args.gentle_promotion,
        boost_factor=args.boost_factor,
        coop_threshold=args.coop_threshold,
        semantic_risk_threshold=args.semantic_risk_threshold,
        affinity_scale=args.affinity_scale,
        risk_penalty_scale=args.risk_penalty_scale,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result["final"], indent=2))
