"""
run_test.py -- multi-provider generation + data-analysis run of the Nisaba pipeline over the full Evaluation_SoSyM set.

Runs the full flow for each of the FOUR generator providers (xAI, Google, DeepSeek, Z-AI), each run
with reasoning disabled. For each (model x provider):

  intermediary (function calling, generator) -> reconstruction (structured output, generator)
  -> zero-shot description (generator)
  -> Nisaba RST narrative

Combos run in PARALLEL (thread pool). Steps are RESUMABLE (skip if output exists).
Then batch data analysis with OBJECTIVE instruments only: complexity (Orig vs every provider's
reconstruction), readability (FRE), multi-perspective semantic distance, and the RST relation metric
(rst_probe). Comprehension QA (functional understandability) is in understandability_test.py.

Artifacts: evaluation_sosym_test/<SourceProvider>/<Batch>/model<ID>/model<ID>_<family>_*.txt
(the folder path is the model's Terpsichora origin; the filename prefix is the pipeline provider).
"""
import os, re, json, shutil, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from typing import List, Literal

import nisaba_port as N

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Evaluation_SoSyM/ (scripts live in pipeline/)
SRC    = ROOT                                                          # source models + generated outputs share this tree
LOGJL  = os.path.join(ROOT, "llm_calls.jsonl")
N.set_log_path(LOGJL)

# Four generators (2 proprietary + 2 open-weight), all run with reasoning disabled. Every generator
# scores below the weakest reader (see understandability_test.py: qwen3.7-plus 34.6, kimi-k2.5 34.0),
# so descriptions are assessed by models strictly more capable than the one that produced them.
PROVIDERS = {
    "anthropic": {"gen": "anthropic/claude-sonnet-5", "gp": {"reasoning_effort": "none"}},   # 32.6
    "xai":       {"gen": "x-ai/grok-4.20",            "gp": {"reasoning_effort": "none"}},   # 32.3
    "deepseek":  {"gen": "deepseek/deepseek-v4-flash", "gp": {"reasoning_effort": "none"}},   # 33.6
    "zai":       {"gen": "z-ai/glm-5.2",              "gp": {"reasoning_effort": "none"}},   # 33.8
}
PRICING = {  # per 1M tokens (in, out), OpenRouter 2026-07 -- 4 generators + 2 readers
    "anthropic/claude-sonnet-5": (2.0, 10.0), "x-ai/grok-4.20": (1.25, 2.5),
    "deepseek/deepseek-v4-flash": (0.084, 0.168), "z-ai/glm-5.2": (0.392, 1.232),
    "qwen/qwen3.7-plus": (0.32, 1.28), "moonshotai/kimi-k2.5": (0.375, 2.025),
}
import glob as _glob
# The full evaluation set: every model under Evaluation_SoSyM (Provider x FC/SO x Batch), auto-discovered.
MODELS = [
    {"id": os.path.basename(os.path.dirname(_p)),
     "provider": _p.replace("\\", "/").split("/")[-4],
     "batch": _p.replace("\\", "/").split("/")[-3]}
    for _p in sorted(_glob.glob(os.path.join(SRC, "*", "* Batch *", "model*", "model*.decl")))
    if re.search(r"model\d+\.decl$", _p.replace("\\", "/"))
]
NARRATIVES = [("legacy", N.narrative_legacy, "")]

def dest_dir(m):
    d = os.path.join(ROOT, m["provider"], m["batch"], m["id"]); os.makedirs(d, exist_ok=True); return d
def w(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text if isinstance(text, str) else str(text))
def has(path): return os.path.exists(path) and os.path.getsize(path) > 0
def log(msg): print(msg, flush=True)

# ---------------------------------------------------------------------------
def run_combo(m, fam):
    """Full generation + judging for one (model, provider family). Resumable."""
    cfg = PROVIDERS[fam]; mid = m["id"]; tag = f"[{mid}/{fam}]"
    src = os.path.join(SRC, m["provider"], m["batch"], mid)
    dst = dest_dir(m); pre = os.path.join(dst, f"{mid}_{fam}")
    decl = open(os.path.join(src, f"{mid}.decl"), encoding="utf-8").read()
    for ext in ("decl", "json"):                          # copy source in (idempotent)
        d = os.path.join(dst, f"{mid}.{ext}")
        if not has(d): shutil.copy2(os.path.join(src, f"{mid}.{ext}"), d)
    N.clear_step(); N.set_step(model=mid, provider=fam, source=m["provider"], batch=m["batch"])

    # 1 intermediary
    ipath = pre + "_intermediary.txt"
    if has(ipath):
        intermediary = open(ipath, encoding="utf-8").read()
    else:
        log(f"{tag} intermediary ...")
        intermediary, _ = N.gen_intermediary(decl, cfg["gen"], cfg["gp"]); w(ipath, intermediary)

    # 2 zero-shot description
    zpath = pre + "_zero-shot_description.txt"
    if has(zpath):
        zeroshot = open(zpath, encoding="utf-8").read()
    else:
        log(f"{tag} zero-shot ...")
        zeroshot = N.zero_shot(decl, cfg["gen"], cfg["gp"]); w(zpath, zeroshot)

    # 3 narrative. The FINAL Nisaba description (composite) is the canonical generation
    #   output -- it (not the intermediary) is what gets reconstructed for the complexity comparison.
    nisaba_final = None
    for variant, fn, suf in NARRATIVES:
        npath = pre + f"_nisaba_description{suf}.txt"
        if has(npath):
            nisaba = open(npath, encoding="utf-8").read()
        else:
            log(f"{tag} narrative/{variant} ...")
            narrative = fn(intermediary, cfg["gen"], cfg["gp"])
            nisaba = (f"#MP-Declare Model\n{intermediary}\n## Interleaved Description\n{narrative}"
                      if variant == "legacy" else narrative)
            w(npath, nisaba)
        if suf == "":
            nisaba_final = nisaba                     # canonical description feeds reconstruction

    # 4 reconstruction FROM THE FINAL descriptions (not the intermediary) -> compare the
    #   reconstructed .decl's complexity / semantic distance against the original, symmetrically
    #   for the Nisaba final description and the zero-shot description.
    rdecl = pre + "_reconstructed.decl"
    if not has(rdecl):
        log(f"{tag} nisaba reconstruction (from final description) ...")
        recon = N.reconstruct(nisaba_final, cfg["gen"], cfg["gp"])
        with open(pre + "_reconstructed.json", "w", encoding="utf-8") as f:
            json.dump(recon.model_dump_json(), f, indent=4)
        w(rdecl, recon.convert_to_string())
    zrdecl = pre + "_zeroshot_reconstructed.decl"
    if not has(zrdecl):
        log(f"{tag} zero-shot reconstruction (from zero-shot description) ...")
        zrecon = N.reconstruct(zeroshot, cfg["gen"], cfg["gp"])
        with open(pre + "_zeroshot_reconstructed.json", "w", encoding="utf-8") as f:
            json.dump(zrecon.model_dump_json(), f, indent=4)
        w(zrdecl, zrecon.convert_to_string())
    return f"{mid}/{fam} OK"

def data_analysis():
    log("\n=== DATA ANALYSIS ===")
    # complexity -- walk ONLY the model .json files (original + reconstructions), not the run-level
    # jsons (run_meta.json), using the notebook's per-model metric functions.
    try:
        import pandas as pd
        from glob import glob
        rows = []
        for p in sorted(glob(os.path.join(ROOT, "*", "*", "*", "model*.json"))):
            try:
                data = json.loads(json.loads(open(p, encoding="utf-8").read()))
            except Exception as e:
                log(f"  complexity skip {os.path.basename(p)}: {e}"); continue
            rows.append({"File": os.path.basename(p),
                         "Size Metric": N.calculate_size_metric(data),
                         "Density Metric": N.calculate_density_metric(data),
                         "Separability Metric": N.calculate_separability_metric(data),
                         "Constraint Variability Metric": N.calculate_constraint_variability_metric(data)})
        df = pd.DataFrame(rows).sort_values("File")
        df.to_csv(os.path.join(ROOT, "complexity_metrics.csv"), index=False)
        log(f"  complexity: {len(df)} model jsons")
    except Exception as e:
        log(f"  complexity ERR: {e}"); traceback.print_exc()
    # readability
    try:
        import textstat, pandas as pd
        def md(s):
            s = re.sub(r"[`*#>_]+", " ", s); s = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", s)
            return re.sub(r"\s+", " ", s).strip()
        rows = []
        for root, _, files in os.walk(ROOT):
            for fn in files:
                if fn.endswith(".txt") and "description" in fn:
                    t = md(open(os.path.join(root, fn), encoding="utf-8").read())
                    rows.append({"filename": fn, "flesch_reading_ease": round(textstat.flesch_reading_ease(t), 2)})
        pd.DataFrame(rows).sort_values("filename").to_csv(os.path.join(ROOT, "readability_scores.csv"), index=False)
        log(f"  readability: {len(rows)} descriptions")
    except Exception as e:
        log(f"  readability ERR: {e}"); traceback.print_exc()
    # semantic distance (per model x provider)
    try:
        import pandas as pd
        embedder, emb_name, tau = N.load_embedder(); log(f"  embedder: {emb_name} tau={tau}")
        sd = []
        for m in MODELS:
            dst = dest_dir(m); orig = open(os.path.join(dst, f"{m['id']}.decl"), encoding="utf-8").read()
            for fam in PROVIDERS:
                for src, suffix in [("nisaba", "_reconstructed.decl"),
                                    ("zeroshot", "_zeroshot_reconstructed.decl")]:
                    rp = os.path.join(dst, f"{m['id']}_{fam}{suffix}")
                    if not has(rp): continue
                    try:                                     # a single broken reconstruction must not abort all SD
                        d = N.compute_semantic_distance(orig, open(rp, encoding="utf-8").read(), embedder, tau=tau)
                    except Exception as e:
                        log(f"    SD skip {m['id']}/{fam}/{src}: {type(e).__name__}: {str(e)[:80]}"); continue
                    row = {"model": m["id"], "family": fam, "recon_source": src,
                           "embedder": emb_name, "tau": tau, "total": round(d["total"], 4)}
                    for k, v in d["perspectives"].items():
                        row[k] = None if v is None else round(v, 4)
                    sd.append(row)
        pd.DataFrame(sd).to_csv(os.path.join(ROOT, "semantic_distance.csv"), index=False)
    except Exception as e:
        log(f"  semantic distance ERR: {str(e)[:200]}"); traceback.print_exc()
    # RST relation metric -- objective, generator-independent rhetorical-structure fidelity
    try:
        import rst_probe
        rows = rst_probe.analyze_dir(ROOT, write_csv=True)
        if rows:
            mA = sum(r["nis"]["aligned_rate"] for r in rows) / len(rows)
            zA = sum(r["zs"]["aligned_rate"] for r in rows) / len(rows)
            log(f"  RST relation metric: aligned_rate Nisaba {mA:.2f} vs zero-shot {zA:.2f} "
                f"({len(rows)} runs) -> rst_metrics.csv")
    except Exception as e:
        log(f"  RST metric ERR: {str(e)[:200]}"); traceback.print_exc()

def cost_report():
    usage = {}
    if os.path.exists(LOGJL):
        for line in open(LOGJL, encoding="utf-8"):
            try:
                r = json.loads(line); mdl = r.get("model"); u = r.get("usage") or {}
                a = usage.setdefault(mdl, {"prompt": 0, "completion": 0, "calls": 0})
                a["prompt"] += u.get("prompt_tokens") or 0; a["completion"] += u.get("completion_tokens") or 0
                a["calls"] += 1
            except Exception: pass
    total = 0.0
    for mdl, a in usage.items():
        pi, po = PRICING.get(mdl, (0, 0)); c = a["prompt"]/1e6*pi + a["completion"]/1e6*po
        a["cost_usd"] = round(c, 4); total += c
    return usage, round(total, 3)

# ---------------------------------------------------------------------------
def derived_stats():
    """Aggregate the per-combo metrics into the summary statistics the paper's summary tables report:
    complexity exact-reconstruction rate + max drift, Semantic Distance preservation, RST per-family gap,
    and the paired FRE effect size + directional consistency. Writes derived_summary.json."""
    import pandas as pd
    der = {}
    METRICS = [("Size", N.calculate_size_metric), ("Density", N.calculate_density_metric),
               ("Separability", N.calculate_separability_metric),
               ("Constraint variability", N.calculate_constraint_variability_metric)]
    cx = {m: {"exact": 0, "n": 0, "max": 0.0} for m, _ in METRICS}
    for m in MODELS:
        dst = dest_dir(m); sj = os.path.join(dst, m["id"] + ".json")
        if not has(sj):
            continue
        try:
            src = json.loads(json.loads(open(sj, encoding="utf-8").read()))
        except Exception:
            continue
        for fam in PROVIDERS:
            rj = os.path.join(dst, f"{m['id']}_{fam}_reconstructed.json")
            if not has(rj):
                continue
            try:
                rec = json.loads(json.loads(open(rj, encoding="utf-8").read()))
            except Exception:
                continue
            for mn, fn in METRICS:
                d = abs(fn(src) - fn(rec)); cx[mn]["n"] += 1
                cx[mn]["exact"] += (1 if d == 0 else 0); cx[mn]["max"] = max(cx[mn]["max"], d)
    der["complexity"] = {m: {"exact": f"{v['exact']}/{v['n']}", "max_abs_delta": round(v["max"], 4)}
                         for m, v in cx.items()}
    sdf = os.path.join(ROOT, "semantic_distance.csv")
    if os.path.exists(sdf):
        sd_by = {}
        for r in pd.read_csv(sdf).to_dict("records"):
            if r.get("recon_source") != "nisaba":
                continue
            for k in ["control_flow", "temporal", "data_conditions", "binds", "attr_domains", "total"]:
                v = r.get(k)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                v = float(v); s = sd_by.setdefault(k, {"zero": 0, "n": 0, "max": 0.0})
                s["n"] += 1; s["zero"] += (1 if v == 0 else 0); s["max"] = max(s["max"], v)
        der["semantic_distance"] = {k: {"preserved": f"{v['zero']}/{v['n']}", "max": round(v["max"], 4)}
                                    for k, v in sd_by.items()}
    rf = os.path.join(ROOT, "rst_relation_families.csv")
    if os.path.exists(rf):
        der["rst_gap"] = [{"family": r["constraint_type"], "prescribed": r["prescribed_relation"],
                           "nisaba_rate": r.get("nisaba_rate"), "zeroshot_rate": r.get("zeroshot_rate"),
                           "gap": r.get("gap")} for r in pd.read_csv(rf).to_dict("records")]
    rdf = os.path.join(ROOT, "readability_scores.csv")
    if os.path.exists(rdf):
        fre = {}
        for r in pd.read_csv(rdf).to_dict("records"):
            fn = str(r["filename"])
            if "_nisaba_description" in fn:
                fre.setdefault(fn.replace("_nisaba_description", ""), {})["n"] = r["flesch_reading_ease"]
            elif "_zero-shot_description" in fn:
                fre.setdefault(fn.replace("_zero-shot_description", ""), {})["z"] = r["flesch_reading_ease"]
        pairs = [(v["n"], v["z"]) for v in fre.values() if "n" in v and "z" in v]
        if pairs:
            nis = [p[0] for p in pairs]; zs = [p[1] for p in pairs]
            der["fre"] = {"mean_diff": round(sum(z - n for n, z in pairs) / len(pairs), 2),
                          "paired_d_z": N.cohens_d_paired(zs, nis),
                          "nisaba_lower": "%d/%d" % N.direction_consistency(nis, zs, "lt")}
    with open(os.path.join(ROOT, "derived_summary.json"), "w", encoding="utf-8") as f:
        json.dump(der, f, indent=2)
    log("  derived_summary.json written")
    return der

def main():
    assert "OPENROUTER_API_KEY" in os.environ
    t0 = time.time()
    combos = [(m, fam) for m in MODELS for fam in PROVIDERS]
    log(f"Running {len(combos)} (model x provider) combos in parallel ...")
    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = {ex.submit(run_combo, m, fam): (m["id"], fam) for m, fam in combos}
        failures = 0
        for fu in as_completed(futs):
            mid, fam = futs[fu]
            try:
                log("  done: " + fu.result())
            except Exception as e:
                failures += 1
                log(f"  FAILED {mid}/{fam}: {str(e)[:200]}")
    if failures == 0:
        data_analysis()
        derived_stats()
    else:
        log(f"\n{failures} combo(s) FAILED -> NOT all {len(combos)} complete; SKIPPING data analysis. "
            f"Resume to reprocess the failed combos; analysis runs automatically once 0 fail.")
    usage, total = cost_report()
    meta = {"providers": PROVIDERS, "models": MODELS, "usage_by_model": usage,
            "est_cost_usd": total, "seconds": round(time.time()-t0, 1)}
    with open(os.path.join(ROOT, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"\n=== DONE in {time.time()-t0:.0f}s | est ${total} ===")
    for mdl, a in sorted(usage.items()):
        log(f"   {mdl:34} calls={a['calls']:3} in={a['prompt']:7} out={a['completion']:7} ${a['cost_usd']}")

if __name__ == "__main__":
    main()
