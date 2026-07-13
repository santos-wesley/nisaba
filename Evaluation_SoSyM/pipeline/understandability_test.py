"""
understandability_test.py -- OUTCOME-based understandability measurement (reader-panel comprehension
QA + reader reconstruction from the description alone), run on the evaluation_sosym_test artifacts.

Understandability is assessed by (i) the reader-panel comprehension QA (balanced accuracy) and
(ii) reader reconstruction (Semantic Distance from the description alone). QA generation and
grading are deterministic (no API key); the reader calls require OPENROUTER_API_KEY.
"""
import os, re, json, glob, hashlib, random, threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Evaluation_SoSyM/ (scripts live in pipeline/)

# The local embedder (torch/sentence-transformers) is NOT safe under concurrent forward passes;
# serialize every compute_semantic_distance call with this lock. SD is fast so this is cheap.
_EMB_LOCK = threading.Lock()

# MODE: "composite" (default) = the delivered Nisaba artifact (intermediary lists + interleaved
# narrative); "narrative" = ONLY the "## Interleaved Description" prose, to isolate whether the
# accessibility loss is the formal scaffolding or the RST narrative style itself.
MODE = os.environ.get("UND_MODE", "composite")
SUF = "" if MODE == "composite" else "_narr"
def nisaba_narrative(text):
    i = text.find("## Interleaved Description")
    return text[i:] if i >= 0 else text

# fixed models (real OpenRouter slugs confirmed in the pilot)
# Reader PANEL (2, cross-vendor, comparable pm-llm-benchmark, reasoning OFF): QA + reconstruction are
# AVERAGED over both readers so the anchor is not hostage to one reader's idiosyncrasies.
# The panel avoids OpenAI and Google (they authored the 40 source models being described -> reader
# independence) and the pipeline's own generators (Anthropic, xAI, DeepSeek, z-ai).
#   qwen/qwen3.7-plus (Alibaba, closed) = 34.6  |  moonshotai/kimi-k2.5 (open-weight) = 34.0  (delta 0.6)
# Reasoning-off is applied via READER_PARAMS reasoning_effort:"none", which nisaba_port._reasoning_extra
# translates to extra_body={"reasoning":{"enabled":False}}; verified qwen honors it (reasoning_tokens=0).
READER_MODELS = {"qwen": "qwen/qwen3.7-plus", "kimi": "moonshotai/kimi-k2.5"}
READER_PARAMS = {"reasoning_effort": "none"}

# kimi-k2.5 is served by many OpenRouter providers; its single default (price-sorted) provider throttles
# hard under concurrency (~4 reconstructions/min), stalling the panel. qwen has ONE provider (Alibaba) that
# handles the load, so only kimi needs spreading. Round-robin kimi calls across its up providers so the load
# is split ~N-fold; allow_fallbacks keeps a request alive if a chosen provider is momentarily unavailable.
KIMI_PROVIDERS = ["ModelRun", "Chutes", "DeepInfra", "SiliconFlow", "AtlasCloud",
                  "StreamLake", "Novita", "Moonshot AI", "Phala", "Venice"]
def _reader_params(rname):
    if rname == "kimi":
        return {**READER_PARAMS, "provider": {"order": [random.choice(KIMI_PROVIDERS)], "allow_fallbacks": True}}
    return READER_PARAMS

# ---------------------------------------------------------------------------
# 1. Deterministic comprehension QA from the .decl
#    Semantics verified against the notebook's own Interleaved Description of model15
#    (e.g. Precedence[A,B] => "B permitted only after A"; Response[A,B] => "if A then B after";
#     Not Co-Existence[A,B] => "cannot both occur"; Absence[X] => "X must not occur").
# ---------------------------------------------------------------------------
def parse_constraints(decl):
    out = []
    for line in decl.splitlines():
        m = re.match(r"^\s*([A-Za-z][A-Za-z \-]*?)\s*\[(.*?)\]", line)
        if not m:
            continue
        head = m.group(1).strip()
        if head.lower() in ("activity", "bind"):
            continue
        acts = [a.strip() for a in m.group(2).split(",") if a.strip()]
        out.append((head, acts))
    return out

def build_qa(decl):
    """Return list of {q, a, type}. a in {'yes','no'}. Only unambiguous control-flow templates."""
    qa = []
    def add(q, a, t): qa.append({"q": q, "a": a, "type": t})
    cons = parse_constraints(decl)
    # forward-ordering pairs -> a reverse-direction DISTRACTOR is only added when the reverse is NOT
    # itself constrained (otherwise its 'no' ground truth would be wrong).
    fwd = {(a[0], a[1]) for ct, a in cons if len(a) > 1 and ct.lower() in
           ("response", "alternate response", "chain response",
            "succession", "alternate succession", "chain succession")}
    prec = {(a[0], a[1]) for ct, a in cons if len(a) > 1 and ct.lower() in
            ("precedence", "alternate precedence", "chain precedence")}
    for ctype, acts in cons:
        t = ctype.lower()
        A = acts[0] if acts else None
        B = acts[1] if len(acts) > 1 else None
        # DROPPED as GUESSABLE from activity names / domain common sense (a capable reader answers them
        # WITHOUT reading the description -> no discrimination): Init ("begin with X?"), End ("end with
        # X?"), Existence/Exactly ("must X occur?"). Discrimination comes from RELATIONS and PROHIBITIONS
        # below, whose answer requires the specific constraint and often violates the naive prior.
        if t == "absence" and A:
            add(f"Is '{A}' allowed to occur in the process?", "no", ctype)   # prior-violating: X is listed -> naive 'yes'
        # response family -> if A then B afterwards (activation=arg0; NOT inverted by declare4py)
        elif t in ("response", "alternate response", "chain response") and A and B:
            add(f"If '{A}' occurs, must '{B}' occur afterwards?", "yes", ctype)
            if A != B and (B, A) not in fwd:                 # reverse-direction DISTRACTOR -> 'no'
                add(f"If '{B}' occurs, must '{A}' occur afterwards?", "no", ctype + "-rev")
        # succession family -> both directions; ask the forward one
        elif t in ("succession", "alternate succession", "chain succession") and A and B:
            add(f"If '{A}' occurs, must '{B}' occur after it?", "yes", ctype)
            if A != B and (B, A) not in fwd:                 # reverse-direction DISTRACTOR -> 'no'
                add(f"If '{B}' occurs, must '{A}' occur after it?", "no", ctype + "-rev")
        # precedence family -> X precedes Y (Y needs X before). VERIFIED empirically via declare4py's
        # conformance checker: Precedence/Chain/Alternate[A,B] => trace [A,B] SATISFIED, [B,A] VIOLATED.
        # (reverseActivationTarget only swaps the DATA conditions A./T., NOT the temporal order.)
        elif t in ("precedence", "alternate precedence") and A and B:
            add(f"Can '{B}' occur without '{A}' having occurred before it?", "no", ctype)
            if A != B and (B, A) not in prec:                # reverse has no such requirement -> 'yes'
                add(f"Can '{A}' occur without '{B}' having occurred before it?", "yes", ctype + "-rev")
        elif t == "chain precedence" and A and B:
            add(f"Can '{B}' occur without '{A}' occurring immediately before it?", "no", ctype)
        elif t in ("not precedence", "not chain precedence") and A and B:
            add(f"Is '{B}' allowed to occur after '{A}' has already occurred earlier?", "no", ctype)
        elif t == "responded existence" and A and B:
            add(f"If '{A}' occurs, must '{B}' also occur in the same case?", "yes", ctype)
        elif t == "co-existence" and A and B:
            add(f"If '{A}' occurs, must '{B}' also occur in the same case?", "yes", ctype)
        elif t == "choice" and A and B:
            add(f"Must at least one of '{A}' or '{B}' occur?", "yes", ctype)
        elif t == "exclusive choice" and A and B:
            add(f"Can both '{A}' and '{B}' occur in the same case?", "no", ctype)
        elif t == "not co-existence" and A and B:
            add(f"Can both '{A}' and '{B}' occur in the same case?", "no", ctype)
        elif t in ("not responded existence", "not response", "not succession") and A and B:
            add(f"If '{A}' occurs, is '{B}' allowed to occur in response/relation to it?", "no", ctype)
        elif t in ("not chain succession", "not chain response") and A and B:
            same = (A == B)
            if same:
                add(f"Can two '{A}' activities occur immediately one after another (back-to-back)?", "no", ctype)
            else:
                add(f"Can '{B}' occur immediately after '{A}'?", "no", ctype)
    seen = set(); uniq = []                                   # dedup (a model may list a constraint twice)
    for x in qa:
        if x["q"] not in seen:
            seen.add(x["q"]); uniq.append(x)
    return uniq

def balanced_accuracy(qa, answers):
    """Mean of per-class recall (yes-recall + no-recall)/2 -> neutralizes the yes/no base-rate, so a
    constant-'yes' reader scores 0.5. None if fewer than 2 classes present (can't balance)."""
    tot = {"yes": 0, "no": 0}; cor = {"yes": 0, "no": 0}
    for x, a in zip(qa, answers):
        if x["a"] in tot:
            tot[x["a"]] += 1
            if a == x["a"]: cor[x["a"]] += 1
    classes = [c for c in ("yes", "no") if tot[c] > 0]
    if len(classes) < 2:
        return None
    return round(sum(cor[c] / tot[c] for c in classes) / len(classes), 3)

# ---------------------------------------------------------------------------
# 3. LLM-dependent parts (need OPENROUTER_API_KEY) -- imported lazily
# ---------------------------------------------------------------------------
def run_llm(rows):
    """rows: list of dicts with model/family/decl/nisaba/zeroshot text. Mutates rows in place with
    reader-recon distance and QA accuracy. Requires the API key + nisaba_port."""
    import nisaba_port as N
    from pydantic import BaseModel
    from typing import List, Literal
    N.set_log_path(os.path.join(HERE, "understandability_calls.jsonl"))
    embedder, emb_name, tau = N.load_embedder()

    class QAAns(BaseModel):
        answers: List[Literal["yes", "no", "unknown"]]

    FIELDS = ["recon_nisaba", "recon_zeroshot", "qa_nisaba", "qa_zeroshot",
              "qaraw_nisaba", "qaraw_zeroshot", "qa_n", "qa_yes", "qa_no",
              "ac1_nisaba", "ac1_zeroshot", "rawagree_nisaba", "rawagree_zeroshot", "per_reader"]
    def _process_row(r):
        pre = r["pre"]; cache = f"{pre}_understandability_v2{SUF}.json"   # v2 = reader panel + balanced discriminative QA
        if os.path.exists(cache):                       # resumable: reuse a completed combo
            r.update(json.load(open(cache, encoding="utf-8"))); print(f"  [cached] {r['model']}/{r['family']}"); return
        try:
            qa = build_qa(r["decl"])
            r["qa_n"] = len(qa); r["qa_yes"] = sum(1 for x in qa if x["a"] == "yes"); r["qa_no"] = len(qa) - r["qa_yes"]
            qtext = "\n".join(f"{i+1}. {x['q']}" for i, x in enumerate(qa))
            per_reader = {}
            for rname, rmodel in READER_MODELS.items():          # <-- PANEL: average over 2 readers
                pr = {}
                rparams = _reader_params(rname)                  # kimi: round-robin its providers (beats the throttle)
                # A1. reconstruction from description ONLY
                for tag, desc in (("nisaba", r["nisaba"]), ("zeroshot", r["zeroshot"])):
                    rp = f"{pre}_reader{SUF}_{rname}_{tag}_reconstructed.decl"
                    if not os.path.exists(rp):
                        recon = N.reconstruct(desc, rmodel, rparams)
                        open(rp, "w", encoding="utf-8").write(recon.convert_to_string())
                    with _EMB_LOCK:                              # embedder not concurrency-safe; SD is fast
                        d = N.compute_semantic_distance(r["decl"], open(rp, encoding="utf-8").read(), embedder, tau=tau)
                    pr[f"recon_{tag}"] = round(d["total"], 4)
                # A2. comprehension QA (balanced accuracy), from description ONLY
                for tag, desc in (("nisaba", r["nisaba"]), ("zeroshot", r["zeroshot"])):
                    msg = [{"role": "user", "content":
                            "You are given ONLY a natural-language description of a business process (no formal model). "
                            f"Answer ALL {len(qa)} yes/no questions IN ORDER using only what the description states or "
                            "clearly implies; answer 'unknown' if the description does not say.\n\nDESCRIPTION:\n" + desc +
                            "\n\nQUESTIONS:\n" + qtext}]
                    ans = N.structured(msg, rmodel, QAAns, params=rparams, max_tokens=2000, step=f"qa.{rname}.{tag}").answers
                    pr[f"qa_{tag}"] = balanced_accuracy(qa, ans)
                    pr[f"qaraw_{tag}"] = round(sum(1 for x, a in zip(qa, ans) if a == x["a"]) / len(qa), 3) if qa else None
                    pr[f"answers_{tag}"] = list(ans)
                per_reader[rname] = pr
            rnames = list(READER_MODELS)                          # inter-reader reliability (2-reader panel)
            if len(rnames) == 2:
                for tag in ("nisaba", "zeroshot"):
                    _a = per_reader[rnames[0]].get(f"answers_{tag}"); _b = per_reader[rnames[1]].get(f"answers_{tag}")
                    if _a and _b:
                        r[f"ac1_{tag}"] = N.gwet_ac1(_a, _b, ["yes", "no", "unknown"])
                        r[f"rawagree_{tag}"] = N.raw_agreement(_a, _b)
            def avg(key):
                v = [per_reader[rn][key] for rn in READER_MODELS if per_reader[rn].get(key) is not None]
                return round(sum(v) / len(v), 4) if v else None
            for key in ("recon_nisaba", "recon_zeroshot", "qa_nisaba", "qa_zeroshot", "qaraw_nisaba", "qaraw_zeroshot"):
                r[key] = avg(key)
            r["per_reader"] = per_reader
            json.dump({k: r.get(k) for k in FIELDS}, open(cache, "w", encoding="utf-8"), indent=2)
            print(f"  [done]   {r['model']}/{r['family']}: recon N={r['recon_nisaba']} Z={r['recon_zeroshot']} | "
                  f"balQA N={r['qa_nisaba']} Z={r['qa_zeroshot']} (yes/no={r['qa_yes']}/{r['qa_no']})")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [FAIL]   {r['model']}/{r['family']}: {type(e).__name__}: {str(e)[:160]}")

    # parallelize ACROSS combos: each combo still makes its 8 reader calls sequentially inside
    # _process_row, but many combos run at once (LLM calls are I/O-bound -> threads suffice).
    workers = int(os.environ.get("UND_WORKERS", "12"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(_process_row, r) for r in rows]))
    return rows

# ---------------------------------------------------------------------------
# 4. Orchestration
# ---------------------------------------------------------------------------
def load_rows():
    rows = []
    for nis in sorted(glob.glob(os.path.join(HERE, "*", "*", "model*", "model*_*_nisaba_description.txt"))):
        base = os.path.basename(nis)
        m = re.match(r"(model\d+)_(.+?)_nisaba_description", base)
        model, fam = m.group(1), m.group(2)
        d = os.path.dirname(nis)
        zs = nis.replace("_nisaba_description", "_zero-shot_description")
        decl = os.path.join(d, f"{model}.decl")
        if not (os.path.exists(zs) and os.path.exists(decl)):
            continue
        nisaba_txt = open(nis, encoding="utf-8").read()
        if MODE == "narrative":
            nisaba_txt = nisaba_narrative(nisaba_txt)
        rows.append({"model": model, "family": fam, "pre": os.path.join(d, f"{model}_{fam}"),
                     "decl": open(decl, encoding="utf-8").read(),
                     "nisaba": nisaba_txt,
                     "zeroshot": open(zs, encoding="utf-8").read()})
    return rows

def report(rows):
    def mean(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return round(sum(v) / len(v), 3) if v else None
    print("\n=== A. FUNCTIONAL COMPREHENSION (reader panel: %s) ===" % " + ".join(READER_MODELS.values()))
    print(f"{'model/family':20}{'recon_N':>9}{'recon_Z':>9}{'balQA_N':>9}{'balQA_Z':>9}{'rawQA_N':>9}{'rawQA_Z':>9}")
    for r in rows:
        g = lambda k: r.get(k, '-')
        print(f"{r['model']+'/'+r['family']:20}{g('recon_nisaba'):>9}{g('recon_zeroshot'):>9}"
              f"{g('qa_nisaba'):>9}{g('qa_zeroshot'):>9}{g('qaraw_nisaba'):>9}{g('qaraw_zeroshot'):>9}")
    print(f"MEANS recon:  Nisaba {mean('recon_nisaba')} vs zero-shot {mean('recon_zeroshot')}  (lower=more recoverable)")
    print(f"MEANS balQA:  Nisaba {mean('qa_nisaba')} vs zero-shot {mean('qa_zeroshot')}  (balanced acc, higher=better)")
    print(f"MEANS rawQA:  Nisaba {mean('qaraw_nisaba')} vs zero-shot {mean('qaraw_zeroshot')}")
    print(f"MEANS AC1:    Nisaba {mean('ac1_nisaba')} vs zero-shot {mean('ac1_zeroshot')}  (inter-reader, Gwet)")
    print(f"MEANS rawAgr: Nisaba {mean('rawagree_nisaba')} vs zero-shot {mean('rawagree_zeroshot')}")
    bym = defaultdict(lambda: {"n": [], "z": []})
    for r in rows:
        if isinstance(r.get("qa_nisaba"), (int, float)):
            bym[r["model"]]["n"].append(r["qa_nisaba"]); bym[r["model"]]["z"].append(r["qa_zeroshot"])
    for m, d in sorted(bym.items()):
        if d["n"]:
            print(f"  {m} balQA: Nisaba {sum(d['n'])/len(d['n']):.3f} vs zero-shot {sum(d['z'])/len(d['z']):.3f} "
                  f"(yes/no={next(r['qa_yes'] for r in rows if r['model']==m)}/{next(r['qa_no'] for r in rows if r['model']==m)})")

def main():
    rows = load_rows()
    if os.environ.get("OPENROUTER_API_KEY"):
        run_llm(rows)
        # persist
        keep = ["model","family","recon_nisaba","recon_zeroshot","qa_nisaba","qa_zeroshot",
                "qaraw_nisaba","qaraw_zeroshot","qa_n","qa_yes","qa_no",
                "ac1_nisaba","ac1_zeroshot","rawagree_nisaba","rawagree_zeroshot"]
        json.dump([{k:r.get(k) for k in keep} for r in rows],
                  open(os.path.join(HERE,f"understandability_results_v2{SUF}.json"),"w",encoding="utf-8"), indent=2)
        print(f"\n[MODE={MODE}] wrote understandability_results_v2{SUF}.json")
    else:
        print("[no OPENROUTER_API_KEY -> deterministic QA generation only]\n")
        print("=== comprehension QA generated per model (deterministic ground truth) ===")
        seen=set()
        for r in rows:
            if r["model"] in seen: continue
            seen.add(r["model"])
            qa = build_qa(r["decl"])
            yes=sum(1 for x in qa if x['a']=='yes'); no=len(qa)-yes
            print(f"\n-- {r['model']}: {len(qa)} questions ({yes} yes / {no} no) --")
            for x in qa: print(f"   [{x['a']:3}] ({x['type']}) {x['q']}")
    report(rows)

if __name__ == "__main__":
    main()
