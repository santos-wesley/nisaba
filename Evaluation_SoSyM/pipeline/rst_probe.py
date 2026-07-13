"""
rst_probe.py -- objective, generator-independent measurement of the rhetorical
relations that the generation guidelines prescribe for each constraint.

Idea (why this is NON-circular):
  * The Nisaba GUIDELINES map every MP-Declare constraint TYPE to a prescribed RST
    relation (Init->Background, Response->Sequence, Succession->Cause-Effect,
    Alternate*->Condition-Consequence, Choice/Not*->Contrast, Co-Existence->Conjunction,
    Exclusive Choice->Alternative, ...).
  * Therefore a process model DETERMINISTICALLY defines a *target* relation profile.
  * We measure which relations each description actually REALIZES with a fixed
    discourse-marker lexicon (standard PDTB3 / RST-signalling connectives -- NOT tuned to
    Nisaba's wording), localised PER CONSTRAINT (only sentences that mention the
    constraint's activities count). No LLM, and no grader is ever shown the guidelines => no circularity.

Outputs per (model, provider) for Nisaba(interleaved narrative) vs zero-shot:
  coverage       = fraction of constraints the description even mentions
  aligned_rate   = of mentioned constraints, fraction whose sentence realises the
                   guideline-PRESCRIBED relation
  any_marker_rate= of mentioned constraints, fraction whose sentence has ANY relation marker
  density        = relation markers per 100 words
  profile_cos    = cosine(realised relation profile, model target profile)
"""
import os, re, json, glob, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Evaluation_SoSyM/ (scripts live in pipeline/)

# ---------------------------------------------------------------------------
# 1. Guideline mapping: constraint TYPE -> prescribed RST relation (verbatim table)
#    Longest key must win when matching a type name (Chain Succession before Succession).
# ---------------------------------------------------------------------------
CONSTRAINT_RELATION = {
    "Init": "Background",
    "End": "Conclusion",
    "Existence": "Elaboration",
    "Absence": "Contrast",
    "Exactly": "Summary",
    "Responded Existence": "Cause-Effect",
    "Precedence": "Background",
    "Response": "Sequence",
    "Succession": "Cause-Effect",
    "Alternate Precedence": "Condition-Consequence",
    "Alternate Response": "Condition-Consequence",
    "Alternate Succession": "Condition-Consequence",
    "Chain Precedence": "Sequence",
    "Chain Response": "Cause-Effect",
    "Chain Succession": "Cause-Effect",
    "Co-Existence": "Conjunction",
    "Choice": "Contrast",
    "Exclusive Choice": "Alternative",
    "Not Co-Existence": "Contrast",
    "Not Responded Existence": "Contrast",
    "Not Precedence": "Contrast",
    "Not Response": "Contrast",
    "Not Succession": "Contrast",
    "Not Chain Precedence": "Contrast",
    "Not Chain Response": "Contrast",
    "Not Chain Succession": "Contrast",
}
TYPES_BY_LEN = sorted(CONSTRAINT_RELATION, key=len, reverse=True)

# ---------------------------------------------------------------------------
# 2. Discourse-marker lexicon per RST relation.
#    Grounded in standard connective inventories (PDTB3 senses / RST Signalling Corpus):
#    Temporal/asynchronous -> Sequence; Contingency.cause -> Cause-Effect;
#    Contingency.condition -> Condition-Consequence; Comparison.contrast/concession ->
#    Contrast; Expansion.conjunction -> Conjunction; Expansion.disjunction/alternative ->
#    Alternative; Expansion.level-of-detail -> Elaboration; plus discourse-opening/closing
#    cues for Background/Conclusion and quantity cues for Summary.
#    Markers are matched as whole-word regexes, case-insensitive, on the localised sentence.
# ---------------------------------------------------------------------------
MARKERS = {
    "Background": [r"begins? with", r"\binitially\b", r"initial context", r"at the outset",
                   r"\bbefore\b", r"prior to", r"\bearlier\b", r"\balready\b", r"sets? up",
                   r"has (?:occurred|happened) before", r"must have (?:occurred|happened)"],
    "Conclusion": [r"\bfinally\b", r"\blastly\b", r"conclude[sd]?", r"in conclusion",
                   r"to close", r"ends? with", r"must be the last", r"closes the"],
    "Elaboration": [r"\bspecifically\b", r"in particular", r"\bnotably\b", r"in detail",
                    r"\bfurthermore\b", r"\bmoreover\b", r"\badditionally\b", r"adds? depth",
                    r"at least once", r"\bnamely\b"],
    "Contrast": [r"\bhowever\b", r"in contrast", r"on the other hand", r"\bconversely\b",
                 r"\bwhereas\b", r"\bbut\b", r"\byet\b", r"\balthough\b", r"\bthough\b",
                 r"\bnevertheless\b", r"\bnonetheless\b", r"\binstead\b"],
    "Summary": [r"\bexactly\b", r"\bprecisely\b", r"in total", r"a total of", r"a fixed number",
                r"no more than", r"no fewer than", r"in summary", r"to summari[sz]e"],
    "Cause-Effect": [r"\bbecause\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
                     r"\bconsequently\b", r"as a result", r"so that", r"cause[- ]and[- ]effect",
                     r"\bcausal\b", r"\btriggers?\b", r"leads? to", r"results? in",
                     r"\bensuring\b", r"\bthereby\b"],
    "Sequence": [r"\bthen\b", r"\bafter\b", r"\bafterwards\b", r"\bsubsequently\b", r"\bnext\b",
                 r"\bfollowing\b", r"\bonce\b", r"\beventually\b", r"\blater\b",
                 r"is (?:immediately )?followed by", r"\bfollows\b", r"must (?:eventually )?occur after"],
    "Condition-Consequence": [r"\bif\b", r"\bwhen\b", r"\bwhenever\b", r"provided that",
                              r"in case", r"only if", r"as long as", r"\bunless\b",
                              r"on the condition"],
    "Conjunction": [r"both .* and", r"\btogether\b", r"co[- ]exist", r"\bjointly\b",
                    r"as well as", r"in conjunction", r"must occur together"],
    "Alternative": [r"either .* or", r"\bexclusively\b", r"\balternatively\b",
                    r"mutually exclusive", r"exactly one", r"one of the", r"or else"],
}
COMPILED = {rel: [re.compile(p, re.I) for p in pats] for rel, pats in MARKERS.items()}

def markers_in(sentence, relation):
    return sum(1 for rx in COMPILED[relation] if rx.search(sentence))

def any_markers_in(sentence):
    return sum(markers_in(sentence, rel) for rel in COMPILED)

def profile_of(text):
    """Document-level relation profile: marker hits per relation over the whole text."""
    prof = Counter()
    for rel in COMPILED:
        prof[rel] = sum(len(rx.findall(text)) for rx in COMPILED[rel])
    return prof

# ---------------------------------------------------------------------------
# 3. Parsing helpers
# ---------------------------------------------------------------------------
def parse_constraints(decl_path):
    """Return list of (type, prescribed_relation, [activities])."""
    out = []
    for line in open(decl_path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"^([A-Za-z][A-Za-z \-]*?)\s*\[(.*?)\]", line)
        if not m:
            continue
        head = m.group(1).strip()
        if head.lower() in ("activity", "bind"):
            continue
        ctype = next((t for t in TYPES_BY_LEN if head.lower() == t.lower()), None)
        if ctype is None:
            ctype = next((t for t in TYPES_BY_LEN if head.lower().startswith(t.lower())), None)
        if ctype is None:
            continue
        acts = [a.strip() for a in m.group(2).split(",") if a.strip()]
        out.append((ctype, CONSTRAINT_RELATION[ctype], acts))
    return out

def split_sentences(text):
    # strip markdown headers / bullets / bold so only prose remains
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r"^[-*0-9.\)]+\s*", "", s)      # bullet / list markers
        s = s.replace("**", "")
        lines.append(s)
    joined = " ".join(lines)
    joined = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", joined)   # protect decimals
    parts = re.split(r"(?<=[.!?;:])\s+", joined)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

def nisaba_narrative(nisaba_text):
    """Only the interleaved RST narrative (apples-to-apples vs zero-shot prose)."""
    idx = nisaba_text.find("## Interleaved Description")
    return nisaba_text[idx:] if idx >= 0 else nisaba_text

_STOP = {"the", "a", "an", "of", "to", "and", "or", "for"}
def _stem(tok):
    tok = re.sub(r"[^a-z]", "", tok.lower())
    for suf in ("ing", "ed", "es", "s", "e"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok
def _stems(text):
    return {_stem(t) for t in re.findall(r"[A-Za-z]+", text) if t.lower() not in _STOP}
def sentences_mentioning(sents, activities):
    """A sentence mentions an activity if ALL of the activity's stemmed content tokens
    appear in the sentence (robust to paraphrase/nominalisation of activity labels, e.g.
    'Write Play Script' <- 'Writing the Play Script'; the all-tokens rule avoids the common
    'Play' token cross-matching sibling activities)."""
    act_tok = [{_stem(t) for t in re.findall(r"[A-Za-z]+", a) if t.lower() not in _STOP}
               for a in activities]
    out = []
    for s in sents:
        st = _stems(s)
        if any(toks and toks <= st for toks in act_tok):
            out.append(s)
    return out

def cosine(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

# ---------------------------------------------------------------------------
# 4. Analyse one description against a model's constraints
# ---------------------------------------------------------------------------
def analyse(text, constraints):
    sents = split_sentences(text)
    words = max(1, len(re.findall(r"\w+", text)))
    target = Counter(rel for _, rel, _ in constraints)
    mentioned = aligned = any_marker = 0
    per_type = defaultdict(lambda: [0, 0])   # ctype -> [mentioned, aligned]
    for ctype, rel, acts in constraints:
        cand = sentences_mentioning(sents, acts)
        if not cand:
            continue
        mentioned += 1
        per_type[ctype][0] += 1
        blob = " ".join(cand)
        if markers_in(blob, rel):
            aligned += 1
            per_type[ctype][1] += 1
        if any_markers_in(blob):
            any_marker += 1
    realised = profile_of(text)
    n = len(constraints)
    return {
        "n_constraints": n,
        "coverage": mentioned / n if n else 0,
        "aligned_rate": aligned / mentioned if mentioned else 0,
        "any_marker_rate": any_marker / mentioned if mentioned else 0,
        "density_per100w": 100 * sum(realised.values()) / words,
        "profile_cos": cosine(realised, target),
        "per_type": {k: tuple(v) for k, v in per_type.items()},
    }

# ---------------------------------------------------------------------------
# 5. Walk the test tree
# ---------------------------------------------------------------------------
def collect_rows(test_dir=HERE):
    decls = {os.path.basename(os.path.dirname(p)): p
             for p in glob.glob(os.path.join(test_dir, "*", "*", "model*", "model*.decl"))
             if re.search(r"model\d+\.decl$", p)}
    rows = []
    for model, decl_path in sorted(decls.items()):
        cons = parse_constraints(decl_path)
        base = os.path.dirname(decl_path)
        for nis in sorted(glob.glob(os.path.join(base, f"{model}_*_nisaba_description.txt"))):
            fam = re.search(rf"{model}_(.+?)_nisaba_description", os.path.basename(nis)).group(1)
            zs = nis.replace("_nisaba_description", "_zero-shot_description")
            if not os.path.exists(zs):
                continue
            nis_text = nisaba_narrative(open(nis, encoding="utf-8").read())
            zs_text = open(zs, encoding="utf-8").read()
            rows.append({"model": model, "family": fam,
                         "nis": analyse(nis_text, cons), "zs": analyse(zs_text, cons)})
    return rows

def pooled_per_type(rows):
    """Pool the per-constraint-type (mentioned, aligned) counts over all runs, Nisaba and zero-shot.
    Returns ctype -> [nis_mentioned, nis_aligned, zs_mentioned, zs_aligned]. This is the source of the
    per-relation-family realised/mentioned table in the paper (tab:rst_relation_metric)."""
    pooled = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        for k, (m, a) in r["nis"]["per_type"].items():
            pooled[k][0] += m; pooled[k][1] += a
        for k, (m, a) in r["zs"]["per_type"].items():
            pooled[k][2] += m; pooled[k][3] += a
    return pooled


def analyze_dir(test_dir=HERE, write_csv=True):
    """Run the RST probe over a test dir and (optionally) write rst_metrics.csv (per run) and
    rst_relation_families.csv (pooled per-family realised/mentioned = the RST relation table). Returns the rows.
    This is the pipeline entry point for the objective rhetorical-structure fidelity metric."""
    rows = collect_rows(test_dir)
    if write_csv and rows:
        import csv
        with open(os.path.join(test_dir, "rst_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "family", "n_constraints",
                        "coverage_nisaba", "coverage_zeroshot", "aligned_nisaba", "aligned_zeroshot",
                        "density_nisaba", "density_zeroshot", "profilecos_nisaba", "profilecos_zeroshot"])
            for r in rows:
                n, z = r["nis"], r["zs"]
                w.writerow([r["model"], r["family"], n["n_constraints"],
                            round(n["coverage"], 3), round(z["coverage"], 3),
                            round(n["aligned_rate"], 3), round(z["aligned_rate"], 3),
                            round(n["density_per100w"], 2), round(z["density_per100w"], 2),
                            round(n["profile_cos"], 3), round(z["profile_cos"], 3)])
        pooled = pooled_per_type(rows)
        with open(os.path.join(test_dir, "rst_relation_families.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["constraint_type", "prescribed_relation",
                        "nisaba_aligned", "nisaba_mentioned", "zeroshot_aligned", "zeroshot_mentioned",
                        "nisaba_rate", "zeroshot_rate", "gap"])
            for k in sorted(pooled, key=lambda k: -pooled[k][0]):
                mN, aN, mZ, aZ = pooled[k]
                nr = round(aN / mN, 3) if mN else 0.0
                zr = round(aZ / mZ, 3) if mZ else 0.0
                w.writerow([k, CONSTRAINT_RELATION[k], aN, mN, aZ, mZ, nr, zr, round(nr - zr, 3)])
    return rows

def main():
    decls = {os.path.basename(os.path.dirname(p)): p
             for p in glob.glob(os.path.join(HERE, "*", "*", "model*", "model*.decl"))
             if re.search(r"model\d+\.decl$", p)}
    rows = analyze_dir(HERE, write_csv=True)
    # ---- report ----
    print(f"Constraints & prescribed relations per model:")
    for model, decl_path in sorted(decls.items()):
        cons = parse_constraints(decl_path)
        tgt = Counter(rel for _, rel, _ in cons)
        print(f"  {model}: {len(cons)} constraints; target relations = "
              f"{dict(sorted(tgt.items(), key=lambda x:-x[1]))}")
    print()
    hdr = ("model/family", "n", "cov_N", "cov_Z", "align_N", "align_Z",
           "dens_N", "dens_Z", "cos_N", "cos_Z")
    print("{:<18}{:>4}{:>7}{:>7}{:>9}{:>9}{:>8}{:>8}{:>7}{:>7}".format(*hdr))
    aN, aZ = [], []
    for r in rows:
        n, z = r["nis"], r["zs"]
        aN.append(n["aligned_rate"]); aZ.append(z["aligned_rate"])
        print("{:<18}{:>4}{:>7.2f}{:>7.2f}{:>9.2f}{:>9.2f}{:>8.1f}{:>8.1f}{:>7.2f}{:>7.2f}".format(
            f"{r['model']}/{r['family']}", n["n_constraints"], n["coverage"], z["coverage"],
            n["aligned_rate"], z["aligned_rate"], n["density_per100w"], z["density_per100w"],
            n["profile_cos"], z["profile_cos"]))
    def mean(v): return sum(v) / len(v)
    print("\nMEANS  aligned_rate:  Nisaba {:.2f}  vs  zero-shot {:.2f}".format(mean(aN), mean(aZ)))
    print("MEANS  coverage:      Nisaba {:.2f}  vs  zero-shot {:.2f}".format(
        mean([r['nis']['coverage'] for r in rows]), mean([r['zs']['coverage'] for r in rows])))
    print("MEANS  density/100w:  Nisaba {:.1f}  vs  zero-shot {:.1f}".format(
        mean([r['nis']['density_per100w'] for r in rows]), mean([r['zs']['density_per100w'] for r in rows])))
    print("MEANS  profile_cos:   Nisaba {:.2f}  vs  zero-shot {:.2f}".format(
        mean([r['nis']['profile_cos'] for r in rows]), mean([r['zs']['profile_cos'] for r in rows])))
    # per-constraint-type aligned realisation, pooled (also written to rst_relation_families.csv)
    pooled = pooled_per_type(rows)               # ctype -> [mentN, alignN, mentZ, alignZ]
    print(f"\nPer-constraint-type aligned realisation (pooled over {len(rows)} runs):")
    print("  {:<22}{:<20}{:>14}{:>14}".format("constraint", "prescribed rel", "Nisaba", "zero-shot"))
    for k in sorted(pooled, key=lambda k: -pooled[k][0]):
        mN, aN_, mZ, aZ_ = pooled[k]
        print("  {:<22}{:<20}{:>14}{:>14}".format(
            k, CONSTRAINT_RELATION[k],
            f"{aN_}/{mN}" if mN else "-", f"{aZ_}/{mZ}" if mZ else "-"))

if __name__ == "__main__":
    main()
