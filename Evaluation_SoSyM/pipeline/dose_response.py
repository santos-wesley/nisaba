"""Dose-response probe: does the DEGREE of rhetorical realisation (lexical RST aligned rate, per run)
correlate with the reader panel's comprehension (balanced QA) and recoverability (reader-recon SD)?
Two views: (1) pooled across approaches (320 points: each run x {nisaba, zeroshot}) - uses the full
treatment variance; (2) within-Nisaba only (160 points) - cleaner dose-response, restricted range."""
import csv, json, glob, os
from scipy.stats import spearmanr
os.chdir(r"c:/Projects/Nisaba Journal/Github repo/Evaluation_SoSyM")

rst = {(r["model"], r["family"]): r for r in csv.DictReader(open("rst_metrics.csv", encoding="utf-8"))}
und = {}
for p in glob.glob("*/*/model*/model*_*_understandability_v2.json"):
    base = os.path.basename(p)
    model = base.split("_")[0]; fam = base.split("_")[1]
    und[(model, fam)] = json.load(open(p, encoding="utf-8"))

pooled, within = {"al": [], "qa": [], "rec": []}, {"al": [], "qa": [], "rec": []}
for key, r in rst.items():
    u = und.get(key)
    if not u:
        continue
    for tag in ("nisaba", "zeroshot"):
        al = r.get(f"aligned_{tag}"); qa = u.get(f"qa_{tag}"); rec = u.get(f"recon_{tag}")
        if al in (None, "") or qa is None or rec is None:
            continue
        al = float(al)
        pooled["al"].append(al); pooled["qa"].append(qa); pooled["rec"].append(rec)
        if tag == "nisaba":
            within["al"].append(al); within["qa"].append(qa); within["rec"].append(rec)

def rep(d, label):
    n = len(d["al"])
    r1, p1 = spearmanr(d["al"], d["qa"])
    r2, p2 = spearmanr(d["al"], d["rec"])
    print(f"{label} (n={n}):")
    print(f"  aligned_rate vs balanced QA:      rho={r1:+.3f}  p={p1:.2g}")
    print(f"  aligned_rate vs recon distance:   rho={r2:+.3f}  p={p2:.2g}   (negative = more rhetoric, more recoverable)")

rep(pooled, "POOLED (both approaches, 2 points/run)")
rep(within, "WITHIN-NISABA only")
# extra: marker density as alternative dose
dens_p = [float(rst[k][f"density_{t}"]) for k in rst if k in und for t in ("nisaba","zeroshot") if rst[k].get(f"density_{t}") not in (None,"") and und[k].get(f"qa_{t}") is not None]
qa_p   = [und[k][f"qa_{t}"] for k in rst if k in und for t in ("nisaba","zeroshot") if rst[k].get(f"density_{t}") not in (None,"") and und[k].get(f"qa_{t}") is not None]
r3, p3 = spearmanr(dens_p, qa_p)
print(f"\nmarker DENSITY vs balanced QA (pooled, n={len(dens_p)}): rho={r3:+.3f} p={p3:.2g}")
