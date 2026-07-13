"""Reprocess ALL objective metrics in a fresh process AFTER fix B landed in the notebook's
_sanitize_decl (which the SD pipeline loads verbatim). Recomputes complexity, readability,
semantic distance (now 320/320 — the 2 deepseek zero-shot combos with unbounded floats parse),
and RST, then regenerates derived_summary.json. No API calls (all local)."""
import os
os.environ.setdefault("OPENROUTER_API_KEY", "dummy")  # client() constructs at import; no API is called here
import run_test as R

R.data_analysis()
R.derived_stats()

# quick completeness check on the SD csv
import csv
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "semantic_distance.csv")
rows = list(csv.DictReader(open(p, encoding="utf-8")))
zs = [r for r in rows if r["recon_source"] == "zeroshot"]
nis = [r for r in rows if r["recon_source"] == "nisaba"]
print(f"\nSD rows total={len(rows)}  nisaba={len(nis)}  zeroshot={len(zs)}")
for want in ["model102", "model70"]:
    hit = [r for r in rows if r["model"] == want and r["family"] == "deepseek" and r["recon_source"] == "zeroshot"]
    print(f"  {want}/deepseek/zeroshot -> {'OK total='+hit[0]['total'] if hit else 'STILL MISSING'}")
