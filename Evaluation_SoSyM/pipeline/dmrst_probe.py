"""
dmrst_probe.py -- DMRST neural discourse-parser corroboration of the lexical RST metric, run LOCALLY
over the evaluation_sosym_test output tree. Independent parser (DMRST; Liu/Shi/Chen 2021; xlm-roberta
backbone) that confirms the Nisaba-vs-zero-shot non-temporal relation gap. Uses the GPU if available.

Prereq: DMRST_Parser/ cloned + depth_mode/Savings/multi_all_checkpoint.torchsave downloaded (see dmrst_setup.log).
Outputs: dmrst_pooled.csv (relation -> Nisaba/zero-shot pooled counts, the source of the DMRST table),
         dmrst_batch.csv (per combo: non-temporal counts), dmrst_raw.json (spans + profiles).
"""
import os, re, sys, json, csv, glob
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Evaluation_SoSyM/ (scripts live in pipeline/)
REPO = os.path.join(HERE, "DMRST_Parser")
CKPT = os.path.join(REPO, "depth_mode", "Savings", "multi_all_checkpoint.torchsave")
sys.path.insert(0, REPO)
if not os.path.exists(CKPT) or os.path.getsize(CKPT) < 5e8:
    raise SystemExit(f"DMRST checkpoint missing/incomplete at {CKPT} -- run the setup first (dmrst_setup.log).")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DMRST device: {DEVICE}", flush=True)

from model_depth import ParsingNet   # noqa: E402  (needs REPO on sys.path)

tok = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
bert = AutoModel.from_pretrained("xlm-roberta-base")
for p in bert.parameters():
    p.requires_grad = False
net = ParsingNet(bert, bert_tokenizer=tok)
try:
    state = torch.load(CKPT, map_location="cpu")
except Exception:
    state = torch.load(CKPT, map_location="cpu", weights_only=False)   # torch>=2.6
net.load_state_dict(state, strict=False)
net.eval()
try:
    net = net.to(DEVICE)
except Exception as e:
    print(f"  [device] falling back to CPU: {e}", flush=True); DEVICE = torch.device("cpu")

# DMRST (RST-DT/GUM) relation -> Nisaba 10-relation set
DM2NIS = {"Background": "Background", "Elaboration": "Elaboration", "Contrast": "Contrast",
          "Comparison": "Contrast", "Summary": "Summary", "Cause": "Cause-Effect",
          "Explanation": "Cause-Effect", "Enablement": "Cause-Effect", "Temporal": "Sequence",
          "Condition": "Condition-Consequence", "Joint": "Conjunction"}   # rest -> "Other"
NONTEMPORAL = {"Contrast", "Condition-Consequence", "Cause-Effect"}


def parse_text(text):
    toks = tok.tokenize(text)
    if len(toks) < 2:
        return [], ""
    with torch.no_grad():
        _, _, spans, _, edu = net.TestingLoss([toks], input_EDU_breaks=None, LabelIndex=None,
                                              ParsingIndex=None, GenerateTree=True,
                                              use_pred_segmentation=True)
    return edu[0], (spans[0][0] if spans and spans[0] else "")


def relations_from_span(span):
    prof = Counter()
    for grp in re.findall(r"\([^()]*\)", span):
        for r in {m for m in re.findall(r"=([\w-]+):", grp) if m != "span"}:
            prof[DM2NIS.get(r, "Other")] += 1
    return prof


def nisaba_narrative(t):
    i = t.find("## Interleaved Description")
    return t[i:] if i >= 0 else t


def main():
    pairs = []
    for nis in sorted(glob.glob(os.path.join(HERE, "*", "*", "model*", "*_nisaba_description.txt"))):
        zs = nis.replace("_nisaba_description", "_zero-shot_description")
        if not os.path.exists(zs):
            continue
        m = re.search(r"(model\d+)_(.+?)_nisaba_description", os.path.basename(nis))
        pairs.append(((m.group(1) if m else os.path.basename(nis)), (m.group(2) if m else "-"),
                      nisaba_narrative(open(nis, encoding="utf-8").read()),
                      open(zs, encoding="utf-8").read()))
    print(f"{len(pairs)} (model x generator) description pairs to parse", flush=True)

    raw, rows, poolN, poolZ = {}, [], Counter(), Counter()
    for i, (model, fam, ntext, ztext) in enumerate(pairs):
        en, sn = parse_text(ntext)
        ez, sz = parse_text(ztext)
        pn, pz = relations_from_span(sn), relations_from_span(sz)
        poolN.update(pn); poolZ.update(pz)
        raw[f"{model}/{fam}"] = {"nisaba": {"edus": len(en), "profile": dict(pn)},
                                 "zeroshot": {"edus": len(ez), "profile": dict(pz)}}
        ntN = sum(v for k, v in pn.items() if k in NONTEMPORAL)
        ntZ = sum(v for k, v in pz.items() if k in NONTEMPORAL)
        rows.append([model, fam, len(en), len(ez), sum(pn.values()), sum(pz.values()), ntN, ntZ])
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(pairs)}", flush=True)

    json.dump(raw, open(os.path.join(HERE, "dmrst_raw.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "dmrst_batch.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "edu_nisaba", "edu_zeroshot", "rel_nisaba", "rel_zeroshot",
                    "nontemporal_nisaba", "nontemporal_zeroshot"])
        w.writerows(rows)
    with open(os.path.join(HERE, "dmrst_pooled.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relation", "nisaba", "zeroshot"])
        for rel in sorted(set(poolN) | set(poolZ), key=lambda k: -(poolN[k] + poolZ[k])):
            w.writerow([rel, poolN[rel], poolZ[rel]])

    if rows:
        mN = sum(r[6] for r in rows) / len(rows)
        mZ = sum(r[7] for r in rows) / len(rows)
        print(f"\nMEAN non-temporal rels (Contrast+Condition+Cause): Nisaba {mN:.1f} vs zero-shot {mZ:.1f}")
        print("Pooled DMRST distribution:")
        for rel in sorted(set(poolN) | set(poolZ), key=lambda k: -(poolN[k] + poolZ[k])):
            print(f"  {rel:<22} Nisaba {poolN[rel]:>4}  zero-shot {poolZ[rel]:>4}")
    print("-> dmrst_pooled.csv, dmrst_batch.csv, dmrst_raw.json", flush=True)


if __name__ == "__main__":
    main()
