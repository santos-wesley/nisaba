# -*- coding: utf-8 -*-
"""
Sweep of the pairing cost threshold `tau` (and the 'center' whitening), to calibrate
mitigation (a): reject bad matches instead of forcing the bijection.

Reuses the already-generated variants (variants/<model>/) + the real original↔reconstructed pairs from base_models/.
Embeddings are CACHED per string, so the tau sweep is cheap (tau/whiten only affect
the post-processing of the assignment, not the embeddings).

Metrics per (tau, whiten):
  relabel_detect : fraction of `relabel_single` now DETECTED (total>0)     [want HIGH]
  inv_easy_ok    : fraction of easy invariances (recon_reorder/reorder/syn_values/
                   enum_synonym) that stay ~0                              [want 1.0]
  inv_hard_ok    : same for the hard cases (syn_names/syn_all/recon_paraphrase)
  sens_ok        : fraction of sensitivities still detected               [want ~1.0]
  realpair_max   : largest distance over the identical real pairs       [must be 0]

Usage: [STRESS_EMB=<model>] python tau_sweep.py
"""
import os, csv, logging
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
for _n in ['httpcore', 'httpx', 'urllib3', 'filelock', 'huggingface_hub', 'transformers', 'sentence_transformers']:
    logging.getLogger(_n).setLevel(logging.ERROR)
logging.disable(logging.INFO)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from mp_metric import compute_semantic_distance

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'base_models')  # eight calibration models (.decl) + their pipeline reconstructions (REALPAIR)
VARDIR = os.path.join(HERE, 'variants')
EMB_NAME = os.environ.get('STRESS_EMB', 'Alibaba-NLP/gte-large-en-v1.5')
SHORT = EMB_NAME.split('/')[-1]

INV_EASY = {'recon_reorder', 'reorder', 'syn_values', 'enum_synonym'}
INV_HARD = {'syn_names', 'syn_all', 'recon_paraphrase'}


def fix_position_ids(model):
    for mod in model.modules():
        for bn, buf in list(mod.named_buffers(recurse=False)):
            if bn != 'position_ids' or buf.numel() == 0:
                continue
            flat = buf.flatten()
            k = min(16, flat.numel())
            if not torch.equal(flat[:k].cpu(), torch.arange(k, dtype=buf.dtype)):
                new = torch.arange(flat.numel(), device=buf.device, dtype=buf.dtype).view_as(buf)
                mod.register_buffer('position_ids', new, persistent=False)


print('Loading', EMB_NAME, flush=True)
_M = SentenceTransformer(EMB_NAME, trust_remote_code=True)
fix_position_ids(_M)


class CachingEmbedder:
    """Memoizes encode per string -> the tau sweep does not re-embed."""
    def __init__(self):
        self.cache = {}
    def encode(self, items):
        miss = [x for x in items if x not in self.cache]
        if miss:
            vecs = _M.encode(miss, convert_to_numpy=True, normalize_embeddings=False)
            for s, v in zip(miss, vecs):
                self.cache[s] = np.asarray(v, dtype=float)
        return np.array([self.cache[x] for x in items])
EMB = CachingEmbedder()


def read_rows():
    """(model, variant, category, expected) from the generated variants + the real pairs."""
    rows = []
    # variants: take the list from the gte CSV (same set for any embedder)
    csvp = os.path.join(HERE, 'results_gte-large-en-v1.5.csv')
    with open(csvp, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            base = os.path.join(VARDIR, r['model'], 'base.decl')
            var = os.path.join(VARDIR, r['model'], r['variant'] + '.decl')
            if os.path.exists(base) and os.path.exists(var):
                rows.append((r['model'], r['variant'], r['category'], r['expected'],
                             open(base, encoding='utf-8').read(), open(var, encoding='utf-8').read()))
    # real pairs: each base model vs. its pipeline reconstruction
    for root, _d, files in os.walk(BASE):
        for f in sorted(files):
            if f.endswith('.decl') and 'reconstructed' not in f:
                o = os.path.join(root, f); rc = o.replace('.decl', '_reconstructed.decl')
                if os.path.exists(rc):
                    rows.append((f[:-5].strip(), 'REALPAIR', 'REALPAIR', '(total~0)',
                                 open(o, encoding='utf-8').read(), open(rc, encoding='utf-8').read()))
    return rows


def is_sens_pass(expected, res):
    p = res['perspectives']
    if expected == 'ANY_DATA':
        return (p.get('data_conditions') or 0) > 0 or (p.get('attr_domains') or 0) > 0
    if expected == 'ANY':
        return res['total'] > 1e-9
    return (p.get(expected) or 0) > 1e-9


ROWS = read_rows()
TAUS = [None, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
WHITENS = [None, 'center']

out = []
for whiten in WHITENS:
    for tau in TAUS:
        agg = {'relabel_d': [], 'inv_easy': [], 'inv_hard': [], 'inv_hard_mean': [],
               'sens': [], 'realpair': []}
        for model, variant, cat, expected, bt, vt in ROWS:
            res = compute_semantic_distance(bt, vt, EMB, tau=tau, whiten=whiten)
            t = res['total']
            if cat == 'REALPAIR':
                agg['realpair'].append(t)
            elif cat == 'LIMITATION':                   # relabel_single
                agg['relabel_d'].append(1.0 if t > 1e-9 else 0.0)
            elif cat == 'SENSITIVITY':
                agg['sens'].append(1.0 if is_sens_pass(expected, res) else 0.0)
            elif cat == 'INVARIANCE':
                ok = 1.0 if t <= 0.10 else 0.0
                if variant in INV_EASY:
                    agg['inv_easy'].append(ok)
                elif variant in INV_HARD:
                    agg['inv_hard'].append(ok); agg['inv_hard_mean'].append(t)
        row = {
            'embedder': SHORT, 'tau': 'inf' if tau is None else tau, 'whiten': whiten or 'none',
            'relabel_detect': round(np.mean(agg['relabel_d']), 3),
            'inv_easy_ok': round(np.mean(agg['inv_easy']), 3),
            'inv_hard_ok': round(np.mean(agg['inv_hard']), 3),
            'inv_hard_mean': round(np.mean(agg['inv_hard_mean']), 3),
            'sens_ok': round(np.mean(agg['sens']), 3),
            'realpair_max': round(max(agg['realpair']), 3),
        }
        out.append(row)

# write CSV
csvpath = os.path.join(HERE, 'tau_sweep_%s.csv' % SHORT)
with open(csvpath, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow(r)

# table
print('\nembedder=%s   (relabel_detect and inv_easy_ok: HIGHER is better; realpair_max must be 0)\n' % SHORT)
print('%-8s %-7s | relabel_det inv_easy inv_hard inv_hard_mean sens_ok realpair' % ('whiten', 'tau'))
print('-' * 88)
for r in out:
    print('%-8s %-7s |   %5.3f      %5.3f    %5.3f     %5.3f       %5.3f    %5.3f' % (
        r['whiten'], str(r['tau']), r['relabel_detect'], r['inv_easy_ok'],
        r['inv_hard_ok'], r['inv_hard_mean'], r['sens_ok'], r['realpair_max']))
print('\nCSV:', csvpath)
