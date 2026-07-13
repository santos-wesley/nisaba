# -*- coding: utf-8 -*-
"""
Stress-test of the multi-perspective semantic distance metric.

Generates controlled VARIANTS of each real .decl model (base_models/) and
evaluates the base-vs-variant distance, to expose the metric's behavior.

The framing is the real use case: the metric scores models RECONSTRUCTED from a
natural-language description (round-trip model -> NL -> model). Two opposite
properties are wanted:

  INVARIANCE (meaning preserved -> distance should be ~0):
    recon_reorder   : LLM reorders the label words (Wood Cutting -> Cutting Wood)
    recon_paraphrase: synonym + reordering together
    syn_names       : activities/attributes renamed by SYNONYMS (word-level)
    syn_values      : enum/condition values replaced by synonyms
    syn_all         : both
    enum_synonym    : one enumeration member -> synonym
    reorder         : constraints reordered (Jaccard is a set -> exactly 0)

  SENSITIVITY (meaning changed -> the expected perspective must fire):
    tmpl_confusion  : LLM confuses close templates (Responded Existence -> Response)
    drop_condition  : LLM omits the data/time payload of a constraint
    neg_template    : positive template -> negative (Response -> Not Response)
    swap_acts       : swap activation<->target in a binary constraint
    cardinality     : Existence -> Existence2
    drop_constr     : remove the last constraint
    add_constr      : add a spurious constraint
    op_flip         : flip a condition operator (> -> <)
    val_antonym     : condition/enum value -> ANTONYM
    domain_range    : change the upper bound of a numeric range
    time_window     : change the time window (max)
    time_unit       : change the time unit (d -> h)
    rebind          : move an attribute to another activity
    drop_bind       : remove a bind

  LIMITATION (documented):
    relabel_single  : rename one activity to an UNRELATED term (isomorphism -> the
                      metric is blind at tau=off; detected with the adopted tau=0.2)

Usage: [STRESS_EMB=<model>] [STRESS_TAU=<float|off>] python generate_and_evaluate.py

Note: the SHIPPED metric uses tau=0.2 (see mp_metric.DEFAULT_TAU). This script
characterizes the RAW metric with tau OFF by default (STRESS_TAU unset -> tau=None)
to expose the limitations that motivate the mitigation; tau_sweep.py justifies the
adopted tau=0.2. Set STRESS_TAU=0.2 to reproduce the shipped configuration.

Output: variants/<model>/<variant>.decl + results_<embedder>.csv (a summary is printed).
"""
import os, re, csv, random, logging
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
for _n in ['httpcore', 'httpx', 'urllib3', 'filelock', 'huggingface_hub', 'sentence_transformers']:
    logging.getLogger(_n).setLevel(logging.ERROR)
logging.disable(logging.INFO)

import numpy as np
from mp_metric import compute_semantic_distance, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'base_models')  # eight calibration models (.decl) + their pipeline reconstructions (REALPAIR)
VARDIR = os.path.join(HERE, 'variants')
random.seed(42)

# cost threshold used for the evaluation. Default None (raw characterization);
# set STRESS_TAU=0.2 to evaluate under the shipped configuration.
_TAU_ENV = os.environ.get('STRESS_TAU')
TAU = None if _TAU_ENV in (None, '', 'off', 'none', 'None', 'inf') else float(_TAU_ENV)

# --------------------------- embedder ---------------------------
# Official paper embedder: Alibaba-NLP/gte-large-en-v1.5. The gte-v1.5 custom code
# leaves the non-persistent `position_ids` buffer unmaterialized on meta-load (recent
# transformers) -> IndexError in RoPE. Fix: re-register the buffer with arange after
# loading. Switch models with the STRESS_EMB env.
import torch
from sentence_transformers import SentenceTransformer
_EMB_NAME = os.environ.get('STRESS_EMB', 'Alibaba-NLP/gte-large-en-v1.5')
print('Loading embedder:', _EMB_NAME, '| tau =', TAU, flush=True)
def _fix_position_ids(model):
    """Re-materialize `position_ids` buffers left with GARBAGE by the meta-load
    (gte-v1.5 bug). Only touches actually-corrupted buffers and PRESERVES the shape,
    so it does not break standard models (BERT uses a valid [1, N])."""
    fixed = 0
    for mod in model.modules():
        for bn, buf in list(mod.named_buffers(recurse=False)):
            if bn != 'position_ids' or buf.numel() == 0:
                continue
            flat = buf.flatten()
            k = min(16, flat.numel())
            expected = torch.arange(k, dtype=buf.dtype)
            if not torch.equal(flat[:k].cpu(), expected):     # corrupted buffer
                new = torch.arange(flat.numel(), device=buf.device, dtype=buf.dtype).view_as(buf)
                mod.register_buffer('position_ids', new, persistent=False)
                fixed += 1
    return fixed

_M = SentenceTransformer(_EMB_NAME, trust_remote_code=True)
_fx = _fix_position_ids(_M)
if _fx:
    print('  (fix: position_ids re-registered in %d module(s))' % _fx, flush=True)

class Embedder:
    def encode(self, items):
        return _M.encode(list(items), convert_to_numpy=True, normalize_embeddings=False)
EMB = Embedder()

# --------------------------- thesauri ---------------------------
SYN_WORDS = {
    'order': 'purchase', 'approve': 'authorize', 'request': 'requisition', 'receive': 'accept',
    'inspect': 'examine', 'install': 'set up', 'train': 'educate', 'use': 'operate',
    'maintain': 'service', 'decommission': 'retire', 'gather': 'collect', 'analyze': 'examine',
    'draft': 'compose', 'review': 'assess', 'revise': 'edit', 'publish': 'release',
    'create': 'formulate', 'implement': 'execute', 'collect': 'gather', 'monitor': 'track',
    'schedule': 'arrange', 'report': 'document', 'update': 'revise', 'evaluate': 'assess',
    'perform': 'conduct', 'communicate': 'convey', 'investigate': 'probe', 'certify': 'accredit',
    'conduct': 'perform', 'status': 'state', 'cost': 'price', 'score': 'rating', 'type': 'category',
    'id': 'identifier', 'level': 'degree', 'hours': 'duration', 'frequency': 'rate',
    'reason': 'cause', 'condition': 'state', 'impact': 'effect', 'feedback': 'response',
    'revision': 'edit', 'approval': 'authorization', 'data': 'information', 'equipment': 'apparatus',
    'staff': 'personnel', 'action': 'measure', 'actions': 'measures', 'plan': 'scheme',
    'safety': 'security', 'incident': 'event', 'severity': 'seriousness', 'compliance': 'conformity',
    'risk': 'hazard', 'budget': 'funds', 'allocation': 'assignment', 'breach': 'violation',
    'audit': 'review', 'drill': 'exercise', 'participants': 'attendees', 'measures': 'controls',
    'policies': 'guidelines', 'performance': 'efficiency', 'installation': 'setup',
    'maintenance': 'servicing', 'usage': 'utilization', 'training': 'instruction',
    'inspection': 'examination', 'certification': 'accreditation',
}
SYN_VALUES = {
    'approved': 'accepted', 'completed': 'finished', 'damaged': 'broken', 'obsolete': 'outdated',
    'positive': 'favorable', 'negative': 'unfavorable', 'neutral': 'impartial', 'minor': 'small',
    'major': 'large', 'environmental': 'ecological', 'social': 'societal', 'high': 'elevated',
    'low': 'reduced', 'medium': 'moderate', 'up to date': 'current',
}
ANT_VALUES = {
    'approved': 'rejected', 'completed': 'pending', 'damaged': 'intact', 'obsolete': 'current',
    'positive': 'negative', 'up to date': 'outdated', 'high': 'low', 'low': 'high',
    'minor': 'major', 'major': 'minor',
}
NEG_TEMPLATES = [
    ('Chain Response', 'Not Chain Response'), ('Chain Precedence', 'Not Chain Precedence'),
    ('Chain Succession', 'Not Chain Succession'), ('Responded Existence', 'Not Responded Existence'),
    ('Co-Existence', 'Not Co-Existence'), ('Succession', 'Not Succession'),
    ('Precedence', 'Not Precedence'), ('Response', 'Not Response'),
]
UNRELATED = ['Banana Harvest', 'Volcano Survey', 'Poetry Reading', 'Guitar Tuning', 'Cloud Watching']


def _syn_word(w):
    lw = w.lower()
    if lw in SYN_WORDS:
        rep = SYN_WORDS[lw]
        return rep.capitalize() if w[:1].isupper() else rep
    return w


def _synonymize_name(name):
    return ' '.join(_syn_word(w) for w in name.split(' '))


def _replace_exact(text, mapping):
    """Replace exact names in a single pass (longest first), no cascading."""
    mapping = {k: v for k, v in mapping.items() if v and v != k}
    if not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile('|'.join(re.escape(k) for k in keys))
    return pat.sub(lambda m: mapping[m.group(0)], text)


# --------------------------- mutators ---------------------------
def m_syn_names(text):
    mdl = load_model(text)
    names = set(mdl['activities']) | set(mdl['attr_names'])
    mapping = {n: _synonymize_name(n) for n in names}
    out = _replace_exact(text, mapping)
    return out if out != text else None

def m_syn_values(text):
    return _generic_word_sub(text, SYN_VALUES)

def m_syn_all(text):
    t = m_syn_names(text)
    t = m_syn_values(t or text)
    return t if t and t != text else (m_syn_names(text))

def _generic_word_sub(text, wordmap):
    """Replace words (values) preserving the rest; case-insensitive per word."""
    keys = sorted(wordmap, key=len, reverse=True)
    pat = re.compile(r'(?<![A-Za-z])(' + '|'.join(re.escape(k) for k in keys) + r')(?![A-Za-z])', re.I)
    def rep(m):
        v = wordmap[m.group(0).lower()]
        return v.capitalize() if m.group(0)[:1].isupper() else v
    out = pat.sub(rep, text)
    return out if out != text else None

def m_reorder(text):
    lines = text.split('\n')
    cons = [i for i, l in enumerate(lines) if '[' in l and ']' in l]
    if len(cons) < 2:
        return None
    block = [lines[i] for i in cons]
    shuffled = block[:]
    random.shuffle(shuffled)
    if shuffled == block:
        shuffled = block[::-1]
    for pos, i in enumerate(cons):
        lines[i] = shuffled[pos]
    return '\n'.join(lines)

def m_neg_template(text):
    for pos, neg in NEG_TEMPLATES:
        # constraint at line start, positive template (not already "Not ...")
        pat = re.compile(r'(?m)^(' + re.escape(pos) + r')(\[)')
        if pat.search(text):
            return pat.sub(neg + r'\2', text, count=1)
    return None

def m_swap_acts(text):
    pat = re.compile(r'(?m)^([^\[\n]+\[)([^,\]]+),\s*([^,\]]+)(\])')
    def rep(m):
        return m.group(1) + m.group(3).strip() + ', ' + m.group(2).strip() + m.group(4)
    out, n = pat.subn(rep, text, count=1)
    return out if n else None

def m_cardinality(text):
    # Existence[ -> Existence2[  ; Existence2[ -> Existence3[
    pat = re.compile(r'(?m)^(Existence|Absence)(\d*)(\[)')
    mm = pat.search(text)
    if not mm:
        return None
    cur = int(mm.group(2)) if mm.group(2) else 1
    return text[:mm.start()] + mm.group(1) + str(cur + 1) + mm.group(3) + text[mm.end():]

def m_drop_constr(text):
    lines = text.split('\n')
    cons = [i for i, l in enumerate(lines) if '[' in l and ']' in l]
    if not cons:
        return None
    del lines[cons[-1]]
    return '\n'.join(lines)

def m_add_constr(text):
    mdl = load_model(text)
    if len(mdl['activities']) < 2:
        return None
    a, b = mdl['activities'][0], mdl['activities'][1]
    return text.rstrip('\n') + '\nChoice[' + a + ', ' + b + '] | | |\n'

def m_op_flip(text):
    if re.search(r'>=|<=', text):
        return re.sub(r'>=', '<=', text, count=1) if '>=' in text else re.sub(r'<=', '>=', text, count=1)
    if re.search(r'(?<![<>])>(?!=)', text):
        return re.sub(r'(?<![<>])>(?!=)', '<', text, count=1)
    if re.search(r'(?<![<>])<(?!=)', text):
        return re.sub(r'(?<![<>])<(?!=)', '>', text, count=1)
    return None

def m_val_antonym(text):
    return _generic_word_sub(text, ANT_VALUES)

def m_domain_range(text):
    pat = re.compile(r'(between\s+[+-]?\d+(?:\.\d+)?\s+and\s+)([+-]?\d+)(?:\.\d+)?', re.I)
    mm = pat.search(text)
    if not mm:
        return None
    newval = str(int(mm.group(2)) - 1)
    return text[:mm.start(2)] + newval + text[mm.end(2):]

def m_domain_enum(text):
    # replace 1 enum member by a textual synonym in a domain line (attr: v1, v2, v3)
    for k, v in list(SYN_VALUES.items()) + list(ANT_VALUES.items()):
        pat = re.compile(r'(?<![A-Za-z])(' + re.escape(k) + r')(?![A-Za-z])', re.I)
        # only on domain lines (with ':' and without '[')
        for line in text.split('\n'):
            if ':' in line and '[' not in line and not line.startswith('activity') and not line.startswith('bind'):
                if pat.search(line):
                    newline = pat.sub(v.capitalize() if line[line.lower().index(k)].isupper() else v, line, count=1)
                    return text.replace(line, newline, 1)
    return None

def m_time_window(text):
    pat = re.compile(r'(\d+),(\d+),([smhd])')
    mm = pat.search(text)
    if not mm:
        return None
    newmax = str(int(mm.group(2)) + 3)
    return text[:mm.start()] + mm.group(1) + ',' + newmax + ',' + mm.group(3) + text[mm.end():]

def m_time_unit(text):
    pat = re.compile(r'(\d+,\d+,)([smhd])')
    mm = pat.search(text)
    if not mm:
        return None
    newu = {'d': 'h', 'h': 'd', 'm': 's', 's': 'm'}[mm.group(2)]
    return text[:mm.start()] + mm.group(1) + newu + text[mm.end():]

def m_rebind(text):
    lines = text.split('\n')
    binds = [i for i, l in enumerate(lines) if l.startswith('bind ') and ':' in l and l.split(':', 1)[1].strip()]
    acts = [l[9:].strip() for l in lines if l.startswith('activity ')]
    if len(binds) < 1 or len(acts) < 2:
        return None
    i = binds[0]
    cur_act = lines[i][5:].split(':', 1)[0].strip()
    other = next((a for a in acts if a != cur_act), None)
    if not other:
        return None
    attrs = lines[i].split(':', 1)[1]
    lines[i] = 'bind ' + other + ':' + attrs
    return '\n'.join(lines)

def m_drop_bind(text):
    lines = text.split('\n')
    binds = [i for i, l in enumerate(lines) if l.startswith('bind ') and ':' in l and l.split(':', 1)[1].strip()]
    if not binds:
        return None
    del lines[binds[0]]
    return '\n'.join(lines)

def m_rename_unrel(text):
    mdl = load_model(text)
    if not mdl['activities']:
        return None
    a = mdl['activities'][0]
    return _replace_exact(text, {a: UNRELATED[0]})


# ---- REALISTIC LLM reconstruction artifacts (round-trip model -> NL -> model) ----
def _reorder_words(name):
    ws = name.split(' ')
    if len(ws) == 2:
        return ws[1] + ' ' + ws[0]           # "Wood Cutting" -> "Cutting Wood"
    if len(ws) >= 3:
        return ' '.join(ws[1:] + ws[:1])      # rotate
    return name

def m_recon_reorder(text):
    """LLM reorders the label words (Wood Cutting -> Cutting Wood)."""
    mdl = load_model(text)
    names = set(mdl['activities']) | set(mdl['attr_names'])
    mapping = {n: _reorder_words(n) for n in names if len(n.split(' ')) >= 2}
    out = _replace_exact(text, mapping)
    return out if out != text else None

def m_recon_paraphrase(text):
    """Most realistic case: synonym + reordering together."""
    mdl = load_model(text)
    names = set(mdl['activities']) | set(mdl['attr_names'])
    mapping = {n: _reorder_words(_synonymize_name(n)) for n in names}
    out = _replace_exact(text, mapping)
    return out if out != text else None

def m_tmpl_confusion(text):
    """LLM confuses semantically close templates (should be DETECTED)."""
    pairs = [('Responded Existence', 'Response'), ('Chain Response', 'Response'),
             ('Alternate Response', 'Response'), ('Response', 'Responded Existence'),
             ('Chain Precedence', 'Precedence'), ('Alternate Precedence', 'Precedence')]
    for a, b in pairs:
        pat = re.compile(r'(?m)^' + re.escape(a) + r'\[')
        if pat.search(text):
            return pat.sub(b + '[', text, count=1)
    return None

def m_drop_condition(text):
    """LLM omits the data/time payload of a constraint in the description (completeness loss)."""
    for line in text.split('\n'):
        if '[' in line and ']' in line:
            after = line.split(']', 1)[1]
            if after.replace('|', '').strip():        # has data/time payload
                newline = line.split(']', 1)[0] + '] | | |'
                return text.replace(line, newline, 1)
    return None


MUTATORS = [
    # (name, function, category, expected_perspective)
    # --- LLM reconstruction artifacts that PRESERVE meaning (should be ~0) ---
    ('recon_reorder', m_recon_reorder, 'INVARIANCE', None),
    ('recon_paraphrase', m_recon_paraphrase, 'INVARIANCE', None),
    ('syn_names', m_syn_names, 'INVARIANCE', None),
    ('syn_values', m_syn_values, 'INVARIANCE', None),
    ('syn_all', m_syn_all, 'INVARIANCE', None),
    ('enum_synonym', m_domain_enum, 'INVARIANCE', None),
    ('reorder', m_reorder, 'INVARIANCE', None),
    # --- reconstruction artifacts that LOSE/ALTER meaning (should be detected) ---
    ('tmpl_confusion', m_tmpl_confusion, 'SENSITIVITY', 'control_flow'),
    ('drop_condition', m_drop_condition, 'SENSITIVITY', 'ANY'),
    ('neg_template', m_neg_template, 'SENSITIVITY', 'control_flow'),
    ('swap_acts', m_swap_acts, 'SENSITIVITY', 'control_flow'),
    ('cardinality', m_cardinality, 'SENSITIVITY', 'control_flow'),
    ('drop_constr', m_drop_constr, 'SENSITIVITY', 'control_flow'),
    ('add_constr', m_add_constr, 'SENSITIVITY', 'control_flow'),
    ('op_flip', m_op_flip, 'SENSITIVITY', 'data_conditions'),
    ('val_antonym', m_val_antonym, 'SENSITIVITY', 'ANY_DATA'),
    ('domain_range', m_domain_range, 'SENSITIVITY', 'attr_domains'),
    ('time_window', m_time_window, 'SENSITIVITY', 'temporal'),
    ('time_unit', m_time_unit, 'SENSITIVITY', 'temporal'),
    ('rebind', m_rebind, 'SENSITIVITY', 'binds'),
    ('drop_bind', m_drop_bind, 'SENSITIVITY', 'binds'),
    # Documented LIMITATION: a single consistent relabel = isomorphism -> the metric
    # is blind at tau=off (total~0); with the adopted tau=0.2 it becomes detected.
    ('relabel_single', m_rename_unrel, 'LIMITATION', None),
]

INV_THRESHOLD = 0.10   # invariance: total should stay below this


def base_models():
    out = []
    for root, _d, files in os.walk(BASE):
        for f in sorted(files):
            if f.endswith('.decl') and 'reconstructed' not in f:
                out.append((f[:-5].strip(), os.path.join(root, f)))
    return out


def main():
    rows = []
    for name, path in base_models():
        base = open(path, encoding='utf-8').read()
        outdir = os.path.join(VARDIR, name)
        os.makedirs(outdir, exist_ok=True)
        open(os.path.join(outdir, 'base.decl'), 'w', encoding='utf-8').write(base)
        for vname, fn, cat, persp in MUTATORS:
            try:
                variant = fn(base)
            except Exception:
                variant = None
            if not variant or variant == base:
                continue
            # verify the variant still parses; otherwise, drop it
            try:
                load_model(variant)
            except Exception:
                continue
            open(os.path.join(outdir, vname + '.decl'), 'w', encoding='utf-8').write(variant)
            res = compute_semantic_distance(base, variant, EMB, tau=TAU)
            p = res['perspectives']
            if cat in ('INVARIANCE', 'LIMITATION'):
                ok = res['total'] <= INV_THRESHOLD
            elif persp == 'ANY_DATA':
                ok = (p.get('data_conditions') or 0) > 0 or (p.get('attr_domains') or 0) > 0
            elif persp == 'ANY':
                ok = res['total'] > 1e-9
            else:
                ok = (p.get(persp) or 0) > 1e-9
            rows.append({
                'model': name, 'variant': vname, 'category': cat,
                'expected': persp or '(total~0)', 'total': round(res['total'], 4),
                'control_flow': p['control_flow'], 'temporal': p['temporal'],
                'data_conditions': p['data_conditions'], 'binds': p['binds'],
                'attr_domains': p['attr_domains'], 'pass': ok,
            })

    # write CSV (name includes the embedder for side-by-side comparison)
    csvpath = os.path.join(HERE, 'results_%s.csv' % _EMB_NAME.split('/')[-1])
    with open(csvpath, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary
    def fmt(v):
        return '  .  ' if v is None else ('%.3f' % v)
    print('\n%-9s %-13s %-11s %-15s %6s | %-6s %-6s %-6s %-6s %-6s | %s' % (
        'model', 'variant', 'category', 'expected', 'total', 'ctrl', 'temp', 'data', 'bind', 'dom', 'ok'))
    print('-' * 110)
    for r in rows:
        print('%-9s %-13s %-11s %-15s %6.3f | %-6s %-6s %-6s %-6s %-6s | %s' % (
            r['model'], r['variant'], r['category'][:11], r['expected'][:15], r['total'],
            fmt(r['control_flow']), fmt(r['temporal']), fmt(r['data_conditions']),
            fmt(r['binds']), fmt(r['attr_domains']), 'PASS' if r['pass'] else 'FAIL'))

    inv = [r for r in rows if r['category'] == 'INVARIANCE']
    sen = [r for r in rows if r['category'] == 'SENSITIVITY']
    lim = [r for r in rows if r['category'] == 'LIMITATION']
    print('\nSUMMARY (tau=%s): %d variants | INVARIANCE %d/%d | SENSITIVITY %d/%d | LIMITATION %d/%d' % (
        TAU, len(rows), sum(r['pass'] for r in inv), len(inv),
        sum(r['pass'] for r in sen), len(sen), sum(r['pass'] for r in lim), len(lim)))
    print('Non-conforming:', [(r['model'], r['variant']) for r in rows
                              if (r['category'] != 'LIMITATION' and not r['pass'])] or 'none')
    print('CSV:', csvpath)


if __name__ == '__main__':
    main()
