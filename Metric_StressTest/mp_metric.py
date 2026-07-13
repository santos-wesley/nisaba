# NOTE: mirror of cell 64 of Nisaba_(BISE_Journal).ipynb. Adopted configuration:
# tau=0.2 (cost threshold in the pairing) + whitening off. `whiten` is an optional
# knob. Keep this file in sync with cell 64.
# -*- coding: utf-8 -*-
"""
MULTI-PERSPECTIVE semantic distance for MP-Declare models, built ON TOP of the
official declare4py parser (DeclareModel).

Motivation for the embedding (inherited from the first metric): the round-trip
model->NL->model may introduce SYNONYMS/paraphrases (e.g., "Wood Cutting" ->
"Cut Wood"). Hence we align, via embedding + assignment (Jonker-Volgenant), ALL
the textual tokens where that makes sense:
  - ACTIVITY names       (act_map)
  - ATTRIBUTE names      (attr_map)
  - categorical VALUES   (val_map): enumeration members and the textual right-hand
                                    side of "is" conditions
Numbers (thresholds, ranges), operators, time windows and the TEMPLATE vocabulary
are compared EXACTLY (formal semantics must not be fuzzy).

Perspectives (each with its own Jaccard distance):
  control_flow    : (template, normalized-activities)
  temporal        : (constraint, time-window)
  data_conditions : (constraint, slot, role, normalized-attribute, op, normalized-value)
  binds           : (normalized-activity, normalized-attribute)
  attr_domains    : (normalized-attribute, canonical-domain)   [enum remapped via val_map]

Total = WEIGHTED mean of the ACTIVE perspectives (non-empty union in some model);
weights are customizable via DEFAULT_WEIGHTS or the `weights` argument. Perspectives
that are not applicable (empty in both models) are ignored (excluded from the mean).
"""
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from Declare4Py.ProcessModels.DeclareModel import DeclareModel


# ------------------------- embedding-based alignment -------------------------
def _label_stream():
    import string
    i = 0
    while True:
        n, s = i, ""
        while True:
            s = string.ascii_uppercase[n % 26] + s
            n = n // 26 - 1
            if n < 0:
                break
        yield s
        i += 1


def _l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def semantic_pairing(list_A, list_B, embedding_model, tau=None, whiten=None):
    """Minimum-cost 1-to-1 pairing (cosine distance) via rectangular
    Jonker-Volgenant. Returns (pairs, unpaired_A, unpaired_B).

    knobs (default None = original behavior):
      tau    : REJECT pairs whose cost (1-cos) > tau -> they stay unpaired (avoids
               the spurious match forced by the bijection).
      whiten : 'center' subtracts the mean of {A union B} before normalizing
               (de-anisotropizes the cosine space)."""
    if not list_A or not list_B:
        return [], list(list_A), list(list_B)
    eA = np.asarray(embedding_model.encode(list_A), dtype=float)
    eB = np.asarray(embedding_model.encode(list_B), dtype=float)
    if whiten == 'center':
        mu = np.vstack([eA, eB]).mean(axis=0, keepdims=True)
        eA, eB = eA - mu, eB - mu
    eA, eB = _l2(eA), _l2(eB)
    cost = 1 - eA @ eB.T
    row, col = linear_sum_assignment(cost)
    pairs = []
    up_A_idx, up_B_idx = set(range(len(list_A))), set(range(len(list_B)))
    for i, j in zip(row, col):
        if tau is None or cost[i, j] <= tau:
            pairs.append((list_A[i], list_B[j], float(cost[i, j])))
            up_A_idx.discard(int(i)); up_B_idx.discard(int(j))
        # else: pair rejected by tau -> both remain unpaired
    up_A = [list_A[i] for i in sorted(up_A_idx)]
    up_B = [list_B[j] for j in sorted(up_B_idx)]
    return pairs, up_A, up_B


def build_mapping(list_A, list_B, embedding_model, tau=None, whiten=None):
    """N: paired -> common label; unpaired -> unique label. Takes the two token
    lists (one per model) and returns a dict token->label."""
    pairs, up_A, up_B = semantic_pairing(list_A, list_B, embedding_model, tau=tau, whiten=whiten)
    labels = _label_stream()
    m = {}
    for a, b, _ in pairs:
        lab = next(labels)
        m[a] = lab
        m[b] = lab
    for x in list(up_A) + list(up_B):
        m[x] = next(labels)
    return m


# ------------------------- loading via declare4py -------------------------
_OP = {'>': 'gt', '<': 'lt', '>=': 'ge', '<=': 'le', '=': 'eq', '==': 'eq',
       'is': 'eq', 'is not': 'ne', '!=': 'ne', 'in': 'in'}
_ATOM_RE = re.compile(r'^([ATB])\.(.+?)\s*(>=|<=|!=|==|=|>|<|is not|is|in)\s*(.+)$', re.I)


def _sanitize_decl(text):
    """Works around two declare4py fragilities on imperfect models:
      (1) binds WITHOUT an attribute ('bind X:' after strip) break split(': ');
      (2) domain lines ('Attr: value') whose attribute was never bound by a bind
          raise 'Unable to find attribute'.
    An empty bind carries no data; an orphan domain references a non-existent
    attribute -> both lines are dropped (no useful info for the metric)."""
    lines = text.split("\n")
    bound = set()
    for line in lines:
        s = line.strip()
        if s.startswith("bind ") and ":" in s:
            for a in s.split(":", 1)[1].split(","):
                a = a.strip()
                if a:
                    bound.add(a)
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("bind "):
            after = s.split(":", 1)[1].strip() if ":" in s else ""
            if after == "":
                continue                      # (1) empty bind
            out.append(line)
            continue
        if (":" in s and "[" not in s and not s.startswith("activity ")
                and DeclareModel.is_events_attrs_value_definition(s)):
            lhs = [a.strip() for a in s.split(":", 1)[0].split(",")]
            if any(a not in bound for a in lhs):
                continue                      # (2) orphan domain
        out.append(line)
    return "\n".join(out)


def _norm_val(v):
    """Normalize a condition value. Numeric -> float; otherwise lowercased string."""
    v = v.strip().strip('()').strip()
    try:
        return ('num', float(v))
    except ValueError:
        return ('str', v.lower())


def _parse_atoms(cond):
    """Split a condition into atoms (role, attr, op, ('num'|'str', value))."""
    if not cond:
        return []
    atoms = []
    for part in re.split(r'\s+(?:and|or)\s+', cond.strip(), flags=re.I):
        p = part.strip().strip('()').strip()
        if not p or p.lower() == 'true':
            continue
        m = _ATOM_RE.match(p)
        if m:
            role = m.group(1).upper()
            role = 'T' if role == 'B' else role
            attr = m.group(2).strip()
            op = _OP.get(m.group(3).lower().strip(), m.group(3).lower().strip())
            atoms.append((role, attr, op, _norm_val(m.group(4))))
        else:
            atoms.append(('?', p.lower(), '', ('str', p.lower())))
    return atoms


def _canonical_domain(attr):
    """Canonical domain. For enum, keep the raw (lowercased) members for later
    remapping via val_map; numbers stay exact."""
    av = getattr(attr, 'attr_value', None)
    if av is None:
        return ('none',)
    t = str(av.attribute_value_type)
    try:
        if t in ('integer_range', 'float_range'):
            return (t, float(av.value[0]), float(av.value[1]))
        if t == 'enumeration':
            return ('enum', frozenset(tok.get_name().strip().lower() for tok in av.value))
    except Exception:
        pass
    return (t, str(av.value_original).strip().lower())


def load_model(text):
    """Parse a .decl (string) via declare4py and extract the raw perspectives,
    also collecting the universes of attribute names and categorical values."""
    m = DeclareModel().parse_from_string(_sanitize_decl(text))
    pm = m.parsed_model

    activities = list(m.activities)

    binds = {}
    for _et, evs in pm.events.items():
        for en, ev in evs.items():
            binds[en] = set(ev.attributes.keys())

    domains = {an: _canonical_domain(a) for an, a in pm.attributes_list.items()}

    constraints, cond_attr_names, value_tokens = [], set(), set()
    for _i, t in pm.templates.items():
        acts = [e.get_event_name() for e in t.events_activities if e is not None]
        slots = {}
        for slot, raw in (('act', t.get_activation_condition()),
                          ('tgt', t.get_target_condition())):
            atoms = _parse_atoms(raw)
            slots[slot] = atoms
            for (_role, attr, _op, val) in atoms:
                cond_attr_names.add(attr)
                if val[0] == 'str':
                    value_tokens.add(val[1])
        tm = t.get_time_condition()
        time = None
        if tm and tm.strip():
            parts = [x.strip() for x in tm.split(',')]
            try:
                time = (float(parts[0]), float(parts[1]), parts[2].lower())
            except Exception:
                time = tuple(parts)
        constraints.append({'template': t.get_template_name(), 'acts': acts,
                            'slots': slots, 'time': time})

    # categorical values: enumeration members + textual RHS of conditions
    for dom in domains.values():
        if dom and dom[0] == 'enum':
            value_tokens |= set(dom[1])

    attr_names = set(domains) | {a for s in binds.values() for a in s} | cond_attr_names
    return {'activities': activities, 'binds': binds, 'domains': domains,
            'constraints': constraints, 'attr_names': sorted(attr_names),
            'value_tokens': sorted(value_tokens)}


# ------------------------- perspectives + distance -------------------------
def _jaccard(s1, s2):
    if not s1 and not s2:
        return None            # perspective not applicable (empty in both)
    return 1 - len(s1 & s2) / len(s1 | s2)


def _perspective_sets(model, act_map, attr_map, val_map):
    A = lambda x: act_map.get(x, x)
    T = lambda x: attr_map.get(x, x)
    V = lambda x: val_map.get(x, x)

    def norm_value(val):
        kind, v = val
        return ('num', v) if kind == 'num' else ('str', V(v))

    control, temporal, data, binds, domains = set(), set(), set(), set(), set()

    for c in model['constraints']:
        cid = (c['template'], tuple(A(a) for a in c['acts']))
        control.add(cid)
        if c['time'] is not None:
            temporal.add((cid, c['time']))
        for slot, atoms in c['slots'].items():
            for (role, attr, op, val) in atoms:
                data.add((cid, slot, role, T(attr), op, norm_value(val)))

    for act, attrs in model['binds'].items():
        for at in attrs:
            binds.add((A(act), T(at)))

    for an, dom in model['domains'].items():
        if dom and dom[0] == 'enum':
            dom = ('enum', frozenset(V(v) for v in dom[1]))   # remap synonyms
        domains.add((T(an), dom))

    return {'control_flow': control, 'temporal': temporal,
            'data_conditions': data, 'binds': binds, 'attr_domains': domains}


DEFAULT_WEIGHTS = {'control_flow': 1.0, 'temporal': 1.0, 'data_conditions': 1.0,
                   'binds': 1.0, 'attr_domains': 1.0}

# Cost threshold (1-cos) adopted in the pairing: reject dissimilar matches instead
# of forcing them through the bijection (closes the blindness to unrelated relabels).
# Calibrated in Metric_StressTest/tau_sweep.py for gte-large (compressed space).
# It does NOT affect identical/real pairs (cost 0 is never rejected). Use None to disable.
DEFAULT_TAU = 0.2


def compute_semantic_distance(decl_original, decl_reconstructed, embedding_model,
                              weights=None, tau=DEFAULT_TAU, whiten=None):
    """Multi-perspective distance. Returns a dict with 'total', 'perspectives', 'weights'.
    'total' is the WEIGHTED mean of the active perspectives. Weights are customizable:
    pass `weights` (dict perspective->weight) or edit DEFAULT_WEIGHTS. Default weights
    (1.0) => plain arithmetic mean.
    tau: cost threshold in the pairing (default DEFAULT_TAU=0.2; pass None to disable).
    whiten: 'center' de-anisotropizes the embedding space (see semantic_pairing)."""
    weights = weights or DEFAULT_WEIGHTS
    mo = load_model(decl_original)
    mr = load_model(decl_reconstructed)

    act_map = build_mapping(mo['activities'], mr['activities'], embedding_model, tau=tau, whiten=whiten)
    attr_map = build_mapping(mo['attr_names'], mr['attr_names'], embedding_model, tau=tau, whiten=whiten)
    val_map = build_mapping(mo['value_tokens'], mr['value_tokens'], embedding_model, tau=tau, whiten=whiten)

    so = _perspective_sets(mo, act_map, attr_map, val_map)
    sr = _perspective_sets(mr, act_map, attr_map, val_map)

    persp = {k: _jaccard(so[k], sr[k]) for k in so}

    active = {k: v for k, v in persp.items() if v is not None}
    if not active:
        total = 0.0
    else:
        wsum = sum(weights.get(k, 1.0) for k in active)
        total = sum(weights.get(k, 1.0) * v for k, v in active.items()) / wsum

    return {'total': total, 'perspectives': persp, 'weights': dict(weights)}


def compute_semantic_similarity(decl_original, decl_reconstructed, embedding_model):
    """Compat: returns only the total (float), drop-in for the previous metric."""
    return compute_semantic_distance(decl_original, decl_reconstructed, embedding_model)['total']
