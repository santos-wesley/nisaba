# Stress-test of the Multi-perspective Semantic Distance metric

This directory stress-tests the semantic distance metric (cell 64 of
`Nisaba_(BISE_Journal).ipynb`, mirrored in [`mp_metric.py`](mp_metric.py)) by
generating **controlled variants** of every real `.decl` model in
`base_models/` and measuring the *base → variant* distance.

The framing is the real use case: the metric scores models **reconstructed from a
natural-language description** (round-trip `model → NL → model`). It therefore needs
two opposite properties:

- **Invariance to form** — if the LLM merely rewrites labels (e.g., `Wood Cutting` →
  `Cut Wood`, `Impact Score` → `Score Rating`), the distance must stay ≈ 0.
- **Sensitivity to semantic loss** — if the LLM drops a constraint, swaps a template,
  changes a data condition, a domain range or a time window, the corresponding
  perspective must fire.

## Embedder

The **official paper embedder is `Alibaba-NLP/gte-large-en-v1.5`** and it is the
**default** of this harness. It runs locally after one fix: the gte-v1.5 custom code
leaves the non-persistent `position_ids` buffer **unmaterialized** on the recent
`transformers` meta-load (it holds garbage → `IndexError` in RoPE); `_fix_position_ids()`
re-registers the buffer with `arange` (it only touches corrupted buffers and preserves
the shape, so standard models are unaffected). Switch models with the `STRESS_EMB` env
(e.g., `sentence-transformers/all-MiniLM-L6-v2`).

## Adopted configuration

The shipped metric uses a pairing **cost threshold `tau = 0.2`** (see
`mp_metric.DEFAULT_TAU`) — see *Mitigation (a)* below. It does **not** change the
reported results (identical/real pairs have pairing cost 0 and are never rejected),
and it closes the blindness to unrelated relabels.

## How to reproduce

```bash
cd Metric_StressTest
python generate_and_evaluate.py                                    # gte-large, tau=off (raw characterization)
STRESS_EMB="sentence-transformers/all-MiniLM-L6-v2" python generate_and_evaluate.py
python tau_sweep.py                                                # calibrate tau (both embedders)
STRESS_TAU=0.2 python generate_and_evaluate.py                     # evaluate under the shipped config
```

`generate_and_evaluate.py` characterizes the **raw** metric (`tau=off`) by default to
expose the limitations that motivate the mitigation; `tau_sweep.py` justifies the
adopted `tau=0.2`. Output: `variants/<model>/<variant>.decl` (170 files) and one CSV
per embedder ([`results_gte-large-en-v1.5.csv`](results_gte-large-en-v1.5.csv),
[`results_all-MiniLM-L6-v2.csv`](results_all-MiniLM-L6-v2.csv)).

## Variant taxonomy

| Variant | Category | What it simulates | Expected |
|---|---|---|---|
| `recon_reorder` | invariance | LLM reorders the label words (Wood Cutting→Cutting Wood) | total ≈ 0 |
| `recon_paraphrase` | invariance | synonym **+** reordering (full paraphrase) | total ≈ 0 |
| `syn_names` | invariance | activities/attributes by synonyms | total ≈ 0 |
| `syn_values` / `enum_synonym` | invariance | enum/condition values by synonyms | total ≈ 0 |
| `syn_all` | invariance | names + values by synonyms | total ≈ 0 |
| `reorder` | invariance | constraints reordered (Jaccard is a set) | total = 0 |
| `tmpl_confusion` | sensitivity | LLM confuses close templates (Responded Existence→Response) | control_flow > 0 |
| `drop_condition` | sensitivity | LLM omits a constraint's data/time payload | total > 0 |
| `neg_template` | sensitivity | positive → negative template (Response→Not Response) | control_flow > 0 |
| `swap_acts` | sensitivity | swap activation↔target in a binary constraint | control_flow > 0 |
| `cardinality` | sensitivity | Existence → Existence2 | control_flow > 0 |
| `drop_constr` / `add_constr` | sensitivity | remove/add a constraint | control_flow > 0 |
| `op_flip` | sensitivity | flip a condition operator (> → <) | data_conditions > 0 |
| `val_antonym` | sensitivity | condition/enum value → **antonym** | data/domain > 0 |
| `domain_range` | sensitivity | change a numeric range bound | attr_domains > 0 |
| `time_window` / `time_unit` | sensitivity | change the time window/unit | temporal > 0 |
| `rebind` / `drop_bind` | sensitivity | move/remove an activity↔attribute binding | binds > 0 |
| `relabel_single` | **limitation** | rename 1 activity to an unrelated term | see §Limitations |

## Results (7 models, 162 applicable variants)

Raw characterization (`tau=off`):

| | gte-large-en-v1.5 (official) | all-MiniLM-L6-v2 |
|---|---|---|
| **Invariance** | 28/52 | 41/52 |
| **Sensitivity** | **101/102** | **101/102** |
| **Limitation** (blindness at tau=off) | 8/8 | 8/8 |

Extra sanity (gte-large): the **7 real pairs** original↔reconstructed give **0.000**
across all perspectives (consistent with the paper's Semantic Distance = 0 column);
models from **different domains** give **0.98–1.00** (the metric cleanly separates what
is genuinely different).

### Sensitivity is (almost) perfect — embedder-independent
Every semantic change fires **only** the right perspective. Examples (model246):

| variant | ctrl | temp | data | bind | dom |
|---|---|---|---|---|---|
| `neg_template`  | **0.133** | 0.667\* | 0 | 0 | 0 |
| `op_flip`       | 0 | 0 | **0.250** | 0 | 0 |
| `time_window`   | 0 | **0.667** | 0 | 0 | 0 |
| `rebind`        | 0 | 0 | 0 | **0.182** | 0 |
| `domain_range`  | 0 | 0 | 0 | 0 | **0.182** |
| `drop_condition`| 0 | 0 | **0.143** | 0 | 0 |

\* also touches `temporal` because the negated constraint carried a time window — the
condition↔constraint coupling is intentional.

## Mitigation (a): pairing cost threshold τ

`semantic_pairing` has a `tau` knob: after solving the assignment, it **rejects pairs
whose cost (1−cos) > τ** — both endpoints stay **unpaired** (unique label) instead of
force-matched. This targets **Limitation #2** (invisible relabel): if the "best" match
is still dissimilar, don't pretend it matches. `tau_sweep.py` sweeps τ (and the
`center` whitening) measuring the trade-off. Results:

**gte-large-en-v1.5** (compressed space → needs a tight τ)

| whiten | τ | relabel_detect | inv_easy_ok | sens_ok | realpair_max |
|---|---|---|---|---|---|
| none | ∞ (base) | 0.00 | 0.893 | 0.99 | 0 |
| none | **0.20** | **1.00** | 0.893 | 1.00 | 0 |
| center | 0.6 | 1.00 | 0.893 | 1.00 | 0 |

**all-MiniLM-L6-v2** (well-separated space → a loose τ already works)

| whiten | τ | relabel_detect | inv_easy_ok | sens_ok | realpair_max |
|---|---|---|---|---|---|
| none | ∞ (base) | 0.00 | 1.000 | 0.99 | 0 |
| none | **0.50** | **1.00** | 1.000 | 0.99 | 0 |
| none | 0.40 | 1.00 | 0.964 | 1.00 | 0 |

Conclusions:
- **τ closes Limitation #2 for both embedders**, without breaking the real pairs
  (`realpair_max`=0 always — identical models have cost 0 and are never rejected) or the
  sensitivity. The operating point depends on the space's separability: MiniLM at
  **τ≈0.5** (zero cost to the easy invariance); gte-large at **τ≈0.2**.
- **τ does NOT fix Limitation #1** (templated synonyms): tightening only makes it worse
  (it rejects the correct pair). It is a *ranking* problem, not a forced-bad-match one.
- **Whitening (`center`) does not help**: it separates but degrades the easy invariance;
  keep it off.

Adopted: `tau=0.2` with gte-large (the paper embedder). Limitation #1 is treated as an
embedder limitation (anisotropic space) in the threats to validity.

## Limitations revealed (threats to validity)

### 1. Activity pairing is the weak link — and the stronger embedder is worse
Invariance only fails on `syn_names`/`syn_all`/`recon_paraphrase` (variants that
**substitute** labels with synonyms). Counterintuitively, **gte-large fails more**
(28/52) than MiniLM (41/52): the gte-large cosine space is **compressed/anisotropic**
(high baseline ~0.67–0.75; synonym-vs-unrelated margin ≈ 0.05). On models with
"templated" labels (model246 = all "`X Equipment`", model103 = "`X Safety Y`"), after
synonymizing the shared word the activities become nearly indistinguishable and the
assignment **mispairs**. Evidence that it is *activity* pairing and not the metric:
`recon_reorder` (which keeps the *words*, only reordering) gives **0.000** for both
embedders. A higher MTEB rank does **not** guarantee better pairing here. Possible
mitigations: a cost threshold in the assignment (Mitigation (a) above), embedding
whitening, or a more separable embedder.

### 2. A single consistent relabel is invisible (isomorphism)
`relabel_single` renames **one** activity to an unrelated term consistently → **0.000**
at `tau=off`. Because the pairing is a **bijection**, the others match and the renamed
one is paired by **elimination**. The embedding resolves **ambiguity**; it does not
penalize an isolated relabel. **Closed by the adopted `tau=0.2`** (see Mitigation (a)),
which refuses the dissimilar forced match so the relabel is detected.

### 3. Short categorical antonyms can be masked
An antonym via **operator** (`>`→`<`) or **negated template** (Response→Not Response) is
always detected (exact comparison). But **categorical value** antonyms
("Approved"↔"Rejected", cos ≈ 0.6) can be absorbed by `val_map` — the single sensitivity
miss (`val_antonym` on model247). Textual-value matching inherits the embedder's
(in)ability to separate synonym from antonym.

## Files

- [`mp_metric.py`](mp_metric.py) — the metric (mirror of notebook cell 64 + `tau`/`whiten` knobs; `DEFAULT_TAU=0.2`).
- [`generate_and_evaluate.py`](generate_and_evaluate.py) — generates and evaluates the variants.
- [`tau_sweep.py`](tau_sweep.py) — τ / whitening sweep (Mitigation (a)).
- `variants/<model>/` — `base.decl` + one `.decl` per applicable variant (170 files).
- `results_<embedder>.csv` — full results (total + 5 perspectives + PASS/FAIL).
- `tau_sweep_<embedder>.csv` — detection-vs-invariance trade-off per τ.
