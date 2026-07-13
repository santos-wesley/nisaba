# Evaluation_SoSyM — Evaluation dataset **and pipeline run** (SoSyM revision)

This folder contains the **evaluation model set for the SoSyM revision** of the Nisaba study **together with the full pipeline, the generated artifacts, and the aggregated results**. It **replaces** the eight-model set of our preliminary (BISE-era) evaluation --- preserved in the repository history and, as the metric calibration set, in `Metric_StressTest/base_models/` --- whose final benchmark reused a model (`model250`) that had also been part of the preliminary prompt-tuning phase — a train/test contamination that this fresh, held-out set removes.

## Provenance

All models are drawn from the **Terpsichora** public collection of 2,000 synthetic MP-Declare process models, generated with **OpenAI GPT-4o** and **Google Gemini 2.0 Flash** using two constrained-generation strategies (function calling and structured outputs):

- da Silva Santos, W., Rezende Coutinho, J., Baião, F., Miranda Spyrides, G., Côrtes Vieira Lopes, H. **Terpsichora: A Tool to Generate Synthetic MP-Declare Process Models.** *Process Mining Workshops (ICPM 2024)*, LNBIP 533, pp. 624–636, Springer, 2024. 🏆 Best Paper Award. doi:10.1007/978-3-031-82225-4_46
- da Silva Santos, W., Rezende Coutinho, J., Baião, F., Miranda Spyrides, G., Côrtes Vieira Lopes, H. **Enhancing Declarative Business Process Management Availability Through Generative AI.** *Process Science*, vol. 2, art. 21, Springer, 2025. doi:10.1007/s44311-025-00029-1

MP-Declare models are scarce in public repositories because real models expose strategic business knowledge; Terpsichora is used precisely because no equivalent public collection of multi-perspective declarative models exists.

## Selection design (40 models)

The set is **balanced** across the two factors that structure Terpsichora and **stratified by complexity**:

| Factor | Levels |
|---|---|
| Generator provider | Google (Gemini 2.0 Flash) · OpenAI (GPT-4o) |
| Constrained-generation strategy | Function Calling (FC) · Structured Output (SO) |
| Complexity batch | Batch 1 = simpler (size 16–21) · Batch 2 = complex (size 24–30) |

→ **2 providers × 2 strategies × 2 batches × 5 models = 40**, i.e. 10 per provider×strategy cell.

Within each cell, the 10 models were chosen by **maximum-diversity (farthest-point) sampling** over a standardized feature vector — size, density, number of distinct constraint templates, count of data conditions, temporal conditions, correlation conditions, cardinality constraints, attribute count and type mix (enumeration/integer/float), and number of binds — with an added coverage bonus so that distinct **business domains** are represented. The selection is deterministic (fixed seeding and tie-breaking) and reproducible (see `selection_manifest.csv`).

Every selected model was verified to be **byte-distinct from the eight preliminary-evaluation models** (MD5 exclusion), and all 40 model IDs are globally unique.

### Diversity achieved
- **25** distinct constraint templates across the set (unary existence/`Init`/`End`, positive relations, `Chain`/`Alternate`, and negative templates).
- **13** business domains (Security/Cyber, Finance/Banking, Healthcare, Logistics, Manufacturing, HR, IT, Customer/Sales, Legal/Compliance, Energy, Education, Insurance, Government/Public, Other).
- **35/40** exercise temporal conditions, **30/40** data conditions, **14/40** correlation conditions, **23/40** negative templates.
- Model size spans **16–30**.

See `selection_manifest.csv` for the full per-model feature record.

---

## What the pipeline does

For every **model × generator** combination (40 models × 4 generators = **160 runs**) the pipeline:

1. **Generates** a Nisaba description (staged: function-calling *intermediary* descriptions of each constraint layer, then a single *interleaved* narrative) and a **zero-shot** baseline description of the same model.
2. **Reverse-generates** an MP-Declare model back from each description (`*_reconstructed.decl/.json` for Nisaba, `*_zeroshot_reconstructed.decl/.json` for the baseline).
3. **Measures** the description/reconstruction against the original along objective, generator-independent instruments (no LLM-as-judge):
   - **Completeness** — complexity metrics (size, density, separability, constraint variability) of reconstructed vs. original.
   - **Correctness** — multi-perspective **Semantic Distance** (control-flow, temporal, data conditions, binds, attribute domains) between the reconstructed and original model, aligning tokens with the `Alibaba-NLP/gte-large-en-v1.5` embedder.
   - **Readability** — Flesch Reading Ease.
   - **Rhetorical-structure fidelity** — a **lexical RST relation metric** (does each description realise the guideline-prescribed rhetorical relation?) and its **neural corroboration** by the **DMRST** discourse parser.
   - **Understandability (outcome)** — a **two-model reader panel** that reads each description *in isolation* and (a) answers deterministic comprehension **QA** (balanced accuracy vs. a model-derived ground truth) and (b) **reconstructs** the model from the description alone (*recoverability*, scored by Semantic Distance); with **inter-reader agreement** (Gwet's AC1).

### Models used (all via OpenRouter, reasoning **off**)
- **Generators (4):** `anthropic/claude-sonnet-5`, `x-ai/grok-4.20`, `deepseek/deepseek-v4-flash`, `z-ai/glm-5.2` — two proprietary (Anthropic, xAI) and two open-weight (DeepSeek, Z-AI). Exact per-provider config in `run_meta.json`.
- **Reader panel (2):** `qwen/qwen3.7-plus` (proprietary, Alibaba) + `moonshotai/kimi-k2.5` (open-weight) — cross-vendor, disjoint from both the generators and the Terpsichora source-model authors (OpenAI/Google); `kimi` requests are round-robined across its OpenRouter providers to avoid single-provider throttling.

## Folder structure

```
Evaluation_SoSyM/
  README.md · selection_manifest.csv · run_meta.json      # docs + selection + run config
  .gitignore                                              # excludes DMRST_Parser/ (1.2 GB) + caches
  pipeline/                                               # the code (see below)
  <Provider>/                                             # Google | OpenAI
    <Strategy> Batch <n>/                                 # FC/SO Batch 1|2
      model<ID>/
        model<ID>.decl · model<ID>.json                   # SOURCE model (pipeline input)
        model<ID>_<gen>_intermediary.txt                  # Nisaba stage-1 per-constraint descriptions
        model<ID>_<gen>_nisaba_description.txt            # Nisaba final interleaved description
        model<ID>_<gen>_zero-shot_description.txt         # zero-shot baseline description
        model<ID>_<gen>_reconstructed.decl/.json          # model reverse-generated from the Nisaba description
        model<ID>_<gen>_zeroshot_reconstructed.decl/.json # ... from the zero-shot description
        model<ID>_<gen>_reader_<qwen|kimi>_<nisaba|zeroshot>_reconstructed.decl  # reader recoverability recon
        model<ID>_<gen>_understandability_v2.json         # reader QA + recoverability + AC1 for this combo
  # aggregated results (top level):
  complexity_metrics.csv          # completeness: complexity of source + every reconstruction
  semantic_distance.csv           # correctness: per-combo SD, 5 perspectives + total, Nisaba & zero-shot
  readability_scores.csv          # FRE per description
  rst_metrics.csv                 # lexical RST per combo: aligned rate, coverage, marker density, profile cosine
  rst_relation_families.csv       # RST aligned/mentioned per constraint family (prescribed relation)
  dmrst_pooled.csv · dmrst_batch.csv · dmrst_raw.json     # neural DMRST corroboration
  derived_summary.json            # the paper's aggregate stats (exact-preservation rates, SD preserved, RST gaps, FRE synthesis)
  understandability_results_v2.json                       # per-combo reader results (QA/recoverability/AC1)
  llm_calls.jsonl · understandability_calls.jsonl         # FULL LLM request/response logs (transparency/replicability)
  *.log                           # run logs (generation passes, reader run, DMRST, setup)
  DMRST_Parser/                   # NOT versioned (gitignored, ~1.2 GB) — see "DMRST" below
```

## Pipeline (`pipeline/`)

| Script | Role |
|---|---|
| `nisaba_port.py` | Provider-agnostic engine. Loads all **prompts, the reconstruction schema, the tools, and the metric functions VERBATIM** from `../../Nisaba_(SoSym_Journal).ipynb`, and adds a thin OpenRouter layer. Imported by the others. |
| `provider_probe.py` | Pre-flight: verify each generator supports chat + function-calling + structured output. |
| `run_test.py` | Batch runner over the 160 combos: generation → reverse generation → completeness/correctness/readability/RST. Writes `complexity_metrics.csv`, `semantic_distance.csv`, `readability_scores.csv`, `rst_metrics.csv`, `rst_relation_families.csv`, `derived_summary.json`, `llm_calls.jsonl`. |
| `understandability_test.py` | Reader stage: the 2-model panel's comprehension QA + recoverability + Gwet AC1. Writes `*_understandability_v2.json` (per combo), `understandability_results_v2.json`, `understandability_calls.jsonl`. Tunable via `UND_WORKERS`. |
| `rst_probe.py` | The lexical RST relation metric (invoked by `run_test.py`). |
| `dmrst_probe.py` | Neural DMRST corroboration (needs the checkpoint + a CUDA GPU). Writes `dmrst_pooled.csv`, `dmrst_batch.csv`, `dmrst_raw.json`. |
| `reprocess_sd.py` | Recompute the local metrics (complexity/SD/readability/RST + `derived_summary.json`) from cached reconstructions, without re-calling any LLM. |
| `dose_response.py` | Dose–response probe: Spearman correlation between the per-run lexical-RST realisation rate and the reader panel's comprehension (balanced QA) and recoverability (reader-reconstruction SD), pooled (320 pts) and within-Nisaba (160 pts). Local only (reads `rst_metrics.csv` + the per-combo `*_understandability_v2.json`). |

## Reproducing the evaluation

```bash
export OPENROUTER_API_KEY=...          # all generation/reader calls go through OpenRouter
cd pipeline
python provider_probe.py               # optional pre-flight
python run_test.py                     # generation + completeness/correctness/readability/RST (writes the CSVs)
python understandability_test.py       # reader panel: QA + recoverability + AC1
python dmrst_probe.py                  # neural RST corroboration (requires the DMRST checkpoint + GPU)
```
Scripts are idempotent/resumable: a combo whose artifacts already exist is skipped, so an interrupted run can be re-launched. `reprocess_sd.py` recomputes the objective metrics without re-generating.

### DMRST (not versioned)
`dmrst_probe.py` uses the **DMRST discourse parser** (Liu/Shi/Chen 2021, `xlm-roberta` backbone). The parser code + its `multi_all_checkpoint.torchsave` weight (~1.2 GB) are **git-ignored** under `DMRST_Parser/`; the exact clone + checkpoint-download steps are recorded in `dmrst_setup.log`. It requires a **CUDA-enabled PyTorch** (the parser hardcodes `.cuda()`); this run used `torch 2.13.0+cu130` on a GTX 1650 (`torch_cuda_install.log`).

## Transparency & replicability

Everything needed to audit or replicate the run is kept in-repo: the source models, every generated description/reconstruction, the aggregated CSV/JSON results, `run_meta.json` (model IDs + sampling params), the run logs, and the **complete line-delimited LLM request/response logs** (`llm_calls.jsonl`, `understandability_calls.jsonl`). Only the 1.2 GB DMRST checkpoint is excluded (obtainable via `dmrst_setup.log`). No LLM acts as a judge — all instruments are objective and generator-independent.
