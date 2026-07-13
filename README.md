# Nisaba

Nisaba is an LLM framework that **generates natural language descriptions of Multi-Perspective Declare (MP-Declare) process models** and pairs generation with a **systematic, objective assessment methodology**. Generation is staged: constrained *intermediary* descriptions anchor every construct of the model, and a rhetorically prescribed synthesis step composes them into the final *interleaved* description delivered to the reader. Assessment closes the loop by **reverse-generating** an MP-Declare model back from each delivered description and measuring what survived — no LLM-as-judge anywhere in the loop:

- **Completeness** — complexity metrics of the reconstructed vs. original model;
- **Correctness** — a multi-perspective **Semantic Distance** between reconstructed and original model;
- **Readability** — Flesch Reading Ease;
- **Rhetorical-structure fidelity** — a lexical **RST relation metric**, corroborated by the neural **DMRST** discourse parser;
- **Understandability (outcome)** — a two-model **reader panel** that reads each description in isolation, answers deterministic comprehension **QA** (graded against a model-derived ground truth), and **reconstructs** the model from the text alone (*recoverability*).

## Repository layout

| Path | What it is |
|---|---|
| `Nisaba_(SoSym_Journal).ipynb` | The execution and evaluation bench for the SoSyM study — the full pipeline as documented, runnable cells (setup → generation → reverse generation → all instruments → analysis). |
| `Evaluation_SoSyM/` | The SoSyM evaluation data pack: the 40 held-out Terpsichora source models, the pipeline as standalone scripts (`pipeline/`), every generated artifact (intermediary + final descriptions, reconstructions, reader outputs), the aggregated result CSVs, run logs, and the replayable per-request LLM call logs (`llm_calls.jsonl`, `understandability_calls.jsonl`). **See its `README.md` for the full data dictionary.** |
| `Metric_StressTest/` | The perturbation suite that calibrates and validates the Semantic Distance metric: 162 controlled edits (synonym/antonym/structural) over the eight calibration models in `base_models/`, plus the τ threshold sweep. See its `README.md`. |
| `Evaluation_ICPM/`, `Nisaba (ICPM).ipynb` | Historical: the published ICPM 2024 tool-paper study (see *Lineage* below), kept for reference. |

## Reproducing

1. **Environment.** Python 3.11+. The setup cells of `Nisaba_(SoSym_Journal).ipynb` install the dependencies (version-pinned, except the Colab-side CrewAI/LiteLLM layer); the standalone scripts in `Evaluation_SoSyM/pipeline/` mirror the notebook cell-for-cell.
2. **API access.** All model calls go through [OpenRouter](https://openrouter.ai) under version-pinned slugs, with reasoning disabled and sampling controls fixed where the model exposes them. Set the key as an environment variable (never in code or notebooks): `OPENROUTER_API_KEY`.
3. **Run.** Execute `Nisaba_(SoSym_Journal).ipynb` top-to-bottom, or the `Evaluation_SoSyM/pipeline/` scripts (each script's header states which result files it produces; see the folder's `README.md`).
4. **Audit/replay.** Every request and response of the reported run is logged in `Evaluation_SoSyM/llm_calls.jsonl` (generation and reconstruction) and `Evaluation_SoSyM/understandability_calls.jsonl` (reader panel) — model, messages, response, usage; no credentials — so each reported number can be traced to the calls that produced it even where providers are not bit-exact reproducible.

## DMRST parser (not versioned)

The neural discourse parser used to corroborate the lexical RST metric is a third-party project whose ~1.2 GB trained checkpoint is distributed by its authors via Google Drive (it is not in their git repository, and it exceeds GitHub's 100 MB blob limit), so neither is versioned here (see `.gitignore`). The parser **outputs of the reported run are versioned** (`Evaluation_SoSyM/dmrst_raw.json`, `dmrst_batch.csv`, `dmrst_pooled.csv`, `dmrst_run.log`), so the reported numbers can be verified without re-running. To re-run:

```bash
git clone https://github.com/seq-to-mind/DMRST_Parser.git "Evaluation_SoSyM/DMRST_Parser"
cd "Evaluation_SoSyM/DMRST_Parser" && git checkout 231d8c0d28ba8cba074e29a6ff99e858e4742735
# Checkpoint, as distributed by the DMRST authors (Google Drive file id 12Gc6mC6Qh0R_N_U60mx2jDQqRfuwQzZh):
pip install gdown && gdown 12Gc6mC6Qh0R_N_U60mx2jDQqRfuwQzZh -O depth_mode/Savings/multi_all_checkpoint.torchsave
```

Verify the artifact is the exact one used in the reported run — SHA-256:
`7973b183104784d99dec3a90bbb5e58e2ec7c243c0c7126b5be669029e56c246` — then run `Evaluation_SoSyM/pipeline/dmrst_probe.py`. (The DMRST repository declares no license, so we do not mirror the checkpoint ourselves; the hash pins it instead.)

## Lineage

- **Terpsichora** (source-model collection): da Silva Santos et al., *Terpsichora: A Tool to Generate Synthetic MP-Declare Process Models*, ICPM Workshops 2024, LNBIP 533. doi:10.1007/978-3-031-82225-4_46
- **MP-Declare metamodel / availability**: da Silva Santos et al., *Enhancing Declarative Business Process Management Availability Through Generative AI*, Process Science 2(21), 2025. doi:10.1007/s44311-025-00029-1
- **Nisaba preliminary study** (framework concept and the eight calibration models): da Silva Santos et al., *Nisaba: Towards Generating Natural Language Description of Multi-Perspective Declarative Process Models*, ICPM 2024 Workshops. Its eight models serve as the Semantic Distance calibration set (`Metric_StressTest/base_models/`); the forty-model evaluation set in `Evaluation_SoSyM/` is disjoint from them (MD5-verified).
