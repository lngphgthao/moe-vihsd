# Research pipeline: evidence audit and provisional direction

Date: 2026-11-04 (Asia/Saigon)  
Run: vietnamese-moe-2026-11-04  
Status: **BLOCKED at Stage 1 acceptance; research pipeline incomplete**

## Scope and execution

Execution root is `code/` in the enclosing thesis workspace. Preserve the established empirical thesis: controlled dense/MoE comparisons on ViANLI and ViHSD, with XLM-R, PhoBERT, and mBERT candidates. This is not a proposal to invent a new architecture. The existing `docs/THESIS_SPECIFICATION.md`, `experiments/EXPERIMENT_PLAN.md`, tracker, results, analysis notes, and source code supply the research brief. No research question, dataset, selection policy, historical result, or training implementation was changed.

Resolved defaults: AUTO_PROCEED=true, CODE_REVIEW=true, ARXIV_DOWNLOAD=false, AUTO_WRITE=false, venue unset. No cloud resources were purchased, GPU experiments launched, notifications sent, or test-set evaluations performed. GPU-hours consumed: 0. Historical compute cost is unknown.

The working tree already contained user changes, including training/evaluation, data loading, notebooks, requirements, tracker, and documentation. These remain intact. This audit describes the working tree, not a clean committed revision or the code that necessarily produced historical results.

## Literature landscape

Targeted web search on 2026-11-04 checked direct benchmark work and recent Vietnamese MoE classification leads. This is a preliminary landscape, not an exhaustive survey or completed novelty review.

- Huynh, Nguyen, and Nguyen, *A new benchmark dataset and mixture-of-experts language models for adversarial natural language inference in Vietnamese*, Expert Systems with Applications 306, 130109 (2026), [publisher DOI](https://doi.org/10.1016/j.eswa.2025.130109). The publisher abstract introduces ViANLI and NLIMoE with learned dynamic routing above a shared encoder. Therefore, applying MoE to ViANLI or comparing dynamic and top-k routing alone cannot support a first-use claim. Abstract-level inspection does not establish faithful reproduction of its method or evaluation protocol.
- The authors' [ViHSD dataset card](https://huggingface.co/datasets/uitnlp/vihsd) provides the benchmark source and cites Luu, Nguyen, and Nguyen (2021). Verify preprocessing, label order, splits, and metrics before cross-paper comparison.
- Search surfaced [BamiBERT MoE sentiment model](https://huggingface.co/TheSon2202/bamibert-moe-sentiment?library=transformers), whose card points to DOI `10.18653/v1/2026.semeval-1.25`. This is a lead requiring primary-paper verification, not an accepted citation or evidence of a matching task/protocol. It makes a broad first-Vietnamese-MoE-classifier claim especially unsafe.
- Four PDFs exist in `../papers/`. Their bibliographic descriptions are already recorded in `docs/THESIS_SPECIFICATION.md`; they were inventoried but not independently reread in this run. No new PDFs were downloaded. Gemini discovery was unavailable in the checked command/tool inventory.

The provisional contribution is a reproducible controlled study of quality, stability, cost, and expert behavior across the two established tasks. Its novelty is **UNCONFIRMED** pending primary-paper comparison and external review.

## Ranked ideas

These are investigator-ranked next investigations within the existing scope, not externally generated or accepted new research ideas. No pilots were run; no ranking uses historical test metrics.

| Rank | Investigation | Evidence and discriminating next step | Status |
| --- | --- | --- | --- |
| 1 | Establish trustworthy dense-versus-MoE comparisons | Recover resolved configs and artifacts for existing runs; reproduce only unrecoverable runs under a declared protocol; compare common seeds and report parameter/compute differences | Provisional priority; no verified pilot |
| 2 | Diagnose sequence-head and hybrid failure | Recover anomaly config/history; check labels, per-class predictions, routing, finite gradients, and weighted losses; reuse the existing tiny-overfit profile | Backup investigation; cause unknown |
| 3 | Test routing/placement effects across tasks | First establish a stable matched backbone/architecture control; vary one factor and confirm across seeds before transfer claims | Deferred until controls are verified |

Do not expand into new backbones, new architectures, a full Cartesian sweep, or new datasets before these evidence gaps are resolved. No candidate is empirically eliminated by this audit.

## Evidence audit

Canonical records: `experiments/experiment_tracker.csv`, `experiments/RESULTS.md`, and `experiments/analysis_notes.md`. Tracker percentages and saved metric fractions must not be mixed.

- The ledger documents historical dense baselines and MoE sweeps; its verification fields remain unverified. A reported completion status is not independently reproduced evidence. Do not upgrade rows from filenames or configurations.
- The saved hybrid run `results/vianli-hybrid-moe-xlmr-last6-e6-dynamic-head-e4-top2-s42/run_metrics.json` records validation macro F1 **0.3751085481**, training contradiction F1 **0.0793882010**, classification loss **1.0072895288**, balance loss **21.4069826285**, dynamic loss **1.7176198889**, and total training loss **1.2230769708**. These are observations of one saved file. They do not verify the run's provenance or establish the cause of poor performance.
- Important correction to the existing interpretation: raw auxiliary loss **23.1246025174** is an unweighted sum. `train.py` optimizes `classification_loss + balance_factor * balance_loss + dynamic_factor * dynamic_loss`. The current hybrid config specifies factors 0.01 and 0.001. Applied to the saved raw values, these give auxiliary contribution about **0.2157874421**, consistent with total minus classification loss. This consistency does not recover the missing resolved historical config. It does show that comparing raw auxiliary 23.12 to task loss 1.01 does **not** establish domination of the optimized scalar loss. Gradient effects remain unmeasured.
- Macro F1 has no universal three-class chance value. A low score alone does not diagnose collapse; use class distributions, confusion matrices, predictions, and suitable baselines. The historical anomaly status remains appropriate pending investigation.
- `.agent/context.yaml` says results were manually verified, but the run ledger explicitly says unverified. The ledger retains precedence until run-specific evidence reconciles the discrepancy.
- `configs/groups/vianli_stage1_dense_backbones.yaml` currently requests seeds **43 and 44**, despite the plan's common 42/43/44 confirmation policy. Do not launch it assuming it covers all three. Check the existing ledger and artifacts first; reruns need an explicit reproduction/provenance reason and fresh run IDs.
- The existing specification records finalist comment/config inconsistencies and missing provenance. A filename containing `best` is not a selection receipt.

## Novelty verification

**NOT ACCEPTED.** Direct prior ViANLI MoE work exists. A complete recent-work search, primary-paper comparison, and identity-bearing external novelty verdict have not been obtained. No novelty receipt or reviewer identity is invented.

## External critical review

**REVIEW_UNAVAILABLE through the specified route.** The installed skill references `../shared-references/reviewer-backends.md`, which is missing. No Codex reviewer MCP tool appears in the available tool inventory. The reviewer resolver cannot be followed with the supplied installation, and a self-review is not substituted for its cross-model acceptance gate. No reviewer score, thread ID, positive verdict, or accepted phase is recorded.

## First unfinished gate and recovery

`idea-discovery/SKILL.md` requires `run_state.py` and `idea_discovery_gate.py`, stating that if either is unavailable the final report is BLOCKED. The canonical locations checked were repo `.aris/tools`, repo `tools`, `$ARIS_REPO`, and the `~/.aris/repo` pointer. The repo helper directories and user `.aris` directory were absent, and ARIS_REPO was unset. Shared output/reviewer references were also absent from the installed skill tree. Broad searches were not used as proof that no installation exists anywhere on the machine.

This document preserves audit findings; it is not the helper's state file and is not an accepted Stage 1 artifact. The missing gate prevents Stage 2 deployment and Stage 3 review, rather than creating a discretionary user-confirmation checkpoint. AUTO_PROCEED remains true.

Recovery order:

1. Restore the skill package's referenced shared files and canonical state, evidence-gate, and reviewer-routing helpers. Resolve a functioning supported reviewer backend and record actual returned identities and durable verdict handles.
2. Start the canonical resumable run with ID `vietnamese-moe-2026-11-04`; load this audit as partial context. Stage 1 remains unfinished and must receive its required novelty/review evidence before acceptance.
3. Recover existing resolved configs, code revisions, dataset/model revisions, checkpoints, histories, predictions, and execution metadata. Preserve historical rows and distinguish artifact verification from reproduction.
4. For any unrecoverable control, declare the single comparison factor, preprocessing, training/selection budget, seed, parameter/compute accounting, unique run ID, expected artifacts, and reproduction reason before rerunning. Reuse `scripts/run_group.py`, the trainer, and existing configuration ownership.
5. For the classifier anomaly, reuse `configs/diagnostics/vianli_classifier_moe_tiny_overfit.yaml` after review. It currently points to Hub data and 50 diagnostic epochs; it is not a cheap local smoke command to launch blindly. Match preprocessing to its control.
6. On a GPU-capable authorized environment, run reviewed sanity checks and the smallest discriminating experiment. Only expand stable comparisons to common seeds 42/43/44, then selected ablations and the second dataset. No new paid compute budget is assumed.
7. Select on validation macro F1, report all seeds and sample standard deviation, and retain cost/routing evidence. Lock finalist config and seed set in a durable decision before final test evaluation.

Local environment check returned PyTorch **2.14.0+cpu**, CUDA unavailable, and zero visible GPUs. This establishes that the checked virtual environment cannot execute CUDA training; it does not establish absence of a GPU or another usable environment elsewhere. No external Kaggle/Colab session was accessed.

## Validation and stage disposition

Validation output is recorded in `validation_2026-11-04.txt` after the existing unit suite finishes. Unit tests exercise implementation invariants; they are not empirical pilots, historical-run verification, or independent research review.

| Stage | Disposition |
| --- | --- |
| Idea discovery | Partial audit written; acceptance BLOCKED by missing mandatory helpers and routed reviewer evidence |
| Experiment bridge | Not entered; no training or deployment |
| Auto review | Not entered; no score or positive verdict |
| Summary/writing handoff | Not entered; no final narrative or research-complete claim |
| Paper writing | Disabled by default; venue unset |

No final `NARRATIVE_REPORT.md`, paper PDF, research contract for an accepted idea, or accepted resumable-run state was created. HTML rendering was not reached. The existing plan remains canonical; no overlapping replacement experiment plan was introduced.
