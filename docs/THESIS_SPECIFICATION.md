# Thesis Project Specification

**Status:** Working specification, consolidated from the submitted thesis proposal and this repository on 2026-10-22. It records the project as currently defined; entries marked **Open** require confirmation or completion.  
**Maintainer:** Student researcher, with supervisor review.  
**Official source:** `../paperwork/DECUONG_23521467.pdf` (detailed thesis proposal, six pages; outside this Git repository).

## 1. Project identity

| Field | Specification |
| --- | --- |
| Vietnamese title | Áp dụng kiến trúc Mixture-of-Experts cho bài toán phân lớp |
| English title | Applying Mixture-of-Experts Architectures to Text Classification |
| Student | Lê Ngọc Phương Thảo (23521467), per the proposal |
| Supervisor | TS. Nguyễn Trọng Chỉnh, per the proposal |
| Program | Regular undergraduate program, per the proposal |
| Thesis language | Vietnamese, per the proposal |
| Scheduled period | 2026-09-07 to 2026-12-26, per the proposal |
| Latest documented project stage | `docs/STATUS.md` reports experimental development and evaluation, last updated 2026-09-29. The proposal schedule placed ablations from Oct 21 to Nov 2, analysis/statistical testing Nov 3–15, demo Nov 16–20, and writing thereafter. Actual progress after the status report is not verified in the current project records. |

## 2. Project summary

This is an empirical study of Mixture-of-Experts (MoE) architectures for Vietnamese text classification. It compares dense Transformer classifiers and MoE variants on two tasks with different characteristics: Vietnamese Adversarial Natural Language Inference (ViANLI) and Vietnamese Hate Speech Detection (ViHSD). The research asks when MoE helps, how its design choices affect predictive performance and expert behavior, and what computational costs accompany those choices. Results may be positive, neutral, or negative; the project does not assume MoE will outperform dense models.

ViANLI and ViHSD are evaluation settings for the broader text-classification question. The thesis is not primarily a proposal for a new adversarial-NLI or hate-speech method, and it does not set out to invent an entirely new MoE architecture. A demo is intended to illustrate a selected trained model, not to deliver a production service.

## 3. Motivation and problem

Dense classifiers apply the same learned transformations to every example. MoE introduces multiple expert components and a router that selects or weights experts for each input or token. This conditional computation may allow specialized processing, but its value depends on design choices such as expert count, routing, placement, training, and the task. Additional parameters and routing overhead may also reduce practical efficiency. Therefore, empirical comparisons under controlled conditions are needed to determine whether MoE offers useful benefits and where those benefits or costs occur for Vietnamese classification.

**Research gap as currently evidenced:** The proposal motivates this empirical question and the repository contains implementation and experiment plans. A direct 2026 publication already introduces ViANLI and evaluates an MoE model on it. Therefore, a claim that this thesis is the first application of MoE to ViANLI, or that no MoE method has been studied there, would be unsupported. A defensible contribution may lie in a controlled study across both ViANLI and ViHSD, wider architecture families, ablations, multi-seed stability, and efficiency/expert-behavior analysis—but this needs confirmation against the direct paper and supervisor approval. The four local PDFs and the benchmark papers below are a starting corpus, not a complete literature review.

### Direct benchmark work and prior results

| Source | Established information | How it affects this thesis |
| --- | --- | --- |
| [Huynh, Nguyen, and Nguyen (2026), “A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial Natural Language Inference in Vietnamese,” *Expert Systems with Applications*, 306, article 130109, DOI 10.1016/j.eswa.2025.130109](https://doi.org/10.1016/j.eswa.2025.130109). | The publisher page dates the article to 15 Apr 2026. It introduces ViANLI and proposes NLIMoE. Its abstract reports 47.3% accuracy for NLIMoE versus 45.5% for an XLM-R-large baseline on ViANLI. The current ViANLI dataset card cites this paper and describes the dataset. | This is a close, direct comparator and must be discussed in related work. Its accuracy results are not directly comparable with this project’s tracker macro-F1 values or different backbone/validation/test protocols. Check the full paper’s setup, datasets, split, metric, and architecture before defining the precise research gap. The repo’s `nlimoe` implementation may support an explicit comparison, but must be checked against the paper before being described as a reproduction. |
| [Luu, Nguyen, and Nguyen (2021), “A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts,” arXiv:2103.11528](https://arxiv.org/abs/2103.11528). | The source paper defines the ViHSD labels and numeric IDs as CLEAN (0), OFFENSIVE (1), and HATE (2), describes 33,400 comments and a 7:1:2 split, and reports a BERT baseline. The local cache’s 24,048/2,672/6,680 split totals match that size. | This resolves the canonical numeric class mapping for the local ViHSD cache. Cite the dataset paper and use its mapping in all per-class tables. The current pipeline still labels these classes as the strings `"0"`, `"1"`, and `"2"` in reports because the cached feature is integer `Value` and the config has no label-name override; update/verify reporting before interpreting class-wise results. |

The ViANLI card itself has a metadata conflict: its repository badge says MIT while the dataset-card text says CC BY-NC-SA 4.0. Its card also cites the 2026 paper under a 2025 BibTeX year (the DOI year); use the publisher’s publication metadata and resolve the applicable data terms before redistribution. The ViHSD Hugging Face page displays MIT metadata but gates file access behind accepting its conditions; confirm that these terms and the upstream dataset citation apply to the exact cached revision.

### Literature currently available in the workspace

The following records are from the local PDFs under `../papers/`; that folder is outside this Git repository. The summary is based on bibliographic/front-matter and introductory pages. These are a starting set, not a completed review or validated citation list.

| Local file | What it covers | Relevance and caveat |
| --- | --- | --- |
| `Mixtureofexpertsaliteraturesurvey.pdf` — Saeed Masoudnia and Reza Ebrahimpour, “Mixture of experts: a literature survey,” *Artificial Intelligence Review*, 42, 275–293, DOI `10.1007/s10462-012-9338-y`. | Foundational taxonomy of mixture-of-experts approaches and implicit versus explicit partitioning. The publisher lists online publication in 2012 and the journal volume/pages in 2014. | Useful for definitions/history. Follow the department’s citation style for the online-first versus issue year. |
| `NeurIPS-2022-towards-understanding-the-mixture-of-experts-layer-in-deep-learning-Paper-Conference.pdf` — Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, and Yuanzhi Li, “Towards Understanding the Mixture-of-Experts Layer in Deep Learning,” NeurIPS 2022. | Theory and empirical analysis of expert specialization and routing in a cluster-structured classification setting. | Useful for explaining why expert specialization and data structure matter. The paper’s theoretical/synthetic setting does not establish outcomes on Vietnamese NLP benchmarks. |
| `2503.07137.pdf` — Siyuan Mu and Sen Lin, “A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications,” arXiv:2503.07137v4 (version dated 2026-01-24 in the PDF). | Broad survey of MoE design, training, theory, systems, and applications, including NLP. | Useful for mapping themes and discovering primary studies. The PDF says it was submitted to IEEE; verify current publication status and cite the version actually consulted. |
| `survey-of-next-generation-ai.pdf` — Asifullah Khan et al., “A Survey of Next-Generation AI and Its Evolving Landscape,” TechRxiv DOI `10.36227/techrxiv.175329122.29020847/v3`, posted 2026-02-02. | Broad survey across 16 emerging AI topics, including a section on MoE. | Tangential background only; the PDF labels itself a preprint that has not been peer reviewed. It should not carry the technical MoE argument when primary sources are available. |

**Review task:** extract and read the sections most relevant to this study, trace key claims to primary sources, add recent NLP/text-classification MoE work and Vietnamese benchmark studies, and record the formal citation metadata and peer-review status. Do not infer that no prior Vietnamese work exists from this small local collection.

## 4. Aim, objectives, and research questions

### Aim

Evaluate and analyze the application of MoE architectures to Vietnamese text classification using controlled experiments on ViANLI and ViHSD.

### Objectives

1. Review text classification, Transformer encoders, pretrained language models, MoE experts and routers, routing strategies, and prior MoE applications to NLP.
2. Prepare reproducible pipelines for ViANLI and ViHSD, including required Vietnamese word segmentation and explicit label handling.
3. Establish dense baselines and compare them with selected MoE architecture families under comparable data and training conditions.
4. Run controlled ablations on MoE design choices, such as placement, expert count, routing, shared experts, and selected training or regularization choices.
5. Repeat important comparisons across random seeds, report aggregate results and uncertainty, and use appropriate statistical analysis.
6. Analyze classification quality alongside efficiency and routing/expert behavior where artifacts support it.
7. Produce a small demonstration, reproducible source/configuration artifacts, and the Vietnamese thesis report.

### Working research questions

These questions are synthesized from the proposal and experiment plan; the student and supervisor should approve their final wording and priority.

- **RQ1:** Under matched experimental conditions, how do selected MoE classifiers compare with dense Transformer classifiers on ViANLI and ViHSD?
- **RQ2:** How do MoE placement, expert count, and routing strategy affect classification quality and stability?
- **RQ3:** How do shared-expert or other selected MoE components affect outcomes when capacity and training conditions are made explicit?
- **RQ4:** What trade-offs in parameter count, active computation, memory, training/inference time, and expert utilization accompany the observed performance?
- **RQ5:** Which findings are consistent across the two datasets, and which appear task-specific?

Do not interpret these as confirmed conclusions. Hypotheses and expected directional effects remain open; the proposal explicitly allows MoE to perform worse or show no benefit.

## 5. Scope and boundaries

### Included

- Vietnamese text classification experiments on ViANLI and ViHSD.
- Dense Transformer baselines and implemented MoE families: Transformer-MoE, Classifier-MoE, and Hybrid-MoE where stable and explicitly configured. The repository also implements an `nlimoe` sequence-level head; whether it belongs in the thesis comparison is an **Open** scope decision.
- Candidate encoders documented in the project: XLM-R-base, PhoBERT-base, and multilingual BERT. XLM-R-large is optional and resource-dependent.
- Controlled architecture/component comparisons, multiple seeds for important results, classification metrics, appropriate statistical analysis, and efficiency/behavior analyses where measurable.
- A demonstration and research artifacts sufficient to explain and reproduce selected results.

### Excluded unless formally agreed as a scope change

- Claiming a novel general-purpose MoE architecture.
- Treating either benchmark as the sole definition of the research problem or proposing a domain-specific production solution.
- Treating a demo as a production-ready or deployed commercial system.
- Selecting models or hyperparameters using test-set results.
- Treating planned configurations, filenames, notebook presence, or unverified tracker metrics as evidence that an experiment ran successfully.

Any substantive change to datasets, research questions, evaluation policy, or core comparison should be recorded in `DECISIONS.md` and reflected here.

## 6. Data and task specification

| Dataset | Task and input | Current repository setup | Required verification |
| --- | --- | --- | --- |
| ViANLI (`uitnlp/ViANLI`) | Three-way inference classification from premise and hypothesis; config names labels `entailment`, `contradiction`, `neutral` in that order. | Cached split sizes: train 8,012, validation 1,000, test 1,000. Cached HF metadata records source revision `0fec8d6ecb043a61c609f9b51f80401fdf1e84d3`. Train counts: entailment 2,615; neutral 2,924; contradiction 2,473. Validation/test are 333/334 per class. Maximum length 256; NLI pair tokenization; segmentation script processes both fields. | Config does not pin this revision. The base config points to the Hub ID `uitnlp/ViANLI`; the segmented local cache is used only when `dataset.name` points to `data/vianli_segmented`. The dense-backbone group mixes Hub data for XLM-R/mBERT with segmented local data for PhoBERT. Review the conflicting license statements in the dataset card and make preprocessing consistent across backbones. |
| ViHSD (`uitnlp/vihsd`, cached as `data/vihsd_segmented`) | Hate-speech text classification; input column `free_text`, numeric label `label_id`. Canonical mapping from the source paper: 0=CLEAN, 1=OFFENSIVE, 2=HATE. | Cached split sizes: train 24,048, validation 2,672, test 6,680. Cached HF metadata records source revision `88e81b36ca376867640dad9df295e0f7d499ab80`. Counts by numeric ID: train 0/1/2 = 19,886/1,606/2,556; validation = 2,190/212/270; test = 5,548/444/688. Maximum length 128; preprocessing script applies VnCoreNLP segmentation. | Config does not pin this revision. Confirm dataset access/terms and preprocessing fallback. The current loader reports class names as `"0"`, `"1"`, and `"2"`; make per-class metric labels readable and verify the resulting reports against the source-paper mapping. |

Use the official train and validation splits for fitting and model selection, and reserve test data for final evaluation after configurations/checkpoints and seed sets are locked. Preserve the raw data provenance and the exact preprocessing/tokenizer versions needed for replication. Do not commit private credentials or restricted dataset content.

The counts and revisions above were read from the local `datasets` cache (`dataset_info.json` and `load_from_disk` split contents) on 2026-10-22. The preprocessing scripts load datasets without an explicit revision argument, so cache metadata is evidence about these local copies, not a pinned guarantee for a future run. The split fingerprints in `state.json` identify local serialized data but do not by themselves capture the VnCoreNLP model/software versions.

**Open:** Review dataset license/usage terms; pin source revisions in preprocessing/configuration; ensure the runtime per-class reports use the documented ViHSD ID-to-name mapping; and record segmentation fallback/error counts. The label counts are now available for the local caches, but should be regenerated after pinning the official dataset revision and preprocessing recipe.

## 7. Methods and experimental protocol

### Model families and variables

The experiment plan identifies the following factors:

- Backbone: XLM-R-base, PhoBERT-base, multilingual BERT; optional XLM-R-large.
- Architecture: dense, Transformer-MoE, Classifier-MoE, Hybrid-MoE; clarify whether sequence-level `nlimoe` is in final scope.
- Transformer FFN placement: none, last 4, last 6, alternating, all, or explicit layer sets.
- Routed expert count: commonly 2, 4, or 8; other values only with justification.
- Routing: top-1/top-2/top-4 and implemented dynamic routing.
- Other components as selected: shared expert, expert type/initialization, residual scale, load-balancing/dynamic auxiliary losses, and training settings.
- Training factors: learning rate, weight decay, dropout, epoch budget, and random seed.

This is a candidate design space, not a requirement to exhaustively run every combination. Keep the study feasible and prioritize comparisons that answer the approved research questions.

### Comparison rules

1. Define a question and the factor being changed before launching each comparison.
2. Match dataset split/preprocessing, backbone, training budget, loss, and seed wherever they are not the target factor. If matching parameter count or compute is impossible, report total and trainable parameters and explain the difference.
3. Use an explicit resolved configuration for every run and preserve it with checkpoints and metrics.
4. Use validation macro F1 as the current primary development/selection metric. Track validation accuracy, weighted F1, per-class results, and loss as supporting evidence.
5. Do not use test metrics to choose architecture, hyperparameters, checkpoints, or finalists.
6. The experiment plan calls for seeds 42, 43, and 44 at minimum for promising configurations; aim for 3–5 seeds for final candidates when resources allow. Report mean and standard deviation. Record any resource-driven reduction as a limitation.
7. Distinguish recorded observation, interpretation, hypothesis, and confirmed finding. A single run or uncontrolled comparison does not establish a causal architecture effect.

### Metrics

- **Classification:** macro F1 (primary development metric), accuracy, weighted F1, per-class precision/recall/F1, confusion matrix, and validation loss. State final reported test metrics after the final selection is locked.
- **Efficiency:** total and trainable parameters, active parameters per token where definable, FLOPs if reliably measured, peak GPU memory, training time, and inference latency/throughput. Define hardware, batch size, sequence length, warm-up, and timing procedure for runtime comparisons.
- **MoE behavior:** expert/token utilization, routing entropy, imbalance, active experts per input/token, and auxiliary losses. Define aggregation and padding-token treatment.
- **Statistics:** report mean ± standard deviation across seeds. The approved test (e.g., paired test across common seeds or a suitable prediction-level test) and significance threshold/multiple-comparison handling are **Open** and must be chosen with supervisor/statistical justification before claims of significance.

### Implementation and execution

The current codebase uses Python, PyTorch, Hugging Face Transformers/Datasets, YAML configuration, and supports local, Kaggle, and Colab execution. Run-specific configuration and artifacts should identify the code revision, resolved config, dataset/model revision, seed, platform/hardware, checkpoint, metrics, and provenance. External Kaggle/Colab runs must be manually synchronized into the tracked `experiments/experiment_tracker.csv`; interpret the shared evidence here, and keep local-only `experiments/RESULTS.md` and `analysis_notes.md` synchronized when available.

The repository root (`code/` in the enclosing workspace) is the execution root for scripts: commands such as `python train.py ...` and `python scripts/segment_vianli.py ...` are run from this directory. The README and notebooks provide the current platform walkthroughs. The dependency file declares Transformers 5.5.0 and the VnCoreNLP Python wrapper. Segmentation requires Java 8 or newer on `PATH`. The remaining environment uses lower-bounded dependencies, and NumPy is currently supplied transitively. PhoBERT/XLM-R tokenization may also require SentencePiece support. The README's `pip install -r requirements.txt` command alone is therefore not verified to create a working environment. **Open reproducibility task:** declare and pin the full environment, document Python/Java and GPU requirements, and verify clean setup on the intended local/Colab/Kaggle path before presenting the repository as directly reproducible.

The intended high-level run sequence is: create the environment and authenticate only when private/gated model or dataset access is needed; prepare the segmented dataset using `scripts/segment_vianli.py` and/or `scripts/preprocess_vihsd.py`; run a smoke profile for the selected task; execute planned runs from explicit YAML/group configs with unique IDs; then preserve resolved configs, checkpoints, logs, and metrics. Example command syntax and platform-specific persistence steps live in `README.md`. Do not treat this outline as proof that a clean setup or run has been verified.

In the current implementation, `train.py` evaluates the validation split during training and saves the best-validation checkpoint; it does not need test metrics for checkpoint selection. `evaluate.py` requires the `--final-evaluation` flag before accessing the test split. This flag is a human confirmation guard; the script does not independently verify that the finalist config or seed set was locked. Record that selection decision before using the command.

## 8. Current evidence and project state

At the time this specification was created, `experiments/experiment_tracker.csv` contains 51 rows: 48 marked `completed`, 2 marked `anomaly`, and 1 marked `planned`. All 51 rows are marked `unverified`. Forty-seven rows lack a platform value and 50 lack a config path. Therefore, these counts describe the ledger, not independently verified execution evidence.

The ledger records 37 ViANLI rows and 14 ViHSD rows. Its two anomaly rows include the low Classifier-MoE ViANLI run and the Hybrid-MoE ViANLI run; local-only `experiments/analysis_notes.md` also leaves their causes unresolved. Existing ViANLI dense runs cover three backbones and seeds 42–44, but require artifact/config verification. Most architecture and routing comparisons are single-seed. No final causal findings should be claimed from this record as-is.

The local machine-readable summary `.agent/context.yaml` says results were externally executed and manually verified, but names no experiment IDs or evidence. That blanket statement conflicts with the tracker’s `unverified` status for all 51 rows and the missing provenance/artifacts described here. Do not use it to upgrade result status; reconcile it by run ID and supporting artifacts, then update the context file to match the verified ledger.

**Artifact audit:** this workspace currently has a `run_metrics.json` for the Hybrid-MoE anomaly at `results/vianli-hybrid-moe-xlmr-last6-e6-dynamic-head-e4-top2-s42/`. It records validation macro F1 0.3751, test macro F1 0.3778, training contradiction-class F1 0.0794, and total auxiliary loss 23.1246 (balance 21.4070, dynamic 1.7176). These are values in a saved metrics file, not independently reproduced or verified results. The same run folder has no resolved config, checkpoint, or training history in the current workspace; its saved test value is historical only and must not guide selection. No run-specific metrics folder was found for the Classifier-MoE anomaly. The legacy `results/summary_results/` export is not enough to establish run provenance. `results/` and `checkpoints/` are ignored by Git, so run artifacts need an explicit retained/shared location and tracker paths if another contributor is expected to verify them.

The legacy `results/summary_results/summary_table.csv` contains 39 ViHSD rows and test metrics, but its columns do not include seed, execution platform, resolved config, artifact path, or verification state. The CSV and zipped copy are useful as historical reported summaries only; there is not enough metadata to map them safely onto current tracker rows or treat the scores as verified finalist evidence.

**Configuration inconsistencies to resolve before calling a model a finalist:**

- `configs/finalists/vianli_best.yaml` comments describe a “best confirmed” configuration and mention 4 experts in the run name, while the actual YAML sets `num_experts: 6`; the tracker’s corresponding last-six, dynamic-routing entry is also unverified.
- `configs/finalists/vihsd_best.yaml` comments report multi-seed validation/test summaries, but the tracker currently marks every row unverified and does not show a matching three-seed finalist set. The ledger’s strongest listed single-run validation scores do not by themselves verify that summary.
- The task configs, candidate design space, experiment overlays, tracker, and finalist comments do not always describe the same expert count/routing/placement. Reconcile each intended comparison against its resolved configuration and saved artifacts; preserve historical records and document corrections rather than silently editing history.

“Finalist” in a file name is a repository label, not independent evidence. Verify configurations, artifacts, metric definitions, and provenance before relying on reported results.

## 9. Planned deliverables

1. Vietnamese thesis report with background/literature, problem and questions, method/protocol, controlled results, analysis/statistics, limitations, conclusion, references, and appendices as required by the department.
2. Reproducible source code, environment/dependency record, dataset preparation instructions, configs, and run commands.
3. Verified experiment ledger and result summaries, including negative/anomalous results and provenance.
4. Selected saved model/checkpoint and a small demonstration illustrating text input and predicted class/output. Define UI, hosting, and model choice later.
5. Final presentation/defense material and any university-required forms/formatting.

The repository includes an LLNCS/Springer sample template, but the proposal does not establish it as the required thesis format. Follow the university/program thesis template and submission rules once obtained.

## 10. Schedule from the approved proposal

These are proposal dates, not verified completion dates. Update actual status and revised dates as work proceeds.

| Work package | Proposal dates | Evidence/artifact to close the work |
| --- | --- | --- |
| Literature review and problem definition | Sep 7–20 | Annotated literature and approved research questions |
| Data preparation | Sep 15–18 | Documented, checked dataset pipeline |
| Dense baselines | Sep 18–20 | Verified baseline runs and configs |
| MoE implementation | Sep 21–30 | Model implementation and pipeline checks |
| Main experiments | Oct 1–20 | Controlled baseline/MoE comparisons |
| Ablations | Oct 21–Nov 2 | Component comparisons with complete run records |
| Analysis and statistical testing | Nov 3–15 | Multi-seed summaries, justified tests, plots, interpretation |
| Demo | Nov 16–20 | Reproducible demonstration using a selected model |
| Writing and completion | Nov 20–Dec 1 | Full draft and completed artifacts |
| Final review and defense preparation | Dec 1–20 | Revised thesis and defense material |
| Overall project end date | Dec 26 | Per proposal |

## 11. Open information and decisions register

Resolve these items with evidence or supervisor input. Mark an item complete here and update the source-of-truth file when it is settled.

### Recommended resolution order

This order reflects dependencies, not completed work. Record the decisions and evidence in section 15 as they are resolved.

1. **Approve the research claim and scope:** finish the focused literature comparison with the ViANLI/NLIMoE paper, settle the research questions and required architecture families with the supervisor, and define what contribution the thesis can support.
2. **Freeze the data recipe:** settle source revisions and usage terms, choose one consistent preprocessing path per dataset/backbone, and verify label mappings and split integrity before launching new comparisons.
3. **Freeze the comparison protocol:** select the primary baseline/backbone, controlled factors, seed set, metrics, efficiency measurement procedure, statistical analysis, and validation-only finalist rule.
4. **Make runs reproducible:** repair and verify the dependency environment, agree on artifact retention, and recover run-level provenance where possible. New runs should not inherit the current tracker’s unknown provenance.
5. **Audit and execute:** investigate anomalies and reconcile existing runs against artifacts; then run only the comparisons needed to answer the approved questions, prioritizing multi-seed confirmation over a broad unreplicated sweep.
6. **Close delivery requirements:** confirm thesis format/deadlines and demo expectations, then align the writing, demo, and defense materials to those requirements and the revised schedule.

### Research design

- **Open:** Final supervisor-approved research questions, hypotheses (if any), and primary claim boundaries.
- **Open:** Complete the literature review and define the thesis contribution relative to the 2026 ViANLI/NLIMoE paper and relevant ViHSD work; validate citations and get supervisor approval for the final claim boundary.
- **Open:** Which architecture families are essential, especially whether `nlimoe` and Hybrid-MoE remain in the final comparison.
- **Open:** Decide whether to reproduce/compare against the published NLIMoE ViANLI study directly, or position it as prior context while making the controlled two-dataset and architecture-ablation design the thesis contribution. State the resulting novelty/claim boundary with the supervisor.
- **Open:** Primary backbone/baseline selection and a tie-breaking rule for choosing the comparison backbone based only on validation data.
- **Open:** Which comparisons use matched parameter/compute budgets versus ordinary matched training settings.
- **Open:** Statistical test, unit of analysis, assumptions, significance level, and any multiple-comparison policy.

### Data and reproducibility

- **Open:** Pin the locally recorded dataset revisions and preprocessing recipe; resolve conflicting ViANLI license declarations and confirm ViHSD access/terms for the exact revision. The canonical ViHSD ID mapping is documented above, but the current pipeline’s per-class output names still need correction/verification.
- **Open:** Exact preprocessing policy and dataset/model revisions. Segmentation errors now stop processing before dataset saving.
- **Open:** Complete environment lock and clean-install verification. Transformers is pinned to 5.5.0, the VnCoreNLP wrapper is declared, and Java prerequisites are documented. Remaining dependency versions and any tokenizer-specific SentencePiece requirements need clean-environment verification.
- **Open:** Hardware details, run-time measurement protocol, and artifact retention/location policy.
- **Open:** Verify all existing tracker rows against run artifacts; fill provenance/platform/config path where recoverable. Keep unavailable values explicitly unknown.
- **Open:** Reconcile `.agent/context.yaml`'s unscoped `manually_verified: true` claim with the per-run tracker verification state; only retain it if the verified run IDs and evidence are recorded.
- **Open:** Preserve/share the run-specific artifacts needed for verification. Current results/checkpoints are Git-ignored, and the locally available Hybrid-MoE metrics file lacks its resolved config/checkpoint/history.
- **Open:** Recover provenance for the 39-row legacy ViHSD summary export before reusing its test scores or linking rows to current experiment IDs.
- **Open:** Reconcile finalist YAML comments, actual values, tracker records, and checkpoint evidence.

### Thesis and demo requirements

- **Open:** University-required thesis template, chapter structure, citation style, length, language details, and submission/defense deadlines.
- **Open:** Demo platform/UI, chosen model, input/output behavior, and how inference dependencies/checkpoints will be delivered.
- **Open:** Whether any ethics or data-use approval is required by the department for the selected public datasets and demo.

## 12. Risks and limitations to report

- Compute and the short thesis schedule may limit model size, number of ablations, and repeated seeds.
- The tracker currently has incomplete provenance and all rows unverified; this limits the strength of present result claims.
- Differences between tasks, tokenizers, preprocessing, class distributions, and baseline models constrain cross-dataset generalization.
- MoE total parameter count, active computation, and wall-clock cost are distinct; reporting only accuracy would give an incomplete comparison.
- Multiple ablations and model selection can create selection bias; predefine gates, keep test evaluation locked, and report the search scope.
- Historical automatically produced test values, when present, are provenance only until the approved finalist protocol is satisfied.

## 13. Working workflow for a new contributor

1. Read this specification, [AGENTS.md](../AGENTS.md), the tracked run ledger, and the detailed experiment plan before proposing or interpreting runs. The project and architecture records under `docs/` are also versioned handoff context.
2. Check the source code and resolved config for implementation-specific claims; documentation and config intent are not proof of execution.
3. Work from the repository root (the `code/` directory in the enclosing workspace). Resolve the environment gaps listed above before installing/running. Java >= 8 is required by the VnCoreNLP preprocessing scripts. Use the cached data only after checking its revision and preprocessing provenance.
4. Prepare data only when needed. ViHSD config uses `data/vihsd_segmented`; create it with `python scripts/preprocess_vihsd.py --output-dir data/vihsd_segmented`. The ViANLI base config points to the Hub dataset; to use the local segmented cache, set `dataset.name=data/vianli_segmented` (for example with `--set dataset.name=data/vianli_segmented`) after running `python scripts/segment_vianli.py --output-dir data/vianli_segmented`.
5. Run a smoke profile before a full run. Example commands from the repository root:

   ```bash
   python train.py --config configs/vianli.yaml --smoke-test --run-id vianli-smoke-s42
   python train.py --config configs/vihsd.yaml --smoke-test --run-id vihsd-smoke-s42
   ```

6. For a proposed experiment, state its question, changed factor, controls, seeds, metrics, validation selection rule, and expected artifacts before running it. Use a unique run ID; the trainer will stop rather than overwrite an existing run ID.
7. Preserve the resolved config, checkpoint, metrics, history, code revision, data/model revision, seed, platform/hardware, and artifact locations. Outputs are written to the configured `checkpoints/` and `results/` directories; these are Git-ignored and must be copied to an agreed durable location for review.
8. Record actual run metadata and metrics in the tracked `experiments/experiment_tracker.csv`; update this specification's current evidence and decision-history sections when interpretation or scope changes. If the local-only `experiments/RESULTS.md` and `analysis_notes.md` are available, keep them synchronized, and preserve anomalies/history.
9. Use validation only to select models. Lock the configuration and seed set before final test evaluation. Then run `python evaluate.py --final-evaluation --config configs/vihsd.yaml --run-id your_locked_finalist_run_id` (replace the run ID and config with the locked run). The flag is a human confirmation, so the tracker/spec must record the lock decision.
10. Update this specification when a stable project decision changes: resolve the corresponding item in section 11 and add the date, decision, rationale/evidence, and impact to the decision history in section 15. A local `docs/DECISIONS.md` may also be updated when available, but it is not part of the shared handoff.

## 14. Sources consulted and handoff availability

The specification is intended to be the primary handoff document. The proposal, literature PDFs,
and sample template are outside this Git repository. Evidence or requirements that exist only in
those materials are identified as such; obtain/share the source files when they are needed to
resolve an open item. Historical segmented data may still be tracked in the current checkout;
the data migration required by `AGENTS.md` is intentionally separate from this documentation
refinement.

- `../paperwork/DECUONG_23521467.pdf` — official detailed proposal and schedule (sibling workspace directory; not tracked in this repository).
- `AGENTS.md` — project scope, experiment governance, and evidence rules.
- `docs/PROJECT.md`, `STATUS.md`, `DECISIONS.md`, `ARCHITECTURE.md` — project definition, state, decisions, and implementation overview.
- `experiments/experiment_tracker.csv` — the experiment ledger; this is the only experiment document currently tracked in the shared repository.
- `experiments/EXPERIMENT_PLAN.md`, `RESULTS.md`, `analysis_notes.md`, `experiment_template.md` — versioned planning and interpretation notes.
- `.agent/context.yaml` — ignored local machine-readable summary; its global verification flag conflicts with the tracker and is not run-level evidence.
- `configs/vianli.yaml`, `vihsd.yaml`, `finalists/*`, `README.md` — task setup and configuration claims.
- `data/{vianli_segmented,vihsd_segmented}/**/dataset_info.json`, `state.json`, and local `DatasetDict` contents — runtime data artifacts; their provenance belongs in `data/manifests/`. Existing tracked copies require a separate migration before removal.
- [Current ViANLI dataset card](https://huggingface.co/datasets/uitnlp/ViANLI) — label schema, cited paper, current card text and license declarations; card metadata must be tied to the local source revision before reusing it.
- [Current ViHSD dataset card](https://huggingface.co/datasets/uitnlp/vihsd) — label names, citation, license metadata, and gated-access notice.
- [ViHSD source paper](https://arxiv.org/abs/2103.11528) — canonical numeric label mapping and original split/data description.
- [ViANLI/NLIMoE publisher article](https://doi.org/10.1016/j.eswa.2025.130109) — direct prior MoE work on ViANLI; reported accuracy is protocol-specific.
- [Masoudnia & Ebrahimpour article](https://link.springer.com/article/10.1007/s10462-012-9338-y) — formal publication metadata for the local survey PDF.
- `results/vianli-hybrid-moe-xlmr-last6-e6-dynamic-head-e4-top2-s42/run_metrics.json` — ignored local, unverified anomaly metrics; historical test metrics are provenance only.
- `results/summary_results/summary_table.csv` and `summary_results.zip` — ignored local legacy ViHSD aggregate export; no seed/config/platform/provenance columns.
- `src/`, `data/`, `scripts/` — current model registry, metrics/losses, and preprocessing implementations.
- `../papers/*.pdf` — four local literature items inventoried above (sibling workspace directory; not tracked here); introductory-page review only.
- `../paperwork/LaTeX2e+Proceedings+Template+ZIP/samplepaper.tex` — sample proceedings template in a sibling workspace directory; not confirmed as a university requirement.

## 15. Decision history

When an **Open** item is resolved, preserve the evidence and reasoning here rather than deleting the question without a trace. Keep entries chronological and distinguish supervisor-approved decisions from working assumptions.

| Date | Decision | Rationale and evidence | Approval/status | Impact on this specification or experiments |
| --- | --- | --- | --- | --- |
| 2026-09-29 | Keep PhoBERT, XLM-R, and mBERT as candidate backbones rather than restricting the study to PhoBERT. | Existing local `docs/DECISIONS.md` (D001) says the thesis evaluates MoE for text classification rather than optimizing PhoBERT specifically. | Recorded project decision; supervisor approval not stated in the source. | Preserved in the scope and candidate backbone list; final backbone priorities remain open in section 11. |
| 2026-09-29 | Treat routing strategy as an experimental variable; do not assume top-k is the only relevant approach. | Existing local `docs/DECISIONS.md` (D002). | Recorded project decision; supervisor approval not stated in the source. | Preserved in the candidate design space and experiment plan. |
| 2026-09-29 | Require external experiment results to be manually recorded and verified. | Existing local `docs/DECISIONS.md` (D003) and the experiment governance in `AGENTS.md`. | Recorded project decision; supervisor approval not stated in the source. | The tracker is the run-level record; as of 2026-10-22 all 51 rows remain marked `unverified`, so this policy has not yet yielded verified runs. |

