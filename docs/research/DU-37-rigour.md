# DU-37 — Rigour, experiment selection, and improvement

Reviewed November 14, 2026. This is a research synthesis and evaluation protocol for Crux v2. No new benchmark, lesson-learning comparison, or scientific discovery is reported.

## Three takeaways

1. Use full expected information gain with explicit outcome uncertainty and check it against downstream utility. BED-LLM already combines LLM-derived beliefs, candidate questions, and EIG selection. Its analysis distinguishes unpredictable outcomes from informative outcomes. Crux should log the likelihood source, preserve a null explanation, and test calibration. Its EIG mechanism alone cannot establish novelty. [BED-LLM §§3–4](https://arxiv.org/html/2508.21184)

2. Evaluate experiment selection independently of other improvements. BoxingGym provides generative environments and separates design informativeness from prediction performance. Collaborative Battleship shows that computational Bayesian strategies can improve information seeking, while asking high-EIG questions alone need not maximize task success. Compare Crux EIG with random designs using identical proposed model families and observations available so far. Report held-out prediction error and cost alongside information measures. [BoxingGym §3.2](https://arxiv.org/html/2501.01540), [Collaborative Battleship §§3–4](https://arxiv.org/html/2510.20886)

3. Test artifact checking and procedural memory as separate interventions. HeurekaBench's critic ablations vary by model and placement. BLADE motivates preserving justified analysis alternatives and evaluating the decisions in executable analyses. For Crux, compare critic on/off independently from lessons on/off and preserve unresolved or robust-to-choice outcomes in the report. [HeurekaBench §4.3.2](https://proceedings.iclr.cc/paper_files/paper/2026/file/12d25b1673452b017fe9c866d9f494a0-Paper-Conference.pdf), [BLADE §§4–5](https://arxiv.org/html/2408.09667)

## Prior art and the scope of the contribution

| Source reviewed | Existing mechanism or evidence | Implication for Crux |
| --- | --- | --- |
| BED-LLM v3, §§3–4 and experiments | Builds probabilistic models from LLMs, proposes queries, estimates EIG, and updates beliefs. Discusses overconfidence and incompatibility with prior observations. | Claim an evaluated application to executable scientific-analysis choices and artifacts. Avoid claiming the invention of LLM-guided BED. |
| BoxingGym v2, §3.2, experiments and Location Finding appendix | Uses generative scientific environments, EIG evaluation, predictive error, and communication of discovered models. | A useful controlled test of selection. Our count-matrix artifact pilot addresses a different outcome and must be reported separately. |
| Collaborative Battleship v2, §§3–4 and SMC appendix | Uses Monte Carlo Bayesian inference for information seeking, action selection, and deciding when to ask. The paper reports a weaker-model comparison at about 1% of GPT-5's cost under its game setup. | Supports computing selection and considering whether another diagnostic is worth its cost. The game-specific cost result cannot predict Crux savings. |
| HypoAgents v1, methodology and experiments | Uses novelty/relevance/feasibility priors, literature retrieval, LLM likelihood scores, Bayesian updates, and entropy-driven hypothesis refinement. Evaluation includes judged hypothesis quality and entropy. | Bayesian multi-agent hypothesis refinement is prior art. Validate likelihood calibration and executable outcomes independently of improving a hypothesis-quality judge score. |
| HeurekaBench, §4.3.2 and Table 3 | Studies critic placement with Biomni on single-cell research questions, with model-dependent benefits and harms. | Supports an end-critic ablation and bounded repairs. It cannot establish a universal critic benefit or a Task 9 score increase. |
| BLADE v3, §§3–5 | Evaluates conceptual variables, transformations, and statistical models while accommodating multiple justified analyses and equivalent executable representations. | Analysis-choice disagreement already has a benchmark literature. Preserve substantive alternatives and evaluate their effect on the contracted deliverable. |

Sources for the last three rows are [HypoAgents](https://arxiv.org/html/2508.01746), [HeurekaBench](https://proceedings.iclr.cc/paper_files/paper/2026/file/12d25b1673452b017fe9c866d9f494a0-Paper-Conference.pdf), and [BLADE](https://arxiv.org/html/2408.09667). All paper-specific findings above are reported evidence. The Crux implications are engineering inferences.

The proposed one-sentence contribution claim is **Crux tests whether cost-aware selection of executable diagnostics can resolve agents' analysis disagreements and improve reproducible scientific artifacts under matched budgets.** This describes the contribution being evaluated. A demonstrated improvement requires the corresponding results. This review establishes no worldwide priority claim.

## Correcting two overbroad project summaries

The research library's “critic +22%” should retain its scope. HeurekaBench Table 3 gives GPT-OSS-120B a single-run correctness change from 2.04 to 2.49, about a 22% relative increase. Its three-run means are 2.08 to 2.40. Qwen3's three-run means decrease from 1.85 to 1.73 with the end critic. These are judged correctness scores on a 1–5 scale, not percentage-point accuracy gains. The study also reports harms from plan critics. [HeurekaBench, Table 3, PDF page 9](https://proceedings.iclr.cc/paper_files/paper/2026/file/12d25b1673452b017fe9c866d9f494a0-Paper-Conference.pdf)

The older R4 mapping assigns experiment selection to the critic. Use the current v2 division of responsibilities. Proposers raise alternatives, a designer proposes fixed diagnostics, a neutral predictor supplies outcome likelihoods, the scoring engine/orchestrator selects, execution computes the observation, the referee gates validity, and the end critic checks artifacts. This is an interpretation of the current project contract, without editing older project documents. [Technical spec v2](https://linear.app/mian-abd/document/technical-spec-v2-erd-engine-data-model-labs-devin-briefs-4909aa696e12)

## Concrete mapping to Task 9

| Analysis question | Candidate diagnostic and predeclared outcomes | Effect on the deliverable |
| --- | --- | --- |
| Are two libraries independent or duplicated? | Hash and compare count columns. Outcomes distinguish exact identity, near identity requiring investigation, and ordinary distinct columns. | Flag duplicate biological evidence and run the documented repair or sensitivity analysis. Correlation alone cannot establish the correct sample identity. |
| Is a label swap plausible? | Compare sample-to-sample relationships with the supplied groups using a predeclared statistic. Outcomes can be consistent, discordant, or inconclusive. | Flag uncertainty and compare supported alternatives. Use no evaluator truth labels to infer a repair. |
| Does a defensible filtering choice affect the contracted output? | Run two justified filtering settings with fixed comparison metrics and outcome bins. | Record sensitivity or robust-to-choice results and report unresolved alternatives where needed. |
| Do gene identifiers match the required row key? | Inspect identifiers against the public output contract and detect collisions after version normalization. | Apply the contract-defined transformation or flag collisions. This is often a deterministic contract check and should be performed directly, without spending EIG calls on an already-specified requirement. |

Rank only diagnostics that can distinguish active alternatives and affect the deliverable. The existing categorical score is impact times EIG discounted by estimated runtime. Describe it as that runtime-discounted score. It does not by itself optimize dollar cost across arbitrary model calls. Include actual cost and runtime in evaluation and test ranking sensitivity to noisy likelihoods and omitted hypotheses.

## One central risk and mitigation

A self-confirming evidence loop can produce a confident posterior and a persuasive success story from poorly calibrated likelihoods or contaminated lesson memory. Distinct proposer endorsements remain a heuristic prior. Measure calibration against observed bins, deduplicate evidence, retain invalid and inconclusive results, preserve the null explanation, and freeze all development-derived lessons before held-out comparisons. Statistical validity must be justified separately when the same data are used repeatedly. [BED-LLM §4](https://arxiv.org/html/2508.21184), [HypoAgents methodology](https://arxiv.org/html/2508.01746)

## Protocol for measuring improvement from reviewed lessons

1. Fix the development and held-out cohort before examining outcomes. Use clean plus the designated dev perturbations to develop procedural lessons. Keep evaluator truth, reference artifacts, and grades outside agent prompts and memory.
2. Review each lesson's applicable conditions, source run and code references, mistake, diagnostic, repair, and exceptions. Store only the reviewed procedural content and approved provenance. Freeze the corpus by content hash before the comparison.
3. Run lessons=on and lessons=off on paired held-out cases with identical input hashes, seeds, model IDs, tools, image, dollar/call caps, and time limits. Scope mutable evidence memory per run. Make the frozen lesson corpus the intended context difference and account for its extra prompt tokens.
4. Record official continuous artifact scores, fraction at least 0.80, diagnostics before a detected fault, fault detection/repair where independently reviewed, tokens, cost, runtime, and every failed or timed-out run. Compare paired deltas and report uncertainty. A small planted-fault cohort supports a pilot claim with limited generalizability.
5. Keep all requested cases in the denominator. Treat failures under the predeclared evaluation rule. Freeze selection rules for the hero examples and present aggregate outcomes alongside them.
6. To isolate selection, compare EIG and random selection with other settings fixed. To isolate the critic, compare critic on/off. To test the combined system, compare the same-budget single agent. Changes in more than one mechanism limit causal attribution.
7. Describe a successful result as improvement through reviewed procedural context. Model weights are unchanged by this protocol. Report null or negative results and the retrieval cost as well.

DU-30 at aea8bd1e929244c79d3b94bec0b2412f884ff08a implements corpus provenance, frozen manifests, applicability gates, and paired context checks. Its documented live demo uses a synthetic lesson and no model or analysis pipeline. Actual reviewed study lessons, integration into real runs, and held-out grading remain pending. This research specifies that experiment and does not claim it has passed. [DU-30](https://linear.app/mian-abd/issue/DU-30), [lesson implementation](https://github.com/mian-abd/Crux/blob/aea8bd1e929244c79d3b94bec0b2412f884ff08a/docs/LESSONS.md)
