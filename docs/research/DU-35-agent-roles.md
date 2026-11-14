# DU-35 — Roles and evidence for a scientific swarm

Reviewed November 14, 2026. This is a literature review and a proposed role design for Crux v2. Recommendations below are engineering inferences from the cited research. Crux performance remains to be measured.

## Three takeaways

1. Preserve independent proposals before sharing, then merge with provenance. Different expertise can expose different analysis choices. Keep the Statistician, Bioinformatician, and Sceptic blind to one another until all initial proposals are recorded. Treat persona diversity as a hypothesis to test. Virtual Lab demonstrates specialist collaboration, and Kosmos demonstrates traceable shared context. Neither establishes that these particular three personas are optimal. [Virtual Lab official implementation](https://github.com/zou-group/virtual-lab), [Kosmos §2.1](https://arxiv.org/html/2511.02824v2)

2. Assign verification to explicit boundaries. Use a referee before belief updates and a contract critic after artifacts exist. POPPER motivates separating experimental design from execution and checking the validity of evidence. MAST documents incomplete verification, premature termination, and coordination failures. Crux's end-only critic is a project design choice that needs its own ablation. [POPPER §§2–3](https://arxiv.org/html/2502.09858), [MAST §4 and Appendix A](https://arxiv.org/html/2503.13657)

3. Evaluate the value of coordination at a fixed total budget. Debate and aggregation have positive results on selected language tasks, while stronger sampling baselines often close the gap. Give the single agent the same model, inputs, tools, and total budget, including the opportunity to self-check. Count every proposer, merge, predictor, referee, critic, and reporter call. [Du et al. §§3, 5](https://arxiv.org/html/2305.14325), [Smit et al. §3](https://arxiv.org/html/2311.17371)

## Evidence and comparison limits

| Primary source and reviewed scope | Finding relevant to roles | What it establishes about compute and scientific validity |
| --- | --- | --- |
| [Kosmos v2, §2.1 and methods](https://arxiv.org/html/2511.02824v2) | Parallel search and data-analysis agents share a structured world model with report-to-source links. Expert review evaluated 102 statements from three representative reports. | Supports traceable evidence memory. This selected report audit does not isolate the causal effect of agent count under a matched single-agent budget. |
| [POPPER v1, §§2–4](https://arxiv.org/html/2502.09858) | A design agent proposes falsifiable implications and an execution agent tests them. Sequential error control depends on implication validity and conditionally valid e-values. | Supports evidence validation. Crux's repeated use of one count matrix and model-estimated likelihoods needs separate justification before claiming analogous error control. |
| [MAST v3, §§3–5 and Appendix A](https://arxiv.org/html/2503.13657) | A taxonomy derived from annotated traces identifies system-design, inter-agent, and verification failures across coding, mathematics, and general-agent systems. | Supports explicit responsibilities and stopping checks. Observational failure labels do not prove that adding a critic causes an improvement on BixBench3. |
| [Google AI co-scientist v1, §§3–4](https://arxiv.org/html/2502.18864v1) | Generation, reflection, ranking, evolution, proximity, and meta-review agents refine hypotheses through tournaments, with external experimental validation. | Demonstrates a scientific workflow and test-time scaling. Hypothesis-quality rankings and selected validations leave the equal-budget artifact-reproduction question open. The identifier now also has a revised v2 abstract, which was checked. |
| [Virtual Lab, official repository](https://github.com/zou-group/virtual-lab) and [published abstract](https://pubmed.ncbi.nlm.nih.gov/40730228/) | A principal investigator coordinates specialist meetings with human feedback and a computational nanobody workflow. | Supports specialist coordination in a demonstrated application. It does not establish which Crux roles improve Task 9 at equal spend. The Nature full text was inaccessible in this review, so detailed ablation claims are omitted. |
| [Du et al. v1, Table 1 and §5](https://arxiv.org/html/2305.14325) | Three agents and two debate rounds improve several reasoning scores over the listed single-agent and majority baselines. | Positive evidence for interaction on those tasks. Authors acknowledge extra compute and potentially incorrect consensus. Table 1 alone cannot establish an equal-token scientific-analysis advantage. |
| [Should we be going MAD? v3, §3 and conclusions](https://arxiv.org/html/2311.17371) | Original debate protocols do not reliably outperform self-consistency or the evaluated Medprompt variant. Tuning agreement and other hyperparameters can improve debate. | Strong counterevidence to an unconditional swarm advantage. Cost/accuracy comparisons are informative, but protocol tuning and QA tasks limit direct transfer to Crux. |
| [More Agents Is All You Need v2, §§4–5](https://arxiv.org/html/2402.05120) | Sampling and voting improve results as the ensemble grows. Main comparisons include ensembles of 40 against a single query. | Supports a sampling baseline. Those comparisons increase inference expenditure and therefore do not isolate coordination under identical total compute. |
| [Mixture-of-Agents v1, §§3.3–3.4](https://arxiv.org/html/2406.04692) | Layered proposal and aggregation improves instruction benchmarks, with model-role ablations and cost analysis. | Useful evidence for role-specific aggregation and cost frontiers. Heterogeneous model ensembles and preference-oriented benchmarks do not establish an equal-model BixBench3 advantage. Its parallel TFLOPs proxy also differs from total compute. |

## One central risk and mitigation

Correlated errors can survive consensus, enter shared memory, and make later agents overconfident. Keep initial proposals independent, attach source and execution references to evidence, deduplicate repeated evidence, and retain contradictory and invalid results. Treat proposer endorsement counts as a heuristic prior whose calibration must be measured. The DU-9 endorsement-based prior is specified by the current issue, so changing it requires team agreement. Record its provenance now and test a uniform-prior alternative if scope allows. [Du et al. §5](https://arxiv.org/html/2305.14325), [Kosmos §2.1](https://arxiv.org/html/2511.02824v2)

## Proposed role list

These responsibilities map to the existing v2 architecture. Their benefit is a testable design hypothesis.

| Role | Responsibility and reason |
| --- | --- |
| Statistician proposer | Independently proposes design, filtering, multiplicity, and effect-size alternatives to expose statistical choices. |
| Bioinformatician proposer | Independently examines identifiers, annotation, and pipeline conventions to expose domain-specific incompatibilities. |
| Sceptic proposer | Independently raises label, duplication, and outlier explanations so data-quality failures get explicit hypotheses. |
| Merge step | Clusters duplicates while retaining proposer provenance and the null option to prevent duplicate options from multiplying evidence. |
| Designer | Specifies diagnostic code, outcome bins, and costs before execution so competing explanations can make checkable predictions. |
| Neutral predictor | Predicts outcomes without having authored the options so likelihood elicitation has a distinct responsibility. |
| Executor | Runs the fixed diagnostic in the isolated environment and returns raw outputs and execution provenance. |
| Referee | Checks execution validity before any belief update so crashes and invalid analyses cannot act as scientific evidence. |
| End critic | Checks completed artifacts against the contract and requests bounded repairs so completion depends on observable outputs. |
| Reporter | Builds methods, rejected, robust, and unresolved records from the event log so each conclusion has an evidence trail. |
| Orchestrator and scoring code | Enforces budgets, EIG selection, updates, and stop rules so model suggestions cannot bypass the run contract. |

Search and memory remain tools with explicit access policies. The official grader remains isolated from all agent roles. These boundaries follow the [Crux engineering contract](../ERD.md).

## Three BixBench3 failure modes the critic must catch

| Failure | Enforceable check | Model-assisted review and response |
| --- | --- | --- |
| Incomplete or malformed deliverables | Verify every required path, nonempty parsable table, required numeric columns including padj, unversioned row keys where required, and valid regulation labels. | Explain which requirement failed and request a bounded repair. |
| Plausible-looking output unsupported by execution | Check artifact hashes and execution references, and independently recompute the specified base_mean check for 20 seeded genes using the agreed normalization. | Review whether the stated method matches recorded code and evidence. Passing this spot check verifies only that invariant. |
| Unsupported certainty or concealed unresolved choices | Cross-check methods and the rejected/unresolved ledger against events. Verify that invalid observations never produced belief updates and required sensitivity outputs exist. | Flag conclusions exceeding the evidence and preserve inconclusive results in the report. |

The first two map to DU-11's artifact checks. The third also needs orchestrator and reporter validation. MAST motivates explicit verification, while the Task 9 checks come from the [Crux ERD](../ERD.md).

## Decisions for implementation and evaluation

- DU-9 should record independent proposals, merge mappings, and prior construction. Persona endorsement can initialize a prior, but repeated agreement supplies no new observation.
- DU-11 should keep execution validity, artifact validity, and scientific uncertainty explicit. A nonsignificant result or failed execution alone cannot reject a biological explanation.
- DU-23 should compare paired cases and seeds with identical information, tool access, sandbox image, search policy, and budget caps. Record actual input/output tokens, total cost, calls, runtime, tool compute, and failures. Equal caps alone do not guarantee equal realized compute.
- Use full Crux versus random selection to isolate selection, and full Crux versus no critic to estimate the end critic's contribution. A single-agent comparison measures the combined system. A budget-matched independent-sampling baseline would further test coordination if time permits.
- The reviewed studies support these experiments, but leave Crux's equal-compute advantage unestablished until its own results are available.
