# DU-9 live acceptance — November 14, 2026

Acceptance passed for the clean Task 9 downstream pilot and the designated
duplicate-library development perturbation. These checks validate proposal
generation and event recording. They provide no artifact-grading or comparative
benchmark score.

| Input case | Distinct merged questions | Recorded events | Required event types |
| --- | --- | --- | --- |
| case-a, clean published counts | 8 | 125 | All three present |
| case-b, duplicate-library dev case | 6 | 123 | All three present |

Required types are question.opened, hypothesis.proposed, and hypothesis.merged.
Sequence numbers were checked for continuity. Every final question's priors sum
to one and its null hypothesis has mass 0.15.

## Material differences reviewed

The clean case separates DESeq2 design formulas, low-count filtering, multiple
testing, gene-ID handling, normalization, metadata consistency, replicate
coverage, and sample outliers.

The dev case separates design formulas, sample exclusion, multiple testing,
gene-ID handling, normalization, and shared/specific gene-set definitions.
This satisfies the requirement for at least two materially different questions
on each case. It does not establish that every question is useful or that the
planted fault will be repaired downstream.

## Inputs and isolation

The explicitly approved input was artifact1_tagseq_featurecounts_matrix.tsv
from BixBench3 release v1.0.0, Task 9, paper ID
10_1186_s12915_024_01879_0_v1. Its verified SHA-256 is
9fffdcac8fd8d7098928cd7fb6adf8bd707e1d4818703327cc1670f00b545ab3.
This matrix is the declared input exception for Crux's downstream pilot.
The raw matrix is excluded from this commit.

The sample metadata came from the same release's agent-visible
data/metadata/sample_sheet.tsv. Technical-run rows were collapsed by sample_name,
then matched to the 16 matrix columns. Conditions were normalized from
CRISPR_Control to Ctrl and REST_KO_<clone> to the clone label.

The DU-43 generator's _load, _variants, and _apply functions at revision
0fbeb549cf6be64a7608367e62d762093c1b1c36 were used with numpy seed 43.
Both cases have blinded sample IDs.
Only the first duplicate_library variant was applied, the dev perturbation
specified by DU-43. No held-out case was evaluated. Case identities and the
perturbation metadata were withheld from the proposers.

Profiles were computed from counts and metadata, including library sizes, zero
fractions, log1p-count correlations, and exact column duplicates. The brief used
the public pilot artifact contracts and locally written DESeq2 method guidance,
as allowed by DU-14's fallback. It was not the gated official task prompt.
The existing fake memory adapter supplied empty recall for both independent runs.

## Evidence

Committed under runs/demo/du9-acceptance-case-a-04/ and
runs/demo/du9-acceptance-case-b-04/ are profile.json, proposals.json, and events.jsonl.
Each log is a proposal-stage trace, so the production run's terminal event is
outside its scope.

Clean input after blinding, SHA-256
8696a952bf278c430ed71ff174b2192de09d1da04af77785025d899208fb985e.

Dev input after perturbation, SHA-256
feefd1ecb771360ae07fa97c8854d3346883d2c3c1eb934f6d7d50ca3971b65e.

Clean event log, SHA-256
8cf3802bd37fbd84c9cdb4c8695281922a1ca9e5f13db2a68c6f5a5bbdd4ffae.

Dev event log, SHA-256
4d45bd51008e5f8bd7f0b3ac34c42be5872c6b02f8ecfc92e99b207fbaccf232.

To rerun with a fresh run ID and an API key in the local environment file,
use the saved profile as input.

```sh
uv run --env-file .env python -m scripts.run_proposers --input runs/demo/du9-acceptance-case-a-04/profile.json --run-id du9-clean-repeat
uv run --env-file .env python -m scripts.run_proposers --input runs/demo/du9-acceptance-case-b-04/profile.json --run-id du9-dev-repeat
```

## Failures and fixes observed during acceptance

Attempts 01 and 02 failed structured-output validation because an analysis-model
question was classified as an executable numeric hypothesis without model code.
The persona prompts now explicitly require categorical questions for bixbench3.
Attempt 03 reached merging but grouped options with conflicting answer keys.
The merge handler now partitions incompatible options without discarding them.
A regression test checks that preservation and normalized priors.

Attempt 04 passed both cases. The clean case reused four cached live model
responses from attempt 03. The dev case made four fresh calls to gpt-4.1-mini,
with 20,280 input tokens and 3,260 output tokens reported by the wrapper.
Earlier failed attempts also consumed API calls. The wrapper lacks configured
model pricing, so its zero recorded dollar total must not be interpreted as
zero spend. No exact total cost is claimed.
