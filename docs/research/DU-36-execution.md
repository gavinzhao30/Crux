# DU-36 — Bioinformatics execution recommendation

Reviewed November 14, 2026. This completes the research brief. Runtime estimates below are planning recommendations, with measured values and missing measurements identified separately.

## Three takeaways

1. Keep the working Task 9 micromamba image and preinstall the analysis dependencies. The current pilot starts from counts, so its immediate requirements are R, DESeq2, data.table, Python, pandas, and numpy. Loading a complete raw-read or single-cell environment increases build dependencies without serving that entry point. The DESeq2 workflow takes unnormalized counts and sample metadata. [DESeq2 vignette](https://www.bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html), [Crux image](https://github.com/mian-abd/Crux/blob/ea78f9c180813a7f5444dcbba7febf22370dd796/execution/image.py)

2. Borrow execution patterns selectively. Finch provides notebook-based Python/R/Bash analysis and an optional bioinformatics Docker environment. CellVoyager exposes notebook execution and bounded analysis steps. Biomni offers a much broader environment whose documented full setup exceeds ten hours and requires at least 30 GB disk. These sources inform tooling and repair loops, while Crux's own adapters retain budget and isolation control. [Finch](https://github.com/Future-House/data-analysis-crow), [CellVoyager](https://github.com/zou-group/CellVoyager), [Biomni environment](https://github.com/snap-stanford/Biomni/blob/main/biomni_env/README.md)

3. Cache expensive upstream computation and measure each stage separately. nf-core/rnaseq provides established alignment, quantification, and QC workflows. Use that as a raw bulk-RNA reference, then select the method required by the task. Measure image build, sandbox startup, input verification, diagnostics, analysis, and export separately before scheduling the full comparison. [nf-core/rnaseq 3.27.0](https://nf-co.re/rnaseq/3.27.0/), [Nextflow cache and resume](https://docs.seqera.io/nextflow/cache-and-resume)

## Image decision

Primary recommendation is the existing Modal micromamba image with Python 3.12 and R 4.5, installing from conda-forge and bioconda. The existing minimal fallback contains DESeq2, data.table, pandas, and numpy. Use this fallback if the expanded environment fails to solve. Add edgeR, apeglm, scipy, PyDESeq2, or gseapy only when the selected method or artifact needs them. Preserve stable layers, record resolved package versions and image identity, and reuse the same image across comparison arms. This is a research recommendation, not an image change. [Crux image source](https://github.com/mian-abd/Crux/blob/ea78f9c180813a7f5444dcbba7febf22370dd796/execution/image.py)

| Candidate | Evidence and decision |
| --- | --- |
| Existing Crux micromamba image | Best fit for the active downstream pilot. DU-12/DU-13 report a live DESeq2 contrast, input hashes, read-only mount, forced timeout, and exported outputs. |
| Finch futurehouse/bixbench:aviary-notebook-env | Useful fallback candidate if a required package is missing. It was supplied for the earlier BixBench notebook workflow. Verify Python/R compatibility and pin its digest before adoption. Its relationship to BixBench3 must be evaluated explicitly. |
| Official BixBench3 sandbox | A broad runtime reference. The inspected Dockerfile installs system/Python/R tools and copies Cell Ranger and ARC tarballs unconditionally. Rebuilding it as written therefore requires those build inputs even for a downstream-only task. Reuse selected layers if needed. |
| Biomni full environment | Useful tool catalog. Its documented setup cost and broad dependencies make wholesale adoption a poor fit for the current hackathon pilot. |
| CellVoyager environment | Useful notebook and feedback workflow for single-cell analysis. Its h5ad-oriented workflow differs from this bulk count-matrix entry point. |

Primary evidence for the last four rows is the official [Finch README](https://raw.githubusercontent.com/Future-House/data-analysis-crow/main/README.md), [BixBench3 Dockerfile](https://raw.githubusercontent.com/EdisonScientific/BixBench3/main/sandbox/Dockerfile), [Biomni setup instructions](https://raw.githubusercontent.com/snap-stanford/Biomni/main/biomni_env/README.md), and [CellVoyager README](https://raw.githubusercontent.com/zou-group/CellVoyager/main/README.md).

## Tools and resources

The allocation recommendations are starting points to validate. They are not measured minimum requirements.

| Workflow | Tools | CPU / RAM recommendation | Disk recommendation | Time evidence |
| --- | --- | --- | --- | --- |
| Task 9 profiling and DE from counts | R/DESeq2, data.table, pandas, numpy, hash and schema checks | Keep current 4 vCPU / 16 GiB allocation for paired runs | Reserve 2 GiB scratch/output headroom beyond the image and staged inputs, then measure actual high-water usage | Live one-contrast success reported by DU-12/13. Numeric build/startup/analysis timings unavailable in the inspected evidence. |
| Task 9 raw-read alignment and counting, outside active pilot | Task-compatible STAR/Salmon or other documented alignment/quantification route, samtools or featureCounts where appropriate, QC tools | Planning envelope 16 vCPU / 64 GiB, to revise after a representative pilot | Planning envelope 100 GiB scratch, plus reference/index requirements verified for the chosen aligner | Unmeasured. No end-to-end Modal alignment-plus-counting runtime can be claimed here. |
| Task 11 TempO-Seq stretch | Probe-manifest-specific alignment and counting, then the existing DE environment | DU-46 proposes 32 vCPU / 64 GiB for the compute Function | Planning reserve 60 GiB for the issue's reported 15.8 GB input set plus working files, subject to format and expansion checks | Unmeasured. Pilot one representative FASTQ and extrapolate cautiously before scheduling all 85. |

Resource values for raw workflows are engineering estimates. A generic whole-transcriptome pipeline must not be assumed to implement TempO-Seq probe matching correctly. Confirm task-specific method guidance and supported read layout first. [DU-46 task brief](https://linear.app/mian-abd/issue/DU-46), [nf-core/rnaseq](https://nf-co.re/rnaseq/3.27.0/)

The approved local Task 9 matrix was measured at 60,708 genes by 16 sample columns, 3,366,610 input bytes, and 7,770,624 bytes for a dense int64 count array alone. DataFrame/R objects, model intermediates, libraries, and parallel workers add memory overhead. These input measurements do not establish peak process RAM.

## One risk and mitigation

Environment drift can change scientific results or make paired arms incomparable. The source pins Python and R but leaves several package versions solver-selected. Record the resolved conda package list, image ID, input hashes, git revision, and exact commands in each manifest. Verify imports and one real contrast before the evaluation cohort. Freeze that environment across arms. Keep downloads and installation outside the network-blocked analysis sandbox.

## Existing verification and measurement handoff

DU-12 and DU-13 report live run task9-smoke-live-04, a T98G C10 versus Ctrl DESeq2 table, version stripping preserving _PAR_Y, a read-only mount, and a forced timeout returning invalid. This review inspected the implementation at ea78f9c180813a7f5444dcbba7febf22370dd796 and the issue updates. The numeric runtime log was not present among the inspected tracked files, so the report does not invent cold-start or full-study timings. [DU-12](https://linear.app/mian-abd/issue/DU-12), [DU-13](https://linear.app/mian-abd/issue/DU-13)

For the next execution pilot, capture sandbox.ready startup_seconds, each command's seconds, package/image versions, process peak RSS, and scratch-disk high-water mark. Run the same image once cold and once warm. Report build time separately from per-run time. Verify all required contrasts and exports before treating the one-contrast smoke result as study-level readiness.

The older R3 request for Task 9 alignment-plus-counting time applies to a raw-data workflow. The active pilot uses the published matrix. This research is complete with an explicit measurement gap. It performs no new Modal workload and claims no raw-data timing.
