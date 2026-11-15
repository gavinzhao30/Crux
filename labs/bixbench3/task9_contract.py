"""Agent-visible artifact contracts for the Task 9 downstream pilot."""

import json
from pathlib import Path

TASK9_PAPER_ID = "10_1186_s12915_024_01879_0_v1"
PROTOCOL = "BixBench3 task 9, downstream pilot (published counts as input)"
# Pinned Edison grader commit. Lives here, not in eval/, so run provenance in core/ can
# record it without importing the grader (agent packages must never import eval/).
GRADER_COMMIT = "d0e0bbb41222335b1b8878f533a58466a3a782dc"


def load_contract(*, include_go: bool = False) -> dict:
    artifacts = json.loads(
        Path(__file__).with_name("task9_contract.json").read_text(encoding="utf-8")
    )
    return {
        name: artifact
        for name, artifact in artifacts.items()
        if include_go or artifact["kind"] != "go"
    }


def load_prompt(path: Path | None = None) -> tuple[str, str]:
    if path is not None:
        prompt = path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("The supplied Task 9 prompt is empty")
        return prompt, "supplied Task 9 prompt"
    return (
        (
            "Reproduce the downstream Task 9 analysis from the staged counts and sample sheet. "
            "After checking sample labels and library quality, fit DESeq2 once per cell line "
            "with design ~ condition using all retained conditions and biological replicates "
            "for that cell line. Extract every treatment-versus-Ctrl contrast from that same "
            "fitted model. Produce the shared and D4-specific gene sets requested by the "
            "artifact contract. Record analysis decisions, diagnostics and unresolved issues in "
            "/outputs/METHODS.md. Use the supplied artifact names and column names. "
            "Follow each artifact's row-key requirements and allowed values exactly. Strip the "
            "Ensembl version number from exported gene_id values while preserving _PAR_Y, "
            "and check that row keys are unique before submitting. When "
            "/data/inputs_metadata.json provides gene_symbol_by_gene_id, use that mapping for "
            "gene symbols. If versions differ, match on the stable ID only when it maps "
            "unambiguously. Record unresolved gene-symbol mappings in METHODS.md."
        ),
        "local fallback from public artifact contracts",
    )
