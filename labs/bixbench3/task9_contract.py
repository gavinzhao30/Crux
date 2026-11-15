"""Agent-visible artifact contracts for the Task 9 downstream pilot."""

import json
from pathlib import Path

GRADER_COMMIT = "d0e0bbb41222335b1b8878f533a58466a3a782dc"

TASK9_PAPER_ID = "10_1186_s12915_024_01879_0_v1"
PROTOCOL = "BixBench3 task 9, downstream pilot (published counts as input)"


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
            "For each cell line, produce DESeq2 treatment-versus-Ctrl comparisons and the shared "
            "and D4-specific gene sets requested by the artifact contract. Use biological "
            "replicates and record analysis decisions, diagnostics and unresolved issues in "
            "/outputs/METHODS.md. Use the supplied artifact names and column names."
        ),
        "local fallback from public artifact contracts",
    )
