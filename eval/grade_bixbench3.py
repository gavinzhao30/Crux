"""Run Edison's pinned artifact grader in an evaluator-only subprocess."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from labs.bixbench3.task9_contract import PROTOCOL, TASK9_PAPER_ID, load_contract

GRADER_COMMIT = "d0e0bbb41222335b1b8878f533a58466a3a782dc"
PASS_SCORE = 0.80
_WORKER = """
import json, sys
from pathlib import Path
from types import ModuleType

# Load only deterministic grading. The upstream package initializer also
# imports its optional model-based process judge.
sys.path.insert(0, sys.argv[1])
grading = ModuleType("bixbench3.grading")
grading.__path__ = [str(Path(sys.argv[1]) / "bixbench3" / "grading")]
sys.modules[grading.__name__] = grading
from bixbench3.grading.grader import grade_paper

spec = json.loads(Path(sys.argv[2]).read_text())
names = set(json.loads(sys.argv[5]))
spec["artifacts"] = [item for item in spec["artifacts"] if item["artifact_id"] in names]
if {item["artifact_id"] for item in spec["artifacts"]} != names:
    raise ValueError("Official spec is missing requested artifacts")
grade_paper(spec, ground_truth_root=Path(sys.argv[2]).parent,
            agent_artifacts_dir=Path(sys.argv[3]), artifact_grades_dir=Path(sys.argv[4]))
"""


def _stage_outputs(source: Path, target: Path, contracts: dict) -> None:
    for contract in contracts.values():
        path = source / contract["path"]
        if path.is_symlink():
            raise ValueError(f"Artifact is a symlink: {path.name}")
        if path.is_file():
            shutil.copyfile(path, target / path.name)


def grade_task9(
    *,
    artifacts_dir: Path,
    references: Path,
    output_dir: Path,
    grader_repo: Path,
    python: Path,
    only: list[str] | None = None,
    self_test: bool = False,
) -> dict:
    grader_repo, references = grader_repo.resolve(), references.resolve()
    artifacts_dir, output_dir = artifacts_dir.resolve(), output_dir.resolve()
    if (
        output_dir.is_relative_to(artifacts_dir.parent)
        or artifacts_dir.is_relative_to(output_dir)
        or references.is_relative_to(artifacts_dir.parent)
    ):
        raise ValueError("Keep evaluator output and references outside the agent run directory")
    commit = subprocess.check_output(
        ["git", "-C", str(grader_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(grader_repo), "status", "--porcelain", "--", "src"], text=True
    )
    if commit != GRADER_COMMIT or changes:
        raise ValueError(f"Use an unmodified Edison grader checkout at {GRADER_COMMIT}")

    contracts = load_contract(include_go=True)
    names = only if only is not None else list(load_contract())
    if not names or len(set(names)) != len(names) or set(names) - contracts.keys():
        raise ValueError("Select unique Task 9 downstream artifact IDs")
    spec_file = references / "artifacts_grading.json"
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if spec["paper_id"] != TASK9_PAPER_ID:
        raise ValueError("Expected the official Task 9 grading spec")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output_dir) as temp:
        scratch = Path(temp)
        staged = scratch / "candidate"
        staged.mkdir()
        chosen = {name: contracts[name] for name in names}
        if self_test:
            for artifact in spec["artifacts"]:
                if artifact["artifact_id"] in chosen:
                    path = references / artifact["ground_truth"]["path"]
                    shutil.copyfile(path, staged / chosen[artifact["artifact_id"]]["path"])
        else:
            if not artifacts_dir.is_dir():
                raise ValueError("Agent artifact directory does not exist")
            _stage_outputs(artifacts_dir, staged, chosen)
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
        subprocess.run(
            [
                str(python.absolute()),
                "-I",
                "-c",
                _WORKER,
                str(grader_repo / "src"),
                str(spec_file),
                str(staged),
                str(scratch / "artifact_grades"),
                json.dumps(names),
            ],
            cwd=scratch,
            check=True,
            timeout=300,
            env=env,
        )
        official = json.loads((scratch / "grade.json").read_text(encoding="utf-8"))
        grades = {item["artifact_id"]: item for item in official["artifacts"]}
        scores = [grades[name]["score"] for name in names]
        result = {
            "protocol": PROTOCOL,
            "grader_commit": GRADER_COMMIT,
            "task": 9,
            "self_test": self_test,
            "artifact_count": len(names),
            "mean": sum(scores) / len(names),
            "fraction_at_least_0_80": sum(score >= PASS_SCORE for score in scores) / len(names),
            "artifacts": {name: grades[name] for name in names},
        }
        if self_test and any(abs(score - 1) > 1e-9 for score in scores):
            raise RuntimeError(
                f"Official Task 9 self-test failed: {dict(zip(names, scores, strict=True))}"
            )
        report = scratch / "report.json"
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        report.replace(output_dir / "grade.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=[9], required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grader-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--only", help="Comma-separated artifact IDs; defaults to nine core outputs"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = grade_task9(
        artifacts_dir=args.artifacts_dir,
        references=args.references,
        output_dir=args.output_dir,
        grader_repo=args.grader_repo,
        python=args.python,
        only=args.only.split(",") if args.only else None,
        self_test=args.self_test,
    )
    print(json.dumps({key: result[key] for key in ("protocol", "mean", "fraction_at_least_0_80")}))


if __name__ == "__main__":
    main()
