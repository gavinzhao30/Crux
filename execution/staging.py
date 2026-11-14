"""Stage only the declared Task 9 pilot inputs outside the sandbox."""

import argparse
import json
import shutil
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

COUNTS = "tagseq_featurecounts_matrix.tsv"
SAMPLES = "samples.tsv"
METADATA = "inputs_metadata.json"
MANIFEST = "input_manifest.json"
VOLUME_NAME = "crux-task9-inputs"


@dataclass(frozen=True)
class Task9Inputs:
    bundle_id: str
    files: dict[str, str]

    def __post_init__(self):
        allowed = {COUNTS, SAMPLES, METADATA}
        if not {COUNTS, SAMPLES} <= self.files.keys() <= allowed:
            raise ValueError("Task 9 requires counts and samples, with optional input metadata")
        if any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in self.files.values()
        ):
            raise ValueError("Expected lowercase SHA-256 digests")
        expected = sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest()
        if self.bundle_id != expected:
            raise ValueError("Bundle ID does not match its input digests")

    @classmethod
    def load(cls, path: Path) -> "Task9Inputs":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def json(self) -> str:
        return (
            json.dumps({"bundle_id": self.bundle_id, "files": self.files}, indent=2, sort_keys=True)
            + "\n"
        )


def stage(volume: modal.Volume, counts: Path, samples: Path, metadata: Path | None) -> Task9Inputs:
    if counts.name != COUNTS:
        raise ValueError(f"Supply the published Task 9 matrix named {COUNTS}")
    sources = {COUNTS: counts, SAMPLES: samples}
    if metadata is not None:
        if metadata.name != METADATA:
            raise ValueError(f"Supply only the task's {METADATA} as optional metadata")
        sources[METADATA] = metadata
    with TemporaryDirectory() as directory:
        snapshot = Path(directory)
        digests = {}
        for name, path in sources.items():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Expected a regular input file for {name}")
            destination = snapshot / name
            shutil.copyfile(path, destination)
            with destination.open("rb") as file:
                digests[name] = file_digest(file, "sha256").hexdigest()
        bundle_id = sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
        inputs = Task9Inputs(bundle_id, digests)
        (snapshot / MANIFEST).write_text(inputs.json(), encoding="utf-8")
        with volume.batch_upload(force=True) as upload:
            for name in (*sources, MANIFEST):
                upload.put_file(snapshot / name, f"/{bundle_id}/{name}")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    inputs = stage(volume, args.counts, args.samples, args.metadata)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(inputs.json(), encoding="utf-8")
    print(inputs.bundle_id)


if __name__ == "__main__":
    main()
