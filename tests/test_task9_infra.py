import asyncio
import json
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from execution.sandbox import ModalExecutor, _Run
from execution.staging import COUNTS, MANIFEST, SAMPLES, Task9Inputs, stage
from labs.bixbench3.bins import choose_bin
from labs.bixbench3.task9 import validate_de_table
from search.task9_policy import GuardedSearch, Task9SearchPolicy


def test_stage_snapshots_only_declared_files(tmp_path):
    class Volume:
        def __init__(self):
            self.files = {}

        @contextmanager
        def batch_upload(self, *, force):
            assert force
            yield self

        def put_file(self, source, name):
            self.files[name] = source.read_bytes()

    counts, samples = tmp_path / COUNTS, tmp_path / "my-samples.tsv"
    counts.write_text("matrix")
    samples.write_text("samples")
    (tmp_path / "truth.json").write_text("{}")
    volume = Volume()
    inputs = stage(volume, counts, samples, None)
    assert inputs.files[COUNTS] == sha256(b"matrix").hexdigest()
    assert set(volume.files) == {
        f"/{inputs.bundle_id}/{name}" for name in (COUNTS, SAMPLES, MANIFEST)
    }
    assert Task9Inputs(**json.loads(inputs.json())) == inputs


@pytest.mark.parametrize(
    "stdout,exit_code,valid",
    [
        (['RESULT: {"hits": 1}\n'], 0, True),
        (["RESULT: {}\n", "RESULT: {}\n"], 0, False),
        (["RESULT: [0]\n"], 0, False),
        (["RESULT: {}\n"], 1, False),
        ([], 124, False),
    ],
)
def test_executor_requires_one_result_from_successful_command(tmp_path, stdout, exit_code, valid):
    async def stream(lines):
        for line in lines:
            yield line

    process = SimpleNamespace(
        stdout=stream(stdout),
        stderr=stream(["stderr\n"]),
        wait=SimpleNamespace(aio=AsyncMock(return_value=exit_code)),
    )
    remote = SimpleNamespace(
        exec=SimpleNamespace(aio=AsyncMock(return_value=process)),
        filesystem=SimpleNamespace(write_text=SimpleNamespace(aio=AsyncMock())),
    )
    files = {COUNTS: "b" * 64, SAMPLES: "c" * 64}
    bundle_id = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    executor = ModalExecutor(Task9Inputs(bundle_id, files), tmp_path)
    executor._runs["test"] = _Run(remote, tmp_path, "run")
    result = asyncio.run(executor.exec("test", "python", "print('hello')"))
    assert result["valid"] is valid
    assert result["returncode"] == exit_code
    logs = [json.loads(line) for line in (tmp_path / "execution.jsonl").read_text().splitlines()]
    assert any(item["type"] == "experiment.log" and item["stream"] == "stderr" for item in logs)
    assert result["result"] == ({"hits": 1} if valid else None)


def test_guarded_search_rejects_leaks_and_logs_returned_urls():
    allowed = SimpleNamespace(url="https://bioconductor.org/packages/DESeq2", claim="alpha default")
    leaked = SimpleNamespace(
        url="https://bioconductor.org.example.org/packages/DESeq2", claim="unsafe site"
    )
    snippet = SimpleNamespace(
        url="https://bioconductor.org/another", claim="T98G published results"
    )
    provider = AsyncMock(return_value=[allowed, leaked, snippet])
    events = []
    search = GuardedSearch(
        provider, Task9SearchPolicy(), lambda name, payload: events.append((name, payload))
    )
    with pytest.raises(PermissionError):
        asyncio.run(search.find("Look up T98G results"))
    provider.assert_not_called()
    assert asyncio.run(search.find("DESeq2 default alpha")) == [allowed]
    assert (leaked.url, True) in [
        (payload["url"], payload["blocked"])
        for event, payload in events
        if event == "search.result"
    ]
    assert not search.policy.allows_url("https://bioconductor.org/%31%30.1234%2Fabc")


def test_bins_are_exclusive_and_allow_no_code_execution():
    rules = [("lower", "x <= 2"), ("higher", "x > 2")]
    assert choose_bin(rules, {"x": 3}) == "higher"
    with pytest.raises(ValueError, match="exactly one"):
        choose_bin([("one", "True"), ("two", "True")], {})
    with pytest.raises(ValueError, match="Unsupported"):
        choose_bin([("unsafe", "__import__('os').system('false')")], {})


def test_validate_de_output_rejects_versioned_gene_ids(tmp_path: Path):
    path = tmp_path / "result.csv"
    header = "gene_id,gene_symbol,base_mean,log2_fold_change,lfc_se,stat,p_value,padj\n"
    path.write_text(header + "ENSG00000000001,TEST,4,1,0.1,5,0.01,0.02\n")
    assert validate_de_table(path) == []
    path.write_text(header + "ENSG00000000001.2,TEST,4,1,0.1,5,0.01,0.02\n")
    assert validate_de_table(path) == ["DE table must have unique, unversioned gene IDs"]


def test_export_replaces_previous_snapshot_even_when_remote_is_empty(tmp_path):
    files = {COUNTS: "b" * 64, SAMPLES: "c" * 64}
    bundle_id = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    executor = ModalExecutor(Task9Inputs(bundle_id, files), tmp_path)
    read = AsyncMock(return_value=b"test")
    listing = AsyncMock(side_effect=['{"nested/a.csv": 4}', '{"broken.csv": 6}', "{}"])
    process = SimpleNamespace(
        stdout=SimpleNamespace(read=SimpleNamespace(aio=listing)),
        wait=SimpleNamespace(aio=AsyncMock(return_value=0)),
    )
    remote = SimpleNamespace(
        exec=SimpleNamespace(aio=AsyncMock(return_value=process)),
        filesystem=SimpleNamespace(read_bytes=SimpleNamespace(aio=read)),
    )
    executor._runs["test"] = _Run(remote, tmp_path, "run")

    async def collect():
        first = await executor.collect_outputs("test")
        assert [artifact["name"] for artifact in first] == ["nested/a.csv"]
        previous = tmp_path / "artifacts/nested/a.csv"
        assert previous.read_bytes() == b"test"
        with pytest.raises(ValueError, match="changed during export"):
            await executor.collect_outputs("test")
        assert previous.read_bytes() == b"test"
        assert await executor.collect_outputs("test") == []
        assert list((tmp_path / "artifacts").iterdir()) == []

    asyncio.run(collect())
