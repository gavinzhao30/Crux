"""Check one Task 9 contrast and the sandbox's isolation on Modal."""

import argparse
import asyncio
import json
from pathlib import Path

from execution.sandbox import ModalExecutor
from execution.staging import Task9Inputs
from labs.bixbench3.task9 import validate_de_table


async def smoke(args: argparse.Namespace) -> None:
    executor = ModalExecutor(Task9Inputs.load(args.manifest), minimal_image=args.minimal)
    sandbox_id = await executor.open("bixbench3", args.run_id)
    try:
        await executor.verify_isolation(sandbox_id)
        counter_step = 'from pathlib import Path; Path("/work/probe.txt").write_text("crux")\n'
        read_step = 'from pathlib import Path; assert Path("/work/probe.txt").read_text()=="crux"\n'
        for step in (counter_step, read_step):
            result = await executor.exec(sandbox_id, "python", step + "print('RESULT: {}')\n")
            if not result["valid"]:
                raise RuntimeError(result["error"])
        timeout = await executor.exec(sandbox_id, "python", "import time; time.sleep(10)", 1)
        if timeout["valid"]:
            raise RuntimeError("Timed-out command unexpectedly succeeded")
        script_path = Path(__file__).resolve().parents[1] / "labs/bixbench3/smoke_de.R"
        variables = {
            "cell_line": args.cell_line,
            "treatment": args.treatment,
            "reference": args.reference,
        }
        code = "".join(f"{key} <- {json.dumps(value)}\n" for key, value in variables.items())
        result = await executor.exec(
            sandbox_id,
            "r",
            code + script_path.read_text(encoding="utf-8"),
            600,
        )
        if not result["valid"]:
            raise RuntimeError(f"{result['error']}\n{result['stderr']}")
        output = next(
            item
            for item in await executor.collect_outputs(sandbox_id)
            if item["name"] == "smoke_de.csv"
        )
        if findings := validate_de_table(Path(output["path"])):
            raise RuntimeError("; ".join(findings))
        print(output["path"])
    finally:
        await executor.close(sandbox_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cell-line", default="T98G")
    parser.add_argument("--treatment", default="C10")
    parser.add_argument("--reference", default="Ctrl")
    parser.add_argument("--minimal", action="store_true")
    asyncio.run(smoke(parser.parse_args()))


if __name__ == "__main__":
    main()
