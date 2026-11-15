"""Live documentation smoke check. Requires EXA_API_KEY.

Without --policy it uses the Task 9 policy (search.task9_policy).

uv run python -m search.demo --query "DESeq2 documentation alpha default"
uv run python -m search.demo --policy work/task-policy.json --query "DESeq2 alpha default"
"""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from search.exa import ExaSearch
from search.policy import SearchPolicy


async def main(policy_path: Path | None, query: str, root: Path) -> None:
    if policy_path is None:
        from search.task9_policy import Task9SearchPolicy

        policy = Task9SearchPolicy().rules
    else:
        policy = SearchPolicy.model_validate_json(policy_path.read_text(encoding="utf-8"))
    async with ExaSearch(policy, "search-demo-" + uuid.uuid4().hex, root) as search:
        print(json.dumps((await search.lookup(query)).model_dump(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--root", default=Path("runs"), type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.policy, args.query, args.root))
