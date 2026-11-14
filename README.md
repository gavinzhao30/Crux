# Crux

Crux agents propose competing scientific explanations, select diagnostics by expected information gain and cost, and update their beliefs from observations. The shared engine targets BixBench3 Task 9, BoxingGym Location Finding, and the Jetson bench. See the [engineering requirements](docs/ERD.md) for the lab contracts and repository layout.

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from the repository root. The first sync installs Python 3.12 if needed and resolves the environment from the lockfile. The tests run locally without API credentials.

```sh
uv sync
uv run pytest -q
uv run ruff check .
```

For live integrations, set the environment variables listed in `.env.example`. To load values from a local `.env` file, use `uv run --env-file .env` before your command. Run settings and ablation options are defined in `core/config.py`.

## Leakage controls

Agent-visible inputs exclude benchmark ground truth and planted-fault truth tables. The ERD reserves grading for a separate process in `eval/`. Ordinary run logs and artifacts are ignored by Git. Review files in `runs/demo/` before staging them.

## Which memory holds what

The run event log records observations and belief updates for replay. The planned past.dev integration shares evidence between roles within a run. The role-only configuration provides the comparison for measuring the effect of sharing evidence.

## Credits

The stack uses FastAPI, Uvicorn, Pydantic, HTTPX, the OpenAI SDK, exa-py, Modal, NumPy, pandas, and Langfuse. Development tooling includes uv, pytest, and Ruff. The benchmark designs reference BixBench3 and BoxingGym.

## Contributing

Use one branch per Linear issue and a small pull request into `main`. Run the tests and lint checks above. Credit borrowed code, libraries, models, and APIs here.

Every commit must be reviewed and made by a teammate during the event, using their own Git name, email, and GitHub login. Omit co-author and generated-by trailers from commits and pull requests. List human contributors only.
