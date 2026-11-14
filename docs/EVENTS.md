# Run events: the contract between the engine, the API and the UI

Every run writes one JSON object per line to `runs/<run_id>/events.jsonl`. The orchestrator writes it, the API server streams it (it replays the file, then follows new lines), and the UI renders it. Replay files in `runs/demo/` use the same format. This file is the contract: change it only after saying so in Discord.

## Envelope

```json
{"run_id": "r-20261114-153012-a1b2", "seq": 17, "ts": 1794700212.481, "type": "belief.updated", "payload": {}}
```

* `seq` starts at 0 and increases by 1 within a run, with no gaps.
* `ts` is Unix time in seconds (float).
* Objects inside payloads are the pydantic models from `docs/ERD.md` §4.2 dumped with `model_dump(mode="json")`. Their field names are the contract too.
* Consumers must ignore unknown event types and unknown payload fields.

## Event types and payloads

| Type | Payload | Notes |
| --- | --- | --- |
| `run.started` | `{"lab": str, "goal": str, "config": {...}, "started_at": iso8601}` | `config` holds the ablation flags and the role → model map |
| `profile.done` | `{"evidence": [Evidence]}` | The first look at the data or setup |
| `question.opened` | `{"question": Question}` | Includes the ones raised by the fork library |
| `hypothesis.proposed` | `{"hypothesis": Hypothesis}` | One event per option, including `h0` (`is_null: true`) |
| `hypothesis.merged` | `{"question_id": str, "kept": str, "merged": [str], "priors": {hypothesis_id: float}}` | `priors` is the full prior over the question's options after merging |
| `experiment.proposed` | `{"experiment": Experiment}` | `code` may be long; the UI shows it collapsed |
| `experiment.scored` | `{"question_id": str, "ranking": [{"experiment_id": str, "kind": str, "goal": str, "eig_bits": float, "est_cost_s": float, "est_cost_usd": float, "score": float}], "chosen": str}` | `ranking` is sorted by `score`, best first. For numeric labs `experiment_id` is a design label such as `"x=1.75"` |
| `experiment.started` | `{"experiment_id": str, "question_ids": [str], "kind": str}` | |
| `experiment.log` | `{"experiment_id": str, "stream": "stdout" \| "stderr", "line": str}` | One line per event, truncated to 500 characters |
| `experiment.finished` | `{"observation": Observation}` | `bin_id` is set for categorical labs, `value` for numeric ones |
| `referee.verdict` | `{"experiment_id": str, "valid": bool, "note": str}` | Invalid runs never produce a `belief.updated` |
| `belief.updated` | `{"question_id": str, "experiment_id": str, "observed": str \| float, "prior": {hypothesis_id: float}, "posterior": {hypothesis_id: float}}` | `observed` is the bin id, or the measured value for numeric labs |
| `model.curves` | `{"question_id": str, "x": [float], "families": {hypothesis_id: {"mean": [float], "sd": [float], "weight": float}}, "points": [{"x": float, "y": float}]}` | Numeric labs only, after each update: what the UI plots |
| `hypothesis.rejected` | `{"question_id": str, "hypothesis_id": str, "evidence_id": str \| null, "reason": str}` | Feeds the rejected ledger |
| `question.settled` | `{"question_id": str, "hypothesis_id": str, "posterior": float}` | Max posterior ≥ 0.9 |
| `question.robust` | `{"question_id": str, "answer_key": str}` | Every active option leads to the same deliverable |
| `question.unresolved` | `{"question_id": str, "reason": str, "top": [str]}` | Best EIG under 0.05 bits, or out of experiments |
| `surprise.raised` | `{"question_id": str, "experiment_id": str, "reason": str, "z_score": float \| null, "null_mass": float \| null}` | Followed by new `hypothesis.proposed` events when options are regenerated |
| `search.done` | `{"experiment_id": str, "query": str, "urls": [str], "blocked": [str], "bin_id": str \| null}` | `blocked` lists URLs the leakage policy removed |
| `memory.written` | `{"evidence_id": str, "author_role": str, "backend": str}` | |
| `memory.recalled` | `{"reader_role": str, "query": str, "evidence_ids": [str], "authors": [str], "backend": str, "latency_ms": float}` | `authors[i]` wrote `evidence_ids[i]`; this is how the UI shows one agent using another's record |
| `artifact.written` | `{"artifact": Artifact}` | |
| `critic.finding` | `{"finding": CriticFinding}` | |
| `lesson.written` | `{"lesson": Evidence}` | |
| `budget.tick` | `{"cost_usd": float, "llm_calls": int, "tokens_in": int, "tokens_out": int, "seconds": float}` | After every model call or experiment; drives the header's cost and time |
| `run.finished` | `{"report": Report, "status": "ok" \| "budget_exhausted" \| "stopped"}` | Always the last event of a successful run |
| `run.error` | `{"message": str, "where": str}` | The last event of a failed run |

## Rules

* The orchestrator writes `run.started` first and exactly one of `run.finished` or `run.error` last.
* Replaying a file through `core/replay.py` must rebuild the same final beliefs as the recorded `belief.updated` events.
* Nothing from the grader ever appears in an event file. Grades live in `eval/results/`.
* Keys and other secrets never appear in any payload.
