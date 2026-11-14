# Three independent proposers

Call agents.proposer.propose with the lab brief, profile evidence, injected LLM,
memory adapter, EventBus, and RunConfig. The return value contains questions,
hypotheses, source mappings, and library diagnostic templates. Call once per
proposal round. The caller owns lifecycle events and subsequent execution.

The three initial calls use the existing proposer model configuration and separate
messages. Each role recalls evidence under its own reader identity. The merge
call uses the same configured cheap proposer model after the initial calls finish.

Each initial question has 2–4 substantive options. The engine adds a null option
when enabled. Merge validation checks complete source coverage, uniqueness, kinds,
option ownership, answer keys, and identical code for merged executable models.
Options grouped together with different answer keys or model implementations
are split deterministically to preserve every distinct alternative.
It cannot establish semantic equivalence or scientific correctness on its own.

Prior weights use distinct proposer support plus one for smoothing. They receive
85% total mass when the null option is enabled, and 100% otherwise. Library-only
options receive the smoothing weight without a fabricated proposer endorsement.
Support counts are a heuristic prior and need calibration in evaluation.

Events contain the initial options with pre_merge=true, retained final options,
and merge mappings with the complete final prior. Initial options have zero
prior until the merge assigns probability mass. Consumers should apply the merge
mapping before scoring the question. Generated model code is statically screened
here and must execute through the existing sandbox path later.

## Acceptance run

Prepare an agent-visible JSON file with brief, profile, and optional config keys.
brief contains goal, method guidance, lab, and output contract. profile is a list
of Evidence objects. Use a separate file for the clean Task 9 profile and one
designated dev-fault profile. Exclude truth tables and grader output.

With the API key available through the environment, run each case with a unique ID.

```sh
uv run python -m scripts.run_proposers --input clean-profile.json --run-id du9-clean
uv run python -m scripts.run_proposers --input dev-profile.json --run-id du9-dev
```

The command prints merged output and writes proposal events in runs/<run_id>/events.jsonl.
It exercises proposal generation only. The production caller supplies the real
memory backend. Inspect both outputs for at least two materially different
questions and verify question.opened, hypothesis.proposed, and hypothesis.merged.

Live acceptance passed on the published Task 9 matrix and a fixed-seed
duplicate-library dev case. See [the acceptance record](acceptance/DU-9.md)
for evidence, input provenance, and limitations. This verifies proposal generation.
