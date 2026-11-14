"""Scripted integration scenario for UI development. All measurements and scores are canned."""

from datetime import UTC, datetime

from core.config import RunConfig
from core.events import EventBus
from core.fakes import (
    FakeExecutor,
    FakeLab,
    FakeLLM,
    FakeMemory,
    FakePredictor,
    FakeSearch,
)
from core.schemas import (
    CriticFinding,
    Evidence,
    Experiment,
    Observation,
    OutcomeBin,
    Report,
)


async def record_demo(bus: EventBus, *, fail: bool = False) -> None:
    lab = FakeLab(bus.path.parent / "artifacts")
    memory, search, executor = FakeMemory(), FakeSearch(), FakeExecutor()
    predictor = FakePredictor()
    emit = bus.emit
    settings = RunConfig(lessons=True)
    config = settings.model_dump(mode="json")
    config["llm"]["cache_dir"] = settings.llm.cache_dir.as_posix()
    await emit(
        "run.started",
        {
            "lab": lab.name,
            "goal": lab.brief()["goal"],
            "config": config,
            "started_at": datetime.fromtimestamp(1_794_700_000, UTC).isoformat(),
        },
    )
    if fail:
        await emit(
            "run.error", {"message": "Simulated sensor disconnect", "where": "profile"}
        )
        return
    await emit("profile.done", {"evidence": await lab.profile()})
    for question in lab.questions:
        await emit("question.opened", {"question": question})
        options = lab.hypotheses[question.id]
        for h in options:
            await emit("hypothesis.proposed", {"hypothesis": h})
        await emit(
            "hypothesis.merged",
            {
                "question_id": question.id,
                "kept": options[0].id,
                "merged": [],
                "priors": {h.id: h.prior for h in options},
            },
        )

    sensor, repeat, check, manual = lab.experiments
    manual.bins = [
        OutcomeBin(
            id="aliases", description="Labels are equivalent", rule="same_unit == true"
        ),
        OutcomeBin(
            id="distinct", description="Labels differ", rule="same_unit == false"
        ),
    ]
    manual.likelihood = await predictor.predict(manual, lab.hypotheses["units"], "")
    for h in lab.hypotheses["units"]:
        if not h.is_null:
            manual.likelihood[h.id] = {"aliases": 0.02, "distinct": 0.98}
    for exp in lab.experiments:
        await emit("experiment.proposed", {"experiment": exp})

    async def run(
        exp: Experiment, score: float, other: Experiment | None = None
    ) -> Observation:
        def row(candidate: Experiment, bits: float) -> dict:
            return {
                "experiment_id": candidate.id,
                "kind": candidate.kind,
                "goal": candidate.goal,
                "eig_bits": bits,
                "est_cost_s": candidate.est_cost_s,
                "est_cost_usd": candidate.est_cost_usd,
                "score": bits / 1.1,
            }

        ranking = [row(exp, score)]
        if other is not None:
            ranking.append(row(other, score / 2))
        await emit(
            "experiment.scored",
            {"question_id": exp.question_ids[0], "ranking": ranking, "chosen": exp.id},
        )
        await emit(
            "experiment.started",
            {
                "experiment_id": exp.id,
                "kind": exp.kind,
                "question_ids": exp.question_ids,
            },
        )
        obs = await lab.run(exp)
        sandbox = await executor.open(lab.name, bus.run_id)
        try:
            output = await executor.exec(sandbox, "json", obs.model_dump_json())
        finally:
            await executor.close(sandbox)
        await emit(
            "experiment.log",
            {
                "experiment_id": exp.id,
                "stream": "stdout",
                "line": output["stdout"][:500],
            },
        )
        await emit("experiment.finished", {"observation": obs})
        await emit(
            "referee.verdict",
            {"experiment_id": exp.id, "valid": obs.valid, "note": obs.referee_note},
        )
        await emit(
            "budget.tick",
            {
                "cost_usd": 0,
                "llm_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "seconds": 0,
            },
        )
        return obs

    obs = await run(sensor, 0.8, repeat)
    linear, quadratic, null = lab.hypotheses["curve"]
    prior = {h.id: h.prior for h in (linear, quadratic, null)}
    posterior = {linear.id: 0.01, quadratic.id: 0.95, null.id: 0.04}
    await emit(
        "belief.updated",
        {
            "question_id": "curve",
            "experiment_id": sensor.id,
            "observed": obs.value,
            "prior": prior,
            "posterior": posterior,
        },
    )
    await emit(
        "model.curves",
        {
            "question_id": "curve",
            "x": [1, 2],
            "points": [{"x": 2, "y": obs.value}],
            "families": {
                h.id: {"mean": means, "sd": [sd, sd], "weight": posterior[h.id]}
                for h, means, sd in (
                    (linear, [1, 2], 0.1),
                    (quadratic, [1, 4], 0.1),
                    (null, [2, 3], 2),
                )
            },
        },
    )
    evidence = Evidence(
        id="reading",
        kind="observation",
        claim="Synthetic reading at x=2 was 4.",
        source_ref=sensor.id,
        author_role="experimenter",
    )
    await memory.write(evidence, bus.run_id, evidence.author_role)
    await emit(
        "memory.written",
        {
            "evidence_id": evidence.id,
            "author_role": evidence.author_role,
            "backend": "fake",
        },
    )
    rejection = {
        "question_id": "curve",
        "hypothesis_id": linear.id,
        "evidence_id": evidence.id,
        "reason": "Linear calibration predicts 2 at x=2.",
    }
    await emit("hypothesis.rejected", rejection)
    await emit(
        "question.settled",
        {
            "question_id": "curve",
            "hypothesis_id": quadratic.id,
            "posterior": posterior[quadratic.id],
        },
    )

    invalid = await run(check, 0.4, manual)
    await emit(
        "question.unresolved",
        {
            "question_id": "units",
            "reason": invalid.referee_note,
            "top": ["units-0", "units-1"],
        },
    )
    artifacts = await lab.deliver({})
    await emit("artifact.written", {"artifact": artifacts[0]})
    finding = CriticFinding(
        id="review-units",
        artifact="METHODS.md",
        severity="warn",
        rule="units",
        message="Verify the unit label in the manual.",
        reopens_question="units",
    )
    critic = FakeLLM({CriticFinding: finding.model_dump()})
    [review] = await critic.structured("critic", CriticFinding, [])
    await emit("critic.finding", {"finding": review})
    recalled = await memory.recall("reading", bus.run_id, "critic")
    await emit(
        "memory.recalled",
        {
            "reader_role": "critic",
            "query": "reading",
            "backend": "fake",
            "evidence_ids": [ev.id for ev in recalled],
            "authors": [ev.author_role for ev in recalled],
            "latency_ms": 0,
        },
    )
    results = await search.find(manual.query)
    checked = await run(manual, 0.2)
    await emit(
        "search.done",
        {
            "experiment_id": manual.id,
            "query": manual.query,
            "urls": [ev.url for ev in results],
            "blocked": [],
            "bin_id": checked.bin_id,
        },
    )
    labels = lab.hypotheses["units"]
    weights = {h.id: h.prior * manual.likelihood[h.id][checked.bin_id] for h in labels}
    total = sum(weights.values())
    for h in labels:
        h.posterior = weights[h.id] / total
    await emit(
        "belief.updated",
        {
            "question_id": "units",
            "experiment_id": manual.id,
            "observed": checked.bin_id,
            "prior": {h.id: h.prior for h in labels},
            "posterior": {h.id: h.posterior for h in labels},
        },
    )
    await emit(
        "surprise.raised",
        {
            "question_id": "units",
            "experiment_id": manual.id,
            "reason": "Unexpected aliases; the manual mandates millivolts for output regardless of input label.",
            "z_score": None,
            "null_mass": labels[-1].posterior,
        },
    )
    for h in labels:
        h.answer_key = "millivolts"
        await emit("hypothesis.proposed", {"hypothesis": h})
    await emit("question.robust", {"question_id": "units", "answer_key": "millivolts"})
    lesson = Evidence(
        id="lesson",
        kind="lesson",
        claim="Check unit aliases before comparing labels.",
        source_ref=results[0].id,
        author_role="critic",
    )
    await emit("lesson.written", {"lesson": lesson})
    report = Report(
        run_id=bus.run_id,
        artifacts=artifacts,
        methods_md=lab.methods,
        unresolved=[],
        rejected=[
            {
                "hypothesis": linear.id,
                "question": "curve",
                "evidence_id": evidence.id,
                "reason": rejection["reason"],
            }
        ],
        cost_usd=0,
        seconds=0,
    )
    await emit("run.finished", {"report": report, "status": "ok"})
