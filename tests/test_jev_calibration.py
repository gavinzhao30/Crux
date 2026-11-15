import asyncio
import json

import httpx

from agents.jev_predictor import JevPredictor, discover_model
from eval.calibrate_task9 import eligible_records, evaluate, predict_rows, stated_rows
from eval.predictor_data.dataset import library_index


def test_identical_prediction_inputs_share_one_draw():
    class Predictor:
        calls = 0

        async def predict(self, exp, hypotheses, context):
            self.calls += 1
            return {
                h.id: {b.id: 1 / len(exp.bins) for b in exp.bins} for h in hypotheses
            }

    predictor = Predictor()
    record = {"diagnostic": "labels_swap_search", "profile": "same profile"}
    inputs = [dict(record, case_id=case) for case in ("case-01", "case-02")]
    rows = asyncio.run(predict_rows(predictor, inputs, concurrency=2))
    assert predictor.calls == 1
    assert list(rows) == ["case-01|labels_swap_search", "case-02|labels_swap_search"]
    assert len({json.dumps(row) for row in rows.values()}) == 1


def test_jev_batches_options_and_caches_response(tmp_path):
    requests = []
    entry = library_index()["labels_swap_search"]
    criteria = {b.id for b in entry.experiment.bins}
    hypotheses = list(entry.options.values())

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"models": [{"name": "jev-latest"}]})
        body = json.loads(request.content)
        assert len(body["questions"]) == len(hypotheses)
        for question in body["questions"].values():
            assert question["type"] == "choice"
            assert set(question["criteria"]) == criteria
        return httpx.Response(
            200,
            json={
                "answers": {
                    key: {
                        "probabilities": {
                            bin_id: 1 / len(criteria) for bin_id in criteria
                        }
                    }
                    for key in body["questions"]
                }
            },
        )

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = await discover_model(client)
            predictor = JevPredictor(client, model, tmp_path)
            rows = await predictor.predict(entry.experiment, hypotheses, "same profile")
            assert (
                await predictor.predict(entry.experiment, hypotheses, "same profile")
                == rows
            )
            assert set(rows) == {h.id for h in hypotheses}

    asyncio.run(check())
    assert [r.method for r in requests] == ["GET", "POST"]


def test_calibration_compares_the_same_observations_and_requires_jev_coverage():
    records = []
    for name, entry in library_index().items():
        records.append(
            {
                "case_id": "case-01",
                "split": "test",
                "diagnostic": name,
                "answer_key": entry.null_key,
                "observed_bin": entry.experiment.bins[0].id,
            }
        )
    eligible = eligible_records(records)
    stated = stated_rows(eligible)
    incomplete = evaluate(records, {"stated": stated, "sampled": stated}, {})
    assert incomplete["heldout_log_loss_winner"] is None
    report = evaluate(records, {"stated": stated, "sampled": stated, "jev": stated}, {})
    test = report["splits"]["test"]
    assert test["executed"] == 12
    assert test["eligible"] == test["matched"] == len(eligible)
    assert all(
        m["n"] == len(eligible) and m["ranking_groups"] == 2
        for m in test["methods"].values()
    )
    assert all(m["changed_vs_stated"] == 0 for m in test["methods"].values())
    assert report["jev_beats_both"] is False
