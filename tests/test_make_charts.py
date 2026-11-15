import json

from eval import make_charts as mc


def _results(tmp_path, name, with_baseline=True):
    runs = []
    for arm, base in (("swarm", 0.8), ("baseline", 0.4)):
        if arm == "baseline" and not with_baseline:
            continue
        for i in (1, 2):
            runs.append({
                "arm": arm, "run_id": f"{arm}-{i}", "status": "ok",
                "mean": base + 0.02 * i,
                "artifacts": {"a_degs": base, "b_degs": base + 0.05 * i},
            })
    if with_baseline:
        runs.append({"arm": "baseline", "run_id": "bad", "status": "no record",
                     "mean": None, "artifacts": {}})
    p = tmp_path / name / "results.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return p


def _calibration(tmp_path):
    methods = {
        m: {s: {"stated": {"log_loss": v + (0.05 if s == "shift" else 0)}}
            for s in ("test", "shift")}
        for m, v in (("library", 0.55), ("sampled", 0.6), ("base", 0.6), ("finetuned", 0.47))
    }
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"methods": methods}), encoding="utf-8")
    return p


def test_all_charts_written(tmp_path):
    out = tmp_path / "figs"
    made = mc.build([_results(tmp_path, "c1"), _results(tmp_path, "c2")],
                    _calibration(tmp_path), out)
    names = {p.name for p in made}
    for stem in ("task9_swarm_vs_baseline", "task9_swarm_vs_baseline_per_case",
                 "fault_detection_real_data", "airway_label_swap", "predictor_calibration"):
        for ext in ("png", "svg"):
            assert f"{stem}.{ext}" in names
            assert (out / f"{stem}.{ext}").stat().st_size > 0


def test_swarm_only_results_work(tmp_path):
    out = tmp_path / "figs"
    mc.build([_results(tmp_path, "c1", with_baseline=False)], _calibration(tmp_path), out)
    assert (out / "task9_swarm_vs_baseline.png").is_file()


def test_missing_inputs_are_skipped(tmp_path, capsys):
    out = tmp_path / "figs"
    made = mc.build([tmp_path / "nope.json"], tmp_path / "nope_cal.json", out)
    printed = capsys.readouterr().out
    assert "skipped task9_swarm_vs_baseline" in printed
    assert "skipped predictor_calibration" in printed
    assert not (out / "task9_swarm_vs_baseline.png").exists()
    assert not (out / "predictor_calibration.png").exists()
    assert (out / "fault_detection_real_data.png").is_file()
    assert (out / "airway_label_swap.png").is_file()
    assert len(made) == 4


def test_hardcoded_totals():
    cells = [e for v in mc.FAULT_DETECTION.values() for e in v if e]
    assert (sum(c for c, _ in cells), sum(p for _, p in cells)) == (26, 30)
