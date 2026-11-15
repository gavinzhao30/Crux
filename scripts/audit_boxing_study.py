"""Audit the six-arm DU-41 primary dataset and rebuild its reports, without API calls."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from core.replay import replay_check
from eval.boxing_study import ARMS, export_results, fingerprint
from labs.boxing.location_finding import LocationFinding

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path, nargs='?', default=Path('eval/results/boxing/main-20261114'))
root = parser.parse_args().directory
protocol = json.loads((root / 'protocol.json').read_text())
records = [json.loads(p.read_text()) for p in sorted(root.glob('*/episode.json'))]
expected = {(arm, seed) for arm in ARMS for seed in protocol['seeds']}
assert {(r['arm'], r['seed']) for r in records} == expected
assert len(records) == len(expected) == 120
rows = [row for r in records for row in r['rows']]
assert len(rows) == 360
assert len({(r['arm'], r['seed'], r['budget']) for r in rows}) == 360
for r in records:
    env = LocationFinding(r['seed'])
    assert r['sources_sha256'] == fingerprint(env._sources)
    assert r['test_points_sha256'] == fingerprint(env.evaluation_points())
    assert sorted(x['budget'] for x in r['rows']) == [5, 10, 20]
    assert len(r['history']) <= 20
    assert all(math.isfinite(row['mse']) and row['mse'] >= 0 for row in r['rows'])
    assert all(row['mse'] == 1e12 for row in r['rows'] if row['status'] == 'failed')
    for point, observed in r['history']:
        assert env.observe(point) == observed
    if r['arm'].startswith('crux_'):
        p = json.loads((root / f"proposals-{r['seed']}.json").read_text())
        assert r['paired_proposal_usage'] == p['usage']
        events_path = root / f"{root.name}-{r['arm']}-{r['seed']:02d}" / 'events.jsonl'
        if events_path.exists():
            events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
            emitted = [e['payload']['hypothesis'] for e in events if e['type'] == 'hypothesis.proposed']
            if emitted:
                assert [h['model_code'] for h in emitted] == [f['code'] for f in p['families']]
            else:
                assert all(row['status'] == 'failed' for row in r['rows'])

costs = json.loads((root / 'costs.json').read_text())
usage = costs['actual_wrapper_usage']
proposals = [json.loads(p.read_text()) for p in root.glob('proposals-*.json')]
assert len(proposals) == 20
for key in ['calls', 'input_tokens', 'output_tokens', 'cost_usd']:
    total = sum(r['usage'][key] for r in records) + sum(p['usage'][key] for p in proposals)
    assert math.isclose(usage[key], total, abs_tol=1e-9), (key, total, usage[key])
assert usage['cost_usd'] <= protocol['accounted_spend_cap_usd']
export_results(records, usage, root)
summary = json.loads((root / 'summary.json').read_text())
assert len(summary) == 18 and all(s['n'] == 20 for s in summary)
assert len(list(csv.DictReader((root / 'costs.csv').open()))) == 120
replays = {}
for arm in ['crux_random', 'crux_eig']:
    folder = root / f'{root.name}-{arm}-100'
    check = replay_check(folder)
    replays[arm] = {'ok': check.ok, 'details': str(check)}
    assert check.ok, check
audit = {'episodes': len(records), 'checkpoints': len(rows),
         'paired_seed_count': 20, 'case_hashes_and_observation_replay': 'pass',
         'hypothesis_sources_paired': 'pass', 'cost_reconciliation': 'pass',
         'replays': replays, 'actual_wrapper_usage': usage,
         'failures_by_arm': {arm: sum(x['status'] != 'ok' for x in rows if x['arm'] == arm) for arm in ARMS},
         'failure_reasons': dict(Counter(x['failure'] for x in rows if x['status'] != 'ok'))}
(root / 'audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
