import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { foldAll, foldEvent, initialState, type RunState } from '../fold';
import { artifactChecks, criticSummary, evidenceText, leaderId, visibleOptions } from '../selectors';
import { parseJsonl, replayDelayMs } from '../../data/sources';
import { EVENT_TYPES, type RunEvent } from '../../types';

const FIXTURES = fileURLToPath(new URL('../../../public/fixtures/', import.meta.url));

function load(name: string): RunEvent[] {
  return parseJsonl(readFileSync(`${FIXTURES}${name}.jsonl`, 'utf8'));
}

const categorical = load('demo_categorical');
const numeric = load('demo_numeric');
const failing = load('demo_error');

function lastOf<T extends RunEvent['type']>(events: RunEvent[], type: T) {
  const found = [...events].reverse().find((e) => e.type === type);
  if (!found) throw new Error(`no ${type}`);
  return found.payload as Record<string, unknown>;
}

describe('fixtures follow the event contract', () => {
  for (const [name, events] of [
    ['demo_categorical', categorical],
    ['demo_numeric', numeric],
    ['demo_error', failing],
  ] as const) {
    it(`${name}: seq from 0 without gaps, run.started first, one terminal event last`, () => {
      expect(events.length).toBeGreaterThan(10);
      events.forEach((e, i) => {
        expect(e.seq).toBe(i);
        expect(e.run_id).toBe(events[0]?.run_id);
        if (i > 0) expect(e.ts).toBeGreaterThanOrEqual(events[i - 1]!.ts);
      });
      expect(events[0]?.type).toBe('run.started');
      const terminal = events.filter((e) => e.type === 'run.finished' || e.type === 'run.error');
      expect(terminal).toHaveLength(1);
      expect(events.at(-1)).toBe(terminal[0]);
    });
  }

  it('together they exercise every event type', () => {
    const seen = new Set([...categorical, ...numeric, ...failing].map((e) => e.type));
    for (const t of EVENT_TYPES) expect(seen, t).toContain(t);
    const cat = new Set(categorical.map((e) => e.type));
    const missing = EVENT_TYPES.filter((t) => !cat.has(t));
    expect(missing.sort()).toEqual(['model.curves', 'run.error', 'surprise.raised']);
  });

  it('categorical sizes match the brief', () => {
    expect(categorical.length).toBeGreaterThanOrEqual(100);
    const span = categorical.at(-1)!.ts - categorical[0]!.ts;
    expect(span).toBeGreaterThan(75);
    expect(span).toBeLessThan(110);
    expect(numeric.filter((e) => e.type === 'model.curves').length).toBeGreaterThanOrEqual(8);
  });

  it('categorical beliefs follow Bayes rule from the proposed likelihoods', () => {
    const lik: Record<string, Record<string, Record<string, number>>> = {};
    for (const e of categorical) {
      if (e.type === 'experiment.proposed') {
        const x = (e.payload as { experiment: { id: string; likelihood: Record<string, Record<string, number>> } })
          .experiment;
        lik[x.id] = x.likelihood;
      }
      if (e.type === 'belief.updated') {
        const p = e.payload as {
          experiment_id: string;
          observed: string;
          prior: Record<string, number>;
          posterior: Record<string, number>;
        };
        const L = lik[p.experiment_id];
        expect(L, p.experiment_id).toBeDefined();
        const un = Object.fromEntries(
          Object.entries(p.prior).map(([h, pr]) => [h, pr * (L![h]?.[p.observed] ?? Number.NaN)]),
        );
        const z = Object.values(un).reduce((a, b) => a + b, 0);
        for (const [h, v] of Object.entries(un)) expect(p.posterior[h]).toBeCloseTo(v / z, 3);
      }
      if (e.type === 'experiment.scored') {
        const p = e.payload as { ranking: { experiment_id: string; score: number }[]; chosen: string };
        const scores = p.ranking.map((r) => r.score);
        expect([...scores].sort((a, b) => b - a)).toEqual(scores);
        expect(p.chosen).toBe(p.ranking[0]?.experiment_id);
      }
    }
  });
});

describe('folding demo_categorical', () => {
  const s = foldAll(categorical);

  it('finishes cleanly with no malformed events', () => {
    expect(s.status).toBe('finished');
    expect(s.finishStatus).toBe('ok');
    expect(s.malformed).toBe(0);
    expect(s.eventCount).toBe(categorical.length);
    expect(s.lastSeq).toBe(categorical.length - 1);
    expect(s.lab).toBe('bixbench3');
  });

  it('settles the label question on "inverted"', () => {
    const q1 = s.questions.q1!;
    expect(q1.status).toBe('settled');
    expect(q1.settled?.hypothesisId).toBe('h2');
    expect(q1.options.h2!.posterior).toBeGreaterThan(0.9);
    expect(q1.options.h2!.status).toBe('accepted');
    expect(leaderId(q1)).toBe('h2');
    // The merged duplicate is folded into h2 and hidden.
    expect(q1.options.h3!.status).toBe('merged');
    expect(q1.options.h2!.mergedFrom).toEqual(['h3']);
    expect(visibleOptions(q1).map((o) => o.id)).toEqual(['h1', 'h2', 'h0']);
    expect(visibleOptions(q1).at(-1)?.isNull).toBe(true);
  });

  it('closes the fit question as robust and the filter question as unresolved', () => {
    expect(s.questions.q2!.status).toBe('robust');
    expect(s.questions.q2!.robust?.answerKey).toBe('deg_t98g');
    expect(s.questions.q3!.status).toBe('unresolved');
    expect(s.questions.q3!.unresolved?.top).toEqual(['h2', 'h1']);
  });

  it('records the rejected option with its evidence', () => {
    expect(s.rejected).toHaveLength(1);
    expect(s.rejected[0]).toMatchObject({ questionId: 'q1', hypothesisId: 'h1', evidenceId: 'ev-obs-sample_corr' });
    expect(s.questions.q1!.options.h1!.status).toBe('rejected');
    expect(evidenceText(s, 'ev-obs-sample_corr')).toMatch(/cluster with the knockouts/);
  });

  it('shows the critic block repaired and the warning still open', () => {
    expect(criticSummary(s)).toEqual({ 'de_results_T98G.csv': 'pass', 'methods.md': 'warn' });
    const de = artifactChecks(s).find((c) => c.artifact.name === 'de_results_T98G.csv')!;
    expect(de.rules).toHaveLength(1);
    expect(de.rules[0]).toMatchObject({ resolved: true });
    expect(de.rules[0]!.finding.rule).toBe('required_columns');
    expect(de.artifact.versions).toBe(2);
  });

  it('a re-emit with the same sha256 is the same version', () => {
    const art = { name: 'x.csv', path: 'x.csv', sha256: 'abc', contract_ok: false, findings: [] };
    const ev = (seq: number, artifact: object): RunEvent =>
      ({ run_id: 'r', seq, ts: seq, type: 'artifact.written', payload: { artifact } }) as RunEvent;
    let st = foldEvent(initialState(), ev(0, art));
    st = foldEvent(st, ev(1, { ...art, contract_ok: true }));
    expect(st.artifacts['x.csv']!.versions).toBe(1);
    expect(st.artifacts['x.csv']!.contract_ok).toBe(true);
    st = foldEvent(st, ev(2, { ...art, sha256: 'def' }));
    expect(st.artifacts['x.csv']!.versions).toBe(2);
  });

  it('critic status was "block" before the repair', () => {
    const idx = categorical.findIndex(
      (e) => e.type === 'artifact.written' && (e.payload as { artifact: { name: string; contract_ok: boolean } }).artifact.contract_ok && (e.payload as { artifact: { name: string } }).artifact.name === 'de_results_T98G.csv',
    );
    const before = foldAll(categorical.slice(0, idx));
    expect(criticSummary(before)['de_results_T98G.csv']).toBe('block');
  });

  it('ends with the recorded cost and the experiment trail', () => {
    const tick = lastOf(categorical, 'budget.tick');
    const report = (lastOf(categorical, 'run.finished').report ?? {}) as { cost_usd: number };
    expect(s.budget?.costUsd).toBeCloseTo(tick.cost_usd as number, 6);
    expect(s.budget?.costUsd).toBeCloseTo(report.cost_usd, 6);
    expect(s.budget?.llmCalls).toBe(tick.llm_calls);
    expect(s.completed.map((c) => c.experimentId)).toEqual([
      'rest_targets',
      'sample_corr',
      'docs_deseq2',
      'multiverse_fit',
    ]);
    expect(s.completed.every((c) => c.valid === true)).toBe(true);
    expect(s.rankings.q2!.phase).toBe('done');
    expect(s.rankings.q2!.observed).toBe('concordant');
  });

  it('feeds memory recalls with their authors and search results with blocked URLs', () => {
    const critic = s.feed.find((f) => f.kind === 'recall' && f.reader === 'critic');
    expect(critic).toMatchObject({ authors: ['experimenter', 'experimenter'] });
    const search = s.feed.find((f) => f.kind === 'search');
    expect(search && search.kind === 'search' && search.blocked.length).toBe(1);
    expect(s.console.some((l) => l.kind === 'stderr')).toBe(true);
    expect(s.console.filter((l) => l.kind === 'verdict')).toHaveLength(4);
  });
});

describe('folding demo_numeric', () => {
  const s = foldAll(numeric);

  it('re-settles on the regenerated option after the surprise', () => {
    const q = s.questions.q1!;
    expect(s.status).toBe('finished');
    expect(s.malformed).toBe(0);
    expect(q.status).toBe('settled');
    expect(q.settled?.hypothesisId).toBe('h5');
    expect(q.options.h5!.regenerated).toBe(true);
    expect(q.options.h2!.regenerated).toBe(false);
    expect(q.surprise?.resolved).toBe(true);
    expect(s.surprises).toBe(1);
  });

  it('settled the power law at 95% before the tint', () => {
    const check = numeric.findIndex(
      (e) => e.type === 'experiment.started' && (e.payload as { experiment_id: string }).experiment_id === 'x=128#check',
    );
    const idx = numeric.findIndex((e) => e.type === 'surprise.raised');
    const before = foldAll(numeric.slice(0, check));
    expect(before.questions.q1!.settled?.hypothesisId).toBe('h2');
    expect(before.questions.q1!.options.h2!.posterior).toBeGreaterThanOrEqual(0.95);
    const at = foldAll(numeric.slice(0, idx + 1));
    expect(at.questions.q1!.status).toBe('open');
    expect(at.questions.q1!.surprise?.resolved).toBe(false);
  });

  it('keeps curves, points and a newest-first rejected ledger', () => {
    const c = s.curves.q1!;
    expect(Object.keys(c.families).sort()).toEqual(['h0', 'h1', 'h2', 'h3', 'h4', 'h5']);
    expect(c.points).toHaveLength(10);
    expect(c.families.h2!.mean).toHaveLength(c.x.length);
    expect(s.rejected.map((r) => r.hypothesisId)).toEqual(['h3', 'h1']);
  });
});

describe('folding demo_error', () => {
  it('ends in the error state with the message', () => {
    const s = foldAll(failing);
    expect(s.status).toBe('error');
    expect(s.error?.message).toMatch(/out of memory/);
    expect(s.console.at(-1)?.kind).toBe('error');
    // Nothing keeps pulsing as "running" after the run died.
    expect(s.rankings.q1?.phase).toBe('aborted');
    expect(Object.values(s.experiments).some((e) => e.status === 'running')).toBe(false);
  });
});

describe('robustness', () => {
  it('ignores unknown event types', () => {
    const before = foldAll(categorical.slice(0, 40));
    const after = foldEvent(before, {
      run_id: before.runId,
      seq: 40,
      ts: 0,
      type: 'agent.daydreamed',
      payload: { anything: true },
    });
    expect({ ...after, lastSeq: before.lastSeq }).toEqual(before);
    expect(after.lastSeq).toBe(40);
  });

  it('ignores unknown payload fields', () => {
    const plain = foldAll(categorical);
    const noisy = foldAll(
      categorical.map((e) => ({ ...e, extra: 1, payload: { ...(e.payload as object), future_field: [1, 2] } })),
    );
    expect(noisy).toEqual(plain);
  });

  it('never throws on malformed input', () => {
    const junk: unknown[] = [
      null,
      42,
      'text',
      {},
      { type: 'belief.updated', seq: 0, payload: null },
      { type: 'question.opened', seq: 1, payload: { question: 5 } },
      { type: 'hypothesis.proposed', seq: 2, payload: { hypothesis: { id: 'h1' } } },
      { type: 'belief.updated', seq: 3, payload: { question_id: 'qx', posterior: { h1: 'high', h2: 0.4 } } },
      { type: 'model.curves', seq: 4, payload: { question_id: 'qx', x: 'nope', families: { a: null }, points: [{}] } },
      { type: 'experiment.scored', seq: 5, payload: { question_id: 'qx', ranking: [null, 3] } },
      { type: 'critic.finding', seq: 6, payload: { finding: { artifact: 'a.csv' } } },
      { type: 'run.finished', seq: 7, payload: { report: 'oops' } },
    ];
    let s: RunState = initialState();
    expect(() => {
      for (const j of junk) s = foldEvent(s, j);
    }).not.toThrow();
    expect(s.questions.qx?.options.h2?.posterior).toBe(0.4);
    expect(s.status).toBe('finished');
  });

  it('drops duplicates after a reconnect', () => {
    const once = foldAll(categorical);
    const twice = foldAll([...categorical.slice(0, 60), ...categorical.slice(30)]);
    expect(twice).toEqual(once);
  });

  it('is pure', () => {
    const base = foldAll(categorical.slice(0, 50));
    const snapshot = structuredClone(base);
    foldAll(categorical.slice(50), base);
    expect(base).toEqual(snapshot);
  });
});

describe('local replay timing', () => {
  it('uses recorded gaps, capped at 2 s, divided by speed', () => {
    const ev = (ts: number, seq: number): RunEvent => ({ run_id: 'r', seq, ts, type: 'budget.tick', payload: {} });
    const events = [ev(0, 0), ev(0.5, 1), ev(10, 2)];
    expect(replayDelayMs(events, 0, 1)).toBe(0);
    expect(replayDelayMs(events, 1, 1)).toBe(500);
    expect(replayDelayMs(events, 2, 1)).toBe(2000);
    expect(replayDelayMs(events, 2, 4)).toBe(500);
  });

  it('skips a torn last line', () => {
    const text = `${JSON.stringify(categorical[0])}\n${JSON.stringify(categorical[1])}\n{"run_id": "r", "seq": 2, "ty`;
    expect(parseJsonl(text)).toHaveLength(2);
  });
});

describe('engine event files (runs/demo, written by core/orchestrator.py)', () => {
  const ENGINE = fileURLToPath(new URL('../../../../runs/demo/', import.meta.url));
  const engine = parseJsonl(readFileSync(`${ENGINE}fake_categorical.jsonl`, 'utf8'));

  it('marks the repaired artifact as passing once the critic re-emits its verdict', () => {
    const s = foldAll(engine);
    expect(criticSummary(s)).toEqual({ 'de_table.csv': 'pass', 'qc_summary.csv': 'pass' });
    expect(s.artifacts['de_table.csv']!.versions).toBe(2);
  });

  it('shows the block between the first verdict and the repair', () => {
    const repaired = engine.findIndex(
      (e, i) =>
        i > 0 &&
        e.type === 'artifact.written' &&
        (e.payload as { artifact: { name: string; contract_ok: boolean } }).artifact.name ===
          'de_table.csv' &&
        (e.payload as { artifact: { contract_ok: boolean } }).artifact.contract_ok,
    );
    const before = foldAll(engine.slice(0, repaired));
    expect(criticSummary(before)['de_table.csv']).toBe('block');
  });
});
