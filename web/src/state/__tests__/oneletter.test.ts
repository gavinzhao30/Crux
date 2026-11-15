import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { foldAll, foldEvent, initialState, type RunState } from '../fold';
import {
  activeSurprise,
  currentStripReads,
  isNullVariant,
  isOneLetterLab,
  isRecorded,
  pendingRequest,
  readCells,
  readMatches,
  readerProgress,
  readerUrls,
  sequenceMarks,
  variantName,
  verdictAuthor,
  verdictStanding,
} from '../selectors';
import { parseJsonl } from '../../data/sources';
import { DNA_EVENT_TYPES, type RunEvent } from '../../types';
import { HBB_REF, referenceBases } from '../../lib/hbb';

const FIXTURES = fileURLToPath(new URL('../../../public/fixtures/', import.meta.url));
const load = (name: string): RunEvent[] => parseJsonl(readFileSync(`${FIXTURES}${name}.jsonl`, 'utf8'));
// The engine's recorded fake-reader run (labs/oneletter, runs/demo/oneletter_fake.jsonl).
const oneletter = load('demo_oneletter');
const indexOf = (pred: (e: RunEvent) => boolean) => oneletter.findIndex(pred);
const upTo = (i: number) => foldAll(oneletter.slice(0, i + 1));

/** Contract fields per dna.* type (tools/ONE_LETTER_CONTRACT.md); the engine may add more. */
const CONTRACT: Record<string, string[]> = {
  'dna.request': ['eig_bits', 'reason', 'replicate', 'window'],
  'dna.read': ['bases', 'confidence', 'expected', 'experiment_id', 'mismatch_positions', 'window'],
  'dna.variant': ['amino_acid', 'codon', 'hgvs', 'hypothesis_id', 'legacy', 'name', 'rsid'],
  'dna.verdict': ['author', 'hypothesis_id', 'latency_ms', 'name', 'pdb_id', 'posterior', 'reads', 'residue', 'text'],
  'dna.lesson': ['added', 'lessons_total'],
};

let seq = 0;
function ev(type: string, payload: unknown, run_id = 'r-ol'): RunEvent {
  return { run_id, seq: seq++, ts: 1000 + seq, type, payload };
}
function started(config: Record<string, unknown> = {}): RunState {
  seq = 0;
  return foldEvent(
    initialState(),
    ev('run.started', { lab: 'oneletter', goal: 'g', config, started_at: '2026-11-14T20:00:00Z' }),
  );
}
const READ = {
  experiment_id: 'window-3',
  window: 3,
  bases: 'AAGGAGAAG',
  expected: { 'h-hba': 'GAGGAGAAG', 'h-hbs': 'GTGGAGAAG' },
  mismatch_positions: [0],
  confidence: [0.86, 0.95, 0.97, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
  closest: 'h-hba',
};

describe('demo_oneletter follows the contract', () => {
  it('seq from 0 without gaps, run.started first, one terminal event last', () => {
    expect(oneletter.length).toBeGreaterThan(40);
    oneletter.forEach((e, i) => {
      expect(e.seq).toBe(i);
      expect(e.run_id).toBe(oneletter[0]?.run_id);
      if (i > 0) expect(e.ts).toBeGreaterThanOrEqual(oneletter[i - 1]!.ts);
    });
    expect(oneletter[0]?.type).toBe('run.started');
    const terminal = oneletter.filter((e) => e.type === 'run.finished' || e.type === 'run.error');
    expect(terminal).toHaveLength(1);
    expect(oneletter.at(-1)).toBe(terminal[0]);
  });

  it('run.started names the lab and the reader URLs, and says the reader is fake', () => {
    const p = oneletter[0]!.payload as { lab: string; config: Record<string, unknown> };
    expect(p.lab).toBe('oneletter');
    expect(p.config.annotated_url).toBe(`${p.config.reader_url as string}/annotated_stream`);
    expect(p.config.fake_reader).toBe(true);
  });

  it('uses every dna.* type with at least the contract fields', () => {
    for (const t of DNA_EVENT_TYPES) {
      const of = oneletter.filter((e) => e.type === t);
      expect(of.length, t).toBeGreaterThan(0);
      for (const e of of) {
        const keys = Object.keys(e.payload as object);
        for (const k of CONTRACT[t]!) expect(keys, `${t} seq ${e.seq}`).toContain(k);
      }
    }
  });

  it('reads cite the reference: only window 3 is read, and it differs only at codon 6', () => {
    const reads = oneletter.filter((e) => e.type === 'dna.read').map((e) => e.payload as typeof READ);
    expect(reads.map((r) => r.window)).toEqual([3, 3, 3, 3]);
    expect(reads.map((r) => r.bases)).toEqual(['GTGGAGAAG', 'GTGGAGAAG', 'AAGGAGAAG', 'AAGGAGAAG']);
    for (const r of reads) {
      expect(r.bases.slice(3)).toBe(referenceBases(3).slice(3));
      expect(r.confidence).toHaveLength(9);
    }
    for (const e of oneletter.filter((x) => x.type === 'experiment.scored')) {
      const rows = (e.payload as { ranking: { experiment_id: string; eig_bits: number }[] }).ranking;
      for (const r of rows.filter((x) => /^window-[12](-|$)/.test(x.experiment_id))) expect(r.eig_bits).toBe(0);
    }
  });

  it('every posterior sums to one', () => {
    for (const e of oneletter.filter((x) => x.type === 'belief.updated')) {
      const post = (e.payload as { posterior: Record<string, number> }).posterior;
      expect(Object.values(post).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    }
  });
});

describe('folding demo_oneletter', () => {
  const s = foldAll(oneletter);

  it('folds without malformed events and settles on HbC', () => {
    expect(s.malformed).toBe(0);
    expect(s.status).toBe('finished');
    expect(s.eventCount).toBe(oneletter.length);
    expect(isOneLetterLab(s)).toBe(true);
    const q = s.questions['q-variant']!;
    expect(q.status).toBe('settled');
    expect(q.settled?.hypothesisId).toBe('h-hbc');
    expect(q.options['h-hbc']!.regenerated).toBe(true);
    expect(q.options['h-hbs']!.regenerated).toBe(false);
    expect(q.surprise?.resolved).toBe(true);
  });

  it('keeps the reads, the variants (latest wins), both verdicts and the lesson', () => {
    expect(s.dna.reads).toHaveLength(4);
    expect(s.dna.requests).toBe(4);
    expect(s.dna.variantOrder).toEqual(['h-hba', 'h-hbs', 'h0', 'h-hbc']);
    expect(s.dna.variants['h-hbs']).toMatchObject({ rsid: 'rs334', legacy: 'Glu6Val', codon: 'GTG' });
    expect(s.dna.variants['h-hbc']).toMatchObject({ rsid: 'rs33930165', legacy: 'Glu6Lys', aminoAcid: 'Lys' });
    expect(isNullVariant(s, 'h0')).toBe(true);
    expect(isNullVariant(s, 'h-hba')).toBe(false);
    expect(s.dna.verdicts.map((v) => [v.hypothesisId, v.pdbId, v.reads, v.strip])).toEqual([
      ['h-hbs', '2HBS', 2, 1],
      ['h-hbc', '1A3N', 2, 2],
    ]);
    expect(s.dna.verdict?.author).toBe('template');
    expect(s.dna.lesson).toMatchObject({ added: ['h-hbc'], lessonsTotal: 1 });
    expect(variantName(s, 'h-hbc')).toBe('HbC');
    expect(s.dna.reads.map((r) => r.closest)).toEqual(['h-hbs', 'h-hbs', 'h-hba', 'h-hba']);
  });

  it('a watch-mode reopen starts a new strip: settlement and highlights reset, history kept', () => {
    const reopen = indexOf((e) => e.type === 'question.opened' && (e.payload as { reopened?: boolean }).reopened === true);
    const before = upTo(reopen - 1);
    const after = upTo(reopen);
    expect(before.questions['q-variant']!.status).toBe('settled');
    expect(before.questions['q-variant']!.options['h-hbs']!.status).toBe('accepted');
    const q = after.questions['q-variant']!;
    expect(q.status).toBe('open');
    expect(q.settled).toBeNull();
    expect(q.options['h-hbs']!.status).toBe('active');
    expect(q.options['h-hbs']!.history.length).toBe(before.questions['q-variant']!.options['h-hbs']!.history.length);
    expect(after.dna.strip).toBe(2);
    expect(after.dna.stripSince).toBe(reopen);
    expect(after.dna.reopenReason).toMatch(/new sample/);
    expect(after.dna.reads).toHaveLength(2);
    expect(currentStripReads(after)).toEqual([]);
    expect(sequenceMarks(after)).toEqual({ windowsRead: [], mismatches: [], latest: null });
    expect(verdictStanding(after)).toBe('new-strip');
    // Re-proposed options start from their stated priors.
    expect(upTo(reopen + 7).questions['q-variant']!.options['h-hbs']!.posterior).toBe(0.45);
  });

  it('the swap raises a surprise with the mismatch at codon 6', () => {
    const at = indexOf((e) => e.type === 'surprise.raised');
    const mid = upTo(at);
    expect(activeSurprise(mid)?.reason).toMatch(/no hypothesis predicts/);
    expect(sequenceMarks(mid)).toMatchObject({ windowsRead: [3], mismatches: [18] });
    expect(HBB_REF[18]).toBe('G');
    expect(activeSurprise(s)).toBeNull();
    expect(verdictStanding(s)).toBe('current');
  });

  it('fresh priors without a merge apply to existing options', () => {
    const at = indexOf((e) => e.type === 'hypothesis.merged' && 'h-hbc' in ((e.payload as { priors: object }).priors ?? {}));
    const q = upTo(at).questions['q-variant']!;
    expect(q.options['h-hba']!.posterior).toBe(0.3);
    expect(q.options['h0']!.posterior).toBe(0.1);
    expect(q.options['h-hbs']!.status).toBe('active');
  });

  it('shows the engine progress line while a window is pending', () => {
    const log = indexOf((e) => e.type === 'experiment.log');
    expect(readerProgress(upTo(log))).toMatch(/camera sees no window card/);
    const read = indexOf((e) => e.type === 'dna.read');
    expect(readerProgress(upTo(read))).toBeNull();
  });

  it('no request is pending once the run has finished', () => {
    expect(pendingRequest(s)).toBeNull();
  });

  it('other labs leave the dna state empty', () => {
    for (const name of ['demo_categorical', 'demo_numeric', 'demo_error']) {
      const other = foldAll(load(name));
      expect(isOneLetterLab(other)).toBe(false);
      expect(other.dna).toEqual(initialState().dna);
    }
  });
});

describe('dna.* reducer', () => {
  it('a request stays pending until a read of that window arrives', () => {
    let s = started();
    s = foldEvent(s, ev('dna.request', { window: 3, reason: 'codon 6', eig_bits: 0.94, replicate: false, experiment_id: 'window-3', strip: 1 }));
    expect(pendingRequest(s)).toMatchObject({ window: 3, eigBits: 0.94, replicate: false, experimentId: 'window-3', strip: 1 });
    expect(s.dna.strip).toBe(1);
    s = foldEvent(s, ev('dna.read', { ...READ, window: 2, bases: 'CTGACTCCT', mismatch_positions: [] }));
    expect(pendingRequest(s)?.window).toBe(3);
    s = foldEvent(s, ev('dna.read', READ));
    expect(pendingRequest(s)).toBeNull();
    s = foldEvent(s, ev('dna.request', { window: 3, reason: 'replicate', eig_bits: 0.37, replicate: true }));
    expect(pendingRequest(s)).toMatchObject({ replicate: true, experimentId: null });
  });

  it('ignores malformed payloads without throwing', () => {
    let s = started();
    s = foldEvent(s, ev('dna.request', { window: 'three' }));
    s = foldEvent(s, ev('dna.read', { window: 3 }));
    s = foldEvent(s, ev('dna.read', { bases: 'GAG' }));
    s = foldEvent(s, ev('dna.variant', { name: 'no id' }));
    s = foldEvent(s, ev('dna.verdict', { text: 'no id' }));
    s = foldEvent(s, ev('dna.lesson', null));
    s = foldEvent(s, ev('dna.read', null));
    s = foldEvent(s, ev('hypothesis.merged', { question_id: 'q-x', kept: '', priors: { h1: 0.5 } }));
    expect(s.malformed).toBe(0);
    expect(s.dna.request).toBeNull();
    expect(s.dna.reads).toEqual([]);
    expect(s.dna.variants).toEqual({});
    expect(s.dna.verdict).toBeNull();
    expect(s.dna.lesson).toEqual({ seq: expect.any(Number), added: [], lessonsTotal: 0 });
    expect(s.questions['q-x']).toBeUndefined();
  });

  it('keeps only in-range, unique mismatch positions and upper-cases bases', () => {
    let s = started();
    s = foldEvent(s, ev('dna.read', { ...READ, bases: 'aaggagaag', mismatch_positions: [0, 0, 12, -1, 2.5, 'x'] }));
    const r = s.dna.reads[0]!;
    expect(r.bases).toBe('AAGGAGAAG');
    expect(r.mismatchPositions).toEqual([0]);
    expect(r.closest).toBe('h-hba');
  });

  it('read cells mark mismatches, reference differences and supporting hypotheses', () => {
    let s = started();
    s = foldEvent(s, ev('dna.read', READ));
    const cells = readCells(s.dna.reads[0]!);
    expect(cells).toHaveLength(9);
    expect(cells[0]).toMatchObject({ base: 'A', ref: 'G', mismatch: true, differsFromRef: true, supports: [] });
    expect(cells[1]).toMatchObject({ base: 'A', ref: 'A', mismatch: false, differsFromRef: false, supports: ['h-hba'] });
    expect(readMatches(s.dna.reads[0]!)).toEqual([]);
    s = foldEvent(s, ev('dna.read', { ...READ, bases: 'GTGGAGAAG', mismatch_positions: [] }));
    expect(readMatches(s.dna.reads[1]!)).toEqual(['h-hbs']);
  });

  it('sequence marks map window positions onto the HBB reference', () => {
    let s = started();
    s = foldEvent(s, ev('dna.read', { ...READ, window: 1, bases: 'ATGGTGCAT', mismatch_positions: [] }));
    s = foldEvent(s, ev('dna.read', READ));
    const m = sequenceMarks(s);
    expect(m.windowsRead).toEqual([1, 3]);
    expect(m.mismatches).toEqual([18]);
    expect(m.latest?.window).toBe(3);
  });

  it('variants upsert (latest wins) and verdicts keep a history', () => {
    let s = started();
    s = foldEvent(s, ev('dna.variant', { hypothesis_id: 'h-hbs', name: 'HbS', hgvs: 'c.20A>T', legacy: 'Glu6Val', rsid: 'rs334', codon: 'gtg', amino_acid: 'Val' }));
    s = foldEvent(s, ev('dna.variant', { hypothesis_id: 'h-hbs', name: 'HbS (sickle)', hgvs: 'NM_000518.5:c.20A>T', legacy: 'Glu6Val', rsid: 'rs334', codon: 'GTG', amino_acid: 'Val' }));
    expect(s.dna.variantOrder).toEqual(['h-hbs']);
    expect(s.dna.variants['h-hbs']).toMatchObject({ name: 'HbS (sickle)', hgvs: 'NM_000518.5:c.20A>T', codon: 'GTG' });
    const verdict = { hypothesis_id: 'h-hbs', name: 'HbS', posterior: 0.97, reads: 2, text: 'Sickle.', author: 'jetson/qwen2.5:3b', latency_ms: 1840.5, pdb_id: '2hbs', residue: 6 };
    s = foldEvent(s, ev('dna.verdict', verdict));
    s = foldEvent(s, ev('dna.verdict', { ...verdict, author: 'template', latency_ms: null }));
    expect(s.dna.verdicts).toHaveLength(2);
    expect(s.dna.verdicts[0]).toMatchObject({ pdbId: '2HBS', latencyMs: 1840.5, author: 'jetson/qwen2.5:3b', legacy: 'Glu6Val' });
    expect(s.dna.verdict).toMatchObject({ author: 'template', latencyMs: null });
  });

  it('a surprise after the verdict marks it as reopened by surprise', () => {
    let s = started();
    s = foldEvent(s, ev('question.opened', { question: { id: 'q-variant', text: 'q', kind: 'categorical', impact: 1, status: 'open', raised_by: 'x' } }));
    s = foldEvent(s, ev('dna.verdict', { hypothesis_id: 'h-hbs', name: 'HbS', posterior: 0.97, reads: 1, text: '', author: 'template', latency_ms: null, pdb_id: '2HBS', residue: 6 }));
    expect(verdictStanding(s)).toBe('current');
    s = foldEvent(s, ev('surprise.raised', { question_id: 'q-variant', experiment_id: 'window-3', reason: 'odd', z_score: null, null_mass: 0.4 }));
    expect(verdictStanding(s)).toBe('surprise');
  });

  it('dna events are dropped on replayed seqs like any other event', () => {
    let s = started();
    const e = ev('dna.read', READ);
    s = foldEvent(s, e);
    s = foldEvent(s, e);
    expect(s.dna.reads).toHaveLength(1);
  });
});

describe('One Letter selectors', () => {
  it('names the verdict author', () => {
    expect(verdictAuthor('jetson/qwen2.5:3b')).toEqual({ onJetson: true, label: 'written on the Jetson · qwen2.5:3b' });
    expect(verdictAuthor('template')).toEqual({ onJetson: false, label: 'template' });
    expect(verdictAuthor('')).toEqual({ onJetson: false, label: 'template' });
  });

  it('is recorded for fixtures, API replays and the fake reader only', () => {
    const live = started({ reader_url: 'http://jetson.local:8770' });
    const fake = started({ fake_reader: true });
    expect(isRecorded(live, 'live')).toBe(false);
    expect(isRecorded(live, 'local')).toBe(true);
    expect(isRecorded(live, 'api-replay')).toBe(true);
    expect(isRecorded(fake, 'live')).toBe(true);
    expect(isRecorded(started({ fake_reader: 'true' }), 'live')).toBe(false);
  });

  it('derives reader URLs and rejects anything but http(s)', () => {
    expect(readerUrls(started({ reader_url: 'http://10.0.0.5:8770/', annotated_url: 'http://10.0.0.5:8770/annotated_stream' }))).toEqual({
      reader: 'http://10.0.0.5:8770',
      annotated: 'http://10.0.0.5:8770/annotated_stream',
      strip: 'http://10.0.0.5:8770/strip',
    });
    expect(readerUrls(started({ reader_url: 'http://10.0.0.5:8770' })).annotated).toBe('http://10.0.0.5:8770/annotated_stream');
    expect(readerUrls(started({ reader_url: 'javascript:alert(1)', annotated_url: 'file:///x' }))).toEqual({
      reader: null,
      annotated: null,
      strip: null,
    });
  });

  it('the lab check is exact', () => {
    for (const lab of ['OneLetter', 'oneletter2', '', 'bench']) {
      seq = 0;
      const s = foldEvent(initialState(), ev('run.started', { lab, goal: '', config: {} }));
      expect(isOneLetterLab(s), lab).toBe(false);
    }
  });
});
