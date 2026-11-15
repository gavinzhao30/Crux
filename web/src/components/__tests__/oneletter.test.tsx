import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { foldAll, foldEvent, initialState, type RunState } from '../../state/fold';
import { parseJsonl } from '../../data/sources';
import type { SourceInfo } from '../../data/useRunStream';
import type { RunEvent } from '../../types';
import { Header } from '../Header';
import { RunView, layoutFor } from '../RunView';
import { seqvizProps } from '../oneletter/SequencePanel';
import { residueSelection } from '../oneletter/StructurePanel';
import { RecordedBadge } from '../oneletter/bits';

const FIXTURES = fileURLToPath(new URL('../../../public/fixtures/', import.meta.url));
const load = (name: string): RunEvent[] => parseJsonl(readFileSync(`${FIXTURES}${name}.jsonl`, 'utf8'));
const oneletter = load('demo_oneletter');

const local: SourceInfo = { kind: 'local', name: 'demo_oneletter', speed: 1, status: { state: 'playing' } };
const live: SourceInfo = { kind: 'live', name: 'r-1', speed: 1, status: { state: 'open' } };

const render = (s: RunState, source: SourceInfo = local) => renderToStaticMarkup(<RunView state={s} source={source} />);

function liveRun(config: Record<string, unknown>): RunState {
  return foldEvent(initialState(), {
    run_id: 'r-1',
    seq: 0,
    ts: 1,
    type: 'run.started',
    payload: { lab: 'oneletter', goal: 'g', config },
  });
}

describe('layout choice', () => {
  it('uses the One Letter layout only for lab "oneletter"', () => {
    expect(layoutFor(initialState())).toBe('default');
    expect(layoutFor(foldAll(oneletter))).toBe('oneletter');
    for (const name of ['demo_categorical', 'demo_numeric', 'demo_error']) {
      const s = foldAll(load(name));
      expect(layoutFor(s), name).toBe('default');
      const html = render(s);
      expect(html).toContain('data-layout="default"');
      expect(html).not.toContain('data-layout="oneletter"');
    }
    const html = render(foldAll(oneletter));
    expect(html).toContain('data-layout="oneletter"');
    expect(html).not.toContain('data-layout="default"');
  });
});

describe('One Letter view', () => {
  it('renders after every event of the fixture', () => {
    let s = initialState();
    for (const ev of oneletter) {
      s = foldEvent(s, ev);
      expect(() => render(s), `after seq ${ev.seq} (${ev.type})`).not.toThrow();
    }
    const html = render(s);
    expect(html).toContain('HbC');
    expect(html).toContain('rs33930165');
    expect(html).toContain('NM_000518.5:c.19G&gt;A (p.Glu7Lys)');
    expect(html).toContain('data-testid="verdict-author"');
    expect(html).toContain('>template<');
    expect(html).toContain('Recorded · fake reader');
    // The fake reader's port is dead: no strip link, no camera.
    expect(html).not.toContain('data-testid="strip-link"');
    expect(html).toContain('data-testid="lesson-toast"');
    expect(html).toContain('Which HBB variant is on this strip?');
    expect(html).toContain('strip 2');
    // A recorded run never shows the live camera.
    expect(html).not.toContain('<img');
  });

  it('prompts for the requested window until it is read', () => {
    const req = oneletter.findIndex((e) => e.type === 'dna.request');
    const before = render(foldAll(oneletter.slice(0, req + 1)));
    expect(before).toContain('data-testid="window-prompt"');
    expect(before).toContain('Show window 3');
    expect(before).toContain('0.79 bits');
    const waiting = oneletter.findIndex((e) => e.type === 'experiment.log' && /waiting for a stable read/.test((e.payload as { line: string }).line));
    expect(render(foldAll(oneletter.slice(0, waiting + 1)))).toContain('waiting for a stable read');
    const read = oneletter.findIndex((e) => e.type === 'dna.read');
    expect(render(foldAll(oneletter.slice(0, read + 1)))).not.toContain('data-testid="window-prompt"');
    const rep = oneletter.findIndex((e) => e.type === 'dna.request' && (e.payload as { replicate: boolean }).replicate);
    const again = render(foldAll(oneletter.slice(0, rep + 1)));
    expect(again).toContain('Show window 3 again');
    expect(again).toContain('replicate');
  });

  it('shows the surprise banner while the surprise is open', () => {
    const at = oneletter.findIndex((e) => e.type === 'surprise.raised');
    expect(render(foldAll(oneletter.slice(0, at)))).not.toContain('data-testid="surprise-banner"');
    const mid = render(foldAll(oneletter.slice(0, at + 1)));
    expect(mid).toContain('data-testid="surprise-banner"');
    expect(mid).toContain('previous strip');
    expect(mid).toContain('closest HbA, 1 mismatch');
    expect(render(foldAll(oneletter))).not.toContain('data-testid="surprise-banner"');
  });

  it('shows the live camera and no recorded badge for a live run on the real reader', () => {
    const s = liveRun({ reader_url: 'http://192.168.4.20:8770', annotated_url: 'http://192.168.4.20:8770/annotated_stream' });
    const html = render(s, live);
    expect(html).not.toContain('data-testid="recorded-badge"');
    expect(html).toContain('Connecting to the reader camera');
    expect(renderToStaticMarkup(<Header state={s} source={live} />)).not.toContain('Recorded');
    expect(html).toContain('src="http://192.168.4.20:8770/annotated_stream"');
    expect(html).toContain('http://192.168.4.20:8770/strip');
  });

  it('labels a live run on the fake reader as recorded', () => {
    const html = render(liveRun({ reader_url: 'http://127.0.0.1:8770', fake_reader: true }), live);
    expect(html).toContain('data-testid="recorded-badge"');
    expect(html).toContain('Fake reader');
  });

  it('the header shows the recorded badge only when given one', () => {
    const s = foldAll(oneletter);
    expect(renderToStaticMarkup(<Header state={s} source={local} />)).not.toContain('Recorded');
    expect(renderToStaticMarkup(<Header state={s} source={local} extra={<RecordedBadge fakeReader />} />)).toContain(
      'Recorded · fake reader',
    );
  });
});

describe('SeqViz and Molstar props', () => {
  it('annotates the windows and codon 6, highlights reads and mismatches', () => {
    const at = oneletter.findIndex((e) => e.type === 'surprise.raised');
    const p = seqvizProps(foldAll(oneletter.slice(0, at + 1)));
    expect(p.seq).toBe('ATGGTGCATCTGACTCCTGAGGAGAAG');
    expect(p.viewer).toBe('linear');
    expect(p.annotations.map((a) => [a.start, a.end])).toEqual([
      [0, 9],
      [9, 18],
      [18, 27],
      [18, 21],
    ]);
    expect(p.translations).toEqual([{ name: 'beta-globin', start: 0, end: 27, direction: 1 }]);
    expect(p.highlights.map((h) => [h.start, h.end])).toEqual([
      [18, 27],
      [18, 19],
    ]);
    expect(Object.keys(p.bpColors)).toEqual(['18']);
    expect(seqvizProps(initialState()).highlights).toEqual([]);
  });

  it('selects beta residue 6 on every beta chain', () => {
    const sel = residueSelection(['B', 'D', 'F', 'H'], 6, '#ff6b6b');
    expect(sel.map((q) => q.label_asym_id)).toEqual(['B', 'D', 'F', 'H']);
    expect(sel.every((q) => q.beg_label_seq_id === 6 && q.end_label_seq_id === 6)).toBe(true);
    expect(sel.every((q) => q.representation === 'spacefill' && q.color === '#ff6b6b')).toBe(true);
  });
});
